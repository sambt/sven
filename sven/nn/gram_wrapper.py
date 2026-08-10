"""Gram-matrix (kernel-trick) wrapper for memory-efficient Sven updates.

Builds the ``(M, M)`` Gram matrix ``G = J J^T`` of the per-row Jacobian without
materialising the ``(M, P)`` Jacobian itself.  With the thin SVD
``J = U S V^T`` we have ``G = U S^2 U^T``, and the pseudo-inverse update

    delta = V_k S_k^{-1} U_k^T r = J^T [ U_k S_k^{-2} U_k^T r ] = J^T w

is recovered from one standard backward pass of ``sum_m w_m r_m`` (see
:meth:`GramSvenWrapper.delta_from_w`), exact for any architecture.

Two capture modes build ``G``:

- ``"hooks"``: one forward + one backward of ``sum_m r_m``; per-layer inputs
  ``x_l`` and grad-outputs ``g_l`` are captured and contracted layer-by-layer
  (2-D Linear: ``K += (g_l g_l^T) * (x_l x_l^T)``).  The layer set covers the
  transformer stack: Linear with any lead dims ``(B, *lead, d)``, ``groups=1``
  zero-padded Conv2d, frozen-stats ``_NormBase``, per-position ``nn.LayerNorm``
  and batched-input ``nn.Embedding``.  Exact only when per-sample losses do
  not interact across samples, hence the norm-stat/dropout guards.
- ``"chunked"``: per-parameter-group ``jacrev`` over at most ``chunk_numel``
  active parameters at a time; genuinely ``(M, n_group)`` memory and exact
  for any architecture, including batch-coupled normalisation.

A stochastic parameter mask (``param_fraction < 1`` + ``mask_mode``)
restricts ``G`` and the update to a random parameter subset in either
capture; the masked update is always the full ``J^T w`` backward gathered at
the mask.

Hooks capture masks at unchanged capture cost: the masked Gram is pure
algebra on the SAME one-backward captures.  Row masks factorise per layer,
``G_l^A = (g_A g_A^T) * (x x^T)`` for the selected weight rows plus
``g_A g_A^T`` for their bias entries with ``g_A`` the selected grad-output
columns; whole-tensor masks include or skip each per-tensor term; elementwise
masks contract explicit ``(B, n_active_l)`` masked-gradient matrices — never a
per-sample gradient block.  Memory is ~constant in the fraction for
rows/tensor masks (captures + the ``(B, B)`` kernel + one flat-parameter
transient) and ``O(B * n_active)`` for elementwise.

Chunked capture masks like ``sven.jax.gram``: one sorted flat mask per
:meth:`GramSvenWrapper.loss_and_grad`; per parameter group the group-local
mask indices are located inside the group's contiguous flat span, the masked
columns are gathered from the ``(M, P_grp)`` block before the contraction,
and groups with no masked columns skip their ``jacrev`` entirely — exact for
any architecture, transformers included.  ``mask_mode="rows"`` here means
LEADING-AXIS slices per parameter tensor (embedding rows = vocab entries; a
Linear/Conv weight's rows = its output neurons; a LayerNorm weight = a
fraction of channels), not the hooks-mode neuron-tied weight+bias pairing;
1-D tensors draw a fraction of entries and 0-d tensors are always active.
Memory is **group-bounded**, not fraction-constant: the gather adds a
transient ``(M, n_masked_grp)`` copy on top of the ``(M, P_grp)`` block, and
a tensor at or above ``chunk_numel`` forms its own oversized group that
dominates the peak regardless of masking.  Compute is unchanged from the
unmasked chunked capture — the full group backward runs regardless of the
mask; the savings over the classic Jacobian pipeline remain the bounded
memory and the ``(M, M)`` eigh in place of the ``(M, P)`` SVD.

``G`` is accumulated in ``gram_dtype`` (default float64) regardless of the
model dtype: ``cond(G) = cond(J)^2``, so float32 accumulation destroys the
small singular values that tight ``rtol`` settings rely on.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from torch.nn.modules.batchnorm import _NormBase
from torch.nn.modules.dropout import _DropoutNd

from .sven_wrapper import SvenWrapper


class GramSvenWrapper(SvenWrapper):
    """Wrapper computing the Gram matrix ``G = J J^T`` instead of the Jacobian.

    After :meth:`loss_and_grad` the wrapper exposes ``self.losses`` (B,) raw
    per-sample losses, ``self.residuals`` (M,) transformed rows (microbatch
    mean then ``kappa/2`` power) and ``self.gram`` (M, M) in ``gram_dtype``;
    ``self.grads`` is set to ``None`` so a plain :class:`sven.opt.Sven` fails
    loudly — use :class:`sven.opt.SvenGram`.  With ``param_fraction < 1`` a
    fresh random mask is drawn every :meth:`loss_and_grad` and both ``G``
    and the update are restricted to it — ``self.param_mask`` /
    ``self.actual_param_fraction`` / the selection attributes follow the
    :class:`SvenWrapper` conventions.

    Args:
        model: The PyTorch model to wrap.
        loss_fn: A loss function ``(pred, *args) -> Tensor`` that returns
            **per-sample** losses with shape ``(B,)``.
        device: Device to place the model and parameters on.
        kappa: Exponent for the raw loss, as in :class:`SvenWrapper`.
        param_fraction: Fraction of parameters to restrict the Gram/update to,
            resampled each :meth:`loss_and_grad`.  ``1.0`` uses all parameters.
        microbatch_size: If ``> 1``, aggregate losses within sub-batches of
            this size before the ``kappa`` power, reducing the row dimension.
        capture: ``"hooks"`` (fast, per-sample-decoupled architectures only)
            or ``"chunked"`` (exact for any architecture).
        gram_dtype: Accumulation dtype for ``G``; keep float64 for fp32 models.
        freeze_norm_stats: Switch ``_NormBase`` modules to eval (running
            stats) during every wrapper pass, removing cross-sample coupling.
        chunk_numel: ``"chunked"`` mode only — max total active-parameter
            numel differentiated per ``jacrev`` group.
        mask_mode: Mask structure when ``param_fraction < 1`` — ``"rows"``,
            ``"tensor"`` (whole parameter tensors) or ``"elementwise"``
            (random flat subset).  ``"rows"`` under hooks capture selects
            output rows/channels per exact ``nn.Linear`` / ``groups=1``
            ``nn.Conv2d``, weight rows tied to their bias entries (norm-affine
            masking is future work); under chunked capture it selects
            leading-axis slices independently per parameter tensor, any
            architecture — see the module docstring.  Required when
            ``param_fraction < 1``; ignored at ``1.0``.
    """

    _CONV_BLOCK_ELEMS: int = 2 ** 26  # cap on a materialised (chunk, P_l) conv grad block

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable[..., torch.Tensor],
        device: torch.device | str,
        kappa: float = 2.0,
        param_fraction: float = 1.0,
        microbatch_size: int = 1,
        capture: str = "hooks",
        gram_dtype: torch.dtype = torch.float64,
        freeze_norm_stats: bool = True,
        chunk_numel: int = 2 ** 22,
        mask_mode: str | None = None,
    ) -> None:
        if capture not in ("hooks", "chunked"):
            raise ValueError(f"capture must be 'hooks' or 'chunked', got {capture!r}")
        if param_fraction < 1.0:
            if mask_mode not in ("rows", "tensor", "elementwise"):
                raise ValueError(
                    "param_fraction < 1 requires mask_mode 'rows', 'tensor' or "
                    f"'elementwise', got {mask_mode!r}"
                )
        super().__init__(
            model,
            loss_fn,
            device,
            kappa=kappa,
            param_fraction=param_fraction,
            microbatch_size=microbatch_size,
        )
        # The base class derives a legacy mask_mode; the Gram wrapper keeps
        # None for "no masking" and never does rows-mode forward surgery.
        self.mask_mode = mask_mode if param_fraction < 1.0 else None
        self.mask_by_block = self.mask_mode == "tensor"
        self.capture: str = capture
        self.gram_dtype: torch.dtype = gram_dtype
        self.freeze_norm_stats: bool = freeze_norm_stats
        self.chunk_numel: int = chunk_numel
        if self.mask_mode == "rows" and self.capture == "hooks":
            self._check_mask_rows_supported()
            # Reuse the base rows sampler: it only reads the (path, module,
            # w_start, row_numel, b_start) tuples, which plain modules satisfy.
            # (Chunked-rows samples leading-axis slices instead — no guard, no
            # module index; see _make_param_mask_by_leading_rows.)
            self._row_twins = self._index_row_modules()
        elif self.mask_mode == "elementwise":
            self._flat_slices: dict[str, tuple[int, int]] = {
                name: (start, n) for name, n, start in self.param_names_counts_startIdx
            }

        # Populated by loss_and_grad(), consumed by SvenGram.step()
        self.gram: torch.Tensor | None = None
        self.grads = None  # a plain Sven optimizer must fail loudly on this wrapper
        self._batch: tuple[torch.Tensor, ...] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def loss_and_grad(
        self, batch: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute losses and the Gram matrix ``G = J J^T`` for the optimizer.

        Populates ``self.losses`` (B,), ``self.residuals`` (M,) and
        ``self.gram`` (M, M); the batch is retained for :meth:`delta_from_w`.
        Unlike the base class, ``self.losses`` holds the raw **per-sample**
        losses, not the microbatch-grouped rows.  With ``param_fraction < 1``
        a fresh ``self.param_mask`` is drawn first and ``self.gram`` is the
        masked Gram ``J_A J_A^T``.

        Args:
            batch: Tuple of ``(x, y, ...)`` tensors.

        Returns:
            ``(losses, predictions)`` — both detached from the compute graph.
        """
        self._check_dropout()
        x, *args = batch
        self._batch = batch

        if self.capture == "hooks":
            self._check_hooks_supported()
            if self.param_fraction < 1.0:
                self._sample_param_mask()
            with torch.enable_grad(), self._frozen_norm_stats():
                kernel, losses, preds = self._hooks_kernel(x, args)
            mb = self.microbatch_size
            if mb > 1:
                # Row m of J is the SUM of in-group per-sample gradients, so
                # G_mm' = sum_{b in m, b' in m'} K_bb' — block-sum pooling
                # including the cross terms (diagonal blocks alone are wrong).
                m = kernel.shape[0] // mb
                gram = kernel.view(m, mb, m, mb).sum(dim=(1, 3))
            else:
                gram = kernel
        else:
            if self.param_fraction < 1.0:
                self._sample_param_mask()
            with self._frozen_norm_stats():
                gram, losses, preds = self._chunked_gram(x, args)

        self.losses = losses.detach()
        self.residuals = self._group_losses(self.losses).pow(self.kappa / 2.0).detach()
        self.gram = gram.detach()
        self.grads = None
        return self.losses, preds

    def delta_from_w(self, w: torch.Tensor) -> torch.Tensor:
        """Recover the update ``delta = J^T w`` with one forward + backward.

        ``J^T w`` is the gradient of ``sum_m w_m r_m(theta)`` w.r.t. the flat
        parameters — no ``jacrev``, and exact for any architecture, including
        batch-coupled ones.  With an active ``param_mask`` the full backward
        still runs (one flat-parameter transient) and the masked gather is
        returned, matching ``Sven._apply_update``'s masked route.

        Args:
            w: ``(M,)`` row-space weights, e.g. ``U_k S_k^{-2} U_k^T r``.

        Returns:
            Flat ``(P,)`` update in the parameter dtype, detached —
            ``(n_active,)`` when a ``param_mask`` is active.
        """
        if self._batch is None:
            raise RuntimeError("delta_from_w requires a stored batch; call loss_and_grad first")
        x, *args = self._batch
        w = w.detach().to(device=self.params.device, dtype=self.params.dtype)
        # Not self._loss: with an active mask it expects the (n_active,)
        # masked values as its params argument, not the full flat vector.
        with torch.enable_grad(), self._frozen_norm_stats():
            pred = self._func_call(self.params, x)
            rows = self._group_losses(self.loss_fn(pred, *args)).pow(self.kappa / 2.0)
            (delta,) = torch.autograd.grad((w * rows).sum(), self.params)
        delta = delta.detach()
        if self.param_mask is not None:
            delta = delta[self.param_mask]
        return delta

    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------

    def _check_dropout(self) -> None:
        """Dropout resamples per pass, breaking capture/update consistency."""
        for name, mod in self.model.named_modules():
            if isinstance(mod, _DropoutNd) and mod.training and mod.p > 0:
                raise ValueError(
                    f"train-mode dropout ('{name}') makes the capture and update "
                    "passes inconsistent; call .eval() on it or set p=0"
                )

    def _check_hooks_supported(self) -> None:
        """Hooks capture requires per-sample-decoupled layers.

        The supported set is Linear / Conv2d / ``_NormBase`` / LayerNorm /
        Embedding; masking (``param_fraction < 1``) covers the original
        Linear/Conv2d/``_NormBase`` subset only, so masked transformers stay
        chunked-only.
        """
        n_unique = len(list(self.model.named_parameters()))
        n_total = len(list(self.model.named_parameters(remove_duplicate=False)))
        if n_unique != n_total:
            raise NotImplementedError(
                "tied parameters detected; hooks capture cannot combine "
                "shared-weight contributions — use capture='chunked'"
            )
        for name, mod in self.model.named_modules():
            if isinstance(mod, _NormBase):
                if mod.training or mod.running_mean is None:
                    if not self.freeze_norm_stats:
                        raise ValueError(
                            f"norm layer '{name}' uses batch statistics, coupling "
                            "per-sample losses across the batch: hook-captured "
                            "per-sample gradients are wrong upstream of it.  Set "
                            "freeze_norm_stats=True or use capture='chunked'"
                        )
                    if mod.running_mean is None:
                        raise ValueError(
                            f"norm layer '{name}' has no running statistics to "
                            "freeze (track_running_stats=False); use capture='chunked'"
                        )
            elif isinstance(mod, nn.Conv2d):
                if mod.groups != 1:
                    raise NotImplementedError(
                        f"grouped Conv2d ('{name}') is not supported by hooks "
                        "capture; use capture='chunked'"
                    )
                if mod.padding_mode != "zeros" or isinstance(mod.padding, str):
                    raise NotImplementedError(
                        f"Conv2d ('{name}') with padding_mode='{mod.padding_mode}' "
                        f"or string padding is not supported by hooks capture "
                        "(unfold assumes zero padding); use capture='chunked'"
                    )
            elif isinstance(mod, (nn.LayerNorm, nn.Embedding)):
                if isinstance(mod, nn.LayerNorm) and mod.weight is None:
                    continue  # no parameters: nothing to capture
                if self.param_fraction < 1.0:
                    raise NotImplementedError(
                        f"masked hooks capture supports nn.Linear/nn.Conv2d/"
                        f"_NormBase modules only; '{name}' ({type(mod).__name__}) "
                        "cannot join a param_fraction < 1 Gram — use "
                        "capture='chunked'"
                    )
                if isinstance(mod, nn.Embedding) and (
                    mod.max_norm is not None or mod.scale_grad_by_freq
                ):
                    raise NotImplementedError(
                        f"Embedding '{name}' with max_norm or scale_grad_by_freq "
                        "is not supported by hooks capture; use capture='chunked'"
                    )
            elif not isinstance(mod, nn.Linear):
                if any(True for _ in mod.parameters(recurse=False)):
                    raise NotImplementedError(
                        f"module '{name or '<root>'}' ({type(mod).__name__}) holds "
                        "parameters but is not Linear/Conv2d/_NormBase/LayerNorm/"
                        "Embedding; hooks capture cannot form its per-sample "
                        "gradients — use capture='chunked'"
                    )

    @contextmanager
    def _frozen_norm_stats(self) -> Iterator[None]:
        """Switch train-mode norm layers to eval (running stats) for one pass."""
        switched: list[nn.Module] = []
        if self.freeze_norm_stats:
            for mod in self.model.modules():
                if isinstance(mod, _NormBase) and mod.training:
                    mod.eval()
                    switched.append(mod)
        try:
            yield
        finally:
            for mod in switched:
                mod.train()

    # ------------------------------------------------------------------
    # Stochastic parameter masking
    # ------------------------------------------------------------------

    def _check_mask_rows_supported(self) -> None:
        """Rows-mode masking factorises exact Linear / groups=1 Conv2d kernels only."""
        self._check_rows_supported()
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Conv2d) and mod.groups != 1:
                raise NotImplementedError(
                    f"grouped Conv2d ('{name}') is not supported by rows-mode "
                    "masking: channel selection cannot split grouped filters "
                    f"(got groups={mod.groups})"
                )

    def _index_row_modules(self) -> list[tuple[str, nn.Module, int, int, int | None]]:
        """Locate maskable Linear/Conv2d modules and their flat-vector offsets.

        Same contract (and sort order) as :meth:`SvenWrapper._index_row_twins`,
        but over the plain modules: masking ``G`` needs no forward surgery,
        only the selection bookkeeping.
        """
        offsets = {name: start for name, _, start in self.param_names_counts_startIdx}
        mods: list[tuple[str, nn.Module, int, int, int | None]] = []
        for path, mod in self.model.named_modules():
            if isinstance(mod, (nn.Linear, nn.Conv2d)):
                prefix = f"{path}." if path else ""
                w_start = offsets[f"{prefix}weight"]
                row_numel = mod.weight.numel() // mod.weight.shape[0]
                b_start = offsets[f"{prefix}bias"] if mod.bias is not None else None
                mods.append((path, mod, w_start, row_numel, b_start))
        if not mods:
            raise NotImplementedError(
                "mask_mode='rows' found no nn.Linear/nn.Conv2d submodules to mask"
            )
        mods.sort(key=lambda entry: entry[2])
        return mods

    def _sample_param_mask(self) -> None:
        """Draw this step's ``param_mask``, reusing the base-class samplers.

        ``"tensor"`` and ``"elementwise"`` share the :class:`SvenWrapper`
        samplers in both captures (identical seed => identical mask across
        the hooks, chunked and Jacobian pipelines).  ``"rows"`` diverges:
        hooks capture uses the base neuron-tied sampler, chunked capture the
        leading-axis sampler below.
        """
        if self.mask_mode == "rows":
            if self.capture == "chunked":
                mask = self._make_param_mask_by_leading_rows(self.param_fraction)
            else:
                mask = self._make_param_mask_by_rows(self.param_fraction)
        elif self.mask_mode == "tensor":
            mask = self._make_param_mask_by_block(self.param_fraction)
        else:  # elementwise
            mask = self._make_param_mask()
            self.actual_param_fraction = mask.sum().item() / self.n_params
        self.param_mask = mask.to(self.params.device)

    def _make_param_mask_by_leading_rows(self, fraction: float) -> torch.Tensor:
        """Sample leading-axis slices per parameter tensor (chunked-rows mask).

        Chunked-rows semantics, mirroring ``sven.jax``: every parameter
        tensor is masked independently along its leading axis — embedding
        rows are vocab entries, a Linear/Conv weight's rows are its output
        neurons, a LayerNorm weight draws a fraction of channels — with no
        weight/bias pairing.  Each tensor draws
        ``max(1, round(fraction * dim0))`` distinct sorted leading indices
        (1-D tensors: a fraction of entries; 0-d tensors are always active);
        each selected slice is a contiguous run in the flat layout.  The
        achieved fraction is stored in ``self.actual_param_fraction``.
        """
        mask = torch.zeros(self.n_params, dtype=torch.bool)
        starts = {name: start for name, _, start in self.param_names_counts_startIdx}
        for name, shape, n in self.param_shapes:
            start = starts[name]
            if len(shape) == 0:
                mask[start] = True
                continue
            dim0 = shape[0]
            row_numel = n // dim0
            n_rows = max(1, int(round(fraction * dim0)))
            rows = torch.randperm(dim0)[:n_rows].sort().values
            if row_numel == 1:
                mask[start + rows] = True
            else:
                for o in rows.tolist():
                    mask[start + o * row_numel : start + (o + 1) * row_numel] = True
        self.actual_param_fraction = mask.sum().item() / self.n_params
        return mask

    def _local_mask_idx(self, name: str) -> torch.Tensor:
        """Selected flat indices of parameter ``name``, local to its tensor."""
        start, n = self._flat_slices[name]
        return self.param_mask[start : start + n].nonzero(as_tuple=True)[0]

    # ------------------------------------------------------------------
    # Hooks capture
    # ------------------------------------------------------------------

    def _hooks_kernel(
        self, x: torch.Tensor, args: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-sample kernel ``K`` (B, B) from one forward + one backward.

        The backward targets the captured layer *outputs* (not the params), so
        no parameter gradients are formed; ``g_l`` already carries the chain
        factors of the microbatch mean and ``kappa`` power.
        """
        stores: list[dict] = []
        handles = []

        def mk_hook(store: dict):
            def hook(mod: nn.Module, inp: tuple, out: torch.Tensor) -> None:
                store["calls"].append((inp[0], out))
            return hook

        for name, mod in self.model.named_modules():
            if isinstance(mod, (nn.Linear, nn.Conv2d, nn.Embedding)) or (
                isinstance(mod, (_NormBase, nn.LayerNorm)) and mod.weight is not None
            ):
                store = {"name": name, "mod": mod, "calls": []}
                stores.append(store)
                handles.append(mod.register_forward_hook(mk_hook(store)))

        b = x.shape[0]
        try:
            pred = self._func_call(self.params, x)
            loss = self.loss_fn(pred, *args)
            rows = self._group_losses(loss).pow(self.kappa / 2.0)

            stores = [s for s in stores if s["calls"]]  # unused modules contribute nothing
            for store in stores:
                if len(store["calls"]) > 1:
                    raise NotImplementedError(
                        f"module '{store['name']}' is applied more than once per "
                        "forward; hooks capture cannot separate the applications "
                        "— use capture='chunked'"
                    )
                if isinstance(store["mod"], nn.Embedding):
                    idx = store["calls"][0][0]
                    # A 1-D input is indistinguishable by shape from an
                    # unbatched broadcast lookup (whose cotangent is
                    # batch-summed before the hook sees it, destroying
                    # per-row information — silently wrong when its length
                    # happens to equal B).  Reject 1-D outright.
                    if idx.dim() < 2 or idx.shape[0] != b:
                        raise NotImplementedError(
                            f"Embedding '{store['name']}' input has shape "
                            f"{tuple(idx.shape)}; hooks capture needs batched "
                            f">=2-D indices with leading dim B={b} (e.g. "
                            "torch.arange(t).expand(B, t), or idx[:, None] "
                            "for per-sample scalars) — or use capture='chunked'"
                        )
                if isinstance(store["mod"], nn.LayerNorm):
                    xin = store["calls"][0][0]
                    if xin.dim() == len(store["mod"].normalized_shape):
                        raise NotImplementedError(
                            f"LayerNorm '{store['name']}' normalizes over its "
                            "entire input including the batch dim, coupling "
                            "per-sample losses across the batch: hooks capture "
                            "would be silently wrong — use capture='chunked'"
                        )
            z_list = [store["calls"][0][1] for store in stores]
            gs = torch.autograd.grad(rows.sum(), z_list, allow_unused=True)
        finally:
            for handle in handles:
                handle.remove()

        kernel = torch.zeros(b, b, dtype=self.gram_dtype, device=self.device)
        for store, g in zip(stores, gs):
            if g is None:  # output does not reach the loss
                continue
            xin = store["calls"][0][0].detach()
            g = g.detach()
            mod = store["mod"]
            if self.param_mask is not None:
                self._accum_masked(store["name"], mod, xin, g, kernel)
            elif isinstance(mod, nn.Linear):
                self._accum_linear(mod, xin, g, kernel)
            elif isinstance(mod, nn.Conv2d):
                self._accum_conv2d(mod, xin, g, kernel)
            elif isinstance(mod, nn.LayerNorm):
                self._accum_layernorm_affine(mod, xin, g, kernel)
            elif isinstance(mod, nn.Embedding):
                self._accum_embedding(mod, xin, g, kernel)
            else:
                self._accum_norm_affine(mod, xin, g, kernel)
        return kernel, loss.detach(), pred.detach()

    def _accum_masked(
        self,
        name: str,
        mod: nn.Module,
        xin: torch.Tensor,
        g: torch.Tensor,
        kernel: torch.Tensor,
    ) -> None:
        """Masked kernel contribution of one captured layer.

        rows: the masked kernel factorises through the selected grad-output
        columns, ``G_l^A = (g_A g_A^T) * (x x^T)`` — the unmasked contraction
        runs on ``g_A`` unchanged.  tensor: include or skip the weight/bias
        term per selected tensor.  elementwise: explicit ``(B, n_active_l)``
        masked-gradient matrices, never a per-sample gradient block.
        """
        prefix = f"{name}." if name else ""
        if self.mask_mode == "rows":
            rows = next(r for p, _, r in self._row_selection if p == name)
            if isinstance(mod, nn.Linear):
                g_a = g.index_select(-1, rows.to(g.device))  # output cols, any lead dims
                self._accum_linear(mod, xin, g_a, kernel)
            else:
                g_a = g.index_select(1, rows.to(g.device))
                self._accum_conv2d(mod, xin, g_a, kernel)
        elif self.mask_mode == "tensor":
            selected = {n for n, _, _ in self._block_selection}
            with_w = f"{prefix}weight" in selected
            with_b = f"{prefix}bias" in selected
            if not (with_w or with_b):
                return
            if isinstance(mod, nn.Linear):
                self._accum_linear(mod, xin, g, kernel, with_weight=with_w, with_bias=with_b)
            elif isinstance(mod, nn.Conv2d):
                self._accum_conv2d(mod, xin, g, kernel, with_weight=with_w, with_bias=with_b)
            else:
                self._accum_norm_affine(
                    mod, xin, g, kernel, with_weight=with_w, with_bias=with_b
                )
        else:  # elementwise
            w_idx = self._local_mask_idx(f"{prefix}weight")
            b_idx = self._local_mask_idx(f"{prefix}bias") if mod.bias is not None else None
            if isinstance(mod, nn.Linear):
                self._accum_linear_elementwise(mod, xin, g, kernel, w_idx, b_idx)
            elif isinstance(mod, nn.Conv2d):
                self._accum_conv2d_elementwise(mod, xin, g, kernel, w_idx, b_idx)
            else:
                self._accum_norm_affine_elementwise(mod, xin, g, kernel, w_idx, b_idx)

    def _accum_block_outer(
        self, kernel: torch.Tensor, b: int, p_l: int, block: Callable[[slice], torch.Tensor]
    ) -> None:
        """``K += Psi Psi^T`` for per-sample flattened grad blocks ``Psi``.

        ``block(sl)`` materialises the ``(len(sl), p_l)`` grad block for one
        batch slice; the batch is chunked whenever the block would exceed
        ``_CONV_BLOCK_ELEMS`` elements, and off-diagonal chunk pairs recompute
        one side rather than holding the full block.
        """
        rows = max(1, min(b, self._CONV_BLOCK_ELEMS // p_l))
        if rows >= b:
            psw = block(slice(0, b))
            kernel += psw @ psw.T
        else:
            for i in range(0, b, rows):
                si = slice(i, min(i + rows, b))
                psw_i = block(si)
                kernel[si, si] += psw_i @ psw_i.T
                for j in range(0, i, rows):
                    sj = slice(j, min(j + rows, b))
                    psw_j = block(sj)
                    blk = psw_i @ psw_j.T
                    kernel[si, sj] += blk
                    kernel[sj, si] += blk.T

    def _accum_linear(
        self,
        mod: nn.Linear,
        xin: torch.Tensor,
        g: torch.Tensor,
        kernel: torch.Tensor,
        with_weight: bool = True,
        with_bias: bool = True,
    ) -> None:
        """``K += (g g^T) * (x x^T)`` for 2-D inputs, plus ``g g^T`` for the bias.

        ``(B, *lead, d)`` inputs sum the per-sample weight gradient over every
        lead dim, ``A_b = g_b^T x_b`` with ``g_b (L, d_out)``, ``x_b (L, d_in)``
        — the flattened ``(B, d_out * d_in)`` blocks are formed explicitly and
        contracted immediately, batch-chunked like the conv route.  (The
        token-pair "ghost" contraction ``K_ab = sum_tt' (g_a[t].g_b[t'])
        (x_a[t].x_b[t'])`` never materialises the blocks at ``O(B^2 L^2 d)``
        cost — ~10x slower at our dims; future work for very large
        ``d_out * d_in``.)  The bias block is ``G_b = sum_lead g``, ``K += G_b
        G_b^T``.  Under a rows mask ``g`` arrives pre-gathered to the selected
        columns; under a tensor mask the flags include/skip the per-tensor
        terms.
        """
        gd = g.to(self.gram_dtype)
        if xin.dim() == 2:
            gg = gd @ gd.T
            if with_weight:
                xd = xin.to(self.gram_dtype)
                kernel += gg * (xd @ xd.T)
            if with_bias and mod.bias is not None:
                kernel += gg
            return
        b = g.shape[0]
        gf = gd.reshape(b, -1, g.shape[-1])  # (B, L, d_out)
        if with_weight:
            xf = xin.to(self.gram_dtype).reshape(b, -1, xin.shape[-1])  # (B, L, d_in)

            def block(sl: slice) -> torch.Tensor:
                a = torch.einsum("blo,bli->boi", gf[sl], xf[sl])
                return a.reshape(a.shape[0], -1)

            self._accum_block_outer(kernel, b, g.shape[-1] * xin.shape[-1], block)
        if with_bias and mod.bias is not None:
            gb = gf.sum(dim=1)
            kernel += gb @ gb.T

    def _accum_conv2d(
        self,
        mod: nn.Conv2d,
        xin: torch.Tensor,
        g: torch.Tensor,
        kernel: torch.Tensor,
        with_weight: bool = True,
        with_bias: bool = True,
    ) -> None:
        """Unfold-based per-sample conv grads, contracted immediately.

        The batch is chunked (``_accum_block_outer``) whenever the
        materialised ``(chunk, P_l)`` block would exceed ``_CONV_BLOCK_ELEMS``
        elements.  Under a rows mask ``g`` arrives gathered to the selected
        output channels, so the blocks (and the chunk budget) shrink
        proportionally.
        """
        if xin.dim() != 4:
            raise NotImplementedError(
                f"Conv2d input with {xin.dim()} dims; hooks capture supports 4D "
                "inputs only — use capture='chunked'"
            )
        b = g.shape[0]
        if with_weight:
            gf = g.reshape(b, g.shape[1], -1).to(self.gram_dtype)  # (B, C', L)

            def block(sl: slice) -> torch.Tensor:
                x_unf = F.unfold(
                    xin[sl].to(self.gram_dtype),
                    mod.kernel_size,
                    dilation=mod.dilation,
                    padding=mod.padding,
                    stride=mod.stride,
                )  # (b', C_in*kh*kw, L)
                psw = torch.einsum("bol,bkl->bok", gf[sl], x_unf)
                return psw.reshape(psw.shape[0], -1)

            p_l = gf.shape[1] * (mod.weight.numel() // mod.out_channels)
            self._accum_block_outer(kernel, b, p_l, block)
        if with_bias and mod.bias is not None:
            gb = g.sum(dim=(2, 3)).to(self.gram_dtype)
            kernel += gb @ gb.T

    def _norm_affine_grads(
        self, mod: _NormBase, xin: torch.Tensor, g: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Frozen-stats per-sample affine grads: (sum_sp g * xhat, sum_sp g)."""
        shape = [1, -1] + [1] * (xin.dim() - 2)
        mean = mod.running_mean.view(shape).to(xin.dtype)
        var = mod.running_var.view(shape).to(xin.dtype)
        xhat = (xin - mean) / torch.sqrt(var + mod.eps)
        if xin.dim() > 2:
            spatial = tuple(range(2, xin.dim()))
            return (g * xhat).sum(dim=spatial), g.sum(dim=spatial)
        return g * xhat, g

    def _accum_norm_affine(
        self,
        mod: _NormBase,
        xin: torch.Tensor,
        g: torch.Tensor,
        kernel: torch.Tensor,
        with_weight: bool = True,
        with_bias: bool = True,
    ) -> None:
        """Frozen-stats norm affine: grad_gamma = sum_sp g * xhat, grad_beta = sum_sp g."""
        g_gamma, g_beta = self._norm_affine_grads(mod, xin, g)
        if with_weight:
            g_gamma = g_gamma.to(self.gram_dtype)
            kernel += g_gamma @ g_gamma.T
        if with_bias and mod.bias is not None:
            g_beta = g_beta.to(self.gram_dtype)
            kernel += g_beta @ g_beta.T

    def _accum_layernorm_affine(
        self,
        mod: nn.LayerNorm,
        xin: torch.Tensor,
        g: torch.Tensor,
        kernel: torch.Tensor,
    ) -> None:
        """LayerNorm affine: grad_gamma = sum_lead g * xhat, grad_beta = sum_lead g.

        LayerNorm normalises over its trailing ``normalized_shape`` dims per
        position — per-sample, no cross-row coupling and no frozen stats.
        ``xhat`` is recomputed from the captured input with the module eps
        (biased variance over the normalized dims, matching ``F.layer_norm``);
        lead dims are every dim between the batch and ``normalized_shape``.
        Only affine LayerNorms are hooked, so ``mod.weight`` always exists
        here; a bias-less LayerNorm skips the beta block.
        """
        n_norm = len(mod.normalized_shape)
        norm_dims = tuple(range(xin.dim() - n_norm, xin.dim()))
        mean = xin.mean(dim=norm_dims, keepdim=True)
        var = xin.var(dim=norm_dims, unbiased=False, keepdim=True)
        xhat = (xin - mean) / torch.sqrt(var + mod.eps)
        lead = tuple(range(1, xin.dim() - n_norm))
        g_gamma = (g * xhat).sum(dim=lead) if lead else g * xhat
        g_gamma = g_gamma.reshape(g.shape[0], -1).to(self.gram_dtype)
        kernel += g_gamma @ g_gamma.T
        if mod.bias is not None:
            g_beta = g.sum(dim=lead) if lead else g
            g_beta = g_beta.reshape(g.shape[0], -1).to(self.gram_dtype)
            kernel += g_beta @ g_beta.T

    def _accum_embedding(
        self,
        mod: nn.Embedding,
        xin: torch.Tensor,
        g: torch.Tensor,
        kernel: torch.Tensor,
    ) -> None:
        """Dense-scatter per-sample embedding grads, contracted per batch chunk.

        Row ``v`` of sample ``b``'s ``(V, d)`` gradient accumulates
        ``g[b, t]`` over the positions ``t`` with ``idx[b, t] == v`` — a
        ``scatter_add`` into ``(chunk, V, d)`` blocks, flattened and
        contracted immediately like the conv route; ``padding_idx`` positions
        contribute nothing and are zeroed first.  Fine for byte-level vocabs;
        when even a single-sample block busts ``_CONV_BLOCK_ELEMS`` the dense
        route is hopeless and we raise.  (The index-match kernel ``K_ab =
        sum_{t,t': idx_a[t] == idx_b[t']} g_a[t].g_b[t']`` never materialises
        the blocks — future work for large ``V * d``.)
        """
        v, d = mod.num_embeddings, mod.embedding_dim
        p_l = v * d
        if p_l > self._CONV_BLOCK_ELEMS:
            raise NotImplementedError(
                f"Embedding scatter blocks need {v} x {d} = {p_l} elements per "
                f"sample, over the _CONV_BLOCK_ELEMS={self._CONV_BLOCK_ELEMS} "
                "budget even one sample at a time — use capture='chunked'"
            )
        b = g.shape[0]
        idx = xin.reshape(b, -1)  # (B, L)
        gf = g.reshape(b, idx.shape[1], d).to(self.gram_dtype)  # (B, L, d)
        if mod.padding_idx is not None:
            gf = gf.masked_fill((idx == mod.padding_idx).unsqueeze(-1), 0)
        tgt = idx.unsqueeze(-1).expand(-1, -1, d)  # destination rows per position

        def block(sl: slice) -> torch.Tensor:
            e = torch.zeros(
                gf[sl].shape[0], v, d, dtype=self.gram_dtype, device=kernel.device
            )
            e.scatter_add_(1, tgt[sl], gf[sl])
            return e.reshape(e.shape[0], -1)

        self._accum_block_outer(kernel, b, p_l, block)

    def _accum_linear_elementwise(
        self,
        mod: nn.Linear,
        xin: torch.Tensor,
        g: torch.Tensor,
        kernel: torch.Tensor,
        w_idx: torch.Tensor,
        b_idx: torch.Tensor | None,
    ) -> None:
        """Explicit masked matrix ``U[b, m] = g[b, o_m] * x[b, i_m]``, ``K += U U^T``.

        Two index_selects and a multiply — the ``(B, d_out, d_in)`` per-sample
        gradient block is never materialised; memory is ``O(B * n_active_l)``.
        """
        if xin.dim() != 2:
            raise NotImplementedError(
                f"Linear input with {xin.dim()} dims; elementwise-masked hooks "
                "capture supports 2D inputs only — use capture='chunked'"
            )
        gd = g.to(self.gram_dtype)
        if w_idx.numel():
            o_idx = torch.div(w_idx, mod.in_features, rounding_mode="floor")
            i_idx = w_idx % mod.in_features
            u = gd.index_select(1, o_idx) * xin.to(self.gram_dtype).index_select(1, i_idx)
            kernel += u @ u.T
        if b_idx is not None and b_idx.numel():
            u = gd.index_select(1, b_idx)
            kernel += u @ u.T

    def _accum_conv2d_elementwise(
        self,
        mod: nn.Conv2d,
        xin: torch.Tensor,
        g: torch.Tensor,
        kernel: torch.Tensor,
        w_idx: torch.Tensor,
        b_idx: torch.Tensor | None,
    ) -> None:
        """Explicit masked conv grads: ``U[b, m] = <g_b[o_m, :], x_unf_b[k_m, :]>``.

        Both factors are gathered to the selected (channel, tap) pairs before
        the length-``L`` contraction, batch-chunked under ``_CONV_BLOCK_ELEMS``
        — no ``(B, C_out, P_row)`` block is ever materialised.
        """
        b = g.shape[0]
        if w_idx.numel():
            row_numel = mod.weight.numel() // mod.out_channels
            o_idx = torch.div(w_idx, row_numel, rounding_mode="floor")
            k_idx = w_idx % row_numel
            gf = g.reshape(b, g.shape[1], -1).to(self.gram_dtype)  # (B, C_out, L)
            u = torch.empty(b, w_idx.numel(), dtype=self.gram_dtype, device=kernel.device)
            rows = max(1, min(b, self._CONV_BLOCK_ELEMS // (w_idx.numel() * gf.shape[2])))
            for i in range(0, b, rows):
                sl = slice(i, min(i + rows, b))
                x_unf = F.unfold(
                    xin[sl].to(self.gram_dtype),
                    mod.kernel_size,
                    dilation=mod.dilation,
                    padding=mod.padding,
                    stride=mod.stride,
                )  # (b', C_in*kh*kw, L)
                u[sl] = torch.einsum(
                    "bml,bml->bm",
                    gf[sl].index_select(1, o_idx),
                    x_unf.index_select(1, k_idx),
                )
            kernel += u @ u.T
        if b_idx is not None and b_idx.numel():
            gb = g.sum(dim=(2, 3)).to(self.gram_dtype).index_select(1, b_idx)
            kernel += gb @ gb.T

    def _accum_norm_affine_elementwise(
        self,
        mod: _NormBase,
        xin: torch.Tensor,
        g: torch.Tensor,
        kernel: torch.Tensor,
        w_idx: torch.Tensor,
        b_idx: torch.Tensor | None,
    ) -> None:
        """Frozen-stats norm affine restricted to selected channels."""
        g_gamma, g_beta = self._norm_affine_grads(mod, xin, g)
        if w_idx.numel():
            u = g_gamma.to(self.gram_dtype).index_select(1, w_idx)
            kernel += u @ u.T
        if b_idx is not None and b_idx.numel():
            u = g_beta.to(self.gram_dtype).index_select(1, b_idx)
            kernel += u @ u.T

    # ------------------------------------------------------------------
    # Chunked capture
    # ------------------------------------------------------------------

    def _param_groups(self) -> list[list[tuple[str, torch.Size, int]]]:
        """Partition parameter tensors into groups of <= ``chunk_numel`` elements.

        A tensor at or above the budget forms its own group and is handled by
        batch-chunking its ``jacrev`` instead.
        """
        groups: list[list[tuple[str, torch.Size, int]]] = []
        current: list[tuple[str, torch.Size, int]] = []
        current_n = 0
        for name, shape, n in self.param_shapes:
            if n >= self.chunk_numel:
                if current:
                    groups.append(current)
                    current, current_n = [], 0
                groups.append([(name, shape, n)])
            elif current_n + n > self.chunk_numel:
                groups.append(current)
                current, current_n = [(name, shape, n)], n
            else:
                current.append((name, shape, n))
                current_n += n
        if current:
            groups.append(current)
        return groups

    def _chunked_gram(
        self, x: torch.Tensor, args: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``G`` via per-group dict-``jacrev``: exact for any architecture.

        With an active ``param_mask`` each group's masked columns are
        gathered from its ``(M, P_grp)`` block before the contraction —
        groups are contiguous flat spans, so the mask localises by slicing,
        and groups with no masked columns skip their ``jacrev`` entirely.
        The group backward itself is unchanged by the mask (compute equals
        the unmasked capture; only the gathered contraction shrinks).
        """
        detached = self.params.detach()
        starts = {name: start for name, _, start in self.param_names_counts_startIdx}
        views = {
            name: detached[start : start + n].view(shape)
            for (name, shape, _), (_, n, start) in zip(
                self.param_shapes, self.param_names_counts_startIdx
            )
        }

        def f(
            active_: dict[str, torch.Tensor], x_: torch.Tensor, *args_: torch.Tensor
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            param_dict = {**views, **active_}
            for bname, buffer in self.model.named_buffers():
                param_dict[bname] = buffer
            pred = functional_call(self.model, param_dict, x_)
            loss = self.loss_fn(pred, *args_)
            rows_ = self._group_losses(loss).pow(self.kappa / 2.0)
            return rows_, (loss, pred)

        m = x.shape[0] // self.microbatch_size
        gram = torch.zeros(m, m, dtype=self.gram_dtype, device=self.device)
        losses: torch.Tensor | None = None
        preds: torch.Tensor | None = None
        for group in self._param_groups():
            local_idx: torch.Tensor | None = None
            if self.param_mask is not None:
                g_start = starts[group[0][0]]
                g_end = starts[group[-1][0]] + group[-1][2]
                local_idx = self.param_mask[g_start:g_end].nonzero(as_tuple=True)[0]
                if not local_idx.numel():
                    continue  # no masked columns here: skip the jacrev entirely
            active = {name: views[name] for name, _, _ in group}
            chunk_size = None
            if len(group) == 1 and group[0][2] >= self.chunk_numel:
                chunk_size = max(1, self.chunk_numel // group[0][2])
            jac, (loss, pred) = torch.func.jacrev(
                f, argnums=0, has_aux=True, chunk_size=chunk_size
            )(active, x, *args)
            j_grp = torch.cat(
                [jac[name].reshape(m, -1) for name, _, _ in group], dim=1
            )
            if local_idx is not None:
                j_grp = j_grp.index_select(1, local_idx)
            j_grp = j_grp.to(self.gram_dtype)
            gram += j_grp @ j_grp.T
            del jac, j_grp
            if losses is None:
                losses, preds = loss.detach(), pred.detach()
        if losses is None:  # every group skipped (empty mask): plain forward
            with torch.no_grad():
                preds = self._func_call(self.params, x)
                losses = self.loss_fn(preds, *args)
        return gram, losses, preds

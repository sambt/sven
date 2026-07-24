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
  (Linear: ``K += (g_l g_l^T) * (x_l x_l^T)``).  Exact only when per-sample
  losses do not interact across samples, hence the norm-stat/dropout guards.
- ``"chunked"``: per-parameter-group ``jacrev`` over at most ``chunk_numel``
  active parameters at a time; genuinely ``(M, n_group)`` memory and exact
  for any architecture, including batch-coupled normalisation.

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
    loudly — use :class:`sven.opt.SvenGram`.  ``param_fraction`` is forced to
    ``1.0``: the update always spans all parameters.

    Args:
        model: The PyTorch model to wrap.
        loss_fn: A loss function ``(pred, *args) -> Tensor`` that returns
            **per-sample** losses with shape ``(B,)``.
        device: Device to place the model and parameters on.
        kappa: Exponent for the raw loss, as in :class:`SvenWrapper`.
        microbatch_size: If ``> 1``, aggregate losses within sub-batches of
            this size before the ``kappa`` power, reducing the row dimension.
        capture: ``"hooks"`` (fast, per-sample-decoupled architectures only)
            or ``"chunked"`` (exact for any architecture).
        gram_dtype: Accumulation dtype for ``G``; keep float64 for fp32 models.
        freeze_norm_stats: Switch ``_NormBase`` modules to eval (running
            stats) during every wrapper pass, removing cross-sample coupling.
        chunk_numel: ``"chunked"`` mode only — max total active-parameter
            numel differentiated per ``jacrev`` group.
    """

    _CONV_BLOCK_ELEMS: int = 2 ** 26  # cap on a materialised (chunk, P_l) conv grad block

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable[..., torch.Tensor],
        device: torch.device | str,
        kappa: float = 2.0,
        microbatch_size: int = 1,
        capture: str = "hooks",
        gram_dtype: torch.dtype = torch.float64,
        freeze_norm_stats: bool = True,
        chunk_numel: int = 2 ** 22,
    ) -> None:
        if capture not in ("hooks", "chunked"):
            raise ValueError(f"capture must be 'hooks' or 'chunked', got {capture!r}")
        super().__init__(
            model,
            loss_fn,
            device,
            kappa=kappa,
            param_fraction=1.0,
            microbatch_size=microbatch_size,
        )
        self.capture: str = capture
        self.gram_dtype: torch.dtype = gram_dtype
        self.freeze_norm_stats: bool = freeze_norm_stats
        self.chunk_numel: int = chunk_numel

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
        losses, not the microbatch-grouped rows.

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
        batch-coupled ones.

        Args:
            w: ``(M,)`` row-space weights, e.g. ``U_k S_k^{-2} U_k^T r``.

        Returns:
            Flat ``(P,)`` update in the parameter dtype, detached.
        """
        if self._batch is None:
            raise RuntimeError("delta_from_w requires a stored batch; call loss_and_grad first")
        x, *args = self._batch
        w = w.detach().to(device=self.params.device, dtype=self.params.dtype)
        with torch.enable_grad(), self._frozen_norm_stats():
            rows, _ = self._loss(self.params, x, *args)
            (delta,) = torch.autograd.grad((w * rows).sum(), self.params)
        return delta.detach()

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
        """Hooks capture requires per-sample-decoupled Linear/Conv2d/_NormBase layers."""
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
            elif not isinstance(mod, nn.Linear):
                if any(True for _ in mod.parameters(recurse=False)):
                    raise NotImplementedError(
                        f"module '{name or '<root>'}' ({type(mod).__name__}) holds "
                        "parameters but is not Linear/Conv2d/_NormBase; hooks capture "
                        "cannot form its per-sample gradients — use capture='chunked'"
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
            if isinstance(mod, (nn.Linear, nn.Conv2d)) or (
                isinstance(mod, _NormBase) and mod.weight is not None
            ):
                store = {"name": name, "mod": mod, "calls": []}
                stores.append(store)
                handles.append(mod.register_forward_hook(mk_hook(store)))

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
            z_list = [store["calls"][0][1] for store in stores]
            gs = torch.autograd.grad(rows.sum(), z_list, allow_unused=True)
        finally:
            for handle in handles:
                handle.remove()

        b = x.shape[0]
        kernel = torch.zeros(b, b, dtype=self.gram_dtype, device=self.device)
        for store, g in zip(stores, gs):
            if g is None:  # output does not reach the loss
                continue
            xin = store["calls"][0][0].detach()
            g = g.detach()
            mod = store["mod"]
            if isinstance(mod, nn.Linear):
                self._accum_linear(mod, xin, g, kernel)
            elif isinstance(mod, nn.Conv2d):
                self._accum_conv2d(mod, xin, g, kernel)
            else:
                self._accum_norm_affine(mod, xin, g, kernel)
        return kernel, loss.detach(), pred.detach()

    def _accum_linear(
        self, mod: nn.Linear, xin: torch.Tensor, g: torch.Tensor, kernel: torch.Tensor
    ) -> None:
        """``K += (g g^T) * (x x^T)``, plus ``g g^T`` for the bias block."""
        if xin.dim() != 2:
            raise NotImplementedError(
                f"Linear input with {xin.dim()} dims; hooks capture supports 2D "
                "inputs only — use capture='chunked'"
            )
        gd = g.to(self.gram_dtype)
        xd = xin.to(self.gram_dtype)
        gg = gd @ gd.T
        kernel += gg * (xd @ xd.T)
        if mod.bias is not None:
            kernel += gg

    def _accum_conv2d(
        self, mod: nn.Conv2d, xin: torch.Tensor, g: torch.Tensor, kernel: torch.Tensor
    ) -> None:
        """Unfold-based per-sample conv grads, contracted immediately.

        The batch is chunked whenever the materialised ``(chunk, P_l)`` block
        would exceed ``_CONV_BLOCK_ELEMS`` elements; off-diagonal chunk pairs
        recompute one side rather than holding the full block.
        """
        b = g.shape[0]
        gf = g.reshape(b, mod.out_channels, -1).to(self.gram_dtype)  # (B, C_out, L)

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

        p_l = mod.weight.numel()
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
        if mod.bias is not None:
            gb = g.sum(dim=(2, 3)).to(self.gram_dtype)
            kernel += gb @ gb.T

    def _accum_norm_affine(
        self, mod: _NormBase, xin: torch.Tensor, g: torch.Tensor, kernel: torch.Tensor
    ) -> None:
        """Frozen-stats norm affine: grad_gamma = sum_sp g * xhat, grad_beta = sum_sp g."""
        shape = [1, -1] + [1] * (xin.dim() - 2)
        mean = mod.running_mean.view(shape).to(xin.dtype)
        var = mod.running_var.view(shape).to(xin.dtype)
        xhat = (xin - mean) / torch.sqrt(var + mod.eps)
        if xin.dim() > 2:
            spatial = tuple(range(2, xin.dim()))
            g_gamma = (g * xhat).sum(dim=spatial)
            g_beta = g.sum(dim=spatial)
        else:
            g_gamma = g * xhat
            g_beta = g
        g_gamma = g_gamma.to(self.gram_dtype)
        kernel += g_gamma @ g_gamma.T
        if mod.bias is not None:
            g_beta = g_beta.to(self.gram_dtype)
            kernel += g_beta @ g_beta.T

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
        """``G`` via per-group dict-``jacrev``: exact for any architecture."""
        detached = self.params.detach()
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
            active = {name: views[name] for name, _, _ in group}
            chunk_size = None
            if len(group) == 1 and group[0][2] >= self.chunk_numel:
                chunk_size = max(1, self.chunk_numel // group[0][2])
            jac, (loss, pred) = torch.func.jacrev(
                f, argnums=0, has_aux=True, chunk_size=chunk_size
            )(active, x, *args)
            j_grp = torch.cat(
                [jac[name].reshape(m, -1) for name, _, _ in group], dim=1
            ).to(self.gram_dtype)
            gram += j_grp @ j_grp.T
            del jac, j_grp
            if losses is None:
                losses, preds = loss.detach(), pred.detach()
        return gram, losses, preds

from __future__ import annotations

import inspect
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import torch
import torch.nn as nn
from torch.func import functional_call
from torch.nn.modules.batchnorm import _BatchNorm as _StockBatchNorm
from torch.nn.modules.batchnorm import _NormBase
from torch.nn.modules.instancenorm import _InstanceNorm
from torch.nn.utils import parameters_to_vector

from .masked_modules import RowMaskedConv2d, RowMaskedLinear, replace_with_row_masked


class SvenWrapper:
    """Functional wrapper around a PyTorch model for per-sample Jacobian computation.

    Converts a standard ``nn.Module`` into a functional form so that
    ``torch.func.jacrev`` can compute per-sample Jacobians of the loss with
    respect to a flat parameter vector.

    Memory notes: elementwise masking (``mask_by_block=False``) does **not**
    reduce the peak memory of the Jacobian computation — reverse-mode AD still
    materialises the full ``(B, P)`` cotangent before the masked gather; only
    the SVD input shrinks.  ``jac_chunk_size`` bounds peak memory in all
    ``jacrev`` paths, and ``mask_by_block=True`` differentiates only the
    selected tensors, giving genuine ``(B, n_active)`` memory.
    ``mask_mode="rows"`` differentiates only selected output rows/channels per
    module: genuine ``(B, n_active)`` memory at near-exact fractions.

    Args:
        model: The PyTorch model to wrap.
        loss_fn: A loss function ``(pred, *args) -> Tensor`` that returns
            **per-sample** losses with shape ``(B,)``.
        device: Device to place the model and parameters on.
        kappa: Exponent for raw loss function when computing the Jacobian and updates with L = (L^{kappa/2})^{2/kappa} (default: kappa = 2 for the usual derivatives of the raw loss function).
        residual_fn: Optional ``(pred, *args) -> Tensor`` returning one
            **signed scalar residual per sample**, shape ``(B,)`` or ``(B, 1)``
            (e.g. ``pred - y`` for scalar-output MSE with ``loss = r**2``).
            When given, the Jacobian rows are ``sign(r) * |r|**kappa`` instead
            of ``loss**(kappa/2) = |r|**kappa``. A per-row sign flip leaves the
            Sven update unchanged (it flips a Jacobian row and its residual
            together, which the pseudo-inverse absorbs), so the update is the
            same as the loss path up to floating-point rounding -- but its
            gradient is finite at ``r = 0`` for every ``kappa >= 1``, where
            ``loss**(kappa/2)`` has an ``inf * 0`` NaN for ``kappa < 2``.
            Losses without a scalar signed residual (cross-entropy,
            multi-output MSE such as label regression) must leave this
            ``None``. Incompatible with ``microbatch_size > 1``.
        param_fraction: Fraction of parameters to compute the Jacobian with
            respect to on each step.  ``1.0`` uses all parameters.
        mask_by_block: If ``True``, select **whole parameter tensors** when
            ``param_fraction < 1``: tensors are visited in random order and
            included while they fit in the remaining budget (at least one is
            always selected).  The achieved fraction is approximate — whole
            tensors only — and exposed as ``self.actual_param_fraction``.
        microbatch_size: If ``> 1``, aggregate losses within sub-batches of
            this size before computing the Jacobian, reducing its row dimension.
        jac_chunk_size: Chunk size forwarded to ``torch.func.jacrev``; caps
            the number of Jacobian rows materialised at once.  ``None``
            computes all rows in one pass.
        mask_mode: Mask structure when ``param_fraction < 1`` —
            ``"elementwise"`` (random individual entries), ``"tensor"``
            (whole parameter tensors, equivalent to ``mask_by_block=True``)
            or ``"rows"`` (random output rows/channels per Linear/Conv2d,
            resampled each step).  ``None`` derives the mode from
            ``mask_by_block`` for backwards compatibility.
        bn_mode: Normalisation-statistics policy (C-E2).  ``"batch"``
            (default here) trains with **batch** statistics and advances the
            running statistics from the training batch **exactly once per
            optimizer step**: every wrapper pass runs under
            :meth:`no_norm_stat_updates` and :meth:`loss_and_grad` performs
            one explicit ``no_grad`` train-mode forward that writes them.
            ``"frozen"`` runs every norm layer that owns running statistics in
            eval mode for every pass **and** for :meth:`evaluate_and_loss`, so
            no buffer is ever written.  :meth:`evaluate` is eval-mode and
            side-effect-free under both modes.
        freeze_norm_stats: Backward-compatible alias for ``bn_mode``
            (``True`` -> ``"frozen"``, ``False`` -> ``"batch"``); passing both
            is allowed only when they agree.
    """

    #: ``bn_mode`` when neither ``bn_mode`` nor ``freeze_norm_stats`` is given.
    #: The Jacobian wrapper has always trained with batch statistics.
    _DEFAULT_BN_MODE: str = "batch"

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable[..., torch.Tensor],
        device: torch.device | str,
        kappa: float = 2.0,
        param_fraction: float = 1.0,
        mask_by_block: bool = False,
        microbatch_size: int = 1,
        jac_chunk_size: int | None = None,
        mask_mode: str | None = None,
        residual_fn: Callable[..., torch.Tensor] | None = None,
        bn_mode: str | None = None,
        freeze_norm_stats: bool | None = None,
    ) -> None:
        self.model: nn.Module = model.to(device)
        self.device: torch.device = torch.device(device) if isinstance(device, str) else device
        self.loss_fn: Callable[..., torch.Tensor] = loss_fn
        self.kappa: float = kappa
        if residual_fn is not None and microbatch_size > 1:
            raise ValueError(
                "residual_fn is incompatible with microbatch_size > 1: a microbatch "
                "aggregates losses, and the mean of signed residuals is a different "
                "quantity (it can cancel). Use the loss path for microbatching."
            )
        self.residual_fn: Callable[..., torch.Tensor] | None = residual_fn
        self.bn_mode: str = self._resolve_bn_mode(bn_mode, freeze_norm_stats)
        # Norm modules owning running statistics; listed once on first use —
        # the module tree is fixed after the rows-mode surgery below.
        self._norm_stat_mods: list[_NormBase] | None = None

        if mask_mode is None:
            mask_mode = "tensor" if mask_by_block else "elementwise"
        if mask_mode not in ("elementwise", "tensor", "rows"):
            raise ValueError(
                f"mask_mode must be 'elementwise', 'tensor' or 'rows', got {mask_mode!r}"
            )
        self.mask_mode: str = mask_mode
        self.mask_by_block: bool = mask_mode == "tensor"

        # Rows-mode surgery must precede the flat tie: the tie rebinds every
        # parameter's storage, and the twins must be in place to receive it.
        if mask_mode == "rows" and param_fraction < 1.0:
            self._check_rows_supported()
            replace_with_row_masked(self.model)

        self.param_names_counts_startIdx: list[tuple[str, int, int]] = []
        self.params: torch.Tensor = self._tie_parameters_to_flat(requires_grad=False)
        self.params.requires_grad_(True)

        self.param_fraction: float = param_fraction
        self.param_mask: torch.Tensor | None = None
        self.actual_param_fraction: float = 1.0
        # Running mean of actual_param_fraction over steps (C-R4): plain
        # Python scalars, so reading it never syncs a device.
        self._apf_sum: float = 0.0
        self._apf_steps: int = 0
        self._block_selection: list[tuple[str, int, int]] = []
        self.microbatch_size: int = microbatch_size
        self.jac_chunk_size: int | None = jac_chunk_size
        self.n_params: int = self.params.shape[0]
        self.param_shapes: list[tuple[str, torch.Size, int]] = [
            (name, param.shape, param.numel()) for name, param in model.named_parameters()
        ]
        self._row_twins: list[tuple[str, nn.Module, int, int, int | None]] = (
            self._index_row_twins() if mask_mode == "rows" and param_fraction < 1.0 else []
        )
        self._row_selection: list[tuple[str, nn.Module, torch.Tensor]] = []

        self.num_loss_args: int = len(inspect.signature(loss_fn).parameters) - 1

        # Populated by loss_and_grad(), consumed by optimizer.step()
        self.grads: torch.Tensor = torch.empty(0, device=self.device)
        self.losses: torch.Tensor = torch.empty(0, device=self.device)

    # ------------------------------------------------------------------
    # Normalisation-statistics policy (C-E2)
    # ------------------------------------------------------------------

    def _resolve_bn_mode(self, bn_mode: str | None, freeze_norm_stats: bool | None) -> str:
        """Resolve ``bn_mode`` and its legacy ``freeze_norm_stats`` alias."""
        if freeze_norm_stats is not None:
            alias = "frozen" if freeze_norm_stats else "batch"
            if bn_mode is not None and bn_mode != alias:
                raise ValueError(
                    f"bn_mode={bn_mode!r} conflicts with "
                    f"freeze_norm_stats={freeze_norm_stats!r} (= {alias!r}); "
                    "pass only one of them"
                )
            bn_mode = alias
        if bn_mode is None:
            bn_mode = self._DEFAULT_BN_MODE
        if bn_mode not in ("batch", "frozen"):
            raise ValueError(f"bn_mode must be 'batch' or 'frozen', got {bn_mode!r}")
        return bn_mode

    @property
    def freeze_norm_stats(self) -> bool:
        """Legacy view of :attr:`bn_mode` (``True`` iff ``bn_mode == "frozen"``)."""
        return self.bn_mode == "frozen"

    def _norm_stat_modules(self) -> list[_NormBase]:
        """Norm modules that own running statistics (cached: the tree is fixed)."""
        if self._norm_stat_mods is None:
            self._norm_stat_mods = [
                mod
                for mod in self.model.modules()
                if isinstance(mod, _NormBase) and mod.running_mean is not None
            ]
        return self._norm_stat_mods

    def _check_batch_stat_norms(self) -> None:
        """Guard the batch-stat path's two unsupported norm layers.

        1. A **train-mode stock** ``nn.BatchNormNd``: its ``forward`` mutates
           ``num_batches_tracked`` **in place** whenever it runs in train mode
           with the statistics tracked, and ``torch.func`` transforms refuse
           that — the batch-statistics pipeline is validated only against a
           ``torch.func``-compatible replacement (sv3 vendors one in
           ``experiments/nn/batchnorm.py``).  Eval-mode norm layers never
           reach that branch, so a frozen stock module stays supported.
        2. ``InstanceNorm*d(track_running_stats=True)``: it hands its buffers
           to ``F.instance_norm`` unconditionally (so
           :meth:`no_norm_stat_updates` would not stop the write) and reads
           ``use_input_stats = training or not track_running_stats`` (so
           clearing the flag would change the normalisation).  Frozen mode is
           unaffected: there such a layer runs in eval mode.

        Every other running-stat layer is BatchNorm-like — torch's own
        ``_BatchNorm`` and the ``_NormBase`` subclasses that reimplement its
        forward (sv3's vendored one, this repo's test doubles): they gate on
        ``not self.training or self.track_running_stats``, which is exactly
        what :meth:`no_norm_stat_updates` relies on.
        """
        for mod in self._norm_stat_modules():
            if isinstance(mod, _InstanceNorm):
                raise NotImplementedError(
                    f"{type(mod).__name__} owns running statistics but writes "
                    "them regardless of track_running_stats (and changes its "
                    "normalisation when the flag is cleared), so bn_mode="
                    "'batch' cannot suppress its buffer writes — use "
                    "bn_mode='frozen'"
                )
            if isinstance(mod, _StockBatchNorm) and mod.training and mod.track_running_stats:
                raise NotImplementedError(
                    f"train-mode {type(mod).__name__} cannot run inside "
                    "torch.func transforms (its forward mutates "
                    "num_batches_tracked in place), so bn_mode='batch' needs a "
                    "torch.func-compatible BatchNorm — replace it (see "
                    "experiments/nn/batchnorm.replace_batchnorm in sv3) or use "
                    "bn_mode='frozen'"
                )

    @contextmanager
    def no_norm_stat_updates(self) -> Iterator[None]:
        """Suppress running-statistic writes for one pass, unchanged normalisation.

        Sets ``track_running_stats = False`` on every norm module that owns
        running statistics: a train-mode forward then still normalises with
        **batch** statistics (``F.batch_norm`` receives ``None`` buffers) and
        writes nothing, and an eval-mode one still normalises with the running
        statistics.  Unlike ``.eval()`` this never changes the normalisation,
        hence never changes the Gram matrix.  Each module's own previous flag
        and training mode are restored exactly.

        The contract holds for BatchNorm-like layers (they gate both the write
        and the normalisation on ``not training or track_running_stats``); it
        does **not** hold for ``InstanceNorm*d`` with running statistics,
        which :meth:`_check_batch_stat_norms` therefore rejects on the
        batch-stat path.
        """
        saved = [
            (mod, mod.track_running_stats, mod.training)
            for mod in self._norm_stat_modules()
        ]
        for mod, _, _ in saved:
            mod.track_running_stats = False
        try:
            yield
        finally:
            for mod, track, training in saved:
                mod.track_running_stats = track
                mod.training = training

    @contextmanager
    def _frozen_norm_stats(self) -> Iterator[None]:
        """Switch train-mode norm layers to eval (running stats) for one pass.

        A no-op unless ``bn_mode == "frozen"``.  Restores each module's own
        previous training flag — never a blanket ``.train()``, which would
        wake up layers the caller had deliberately put in eval mode.
        """
        saved: list[tuple[nn.Module, bool]] = []
        if self.bn_mode == "frozen":
            for mod in self.model.modules():
                if isinstance(mod, _NormBase) and mod.training:
                    saved.append((mod, mod.training))
                    mod.training = False
        try:
            yield
        finally:
            for mod, training in saved:
                mod.training = training

    @contextmanager
    def _pass_norm_stats(self) -> Iterator[None]:
        """The norm-statistics policy for one wrapper pass.

        Every repeated forward inside a step (capture, each ``jacrev`` group,
        ``jvp``, ``delta_from_w``, the ``variable_k`` line search) runs under
        this context, so none of them writes a running statistic: frozen mode
        normalises with the running statistics, batch mode with the batch
        statistics exactly as before this change.
        """
        if self.bn_mode == "frozen":
            with self._frozen_norm_stats():
                yield
        else:
            with self.no_norm_stat_updates():
                yield

    @contextmanager
    def _eval_mode(self) -> Iterator[None]:
        """Put the whole module in eval mode, restoring every previous flag."""
        saved = [(mod, mod.training) for mod in self.model.modules() if mod.training]
        for mod, _ in saved:
            mod.training = False
        try:
            yield
        finally:
            for mod, training in saved:
                mod.training = training

    @torch.no_grad()
    def update_norm_running_stats(self, batch: tuple[torch.Tensor, ...]) -> bool:
        """One train-mode forward of ``batch`` that advances the running statistics.

        This is the single writer of the norm buffers under
        ``bn_mode="batch"``; every other pass in the step is suppressed.  The
        forward is a plain (non-functional) call, so ``num_batches_tracked``
        advances too — under ``functional_call`` its out-of-place increment is
        reverted and the buffer stays at zero forever.

        Nothing is forced: only the norm layers a normal train-mode forward
        would write (train mode, statistics tracked) are affected, so a layer
        the caller put in eval mode stays frozen, and when no layer qualifies
        — no norm layer with running statistics, or all of them frozen — the
        forward is skipped entirely.

        Args:
            batch: Tuple of ``(x, y, ...)``; only ``x`` is used.

        Returns:
            ``True`` if the forward ran, ``False`` if it was skipped.
        """
        if not any(
            mod.training and mod.track_running_stats for mod in self._norm_stat_modules()
        ):
            return False
        self.model(batch[0])
        return True

    # ------------------------------------------------------------------
    # Forward / evaluation
    # ------------------------------------------------------------------

    def _func_call(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Functional forward pass through the model."""
        param_dict: dict[str, torch.Tensor] = {}
        start_idx = 0
        for name, shape, size in self.param_shapes:
            param_dict[name] = params[start_idx : start_idx + size].view(shape)
            start_idx += size
        # Fetch fresh buffers on every call (includes updated BatchNorm stats)
        for name, buffer in self.model.named_buffers():
            param_dict[name] = buffer
        return functional_call(self.model, param_dict, x)

    @torch.no_grad()
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Run a side-effect-free forward pass in eval mode.

        Eval mode under both ``bn_mode`` settings (C-E2): normalisation uses
        the running statistics, so a fixed example's prediction does not
        depend on its batch companions, no buffer is written, and every
        module's previous training flag is restored.
        """
        with self._eval_mode():
            return self._func_call(self.params, x)

    @torch.no_grad()
    def evaluate_and_loss(self, x: torch.Tensor, *args: torch.Tensor) -> torch.Tensor:
        """Per-sample losses under the CAPTURE normalisation, writing no buffer.

        Used by the ``variable_k`` line search, whose accept/reject test must
        see the same normalisation as the Jacobian it is stepping along — so
        this is *not* :meth:`evaluate`: under ``bn_mode="batch"`` it keeps
        train-mode batch statistics (with the running-stat writes suppressed),
        under ``"frozen"`` the frozen statistics.

        This resolves the one place where two CONTRACTS.md lines collide:
        "``evaluate_and_loss()`` = train-mode normalisation, no buffer write"
        and "``frozen`` = norm layers in eval mode **always** (train and
        eval)".  Under ``bn_mode="batch"`` both readings agree and this method
        is literally train-mode-no-write; under ``"frozen"`` the frozen
        decision wins, because there eval-mode normalisation *is* the model's
        training-time normalisation (a frozen run would otherwise line-search
        a quantity it never trains on).  The campaign is not affected either
        way: every scan config sets ``variable_k: false`` and
        ``grid.py`` rejects ``use_gram`` together with ``variable_k``.
        """
        with self._pass_norm_stats():
            pred = self._func_call(self.params, x)
        return self.loss_fn(pred, *args)

    # ------------------------------------------------------------------
    # Loss / Jacobian computation
    # ------------------------------------------------------------------

    def _group_losses(self, loss: torch.Tensor) -> torch.Tensor:
        """Aggregate per-sample losses into microbatch rows."""
        if self.microbatch_size > 1:
            loss = loss.view(-1, self.microbatch_size).mean(dim=1)
        return loss

    def _rows(self, pred: torch.Tensor, *args: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Jacobian/residual rows and the raw per-sample losses for ``pred``.

        Loss path (default): ``rows = group(loss) ** (kappa/2)`` (microbatch
        mean, then the ``kappa`` power).  Residual path (``residual_fn`` set):
        ``rows = sign(r) * |r| ** kappa`` with ``r = residual_fn(pred, *args)``
        flattened to ``(B,)`` (plain ``r`` at ``kappa = 1``). Returns
        ``(rows, loss)`` with ``loss`` the UNgrouped ``(B,)`` losses.
        """
        loss = self.loss_fn(pred, *args)
        if self.residual_fn is None:
            return self._group_losses(loss).pow(self.kappa / 2.0), loss
        r = self.residual_fn(pred, *args)
        if r.numel() != loss.shape[0]:
            raise ValueError(
                f"residual_fn must return one scalar residual per sample "
                f"(shape ({loss.shape[0]},) or ({loss.shape[0]}, 1)); got "
                f"{tuple(r.shape)}. Vector residuals are not supported -- use "
                "the loss path (residual_fn=None) for multi-output losses."
            )
        r = r.reshape(-1)
        if self.kappa == 1.0:
            return r, loss
        # sign(r) * |r|^kappa: autograd gives kappa*|r|^(kappa-1)*dr, finite at
        # r = 0 for kappa >= 1 (torch.sign has zero gradient).
        return torch.sign(r) * r.abs().pow(self.kappa), loss

    def _loss(
        self, params: torch.Tensor, x: torch.Tensor, *args: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Compute per-sample losses, returning aux data for ``jacrev``."""
        if self.param_mask is not None:
            input_params = self.params.clone()
            input_params[self.param_mask] = params
        else:
            input_params = params

        pred = self._func_call(input_params, x)
        rows, loss = self._rows(pred, *args)
        # aux: (grouped) losses for logging; rows drive the Jacobian
        return rows, (self._group_losses(loss), pred)

    def _batch_gradient(
        self, batch: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-sample Jacobian via ``jacrev``."""
        if self.param_mask is not None:
            if self.mask_mode == "rows":
                return self._batch_gradient_rows(batch)
            if self.mask_by_block:
                return self._batch_gradient_blocks(batch)
        x, *args = batch
        params = self.params[self.param_mask] if self.param_mask is not None else self.params
        grads, (losses, preds) = torch.func.jacrev(
            self._loss, argnums=0, has_aux=True, chunk_size=self.jac_chunk_size
        )(params, x, *args)
        return grads, losses, preds

    def _batch_gradient_blocks(
        self, batch: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-sample Jacobian w.r.t. the selected whole tensors via dict-``jacrev``.

        Differentiating a dict of the selected tensors keeps the cotangent at
        ``(rows, n_active)``; unselected tensors enter ``functional_call`` as
        detached constants.  Per-tensor pieces are concatenated in flat-vector
        order (ascending start index) — this must match the
        ``params[param_mask]`` gather used by the optimizer.
        """
        x, *args = batch
        detached = self.params.detach()
        views = {
            name: detached[start : start + n].view(shape)
            for (name, shape, _), (_, n, start) in zip(
                self.param_shapes, self.param_names_counts_startIdx
            )
        }
        active = {name: views[name] for name, _, _ in self._block_selection}

        def f(
            active_: dict[str, torch.Tensor], x_: torch.Tensor, *args_: torch.Tensor
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            param_dict = {**views, **active_}
            for bname, buffer in self.model.named_buffers():
                param_dict[bname] = buffer
            pred = functional_call(self.model, param_dict, x_)
            rows, loss = self._rows(pred, *args_)
            return rows, (self._group_losses(loss), pred)

        jac, (losses, preds) = torch.func.jacrev(
            f, argnums=0, has_aux=True, chunk_size=self.jac_chunk_size
        )(active, x, *args)
        rows = losses.shape[0]
        grads = torch.cat(
            [jac[name].reshape(rows, -1) for name, _, _ in self._block_selection], dim=1
        )
        return grads, losses, preds

    def _batch_gradient_rows(
        self, batch: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-sample Jacobian w.r.t. the selected rows via the split forward.

        The differentiated argument is a dict of the selected weight-row and
        bias-entry values; the traced function injects them into the twins and
        calls the model directly, so frozen rows come from the twins' own tied
        parameters as detached constants — the cotangent stays at
        ``(rows, n_active)``.  Twin pieces are concatenated in flat-vector
        order (twins ascending, weight rows before bias, rows sorted) — this
        must match the ``params[param_mask]`` gather used by the optimizer.
        """
        x, *args = batch
        active: dict[str, torch.Tensor] = {}
        keys: list[tuple[nn.Module, str, str | None]] = []
        for path, mod, rows in self._row_selection:
            # Build the frozen complement on CPU (rows is a CPU tensor from
            # sampling) BEFORE moving to device: CUDA indices into a CPU bool
            # tensor are rejected by index_put_.
            frozen = torch.ones(mod.weight.shape[0], dtype=torch.bool)
            frozen[rows] = False
            rows = rows.to(self.device)
            mod._active_rows = rows
            mod._frozen_rows = frozen.nonzero(as_tuple=True)[0].to(self.device)
            prefix = f"{path}." if path else ""
            w_key, b_key = f"{prefix}weight", f"{prefix}bias"
            active[w_key] = mod.weight.detach().index_select(0, rows)
            if mod.bias is not None:
                active[b_key] = mod.bias.detach().index_select(0, rows)
            keys.append((mod, w_key, b_key if mod.bias is not None else None))

        def f(
            active_: dict[str, torch.Tensor], x_: torch.Tensor, *args_: torch.Tensor
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            for mod, w_key, b_key in keys:
                mod._active_weight = active_[w_key]
                mod._active_bias = active_[b_key] if b_key is not None else None
            pred = self.model(x_)
            rows, loss = self._rows(pred, *args_)
            return rows, (self._group_losses(loss), pred)

        try:
            jac, (losses, preds) = torch.func.jacrev(
                f, argnums=0, has_aux=True, chunk_size=self.jac_chunk_size
            )(active, x, *args)
        finally:
            # evaluate() and later passes must see clean twins
            for mod, _, _ in keys:
                mod._active_rows = mod._frozen_rows = None
                mod._active_weight = mod._active_bias = None

        rows_dim = losses.shape[0]
        pieces: list[torch.Tensor] = []
        for _, w_key, b_key in keys:
            pieces.append(jac[w_key].reshape(rows_dim, -1))
            if b_key is not None:
                pieces.append(jac[b_key])
        grads = torch.cat(pieces, dim=1)
        return grads, losses, preds

    def loss_and_grad(
        self, batch: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute losses and per-sample Jacobian for the optimizer.

        The Jacobian is stored in ``self.grads`` and losses in ``self.losses``
        for consumption by :meth:`Sven.step`.

        This call starts an optimizer step, so under ``bn_mode="batch"`` it is
        where the single running-statistic update of the step happens; the
        Jacobian pass itself then runs with the writes suppressed.

        Args:
            batch: Tuple of ``(x, y, ...)`` tensors.

        Returns:
            ``(losses, predictions)`` — both detached from the compute graph.
        """
        if self.param_fraction < 1.0:
            if self.mask_mode == "rows":
                self.param_mask = self._make_param_mask_by_rows(self.param_fraction).to(
                    self.params.device
                )
            elif self.mask_by_block:
                self.param_mask = self._make_param_mask_by_block(self.param_fraction).to(
                    self.params.device
                )
            else:
                self.param_mask = self._make_param_mask().to(self.params.device)
            self._record_actual_param_fraction()

        # The mask is drawn first so the RNG stream is untouched by the policy.
        if self.bn_mode == "batch":
            self._check_batch_stat_norms()
            self.update_norm_running_stats(batch)

        with self._pass_norm_stats():
            grads, losses, preds = self._batch_gradient(batch)

        self.grads = grads.detach()
        self.losses = losses.detach()
        if self.residual_fn is None:
            self.residuals = self.losses.pow(self.kappa / 2.0).detach() # store the "residuals" (loss^(kappa/2)) for use in the update step
        else:
            with torch.no_grad():
                self.residuals = self._rows(preds, *batch[1:])[0].detach()

        return self.losses, preds

    # ------------------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------------------

    def _tie_parameters_to_flat(self, requires_grad: bool = False) -> torch.Tensor:
        """Flatten all model parameters into a single vector and rebind them as views."""
        flat = parameters_to_vector(self.model.parameters()).detach()
        flat = flat.requires_grad_(requires_grad)

        start = 0
        for name, p in self.model.named_parameters():
            n = p.numel()
            view = flat[start : start + n].view_as(p)
            self.param_names_counts_startIdx.append((name, n, start))
            start += n

            # Walk to the owning module and replace the parameter storage
            mod: nn.Module = self.model
            *prefix, leaf = name.split(".")
            for part in prefix:
                mod = getattr(mod, part)
            mod._parameters[leaf] = nn.Parameter(view, requires_grad=requires_grad)

        return flat

    def _record_actual_param_fraction(self) -> None:
        """Fold this step's ``actual_param_fraction`` into the running mean."""
        self._apf_sum += float(self.actual_param_fraction)
        self._apf_steps += 1

    @property
    def mean_actual_param_fraction(self) -> float:
        """Mean ``actual_param_fraction`` over the steps taken so far (C-R4).

        The mask is redrawn every step, so a single attribute cannot answer
        "which fraction did this run actually use"; this is the running mean,
        accumulated as Python floats (no device sync).  Before the first
        masked step it falls back to the *requested* ``param_fraction``, so a
        masked run read too early cannot be mistaken for an unmasked one
        (``1.0``); a NaN sentinel is deliberately avoided — this value goes
        into the jsonl records.
        """
        if self._apf_steps == 0:
            return float(self.param_fraction)
        return self._apf_sum / self._apf_steps

    def _make_param_mask(self) -> torch.Tensor:
        """Create a random mask selecting ``param_fraction`` of parameters."""
        n_active = int(self.param_fraction * self.n_params)
        mask = torch.zeros(self.n_params, dtype=torch.bool)
        mask[torch.randperm(self.n_params)[:n_active]] = True
        # Exactly n_active entries are set, so no mask.sum() (device sync) is
        # needed; without this the elementwise path reported 1.0 forever (F31).
        self.actual_param_fraction = n_active / self.n_params
        return mask

    def _make_param_mask_by_block(self, fraction: float) -> torch.Tensor:
        """Select whole parameter tensors totalling at most ``fraction`` of parameters.

        Tensors are visited in random order and included whenever they fit in
        the remaining budget; at least one tensor (the smallest) is always
        selected.  The achieved fraction is approximate (whole tensors only)
        and stored in ``self.actual_param_fraction``; the selection, sorted by
        flat start index, is stored in ``self._block_selection``.
        """
        blocks = self.param_names_counts_startIdx
        target = int(fraction * self.n_params)
        selected: list[tuple[str, int, int]] = []
        running = 0
        for i in torch.randperm(len(blocks)):
            name, nparam, start_idx = blocks[i]
            if running + nparam <= target:
                selected.append((name, nparam, start_idx))
                running += nparam
        if not selected:
            selected.append(min(blocks, key=lambda blk: blk[1]))
            running = selected[0][1]

        selected.sort(key=lambda blk: blk[2])  # flat order: must match params[mask] gather
        self._block_selection = selected
        self.actual_param_fraction = running / self.n_params

        mask = torch.zeros(self.n_params, dtype=torch.bool)
        for _, nparam, start_idx in selected:
            mask[start_idx : start_idx + nparam] = True
        return mask

    # ------------------------------------------------------------------
    # Row-block masking
    # ------------------------------------------------------------------

    def _check_rows_supported(self) -> None:
        """Rows-mode surgery replaces exact ``nn.Linear``/``nn.Conv2d`` modules only."""
        supported = (nn.Linear, nn.Conv2d, RowMaskedLinear, RowMaskedConv2d)
        for name, mod in self.model.named_modules():
            if type(mod) in supported:
                continue
            if any(True for _ in mod.parameters(recurse=False)):
                raise NotImplementedError(
                    f"module '{name or '<root>'}' ({type(mod).__name__}) holds "
                    "parameters but is not nn.Linear/nn.Conv2d; rows-mode masking "
                    "cannot split it (norm-affine support is future work)"
                )

    def _index_row_twins(self) -> list[tuple[str, nn.Module, int, int, int | None]]:
        """Locate the mask-aware twins and their flat-vector offsets.

        Returns:
            ``(path, module, weight start, per-row numel, bias start)`` per
            twin, sorted by weight start — ascending flat order, which the
            rows-mode Jacobian assembly relies on.
        """
        offsets = {name: start for name, _, start in self.param_names_counts_startIdx}
        twins: list[tuple[str, nn.Module, int, int, int | None]] = []
        for path, mod in self.model.named_modules():
            if isinstance(mod, (RowMaskedLinear, RowMaskedConv2d)):
                prefix = f"{path}." if path else ""
                w_start = offsets[f"{prefix}weight"]
                row_numel = mod.weight.numel() // mod.weight.shape[0]
                b_start = offsets[f"{prefix}bias"] if mod.bias is not None else None
                twins.append((path, mod, w_start, row_numel, b_start))
        if not twins:
            raise NotImplementedError(
                "mask_mode='rows' found no nn.Linear/nn.Conv2d submodules to mask"
            )
        twins.sort(key=lambda twin: twin[2])
        return twins

    def _make_param_mask_by_rows(self, fraction: float) -> torch.Tensor:
        """Sample ``fraction`` of output rows per twin and build the flat mask.

        Each twin draws ``max(1, round(fraction * out))`` distinct rows; the
        mask covers the selected weight rows plus matching bias entries.  The
        achieved fraction is stored in ``self.actual_param_fraction`` and the
        selection, rows sorted ascending, in ``self._row_selection``.
        """
        mask = torch.zeros(self.n_params, dtype=torch.bool)
        selection: list[tuple[str, nn.Module, torch.Tensor]] = []
        for path, mod, w_start, row_numel, b_start in self._row_twins:
            out_dim = mod.weight.shape[0]
            n_rows = max(1, int(round(fraction * out_dim)))
            rows = torch.randperm(out_dim)[:n_rows].sort().values
            selection.append((path, mod, rows))
            for o in rows.tolist():
                mask[w_start + o * row_numel : w_start + (o + 1) * row_numel] = True
            if b_start is not None:
                mask[b_start + rows] = True
        self._row_selection = selection
        self.actual_param_fraction = mask.sum().item() / self.n_params
        return mask

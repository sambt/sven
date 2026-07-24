"""Gram/kernel-trick wrapper for JAX: Sven updates without the Jacobian.

With ``J = U S V^T`` the thin SVD of the ``(M, P)`` residual Jacobian, the
Sven update factors through the tiny Gram matrix ``G = J J^T = U S^2 U^T``:

    delta = V_k diag(1/sigma_k) U_k^T r
          = J^T [ U_k diag(1/sigma_k^2) U_k^T r ]
          = J^T w,

and ``J^T w`` is a single reverse-mode VJP of the residual function — one
standard backward. The ``(M, P)`` Jacobian is never materialised in full.

``G`` is accumulated per parameter *group*: pytree leaves are partitioned at
construction into contiguous groups of at most ``group_numel`` entries; per
group, ``jacrev`` differentiates the residuals w.r.t. only that sub-pytree
(remaining leaves behind ``lax.stop_gradient``), giving a genuinely
``(M, P_grp)`` block whose product ``J_grp J_grp^T`` is contracted on device
and immediately discarded. Because each block is real AD, this is exact for
ANY ``apply_fn``, including batch-coupled ones (e.g. train-mode BatchNorm).
Microbatch grouping and ``kappa`` come free through the residual function:
the Gram of the grouped-residual Jacobian is what gets accumulated.

Stochastic parameter masking
----------------------------

With ``param_fraction < 1`` and a ``mask_mode`` (``"rows"`` / ``"tensor"`` /
``"elementwise"``), a fresh flat mask is drawn every :meth:`loss_and_grad`
via the samplers shared with :class:`sven.jax.SvenWrapper` (same key =>
identical ``mask_indices`` in both pipelines). All granularities are handled
uniformly: since the per-group block ``J_grp`` exists before contraction,
each group's masked columns are gathered from the ``(M, P_grp)`` block and
``G += J_A_grp J_A_grp^T`` is accumulated.

Memory is **group-bounded**, not constant in the fraction: the gather adds a
transient ``(M, n_masked_grp)`` copy on top of the ``(M, P_grp)`` block, so
peak memory grows monotonically with the fraction (measured +46% from
fraction 0.1 to 1.0 on a model whose largest leaf exceeds ``group_numel``),
and a leaf larger than ``group_numel`` forms its own oversized group that
dominates the peak regardless of masking. It stays within ~20% across
fractions only when every leaf fits ``group_numel``. Masked never exceeds
unmasked. (Sub-leaf chunking of oversized leaves — jacrev over leading-axis
slices — would tighten the bound; future work. The torch hooks-capture
masked gram IS fraction-constant; prefer it where applicable.)

Groups containing no masked columns skip their ``jacrev`` entirely (with
small fractions and tensor masks this saves most groups). The masked group
function recompiles once per (group, masked-column-count) pair: counts are
deterministic per fraction for ``"rows"``, vary with the leaf selection for
``"tensor"``, and vary hypergeometrically per draw for ``"elementwise"`` —
elementwise masked gram can therefore recompile every group for the first
several steps; prefer ``"rows"``/``"tensor"`` in this pipeline. The masked
update is the full ``J^T w`` VJP gathered at ``mask_indices``, and
:meth:`apply_update` scatters it back.

Precision
---------

``cond(G) = cond(J)^2``, so ``G`` wants more precision than the model. Each
group product is contracted in the device dtype, converted to numpy float64
host-side, and the group products are summed (and later eigendecomposed) in
float64 — ``M x M`` is tiny. The within-group contraction remains limited by
the device dtype: without ``jax_enable_x64`` everything on device is float32,
which is fine at the package default ``rtol=1e-3`` but loses small singular
directions at ``rtol <~ 1e-6``. For tight tolerances enable
``jax.config.update("jax_enable_x64", True)``.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from .wrapper import (
    leafset_mask_indices,
    make_leafset_residual_fn,
    rows_mask_indices,
    sample_elementwise_indices,
    sample_leading_rows,
    select_block_leaves,
)


class GramSvenWrapper:
    """Functional wrapper accumulating the residual Gram matrix ``J J^T``.

    Drop-in alternative to :class:`SvenWrapper` for use with
    :class:`sven.jax.SvenGram`: ``loss_and_grad`` fills ``.gram`` (numpy
    float64, shape ``(M, M)``) and ``.residuals`` instead of ``.jac``, which
    stays ``None``. With ``param_fraction < 1`` a fresh mask is drawn every
    :meth:`loss_and_grad` (a ``key`` is then required), ``.gram`` is the
    masked ``J_A J_A^T``, and the update is restricted to ``mask_indices``.

    Args:
        apply_fn: Forward function ``(params, x) -> preds``.
        params: Initial parameter pytree.
        loss_fn: ``(preds, *args) -> per_sample_losses`` with shape ``(B,)``.
        kappa: Exponent such that updates are computed on ``loss ** (kappa/2)``
            (default 2.0 = gradients of the raw loss).
        param_fraction: Fraction of parameters to restrict the Gram/update to,
            resampled each :meth:`loss_and_grad`. ``1.0`` uses all parameters.
        microbatch_size: Aggregate losses into groups of this size (mean) to
            shrink the Gram's row dimension ``M``.
        group_numel: Max total leaf entries differentiated at once; bounds the
            transient per-group Jacobian at ``M * group_numel`` device floats
            (masked or not — the block is built before the column gather).
            A single leaf larger than this gets its own group.
        mask_mode: Mask structure when ``param_fraction < 1`` — ``"rows"``
            (random leading-axis slices per leaf, sampler shared with
            :class:`SvenWrapper`), ``"tensor"`` (whole pytree leaves) or
            ``"elementwise"`` (random flat subset). Required when
            ``param_fraction < 1``; ignored at ``1.0``.
    """

    def __init__(
        self,
        apply_fn: Callable[..., Any],
        params: Any,
        loss_fn: Callable[..., jnp.ndarray],
        *,
        kappa: float = 2.0,
        param_fraction: float = 1.0,
        microbatch_size: int = 1,
        group_numel: int = 2**22,
        mask_mode: str | None = None,
    ) -> None:
        self.apply_fn = apply_fn
        self.loss_fn = loss_fn
        self.kappa = float(kappa)
        self.param_fraction = float(param_fraction)
        if self.param_fraction < 1.0 and mask_mode not in (
            "rows",
            "tensor",
            "elementwise",
        ):
            raise ValueError(
                "param_fraction < 1 requires mask_mode 'rows', 'tensor' or "
                f"'elementwise', got {mask_mode!r}"
            )
        self.mask_mode: str | None = mask_mode if self.param_fraction < 1.0 else None
        self.microbatch_size = int(microbatch_size)
        self.group_numel = int(group_numel)
        if self.group_numel < 1:
            raise ValueError("group_numel must be >= 1")

        flat, unravel = ravel_pytree(params)
        self.flat_params: jnp.ndarray = flat
        self._unravel = unravel
        self.n_params: int = int(flat.size)
        self._treedef = jax.tree_util.tree_structure(params)

        leaves = jax.tree_util.tree_leaves(params)
        self._n_leaves = len(leaves)
        self._leaf_shapes = tuple(tuple(l.shape) for l in leaves)
        sizes = [int(np.prod(l.shape)) for l in leaves]
        self._leaf_sizes = np.asarray(sizes, dtype=np.int64)
        self._leaf_starts = np.concatenate([[0], np.cumsum(self._leaf_sizes)])[:-1]

        # Fixed contiguous leaf groups (fixed shapes -> each group fn compiles
        # exactly once).
        groups: list[tuple[int, ...]] = []
        cur: list[int] = []
        cur_numel = 0
        for i, s in enumerate(sizes):
            if cur and cur_numel + s > self.group_numel:
                groups.append(tuple(cur))
                cur, cur_numel = [], 0
            cur.append(i)
            cur_numel += s
        if cur:
            groups.append(tuple(cur))
        self._groups = groups
        # Flat (ravel-offset) span covered by each group's columns; groups are
        # contiguous leaf runs, so a sorted mask slices per group cheaply.
        self._group_spans = [
            (
                int(self._leaf_starts[g[0]]),
                int(self._leaf_starts[g[-1]] + self._leaf_sizes[g[-1]]),
            )
            for g in groups
        ]
        self._group_fns = [self._build_group_fn(g) for g in groups]
        self._delta_fn = self._build_delta_fn()

        # Mask state; resampled on every masked ``loss_and_grad``.
        self.mask_indices: jnp.ndarray | None = None
        self._mask_np: np.ndarray | None = None
        self.actual_param_fraction: float | None = (
            None if self.mask_mode is not None else 1.0
        )

        # Populated by ``loss_and_grad``; consumed by ``SvenGram.step``.
        self.losses: jnp.ndarray | None = None
        self.preds: Any = None
        self.residuals: jnp.ndarray | None = None
        self.gram: np.ndarray | None = None
        self.jac: None = None
        self._last_batch: tuple[Any, tuple] | None = None

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    @property
    def params(self):
        return self._unravel(self.flat_params)

    def evaluate(self, x):
        return self.apply_fn(self.params, x)

    def evaluate_and_loss(self, x, *args):
        return self.loss_fn(self.apply_fn(self.params, x), *args)

    # ------------------------------------------------------------------
    # Compiled group / VJP functions
    # ------------------------------------------------------------------

    def _build_group_fn(self, group: tuple[int, ...]):
        """Jitted group contractions, unmasked and masked.

        Returns ``(fn, masked_fn, frozen_ids)``: ``fn(active, frozen, x,
        args)`` contracts the full block ``J_grp J_grp^T``; ``masked_fn``
        additionally takes the group-local masked column indices and
        contracts ``J_A_grp J_A_grp^T`` after gathering those columns from
        the ``(M, P_grp)`` block. ``masked_fn`` recompiles once per masked
        column count (jit caches on the index shape); counts are
        deterministic per fraction only for ``"rows"`` masks — ``"tensor"``
        counts vary with the leaf selection and ``"elementwise"`` counts vary
        per draw, so those modes accumulate more traces (see module
        docstring).
        """
        residuals, frozen_ids = make_leafset_residual_fn(
            self.apply_fn,
            self.loss_fn,
            self.kappa,
            self.microbatch_size,
            self._treedef,
            self._n_leaves,
            group,
        )
        jac = jax.jacrev(residuals, argnums=0, has_aux=True)

        @jax.jit
        def _fn(active, frozen, x, args):
            jac_tree, aux = jac(active, frozen, x, args)
            J = jnp.concatenate(
                [j.reshape(j.shape[0], -1) for j in jac_tree], axis=1
            )
            return J @ J.T, aux

        @jax.jit
        def _masked_fn(active, frozen, local_idx, x, args):
            jac_tree, aux = jac(active, frozen, x, args)
            J = jnp.concatenate(
                [j.reshape(j.shape[0], -1) for j in jac_tree], axis=1
            )
            J_a = J[:, local_idx]
            return J_a @ J_a.T, aux

        return _fn, _masked_fn, frozen_ids

    def _residuals_flat(self, flat_params, x, args):
        params = self._unravel(flat_params)
        preds = self.apply_fn(params, x)
        losses = self.loss_fn(preds, *args)
        if self.microbatch_size > 1:
            losses = losses.reshape(-1, self.microbatch_size).mean(axis=1)
        return jnp.power(losses, self.kappa / 2.0)

    def _build_delta_fn(self):
        @jax.jit
        def _fn(flat_params, x, args, w):
            _, vjp = jax.vjp(lambda f: self._residuals_flat(f, x, args), flat_params)
            (delta,) = vjp(w)
            return delta

        return _fn

    # ------------------------------------------------------------------
    # Mask sampling (host-side — cheap integer work)
    # ------------------------------------------------------------------

    def _sample_mask(self, key: jax.Array) -> None:
        """Draw this call's ``mask_indices`` via the shared samplers.

        The same functions (and key protocol) back :class:`SvenWrapper`'s
        masked paths, so the same key yields identical ``mask_indices`` in
        the Jacobian and Gram pipelines.
        """
        if self.mask_mode == "rows":
            rows = sample_leading_rows(key, self._leaf_shapes, self.param_fraction)
            idx = rows_mask_indices(
                rows, self._leaf_starts, self._leaf_shapes, self._leaf_sizes
            )
        elif self.mask_mode == "tensor":
            selected = tuple(
                sorted(select_block_leaves(key, self._leaf_sizes, self.param_fraction))
            )
            idx = leafset_mask_indices(selected, self._leaf_starts, self._leaf_sizes)
        else:  # elementwise
            n_active = int(self.param_fraction * self.n_params)
            idx = sample_elementwise_indices(key, self.n_params, n_active)
        self._mask_np = idx  # sorted host copy for the per-group slicing
        self.mask_indices = jnp.asarray(idx, dtype=jnp.int32)
        self.actual_param_fraction = idx.size / self.n_params

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def loss_and_grad(
        self,
        batch: Sequence[jnp.ndarray],
        key: jax.Array | None = None,
    ) -> tuple[jnp.ndarray, Any]:
        """Accumulate ``G = J J^T`` group by group; never build ``J``.

        With a mask, ``G`` is the masked ``J_A J_A^T``: each group's masked
        columns are gathered from its ``(M, P_grp)`` block before the
        contraction, and groups with no masked columns are skipped entirely.

        Args:
            batch: ``(x, *loss_args)``.
            key: Required when ``param_fraction < 1``; a fresh mask is drawn
                every call.
        """
        x, *args = batch
        args = tuple(args)

        if self.mask_mode is not None:
            if key is None:
                raise ValueError("`key` is required when param_fraction < 1.0")
            self._sample_mask(key)

        leaves = jax.tree_util.tree_leaves(self.params)
        gram: np.ndarray | None = None
        aux = None
        for group, (fn, masked_fn, frozen_ids), (g_start, g_end) in zip(
            self._groups, self._group_fns, self._group_spans
        ):
            if self._mask_np is not None:
                lo, hi = np.searchsorted(self._mask_np, [g_start, g_end])
                if lo == hi:
                    continue  # no masked columns here: skip the jacrev entirely
                local_idx = jnp.asarray(
                    self._mask_np[lo:hi] - g_start, dtype=jnp.int32
                )
            active = tuple(leaves[i] for i in group)
            frozen = tuple(leaves[i] for i in frozen_ids)
            if self._mask_np is not None:
                G_grp, aux = masked_fn(active, frozen, local_idx, x, args)
            else:
                G_grp, aux = fn(active, frozen, x, args)
            # Sum group products host-side in float64: cond(G) = cond(J)^2.
            G_grp64 = np.asarray(G_grp, dtype=np.float64)
            gram = G_grp64 if gram is None else gram + G_grp64

        losses, preds = aux
        self.losses = losses
        self.preds = preds
        self.residuals = jnp.power(losses, self.kappa / 2.0)
        self.gram = gram
        self.jac = None
        self._last_batch = (x, args)
        return losses, preds

    def delta_from_w(self, w) -> jnp.ndarray:
        """``J^T w`` as one VJP of the residuals at the current params.

        With a mask, the full VJP still runs (one flat transient) and the
        gather at ``mask_indices`` is returned, matching the masked scatter
        in :meth:`apply_update`.

        Args:
            w: Row weights, shape ``(M,)``.

        Returns:
            Flat update direction, shape ``(n_params,)`` — ``(n_active,)``
            when a mask is active.
        """
        if self._last_batch is None:
            raise RuntimeError("Call `loss_and_grad(batch)` before `delta_from_w`.")
        x, args = self._last_batch
        w = jnp.asarray(w, dtype=self.flat_params.dtype)
        delta = self._delta_fn(self.flat_params, x, args, w)
        if self.mask_indices is not None:
            delta = delta[self.mask_indices]
        return delta

    def apply_update(self, delta: jnp.ndarray) -> None:
        """Apply ``delta`` to the flat params (``(n_active,)`` scattered at
        ``mask_indices`` when a mask is active, ``(n_params,)`` otherwise)."""
        if self.mask_indices is not None:
            self.flat_params = self.flat_params.at[self.mask_indices].add(delta)
        else:
            self.flat_params = self.flat_params + delta

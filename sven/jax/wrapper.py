"""SvenWrapper for JAX / Flax models.

Provides a functional wrapper that computes per-sample Jacobians of a loss
function with respect to a flat parameter vector. Mirrors the PyTorch
``SvenWrapper`` API so training loops translate directly.

Masked Jacobians
----------------

Three masking modes exist when ``param_fraction < 1.0``, selected by
``mask_mode`` (``None`` derives the legacy mode from ``mask_by_block``):

* Elementwise (``mask_mode="elementwise"``): a random subset of individual
  flat entries is differentiated by scattering the active slice into the
  frozen flat vector. **This saves neither memory nor compute**: reverse-mode
  AD materialises the full ``(B, n_params)`` cotangent before the gather (XLA
  HLO verified: concatenate -> full cotangent -> gather; FLOPs identical to
  the full Jacobian, temporaries slightly larger). Elementwise masking only
  shrinks the *returned* array — useful for the downstream SVD, not for the
  Jacobian computation itself.
* Structural (``mask_mode="tensor"``, legacy ``mask_by_block=True``): whole
  pytree leaves are selected at random and ``jacrev`` differentiates only
  that sub-pytree; the remaining leaves are constants behind
  ``lax.stop_gradient``. Real AD w.r.t. fewer inputs, so the Jacobian is
  genuinely ``(B, n_active)`` in both memory and compute, for any
  ``apply_fn``. Each distinct leaf selection has its own shapes and therefore
  its own compiled Jacobian function; these are cached on the tuple of
  selected leaf ids, and ``resample_every`` amortises resampling (hence
  recompilation) across steps.
* Rows (``mask_mode="rows"``): a random subset of **leading-axis slices** of
  every leaf is differentiated: ``active_l = leaf[rows_l]`` is scattered back
  via ``stop_gradient(leaf).at[rows_l].set(active_l)``. Honest memory caveat:
  the scatter's backward materialises a transient ``(B, P_leaf)`` cotangent
  per leaf before the gather — weaker than the torch wrapper's op-level
  split-matmul pruning, but bounded by the *largest leaf* rather than the
  full ``P`` (the masked ``GramSvenWrapper`` is the recommended path when
  memory is the constraint). Slice counts are deterministic per fraction
  (``max(1, round(fraction * leading_dim))``; 1-d leaves: that fraction of
  entries; 0-d leaves: always active), so all shapes are fixed and each
  fraction compiles ONCE — the indices are traced data, no per-selection
  cache. Note that "rows" means leading-axis slices, NOT torch's neuron-tied
  output rows: flax-convention ``(in, out)`` kernels get input-side slices,
  and bias entries are sampled independently of any kernel.

The host-side samplers live at module level and are shared with
:class:`sven.jax.GramSvenWrapper`, so the same key yields identical
``mask_indices`` in the Jacobian and Gram pipelines.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree


def make_leafset_residual_fn(
    apply_fn: Callable[..., Any],
    loss_fn: Callable[..., jnp.ndarray],
    kappa: float,
    microbatch_size: int,
    treedef: Any,
    n_leaves: int,
    selected: tuple[int, ...],
) -> tuple[Callable[..., Any], tuple[int, ...]]:
    """Residual function differentiating only the ``selected`` pytree leaves.

    Returns ``(residuals, frozen_ids)`` where ``residuals(active, frozen, x,
    args) -> (r, (losses, preds))`` takes ``active`` / ``frozen`` as tuples of
    leaf arrays (ids ``selected`` / ``frozen_ids``, in that order). Frozen
    leaves sit behind ``lax.stop_gradient`` so reverse-mode AD propagates
    cotangents only to the active leaves — the Jacobian w.r.t. ``active`` is
    genuinely restricted to those leaves.
    """
    frozen_ids = tuple(i for i in range(n_leaves) if i not in set(selected))

    def residuals(active, frozen, x, args):
        leaves: list[Any] = [None] * n_leaves
        for i, leaf in zip(selected, active):
            leaves[i] = leaf
        for i, leaf in zip(frozen_ids, frozen):
            leaves[i] = jax.lax.stop_gradient(leaf)
        params = jax.tree_util.tree_unflatten(treedef, leaves)
        preds = apply_fn(params, x)
        losses = loss_fn(preds, *args)
        if microbatch_size > 1:
            losses = losses.reshape(-1, microbatch_size).mean(axis=1)
        r = jnp.power(losses, kappa / 2.0)
        return r, (losses, preds)

    return residuals, frozen_ids


# ----------------------------------------------------------------------
# Shared host-side mask samplers (cheap integer work)
#
# Both SvenWrapper and GramSvenWrapper draw their masks through these
# functions, so the same key yields identical ``mask_indices`` in the
# Jacobian and Gram pipelines.
# ----------------------------------------------------------------------


def _np_rng_from_key(key: jax.Array) -> np.random.Generator:
    """Host-side numpy RNG derived from a JAX PRNG key."""
    seed = int(jax.random.randint(key, (), 0, np.iinfo(np.int32).max))
    return np.random.default_rng(seed)


def sample_elementwise_indices(
    key: jax.Array, n_params: int, n_active: int
) -> np.ndarray:
    """Sorted random subset of ``n_active`` out of ``n_params`` flat indices."""
    rng = _np_rng_from_key(key)
    idx = rng.choice(n_params, size=n_active, replace=False)
    idx.sort()
    return idx


def select_block_leaves(
    key: jax.Array, leaf_sizes: np.ndarray, fraction: float
) -> tuple[int, ...]:
    """Greedy random whole-leaf selection (include if it fits, else skip).

    Returns leaf ids in inclusion order; at least one leaf is guaranteed
    (budget below every leaf falls back to the smallest one).
    """
    rng = _np_rng_from_key(key)
    budget = fraction * int(leaf_sizes.sum())
    chosen: list[int] = []
    running = 0
    for i in rng.permutation(len(leaf_sizes)):
        s = int(leaf_sizes[i])
        if running + s <= budget:
            chosen.append(int(i))
            running += s
    if not chosen:
        chosen = [int(np.argmin(leaf_sizes))]
    return tuple(chosen)


def leading_row_counts(
    leaf_shapes: Sequence[tuple[int, ...]], fraction: float
) -> list[int]:
    """Deterministic per-leaf leading-axis slice counts (torch rows rule).

    ``max(1, round(fraction * leading_dim))`` per >= 1-d leaf; 0-d leaves are
    always fully active (count 1). Counts depend only on the shapes and the
    fraction, so rows-mode Jacobian shapes are fixed across resamples.
    """
    return [
        1 if len(shape) == 0 else max(1, int(round(fraction * shape[0])))
        for shape in leaf_shapes
    ]


def sample_leading_rows(
    key: jax.Array, leaf_shapes: Sequence[tuple[int, ...]], fraction: float
) -> list[np.ndarray | None]:
    """Sorted random leading-axis indices per leaf (``None`` = 0-d, fully active)."""
    rng = _np_rng_from_key(key)
    counts = leading_row_counts(leaf_shapes, fraction)
    rows: list[np.ndarray | None] = []
    for shape, n in zip(leaf_shapes, counts):
        if len(shape) == 0:
            rows.append(None)
            continue
        idx = rng.choice(shape[0], size=n, replace=False)
        idx.sort()
        rows.append(idx.astype(np.int64))
    return rows


def leafset_mask_indices(
    selected: Sequence[int], leaf_starts: np.ndarray, leaf_sizes: np.ndarray
) -> np.ndarray:
    """Flat (ravel-offset) indices of the ``selected`` leaves, in that order."""
    return np.concatenate(
        [
            np.arange(
                leaf_starts[i], leaf_starts[i] + leaf_sizes[i], dtype=np.int64
            )
            for i in selected
        ]
    )


def rows_mask_indices(
    rows: Sequence[np.ndarray | None],
    leaf_starts: np.ndarray,
    leaf_shapes: Sequence[tuple[int, ...]],
    leaf_sizes: np.ndarray,
) -> np.ndarray:
    """Flat (ravel-offset) indices of the selected leading-axis slices.

    Ascending: leaves in ravel order, rows sorted within each leaf, entries
    contiguous within each slice — the exact order the per-leaf Jacobian
    pieces are concatenated in.
    """
    pieces: list[np.ndarray] = []
    for r, start, shape, size in zip(rows, leaf_starts, leaf_shapes, leaf_sizes):
        if r is None:
            pieces.append(np.arange(start, start + size, dtype=np.int64))
        else:
            slice_numel = int(size) // shape[0]
            offs = (
                int(start)
                + r[:, None] * slice_numel
                + np.arange(slice_numel, dtype=np.int64)[None, :]
            )
            pieces.append(offs.reshape(-1))
    return np.concatenate(pieces)


class SvenWrapper:
    """Functional wrapper around a JAX/Flax model for per-sample Jacobians.

    Args:
        apply_fn: Forward function ``(params, x) -> preds``. For Flax Linen use
            ``model.apply``; for ``flax.nnx`` use ``nnx.call`` or split the
            model into (graphdef, state) and close over graphdef.
        params: Initial parameter pytree.
        loss_fn: ``(preds, *args) -> per_sample_losses`` with shape ``(B,)``.
        kappa: Exponent such that updates are computed on ``loss ** (kappa/2)``
            (default 2.0 = gradients of the raw loss).
        param_fraction: Fraction of parameters to differentiate each step.
        mask_by_block: Legacy switch: ``True`` selects whole pytree leaves
            (equivalent to ``mask_mode="tensor"``), ``False`` individual
            entries (``mask_mode="elementwise"``). Ignored when ``mask_mode``
            is given explicitly.
        microbatch_size: Aggregate losses into groups of this size (mean) to
            shrink the Jacobian's row dimension.
        resample_every: Resample the mask every this many ``loss_and_grad``
            calls. Structural masking compiles one Jacobian function per leaf
            selection, so values > 1 amortise recompilation (rows mode
            compiles once regardless).
        mask_mode: Mask structure when ``param_fraction < 1`` —
            ``"elementwise"`` (random flat entries, no memory saving),
            ``"tensor"`` (whole pytree leaves, genuine memory saving) or
            ``"rows"`` (random leading-axis slices per leaf; genuine saving
            up to a transient ``(B, P_leaf)`` cotangent per leaf, see module
            docstring). ``None`` derives the mode from ``mask_by_block`` for
            backwards compatibility. The realised fraction is exposed as
            ``actual_param_fraction`` (tensor: set per resample, at least one
            leaf always selected; rows: deterministic, set at construction).
    """

    def __init__(
        self,
        apply_fn: Callable[..., Any],
        params: Any,
        loss_fn: Callable[..., jnp.ndarray],
        *,
        kappa: float = 2.0,
        param_fraction: float = 1.0,
        mask_by_block: bool = False,
        microbatch_size: int = 1,
        resample_every: int = 1,
        mask_mode: str | None = None,
    ) -> None:
        self.apply_fn = apply_fn
        self.loss_fn = loss_fn
        self.kappa = float(kappa)
        self.param_fraction = float(param_fraction)
        if mask_mode is None:
            mask_mode = "tensor" if mask_by_block else "elementwise"
        if mask_mode not in ("elementwise", "tensor", "rows"):
            raise ValueError(
                f"mask_mode must be 'elementwise', 'tensor' or 'rows', got {mask_mode!r}"
            )
        self.mask_mode: str = mask_mode
        self.mask_by_block: bool = mask_mode == "tensor"
        self.microbatch_size = int(microbatch_size)
        self.resample_every = int(resample_every)
        if self.resample_every < 1:
            raise ValueError("resample_every must be >= 1")

        flat, unravel = ravel_pytree(params)
        self.flat_params: jnp.ndarray = flat
        self._unravel = unravel
        self.n_params: int = int(flat.size)
        self._treedef = jax.tree_util.tree_structure(params)

        # Leaf boundaries (in ravel order) for structural / rows masking.
        leaves = jax.tree_util.tree_leaves(params)
        self._n_leaves = len(leaves)
        self._leaf_shapes = tuple(tuple(l.shape) for l in leaves)
        sizes = np.asarray([int(np.prod(l.shape)) for l in leaves], dtype=np.int64)
        self._leaf_sizes = sizes
        self._leaf_starts = np.concatenate([[0], np.cumsum(sizes)])[:-1]

        # Mask state; a resample is due on call 0 and every `resample_every`
        # calls thereafter.
        self._mask_calls = 0
        self._selected_leaf_ids: tuple[int, ...] | None = None
        # Compiled per-selection Jacobian fns keyed on the sorted tuple of
        # selected leaf ids (each selection has its own argument shapes).
        self._structural_cache: dict[
            tuple[int, ...], tuple[Callable[..., Any], tuple[int, ...]]
        ] = {}

        # Rows-mode selection state (per-leaf sorted leading-axis indices).
        self._row_indices: list[np.ndarray | None] = []

        if self.param_fraction >= 1.0:
            self.n_active = self.n_params
            self.actual_param_fraction: float | None = 1.0
            self._jac_fn = self._build_full_jac_fn()
        elif self.mask_mode == "rows":
            # Slice counts are deterministic per fraction, so the active total
            # (hence every shape) is fixed and known at construction.
            counts = leading_row_counts(self._leaf_shapes, self.param_fraction)
            self.n_active = int(
                sum(
                    n * (int(size) // shape[0] if shape else int(size))
                    for n, shape, size in zip(counts, self._leaf_shapes, sizes)
                )
            )
            self.actual_param_fraction = self.n_active / self.n_params
            self._jac_fn = self._build_rows_jac_fn()
        elif self.mask_by_block:
            # Estimate only; updated to the realised leaf total per resample.
            self.n_active = int(self.param_fraction * self.n_params)
            self.actual_param_fraction = None
            self._jac_fn = None
        else:
            self.n_active = int(self.param_fraction * self.n_params)
            self.actual_param_fraction = self.n_active / self.n_params
            self._jac_fn = self._build_masked_jac_fn()

        # Populated by ``loss_and_grad``; consumed by ``Sven.step``.
        self.losses: jnp.ndarray | None = None
        self.preds: Any = None
        self.residuals: jnp.ndarray | None = None
        self.jac: jnp.ndarray | None = None
        self.mask_indices: jnp.ndarray | None = None

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
    # Jacobian functions (JIT-compiled)
    # ------------------------------------------------------------------

    def _residuals_and_aux_full(self, flat_params, x, args):
        params = self._unravel(flat_params)
        preds = self.apply_fn(params, x)
        losses = self.loss_fn(preds, *args)
        if self.microbatch_size > 1:
            losses = losses.reshape(-1, self.microbatch_size).mean(axis=1)
        residuals = jnp.power(losses, self.kappa / 2.0)
        return residuals, (losses, preds)

    def _residuals_and_aux_masked(self, active, frozen_flat, indices, x, args):
        flat = frozen_flat.at[indices].set(active)
        params = self._unravel(flat)
        preds = self.apply_fn(params, x)
        losses = self.loss_fn(preds, *args)
        if self.microbatch_size > 1:
            losses = losses.reshape(-1, self.microbatch_size).mean(axis=1)
        residuals = jnp.power(losses, self.kappa / 2.0)
        return residuals, (losses, preds)

    def _build_full_jac_fn(self):
        jac = jax.jacrev(self._residuals_and_aux_full, argnums=0, has_aux=True)

        @jax.jit
        def _fn(flat_params, x, args):
            return jac(flat_params, x, args)

        return _fn

    def _build_masked_jac_fn(self):
        jac = jax.jacrev(self._residuals_and_aux_masked, argnums=0, has_aux=True)

        @jax.jit
        def _fn(active, frozen_flat, indices, x, args):
            return jac(active, frozen_flat, indices, x, args)

        return _fn

    def _residuals_and_aux_rows(self, active, frozen, rows, x, args):
        leaves: list[Any] = []
        for act, leaf, r in zip(active, frozen, rows):
            leaf = jax.lax.stop_gradient(leaf)
            leaves.append(act if r is None else leaf.at[r].set(act))
        params = jax.tree_util.tree_unflatten(self._treedef, leaves)
        preds = self.apply_fn(params, x)
        losses = self.loss_fn(preds, *args)
        if self.microbatch_size > 1:
            losses = losses.reshape(-1, self.microbatch_size).mean(axis=1)
        residuals = jnp.power(losses, self.kappa / 2.0)
        return residuals, (losses, preds)

    def _build_rows_jac_fn(self):
        """Compiled rows-mode Jacobian fn: one trace covers every selection.

        All argument shapes are fixed by the deterministic slice counts; the
        row indices are traced data, so resampling never retraces. The
        scatter's backward still materialises a transient ``(B, P_leaf)``
        cotangent per leaf before the gather (see module docstring).
        """
        jac = jax.jacrev(self._residuals_and_aux_rows, argnums=0, has_aux=True)

        @jax.jit
        def _fn(active, frozen, rows, x, args):
            jac_tree, aux = jac(active, frozen, rows, x, args)
            # Ascending leaf id == ascending ravel offset, rows sorted within
            # each leaf — matches the ``flat_params[mask_indices]`` gather.
            J = jnp.concatenate(
                [j.reshape(j.shape[0], -1) for j in jac_tree], axis=1
            )
            return J, aux

        return _fn

    def _structural_jac_fn(self, selected: tuple[int, ...]):
        """Compiled Jacobian fn for one leaf selection (cached on ids)."""
        cached = self._structural_cache.get(selected)
        if cached is not None:
            return cached

        residuals, frozen_ids = make_leafset_residual_fn(
            self.apply_fn,
            self.loss_fn,
            self.kappa,
            self.microbatch_size,
            self._treedef,
            self._n_leaves,
            selected,
        )
        jac = jax.jacrev(residuals, argnums=0, has_aux=True)

        @jax.jit
        def _fn(active, frozen, x, args):
            jac_tree, aux = jac(active, frozen, x, args)
            # Ascending leaf id == ascending ravel offset, so this matches how
            # ``mask_indices`` gathers ``flat_params``.
            J = jnp.concatenate(
                [j.reshape(j.shape[0], -1) for j in jac_tree], axis=1
            )
            return J, aux

        self._structural_cache[selected] = (_fn, frozen_ids)
        return _fn, frozen_ids

    # ------------------------------------------------------------------
    # Mask sampling (host-side — cheap integer work)
    # ------------------------------------------------------------------

    def _sample_mask_indices(self, key: jax.Array) -> jnp.ndarray:
        # Elementwise random subset. Numpy for speed and to keep index shape
        # stable (host ints); upload once per step.
        idx = sample_elementwise_indices(key, self.n_params, self.n_active)
        return jnp.asarray(idx, dtype=jnp.int32)

    def _select_block_leaves(self, key: jax.Array) -> tuple[int, ...]:
        """Greedy random whole-leaf selection (see :func:`select_block_leaves`)."""
        return select_block_leaves(key, self._leaf_sizes, self.param_fraction)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _resample_due(self) -> bool:
        return (
            self.mask_indices is None
            or self._mask_calls % self.resample_every == 0
        )

    def _structural_loss_and_grad(self, x, args, key):
        if self._resample_due():
            if key is None:
                raise ValueError("`key` is required when param_fraction < 1.0")
            selected = tuple(sorted(self._select_block_leaves(key)))
            self._selected_leaf_ids = selected
            self.n_active = int(self._leaf_sizes[list(selected)].sum())
            self.actual_param_fraction = self.n_active / self.n_params
            idx = leafset_mask_indices(selected, self._leaf_starts, self._leaf_sizes)
            self.mask_indices = jnp.asarray(idx, dtype=jnp.int32)
        self._mask_calls += 1

        fn, frozen_ids = self._structural_jac_fn(self._selected_leaf_ids)
        leaves = jax.tree_util.tree_leaves(self.params)
        active = tuple(leaves[i] for i in self._selected_leaf_ids)
        frozen = tuple(leaves[i] for i in frozen_ids)
        return fn(active, frozen, x, args)

    def _rows_loss_and_grad(self, x, args, key):
        if self._resample_due():
            if key is None:
                raise ValueError("`key` is required when param_fraction < 1.0")
            self._row_indices = sample_leading_rows(
                key, self._leaf_shapes, self.param_fraction
            )
            idx = rows_mask_indices(
                self._row_indices,
                self._leaf_starts,
                self._leaf_shapes,
                self._leaf_sizes,
            )
            self.mask_indices = jnp.asarray(idx, dtype=jnp.int32)
        self._mask_calls += 1

        leaves = jax.tree_util.tree_leaves(self.params)
        rows = tuple(
            None if r is None else jnp.asarray(r, dtype=jnp.int32)
            for r in self._row_indices
        )
        active = tuple(
            leaf if r is None else leaf[r] for leaf, r in zip(leaves, rows)
        )
        return self._jac_fn(active, tuple(leaves), rows, x, args)

    def loss_and_grad(
        self,
        batch: Sequence[jnp.ndarray],
        key: jax.Array | None = None,
    ) -> tuple[jnp.ndarray, Any]:
        """Compute per-sample losses and Jacobian.

        Args:
            batch: ``(x, *loss_args)``.
            key: Required when ``param_fraction < 1`` and a mask resample is
                due (call 0 and every ``resample_every`` calls); ignored while
                the previous mask is being reused.
        """
        x, *args = batch
        args = tuple(args)

        if self.param_fraction >= 1.0:
            jac, (losses, preds) = self._jac_fn(self.flat_params, x, args)
            self.mask_indices = None
        elif self.mask_mode == "rows":
            jac, (losses, preds) = self._rows_loss_and_grad(x, args, key)
        elif self.mask_by_block:
            jac, (losses, preds) = self._structural_loss_and_grad(x, args, key)
        else:
            if self._resample_due():
                if key is None:
                    raise ValueError("`key` is required when param_fraction < 1.0")
                self.mask_indices = self._sample_mask_indices(key)
            self._mask_calls += 1
            active = self.flat_params[self.mask_indices]
            frozen = jax.lax.stop_gradient(self.flat_params)
            jac, (losses, preds) = self._jac_fn(
                active, frozen, self.mask_indices, x, args
            )

        self.jac = jac
        self.losses = losses
        self.preds = preds
        self.residuals = jnp.power(losses, self.kappa / 2.0)
        return losses, preds

    def apply_update(self, delta: jnp.ndarray) -> None:
        """Apply ``delta`` (shape ``(n_active,)``) to the flat params."""
        if self.mask_indices is not None:
            self.flat_params = self.flat_params.at[self.mask_indices].add(delta)
        else:
            self.flat_params = self.flat_params + delta

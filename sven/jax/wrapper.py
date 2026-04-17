"""SvenWrapper for JAX / Flax models.

Provides a functional wrapper that computes per-sample Jacobians of a loss
function with respect to a flat parameter vector. Mirrors the PyTorch
``SvenWrapper`` API so training loops translate directly.

Efficient masked Jacobian
-------------------------

When ``param_fraction < 1.0`` the PyTorch version still computes the full
Jacobian w.r.t. **all** parameters under the hood (``jacrev`` differentiates
w.r.t. the flat vector, then indexes). In JAX we differentiate w.r.t. only the
**active** slice: the frozen portion of the parameter vector is placed behind
``lax.stop_gradient`` and the active slice is scattered into the correct
positions with ``.at[indices].set(active)``. Reverse-mode AD propagates
cotangents only to the active slice, so the Jacobian is genuinely
``(B, n_active)`` both in memory and in compute.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree


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
        mask_by_block: If ``True``, mask whole pytree leaves instead of
            individual entries. Mandatory when leaves live on different
            shardings; otherwise elementwise is more expressive.
        microbatch_size: Aggregate losses into groups of this size (mean) to
            shrink the Jacobian's row dimension.
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
    ) -> None:
        self.apply_fn = apply_fn
        self.loss_fn = loss_fn
        self.kappa = float(kappa)
        self.param_fraction = float(param_fraction)
        self.mask_by_block = bool(mask_by_block)
        self.microbatch_size = int(microbatch_size)

        flat, unravel = ravel_pytree(params)
        self.flat_params: jnp.ndarray = flat
        self._unravel = unravel
        self.n_params: int = int(flat.size)

        # Block boundaries (in ravel order) for block-level masking.
        leaves = jax.tree_util.tree_leaves(params)
        sizes = np.asarray([int(np.prod(l.shape)) for l in leaves], dtype=np.int64)
        self._leaf_sizes = sizes
        self._leaf_starts = np.concatenate([[0], np.cumsum(sizes)])[:-1]

        # Precompute n_active and a compiled Jacobian function.
        if self.param_fraction < 1.0:
            self.n_active = int(self.param_fraction * self.n_params)
            self._jac_fn = self._build_masked_jac_fn()
        else:
            self.n_active = self.n_params
            self._jac_fn = self._build_full_jac_fn()

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

    # ------------------------------------------------------------------
    # Mask sampling (host-side — cheap integer work)
    # ------------------------------------------------------------------

    def _sample_mask_indices(self, key: jax.Array) -> jnp.ndarray:
        if self.mask_by_block:
            return self._sample_block_indices(key)
        # Elementwise random subset. Use numpy for speed and to keep index shape
        # stable (host ints); upload once per step.
        seed = int(jax.random.randint(key, (), 0, np.iinfo(np.int32).max))
        rng = np.random.default_rng(seed)
        idx = rng.choice(self.n_params, size=self.n_active, replace=False)
        idx.sort()
        return jnp.asarray(idx, dtype=jnp.int32)

    def _sample_block_indices(self, key: jax.Array) -> jnp.ndarray:
        seed = int(jax.random.randint(key, (), 0, np.iinfo(np.int32).max))
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self._leaf_sizes))
        target = self.n_active
        running = 0
        pieces: list[np.ndarray] = []
        for i in order:
            s = int(self._leaf_sizes[i])
            start = int(self._leaf_starts[i])
            if running + s >= target:
                n_to_use = target - running
                if n_to_use > 0:
                    pieces.append(np.arange(start, start + n_to_use, dtype=np.int64))
                break
            pieces.append(np.arange(start, start + s, dtype=np.int64))
            running += s
        idx = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int64)
        # Pad with zeros if we somehow came up short (last-block clip); this
        # only happens when the leaf-sum doesn't exactly hit n_active.
        if idx.size < self.n_active:
            pad = np.zeros(self.n_active - idx.size, dtype=np.int64)
            idx = np.concatenate([idx, pad])
        return jnp.asarray(idx[: self.n_active], dtype=jnp.int32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def loss_and_grad(
        self,
        batch: Sequence[jnp.ndarray],
        key: jax.Array | None = None,
    ) -> tuple[jnp.ndarray, Any]:
        """Compute per-sample losses and Jacobian.

        Args:
            batch: ``(x, *loss_args)``.
            key: Required when ``param_fraction < 1`` (used to resample the
                active subset each step).
        """
        x, *args = batch
        args = tuple(args)

        if self.param_fraction >= 1.0:
            jac, (losses, preds) = self._jac_fn(self.flat_params, x, args)
            self.mask_indices = None
        else:
            if key is None:
                raise ValueError("`key` is required when param_fraction < 1.0")
            indices = self._sample_mask_indices(key)
            active = self.flat_params[indices]
            frozen = jax.lax.stop_gradient(self.flat_params)
            jac, (losses, preds) = self._jac_fn(active, frozen, indices, x, args)
            self.mask_indices = indices

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

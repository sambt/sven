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

from .wrapper import make_leafset_residual_fn


class GramSvenWrapper:
    """Functional wrapper accumulating the residual Gram matrix ``J J^T``.

    Drop-in alternative to :class:`SvenWrapper` for use with
    :class:`sven.jax.SvenGram`: ``loss_and_grad`` fills ``.gram`` (numpy
    float64, shape ``(M, M)``) and ``.residuals`` instead of ``.jac``, which
    stays ``None``.

    Args:
        apply_fn: Forward function ``(params, x) -> preds``.
        params: Initial parameter pytree.
        loss_fn: ``(preds, *args) -> per_sample_losses`` with shape ``(B,)``.
        kappa: Exponent such that updates are computed on ``loss ** (kappa/2)``
            (default 2.0 = gradients of the raw loss).
        microbatch_size: Aggregate losses into groups of this size (mean) to
            shrink the Gram's row dimension ``M``.
        group_numel: Max total leaf entries differentiated at once; bounds the
            transient per-group Jacobian at ``M * group_numel`` device floats.
            A single leaf larger than this gets its own group.
    """

    def __init__(
        self,
        apply_fn: Callable[..., Any],
        params: Any,
        loss_fn: Callable[..., jnp.ndarray],
        *,
        kappa: float = 2.0,
        microbatch_size: int = 1,
        group_numel: int = 2**22,
    ) -> None:
        self.apply_fn = apply_fn
        self.loss_fn = loss_fn
        self.kappa = float(kappa)
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
        sizes = [int(np.prod(l.shape)) for l in leaves]

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
        self._group_fns = [self._build_group_fn(g) for g in groups]
        self._delta_fn = self._build_delta_fn()

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
        """Jitted ``(active, frozen, x, args) -> (J_grp J_grp^T, aux)``."""
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

        return _fn, frozen_ids

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
    # Public API
    # ------------------------------------------------------------------

    def loss_and_grad(
        self, batch: Sequence[jnp.ndarray]
    ) -> tuple[jnp.ndarray, Any]:
        """Accumulate ``G = J J^T`` group by group; never build ``J``.

        Args:
            batch: ``(x, *loss_args)``.
        """
        x, *args = batch
        args = tuple(args)

        leaves = jax.tree_util.tree_leaves(self.params)
        gram: np.ndarray | None = None
        aux = None
        for group, (fn, frozen_ids) in zip(self._groups, self._group_fns):
            active = tuple(leaves[i] for i in group)
            frozen = tuple(leaves[i] for i in frozen_ids)
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

        Args:
            w: Row weights, shape ``(M,)``.

        Returns:
            Flat update direction, shape ``(n_params,)``.
        """
        if self._last_batch is None:
            raise RuntimeError("Call `loss_and_grad(batch)` before `delta_from_w`.")
        x, args = self._last_batch
        w = jnp.asarray(w, dtype=self.flat_params.dtype)
        return self._delta_fn(self.flat_params, x, args, w)

    def apply_update(self, delta: jnp.ndarray) -> None:
        """Apply ``delta`` (shape ``(n_params,)``) to the flat params."""
        self.flat_params = self.flat_params + delta

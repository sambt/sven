"""Sven optimizer in JAX.

Given a :class:`SvenWrapper` that has just computed per-sample residuals and
their Jacobian, :meth:`Sven.step` applies the minimum-norm update

    delta = -lr * J^+ r

via a truncated SVD of ``J``. The pseudo-inverse is never materialised — we
keep the factors ``(Vh^T, S_inv, U^T)`` and apply them sequentially.

Optional RMSProp scaling is supported in both ``pre`` (scale gradients before
the pseudo-inverse) and ``post`` (scale the final update) modes.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from .pinv import SVDMode, pinv
from .wrapper import SvenWrapper


@jax.jit
def _compute_delta(U_T, S_inv, VhT, residuals):
    return VhT @ (S_inv * (U_T @ residuals))


class Sven:
    """SVD-based optimizer.

    Args:
        wrapper: :class:`SvenWrapper` providing ``jac`` and ``residuals``.
        lr: Learning rate.
        k: Number of singular components to keep.
        rtol: Relative tolerance; singular values below ``rtol * sigma_max``
            are zeroed.
        svd_mode: ``'randomized'`` or ``'full'``.
        power_iterations: Power iterations for the randomized SVD.
        use_rmsprop: Enable RMSProp-style adaptive scaling.
        alpha_rmsprop: EMA decay for RMSProp.
        eps_rmsprop: RMSProp denominator epsilon.
        mu_rmsprop: Momentum for RMSProp (post mode only).
        rmsprop_post: If ``True``, scale the final update; else scale the
            per-sample-mean gradient before the pseudo-inverse.
        seed: Seed for the internal PRNG (used to draw random projections in
            randomized SVD).
    """

    def __init__(
        self,
        wrapper: SvenWrapper,
        lr: float,
        k: int,
        rtol: float = 1e-3,
        svd_mode: SVDMode = "randomized",
        power_iterations: int = 1,
        use_rmsprop: bool = False,
        alpha_rmsprop: float = 0.99,
        eps_rmsprop: float = 1e-8,
        mu_rmsprop: float = 0.0,
        rmsprop_post: bool = False,
        track_svd_info: bool = False,
        seed: int = 0,
    ) -> None:
        self.wrapper = wrapper
        self.lr = float(lr)
        self.k = int(k)
        self.rtol = float(rtol)
        self.svd_mode: SVDMode = svd_mode
        self.power_iterations = int(power_iterations)

        self.use_rmsprop = bool(use_rmsprop)
        self.rmsprop_post = bool(rmsprop_post)
        self.alpha_rmsprop = float(alpha_rmsprop)
        self.eps_rmsprop = float(eps_rmsprop)
        self.mu_rmsprop = float(mu_rmsprop)
        self.v: jnp.ndarray | None = None
        self.b: jnp.ndarray | None = None

        self.track_svd_info = bool(track_svd_info)
        self.svd_info: dict[str, list[Any]] = {"svs": [], "num_nonzero_svs": []}

        self._key = jax.random.PRNGKey(seed)

    # ------------------------------------------------------------------
    def _next_key(self) -> jax.Array:
        self._key, sub = jax.random.split(self._key)
        return sub

    # ------------------------------------------------------------------
    def step(self) -> None:
        """Compute and apply the pseudo-inverse parameter update."""
        jac = self.wrapper.jac
        residuals = self.wrapper.residuals
        if jac is None or residuals is None:
            raise RuntimeError(
                "Call `wrapper.loss_and_grad(batch)` before `optimizer.step()`."
            )

        if self.use_rmsprop and not self.rmsprop_post:
            mean_grad = jac.mean(axis=0)
            if self.v is None:
                self.v = jnp.zeros_like(mean_grad)
            self.v = self.alpha_rmsprop * self.v + (1 - self.alpha_rmsprop) * mean_grad ** 2
            jac = jac / (jnp.sqrt(self.v) + self.eps_rmsprop)

        VhT, S_inv, U_T = pinv(
            jac,
            k=self.k,
            key=self._next_key(),
            rtol=self.rtol,
            mode=self.svd_mode,
            power_iter=self.power_iterations,
        )

        delta = _compute_delta(U_T, S_inv, VhT, residuals)

        if self.use_rmsprop and self.rmsprop_post:
            if self.v is None:
                self.v = jnp.zeros_like(delta)
            self.v = self.alpha_rmsprop * self.v + (1 - self.alpha_rmsprop) * delta ** 2
            delta = delta / (jnp.sqrt(self.v) + self.eps_rmsprop)
            if self.mu_rmsprop > 0:
                if self.b is None:
                    self.b = jnp.zeros_like(delta)
                self.b = self.mu_rmsprop * self.b + delta
                delta = self.b

        self.wrapper.apply_update(-self.lr * delta)

        if self.track_svd_info:
            # Keep on-device; caller can .block_until_ready or convert later.
            self.svd_info["svs"].append(S_inv)
            self.svd_info["num_nonzero_svs"].append(jnp.sum(S_inv > 0))

        # Clear the Jacobian so a stale one can't be reused by a later step.
        self.wrapper.jac = None
        self.wrapper.residuals = None

"""Sven optimizer in JAX.

Given a :class:`SvenWrapper` that has just computed per-sample residuals and
their Jacobian, :meth:`Sven.step` applies the minimum-norm update

    delta = -lr * J^+ r

via a truncated SVD of ``J``. The pseudo-inverse is never materialised — we
keep the factors ``(Vh^T, S_inv, U^T)`` and apply them sequentially.

:class:`SvenGram` applies the same update through the Gram matrix
``G = J J^T`` accumulated by :class:`sven.jax.GramSvenWrapper`, so the
``(M, P)`` Jacobian itself is never needed.

Optional RMSProp scaling is supported in both ``pre`` (scale gradients before
the pseudo-inverse) and ``post`` (scale the final update) modes; the Gram
variant supports ``post`` only.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .gram import GramSvenWrapper
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


class SvenGram:
    """SVD-based optimizer driven by the residual Gram matrix.

    Consumes a :class:`GramSvenWrapper`: eigendecomposes ``G = J J^T`` in
    numpy float64 (``M x M`` is tiny), forms

        w = U_k diag(1/sigma_k^2) U_k^T r,

    and applies ``delta = J^T w`` via one VJP — the exact :class:`Sven`
    update without ever materialising ``J``. Truncation matches
    :func:`sven.jax.pinv`: truncate to rank ``k``, then zero ``1/sigma``
    where ``sigma <= max(rtol * sigma_max, tol)`` (masking at fixed rank).

    RMSProp is supported in ``post`` mode only; the ``pre`` mode rescales
    Jacobian columns, which cannot be expressed through ``G`` alone.

    Args:
        wrapper: :class:`GramSvenWrapper` providing ``gram`` and ``residuals``.
        lr: Learning rate.
        k: Number of singular components to keep.
        rtol: Relative tolerance; singular values below ``rtol * sigma_max``
            are zeroed.
        tol: Absolute zero cut (matches :func:`sven.jax.pinv`).
        use_rmsprop: Enable RMSProp-style adaptive scaling (post mode only).
        alpha_rmsprop: EMA decay for RMSProp.
        eps_rmsprop: RMSProp denominator epsilon.
        mu_rmsprop: Momentum for RMSProp.
        rmsprop_post: Must be ``True`` when ``use_rmsprop`` is set.
        track_svd_info: Record the kept singular values each step.
    """

    def __init__(
        self,
        wrapper: GramSvenWrapper,
        lr: float,
        k: int,
        rtol: float = 1e-3,
        tol: float = 1e-10,
        use_rmsprop: bool = False,
        alpha_rmsprop: float = 0.99,
        eps_rmsprop: float = 1e-8,
        mu_rmsprop: float = 0.0,
        rmsprop_post: bool = False,
        track_svd_info: bool = False,
    ) -> None:
        if use_rmsprop and not rmsprop_post:
            raise NotImplementedError(
                "RMSProp 'pre' mode rescales Jacobian columns and needs the "
                "full Jacobian; use rmsprop_post=True or the base Sven."
            )
        self.wrapper = wrapper
        self.lr = float(lr)
        self.k = int(k)
        self.rtol = float(rtol)
        self.tol = float(tol)

        self.use_rmsprop = bool(use_rmsprop)
        self.rmsprop_post = bool(rmsprop_post)
        self.alpha_rmsprop = float(alpha_rmsprop)
        self.eps_rmsprop = float(eps_rmsprop)
        self.mu_rmsprop = float(mu_rmsprop)
        self.v: jnp.ndarray | None = None
        self.b: jnp.ndarray | None = None

        self.track_svd_info = bool(track_svd_info)
        self.svd_info: dict[str, list[Any]] = {"svs": [], "num_nonzero_svs": []}

    # ------------------------------------------------------------------
    def step(self) -> None:
        """Compute and apply the pseudo-inverse update via the Gram matrix."""
        gram = self.wrapper.gram
        residuals = self.wrapper.residuals
        if gram is None or residuals is None:
            raise RuntimeError(
                "Call `wrapper.loss_and_grad(batch)` before `optimizer.step()`."
            )

        r = np.asarray(residuals, dtype=np.float64)
        evals, evecs = np.linalg.eigh(gram)          # ascending
        sigma = np.sqrt(np.clip(evals[::-1], 0.0, None))
        U = evecs[:, ::-1]
        k = min(self.k, sigma.shape[0])
        sigma = sigma[:k]
        U = U[:, :k]

        # pinv() semantics at fixed rank: mask, don't shrink.
        cutoff = max(self.rtol * sigma[0], self.tol)
        keep = sigma > cutoff
        s_inv_sq = np.where(keep, 1.0 / np.where(keep, sigma**2, 1.0), 0.0)
        w = U @ (s_inv_sq * (U.T @ r))

        delta = self.wrapper.delta_from_w(w)

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
            self.svd_info["svs"].append(sigma)
            self.svd_info["num_nonzero_svs"].append(int(keep.sum()))

        # Clear so a stale Gram can't be reused by a later step.
        self.wrapper.gram = None
        self.wrapper.residuals = None

"""Truncated SVD pseudo-inverse for JAX.

Returns factored components ``(Vh^T, S_inv, U^T)`` rather than materialising
the full ``(P, B)`` pseudo-inverse. Singular values below ``rtol * sigma_max``
are zeroed out by masking ``S_inv`` (the shape stays fixed at ``k``) so the
routine is JIT-friendly: a single trace handles any data.
"""

from __future__ import annotations

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp

SVDMode = Literal["randomized", "full"]


@partial(jax.jit, static_argnames=("k", "power_iter"))
def _randomized_svd(
    A: jnp.ndarray,
    k: int,
    key: jax.Array,
    power_iter: int = 1,
    oversample: int = 5,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Randomized SVD with the Gram-matrix trick from the PyTorch ``randomized_v2``.

    Avoids calling ``jnp.linalg.svd`` on ``B = Q^T A`` of shape ``(r, P)`` which
    is unstable / slow when ``P`` is in the millions. Instead we diagonalise
    the tiny Gram matrix ``G = B B^T`` of shape ``(r, r)``.
    """
    m, n = A.shape
    r = min(k + oversample, min(m, n))

    Omega = jax.random.normal(key, (n, r), dtype=A.dtype)
    Y = A @ Omega

    def _power_step(Y, _):
        return A @ (A.T @ Y), None

    if power_iter > 0:
        Y, _ = jax.lax.scan(_power_step, Y, None, length=power_iter)

    Q, _ = jnp.linalg.qr(Y)                  # (m, r)
    B = Q.T @ A                              # (r, n)

    G = B @ B.T                              # (r, r)
    evals, evecs = jnp.linalg.eigh(G)        # ascending
    evals = evals[::-1]
    evecs = evecs[:, ::-1]

    S = jnp.sqrt(jnp.maximum(evals, 0.0))
    U = Q @ evecs                            # (m, r)

    eps = jnp.finfo(A.dtype).eps
    S_safe = jnp.maximum(S, eps)
    Vh = (U.T / S_safe[:, None]) @ A         # (r, n)

    return U[:, :k], S[:k], Vh[:k, :]


@partial(jax.jit, static_argnames=("k",))
def _full_truncated_svd(A: jnp.ndarray, k: int):
    U, S, Vh = jnp.linalg.svd(A, full_matrices=False)
    return U[:, :k], S[:k], Vh[:k, :]


@partial(jax.jit, static_argnames=("k", "mode", "power_iter"))
def pinv(
    M: jnp.ndarray,
    k: int,
    key: jax.Array,
    rtol: float = 1e-3,
    tol: float = 1e-10,
    mode: SVDMode = "randomized",
    power_iter: int = 1,
):
    """Pseudo-inverse factors of ``M`` via rank-``k`` truncated SVD.

    Returns ``(Vh^T, S_inv, U^T)``. Singular values below ``max(rtol * S[0], tol)``
    give ``S_inv = 0``; the shape stays ``(k,)`` so the trace is static.

    Args:
        M: Jacobian, shape ``(B, P)``.
        k: target rank.
        key: PRNG key (used only for ``mode='randomized'``).
        rtol: zero singular values below ``rtol * sigma_max``.
        tol: absolute zero cut.
        mode: ``'randomized'`` (recommended for large ``P``) or ``'full'``.
        power_iter: number of power iterations for randomized mode.
    """
    if mode == "randomized":
        U, S, Vh = _randomized_svd(M, k, key, power_iter=power_iter)
    elif mode == "full":
        U, S, Vh = _full_truncated_svd(M, k)
    else:
        raise ValueError(f"invalid SVD mode {mode!r}")

    cutoff = jnp.maximum(rtol * S[0], tol)
    keep = S > cutoff
    S_inv = jnp.where(keep, 1.0 / jnp.where(keep, S, 1.0), 0.0)

    return Vh.T, S_inv, U.T

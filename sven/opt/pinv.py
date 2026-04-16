"""Truncated SVD pseudo-inverse implementations.

All functions return the factored pseudo-inverse components ``(Vh^T, S_inv, U^T)``
so the full pseudo-inverse is never materialised in memory.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

# Type alias for the three-matrix SVD factorisation
SVDResult = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

VALID_MODES = ("randomized", "randomized_v2", "scipy", "torch", "lobpcg")
SVDMode = Literal["randomized", "randomized_v2", "scipy", "torch", "lobpcg"]


@torch.no_grad()
def pinv(
    M: torch.Tensor,
    k: int = 2,
    tol: float = 1e-10,
    rtol: float = 1e-3,
    mode: SVDMode = "randomized",
    power_iter: int = 1,
) -> SVDResult:
    """Compute the pseudo-inverse via truncated SVD.

    Returns ``(Vh^T, S_inv, U^T)`` — the factored components of M+ — to
    avoid storing the full ``(P x B)`` pseudo-inverse matrix.

    Args:
        M: The ``(B, P)`` Jacobian matrix.
        k: Number of singular values to keep.
        tol: Absolute tolerance — singular values below this are zeroed.
        rtol: Relative tolerance — singular values below ``rtol * sigma_max``
            are discarded.
        mode: SVD algorithm to use.
        power_iter: Number of power iterations (randomized mode only).
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid SVD mode '{mode}'. Choose from {VALID_MODES}.")

    M = M.detach()

    if mode == "torch":
        U, S, Vh = _truncated_svd_torch(M, k=k)
    elif mode == "randomized":
        U, S, Vh = _randomized_svd(M, k=k, q=power_iter)
    elif mode == "randomized_v2":
        U, S, Vh = _randomized_svd_v2(M, k=k, q=power_iter)
    elif mode == "scipy":
        U, S, Vh = _truncated_svd_scipy(M, k=k)
    elif mode == "lobpcg":
        U, S, Vh = _truncated_svd_lobpcg(M, k=k)

    # Discard singular values below rtol * sigma_max
    kmax: int = 1 + int((S > rtol * S[0]).nonzero(as_tuple=True)[0].max().item())
    U = U[:, :kmax]
    S = S[:kmax]
    Vh = Vh[:kmax, :]

    S_inv = torch.where(S > tol, 1.0 / S, torch.zeros_like(S))

    return Vh.T.detach(), S_inv.detach(), U.T.detach()


# ------------------------------------------------------------------
# SVD backends
# ------------------------------------------------------------------


@torch.no_grad()
def _truncated_svd_torch(A: torch.Tensor, k: int = 2) -> SVDResult:
    """Full SVD via ``torch.linalg.svd``, then truncate to rank *k*."""
    A = A.detach()
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    return U[:, :k].detach(), S[:k].detach(), Vh[:k, :].detach()


@torch.no_grad()
def _randomized_svd(
    A: torch.Tensor,
    k: int,
    p: int = 5,
    q: int = 1,
) -> SVDResult:
    """Randomized SVD via random projection + power iteration.

    Args:
        A: Input matrix of shape ``(m, n)``.
        k: Target rank.
        p: Oversampling parameter.
        q: Number of power iterations for improved accuracy.
    """
    m, n = A.shape
    r = min(k + p, min(m, n))

    # Random projection
    Omega = torch.randn(n, r, device=A.device, dtype=A.dtype)
    Y = A @ Omega

    # Power iterations for better accuracy
    for _ in range(q):
        Y = A @ (A.T @ Y)

    Q, _ = torch.linalg.qr(Y)

    # Project and compute SVD of smaller matrix
    B = Q.T @ A
    Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ Ub

    return U[:, :k].contiguous(), S[:k].contiguous(), Vh[:k, :].contiguous()


@torch.no_grad()
def _randomized_svd_v2(
    A: torch.Tensor,
    k: int,
    p: int = 5,
    q: int = 1,
) -> SVDResult:
    """Randomized SVD via random projection + power iteration.

    Identical to ``_randomized_svd`` but avoids calling ``torch.linalg.svd``
    on the projected matrix ``B = Q.T @ A`` of shape ``(r, n)``.  When ``n``
    is very large (e.g. ~11M parameters in ResNet18), cuSolver's ``gesvdj``
    algorithm returns ``CUSOLVER_STATUS_INVALID_VALUE`` and the run fails.

    Fix: form the small ``(r, r)`` Gram matrix ``G = B @ B.T`` and use
    ``torch.linalg.eigh`` to recover singular values and left singular vectors.
    Right singular vectors are then recovered via ``Vh = diag(1/S) @ U.T @ A``,
    which is a standard matmul that cuBLAS handles for any ``n``.

    Args:
        A: Input matrix of shape ``(m, n)``.
        k: Target rank.
        p: Oversampling parameter.
        q: Number of power iterations for improved accuracy.
    """
    m, n = A.shape
    r = min(k + p, min(m, n))

    # Random projection
    Omega = torch.randn(n, r, device=A.device, dtype=A.dtype)
    Y = A @ Omega

    # Power iterations for better accuracy
    for _ in range(q):
        Y = A @ (A.T @ Y)

    Q, _ = torch.linalg.qr(Y)  # (m, r)
    B = Q.T @ A  # (r, n)

    # Eigendecomposition of the small (r, r) Gram matrix instead of svd(B)
    G = B @ B.T  # (r, r)
    evals, evecs = torch.linalg.eigh(G)  # ascending order; evecs: (r, r)

    # Sort descending
    idx = torch.arange(r - 1, -1, -1, device=A.device)
    evals = evals[idx]
    evecs = evecs[:, idx]

    S = evals.clamp_min(0).sqrt()  # (r,)
    U = Q @ evecs  # (m, r)

    # Recover Vh = diag(1/S) @ U.T @ A
    eps = torch.finfo(A.dtype).eps
    S_safe = S.clamp_min(eps)
    Vh = (U.T / S_safe.unsqueeze(1)) @ A  # (r, n)

    return U[:, :k].contiguous(), S[:k].contiguous(), Vh[:k, :].contiguous()


@torch.no_grad()
def _truncated_svd_lobpcg(A: torch.Tensor, k: int = 2) -> SVDResult:
    """Truncated SVD via LOBPCG on the Gram matrix."""
    A = A.detach()
    m, n = A.shape
    eps = torch.finfo(A.dtype).eps

    if n <= m:
        C = A.T @ A
        evals, evecs = torch.lobpcg(C, k=k, largest=True)
        s = evals.clamp_min(0).sqrt()
        V = evecs
        U = (A @ V) / s.clamp_min(eps)
        Vh = V.T
    else:
        C = A @ A.T
        evals, evecs = torch.lobpcg(C, k=k, largest=True)
        s = evals.clamp_min(0).sqrt()
        U = evecs
        Vh = ((U.T @ A) / s.clamp_min(eps).reshape(-1, 1)).conj()

    return U.detach(), s.detach(), Vh.detach()


def _truncated_svd_scipy(A: torch.Tensor, k: int) -> SVDResult:
    """Truncated SVD via ``scipy.sparse.linalg.svds`` (CPU fallback)."""
    from scipy.sparse.linalg import svds

    M = A.cpu().numpy()
    k = min(k, min(M.shape) - 1)
    U, S, Vh = svds(M, k=int(k))

    # svds does not guarantee descending order
    idx = np.argsort(S)[::-1]
    S = S[idx]
    U = U[:, idx]
    Vh = Vh[idx, :]

    return (
        torch.from_numpy(U).to(A.device, A.dtype),
        torch.from_numpy(S.copy()).to(A.device, A.dtype),
        torch.from_numpy(Vh).to(A.device, A.dtype),
    )

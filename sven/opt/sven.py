"""Sven: SVD-based optimizer using the Moore-Penrose pseudo-inverse."""

from __future__ import annotations

from typing import Any

import torch

from .pinv import SVDMode, pinv
from sven.nn.sven_wrapper import SvenWrapper
from sven.nn.gram_wrapper import GramSvenWrapper


class Sven:
    """SVD-based optimizer that computes parameter updates via the
    Moore-Penrose pseudo-inverse of the per-sample Jacobian.

    Args:
        model: A :class:`SvenWrapper` instance holding the model and its
            computed Jacobian / losses.
        lr: Learning rate
        k: Number of singular values to keep in the truncated SVD.
        rtol: Relative tolerance for singular-value truncation.
        track_svd_info: If ``True``, record singular values and rank info each
            step (useful for diagnostics, costs extra memory).
        svd_mode: SVD algorithm — ``"torch"``, ``"randomized"``, ``"scipy"``,
            or ``"lobpcg"``.
        power_iterations: Number of power iterations (randomized SVD only).
        use_rmsprop: Apply RMSProp-style adaptive scaling.
        alpha_rmsprop: Decay rate for the RMSProp running average.
        eps_rmsprop: Numerical stability constant for RMSProp.
        mu_rmsprop: Momentum coefficient for RMSProp. ``0`` disables momentum.
        rmsprop_post: If ``True``, apply RMSProp to the *update* vector; if
            ``False``, apply it to the *gradients* before computing the
            pseudo-inverse.
        variable_k: If ``True``, greedily add singular-value components one at
            a time, stopping when the loss increases.
    """

    def __init__(
        self,
        model: SvenWrapper,
        lr: float,
        k: int,
        rtol: float,
        track_svd_info: bool = False,
        svd_mode: SVDMode = "torch",
        power_iterations: int = 1,
        use_rmsprop: bool = False,
        alpha_rmsprop: float = 0.99,
        eps_rmsprop: float = 1e-8,
        mu_rmsprop: float = 0,
        rmsprop_post: bool = False,
        variable_k: bool = False
    ) -> None:
        self.model = model
        self.lr = lr
        self.k = k
        self.rtol = rtol
        self.power_iterations = power_iterations
        self.track_svd_info = track_svd_info
        self.svd_mode: SVDMode = svd_mode
        self.variable_k = variable_k
        self.use_rmsprop = use_rmsprop
        self.rmsprop_post = rmsprop_post

        self.svd_info: dict[str, list[Any]] = {
            "svs": [],
            "num_nonzero_svs": [],
            "k_used": [],
            "variable_k_substep_losses": [],
        }

        # RMSProp state (lazily initialised on first step)
        self.v: torch.Tensor | None = None
        self.b: torch.Tensor | None = None
        if use_rmsprop:
            self.alpha_rmsprop = alpha_rmsprop
            self.eps_rmsprop = eps_rmsprop
            self.mu_rmsprop = mu_rmsprop

    # ------------------------------------------------------------------
    # Core update helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_delta(
        U_T: torch.Tensor,
        S_inv: torch.Tensor,
        VhT: torch.Tensor,
        residuals: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the full-rank parameter update ``Vh^T diag(S_inv) U^T resisduals``."""
        delta = U_T @ residuals        # (k,)
        delta = S_inv * delta        # element-wise
        delta = VhT @ delta          # (P,)
        return delta

    @staticmethod
    def _compute_delta_k(
        k: int,
        U_T: torch.Tensor,
        S_inv: torch.Tensor,
        VhT: torch.Tensor,
        residuals: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the rank-1 update from the *k*-th singular component."""
        delta = U_T[k : k + 1, :] @ residuals
        delta = S_inv[k] * delta
        delta = VhT[:, k : k + 1] @ delta
        return delta.squeeze()

    def _get_lr(self) -> float:
        """Return the learning rate."""
        return self.lr

    def _rmsprop_post(self, update: torch.Tensor) -> torch.Tensor:
        """RMSProp-style rescaling of the update vector (post-pseudo-inverse)."""
        if self.v is None:
            self.v = torch.zeros_like(update)
        self.v.mul_(self.alpha_rmsprop).addcmul_(update, update, value=1 - self.alpha_rmsprop)
        update = update / (self.v.sqrt() + self.eps_rmsprop)

        if self.mu_rmsprop > 0:
            if self.b is None:
                self.b = torch.zeros_like(update)
            self.b.mul_(self.mu_rmsprop).add_(update)
            update = self.b
        return update

    def _apply_update(self, update: torch.Tensor) -> None:
        """Scale by learning rate and apply to model parameters."""
        scaled = -self._get_lr() * update
        if self.model.param_mask is not None:
            self.model.params[self.model.param_mask] += scaled
        else:
            self.model.params += scaled

    # ------------------------------------------------------------------
    # Parameter update strategies
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _update_params(
        self,
        U_T: torch.Tensor,
        S_inv: torch.Tensor,
        VhT: torch.Tensor,
        residuals: torch.Tensor,
    ) -> None:
        """Standard pseudo-inverse parameter update, optionally with RMSProp."""
        update = self._compute_delta(U_T, S_inv, VhT, residuals)

        if self.use_rmsprop and self.rmsprop_post:
            update = self._rmsprop_post(update)

        self._apply_update(update)

    @torch.no_grad()
    def _update_params_variable_k(
        self,
        batch: tuple[torch.Tensor, ...],
        U_T: torch.Tensor,
        S_inv: torch.Tensor,
        VhT: torch.Tensor,
        residuals: torch.Tensor,
        losses: torch.Tensor
    ) -> tuple[int, list[torch.Tensor]]:
        """Greedy rank-1 updates, stopping when the loss increases."""
        original_loss = losses.mean()
        kmax = len(S_inv)
        x, *args = batch

        substep_losses: list[torch.Tensor] = [original_loss]
        k_used = 0
        while k_used < kmax:
            update = self._compute_delta_k(k_used, U_T, S_inv, VhT, residuals)
            self._apply_update(update)

            new_loss = self.model.evaluate_and_loss(x, *args).mean()
            if new_loss > original_loss:
                self._apply_update(-update)
                break

            substep_losses.append(new_loss)
            k_used += 1

        return k_used, substep_losses

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, batch: tuple[torch.Tensor, ...] | None = None) -> None:
        """Compute and apply the pseudo-inverse parameter update.

        Args:
            batch: Required when ``variable_k=True`` so that the loss can be
                re-evaluated after each rank-1 update.  Ignored otherwise.
        """
        jacobian = self.model.grads
        residuals = self.model.residuals
        losses = self.model.losses

        # Optionally apply RMSProp to gradients *before* the pseudo-inverse
        if self.use_rmsprop and not self.rmsprop_post:
            mean_grad = jacobian.mean(dim=0)
            if self.v is None:
                self.v = torch.zeros_like(mean_grad)
            self.v.mul_(self.alpha_rmsprop).addcmul_(mean_grad, mean_grad, value=1 - self.alpha_rmsprop)
            jacobian = jacobian / (self.v.sqrt() + self.eps_rmsprop)

        VhT, S_inv, U_T = pinv(
            jacobian,
            k=self.k,
            rtol=self.rtol,
            mode=self.svd_mode,
            power_iter=self.power_iterations,
        )

        del jacobian
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Update parameters
        if self.variable_k:
            if batch is None:
                raise ValueError("batch must be provided when variable_k=True")
            k_used, substep_losses = self._update_params_variable_k(
                batch, U_T, S_inv, VhT, residuals, losses
            )
        else:
            self._update_params(U_T, S_inv, VhT, residuals)

        # Record diagnostics
        if self.track_svd_info:
            self.svd_info["svs"].append(1.0 / S_inv[S_inv > 0].cpu().numpy())
            self.svd_info["num_nonzero_svs"].append(torch.count_nonzero(S_inv).item())
            if self.variable_k:
                self.svd_info["k_used"].append(k_used)
                self.svd_info["variable_k_substep_losses"].append(substep_losses)

        del VhT, S_inv, U_T
        del self.model.residuals, self.model.grads, self.model.losses
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SvenGram(Sven):
    """Gram-matrix (kernel-trick) variant of :class:`Sven`.

    Consumes ``model.gram = J J^T`` (M, M) and ``model.residuals`` from a
    :class:`sven.nn.GramSvenWrapper` instead of the (M, P) Jacobian.  With
    ``G = U S^2 U^T`` the pseudo-inverse update is recovered as
    ``delta = J^T w`` with ``w = U_k S_k^{-2} U_k^T r`` via one backward pass
    (:meth:`GramSvenWrapper.delta_from_w`), so the Jacobian is never
    materialised.  Truncation replicates :func:`sven.opt.pinv` exactly.

    ``use_rmsprop`` with ``rmsprop_post=False`` and ``variable_k`` both need
    the Jacobian itself and raise ``NotImplementedError``.

    Args:
        model: A :class:`GramSvenWrapper` instance.
        lr: Learning rate.
        k: Number of singular values to keep in the truncated eig of ``G``.
        rtol: Relative tolerance for singular-value truncation.
        track_svd_info: If ``True``, record singular values and rank info.
        use_rmsprop: Apply RMSProp-style scaling (``rmsprop_post=True`` only).
        alpha_rmsprop: Decay rate for the RMSProp running average.
        eps_rmsprop: Numerical stability constant for RMSProp.
        mu_rmsprop: Momentum coefficient for RMSProp. ``0`` disables momentum.
        rmsprop_post: Must be ``True`` when ``use_rmsprop`` is set.
        variable_k: Unsupported — raises ``NotImplementedError``.
    """

    _SIGMA_TOL: float = 1e-10  # absolute sigma cutoff, matches pinv() default

    def __init__(
        self,
        model: GramSvenWrapper,
        lr: float,
        k: int,
        rtol: float,
        track_svd_info: bool = False,
        use_rmsprop: bool = False,
        alpha_rmsprop: float = 0.99,
        eps_rmsprop: float = 1e-8,
        mu_rmsprop: float = 0,
        rmsprop_post: bool = False,
        variable_k: bool = False,
    ) -> None:
        if variable_k:
            raise NotImplementedError(
                "variable_k needs per-component Jacobian updates; use Sven with SvenWrapper"
            )
        if use_rmsprop and not rmsprop_post:
            raise NotImplementedError(
                "pre-pseudo-inverse RMSProp rescales the Jacobian itself, which "
                "SvenGram never materialises; use rmsprop_post=True"
            )
        super().__init__(
            model,
            lr,
            k,
            rtol,
            track_svd_info=track_svd_info,
            use_rmsprop=use_rmsprop,
            alpha_rmsprop=alpha_rmsprop,
            eps_rmsprop=eps_rmsprop,
            mu_rmsprop=mu_rmsprop,
            rmsprop_post=rmsprop_post,
        )

    @torch.no_grad()
    def step(self, batch: tuple[torch.Tensor, ...] | None = None) -> None:
        """Compute and apply the Gram-based pseudo-inverse update.

        Args:
            batch: Ignored — kept for signature compatibility with ``Sven``.
        """
        gram = getattr(self.model, "gram", None)
        if gram is None:
            raise TypeError(
                "SvenGram needs model.gram: wrap the model with GramSvenWrapper "
                "and call loss_and_grad() before each step"
            )
        residuals = self.model.residuals

        # Eig of G = U S^2 U^T in fp64; eigh returns ascending order
        evals, evecs = torch.linalg.eigh(gram.detach().to(torch.float64))
        sigma = evals.flip(0).clamp_min(0.0).sqrt()
        U = evecs.flip(1)
        sigma = sigma[: self.k]
        U = U[:, : self.k]

        # pinv() truncation semantics: rtol relative to sigma_max, then abs tol
        kmax: int = 1 + int((sigma > self.rtol * sigma[0]).nonzero(as_tuple=True)[0].max().item())
        sigma = sigma[:kmax]
        U = U[:, :kmax]
        s_inv_sq = torch.where(
            sigma > self._SIGMA_TOL, 1.0 / sigma.pow(2), torch.zeros_like(sigma)
        )

        w = U @ (s_inv_sq * (U.T @ residuals.detach().to(torch.float64)))
        update = self.model.delta_from_w(w)  # J^T w, flat (P,)

        if self.use_rmsprop and self.rmsprop_post:
            update = self._rmsprop_post(update)
        self._apply_update(update)

        # Record diagnostics
        if self.track_svd_info:
            self.svd_info["svs"].append(sigma[s_inv_sq > 0].cpu().numpy())
            self.svd_info["num_nonzero_svs"].append(int(torch.count_nonzero(s_inv_sq).item()))

        del evals, evecs, sigma, U, s_inv_sq, w
        del self.model.gram, self.model.residuals, self.model.losses
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

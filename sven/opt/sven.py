"""Sven: SVD-based optimizer using the Moore-Penrose pseudo-inverse."""

from __future__ import annotations

from typing import Any

import numpy as np
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
        variable_k: If ``True``, greedily add singular-value components one at
            a time, stopping when the loss increases.
        empty_cache: If ``True``, call ``torch.cuda.empty_cache()`` after the
            solve and at the end of every step.  Each call synchronises the
            device AND drops the caching allocator's blocks, which measured
            4.5x slower per step on CIFAR ResNet18, hence the ``False``
            default; pass ``True`` to reproduce pre-2026-09 runs.

    Diagnostics (``track_svd_info=True``).  Set the attribute
    :attr:`log_this_step` to ``False`` before :meth:`step` to skip the
    per-step spectrum record and the device transfers it needs;
    ``num_nonzero_svs`` is still recorded on every step.  ``svd_info`` then
    holds, per **logged** step: ``step`` (this optimizer's own step index),
    ``svs`` (all M singular values, before the ``k`` / ``rtol`` cut), ``utr``
    (``U^T r``, all M, with ``U`` the Gram eigenvectors in descending-sigma
    order and ``r`` the vector actually solved against), ``update_norm``,
    ``resid_norm``, ``sv_min_kept`` (the smallest singular value actually
    inverted) and ``sv_noise_floor``.  Two conventions worth stating, because
    neither is recoverable from the numbers alone:

    * ``update_norm`` is the norm of the parameter change **as applied**,
      i.e. ``||lr * update|| = ||theta_new - theta_old||``, not the norm of
      the raw solve.  Divide by the run's ``lr`` for the latter.
    * the per-direction SIGN of ``utr`` is arbitrary: ``eigh`` fixes no sign
      convention and within a degenerate sigma cluster the whole sub-basis is
      arbitrary.  Only ``|utr|`` (or ``utr**2``) is comparable across steps
      and runs; averaging signed ``utr`` gives ~0 for no physical reason.

    Both paths take sigma from a float64 ``eigh`` of ``J J^T``, which squares
    the condition number: with float32 parameters (the campaign default) the
    error on sigma_i is ~ ``eps * sigma_max^2 / (2 sigma_i)``, so values below
    ``sv_noise_floor = sqrt(eps) * sigma_max`` (~3e-4 sigma_max in float32)
    are noise, NOT the ~1e-7 floor an ``svdvals(J)`` would give.  The floor is
    recorded per logged step so analysis can truncate instead of plotting
    noise.

    ``svd_info`` holds only Python numbers and numpy arrays (plus the ragged
    ``variable_k_substep_losses``), on GPU as on CPU, so it can be handed
    straight to ``np.asarray``.  :meth:`finalize_svd_info` is a convenience
    that returns the same record as numpy arrays.
    """

    #: ``J J^T`` is accumulated in float64 over blocks of at most this many
    #: Jacobian entries: at ResNet scale ``J.double()`` alone is several GB.
    _GRAM_BLOCK_ELEMS: int = 1 << 23

    def __init__(
        self,
        model: SvenWrapper,
        lr: float,
        k: int,
        rtol: float,
        track_svd_info: bool = False,
        svd_mode: SVDMode = "torch",
        power_iterations: int = 1,
        variable_k: bool = False,
        empty_cache: bool = False,
    ) -> None:
        self.model = model
        self.lr = lr
        self.k = k
        self.rtol = rtol
        self.power_iterations = power_iterations
        self.track_svd_info = track_svd_info
        self.svd_mode: SVDMode = svd_mode
        self.variable_k = variable_k
        self.empty_cache = empty_cache

        # Per-step logging switch, owned by the training loop; ``True`` keeps
        # the legacy behaviour of logging every step.
        self.log_this_step: bool = True
        self.step_count: int = 0

        self.svd_info: dict[str, list[Any]] = {
            "step": [],
            "svs": [],
            "utr": [],
            "update_norm": [],
            "resid_norm": [],
            "sv_min_kept": [],
            "sv_noise_floor": [],
            "num_nonzero_svs": [],
            "k_used": [],
            "variable_k_substep_losses": [],
        }

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

    def _maybe_empty_cache(self) -> None:
        """Release the caching allocator, unless ``empty_cache=False``."""
        if self.empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _apply_update(self, update: torch.Tensor) -> None:
        """Scale by learning rate and apply to model parameters."""
        scaled = -self._get_lr() * update
        if self.model.param_mask is not None:
            self.model.params[self.model.param_mask] += scaled
        else:
            self.model.params += scaled

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @classmethod
    def _gram_fp64(cls, jacobian: torch.Tensor) -> torch.Tensor:
        """``J J^T`` in float64, accumulated over blocks of ``J^T``'s rows.

        Casting the whole ``(M, P)`` Jacobian to float64 is not affordable
        (~6 GB at M=64, P=11.2M), so one block of parameter columns is cast at
        a time and only the ``(M, M)`` result is kept.
        """
        m, p = jacobian.shape
        block = max(1, cls._GRAM_BLOCK_ELEMS // max(m, 1))
        gram = torch.zeros((m, m), dtype=torch.float64, device=jacobian.device)
        for start in range(0, p, block):
            chunk = jacobian[:, start : start + block].to(torch.float64)
            gram += chunk @ chunk.T
        return gram

    @staticmethod
    def _spectrum_from_gram(
        gram: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(sigma, sigma^2, U)`` of ``G = U diag(sigma^2) U^T``, descending.

        ``eigh`` returns ascending order, hence the flips.  The squares are
        returned as the clamped eigenvalues themselves, not as ``sigma**2``.
        """
        evals, evecs = torch.linalg.eigh(gram)
        sigma_sq = evals.flip(0).clamp_min(0.0)
        return sigma_sq.sqrt(), sigma_sq, evecs.flip(1)

    @staticmethod
    def _sv_min_kept(sigma: torch.Tensor, filt: torch.Tensor) -> torch.Tensor:
        """Smallest singular value the filter actually inverts (no sync)."""
        return torch.where(filt > 0, sigma, torch.full_like(sigma, float("inf"))).min()

    def _record_rank(self, kept: torch.Tensor) -> None:
        """Append the per-step retained rank as a Python ``int``.

        Eager on purpose: the runner reads ``svd_info`` with ``np.asarray``,
        which cannot convert a CUDA tensor, and both step paths already
        synchronise once for correctness anyway (the divergence guard here,
        ``pinv``'s rtol slice on the classic path), so deferring this bought
        nothing.  Where the step's own sync can carry the count, it does
        (see :meth:`SvenGram.step`) and this helper is not used.
        """
        self.svd_info["num_nonzero_svs"].append(int(torch.count_nonzero(kept)))

    def _log_step(
        self,
        sigma_full: torch.Tensor,
        utr: torch.Tensor,
        update: torch.Tensor,
        resid: torch.Tensor,
        sv_min_kept: torch.Tensor,
    ) -> None:
        """Append one logged step's spectrum record to ``svd_info``.

        ``update`` is the raw solve; what is recorded is the applied change
        ``lr * update`` (see the class docstring).
        """
        info = self.svd_info
        info["step"].append(self.step_count)
        info["svs"].append(sigma_full.cpu().numpy())
        info["utr"].append(utr.cpu().numpy())
        # The four scalars travel in one transfer
        eps = torch.finfo(self.model.params.dtype).eps
        scalars = torch.stack(
            (
                (self._get_lr() * update.detach().norm()).to(torch.float64),
                resid.detach().norm().to(torch.float64),
                sv_min_kept.to(torch.float64),
                (eps ** 0.5 * sigma_full[0]).to(torch.float64),
            )
        ).tolist()
        info["update_norm"].append(scalars[0])
        info["resid_norm"].append(scalars[1])
        info["sv_min_kept"].append(scalars[2])
        info["sv_noise_floor"].append(scalars[3])

    def finalize_svd_info(self) -> dict[str, Any]:
        """Return the record as numpy arrays.  Convenience, not a requirement.

        ``svd_info`` is always readable directly (per-step lists of Python
        numbers / numpy arrays); this only stacks it.  Idempotent: safe to
        call after every epoch and again at the end of a run.
        """
        info = self.svd_info
        out: dict[str, Any] = {}
        for key, values in info.items():
            if key == "variable_k_substep_losses":
                out[key] = values  # ragged nested lists of tensors
            elif key in ("svs", "utr"):
                out[key] = [np.asarray(v) for v in values]
            elif key in ("step", "num_nonzero_svs", "k_used"):
                out[key] = np.asarray(values, dtype=np.int64)
            else:
                out[key] = np.asarray(values, dtype=np.float64)
        return out

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
    ) -> torch.Tensor:
        """Standard pseudo-inverse parameter update; returns the update."""
        update = self._compute_delta(U_T, S_inv, VhT, residuals)
        self._apply_update(update)
        return update

    @torch.no_grad()
    def _update_params_variable_k(
        self,
        batch: tuple[torch.Tensor, ...],
        U_T: torch.Tensor,
        S_inv: torch.Tensor,
        VhT: torch.Tensor,
        residuals: torch.Tensor,
        losses: torch.Tensor
    ) -> tuple[int, list[torch.Tensor], torch.Tensor]:
        """Greedy rank-1 updates, stopping when the loss increases.

        Returns ``(k_used, substep_losses, applied)`` — ``applied`` is the sum
        of the accepted rank-1 updates, for ``update_norm``.
        """
        original_loss = losses.mean()
        kmax = len(S_inv)
        x, *args = batch

        substep_losses: list[torch.Tensor] = [original_loss]
        k_used = 0
        # accumulated in place: one extra (P,) tensor per step, not per component
        applied: torch.Tensor | None = None
        while k_used < kmax:
            update = self._compute_delta_k(k_used, U_T, S_inv, VhT, residuals)
            self._apply_update(update)

            new_loss = self.model.evaluate_and_loss(x, *args).mean()
            if new_loss > original_loss:
                self._apply_update(-update)
                break

            applied = update.clone() if applied is None else applied.add_(update)
            substep_losses.append(new_loss)
            k_used += 1

        if applied is None:  # no component accepted: zero-norm sentinel
            applied = torch.zeros(
                (), dtype=self.model.params.dtype, device=self.model.params.device
            )
        return k_used, substep_losses, applied

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

        VhT, S_inv, U_T = pinv(
            jacobian,
            k=self.k,
            rtol=self.rtol,
            mode=self.svd_mode,
            power_iter=self.power_iterations,
        )

        # pinv() only returns the k (+ oversampling) singular values left after
        # its rtol cut, so the FULL spectrum cannot be recovered from it; on a
        # logged step recompute it from a float64 eigh of J J^T while J is
        # still alive.  U is then the Gram eigenbasis, i.e. the left singular
        # vectors of J, which is the basis ``utr`` is expressed in.
        log = self.track_svd_info and self.log_this_step
        if log:
            sigma_full, _, U_full = self._spectrum_from_gram(self._gram_fp64(jacobian))
            resid64 = residuals.detach().to(torch.float64)
            utr = U_full.T @ resid64
            del U_full

        del jacobian
        self._maybe_empty_cache()

        # Update parameters
        if self.variable_k:
            if batch is None:
                raise ValueError("batch must be provided when variable_k=True")
            k_used, substep_losses, update = self._update_params_variable_k(
                batch, U_T, S_inv, VhT, residuals, losses
            )
        else:
            update = self._update_params(U_T, S_inv, VhT, residuals)

        # Record diagnostics
        if self.track_svd_info:
            if log:
                # S_inv is >= 0 with the discarded entries zeroed, so 1/max is
                # the smallest singular value inverted; under variable_k only
                # the k_used accepted components were applied
                kept = S_inv[:k_used] if self.variable_k else S_inv
                inv_max = kept.max() if kept.numel() else S_inv.new_zeros(())
                self._log_step(sigma_full, utr, update, resid64, 1.0 / inv_max)
                del sigma_full, utr, resid64
            self._record_rank(S_inv)
            if self.variable_k:
                self.svd_info["k_used"].append(k_used)
                self.svd_info["variable_k_substep_losses"].append(substep_losses)

        self.step_count += 1
        del VhT, S_inv, U_T, update
        del self.model.residuals, self.model.grads, self.model.losses
        self._maybe_empty_cache()


class SvenGram(Sven):
    """Gram-matrix (kernel-trick) variant of :class:`Sven`.

    Consumes ``model.gram = J J^T`` (M, M) and ``model.residuals`` from a
    :class:`sven.nn.GramSvenWrapper` instead of the (M, P) Jacobian.  With
    ``G = U S^2 U^T`` the pseudo-inverse update is recovered as
    ``delta = J^T w`` with ``w = U_k S_k^{-2} U_k^T r`` via one backward pass
    (:meth:`GramSvenWrapper.delta_from_w`), so the Jacobian is never
    materialised.  Truncation replicates :func:`sven.opt.pinv` exactly.

    ``variable_k`` needs the Jacobian itself and raises ``NotImplementedError``.

    Args:
        model: A :class:`GramSvenWrapper` instance.
        lr: Learning rate.
        k: Number of singular values to keep in the truncated eig of ``G``.
        rtol: Relative tolerance for singular-value truncation.
        track_svd_info: If ``True``, record singular values and rank info.
        variable_k: Unsupported — raises ``NotImplementedError``.
        empty_cache: See :class:`Sven`.
    """

    _SIGMA_TOL: float = 1e-10  # absolute sigma cutoff, matches pinv() default

    def __init__(
        self,
        model: GramSvenWrapper,
        lr: float,
        k: int,
        rtol: float,
        track_svd_info: bool = False,
        variable_k: bool = False,
        empty_cache: bool = False,
    ) -> None:
        if variable_k:
            raise NotImplementedError(
                "variable_k needs per-component Jacobian updates; use Sven with SvenWrapper"
            )
        super().__init__(
            model, lr, k, rtol, track_svd_info=track_svd_info, empty_cache=empty_cache
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

        # Eig of G = U S^2 U^T in fp64, descending.  The FULL spectrum of the
        # batch Jacobian (all M values, before the k / rtol cut) is free here;
        # keep it for the diagnostics so the recorded spectrum is not truncated
        # at rtol.  ``num_nonzero_svs`` still counts the SVs used.
        sigma_full, _, U_full = self._spectrum_from_gram(gram.detach().to(torch.float64))
        sigma = sigma_full[: self.k]
        U = U_full[:, : self.k]

        # pinv() truncation semantics: rtol relative to sigma_max, then abs
        # tol.  Applied as a boolean mask multiplied into the filter, not as a
        # slice: sigma is descending, so the mask is that same prefix, and
        # finding its length would cost a device sync on every step.
        keep = sigma > self.rtol * sigma[0]
        s_inv_sq = torch.where(
            keep & (sigma > self._SIGMA_TOL), 1.0 / sigma.pow(2), torch.zeros_like(sigma)
        )
        # ONE host transfer carries both the divergence guard and the retained
        # rank, so a step (logged or not) costs no more syncs than correctness
        # already needs.
        any_kept, n_kept = torch.stack(
            (keep.any().to(torch.int64), torch.count_nonzero(s_inv_sq))
        ).tolist()
        if not any_kept:  # Gram is zero or NaN: the run has diverged
            raise RuntimeError(
                "SvenGram: no singular value above rtol * sigma_max "
                f"(sigma_max={sigma[0].item():.3g}); the Gram matrix is zero or non-finite -- run diverged"
            )

        rhs = residuals.detach().to(torch.float64)
        w = U @ (s_inv_sq * (U.T @ rhs))
        update = self.model.delta_from_w(w)  # J^T w, flat (P,)
        self._apply_update(update)

        # Record diagnostics
        if self.track_svd_info:
            if self.log_this_step:
                self._log_step(
                    sigma_full,
                    U_full.T @ rhs,
                    update,
                    rhs,
                    self._sv_min_kept(sigma, s_inv_sq),
                )
            self.svd_info["num_nonzero_svs"].append(n_kept)  # from the guard's transfer

        self.step_count += 1
        del sigma_full, U_full, sigma, U, keep, s_inv_sq, rhs, w, update
        del self.model.gram, self.model.residuals, self.model.losses
        self._maybe_empty_cache()


class SvenGramReg(SvenGram):
    """:class:`SvenGram` with weight decay and Tikhonov damping, solved exactly.

    Per step, solves the penalized linearized problem

        min_d ||R + M d||^2 + mu ||d||^2 + lam_E ||theta + d||^2
                                         + lam_F ||M (theta + d)||^2,

    whose exact solution, via the Woodbury/push-through identity with
    ``c = lam_E + mu`` and ``G = M M^T``, is

        d = -M^T [(1 + lam_F) G + c I]^{-1} (R + jvp_coef * M theta)
            - (lam_E / c) theta,        jvp_coef = lam_F - (lam_E/c)(1 + lam_F).

    The only extra cost over :class:`SvenGram` is one forward-mode JVP for
    ``M theta`` (skipped when ``jvp_coef = 0``).  The three weight-decay modes,
    and what they do per eigendirection of ``M^T M`` (singular value sigma_i):

    - ``decoupled_weight_decay`` (AdamW-style): ``theta <- (1 - lr * wd) theta``
      applied outside the solve, exactly AdamW's decoupled decay.  Uniform
      full-rate decay of every parameter, blind to the data.
    - ``weight_decay`` (lam_E, coupled Euclidean ridge on the *new* parameters):
      per-direction decay ``theta_i <- theta_i (1 - lr lam_E/(sigma_i^2 + c))``
      — full-rate on directions the batch Jacobian does not constrain
      (null/gauge directions), attenuated on constrained ones.
    - ``fisher_decay`` (lam_F, the same ridge measured in Sven's implied metric
      ``M^T M``): uniform decay of the *constrained* directions only
      (``theta_i <- theta_i (1 - lr lam_F/(1 + lam_F))`` for sigma_i > 0 at
      mu = 0) and none of the null space — margin control in function space,
      no pull on parameters invisible to the data.

    ``damping`` (mu) is Tikhonov/Levenberg-Marquardt damping of the update
    itself: it soft-filters the solve (``1/(sigma^2 + c)``) and exerts no pull
    on the parameters — it is not a weight decay.

    Units: with ``relative=True`` (default), ``lam_E`` and ``mu`` are in units
    of the current ``sigma_max^2`` of ``G``, so the spectral crossover
    ``sigma^2 ~ c`` tracks the spectrum scale; the per-step decay rate of
    unconstrained directions is ``lr * lam_E/(lam_E + mu)`` in either mode.
    ``fisher_decay`` and ``decoupled_weight_decay`` are dimensionless.

    Truncation semantics: with a coupled regularizer active (``c > 0`` or
    ``lam_F > 0``) the solve uses the FULL spectrum and ``k`` / ``rtol`` are
    inert — hard truncation would silently break the exact-solution property
    (dropped-but-constrained directions would receive full-rate decay and no
    data term).  Directions at the parameter-dtype noise floor
    (``sigma <= eps * sigma_max``) are dropped, an exact-arithmetic no-op
    (``M^T u = 0`` there) that avoids amplifying backward-pass noise by
    ``1/c``.  With every knob at zero the class reproduces stock
    :class:`SvenGram` (``k`` / ``rtol`` hard truncation) bit-for-bit.  A
    numerically zero Gram yields the well-defined degenerate answer: no data
    term, pure decay (stock raises here instead).

    With ``param_fraction < 1`` the penalties apply to the step's active
    subset: ``M`` is the masked Jacobian and ``theta`` the active parameter
    values, so each step decays only the parameters it updates (average decay
    rate scales with the fraction).

    With ``track_svd_info`` and a coupled regularizer active,
    ``num_nonzero_svs`` records the number of data-dominated directions
    (``sigma^2 > c``) — the effective rank of the soft-filtered solve.  The
    logged ``utr`` / ``resid_norm`` refer to the rhs actually solved against,
    which with ``jvp_coef != 0`` is ``r + jvp_coef * M theta`` rather than the
    residual rows alone, and ``sv_min_kept`` is the smallest singular value
    the (soft) filter inverts, i.e. the noise-floor cut in the regularized
    branch and the ``k`` / ``rtol`` cut in the stock branch.

    Args (beyond :class:`SvenGram`'s):
        damping: mu >= 0, Tikhonov damping of the update.
        weight_decay: lam_E >= 0, coupled Euclidean L2 penalty coefficient.
        fisher_decay: lam_F >= 0, coupled Fisher-metric penalty coefficient.
        decoupled_weight_decay: AdamW-style decay rate (per step: ``lr`` times
            this), applied outside the solve.
        relative: if ``True`` (default), ``lam_E`` and ``mu`` are in units of
            the current ``sigma_max^2`` of ``G``.
    """

    def __init__(
        self,
        model: GramSvenWrapper,
        lr: float,
        k: int,
        rtol: float,
        damping: float = 0.0,
        weight_decay: float = 0.0,
        fisher_decay: float = 0.0,
        decoupled_weight_decay: float = 0.0,
        relative: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(model, lr, k, rtol, **kwargs)
        if min(damping, weight_decay, fisher_decay, decoupled_weight_decay) < 0:
            raise ValueError(
                "damping, weight_decay, fisher_decay and decoupled_weight_decay "
                "must be >= 0"
            )
        self.damping = damping
        self.weight_decay = weight_decay
        self.fisher_decay = fisher_decay
        self.decoupled_weight_decay = decoupled_weight_decay
        self.relative = relative

    def _active_params(self) -> torch.Tensor:
        """The step's decayed parameter vector: ``params`` gathered at the mask."""
        theta = self.model.params.detach()
        if self.model.param_mask is not None:
            theta = theta[self.model.param_mask]
        return theta

    def _rows_jvp(self) -> torch.Tensor:
        """``M @ theta``: directional derivative of the residual rows along theta."""
        x, *args = self.model._batch
        theta = self._active_params()
        rows_fn = lambda flat: self.model._loss(flat, x, *args)[0]
        with self.model._pass_norm_stats():
            _, v = torch.func.jvp(rows_fn, (theta,), (theta,))
        return v.detach().to(torch.float64)

    @torch.no_grad()
    def step(self, batch: tuple[torch.Tensor, ...] | None = None) -> None:
        """Compute and apply the regularized Gram-based update.

        Args:
            batch: Ignored — kept for signature compatibility with ``Sven``.
        """
        gram = getattr(self.model, "gram", None)
        if gram is None:
            raise TypeError(
                "SvenGramReg needs model.gram: wrap the model with GramSvenWrapper "
                "and call loss_and_grad() before each step"
            )
        # Eig of G = U S^2 U^T in fp64, descending
        sigma, sigma_sq, U = self._spectrum_from_gram(gram.detach().to(torch.float64))
        sigma_full, U_full = sigma, U  # full spectrum / eigenbasis, for diagnostics

        scale = sigma_sq[0] if self.relative else 1.0
        lam_e = self.weight_decay * scale
        lam_f = float(self.fisher_decay)  # dimensionless: the metric carries the sigma^2 units
        mu = self.damping * scale
        c = lam_e + mu
        # lam_e/c is scale-invariant; the raw ratio stays defined when a zero
        # Gram collapses the relative scale (degenerate case: pure decay)
        raw_c = self.weight_decay + self.damping
        shrink = float(self.weight_decay / raw_c) if raw_c > 0 else 0.0
        # rhs coefficient on the JVP M theta from the Woodbury split (see class docstring)
        jvp_coef = lam_f - shrink * (1.0 + lam_f)

        # Both branches discard directions by multiplying a boolean mask into
        # the filter rather than index-slicing: the slice length is
        # data-dependent and would synchronise the device on every step.
        if lam_f > 0 or raw_c > 0:
            # exact regularized solve: full spectrum, soft filter; k/rtol inert.
            # Drop parameter-dtype noise-floor directions — exact-arithmetic
            # no-op (M^T u = 0) that avoids 1/c amplification of fp32 noise.
            eps = torch.finfo(self.model.params.dtype).eps
            keep = sigma > eps * sigma[0]
            filt = torch.where(
                keep, 1.0 / ((1.0 + lam_f) * sigma_sq + c), torch.zeros_like(sigma)
            )
        else:
            # coupled knobs zero: reproduce stock SvenGram exactly (k/rtol
            # truncation, 1/sigma^2 via the same clamp -> sqrt -> pow sequence).
            # A numerically zero G leaves every entry masked out: no data term.
            sigma, sigma_sq, U = sigma[: self.k], sigma_sq[: self.k], U[:, : self.k]
            keep = sigma > self.rtol * sigma[0]
            filt = torch.where(
                keep & (sigma > self._SIGMA_TOL), 1.0 / sigma.pow(2), torch.zeros_like(sigma)
            )

        # data-dominated directions with a coupled regularizer, kept rank otherwise
        rank_mask = (sigma_sq > c) & keep if (raw_c > 0 or lam_f > 0) else filt
        # filt >= 0 by construction, so count_nonzero(filt) > 0 is the "is there
        # a data term at all" guard; both counts ride one host transfer.
        if filt.numel():
            n_filt, n_rank = torch.stack(
                (torch.count_nonzero(filt), torch.count_nonzero(rank_mask))
            ).tolist()
        else:
            n_filt, n_rank = 0, 0

        rhs = self.model.residuals.detach().to(torch.float64)
        if n_filt:
            if jvp_coef != 0.0:
                rhs = rhs + jvp_coef * self._rows_jvp()
            w = U @ (filt * (U.T @ rhs))
            update = self.model.delta_from_w(w)  # M^T w, flat (P,) or (n_active,)
        else:
            update = torch.zeros(
                self._active_params().shape,
                dtype=self.model.params.dtype,
                device=self.model.params.device,
            )

        if shrink > 0 or self.decoupled_weight_decay > 0:
            # theta <- (1 - lr*(lam_E/c + wd_dec)) theta - lr M^T w
            update = update + (shrink + self.decoupled_weight_decay) * self._active_params()

        self._apply_update(update)

        if self.track_svd_info:
            if self.log_this_step:
                self._log_step(
                    sigma_full,
                    U_full.T @ rhs,
                    update,
                    rhs,
                    self._sv_min_kept(sigma, filt),
                )
            self.svd_info["num_nonzero_svs"].append(n_rank)  # from the guard's transfer

        self.step_count += 1
        del sigma, sigma_full, sigma_sq, U, U_full, keep, filt, rank_mask, rhs, update
        del self.model.gram, self.model.residuals, self.model.losses
        self._maybe_empty_cache()

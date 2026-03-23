"""Polyak SGD optimizer with automatic step-size selection."""

from __future__ import annotations

from typing import Callable

import torch
from torch.optim import Optimizer


class PolyakSGD(Optimizer):
    """SGD with Polyak step sizes.

    The learning rate is computed automatically each step as::

        lr = (loss - f_star) / (||grad||^2 + eps)

    This removes the need for learning-rate tuning but requires a reasonable
    estimate of the minimum achievable loss ``f_star``.

    Args:
        params: Model parameters.
        f_star: Estimated minimum loss (``0.0`` for interpolating models).
        max_lr: Upper bound on the step size to prevent huge steps.
        eps: Numerical stability constant.
    """

    def __init__(
        self,
        params,
        f_star: float = 0.0,
        max_lr: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        defaults = dict(f_star=f_star, max_lr=max_lr, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Perform a single optimisation step.

        Args:
            closure: A callable that re-evaluates the model and returns the
                loss.  Called with gradients enabled.
        """
        with torch.enable_grad():
            loss = closure()

        for group in self.param_groups:
            f_star: float = group["f_star"]
            max_lr: float = group["max_lr"]
            eps: float = group["eps"]

            grad_sq_norm = sum(
                p.grad.detach().norm() ** 2
                for p in group["params"]
                if p.grad is not None
            )

            lr = (loss.item() - f_star) / (grad_sq_norm.item() + eps)
            lr = min(lr, max_lr)

            for p in group["params"]:
                if p.grad is not None:
                    p.data.add_(p.grad.detach(), alpha=-lr)

        return loss

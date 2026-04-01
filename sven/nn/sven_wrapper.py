from __future__ import annotations

import inspect
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.func import functional_call
from torch.nn.utils import parameters_to_vector


class SvenWrapper:
    """Functional wrapper around a PyTorch model for per-sample Jacobian computation.

    Converts a standard ``nn.Module`` into a functional form so that
    ``torch.func.jacrev`` can compute per-sample Jacobians of the loss with
    respect to a flat parameter vector.

    Args:
        model: The PyTorch model to wrap.
        loss_fn: A loss function ``(pred, *args) -> Tensor`` that returns
            **per-sample** losses with shape ``(B,)``.
        device: Device to place the model and parameters on.
        kappa: Exponent for raw loss function when computing the Jacobian and updates with L = (L^{kappa/2})^{2/kappa} (default: kappa = 2 for the usual derivatives of the raw loss function).
        param_fraction: Fraction of parameters to compute the Jacobian with
            respect to on each step.  ``1.0`` uses all parameters.
        mask_by_block: If ``True``, mask entire parameter blocks (layers)
            instead of individual parameters when ``param_fraction < 1``.
        microbatch_size: If ``> 1``, aggregate losses within sub-batches of
            this size before computing the Jacobian, reducing its row dimension.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable[..., torch.Tensor],
        device: torch.device | str,
        kappa: float = 2.0,
        param_fraction: float = 1.0,
        mask_by_block: bool = False,
        microbatch_size: int = 1,
    ) -> None:
        self.model: nn.Module = model.to(device)
        self.device: torch.device = torch.device(device) if isinstance(device, str) else device
        self.loss_fn: Callable[..., torch.Tensor] = loss_fn
        self.kappa: float = kappa
        self.param_names_counts_startIdx: list[tuple[str, int, int]] = []
        self.params: torch.Tensor = self._tie_parameters_to_flat(requires_grad=False)
        self.params.requires_grad_(True)

        self.param_fraction: float = param_fraction
        self.mask_by_block: bool = mask_by_block
        self.param_mask: torch.Tensor | None = None
        self.microbatch_size: int = microbatch_size
        self.n_params: int = self.params.shape[0]
        self.param_shapes: list[tuple[str, torch.Size, int]] = [
            (name, param.shape, param.numel()) for name, param in model.named_parameters()
        ]

        self.num_loss_args: int = len(inspect.signature(loss_fn).parameters) - 1

        # Populated by loss_and_grad(), consumed by optimizer.step()
        self.grads: torch.Tensor = torch.empty(0, device=self.device)
        self.losses: torch.Tensor = torch.empty(0, device=self.device)

    # ------------------------------------------------------------------
    # Forward / evaluation
    # ------------------------------------------------------------------

    def _func_call(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Functional forward pass through the model."""
        param_dict: dict[str, torch.Tensor] = {}
        start_idx = 0
        for name, shape, size in self.param_shapes:
            param_dict[name] = params[start_idx : start_idx + size].view(shape)
            start_idx += size
        # Fetch fresh buffers on every call (includes updated BatchNorm stats)
        for name, buffer in self.model.named_buffers():
            param_dict[name] = buffer
        return functional_call(self.model, param_dict, x)

    @torch.no_grad()
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass without gradient tracking."""
        return self._func_call(self.params, x)

    @torch.no_grad()
    def evaluate_and_loss(self, x: torch.Tensor, *args: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return per-sample losses."""
        pred = self.evaluate(x)
        return self.loss_fn(pred, *args)

    # ------------------------------------------------------------------
    # Loss / Jacobian computation
    # ------------------------------------------------------------------

    def _loss(
        self, params: torch.Tensor, x: torch.Tensor, *args: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Compute per-sample losses, returning aux data for ``jacrev``."""
        if self.param_mask is not None:
            input_params = self.params.clone()
            input_params[self.param_mask] = params
        else:
            input_params = params

        pred = self._func_call(input_params, x)
        loss = self.loss_fn(pred, *args)
        if self.microbatch_size > 1:
            loss = loss.view(-1, self.microbatch_size).mean(dim=1)
        return loss.pow(self.kappa / 2.0), (loss, pred) # return raw loss in auxiliary output, use loss ^ (kappa / 2) for gradients

    def _batch_gradient(
        self, batch: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-sample Jacobian via ``jacrev``."""
        x, *args = batch
        params = self.params[self.param_mask] if self.param_mask is not None else self.params
        grads, (losses, preds) = torch.func.jacrev(self._loss, argnums=0, has_aux=True)(
            params, x, *args
        )
        return grads, losses, preds

    def loss_and_grad(
        self, batch: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute losses and per-sample Jacobian for the optimizer.

        The Jacobian is stored in ``self.grads`` and losses in ``self.losses``
        for consumption by :meth:`Sven.step`.

        Args:
            batch: Tuple of ``(x, y, ...)`` tensors.

        Returns:
            ``(losses, predictions)`` — both detached from the compute graph.
        """
        if self.param_fraction < 1.0:
            if self.mask_by_block:
                self.param_mask = self._make_param_mask_by_block(self.param_fraction).to(
                    self.params.device
                )
            else:
                self.param_mask = self._make_param_mask().to(self.params.device)

        grads, losses, preds = self._batch_gradient(batch)

        self.grads = grads.detach()
        self.losses = losses.detach()
        self.residuals = self.losses.pow(self.kappa / 2.0).detach() # store the "residuals" (loss^(kappa/2)) for use in the update step

        return self.losses, preds

    # ------------------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------------------

    def _tie_parameters_to_flat(self, requires_grad: bool = False) -> torch.Tensor:
        """Flatten all model parameters into a single vector and rebind them as views."""
        flat = parameters_to_vector(self.model.parameters()).detach()
        flat = flat.requires_grad_(requires_grad)

        start = 0
        for name, p in self.model.named_parameters():
            n = p.numel()
            view = flat[start : start + n].view_as(p)
            self.param_names_counts_startIdx.append((name, n, start))
            start += n

            # Walk to the owning module and replace the parameter storage
            mod: nn.Module = self.model
            *prefix, leaf = name.split(".")
            for part in prefix:
                mod = getattr(mod, part)
            mod._parameters[leaf] = nn.Parameter(view, requires_grad=requires_grad)

        return flat

    def _make_param_mask(self) -> torch.Tensor:
        """Create a random mask selecting ``param_fraction`` of parameters."""
        n_active = int(self.param_fraction * self.n_params)
        mask = torch.zeros(self.n_params, dtype=torch.bool)
        mask[torch.randperm(self.n_params)[:n_active]] = True
        return mask

    def _make_param_mask_by_block(self, fraction: float) -> torch.Tensor:
        """Create a mask that selects entire parameter blocks (layers) randomly."""
        n_blocks = len(self.param_names_counts_startIdx)
        random_order = torch.randperm(n_blocks)

        running = 0
        target = int(fraction * self.n_params)
        mask = torch.zeros(self.n_params, dtype=torch.bool)
        for i in random_order:
            _, nparam, start_idx = self.param_names_counts_startIdx[i]
            if running + nparam >= target:
                n_to_use = target - running
                mask[start_idx : start_idx + n_to_use] = True
                break
            mask[start_idx : start_idx + nparam] = True
            running += nparam

        return mask

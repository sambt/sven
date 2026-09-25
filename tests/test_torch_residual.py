"""Signed-residual rows (``residual_fn``) for scalar-output regression.

For a scalar residual r with loss = r**2 the rows ``sign(r)|r|**kappa`` differ
from the loss-path rows ``loss**(kappa/2) = |r|**kappa`` only by a per-row
sign, which flips a Jacobian row and its residual together, so the Sven update
is identical (up to rounding) -- but the residual path has a finite gradient
at r = 0, where the loss path produces inf * 0 = NaN for kappa < 2.
All on CPU in float64.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from sven.nn import GramSvenWrapper, SvenWrapper
from sven.opt import Sven, SvenGram

DT = torch.float64
DEVICE = "cpu"


def per_sample_mse(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((pred - y) ** 2).sum(dim=-1)


def residual(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return pred - y  # (B, 1)


def make_mlp(seed: int = 0, d_out: int = 1) -> nn.Sequential:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 8), nn.Tanh(),
                         nn.Linear(8, d_out)).to(DT)


def make_batch(b: int = 12, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(b, 2, generator=g, dtype=DT),
            torch.randn(b, 1, generator=g, dtype=DT))


def one_step(backend: str, kappa: float, use_residual: bool, batch, k: int = 6,
             lr: float = 0.5, rtol: float = 1e-6, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat params after one Sven step and the wrapper's residual vector."""
    model = make_mlp(seed)
    rf = residual if use_residual else None
    if backend == "classic":
        w = SvenWrapper(model, per_sample_mse, DEVICE, kappa=kappa, residual_fn=rf)
        opt = Sven(w, lr=lr, k=k, rtol=rtol, svd_mode="torch")
    else:
        w = GramSvenWrapper(model, per_sample_mse, DEVICE, kappa=kappa, residual_fn=rf,
                            capture=backend)
        opt = SvenGram(w, lr=lr, k=k, rtol=rtol)
    w.loss_and_grad(batch)
    res = w.residuals.detach().clone()
    opt.step(batch)
    return w.params.detach().clone(), res


@pytest.mark.parametrize("backend", ["classic", "hooks", "chunked"])
@pytest.mark.parametrize("kappa", [1.0, 1.5, 2.0, 3.0])
def test_residual_path_matches_loss_path(backend: str, kappa: float) -> None:
    batch = make_batch()
    p_loss, r_loss = one_step(backend, kappa, False, batch)
    p_res, r_res = one_step(backend, kappa, True, batch)
    # residuals agree up to a per-row sign: |rows| identical
    torch.testing.assert_close(r_res.abs(), r_loss.abs(), rtol=1e-10, atol=1e-12)
    assert (r_res < 0).any() and (r_res > 0).any(), "signed residual should carry both signs"
    # and the update is identical
    torch.testing.assert_close(p_res, p_loss, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("backend", ["classic", "hooks", "chunked"])
def test_backends_agree_on_residual_path(backend: str) -> None:
    batch = make_batch()
    p_ref, _ = one_step("classic", 1.0, True, batch)
    p, _ = one_step(backend, 1.0, True, batch)
    torch.testing.assert_close(p, p_ref, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("backend", ["classic", "hooks", "chunked"])
@pytest.mark.parametrize("kappa", [1.0, 1.5])
def test_residual_path_finite_at_zero_residual(backend: str, kappa: float) -> None:
    """A sample with r == 0 exactly: loss path NaNs (documented), residual
    path stays finite and the surviving update matches the loss path with
    that sample's row removed from the pseudo-inverse."""
    x, y = make_batch()
    model = make_mlp()
    with torch.no_grad():
        y[0] = model(x[:1])  # exact zero residual for sample 0
    batch = (x, y)
    p_res, r_res = one_step(backend, kappa, True, batch)
    assert torch.isfinite(p_res).all()
    assert r_res[0] == 0.0 and torch.isfinite(r_res).all()
    # The loss path hits inf * 0 in the Jacobian: the classic pinv raises on a
    # non-finite matrix, the Gram path propagates NaN into the parameters.
    try:
        p_loss, _ = one_step(backend, kappa, False, batch)
    except (RuntimeError, torch._C._LinAlgError):
        return
    assert not torch.isfinite(p_loss).all(), "loss path was expected to NaN at r == 0"


def test_vector_residual_rejected() -> None:
    model = make_mlp(d_out=3)
    w = SvenWrapper(model, per_sample_mse, DEVICE, kappa=1.0, residual_fn=residual)
    x = torch.randn(4, 2, dtype=DT)
    y = torch.randn(4, 3, dtype=DT)
    with pytest.raises(ValueError, match="one scalar residual per sample"):
        w.loss_and_grad((x, y))


def test_microbatch_rejected() -> None:
    with pytest.raises(ValueError, match="microbatch_size"):
        SvenWrapper(make_mlp(), per_sample_mse, DEVICE, microbatch_size=2, residual_fn=residual)
    with pytest.raises(ValueError, match="microbatch_size"):
        GramSvenWrapper(make_mlp(), per_sample_mse, DEVICE, microbatch_size=2,
                        residual_fn=residual)

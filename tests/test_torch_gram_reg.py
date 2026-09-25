"""Correctness tests for SvenGramReg (weight decay / damping on SvenGram).

Every regularized update is checked against the dense normal-equations solve

    [(1 + lam_F) M^T M + (lam_E + mu) I] d = -M^T (R + lam_F M theta) - lam_E theta

with the (M, P) Jacobian materialized via jacrev on the same wrapper (same
batch, same mask).  All on CPU in float64 so exactness assertions are
meaningful.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from sven.nn import GramSvenWrapper
from sven.opt import SvenGram, SvenGramReg

DT = torch.float64
DEVICE = "cpu"


def per_sample_mse(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((pred - y) ** 2).mean(dim=1)


def make_mlp(dims: list[int], seed: int = 0) -> nn.Sequential:
    torch.manual_seed(seed)
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers).to(DT)


def make_data(b: int, d_in: int, d_out: int, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, d_in, generator=g, dtype=DT)
    y = torch.randn(b, d_out, generator=g, dtype=DT)
    return x, y


def run_reg_step(
    template: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    opt_kwargs: dict,
    wrapper_kwargs: dict | None = None,
    lr: float = 1.0,
    seed: int = 7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, GramSvenWrapper]:
    """One SvenGramReg step.  Returns (theta_before_active, applied_delta,
    dense_jacobian_at_mask, wrapper); the mask (if any) is the one the step
    actually drew."""
    wrapper = GramSvenWrapper(
        copy.deepcopy(template), per_sample_mse, DEVICE, **(wrapper_kwargs or {})
    )
    opt = SvenGramReg(wrapper, lr=lr, k=x.shape[0], rtol=0.0, **opt_kwargs)
    torch.manual_seed(seed)  # fixes the mask draw
    wrapper.loss_and_grad((x, y))
    # Dense Jacobian at the drawn mask, for the reference solve
    theta0_full = wrapper.params.detach().clone()
    theta0 = theta0_full[wrapper.param_mask] if wrapper.param_mask is not None else theta0_full
    jac = torch.func.jacrev(lambda p: wrapper._loss(p, x, y)[0])(theta0)
    opt.step()
    theta1_full = wrapper.params.detach()
    theta1 = theta1_full[wrapper.param_mask] if wrapper.param_mask is not None else theta1_full
    if wrapper.param_mask is not None:
        # untouched parameters must be bit-identical
        inv = ~wrapper.param_mask
        assert torch.equal(theta1_full[inv], theta0_full[inv])
    return theta0, theta1 - theta0, jac.detach(), wrapper


def dense_reference(
    jac: torch.Tensor,
    residuals: torch.Tensor,
    theta0: torch.Tensor,
    lam_e: float,
    lam_f: float,
    mu: float,
) -> torch.Tensor:
    p = jac.shape[1]
    a = (1.0 + lam_f) * (jac.T @ jac) + (lam_e + mu) * torch.eye(p, dtype=DT)
    rhs = -jac.T @ (residuals + lam_f * (jac @ theta0)) - lam_e * theta0
    if lam_e + mu == 0.0:
        # fisher-only: the normal matrix is singular off row(J^T); the rhs
        # lies in row(J^T) and the optimizer returns the min-norm solution
        return torch.linalg.pinv(a) @ rhs
    return torch.linalg.solve(a, rhs)


def rel_scale(jac: torch.Tensor) -> float:
    """sigma_max^2 of G, the relative=True unit for lam_E and mu."""
    g = jac @ jac.T
    return float(torch.linalg.eigvalsh(g)[-1])


@pytest.mark.parametrize(
    "opt_kwargs",
    [
        {"weight_decay": 1e-2},
        {"weight_decay": 1e-2, "damping": 1e-3},
        {"damping": 1e-3},
        {"fisher_decay": 0.5},
        {"fisher_decay": 0.5, "weight_decay": 1e-2, "damping": 1e-3},
    ],
)
@pytest.mark.parametrize("relative", [True, False])
def test_reg_update_matches_dense_solve(opt_kwargs, relative):
    model = make_mlp([3, 10, 2])
    x, y = make_data(8, 3, 2)
    theta0, delta, jac, wrapper = run_reg_step(
        model, x, y, {**opt_kwargs, "relative": relative}
    )
    residuals = wrapper_residuals(wrapper, model, x, y)
    scale = rel_scale(jac) if relative else 1.0
    ref = dense_reference(
        jac,
        residuals,
        theta0,
        lam_e=opt_kwargs.get("weight_decay", 0.0) * scale,
        lam_f=opt_kwargs.get("fisher_decay", 0.0),
        mu=opt_kwargs.get("damping", 0.0) * scale,
    )
    torch.testing.assert_close(delta, ref, rtol=1e-8, atol=1e-10)


def test_reg_update_matches_dense_solve_kappa1():
    model = make_mlp([3, 10, 2])
    x, y = make_data(8, 3, 2)
    theta0, delta, jac, wrapper = run_reg_step(
        model, x, y, {"weight_decay": 1e-2, "damping": 1e-3},
        wrapper_kwargs={"kappa": 1.0},
    )
    scale = rel_scale(jac)
    ref = dense_reference(
        jac, wrapper_residuals(wrapper, model, x, y, kappa=1.0), theta0,
        lam_e=1e-2 * scale, lam_f=0.0, mu=1e-3 * scale,
    )
    torch.testing.assert_close(delta, ref, rtol=1e-8, atol=1e-10)


def wrapper_residuals(wrapper, model, x, y, kappa=2.0):
    loss = per_sample_mse(
        torch.func.functional_call(model, dict(model.named_parameters()), x), y
    )
    return loss.pow(kappa / 2.0).detach()


@pytest.mark.parametrize("mask_mode", ["elementwise", "rows", "tensor"])
def test_reg_update_matches_dense_solve_masked(mask_mode):
    model = make_mlp([3, 12, 2])
    x, y = make_data(8, 3, 2)
    theta0, delta, jac, wrapper = run_reg_step(
        model, x, y,
        {"weight_decay": 1e-2, "fisher_decay": 0.3, "damping": 1e-3},
        wrapper_kwargs={"param_fraction": 0.5, "mask_mode": mask_mode},
    )
    assert wrapper.param_mask is not None and jac.shape[1] == int(wrapper.param_mask.sum())
    scale = rel_scale(jac)
    ref = dense_reference(
        jac, wrapper_residuals(wrapper, model, x, y), theta0,
        lam_e=1e-2 * scale, lam_f=0.3, mu=1e-3 * scale,
    )
    torch.testing.assert_close(delta, ref, rtol=1e-8, atol=1e-10)


def test_all_knobs_zero_is_bitwise_stock():
    model = make_mlp([3, 10, 2])
    x, y = make_data(8, 3, 2)
    deltas = []
    for cls in (SvenGram, SvenGramReg):
        wrapper = GramSvenWrapper(copy.deepcopy(model), per_sample_mse, DEVICE)
        opt = cls(wrapper, lr=0.1, k=5, rtol=1e-4)
        theta0 = wrapper.params.detach().clone()
        wrapper.loss_and_grad((x, y))
        opt.step()
        deltas.append(wrapper.params.detach() - theta0)
    assert torch.equal(deltas[0], deltas[1])


def test_decoupled_weight_decay_is_adamw_shrink():
    model = make_mlp([3, 10, 2])
    x, y = make_data(8, 3, 2)
    lr, wd = 0.1, 0.3
    thetas = []
    for kwargs in ({}, {"decoupled_weight_decay": wd}):
        wrapper = GramSvenWrapper(copy.deepcopy(model), per_sample_mse, DEVICE)
        opt = SvenGramReg(wrapper, lr=lr, k=5, rtol=1e-4, **kwargs)
        wrapper.loss_and_grad((x, y))
        theta0 = wrapper.params.detach().clone()
        opt.step()
        thetas.append((theta0, wrapper.params.detach().clone()))
    (theta0, plain), (_, decayed) = thetas
    torch.testing.assert_close(decayed, plain - lr * wd * theta0, rtol=1e-12, atol=1e-14)


def test_decoupled_weight_decay_masked_decays_active_only():
    model = make_mlp([3, 12, 2])
    x, y = make_data(8, 3, 2)
    lr, wd = 0.1, 0.3
    thetas = []
    for kwargs in ({}, {"decoupled_weight_decay": wd}):
        wrapper = GramSvenWrapper(
            copy.deepcopy(model), per_sample_mse, DEVICE,
            param_fraction=0.5, mask_mode="elementwise",
        )
        opt = SvenGramReg(wrapper, lr=lr, k=8, rtol=1e-4, **kwargs)
        torch.manual_seed(11)  # identical mask across the two runs
        wrapper.loss_and_grad((x, y))
        theta0 = wrapper.params.detach().clone()
        opt.step()
        thetas.append((theta0, wrapper.params.detach().clone(), wrapper.param_mask))
    (theta0, plain, mask_a), (_, decayed, mask_b) = thetas
    assert torch.equal(mask_a, mask_b)
    expected = plain.clone()
    expected[mask_a] -= lr * wd * theta0[mask_a]
    torch.testing.assert_close(decayed, expected, rtol=1e-12, atol=1e-14)


def test_zero_gram_degenerate_pure_decay():
    """G numerically zero + coupled decay: pure shrink, no crash."""
    model = make_mlp([3, 10, 2])
    x, y = make_data(8, 3, 2)
    y = torch.func.functional_call(model, dict(model.named_parameters()), x).detach()
    # zero residuals => zero rows for kappa=2 (loss^1) ... but the Jacobian of
    # loss^{kappa/2} at zero loss also vanishes, so G = 0 exactly.
    wrapper = GramSvenWrapper(copy.deepcopy(model), per_sample_mse, DEVICE)
    lr, lam, mu = 0.5, 1e-2, 1e-3
    opt = SvenGramReg(wrapper, lr=lr, k=8, rtol=1e-4, weight_decay=lam, damping=mu)
    wrapper.loss_and_grad((x, y))
    assert torch.count_nonzero(wrapper.gram) == 0
    theta0 = wrapper.params.detach().clone()
    opt.step()
    shrink = lam / (lam + mu)
    torch.testing.assert_close(
        wrapper.params.detach(), (1 - lr * shrink) * theta0, rtol=1e-12, atol=1e-14
    )


def test_negative_knobs_raise():
    model = make_mlp([3, 6, 2])
    wrapper = GramSvenWrapper(model, per_sample_mse, DEVICE)
    with pytest.raises(ValueError):
        SvenGramReg(wrapper, lr=0.1, k=4, rtol=1e-4, weight_decay=-1.0)
    with pytest.raises(ValueError):
        SvenGramReg(wrapper, lr=0.1, k=4, rtol=1e-4, decoupled_weight_decay=-0.1)

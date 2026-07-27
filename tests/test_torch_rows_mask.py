"""Correctness tests for SvenWrapper row-block structural masking.

Rows mode (``mask_mode="rows"``) must select output rows/channels per
Linear/Conv2d, compute the Jacobian only w.r.t. them via the split forward of
the mask-aware twins, and concatenate the per-twin pieces in flat-vector order
so that ``Sven._apply_update``'s ``params[param_mask] += ...`` writes the
right entries.  All on CPU in float64.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from sven.nn import SvenWrapper
from sven.opt import Sven
from sven.opt.pinv import pinv

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


def make_data(b: int, d_in: int, d_out: int, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(b, d_in, generator=g, dtype=DT),
        torch.randn(b, d_out, generator=g, dtype=DT),
    )


def full_jacobian(template: nn.Module, x, y, **wrapper_kwargs) -> torch.Tensor:
    wrapper = SvenWrapper(copy.deepcopy(template), per_sample_mse, DEVICE, **wrapper_kwargs)
    wrapper.loss_and_grad((x, y))
    return wrapper.grads.detach()


def rows_wrapper(template: nn.Module, fraction: float, **wrapper_kwargs) -> SvenWrapper:
    return SvenWrapper(
        copy.deepcopy(template), per_sample_mse, DEVICE,
        param_fraction=fraction, mask_mode="rows", **wrapper_kwargs,
    )


# ----------------------------------------------------------------------
# (a) rows-mode Jacobian columns == full Jacobian columns
# ----------------------------------------------------------------------


def test_rows_mask_columns_match_full_jacobian():
    dims = [5, 7, 11, 13, 9, 3]  # unequal widths: per-layer row counts differ
    template = make_mlp(dims)
    x, y = make_data(8, 5, 3)
    j_full = full_jacobian(template, x, y)

    for seed in range(4):
        wrapper = rows_wrapper(template, 0.3)
        torch.manual_seed(seed)
        wrapper.loss_and_grad((x, y))
        mask = wrapper.param_mask
        n_active = int(mask.sum())
        assert 0 < n_active < wrapper.n_params  # nontrivial mask
        assert len(wrapper._row_selection) >= 2  # spans >= 2 modules
        touched = sum(
            1 for _, n, start in wrapper.param_names_counts_startIdx
            if mask[start : start + n].any()
        )
        assert touched >= 4  # weight + bias of >= 2 modules

        assert wrapper.grads.shape == (8, n_active)
        assert (wrapper.grads - j_full[:, mask]).abs().max() < 1e-12


# ----------------------------------------------------------------------
# (b) column order matches the params[param_mask] gather (plain autograd)
# ----------------------------------------------------------------------


def test_rows_mask_column_order_matches_flat_gather():
    template = make_mlp([5, 7, 11, 3])
    x, y = make_data(6, 5, 3)
    wrapper = rows_wrapper(template, 0.4)
    torch.manual_seed(2)
    wrapper.loss_and_grad((x, y))
    mask_idx = wrapper.param_mask.nonzero(as_tuple=True)[0]
    n = mask_idx.numel()

    # Ground truth without jacrev: plain autograd on an untouched model copy;
    # flat order = parameters_to_vector order = concat of parameter tensors.
    model = copy.deepcopy(template)
    losses = per_sample_mse(model(x), y)
    params_list = list(model.parameters())
    for m in range(0, 6, 2):
        gs = torch.autograd.grad(losses[m], params_list, retain_graph=True)
        g_flat = torch.cat([g.reshape(-1) for g in gs])
        for pos in (0, n // 3, (2 * n) // 3, n - 1):
            i = int(mask_idx[pos])
            assert (wrapper.grads[m, pos] - g_flat[i]).abs() < 1e-12


# ----------------------------------------------------------------------
# (c) Conv2d: split-channel exactness; grouped conv rejected
# ----------------------------------------------------------------------


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, stride=2, padding=1)            # bias=True
        self.conv2 = nn.Conv2d(4, 6, 3, padding=1, bias=False)
        self.fc = nn.Linear(6 * 4 * 4, 2)

    def forward(self, x):
        h = torch.tanh(self.conv1(x))  # 8x8 -> 4x4
        h = torch.tanh(self.conv2(h))
        return self.fc(h.flatten(1))


def test_cnn_rows_mask_matches_full_jacobian():
    torch.manual_seed(7)
    cnn = SmallCNN().to(DT)
    x = torch.randn(6, 1, 8, 8, dtype=DT)
    y = torch.randn(6, 2, dtype=DT)
    j_full = full_jacobian(cnn, x, y)

    wrapper = rows_wrapper(cnn, 0.4)
    torch.manual_seed(1)
    wrapper.loss_and_grad((x, y))
    assert wrapper.grads.shape == (6, int(wrapper.param_mask.sum()))
    assert (wrapper.grads - j_full[:, wrapper.param_mask]).abs().max() < 1e-12


def test_grouped_conv_raises():
    grouped = nn.Sequential(
        nn.Conv2d(4, 4, 3, padding=1, groups=2), nn.Flatten(), nn.Linear(4 * 4 * 4, 2)
    ).to(DT)
    with pytest.raises(NotImplementedError, match="groups"):
        SvenWrapper(grouped, per_sample_mse, DEVICE, param_fraction=0.5, mask_mode="rows")


# ----------------------------------------------------------------------
# (d) Sven.step on the rows-masked wrapper
# ----------------------------------------------------------------------


def test_rows_mask_step_matches_manual_pinv_update():
    template = make_mlp([5, 7, 11, 13, 9, 3])
    x, y = make_data(8, 5, 3)
    lr, k, rtol = 0.1, 8, 1e-10

    wrapper = rows_wrapper(template, 0.3)
    torch.manual_seed(3)
    wrapper.loss_and_grad((x, y))
    p0 = wrapper.params.detach().clone()
    mask = wrapper.param_mask.clone()

    # Hand-computed update from the masked Jacobian (step() consumes grads)
    VhT, S_inv, U_T = pinv(wrapper.grads.clone(), k=k, rtol=rtol, mode="torch")
    delta = Sven._compute_delta(U_T, S_inv, VhT, wrapper.residuals.clone())
    expected = p0.clone()
    expected[mask] -= lr * delta

    Sven(wrapper, lr=lr, k=k, rtol=rtol).step()
    p1 = wrapper.params.detach()
    assert ((p1 - expected).norm() / expected.norm()).item() < 1e-9
    assert torch.equal(p1[~mask], p0[~mask])  # only masked entries touched


# ----------------------------------------------------------------------
# (e) surgery transparency; elementwise mode unaffected
# ----------------------------------------------------------------------


def test_surgery_transparent_and_elementwise_unaffected():
    template = make_mlp([5, 7, 11, 3])
    x, y = make_data(8, 5, 3)
    with torch.no_grad():
        expected = copy.deepcopy(template)(x)

    wrapper = rows_wrapper(template, 0.3)
    assert torch.equal(wrapper.evaluate(x), expected)  # bitwise: same ops
    torch.manual_seed(5)
    wrapper.loss_and_grad((x, y))  # must leave the twins' state cleared
    assert torch.equal(wrapper.evaluate(x), expected)

    j_full = full_jacobian(template, x, y)
    fresh = SvenWrapper(
        copy.deepcopy(template), per_sample_mse, DEVICE,
        param_fraction=0.3, mask_mode="elementwise",
    )
    torch.manual_seed(21)
    fresh.loss_and_grad((x, y))
    assert (fresh.grads - j_full[:, fresh.param_mask]).abs().max() < 1e-12


# ----------------------------------------------------------------------
# (f) kappa != 2 combined with microbatching
# ----------------------------------------------------------------------


def test_rows_mask_kappa_microbatch():
    template = make_mlp([5, 7, 9, 3])
    x, y = make_data(8, 5, 3)
    j_full = full_jacobian(template, x, y, kappa=1.4, microbatch_size=2)

    wrapper = rows_wrapper(template, 0.3, kappa=1.4, microbatch_size=2)
    torch.manual_seed(6)
    wrapper.loss_and_grad((x, y))
    assert wrapper.grads.shape == (4, int(wrapper.param_mask.sum()))
    assert (wrapper.grads - j_full[:, wrapper.param_mask]).abs().max() < 1e-12
    assert torch.equal(wrapper.residuals, wrapper.losses.pow(0.7))


def test_rows_mask_jac_chunk_size_bitwise_equal():
    template = make_mlp([5, 7, 11, 3])
    x, y = make_data(8, 5, 3)
    plain = rows_wrapper(template, 0.3)
    torch.manual_seed(9)
    plain.loss_and_grad((x, y))
    chunked = rows_wrapper(template, 0.3, jac_chunk_size=3)
    torch.manual_seed(9)
    chunked.loss_and_grad((x, y))
    assert torch.equal(plain.param_mask, chunked.param_mask)
    assert torch.equal(plain.grads, chunked.grads)


# ----------------------------------------------------------------------
# (g) masks resampled every call; union covers (nearly) all parameters
# ----------------------------------------------------------------------


def test_rows_mask_resampled_and_covers_parameters():
    template = make_mlp([5, 7, 11, 3])
    x, y = make_data(4, 5, 3)
    wrapper = rows_wrapper(template, 0.25)
    torch.manual_seed(0)
    union = torch.zeros(wrapper.n_params, dtype=torch.bool)
    masks = []
    for _ in range(20):
        wrapper.loss_and_grad((x, y))
        masks.append(wrapper.param_mask.clone())
        union |= wrapper.param_mask
    assert any(not torch.equal(masks[0], m) for m in masks[1:])
    assert union.sum().item() > 0.9 * wrapper.n_params  # every parameter reachable


# ----------------------------------------------------------------------
# (h) achieved fraction
# ----------------------------------------------------------------------


def test_achieved_fraction_close_to_requested():
    template = make_mlp([8, 16, 16, 4])
    x, y = make_data(4, 8, 4)
    for fraction in (0.1, 0.25, 0.5):
        wrapper = rows_wrapper(template, fraction)
        torch.manual_seed(1)
        wrapper.loss_and_grad((x, y))
        achieved = wrapper.actual_param_fraction
        assert 0.5 * fraction <= achieved <= 1.5 * fraction
        assert achieved == wrapper.param_mask.sum().item() / wrapper.n_params


# ----------------------------------------------------------------------
# (i) unsupported modules / modes
# ----------------------------------------------------------------------


def test_unsupported_module_raises_rows_only():
    def build() -> nn.Sequential:
        torch.manual_seed(2)
        model = nn.Sequential(
            nn.Linear(5, 8), nn.BatchNorm1d(8), nn.Tanh(), nn.Linear(8, 3)
        ).to(DT)
        model.eval()  # running stats: keeps the reference per-sample-decoupled
        return model

    with pytest.raises(NotImplementedError, match="BatchNorm1d"):
        SvenWrapper(build(), per_sample_mse, DEVICE, param_fraction=0.3, mask_mode="rows")

    # elementwise mode handles the same model exactly
    x, y = make_data(6, 5, 3)
    j_full = full_jacobian(build(), x, y)
    wrapper = SvenWrapper(
        build(), per_sample_mse, DEVICE, param_fraction=0.3, mask_mode="elementwise"
    )
    torch.manual_seed(8)
    wrapper.loss_and_grad((x, y))
    assert (wrapper.grads - j_full[:, wrapper.param_mask]).abs().max() < 1e-12


def test_unknown_mask_mode_raises():
    template = make_mlp([5, 7, 3])
    with pytest.raises(ValueError, match="mask_mode"):
        SvenWrapper(template, per_sample_mse, DEVICE, mask_mode="banana")

"""Correctness tests for SvenWrapper structural block masking and jac_chunk_size.

Block masking (``mask_by_block=True``) must select whole tensors, compute the
Jacobian only w.r.t. them, and concatenate the per-tensor pieces in flat-vector
order so that ``Sven._apply_update``'s ``params[param_mask] += ...`` writes the
right entries.  All on CPU in float64.
"""

from __future__ import annotations

import copy

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


def full_jacobian(template: nn.Module, x, y) -> torch.Tensor:
    wrapper = SvenWrapper(copy.deepcopy(template), per_sample_mse, DEVICE)
    wrapper.loss_and_grad((x, y))
    return wrapper.grads.detach()


def simulate_block_selection(
    blocks: list[tuple[str, int, int]], n_params: int, fraction: float, seed: int
):
    """Replicate _make_param_mask_by_block's greedy pass, keeping VISIT order."""
    torch.manual_seed(seed)
    target = int(fraction * n_params)
    visited: list[tuple[str, int, int]] = []
    running = 0
    for i in torch.randperm(len(blocks)):
        name, n, start = blocks[i]
        if running + n <= target:
            visited.append((name, n, start))
            running += n
    return visited, running


# ----------------------------------------------------------------------
# (1) block-masked Jacobian columns == full Jacobian columns
# ----------------------------------------------------------------------


def test_block_mask_columns_match_full_jacobian():
    dims = [5, 7, 11, 13, 9, 3]
    template = make_mlp(dims)
    x, y = make_data(8, 5, 3)
    j_full = full_jacobian(template, x, y)

    wrapper = SvenWrapper(
        copy.deepcopy(template), per_sample_mse, DEVICE,
        param_fraction=0.4, mask_by_block=True,
    )

    # Engineer a seed whose greedy VISIT order of the selected blocks is not
    # ascending in flat start index: a bug that concatenates the per-tensor
    # Jacobian pieces in visit order instead of flat order must fail below.
    seed = None
    for candidate in range(100):
        visited, _ = simulate_block_selection(
            wrapper.param_names_counts_startIdx, wrapper.n_params, 0.4, candidate
        )
        starts = [start for _, _, start in visited]
        if len(starts) >= 2 and starts != sorted(starts):
            seed = candidate
            break
    assert seed is not None, "no out-of-order seed found"

    torch.manual_seed(seed)
    wrapper.loss_and_grad((x, y))
    visited, running = simulate_block_selection(
        wrapper.param_names_counts_startIdx, wrapper.n_params, 0.4, seed
    )
    assert sorted(visited, key=lambda blk: blk[2]) == wrapper._block_selection
    assert wrapper.actual_param_fraction == running / wrapper.n_params

    assert wrapper.grads.shape == (8, int(wrapper.param_mask.sum()))
    assert (wrapper.grads - j_full[:, wrapper.param_mask]).abs().max() < 1e-12


# ----------------------------------------------------------------------
# (2) Sven.step on the block-masked wrapper
# ----------------------------------------------------------------------


def test_block_mask_step_matches_manual_pinv_update():
    template = make_mlp([5, 7, 11, 13, 9, 3])
    x, y = make_data(8, 5, 3)
    lr, k, rtol = 0.1, 8, 1e-10

    wrapper = SvenWrapper(
        copy.deepcopy(template), per_sample_mse, DEVICE,
        param_fraction=0.4, mask_by_block=True,
    )
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
# (3) achieved fraction
# ----------------------------------------------------------------------


def test_achieved_fraction_within_bounds():
    dims = [8] + [16] * 12 + [4]  # many similar-size blocks
    template = make_mlp(dims)
    x, y = make_data(4, 8, 4)
    fraction = 0.35
    wrapper = SvenWrapper(
        template, per_sample_mse, DEVICE, param_fraction=fraction, mask_by_block=True
    )
    for seed in range(5):
        torch.manual_seed(seed)
        wrapper.loss_and_grad((x, y))
        achieved = wrapper.actual_param_fraction
        assert 0.5 * fraction <= achieved <= fraction
        assert achieved == wrapper.param_mask.sum().item() / wrapper.n_params


# ----------------------------------------------------------------------
# (4) jac_chunk_size is bitwise-neutral
# ----------------------------------------------------------------------


def test_jac_chunk_size_bitwise_equal():
    template = make_mlp([5, 7, 11, 3])
    x, y = make_data(8, 5, 3)

    plain = SvenWrapper(copy.deepcopy(template), per_sample_mse, DEVICE)
    plain.loss_and_grad((x, y))
    chunked = SvenWrapper(copy.deepcopy(template), per_sample_mse, DEVICE, jac_chunk_size=4)
    chunked.loss_and_grad((x, y))
    assert torch.equal(plain.grads, chunked.grads)

    # elementwise-masked path, same seeded mask
    masked = SvenWrapper(
        copy.deepcopy(template), per_sample_mse, DEVICE, param_fraction=0.3
    )
    torch.manual_seed(9)
    masked.loss_and_grad((x, y))
    masked_chunked = SvenWrapper(
        copy.deepcopy(template), per_sample_mse, DEVICE,
        param_fraction=0.3, jac_chunk_size=4,
    )
    torch.manual_seed(9)
    masked_chunked.loss_and_grad((x, y))
    assert torch.equal(masked.param_mask, masked_chunked.param_mask)
    assert torch.equal(masked.grads, masked_chunked.grads)


# ----------------------------------------------------------------------
# (5) legacy elementwise masking unchanged
# ----------------------------------------------------------------------


def test_elementwise_mask_columns_match_full_jacobian():
    template = make_mlp([5, 7, 11, 3])
    x, y = make_data(8, 5, 3)
    j_full = full_jacobian(template, x, y)

    wrapper = SvenWrapper(
        copy.deepcopy(template), per_sample_mse, DEVICE, param_fraction=0.3
    )
    torch.manual_seed(21)
    wrapper.loss_and_grad((x, y))
    assert wrapper.grads.shape == (8, int(wrapper.param_mask.sum()))
    assert (wrapper.grads - j_full[:, wrapper.param_mask]).abs().max() < 1e-12

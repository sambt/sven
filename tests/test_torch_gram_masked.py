"""Correctness tests for stochastic parameter masking on GramSvenWrapper.

Masked Gram accumulation (mask_mode "rows"/"tensor"/"elementwise") must
reproduce the reference masked pipeline — SvenWrapper with the SAME mask +
pinv + Sven._compute_delta — from the unchanged one-forward-one-backward
captures.  Both pipelines draw their masks through the same base-class
samplers, so seeding ``torch.manual_seed`` identically forces identical
selections (asserted before every delta comparison).  All on CPU in float64
so exactness assertions are meaningful.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from sven.nn import GramSvenWrapper, SvenWrapper
from sven.opt import Sven, SvenGram
from sven.opt.pinv import pinv

DT = torch.float64
DEVICE = "cpu"
DELTA_RTOL = 1e-9
GRAM_ATOL = 1e-10

MODES = ("rows", "tensor", "elementwise")


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


def make_data(
    b: int, d_in: int, d_out: int, seed: int = 1, dup_scales: tuple[float, ...] = ()
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random data; rows 1.. optionally near-duplicate row 0 to force tiny
    singular values so the rtol truncation rule actually fires."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, d_in, generator=g, dtype=DT)
    y = torch.randn(b, d_out, generator=g, dtype=DT)
    for i, scale in enumerate(dup_scales, start=1):
        x[i] = x[0] + scale * torch.randn(d_in, generator=g, dtype=DT)
        y[i] = y[0] + scale * torch.randn(d_out, generator=g, dtype=DT)
    return x, y


def full_jacobian(template: nn.Module, x, y, **wrapper_kwargs) -> torch.Tensor:
    wrapper = SvenWrapper(copy.deepcopy(template), per_sample_mse, DEVICE, **wrapper_kwargs)
    wrapper.loss_and_grad((x, y))
    return wrapper.grads.detach()


def masked_reference_delta(
    template: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
    rtol: float,
    fraction: float,
    mode: str,
    seed: int,
    **wrapper_kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """delta via the masked-Jacobian pipeline. Returns (delta, mask, J_A, kmax)."""
    wrapper = SvenWrapper(
        copy.deepcopy(template), per_sample_mse, DEVICE,
        param_fraction=fraction, mask_mode=mode, **wrapper_kwargs,
    )
    torch.manual_seed(seed)
    wrapper.loss_and_grad((x, y))
    VhT, S_inv, U_T = pinv(wrapper.grads, k=k, rtol=rtol, mode="torch")
    delta = Sven._compute_delta(U_T, S_inv, VhT, wrapper.residuals)
    return (
        delta.detach(),
        wrapper.param_mask.clone(),
        wrapper.grads.detach(),
        int(S_inv.shape[0]),
    )


def masked_gram_wrapper(
    template: nn.Module, fraction: float, mode: str, **wrapper_kwargs
) -> GramSvenWrapper:
    return GramSvenWrapper(
        copy.deepcopy(template), per_sample_mse, DEVICE, capture="hooks",
        param_fraction=fraction, mask_mode=mode, **wrapper_kwargs,
    )


def gram_step(
    wrapper: GramSvenWrapper,
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
    rtol: float,
    seed: int,
    lr: float = 1.0,
    opt_kwargs: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, SvenGram]:
    """Apply a real SvenGram step; returns (params before, params after, opt)."""
    p0 = wrapper.params.detach().clone()
    torch.manual_seed(seed)
    wrapper.loss_and_grad((x, y))
    opt = SvenGram(wrapper, lr=lr, k=k, rtol=rtol, **(opt_kwargs or {}))
    opt.step()
    return p0, wrapper.params.detach(), opt


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


# ----------------------------------------------------------------------
# (a) masked step delta == reference masked pinv delta, per mode
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "k,rtol,dup_scales",
    [(12, 1e-12, ()), (5, 1e-12, ()), (12, 1e-3, (1e-4, 1e-5))],
    ids=["k_full", "k_small", "rtol_trunc"],
)
@pytest.mark.parametrize("mode", MODES)
def test_masked_delta_matches_reference(mode, k, rtol, dup_scales):
    template = make_mlp([6, 24, 16, 4])  # unequal widths: per-layer rows differ
    x, y = make_data(12, 6, 4, dup_scales=dup_scales)
    frac, seed = 0.3, 3
    delta_ref, mask_ref, _, kmax_ref = masked_reference_delta(
        template, x, y, k, rtol, frac, mode, seed
    )
    wrapper = masked_gram_wrapper(template, frac, mode)
    p0, p1, opt = gram_step(
        wrapper, x, y, k, rtol, seed, opt_kwargs={"track_svd_info": True}
    )
    assert torch.equal(wrapper.param_mask, mask_ref)  # SAME selection, both pipelines
    assert 0 < int(mask_ref.sum()) < wrapper.n_params  # nontrivial mask
    delta = (p0 - p1)[mask_ref]  # params += -lr * delta with lr=1
    assert rel_err(delta, delta_ref) < DELTA_RTOL
    assert torch.equal(p1[~mask_ref], p0[~mask_ref])  # only masked entries touched
    assert opt.svd_info["num_nonzero_svs"] == [kmax_ref]
    if dup_scales:
        assert kmax_ref < min(12, k)  # the rtol rule actually truncated


# ----------------------------------------------------------------------
# (b) masked G identity: wrapper gram == J_A @ J_A^T
# ----------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_masked_gram_identity(mode):
    template = make_mlp([6, 24, 16, 4])
    x, y = make_data(12, 6, 4)
    j_full = full_jacobian(template, x, y)
    wrapper = masked_gram_wrapper(template, 0.3, mode)
    torch.manual_seed(4)
    wrapper.loss_and_grad((x, y))
    j_a = j_full[:, wrapper.param_mask]
    assert (wrapper.gram - j_a @ j_a.T).abs().max() < GRAM_ATOL


# ----------------------------------------------------------------------
# (c) rows + kappa != 2 + microbatching (cross terms nonzero)
# ----------------------------------------------------------------------


def test_rows_kappa_microbatch_matches_reference():
    template = make_mlp([6, 24, 24, 4])
    x, y = make_data(12, 6, 4)
    k, rtol, frac, seed = 6, 1e-12, 0.3, 6

    # In-group cross terms of the masked per-sample kernel must be nonzero:
    # diagonal-block-only pooling would be wrong.
    _, mask_ps, j_ps, _ = masked_reference_delta(
        template, x, y, 12, 1e-12, frac, "rows", seed, kappa=1.4
    )
    kernel = j_ps @ j_ps.T
    in_group_cross = torch.tensor([kernel[2 * i, 2 * i + 1] for i in range(6)])
    assert in_group_cross.abs().max() > 1e-6

    delta_ref, mask_ref, _, _ = masked_reference_delta(
        template, x, y, k, rtol, frac, "rows", seed, kappa=1.4, microbatch_size=2
    )
    assert torch.equal(mask_ps, mask_ref)  # sampling is microbatch-independent
    wrapper = masked_gram_wrapper(template, frac, "rows", kappa=1.4, microbatch_size=2)
    p0, p1, _ = gram_step(wrapper, x, y, k, rtol, seed)
    assert torch.equal(wrapper.param_mask, mask_ref)
    assert rel_err((p0 - p1)[mask_ref], delta_ref) < DELTA_RTOL


# ----------------------------------------------------------------------
# (d) CNN masking: rows exactness incl. forced channel chunking;
#     elementwise/tensor G identity with conv + frozen BatchNorm2d
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


def test_cnn_rows_matches_reference():
    torch.manual_seed(7)
    cnn = SmallCNN().to(DT)
    x = torch.randn(6, 1, 8, 8, dtype=DT)
    y = torch.randn(6, 2, dtype=DT)
    k, rtol, frac, seed = 6, 1e-10, 0.4, 1

    delta_ref, mask_ref, _, _ = masked_reference_delta(cnn, x, y, k, rtol, frac, "rows", seed)
    wrapper = masked_gram_wrapper(cnn, frac, "rows")
    p0, p1, _ = gram_step(wrapper, x, y, k, rtol, seed)
    assert torch.equal(wrapper.param_mask, mask_ref)
    assert rel_err((p0 - p1)[mask_ref], delta_ref) < DELTA_RTOL

    # tiny _CONV_BLOCK_ELEMS forces the pairwise channel-chunked contraction
    plain = masked_gram_wrapper(cnn, frac, "rows")
    torch.manual_seed(seed)
    plain.loss_and_grad((x, y))
    tiny = masked_gram_wrapper(cnn, frac, "rows")
    tiny._CONV_BLOCK_ELEMS = 100
    torch.manual_seed(seed)
    tiny.loss_and_grad((x, y))
    assert torch.equal(plain.param_mask, tiny.param_mask)
    assert (plain.gram - tiny.gram).abs().max() < GRAM_ATOL


class BNCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)              # bias=True
        self.bn = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.fc = nn.Linear(4 * 4 * 4, 2)

    def forward(self, x):
        h = torch.tanh(self.bn(self.conv1(x)))
        h = torch.tanh(self.conv2(h))
        return self.fc(h.flatten(1))


def test_cnn_elementwise_tensor_gram_identity():
    torch.manual_seed(9)
    cnn = BNCNN().to(DT)
    # Non-trivial running stats, model left in train mode: the wrapper's
    # freeze_norm_stats must do the eval switch itself.
    cnn.train()
    with torch.no_grad():
        for _ in range(3):
            cnn(torch.randn(8, 1, 4, 4, dtype=DT))
    x = torch.randn(8, 1, 4, 4, dtype=DT)
    y = torch.randn(8, 2, dtype=DT)
    ref_model = copy.deepcopy(cnn)
    ref_model.bn.eval()
    j_full = full_jacobian(ref_model, x, y)

    for mode in ("elementwise", "tensor"):
        wrapper = masked_gram_wrapper(cnn, 0.3, mode)
        torch.manual_seed(2)
        wrapper.loss_and_grad((x, y))
        j_a = j_full[:, wrapper.param_mask]
        assert (wrapper.gram - j_a @ j_a.T).abs().max() < GRAM_ATOL, mode

    # elementwise conv contraction is exact under forced batch chunking too
    plain = masked_gram_wrapper(cnn, 0.3, "elementwise")
    torch.manual_seed(2)
    plain.loss_and_grad((x, y))
    tiny = masked_gram_wrapper(cnn, 0.3, "elementwise")
    tiny._CONV_BLOCK_ELEMS = 50
    torch.manual_seed(2)
    tiny.loss_and_grad((x, y))
    assert torch.equal(plain.param_mask, tiny.param_mask)
    assert (plain.gram - tiny.gram).abs().max() < GRAM_ATOL


# ----------------------------------------------------------------------
# (e) update application and achieved fraction
# ----------------------------------------------------------------------


def test_masked_update_application_and_fraction():
    template = make_mlp([8, 16, 16, 4])
    x, y = make_data(8, 8, 4)
    for frac in (0.1, 0.5):
        wrapper = masked_gram_wrapper(template, frac, "rows")
        p0, p1, _ = gram_step(wrapper, x, y, 8, 1e-10, seed=1, lr=0.1)
        mask = wrapper.param_mask
        assert torch.equal(p1[~mask], p0[~mask])    # frozen entries bitwise untouched
        assert not torch.equal(p1[mask], p0[mask])  # masked entries actually moved
        achieved = wrapper.actual_param_fraction
        assert 0.5 * frac <= achieved <= 1.5 * frac
        assert achieved == mask.sum().item() / wrapper.n_params


# ----------------------------------------------------------------------
# (f) guards; param_fraction=1.0 ignores mask_mode
# ----------------------------------------------------------------------


def test_masked_guards_raise():
    template = make_mlp([6, 16, 4])

    # param_fraction < 1 without an explicit mask_mode
    with pytest.raises(ValueError, match="mask_mode"):
        GramSvenWrapper(
            copy.deepcopy(template), per_sample_mse, DEVICE, param_fraction=0.5
        )

    # rows masking rejects non-Linear/Conv2d parameter holders by name
    bn_model = nn.Sequential(nn.Linear(6, 8), nn.BatchNorm1d(8), nn.Linear(8, 4)).to(DT)
    with pytest.raises(NotImplementedError, match="BatchNorm1d"):
        GramSvenWrapper(
            bn_model, per_sample_mse, DEVICE, param_fraction=0.5, mask_mode="rows"
        )

    # rows masking rejects grouped Conv2d (cannot split grouped filters)
    grouped = nn.Sequential(
        nn.Conv2d(4, 4, 3, padding=1, groups=2), nn.Flatten(), nn.Linear(4 * 4 * 4, 2)
    ).to(DT)
    with pytest.raises(NotImplementedError, match="groups"):
        GramSvenWrapper(
            grouped, per_sample_mse, DEVICE, param_fraction=0.5, mask_mode="rows"
        )


def test_full_fraction_ignores_mask_mode():
    template = make_mlp([6, 16, 4])
    x, y = make_data(8, 6, 4)
    plain = GramSvenWrapper(copy.deepcopy(template), per_sample_mse, DEVICE)
    masked = GramSvenWrapper(
        copy.deepcopy(template), per_sample_mse, DEVICE,
        param_fraction=1.0, mask_mode="rows",
    )
    assert masked.mask_mode is None
    for wrapper in (plain, masked):
        torch.manual_seed(0)
        wrapper.loss_and_grad((x, y))
        SvenGram(wrapper, lr=0.5, k=8, rtol=1e-8).step()
    assert masked.param_mask is None
    assert torch.equal(masked.params.detach(), plain.params.detach())  # bitwise


# ----------------------------------------------------------------------
# (g) masks resampled every call
# ----------------------------------------------------------------------


def test_masks_resampled_each_call():
    template = make_mlp([6, 16, 16, 4])
    x, y = make_data(4, 6, 4)
    for mode in MODES:
        wrapper = masked_gram_wrapper(template, 0.25, mode)
        torch.manual_seed(0)
        masks = []
        for _ in range(6):
            wrapper.loss_and_grad((x, y))
            masks.append(wrapper.param_mask.clone())
        assert any(not torch.equal(masks[0], m) for m in masks[1:]), mode

"""Correctness tests for GramSvenWrapper + SvenGram.

Every case checks the Gram/kernel-trick pipeline against the exact reference
pipeline (SvenWrapper.loss_and_grad -> pinv -> Sven._compute_delta).  All on
CPU in float64 so exactness assertions are meaningful.
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


def reference_delta(
    template: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
    rtol: float,
    **wrapper_kwargs,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """delta via the real Jacobian pipeline. Returns (delta, J, kmax)."""
    wrapper = SvenWrapper(copy.deepcopy(template), per_sample_mse, DEVICE, **wrapper_kwargs)
    wrapper.loss_and_grad((x, y))
    VhT, S_inv, U_T = pinv(wrapper.grads, k=k, rtol=rtol, mode="torch")
    delta = Sven._compute_delta(U_T, S_inv, VhT, wrapper.residuals)
    return delta.detach(), wrapper.grads.detach(), int(S_inv.shape[0])


def gram_step_delta(
    template: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
    rtol: float,
    capture: str = "hooks",
    opt_kwargs: dict | None = None,
    **wrapper_kwargs,
) -> tuple[torch.Tensor, GramSvenWrapper, SvenGram]:
    """Applied update recovered from a real SvenGram step at lr=1."""
    wrapper = GramSvenWrapper(
        copy.deepcopy(template), per_sample_mse, DEVICE, capture=capture, **wrapper_kwargs
    )
    p0 = wrapper.params.detach().clone()
    wrapper.loss_and_grad((x, y))
    opt = SvenGram(wrapper, lr=1.0, k=k, rtol=rtol, **(opt_kwargs or {}))
    opt.step()
    delta = p0 - wrapper.params.detach()  # params += -lr * delta with lr=1
    return delta, wrapper, opt


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


# ----------------------------------------------------------------------
# (1) hooks-mode step delta == reference pinv delta
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "k,rtol,dup_scales",
    [(12, 1e-12, ()), (5, 1e-12, ()), (12, 1e-3, (1e-4, 1e-5))],
    ids=["k_full", "k_small", "rtol_trunc"],
)
def test_hooks_delta_matches_reference(k, rtol, dup_scales):
    template = make_mlp([6, 24, 24, 4])
    x, y = make_data(12, 6, 4, dup_scales=dup_scales)
    delta_ref, _, kmax_ref = reference_delta(template, x, y, k, rtol)
    delta, wrapper, opt = gram_step_delta(
        template, x, y, k, rtol, opt_kwargs={"track_svd_info": True}
    )
    assert rel_err(delta, delta_ref) < DELTA_RTOL
    assert opt.svd_info["num_nonzero_svs"] == [kmax_ref]
    if dup_scales:
        assert kmax_ref < min(12, k)  # the rtol rule actually truncated
    # wrapper contract: step() consumed and deleted the gram/residuals/losses
    assert wrapper.grads is None
    assert not hasattr(wrapper, "gram")


def test_gram_wrapper_attributes():
    template = make_mlp([6, 16, 4])
    x, y = make_data(8, 6, 4)
    wrapper = GramSvenWrapper(template, per_sample_mse, DEVICE, microbatch_size=2)
    wrapper.loss_and_grad((x, y))
    assert wrapper.losses.shape == (8,)      # raw per-sample losses
    assert wrapper.residuals.shape == (4,)   # microbatch rows
    assert wrapper.gram.shape == (4, 4)
    assert wrapper.gram.dtype == torch.float64
    assert wrapper.grads is None


# ----------------------------------------------------------------------
# (2) kappa != 2 and microbatching
# ----------------------------------------------------------------------


def test_hooks_delta_kappa():
    template = make_mlp([6, 24, 24, 4])
    x, y = make_data(12, 6, 4, dup_scales=(1e-4,))
    delta_ref, _, _ = reference_delta(template, x, y, 12, 1e-3, kappa=1.4)
    delta, _, _ = gram_step_delta(template, x, y, 12, 1e-3, kappa=1.4)
    assert rel_err(delta, delta_ref) < DELTA_RTOL


def test_hooks_delta_microbatch():
    template = make_mlp([6, 24, 24, 4])
    x, y = make_data(12, 6, 4)

    # The test is only meaningful if cross-sample kernel terms within a
    # microbatch group are nonzero: diagonal-block-only pooling must be wrong.
    _, j_per_sample, _ = reference_delta(template, x, y, 12, 1e-12)
    kernel = j_per_sample @ j_per_sample.T
    in_group_cross = torch.tensor([kernel[2 * i, 2 * i + 1] for i in range(6)])
    assert in_group_cross.abs().max() > 1e-6

    delta_ref, j_grouped, _ = reference_delta(template, x, y, 6, 1e-12, microbatch_size=2)
    delta, wrapper, _ = gram_step_delta(template, x, y, 6, 1e-12, microbatch_size=2)
    assert rel_err(delta, delta_ref) < DELTA_RTOL

    # combined kappa + microbatch
    delta_ref2, _, _ = reference_delta(
        template, x, y, 6, 1e-12, microbatch_size=2, kappa=1.4
    )
    delta2, _, _ = gram_step_delta(template, x, y, 6, 1e-12, microbatch_size=2, kappa=1.4)
    assert rel_err(delta2, delta_ref2) < DELTA_RTOL


# ----------------------------------------------------------------------
# (3) chunked G == hooks G == J @ J^T
# ----------------------------------------------------------------------


def test_gram_matrix_hooks_chunked_jacobian_agree():
    template = make_mlp([6, 24, 24, 4])
    x, y = make_data(12, 6, 4)
    _, j_full, _ = reference_delta(template, x, y, 12, 1e-12)
    gram_direct = j_full @ j_full.T

    hooks = GramSvenWrapper(copy.deepcopy(template), per_sample_mse, DEVICE, capture="hooks")
    hooks.loss_and_grad((x, y))
    # chunk_numel=200 forces several groups AND batch-chunked jacrev for the
    # 24x24 = 576-element hidden weights
    chunked = GramSvenWrapper(
        copy.deepcopy(template), per_sample_mse, DEVICE, capture="chunked", chunk_numel=200
    )
    chunked.loss_and_grad((x, y))

    assert (hooks.gram - gram_direct).abs().max() < GRAM_ATOL
    assert (chunked.gram - gram_direct).abs().max() < GRAM_ATOL
    assert (hooks.gram - chunked.gram).abs().max() < GRAM_ATOL


# ----------------------------------------------------------------------
# (4) CNN with conv (with/without bias) + frozen BatchNorm2d
# ----------------------------------------------------------------------


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)              # bias=True
        self.bn = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(3, 4, 3, padding=1, bias=False)  # bias=False
        self.fc = nn.Linear(4 * 4 * 4, 2)

    def forward(self, x):
        h = torch.tanh(self.bn(self.conv1(x)))
        h = torch.tanh(self.conv2(h))
        return self.fc(h.flatten(1))


def make_cnn_and_data() -> tuple[SmallCNN, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    cnn = SmallCNN().to(DT)
    x = torch.randn(8, 1, 4, 4, dtype=DT)
    y = torch.randn(8, 2, dtype=DT)
    # Non-trivial running stats, model left in train mode: the wrapper's
    # freeze_norm_stats must do the eval switch itself.
    cnn.train()
    with torch.no_grad():
        for _ in range(3):
            cnn(torch.randn(8, 1, 4, 4, dtype=DT))
    return cnn, x, y


def test_cnn_hooks_chunked_match_reference():
    cnn, x, y = make_cnn_and_data()
    # Reference pipeline sees the frozen-stats model explicitly
    ref_model = copy.deepcopy(cnn)
    ref_model.bn.eval()
    delta_ref, j_full, _ = reference_delta(ref_model, x, y, 8, 1e-10)
    gram_direct = j_full @ j_full.T

    for capture, kwargs in [
        ("hooks", {}),
        ("chunked", {"chunk_numel": 64}),
    ]:
        delta, wrapper, _ = gram_step_delta(cnn, x, y, 8, 1e-10, capture=capture, **kwargs)
        assert rel_err(delta, delta_ref) < DELTA_RTOL, capture
        assert wrapper.model.bn.training  # freeze restored after every pass

    hooks = GramSvenWrapper(copy.deepcopy(cnn), per_sample_mse, DEVICE, capture="hooks")
    hooks.loss_and_grad((x, y))
    assert (hooks.gram - gram_direct).abs().max() < GRAM_ATOL


def test_cnn_conv_batch_chunking_matches():
    """Tiny _CONV_BLOCK_ELEMS forces the pairwise batch-chunked conv path."""
    cnn, x, y = make_cnn_and_data()
    plain = GramSvenWrapper(copy.deepcopy(cnn), per_sample_mse, DEVICE, capture="hooks")
    plain.loss_and_grad((x, y))
    tiny = GramSvenWrapper(copy.deepcopy(cnn), per_sample_mse, DEVICE, capture="hooks")
    tiny._CONV_BLOCK_ELEMS = 100  # conv per-sample grad blocks: 1-2 rows at a time
    tiny.loss_and_grad((x, y))
    assert (plain.gram - tiny.gram).abs().max() < GRAM_ATOL


# ----------------------------------------------------------------------
# (5) batch-stat coupling: chunked stays exact, hooks guards raise
# ----------------------------------------------------------------------


class FuncBatchNorm2d(nn.Module):
    """Train-mode batch statistics computed functionally (jacrev-friendly),
    coupling every sample's loss to the whole batch."""

    def __init__(self, c: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c, dtype=DT))
        self.bias = nn.Parameter(torch.zeros(c, dtype=DT))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        return xhat * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class BatchCoupledCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, 3, padding=1)
        self.bn = FuncBatchNorm2d(3)
        self.fc = nn.Linear(3 * 4 * 4, 2)

    def forward(self, x):
        h = torch.tanh(self.bn(self.conv(x)))
        return self.fc(h.flatten(1))


def test_chunked_exact_with_batch_coupling():
    torch.manual_seed(5)
    net = BatchCoupledCNN().to(DT)
    x = torch.randn(8, 1, 4, 4, dtype=DT)
    y = torch.randn(8, 2, dtype=DT)
    delta_ref, j_full, _ = reference_delta(net, x, y, 8, 1e-10)
    delta, wrapper, _ = gram_step_delta(
        net, x, y, 8, 1e-10, capture="chunked", chunk_numel=32
    )
    assert rel_err(delta, delta_ref) < DELTA_RTOL
    chunked = GramSvenWrapper(
        copy.deepcopy(net), per_sample_mse, DEVICE, capture="chunked", chunk_numel=32
    )
    chunked.loss_and_grad((x, y))
    assert (chunked.gram - j_full @ j_full.T).abs().max() < GRAM_ATOL


def test_hooks_guards_raise():
    x = torch.randn(6, 4, dtype=DT)
    y = torch.randn(6, 2, dtype=DT)

    # train-mode BatchNorm with batch stats, freeze declined -> ValueError
    bn_model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Linear(4, 2)).to(DT)
    bn_model.train()
    hooks = GramSvenWrapper(
        bn_model, per_sample_mse, DEVICE, capture="hooks", freeze_norm_stats=False
    )
    with pytest.raises(ValueError, match="batch statistics"):
        hooks.loss_and_grad((x, y))

    # unsupported parameter-holding module -> NotImplementedError naming it
    torch.manual_seed(5)
    net = BatchCoupledCNN().to(DT)
    hooks = GramSvenWrapper(net, per_sample_mse, DEVICE, capture="hooks")
    with pytest.raises(NotImplementedError, match="FuncBatchNorm2d"):
        hooks.loss_and_grad((torch.randn(6, 1, 4, 4, dtype=DT), y))

    # train-mode dropout -> ValueError in BOTH capture modes
    for capture in ("hooks", "chunked"):
        drop_model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5), nn.Linear(4, 2)).to(DT)
        drop_model.train()
        wrapper = GramSvenWrapper(drop_model, per_sample_mse, DEVICE, capture=capture)
        with pytest.raises(ValueError, match="dropout"):
            wrapper.loss_and_grad((x, y))

    # non-zero padding_mode / string padding -> NotImplementedError (unfold
    # zero-pads, so the hooks contraction would be silently wrong)
    x_img = torch.randn(6, 1, 4, 4, dtype=DT)
    for conv_kwargs in ({"padding_mode": "circular", "padding": 1}, {"padding": "same"}):
        pad_model = nn.Sequential(
            nn.Conv2d(1, 2, 3, **conv_kwargs), nn.Flatten(), nn.Linear(2 * 4 * 4, 2)
        ).to(DT)
        hooks = GramSvenWrapper(pad_model, per_sample_mse, DEVICE, capture="hooks")
        with pytest.raises(NotImplementedError, match="padding"):
            hooks.loss_and_grad((x_img, y))
        chunked = GramSvenWrapper(
            copy.deepcopy(pad_model), per_sample_mse, DEVICE, capture="chunked"
        )
        chunked.loss_and_grad((x_img, y))
        ref = SvenWrapper(copy.deepcopy(pad_model), per_sample_mse, DEVICE)
        ref.loss_and_grad((x_img, y))
        j_ref = ref.grads
        assert (chunked.gram - j_ref @ j_ref.T).abs().max() < GRAM_ATOL


# ----------------------------------------------------------------------
# (6) end-to-end training smoke
# ----------------------------------------------------------------------


def test_svengram_trains_toy_1d():
    torch.manual_seed(11)
    net = make_mlp([1, 24, 24, 1], seed=11)
    x = torch.linspace(-1, 1, 32, dtype=DT).unsqueeze(1)
    y = torch.sin(3 * x)
    wrapper = GramSvenWrapper(net, per_sample_mse, DEVICE, capture="hooks")
    opt = SvenGram(wrapper, lr=0.5, k=32, rtol=1e-8)
    history = []
    for _ in range(20):
        losses, _ = wrapper.loss_and_grad((x, y))
        history.append(losses.mean().item())
        opt.step()
    assert history[-1] < 0.5 * history[0]


# ----------------------------------------------------------------------
# (7) optimizer option handling
# ----------------------------------------------------------------------


def test_rmsprop_post_runs_and_differs():
    template = make_mlp([6, 16, 4])
    x, y = make_data(8, 6, 4)
    delta_plain, _, _ = gram_step_delta(template, x, y, 8, 1e-10)
    delta_rms, _, _ = gram_step_delta(
        template, x, y, 8, 1e-10,
        opt_kwargs={"use_rmsprop": True, "rmsprop_post": True},
    )
    assert not torch.allclose(delta_plain, delta_rms)


def test_unsupported_options_raise():
    template = make_mlp([6, 16, 4])
    wrapper = GramSvenWrapper(template, per_sample_mse, DEVICE)
    with pytest.raises(NotImplementedError):
        SvenGram(wrapper, lr=0.1, k=8, rtol=1e-6, variable_k=True)
    with pytest.raises(NotImplementedError):
        SvenGram(wrapper, lr=0.1, k=8, rtol=1e-6, use_rmsprop=True, rmsprop_post=False)


def test_svengram_rejects_plain_wrapper():
    template = make_mlp([6, 16, 4])
    x, y = make_data(8, 6, 4)
    wrapper = SvenWrapper(template, per_sample_mse, DEVICE)
    wrapper.loss_and_grad((x, y))
    opt = SvenGram(wrapper, lr=0.1, k=8, rtol=1e-6)
    with pytest.raises(TypeError, match="GramSvenWrapper"):
        opt.step()

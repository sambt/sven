"""Correctness tests for stochastic parameter masking on the CHUNKED capture.

Masked chunked Gram accumulation (mask_mode "rows"/"tensor"/"elementwise")
must reproduce the reference masked pipeline — SvenWrapper carrying the SAME
flat mask + pinv + Sven._compute_delta — while skipping the jacrev of every
group without masked columns.  "tensor"/"elementwise" reuse the base-class
samplers, so identical seeds force identical masks across pipelines
(asserted); chunked "rows" means leading-axis slices per parameter tensor
(NOT the hooks-mode neuron-tied weight+bias pairing), so its reference masks
the full Jacobian columns with the chunked-sampled mask directly.  All on
CPU in float64 so exactness assertions are meaningful.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def full_jacobian_and_residuals(
    template: nn.Module, batch: tuple, loss_fn=per_sample_mse, **wrapper_kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    wrapper = SvenWrapper(copy.deepcopy(template), loss_fn, DEVICE, **wrapper_kwargs)
    wrapper.loss_and_grad(batch)
    return wrapper.grads.detach(), wrapper.residuals.detach()


def chunked_masked_wrapper(
    template: nn.Module, fraction: float, mode: str, loss_fn=per_sample_mse, **kwargs
) -> GramSvenWrapper:
    return GramSvenWrapper(
        copy.deepcopy(template), loss_fn, DEVICE, capture="chunked",
        param_fraction=fraction, mask_mode=mode, **kwargs,
    )


def gram_step(
    wrapper: GramSvenWrapper,
    batch: tuple,
    k: int,
    rtol: float,
    seed: int,
    lr: float = 1.0,
    opt_kwargs: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, SvenGram]:
    """Apply a real SvenGram step; returns (params before, params after, opt)."""
    p0 = wrapper.params.detach().clone()
    torch.manual_seed(seed)
    wrapper.loss_and_grad(batch)
    opt = SvenGram(wrapper, lr=lr, k=k, rtol=rtol, **(opt_kwargs or {}))
    opt.step()
    return p0, wrapper.params.detach(), opt


def injected_mask_reference_delta(
    j_full: torch.Tensor,
    residuals: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    rtol: float,
) -> tuple[torch.Tensor, int]:
    """delta from the full Jacobian restricted to the given flat mask."""
    j_a = j_full[:, mask]
    VhT, S_inv, U_T = pinv(j_a, k=k, rtol=rtol, mode="torch")
    return Sven._compute_delta(U_T, S_inv, VhT, residuals).detach(), int(S_inv.shape[0])


def sampled_mask_reference_delta(
    template: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
    rtol: float,
    fraction: float,
    mode: str,
    seed: int,
    **wrapper_kwargs,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """delta via SvenWrapper drawing the SAME mask through the shared sampler."""
    wrapper = SvenWrapper(
        copy.deepcopy(template), per_sample_mse, DEVICE,
        param_fraction=fraction, mask_mode=mode, **wrapper_kwargs,
    )
    torch.manual_seed(seed)
    wrapper.loss_and_grad((x, y))
    VhT, S_inv, U_T = pinv(wrapper.grads, k=k, rtol=rtol, mode="torch")
    delta = Sven._compute_delta(U_T, S_inv, VhT, wrapper.residuals)
    return delta.detach(), wrapper.param_mask.clone(), int(S_inv.shape[0])


def group_spans(wrapper: GramSvenWrapper) -> list[tuple[int, int]]:
    """Contiguous flat span [start, end) covered by each parameter group."""
    starts = {name: s for name, _, s in wrapper.param_names_counts_startIdx}
    return [
        (starts[g[0][0]], starts[g[-1][0]] + g[-1][2])
        for g in wrapper._param_groups()
    ]


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


# ----------------------------------------------------------------------
# Tiny transformer (cribbed from the verified chunked-capture smoke): manual
# causal multi-head attention, LayerNorm, tok+pos embeddings, untied head.
# No dropout — two-pass determinism is required by the wrapper.
# ----------------------------------------------------------------------

B, T, V, D, H, NL = 6, 8, 30, 16, 2, 2


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x):
        Bx, Tx, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        q = q.view(Bx, Tx, H, D // H).transpose(1, 2)
        k = k.view(Bx, Tx, H, D // H).transpose(1, 2)
        v = v.view(Bx, Tx, H, D // H).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D // H)
        mask = torch.triu(torch.ones(Tx, Tx, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf")).softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(Bx, Tx, D)
        x = x + self.proj(y)
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(T, D)
        self.blocks = nn.Sequential(*[Block() for _ in range(NL)])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, V, bias=False)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        return self.head(self.ln_f(self.blocks(x)))


def per_seq_ce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    Bx, Tx, Vx = logits.shape
    return F.cross_entropy(
        logits.reshape(-1, Vx), y.reshape(-1), reduction="none"
    ).view(Bx, Tx).mean(dim=1)


def make_gpt_and_data() -> tuple[TinyGPT, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    model = TinyGPT().to(DT)
    x = torch.randint(0, V, (B, T))
    y = torch.randint(0, V, (B, T))
    return model, x, y


# ----------------------------------------------------------------------
# (a) masked chunked step delta == reference masked pinv delta, per mode
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "k,rtol,dup_scales",
    [(5, 1e-12, ()), (12, 1e-3, (1e-4, 1e-5))],
    ids=["k_small", "rtol_trunc"],
)
@pytest.mark.parametrize("mode", MODES)
def test_masked_chunked_delta_matches_reference(mode, k, rtol, dup_scales):
    template = make_mlp([6, 24, 16, 4])  # unequal widths and tensor sizes
    x, y = make_data(12, 6, 4, dup_scales=dup_scales)
    frac, seed = 0.3, 3
    wrapper = chunked_masked_wrapper(template, frac, mode, chunk_numel=200)
    p0, p1, opt = gram_step(
        wrapper, (x, y), k, rtol, seed, opt_kwargs={"track_svd_info": True}
    )
    mask = wrapper.param_mask
    assert 0 < int(mask.sum()) < wrapper.n_params  # nontrivial mask

    if mode == "rows":
        # chunked-rows = leading-axis slices, so no SvenWrapper sampler
        # matches: inject the chunked-sampled mask into the full Jacobian.
        j_full, residuals = full_jacobian_and_residuals(template, (x, y))
        delta_ref, kmax_ref = injected_mask_reference_delta(
            j_full, residuals, mask, k, rtol
        )
    else:
        delta_ref, mask_ref, kmax_ref = sampled_mask_reference_delta(
            template, x, y, k, rtol, frac, mode, seed
        )
        assert torch.equal(mask, mask_ref)  # shared sampler, same seed

    delta = (p0 - p1)[mask]  # params += -lr * delta with lr=1
    assert rel_err(delta, delta_ref) < DELTA_RTOL
    assert torch.equal(p1[~mask], p0[~mask])  # only masked entries touched
    assert opt.svd_info["num_nonzero_svs"] == [kmax_ref]
    if dup_scales:
        assert kmax_ref < min(12, k)  # the rtol rule actually truncated


# ----------------------------------------------------------------------
# (b) transformer exactness: rows and tensor masks vs the masked full
#     Jacobian, several groups, at least one skipped, emb + LN in the mask
# ----------------------------------------------------------------------


def test_transformer_masked_chunked_matches_reference():
    model, x, y = make_gpt_and_data()
    j_full, residuals = full_jacobian_and_residuals(model, (x, y), loss_fn=per_seq_ce)

    for mode, frac, seed in [("rows", 0.35, 3), ("tensor", 0.3, 1)]:
        wrapper = chunked_masked_wrapper(
            model, frac, mode, loss_fn=per_seq_ce, chunk_numel=1024
        )
        spans = group_spans(wrapper)
        assert len(spans) >= 4  # chunk_numel actually split the model
        p0, p1, _ = gram_step(wrapper, (x, y), B, 1e-10, seed)
        mask = wrapper.param_mask
        starts = {name: s for name, _, s in wrapper.param_names_counts_startIdx}

        # embedding rows and LayerNorm entries participate in the mask
        tok_s, ln_s = starts["tok.weight"], starts["ln_f.weight"]
        emb_or_ln = mask[tok_s : tok_s + V * D].any() or mask[ln_s : ln_s + D].any()
        if mode == "rows":  # leading-axis: every tensor holds active slices
            assert mask[tok_s : tok_s + V * D].any()
            assert not mask[tok_s : tok_s + V * D].all()  # a strict vocab subset
            assert mask[ln_s : ln_s + D].any()
        else:
            assert emb_or_ln
            # tensor mask leaves whole groups empty (their jacrev is skipped)
            assert any(not mask[a:b].any() for a, b in spans)

        delta_ref, _ = injected_mask_reference_delta(j_full, residuals, mask, B, 1e-10)
        assert rel_err((p0 - p1)[mask], delta_ref) < DELTA_RTOL, mode
        assert torch.equal(p1[~mask], p0[~mask]), mode


def test_chunked_rows_mask_is_leading_axis_slices():
    """Chunked-rows semantics: whole leading-axis slices per tensor, each a
    contiguous flat run, ``max(1, round(f * dim0))`` of them (1-D tensors: a
    fraction of entries) — embeddings, LayerNorms and biases included."""
    model, _, _ = make_gpt_and_data()
    frac = 0.3
    wrapper = chunked_masked_wrapper(model, frac, "rows", loss_fn=per_seq_ce)
    torch.manual_seed(5)
    wrapper._sample_param_mask()
    mask = wrapper.param_mask
    starts = {name: s for name, _, s in wrapper.param_names_counts_startIdx}
    for name, shape, n in wrapper.param_shapes:
        view = mask[starts[name] : starts[name] + n].view(shape[0], -1)
        row_sel = view.any(dim=1)
        # whole slices only: relies on leading-axis slices being flat-contiguous
        assert torch.equal(view, row_sel[:, None].expand_as(view)), name
        assert int(row_sel.sum()) == max(1, int(round(frac * shape[0]))), name
    assert wrapper.actual_param_fraction == mask.sum().item() / wrapper.n_params


# ----------------------------------------------------------------------
# (c) masked G identity + group-local offset trap
# ----------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_masked_gram_identity_and_group_offsets(mode):
    template = make_mlp([6, 24, 16, 4])
    x, y = make_data(12, 6, 4)
    j_full, _ = full_jacobian_and_residuals(template, (x, y))
    # chunk_numel=150 -> very unequal groups: [w0]=144, [b0]=24, [w1]=384
    # (oversized, batch-chunked jacrev), [b1,w2,b2]=84
    wrapper = chunked_masked_wrapper(template, 0.3, mode, chunk_numel=150)
    spans = group_spans(wrapper)
    assert len(spans) >= 4
    torch.manual_seed(4)
    wrapper.loss_and_grad((x, y))
    mask = wrapper.param_mask
    j_a = j_full[:, mask]
    assert (wrapper.gram - j_a @ j_a.T).abs().max() < GRAM_ATOL
    # offset trap: masked columns straddle group boundaries, so confusing
    # global flat indices with group-local ones would break the identity
    nonempty = [(a, b) for a, b in spans if mask[a:b].any()]
    assert len(nonempty) >= 2
    assert any(a > 0 for a, _ in nonempty)  # a group needing local re-indexing


# ----------------------------------------------------------------------
# (d) kappa != 2 + microbatching under a chunked mask
# ----------------------------------------------------------------------


def test_masked_chunked_kappa_microbatch():
    template = make_mlp([6, 24, 24, 4])
    x, y = make_data(12, 6, 4)
    k, rtol, frac, seed = 6, 1e-12, 0.3, 6
    delta_ref, mask_ref, _ = sampled_mask_reference_delta(
        template, x, y, k, rtol, frac, "tensor", seed, kappa=1.4, microbatch_size=2
    )
    wrapper = chunked_masked_wrapper(
        template, frac, "tensor", kappa=1.4, microbatch_size=2, chunk_numel=200
    )
    p0 = wrapper.params.detach().clone()
    torch.manual_seed(seed)
    wrapper.loss_and_grad((x, y))
    assert torch.equal(wrapper.param_mask, mask_ref)
    assert wrapper.residuals.shape == (6,)  # microbatch rows, not samples
    assert wrapper.gram.shape == (6, 6)
    SvenGram(wrapper, lr=1.0, k=k, rtol=rtol).step()
    p1 = wrapper.params.detach()
    assert rel_err((p0 - p1)[mask_ref], delta_ref) < DELTA_RTOL


# ----------------------------------------------------------------------
# (e) guards and edge behavior
# ----------------------------------------------------------------------


def test_full_fraction_ignores_mask_mode_chunked():
    template = make_mlp([6, 16, 4])
    x, y = make_data(8, 6, 4)
    plain = GramSvenWrapper(
        copy.deepcopy(template), per_sample_mse, DEVICE,
        capture="chunked", chunk_numel=100,
    )
    masked = GramSvenWrapper(
        copy.deepcopy(template), per_sample_mse, DEVICE,
        capture="chunked", chunk_numel=100, param_fraction=1.0, mask_mode="rows",
    )
    assert masked.mask_mode is None
    for wrapper in (plain, masked):
        torch.manual_seed(0)
        wrapper.loss_and_grad((x, y))
        SvenGram(wrapper, lr=0.5, k=8, rtol=1e-8).step()
    assert masked.param_mask is None
    assert torch.equal(masked.params.detach(), plain.params.detach())  # bitwise


def test_chunked_requires_mask_mode():
    template = make_mlp([6, 16, 4])
    with pytest.raises(ValueError, match="mask_mode"):
        GramSvenWrapper(
            copy.deepcopy(template), per_sample_mse, DEVICE,
            capture="chunked", param_fraction=0.5,
        )


def test_empty_groups_skip_jacrev(monkeypatch):
    """Groups without masked columns must not run their jacrev at all."""
    template = make_mlp([6, 24, 16, 4])
    x, y = make_data(8, 6, 4)
    # frac 0.25 -> budget 159 of 636 params: the 384-param hidden weight can
    # never fit the tensor budget, so its singleton group is always empty.
    wrapper = chunked_masked_wrapper(template, 0.25, "tensor", chunk_numel=150)
    spans = group_spans(wrapper)

    calls = {"n": 0}
    real_jacrev = torch.func.jacrev

    def counting_jacrev(*args, **kwargs):
        calls["n"] += 1
        return real_jacrev(*args, **kwargs)

    monkeypatch.setattr(torch.func, "jacrev", counting_jacrev)
    torch.manual_seed(2)
    wrapper.loss_and_grad((x, y))
    mask = wrapper.param_mask
    n_nonempty = sum(bool(mask[a:b].any()) for a, b in spans)
    assert calls["n"] == n_nonempty  # one jacrev per nonempty group, no more
    assert n_nonempty < len(spans)  # at least one group was actually skipped
    assert wrapper.losses.shape == (8,)  # aux still populated from a live group


def test_masks_resampled_each_call():
    template = make_mlp([6, 16, 16, 4])
    x, y = make_data(4, 6, 4)
    for mode in MODES:
        wrapper = chunked_masked_wrapper(template, 0.25, mode)
        torch.manual_seed(0)
        masks = []
        for _ in range(4):
            wrapper.loss_and_grad((x, y))
            masks.append(wrapper.param_mask.clone())
        assert any(not torch.equal(masks[0], m) for m in masks[1:]), mode


def test_actual_fraction_and_update_application():
    template = make_mlp([8, 16, 16, 4])
    x, y = make_data(8, 8, 4)
    for frac in (0.1, 0.5):
        wrapper = chunked_masked_wrapper(template, frac, "rows")
        p0, p1, _ = gram_step(wrapper, (x, y), 8, 1e-10, seed=1, lr=0.1)
        mask = wrapper.param_mask
        assert torch.equal(p1[~mask], p0[~mask])    # frozen entries bitwise untouched
        assert not torch.equal(p1[mask], p0[mask])  # masked entries actually moved
        achieved = wrapper.actual_param_fraction
        assert 0.5 * frac <= achieved <= 1.5 * frac
        assert achieved == mask.sum().item() / wrapper.n_params

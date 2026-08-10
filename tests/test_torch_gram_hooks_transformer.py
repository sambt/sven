"""Correctness tests for the hooks Gram capture on transformer layers.

The hooks capture now covers the transformer layer set — Linear with lead
dims ``(B, *lead, d)``, per-position ``nn.LayerNorm`` and batched-input
``nn.Embedding`` — so the tiny GPT from comparisons/transformer runs on the
fast one-forward + one-backward path.  Every case checks hooks against the
exact references (chunked capture and/or the classic SvenWrapper + pinv +
Sven._compute_delta pipeline), plus the batched-embedding-input guard, the
untouched hooks guards (dropout / tied weights / batch-stats norms / masked
hooks), and a loose perf smoke proving the fast path is actually fast.  All
exactness checks on CPU in float64.
"""

from __future__ import annotations

import copy
import importlib.util
import os
import time

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

_TFM_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "comparisons", "transformer",
)


def _load(alias: str, fname: str):
    """Import a comparisons/transformer module under a collision-free alias
    (comparisons/ is not a package and has same-named top-level modules)."""
    spec = importlib.util.spec_from_file_location(alias, os.path.join(_TFM_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tfm_model = _load("tfm_model_hooks", "model.py")


def per_sample_mse(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, pred.dim()))
    return ((pred - y) ** 2).mean(dim=dims)


def make_tiny_gpt(seed: int = 0) -> torch.nn.Module:
    """fp64 tiny GPT at the bench architecture: V=50, T=12, d=16, L=2, H=2."""
    torch.manual_seed(seed)
    return tfm_model.TinyGPT(
        vocab=50, d_model=16, n_layer=2, n_head=2, seq_len=12
    ).to(DT)


def make_batch(b: int = 6, t: int = 12, v: int = 50, seed: int = 1, n_dup: int = 0):
    """Random token batch; the first ``n_dup`` rows duplicate row 0 exactly,
    forcing zero singular values so the rtol truncation rule actually fires."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, v, (b, t), generator=g)
    y = torch.randint(0, v, (b, t), generator=g)
    for i in range(1, 1 + n_dup):
        x[i], y[i] = x[0], y[0]
    return x, y


def reference_delta(
    template: nn.Module,
    batch: tuple,
    k: int,
    rtol: float,
    loss_fn=per_sample_mse,
    **wrapper_kwargs,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """delta via the real Jacobian pipeline. Returns (delta, J, kmax)."""
    wrapper = SvenWrapper(copy.deepcopy(template), loss_fn, DEVICE, **wrapper_kwargs)
    wrapper.loss_and_grad(batch)
    VhT, S_inv, U_T = pinv(wrapper.grads, k=k, rtol=rtol, mode="torch")
    delta = Sven._compute_delta(U_T, S_inv, VhT, wrapper.residuals)
    return delta.detach(), wrapper.grads.detach(), int(S_inv.shape[0])


def gram_wrapper(
    template: nn.Module, capture: str, loss_fn=per_sample_mse, **kwargs
) -> GramSvenWrapper:
    return GramSvenWrapper(
        copy.deepcopy(template), loss_fn, DEVICE, capture=capture, **kwargs
    )


def hooks_vs_chunked_gram(
    template: nn.Module, batch: tuple, loss_fn=per_sample_mse, **kwargs
) -> tuple[GramSvenWrapper, GramSvenWrapper]:
    """Run both captures and assert their Grams agree."""
    hooks = gram_wrapper(template, "hooks", loss_fn=loss_fn, **kwargs)
    hooks.loss_and_grad(batch)
    chunked = gram_wrapper(template, "chunked", loss_fn=loss_fn, **kwargs)
    chunked.loss_and_grad(batch)
    assert (hooks.gram - chunked.gram).abs().max() < GRAM_ATOL
    return hooks, chunked


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


# ----------------------------------------------------------------------
# (a) tiny GPT end-to-end: hooks G == chunked G == J J^T, step delta ==
#     classic pinv pipeline with k < M and the rtol rule firing
# ----------------------------------------------------------------------


def test_tiny_gpt_hooks_gram_matches_chunked_and_jacobian():
    template = make_tiny_gpt()
    batch = make_batch()
    _, j_full, _ = reference_delta(template, batch, 6, 1e-12, loss_fn=tfm_model.per_seq_ce)
    gram_direct = (j_full @ j_full.T).to(torch.float64)
    hooks, chunked = hooks_vs_chunked_gram(
        template, batch, loss_fn=tfm_model.per_seq_ce, chunk_numel=4096
    )
    assert (hooks.gram - gram_direct).abs().max() < GRAM_ATOL
    assert (chunked.gram - gram_direct).abs().max() < GRAM_ATOL


def test_tiny_gpt_hooks_delta_matches_reference():
    template = make_tiny_gpt()
    # Two duplicated sequences -> rank 4, so k=5 < M=6 AND the rtol rule
    # truncates inside the kept spectrum (kmax = 4 < k).
    batch = make_batch(n_dup=2)
    k, rtol = 5, 1e-3
    delta_ref, _, kmax_ref = reference_delta(
        template, batch, k, rtol, loss_fn=tfm_model.per_seq_ce
    )
    assert kmax_ref < k  # the rtol rule actually truncated

    wrapper = gram_wrapper(template, "hooks", loss_fn=tfm_model.per_seq_ce)
    p0 = wrapper.params.detach().clone()
    wrapper.loss_and_grad(batch)
    opt = SvenGram(wrapper, lr=1.0, k=k, rtol=rtol, track_svd_info=True)
    opt.step()
    delta = p0 - wrapper.params.detach()  # params += -lr * delta with lr=1
    assert rel_err(delta, delta_ref) < DELTA_RTOL
    assert opt.svd_info["num_nonzero_svs"] == [kmax_ref]


# ----------------------------------------------------------------------
# (b) N-D Linear alone: (B, *lead, d) MLP, bias-less layer included
# ----------------------------------------------------------------------


def make_nd_mlp(seed: int = 2) -> nn.Sequential:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(5, 14), nn.Tanh(),
        nn.Linear(14, 14, bias=False), nn.Tanh(),  # bias-less N-D Linear
        nn.Linear(14, 3),
    ).to(DT)


@pytest.mark.parametrize("lead", [(7,), (3, 4)], ids=["3d_input", "4d_input"])
def test_nd_linear_mlp_hooks_matches_chunked(lead):
    template = make_nd_mlp()
    g = torch.Generator().manual_seed(3)
    x = torch.randn(6, *lead, 5, generator=g, dtype=DT)
    y = torch.randn(6, *lead, 3, generator=g, dtype=DT)
    _, j_full, _ = reference_delta(template, (x, y), 6, 1e-12)
    hooks, _ = hooks_vs_chunked_gram(template, (x, y), chunk_numel=100)
    assert (hooks.gram - (j_full @ j_full.T).to(torch.float64)).abs().max() < GRAM_ATOL


def test_nd_linear_batch_chunking_matches():
    """Tiny _CONV_BLOCK_ELEMS forces the pairwise batch-chunked Linear path."""
    template = make_nd_mlp()
    g = torch.Generator().manual_seed(4)
    x = torch.randn(6, 7, 5, generator=g, dtype=DT)
    y = torch.randn(6, 7, 3, generator=g, dtype=DT)
    plain = gram_wrapper(template, "hooks")
    plain.loss_and_grad((x, y))
    tiny = gram_wrapper(template, "hooks")
    tiny._CONV_BLOCK_ELEMS = 250  # (B, d_out*d_in) blocks: 1-2 samples at a time
    tiny.loss_and_grad((x, y))
    assert (plain.gram - tiny.gram).abs().max() < GRAM_ATOL


# ----------------------------------------------------------------------
# (c) LayerNorm variants: affine +/- bias, affine=False, trailing rank > 1
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "ln_kwargs,shape_rank",
    [
        ({}, 1),
        ({"bias": False}, 1),
        ({"elementwise_affine": False}, 1),  # no params: must not crash
        ({}, 2),
    ],
    ids=["affine_bias", "affine_no_bias", "no_affine", "rank2_shape"],
)
def test_layernorm_hooks_matches_chunked(ln_kwargs, shape_rank):
    t = 7
    normalized_shape = [t, 10] if shape_rank == 2 else 10
    torch.manual_seed(5)
    template = nn.Sequential(
        nn.Linear(6, 10),
        nn.LayerNorm(normalized_shape, **ln_kwargs),
        nn.Tanh(),
        nn.Linear(10, 4),
    ).to(DT)
    g = torch.Generator().manual_seed(6)
    x = torch.randn(6, t, 6, generator=g, dtype=DT)
    y = torch.randn(6, t, 4, generator=g, dtype=DT)
    _, j_full, _ = reference_delta(template, (x, y), 6, 1e-12)
    hooks, _ = hooks_vs_chunked_gram(template, (x, y))
    assert (hooks.gram - (j_full @ j_full.T).to(torch.float64)).abs().max() < GRAM_ATOL


# ----------------------------------------------------------------------
# (d) Embedding: repeated tokens, batched positions, unbatched-input guard
# ----------------------------------------------------------------------

EMB_V, EMB_T, EMB_D = 13, 10, 7


class EmbNet(nn.Module):
    """Token embedding -> mean over positions -> linear readout."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(EMB_V, EMB_D)
        self.fc = nn.Linear(EMB_D, 3)

    def forward(self, idx):
        return self.fc(self.emb(idx).mean(dim=1))


class PosNet(nn.Module):
    """Token + positional embeddings; ``batched=False`` reproduces the
    broadcast 1-D arange bug the hooks capture must detect."""

    def __init__(self, batched: bool):
        super().__init__()
        self.tok = nn.Embedding(EMB_V, EMB_D)
        self.pos = nn.Embedding(EMB_T, EMB_D)
        self.fc = nn.Linear(EMB_D, 3)
        self.batched = batched

    def forward(self, idx):
        pos_idx = torch.arange(idx.shape[1], device=idx.device)
        if self.batched:
            pos_idx = pos_idx.expand(idx.shape)
        return self.fc((self.tok(idx) + self.pos(pos_idx)).mean(dim=1))


def make_emb_batch(seed: int = 7):
    """Token batch with repeats within AND across sequences (the scatter
    accumulation path), plus a regression target."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, EMB_V, (6, EMB_T), generator=g)
    x[0, :4] = x[0, 0]      # within-sequence repeats
    x[3, :3] = x[0, 0]      # the same token across sequences
    y = torch.randn(6, 3, generator=g, dtype=DT)
    return x, y


def test_embedding_hooks_matches_chunked_repeated_tokens():
    torch.manual_seed(8)
    template = EmbNet().to(DT)
    x, y = make_emb_batch()
    _, j_full, _ = reference_delta(template, (x, y), 6, 1e-12)
    hooks, _ = hooks_vs_chunked_gram(template, (x, y))
    assert (hooks.gram - (j_full @ j_full.T).to(torch.float64)).abs().max() < GRAM_ATOL

    # scatter blocks batch-chunked: V*d = 91, budget 200 -> 2 samples per block
    tiny = gram_wrapper(template, "hooks")
    tiny._CONV_BLOCK_ELEMS = 200
    tiny.loss_and_grad((x, y))
    assert (hooks.gram - tiny.gram).abs().max() < GRAM_ATOL

    # even a single-sample block over budget: dense scatter refuses clearly
    over = gram_wrapper(template, "hooks")
    over._CONV_BLOCK_ELEMS = 50
    with pytest.raises(NotImplementedError, match="capture='chunked'"):
        over.loss_and_grad((x, y))


def test_positional_embedding_batched_matches_unbatched_raises():
    x, y = make_emb_batch()

    torch.manual_seed(9)
    hooks, _ = hooks_vs_chunked_gram(PosNet(batched=True).to(DT), (x, y))

    torch.manual_seed(9)
    unbatched = gram_wrapper(PosNet(batched=False).to(DT), "hooks")
    with pytest.raises(NotImplementedError, match=r"Embedding 'pos'.*batched"):
        unbatched.loss_and_grad((x, y))


def test_unbatched_embedding_raises_even_when_length_equals_batch():
    # The B == T trap: a broadcast arange of length B is indistinguishable by
    # leading dim from a batched input and used to pass the guard silently
    # wrong (hooks-vs-chunked |dG| ~ 8e-2).  1-D embedding inputs must raise.
    x, y = make_emb_batch()
    b = x.shape[0]
    torch.manual_seed(9)
    wrapper = gram_wrapper(PosNet(batched=False).to(DT), "hooks")
    with pytest.raises(NotImplementedError, match=r"Embedding 'pos'.*batched"):
        wrapper.loss_and_grad((x[:, :b], y))  # T == B: leading-dim check alone passes


def test_layernorm_over_batch_dim_raises():
    # LayerNorm whose normalized_shape spans the ENTIRE input normalizes over
    # the batch dim, coupling per-sample losses — silently wrong under hooks.
    b, d = 5, 6
    net = nn.Sequential(nn.Linear(4, d), nn.LayerNorm([b, d]), nn.Linear(d, 3)).to(DT)
    wrapper = GramSvenWrapper(
        net, lambda p, t: ((p - t) ** 2).mean(dim=-1), "cpu", capture="hooks"
    )
    with pytest.raises(NotImplementedError, match="entire input including the batch"):
        wrapper.loss_and_grad((torch.randn(b, 4, dtype=DT), torch.randn(b, 3, dtype=DT)))


# ----------------------------------------------------------------------
# (e) kappa != 2 + microbatching on the tiny GPT
# ----------------------------------------------------------------------


def test_tiny_gpt_hooks_kappa_microbatch():
    template = make_tiny_gpt()
    batch = make_batch()
    kw = {"kappa": 1.4, "microbatch_size": 2}
    delta_ref, j_grouped, _ = reference_delta(
        template, batch, 3, 1e-12, loss_fn=tfm_model.per_seq_ce, **kw
    )
    hooks, _ = hooks_vs_chunked_gram(template, batch, loss_fn=tfm_model.per_seq_ce, **kw)
    assert hooks.gram.shape == (3, 3)  # microbatch rows, not samples
    assert (hooks.gram - (j_grouped @ j_grouped.T).to(torch.float64)).abs().max() < GRAM_ATOL

    wrapper = gram_wrapper(template, "hooks", loss_fn=tfm_model.per_seq_ce, **kw)
    p0 = wrapper.params.detach().clone()
    wrapper.loss_and_grad(batch)
    SvenGram(wrapper, lr=1.0, k=3, rtol=1e-12).step()
    assert rel_err(p0 - wrapper.params.detach(), delta_ref) < DELTA_RTOL


# ----------------------------------------------------------------------
# (f) existing guards intact on the extended hooks capture
# ----------------------------------------------------------------------


def test_hooks_guards_intact_on_transformer():
    batch = make_batch()

    # train-mode dropout inside a block -> ValueError in the capture
    drop_gpt = make_tiny_gpt()
    drop_gpt.blocks[0].mlp.insert(2, nn.Dropout(0.5))
    with pytest.raises(ValueError, match="dropout"):
        gram_wrapper(drop_gpt, "hooks", loss_fn=tfm_model.per_seq_ce).loss_and_grad(batch)

    # tied tok/head weights -> NotImplementedError.  Tie on the wrapped model:
    # the flat-parameter tie at construction rebinds every parameter storage,
    # which would silently sever a pre-existing tie before the guard runs.
    tied = gram_wrapper(make_tiny_gpt(), "hooks", loss_fn=tfm_model.per_seq_ce)
    tied.model.head.weight = tied.model.tok.weight
    with pytest.raises(NotImplementedError, match="tied"):
        tied.loss_and_grad(batch)

    # batch-stats BatchNorm with freeze declined -> ValueError, as before
    bn_model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Linear(4, 2)).to(DT)
    bn_model.train()
    bn_wrapper = GramSvenWrapper(
        bn_model, per_sample_mse, DEVICE, capture="hooks", freeze_norm_stats=False
    )
    with pytest.raises(ValueError, match="batch statistics"):
        bn_wrapper.loss_and_grad((torch.randn(6, 4, dtype=DT), torch.randn(6, 2, dtype=DT)))


def test_masked_hooks_still_refuse_transformer():
    """param_fraction < 1 hooks stays Linear/Conv2d(/_NormBase)-only: the
    masked transformer is chunked-only, unchanged scope."""
    with pytest.raises(NotImplementedError, match="nn.Linear/nn.Conv2d"):
        gram_wrapper(  # rows-mode surgery guard fires at construction
            make_tiny_gpt(), "hooks", loss_fn=tfm_model.per_seq_ce,
            param_fraction=0.5, mask_mode="rows",
        )
    for mode in ("tensor", "elementwise"):
        wrapper = gram_wrapper(
            make_tiny_gpt(), "hooks", loss_fn=tfm_model.per_seq_ce,
            param_fraction=0.5, mask_mode=mode,
        )
        with pytest.raises(NotImplementedError, match="nn.Linear/nn.Conv2d"):
            wrapper.loss_and_grad(make_batch())


# ----------------------------------------------------------------------
# (g) perf smoke: the fast path is actually fast on the bench train tier
# ----------------------------------------------------------------------


def test_hooks_step_faster_than_chunked_on_train_tier():
    """One fp32 train-tier loss_and_grad + step per capture; hooks must be
    well under the chunked wall time (loose 0.5x bound, not a benchmark)."""
    g = torch.Generator().manual_seed(10)
    x = torch.randint(0, 256, (32, 64), generator=g)
    y = torch.randint(0, 256, (32, 64), generator=g)

    times: dict[str, float] = {}
    for capture, kwargs in [("chunked", {"chunk_numel": 2 ** 20}), ("hooks", {})]:
        wrapper = GramSvenWrapper(
            tfm_model.build_model("train", seed=0), tfm_model.per_seq_ce, DEVICE,
            capture=capture, **kwargs,
        )
        t0 = time.perf_counter()
        wrapper.loss_and_grad((x, y))
        SvenGram(wrapper, lr=0.1, k=8, rtol=1e-4).step()
        times[capture] = time.perf_counter() - t0
        assert torch.isfinite(wrapper.params).all()
    assert times["hooks"] < 0.5 * times["chunked"], times

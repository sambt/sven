"""Correctness tests for the transformer benchmark (comparisons/transformer).

The chunked Gram capture is exact for ANY architecture; these tests promote
the verified transformer smoke into the suite: on a tiny fp64 GPT (manual
causal multi-head attention, pre-LN, tok+pos embeddings, untied head) the
chunked-gram step delta must equal the classic SvenWrapper + pinv +
Sven._compute_delta pipeline, the MASKED hooks capture must refuse the
architecture (LayerNorm / Embedding cannot join a param_fraction < 1 Gram;
unmasked hooks support is covered in test_torch_gram_hooks_transformer.py),
and the byte-level data pipeline must be deterministic with a contiguous
disjoint train/val split.  All exactness checks on CPU in float64.
"""

from __future__ import annotations

import copy
import importlib.util
import os

import pytest
import torch

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


tfm_model = _load("tfm_model", "model.py")
tfm_data = _load("tfm_data", "data.py")

_HAS_SHARD = os.path.isfile(tfm_data.DEFAULT_SHARD)


def make_tiny_gpt(seed: int = 0) -> torch.nn.Module:
    """fp64 tiny GPT: V=50, T=16, d=24, H=2, L=2 — the verified smoke shapes."""
    torch.manual_seed(seed)
    return tfm_model.TinyGPT(
        vocab=50, d_model=24, n_layer=2, n_head=2, seq_len=16
    ).to(DT)


def make_batch(b: int = 6, t: int = 16, v: int = 50, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, v, (b, t), generator=g)
    y = torch.randint(0, v, (b, t), generator=g)
    return x, y


# ----------------------------------------------------------------------
# (1) chunked-gram step delta == classic SvenWrapper + pinv pipeline
# ----------------------------------------------------------------------


def test_chunked_gram_matches_classic_on_transformer():
    template = make_tiny_gpt()
    x, y = make_batch()
    k, rtol = 6, 1e-10

    ref = SvenWrapper(copy.deepcopy(template), tfm_model.per_seq_ce, DEVICE)
    ref.loss_and_grad((x, y))
    j = ref.grads.detach()
    VhT, S_inv, U_T = pinv(j, k=k, rtol=rtol, mode="torch")
    delta_ref = Sven._compute_delta(U_T, S_inv, VhT, ref.residuals).detach()

    # chunk_numel=4096 forces several parameter groups (P ~ 46k)
    wrapper = GramSvenWrapper(
        copy.deepcopy(template), tfm_model.per_seq_ce, DEVICE,
        capture="chunked", chunk_numel=4096,
    )
    assert len(wrapper._param_groups()) > 1
    p0 = wrapper.params.detach().clone()
    wrapper.loss_and_grad((x, y))
    assert (wrapper.losses - ref.losses).abs().max() < GRAM_ATOL
    assert (wrapper.gram - (j @ j.T).to(torch.float64)).abs().max() < GRAM_ATOL
    SvenGram(wrapper, lr=1.0, k=k, rtol=rtol).step()
    delta = p0 - wrapper.params.detach()  # params += -lr * delta with lr=1
    assert ((delta - delta_ref).norm() / delta_ref.norm()).item() < DELTA_RTOL


# ----------------------------------------------------------------------
# (2) masked hooks capture refuses the transformer (documents the boundary:
#     the masked transformer stays chunked-only)
# ----------------------------------------------------------------------


def test_masked_hooks_capture_raises_on_transformer():
    wrapper = GramSvenWrapper(
        make_tiny_gpt(), tfm_model.per_seq_ce, DEVICE, capture="hooks",
        param_fraction=0.5, mask_mode="tensor",
    )
    with pytest.raises(NotImplementedError):
        wrapper.loss_and_grad(make_batch())


# ----------------------------------------------------------------------
# (3) data pipeline: determinism, shifted targets, contiguous disjoint split
# ----------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_SHARD, reason="local OpenWebText shard not present")
def test_data_determinism_and_split():
    n_train, n_val, t = 200_000, 50_000, 16
    train, val = tfm_data.load_openwebtext_bytes(n_train, n_val)
    assert train.dtype == torch.uint8 and val.dtype == torch.uint8
    assert train.shape[0] == n_train and val.shape[0] == n_val

    # contiguous split of one stream => train/val byte ranges are disjoint
    stream = tfm_data.byte_stream(n_train + n_val)
    assert torch.equal(torch.cat([train, val]), stream)

    # same seed+step -> identical batches; another step -> different windows
    x1, y1 = tfm_data.train_batch(train, t, 4, seed=123, step=5)
    x2, y2 = tfm_data.train_batch(train, t, 4, seed=123, step=5)
    x3, _ = tfm_data.train_batch(train, t, 4, seed=123, step=6)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    assert not torch.equal(x1, x3)
    assert x1.shape == (4, t) and torch.equal(x1[:, 1:], y1[:, :-1])  # shifted targets

    # validation windows are fixed and shifted the same way
    xv1, yv1 = tfm_data.val_windows(val, t, 32)
    xv2, _ = tfm_data.val_windows(val, t, 32)
    assert torch.equal(xv1, xv2)
    assert xv1.shape == (32, t) and torch.equal(xv1[:, 1:], yv1[:, :-1])


# ----------------------------------------------------------------------
# (4) train-tier model: one adamw and one gram-chunked step run end-to-end
# ----------------------------------------------------------------------


def test_train_tier_end_to_end_steps():
    x, y = make_batch(b=8, t=64, v=256, seed=2)

    model = tfm_model.build_model("train", seed=0)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss = tfm_model.per_seq_ce(model(x), y).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    assert torch.isfinite(tfm_model.per_seq_ce(model(x), y).mean())

    wrapper = GramSvenWrapper(
        tfm_model.build_model("train", seed=0), tfm_model.per_seq_ce, DEVICE,
        capture="chunked", chunk_numel=2 ** 20,
    )
    p0 = wrapper.params.detach().clone()
    losses, _ = wrapper.loss_and_grad((x, y))
    SvenGram(wrapper, lr=0.3, k=8, rtol=1e-4).step()
    assert torch.isfinite(losses).all()
    assert torch.isfinite(wrapper.params).all()
    assert not torch.equal(wrapper.params.detach(), p0)

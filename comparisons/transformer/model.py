"""Minimal GPT for the transformer benchmark — cribbed from the verified smoke.

Manual causal multi-head attention (no sdpa / ``nn.MultiheadAttention``),
pre-LN blocks, GELU MLP, learned token+position embeddings, an untied head
and no dropout: every operation is plain differentiable algebra, so the
chunked Gram capture is exact on it, and the two-pass capture/update
determinism requirement holds trivially (dropout would violate it).

Two tiers share the architecture:

- ``"train"``:   V=256, T=64,  d=128, L=4, H=4 → ~0.87M params; B=32.  Used
  for the actual training comparison (M=32 rows, per-sequence mean CE).
- ``"profile"``: V=256, T=128, d=192, L=4, H=6 → ~1.90M params; B=64.  Used
  only for per-step cost probes (``mem_probe.py``), never full training.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TIERS: dict[str, dict] = {
    "train": dict(vocab=256, seq_len=64, d_model=128, n_layer=4, n_head=4, batch_size=32),
    "profile": dict(vocab=256, seq_len=128, d_model=192, n_layer=4, n_head=6, batch_size=64),
}


class Block(nn.Module):
    """Pre-LN transformer block with manual causal multi-head attention."""

    def __init__(self, d_model: int, n_head: int) -> None:
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.n_head
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=-1)
        q = q.view(b, t, h, d // h).transpose(1, 2)
        k = k.view(b, t, h, d // h).transpose(1, 2)
        v = v.view(b, t, h, d // h).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(d // h)
        att = att.masked_fill(causal_mask, float("-inf")).softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(b, t, d)
        x = x + self.proj(y)
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    """Byte-level GPT: tok+pos embeddings, pre-LN blocks, untied lm head."""

    def __init__(
        self, vocab: int, d_model: int, n_layer: int, n_head: int, seq_len: int
    ) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)  # untied
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1),
            persistent=False,
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(mod: nn.Module) -> None:
        """GPT-2-style init: N(0, 0.02) weights, zero biases."""
        if isinstance(mod, nn.Linear):
            nn.init.normal_(mod.weight, std=0.02)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.Embedding):
            nn.init.normal_(mod.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        t = idx.shape[1]
        # Batched position indices (identical semantics to the broadcast 1-D
        # arange): the hooks Gram capture needs every Embedding input to carry
        # the batch dim, or the broadcast add batch-sums its cotangent.
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device).expand(idx.shape))
        mask = self.causal_mask[:t, :t]
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln_f(x))


def build_model(tier: str, seed: int) -> TinyGPT:
    """Seeded tier model — identical init for every optimizer config."""
    cfg = TIERS[tier]
    torch.manual_seed(seed)
    return TinyGPT(
        cfg["vocab"], cfg["d_model"], cfg["n_layer"], cfg["n_head"], cfg["seq_len"]
    )


def per_seq_ce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-sequence mean cross-entropy over tokens, shape (B,) — the Sven rows."""
    b, t, v = logits.shape
    return (
        F.cross_entropy(logits.reshape(-1, v), y.reshape(-1), reduction="none")
        .view(b, t)
        .mean(dim=1)
    )

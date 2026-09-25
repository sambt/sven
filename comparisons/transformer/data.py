"""Byte-level OpenWebText for the transformer benchmark — local shard only.

Reads the first documents of the single verified local parquet shard
directly with pyarrow (NO network, NO ``datasets.load_dataset`` — it stalls
trying to fetch the missing shards, and the other cached blob is
incomplete), joins them with a separator, encodes to uint8 bytes and splits
train/val contiguously.  Training batches draw seeded random windows —
seeded per step, so every optimizer config sees the identical window
sequence regardless of its own RNG consumption; validation uses a fixed
strided set of windows.
"""

from __future__ import annotations

import os

import numpy as np
import torch

DEFAULT_SHARD = (
    "/Users/sambt/.cache/huggingface/hub/datasets--Skylion007--openwebtext/"
    "blobs/caed9f4b7053d7cd4d1a13ce9ec9224d84a3bba1f11579193562a7e31ebe656e"
)
SEPARATOR = b"\n\n"


def byte_stream(n_bytes: int, shard_path: str | None = None) -> torch.Tensor:
    """First ``n_bytes`` of the shard's separator-joined docs as a uint8 tensor.

    Docs are read in order from the start of the shard and reading stops as
    soon as enough bytes are collected — the full shard (~100k docs) is never
    loaded.
    """
    import pyarrow.parquet as pq

    path = shard_path or os.environ.get("OWT_SHARD", DEFAULT_SHARD)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"OpenWebText shard not found at '{path}'. Point OWT_SHARD at a "
            "parquet file with a 'text' column."
        )
    chunks: list[bytes] = []
    total = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["text"]):
        for text in batch.column("text").to_pylist():
            chunks.append(text.encode("utf-8") + SEPARATOR)
            total += len(chunks[-1])
        if total >= n_bytes:
            break
    if total < n_bytes:
        raise ValueError(f"shard exhausted at {total} bytes < requested {n_bytes}")
    stream = np.frombuffer(b"".join(chunks), dtype=np.uint8)[:n_bytes]
    return torch.from_numpy(stream.copy())


def load_openwebtext_bytes(
    n_train: int = 3_000_000, n_val: int = 500_000, shard_path: str | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contiguous ``(train, val)`` uint8 split of the leading byte stream."""
    stream = byte_stream(n_train + n_val, shard_path)
    return stream[:n_train].clone(), stream[n_train:].clone()


def train_batch(
    data: torch.Tensor, seq_len: int, batch_size: int, seed: int, step: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Seeded random windows for one training step: ``(x, y)`` of shape (B, T).

    The generator is seeded from ``seed + step`` (the epoch-seeded pattern of
    ``comparisons/train_one.py``), so the batch sequence is identical across
    configs and independent of global RNG state.
    """
    g = torch.Generator().manual_seed(seed + step)
    starts = torch.randint(0, data.shape[0] - seq_len - 1, (batch_size,), generator=g)
    win = data[starts[:, None] + torch.arange(seq_len + 1)].long()
    return win[:, :-1], win[:, 1:]


def val_windows(
    data: torch.Tensor, seq_len: int, n_windows: int = 256
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed strided evaluation windows: ``(x, y)`` of shape (n_windows, T)."""
    stride = max(1, (data.shape[0] - seq_len - 1) // n_windows)
    starts = torch.arange(n_windows) * stride
    win = data[starts[:, None] + torch.arange(seq_len + 1)].long()
    return win[:, :-1], win[:, 1:]

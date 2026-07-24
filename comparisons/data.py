"""CIFAR-10 loading for optimizer comparisons — no torchvision dependency.

Reads the standard ``cifar-10-batches-py`` pickled batches directly, flattens
images to 3072-dim vectors and standardises per-feature using statistics of the
selected training subset.  The training subset is a seeded, class-stratified
sample so every optimizer config sees the identical data.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import torch

DEFAULT_CIFAR_DIR = "/Users/sambt/iaifi/autoscidact/data/cifar-10-batches-py"


def _load_batch(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    return d[b"data"], np.asarray(d[b"labels"], dtype=np.int64)


def load_cifar10(
    n_train: int = 4096,
    seed: int = 0,
    data_dir: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(x_train, y_train, x_val, y_val)`` flattened and standardised.

    Args:
        n_train: Size of the class-stratified training subset.
        seed: Seed for the stratified subsample.
        data_dir: Directory holding ``data_batch_*`` / ``test_batch``; defaults
            to ``$CIFAR_DIR`` or the local copy under ``DEFAULT_CIFAR_DIR``.
    """
    root = data_dir or os.environ.get("CIFAR_DIR", DEFAULT_CIFAR_DIR)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"CIFAR-10 batches not found at '{root}'. Point CIFAR_DIR at a "
            "directory containing the extracted cifar-10-batches-py files."
        )

    # Keep uint8 until after subsampling: the fp32 conversion of the full 50k
    # train set is a ~600MB transient that pollutes per-run peak-RSS numbers.
    xs, ys = zip(*(_load_batch(os.path.join(root, f"data_batch_{i}")) for i in range(1, 6)))
    x_train_full = np.concatenate(xs)
    y_train_full = np.concatenate(ys)
    x_val, y_val = _load_batch(os.path.join(root, "test_batch"))
    x_val = x_val.astype(np.float32)

    # Class-stratified subsample: n_train/10 per class, fixed by seed
    rng = np.random.default_rng(seed)
    per_class = n_train // 10
    idx = np.concatenate(
        [
            rng.choice(np.flatnonzero(y_train_full == c), size=per_class, replace=False)
            for c in range(10)
        ]
    )
    rng.shuffle(idx)
    x_train = x_train_full[idx].astype(np.float32)
    y_train = y_train_full[idx]
    del x_train_full

    # Standardise with train-subset statistics
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-7
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    return (
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
        torch.from_numpy(x_val),
        torch.from_numpy(y_val),
    )

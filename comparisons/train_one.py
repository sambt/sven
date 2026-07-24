"""Train ONE optimizer config on CIFAR-10 with a ~1M-param MLP and dump metrics.

Run in a fresh subprocess per config (see ``run_comparison.py``) so that peak
RSS is attributable to the config.  Writes ``<out>/<config>.json`` with
per-epoch train/val loss+accuracy, per-step wall times (gradient/Jacobian phase
and optimizer-step phase separately), peak RSS and Sven diagnostics.

Usage:
    python train_one.py --config sven_gram --epochs 20 --n-train 4096 --out results
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_cifar10

BATCH_SIZE = 128
SEED = 7310
SVEN_KW = dict(lr=0.1, k=128, rtol=1e-4)

# name -> (kind, options)
# The fraction scan uses ELEMENTWISE masks (exact fractions; jac_chunk_size
# bounds the Jacobian peak).  Whole-tensor structural masking cannot express
# 10-80% here: fc1.weight holds ~91% of parameters, so every whole-tensor
# budget below 91% collapses to the same ~9% "everything but fc1.weight"
# selection — that one honest point is included as `sven_struct`.
CONFIGS: dict[str, tuple[str, dict]] = {
    "sven_classic": ("sven", dict(svd_mode="randomized_v3")),
    "sven_frac10": ("sven", dict(svd_mode="randomized_v3", param_fraction=0.10, elementwise=True)),
    "sven_frac25": ("sven", dict(svd_mode="randomized_v3", param_fraction=0.25, elementwise=True)),
    "sven_frac50": ("sven", dict(svd_mode="randomized_v3", param_fraction=0.50, elementwise=True)),
    "sven_frac80": ("sven", dict(svd_mode="randomized_v3", param_fraction=0.80, elementwise=True)),
    "sven_struct": ("sven", dict(svd_mode="randomized_v3", param_fraction=0.10)),
    "sven_gram": ("gram", dict()),
    "adam": ("torch", dict(cls="Adam", lr=1e-3)),
    "adamw": ("torch", dict(cls="AdamW", lr=1e-3, weight_decay=1e-2)),
    "sgd": ("torch", dict(cls="SGD", lr=0.05, momentum=0.9)),
    "lbfgs": ("lbfgs", dict(lr=0.5, max_iter=4, history_size=10)),
    "probe": ("probe", dict()),  # memory baseline: data + model + one forward
}


def build_model() -> nn.Module:
    torch.manual_seed(SEED)
    return nn.Sequential(
        nn.Linear(3072, 300), nn.ReLU(),
        nn.Linear(300, 300), nn.ReLU(),
        nn.Linear(300, 10),
    )


def per_sample_ce(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(pred, y, reduction="none")


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    losses, correct = [], 0
    for i in range(0, len(x), 2048):
        pred = model(x[i : i + 2048])
        losses.append(F.cross_entropy(pred, y[i : i + 2048], reduction="sum").item())
        correct += (pred.argmax(dim=1) == y[i : i + 2048]).sum().item()
    return sum(losses) / len(x), correct / len(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=4096)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    torch.set_num_threads(4)
    kind, opts = CONFIGS[args.config]

    x_train, y_train, x_val, y_val = load_cifar10(n_train=args.n_train, seed=SEED)
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())

    if kind == "probe":
        model(x_train[:BATCH_SIZE])
        result = {"config": args.config, "n_params": n_params,
                  "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"{args.config}.json"), "w") as f:
            json.dump(result, f)
        return

    # --- optimizer construction -------------------------------------------
    wrapper = None
    if kind == "sven":
        from sven.nn import SvenWrapper
        from sven.opt import Sven

        frac = opts.get("param_fraction", 1.0)
        elementwise = opts.get("elementwise", False)
        wrapper = SvenWrapper(
            model, per_sample_ce, "cpu",
            param_fraction=frac,
            mask_by_block=frac < 1.0 and not elementwise,
            jac_chunk_size=16 if elementwise else None,
        )
        opt = Sven(wrapper, **SVEN_KW, svd_mode=opts["svd_mode"], track_svd_info=True)
    elif kind == "gram":
        from sven.nn import GramSvenWrapper
        from sven.opt import SvenGram

        wrapper = GramSvenWrapper(model, per_sample_ce, "cpu", capture="hooks")
        opt = SvenGram(wrapper, **SVEN_KW, track_svd_info=True)
    elif kind == "torch":
        cls = getattr(torch.optim, opts.pop("cls"))
        opt = cls(model.parameters(), **opts)
    elif kind == "lbfgs":
        opt = torch.optim.LBFGS(model.parameters(), line_search_fn="strong_wolfe", **opts)

    # --- training loop ----------------------------------------------------
    n = len(x_train)
    steps_per_epoch = n // BATCH_SIZE
    t_grad_all, t_step_all = [], []
    epochs_log = []
    actual_fracs = []
    cum_opt_time = 0.0

    for epoch in range(args.epochs):
        # Same shuffling for every config
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(SEED + epoch))
        for s in range(steps_per_epoch):
            idx = perm[s * BATCH_SIZE : (s + 1) * BATCH_SIZE]
            xb, yb = x_train[idx], y_train[idx]

            if kind in ("sven", "gram"):
                t0 = time.perf_counter()
                wrapper.loss_and_grad((xb, yb))
                t1 = time.perf_counter()
                opt.step()
                t2 = time.perf_counter()
                if kind == "sven" and wrapper.mask_by_block and wrapper.param_mask is not None:
                    actual_fracs.append(wrapper.actual_param_fraction)
            elif kind == "torch":
                t0 = time.perf_counter()
                loss = F.cross_entropy(model(xb), yb)
                opt.zero_grad()
                loss.backward()
                t1 = time.perf_counter()
                opt.step()
                t2 = time.perf_counter()
            else:  # lbfgs: everything happens inside step(closure)
                def closure():
                    opt.zero_grad()
                    loss = F.cross_entropy(model(xb), yb)
                    loss.backward()
                    return loss

                t0 = t1 = time.perf_counter()
                opt.step(closure)
                t2 = time.perf_counter()

            t_grad_all.append(t1 - t0)
            t_step_all.append(t2 - t1)
            cum_opt_time += t2 - t0

        train_loss, train_acc = evaluate(model, x_train, y_train)
        val_loss, val_acc = evaluate(model, x_val, y_val)
        epochs_log.append(dict(
            epoch=epoch + 1, train_loss=train_loss, train_acc=train_acc,
            val_loss=val_loss, val_acc=val_acc, cum_opt_time=cum_opt_time,
        ))
        print(f"[{args.config}] epoch {epoch+1}/{args.epochs} "
              f"train {train_loss:.4f}/{train_acc:.3f} val {val_loss:.4f}/{val_acc:.3f} "
              f"opt-time {cum_opt_time:.1f}s", flush=True)

    result = {
        "config": args.config,
        "kind": kind,
        "n_params": n_params,
        "n_train": n,
        "batch_size": BATCH_SIZE,
        "steps_per_epoch": steps_per_epoch,
        "epochs": epochs_log,
        "t_grad": t_grad_all,
        "t_step": t_step_all,
        "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,  # bytes on macOS
        "actual_param_fraction": (sum(actual_fracs) / len(actual_fracs)) if actual_fracs else None,
        "num_nonzero_svs": (
            [int(v) for v in opt.svd_info["num_nonzero_svs"]]
            if kind in ("sven", "gram") else None
        ),
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.config}.json"), "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

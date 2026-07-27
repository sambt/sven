"""Train ONE optimizer config on byte-level OpenWebText with a ~0.9M-param GPT.

Run in a fresh subprocess per config (see ``run_comparison.py``) so that peak
RSS is attributable to the config.  Writes ``<out>/<config>.json`` with
periodic train/val cross-entropy (per-token nats and bits-per-byte),
per-step wall times (gradient/Gram phase and optimizer-step phase
separately), peak RSS and Sven diagnostics.

Usage:
    python train_one.py --config sven_gram_c20 --steps 300 --out results
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_openwebtext_bytes, train_batch, val_windows
from model import TIERS, build_model, per_seq_ce

SEED = 7310
TIER = "train"
N_TRAIN_BYTES = 3_000_000
N_VAL_BYTES = 500_000
EVAL_WINDOWS = 256
# lr picked by a 30-step {0.1, 0.3, 1.0} sanity scan on sven_gram_c20: 0.1
# was the only monotone decrease (val CE 5.60 -> 3.10 nats), 0.3 oscillated,
# 1.0 diverged (see README).  Shared by ALL Sven variants.  k=32 keeps every
# row of the M=32 per-sequence system; rtol matches the CIFAR settings.
SVEN_LR = 0.1
SVEN_KW = dict(k=32, rtol=1e-4)

# name -> (kind, options)
# The gram configs are the chunk_numel scan: identical updates (the Gram is
# exact at any chunk size), different capture memory/time.  The rows configs
# use the masked chunked capture (param_fraction + mask_mode="rows" with
# capture="chunked": leading-axis slices per tensor); on a checkout without
# that support their construction raises and this script prints a FAILED
# line and exits nonzero, which run_comparison.py reports without aborting
# the remaining configs.
CONFIGS: dict[str, tuple[str, dict]] = {
    "sven_classic": ("sven", dict(svd_mode="randomized_v3")),
    "sven_gram_c18": ("gram", dict(chunk_numel=2 ** 18)),
    "sven_gram_c20": ("gram", dict(chunk_numel=2 ** 20)),
    "sven_gram_c22": ("gram", dict(chunk_numel=2 ** 22)),
    "sven_gram_rows10": ("gram", dict(chunk_numel=2 ** 20, param_fraction=0.10, mask_mode="rows")),
    "sven_gram_rows25": ("gram", dict(chunk_numel=2 ** 20, param_fraction=0.25, mask_mode="rows")),
    "sven_gram_rows50": ("gram", dict(chunk_numel=2 ** 20, param_fraction=0.50, mask_mode="rows")),
    "adam": ("torch", dict(cls="Adam", lr=3e-4)),
    "adamw": ("torch", dict(cls="AdamW", lr=3e-4, weight_decay=0.01)),
    "sgd": ("torch", dict(cls="SGD", lr=0.1, momentum=0.9)),
    "probe": ("probe", dict()),  # memory baseline: data + model + one forward
}


def rss_bytes() -> int:
    """Peak RSS in BYTES: ru_maxrss is bytes on macOS but kilobytes on Linux."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru * 1024 if sys.platform.startswith("linux") else ru


def resolve_device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def sync(device: str) -> None:
    """Barrier before timing reads: CUDA launches are asynchronous."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def build_wrapper_and_opt(
    kind: str, opts: dict, model: torch.nn.Module, sven_lr: float,
    device: str = "cpu", track: bool = True,
):
    """Construct the (wrapper, optimizer) pair for one Sven/torch config."""
    if kind == "sven":
        from sven.nn import SvenWrapper
        from sven.opt import Sven

        wrapper = SvenWrapper(model, per_seq_ce, device)
        opt = Sven(
            wrapper, lr=sven_lr, **SVEN_KW,
            svd_mode=opts["svd_mode"], track_svd_info=track,
        )
    elif kind == "gram":
        from sven.nn import GramSvenWrapper
        from sven.opt import SvenGram

        gram_kw: dict = dict(capture="chunked", chunk_numel=opts["chunk_numel"])
        if "param_fraction" in opts:
            gram_kw.update(
                param_fraction=opts["param_fraction"], mask_mode=opts["mask_mode"]
            )
        wrapper = GramSvenWrapper(model, per_seq_ce, device, **gram_kw)
        opt = SvenGram(wrapper, lr=sven_lr, **SVEN_KW, track_svd_info=track)
    else:  # torch
        opts = dict(opts)
        wrapper = None
        model.to(device)
        opt = getattr(torch.optim, opts.pop("cls"))(model.parameters(), **opts)
    return wrapper, opt


@torch.no_grad()
def evaluate(model: torch.nn.Module, xw: torch.Tensor, yw: torch.Tensor) -> float:
    """Mean per-token cross-entropy (nats) over fixed windows, chunked."""
    total, ntok = 0.0, 0
    for i in range(0, xw.shape[0], 64):
        logits = model(xw[i : i + 64])
        total += F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), yw[i : i + 64].reshape(-1),
            reduction="sum",
        ).item()
        ntok += yw[i : i + 64].numel()
    return total / ntok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--sven-lr", type=float, default=SVEN_LR)
    ap.add_argument("--device", default="auto", help="cpu, cuda, cuda:N, or auto")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    torch.set_num_threads(4)
    device = resolve_device(args.device)
    kind, opts = CONFIGS[args.config]
    tier = TIERS[TIER]
    seq_len, batch_size = tier["seq_len"], tier["batch_size"]

    train_data, val_data = load_openwebtext_bytes(N_TRAIN_BYTES, N_VAL_BYTES)
    xt, yt = val_windows(train_data, seq_len, EVAL_WINDOWS)  # fixed train-subset eval
    xv, yv = val_windows(val_data, seq_len, EVAL_WINDOWS)
    xt, yt, xv, yv = (t.to(device) for t in (xt, yt, xv, yv))
    model = build_model(TIER, SEED)
    n_params = sum(p.numel() for p in model.parameters())

    if kind == "probe":
        model.to(device)
        xb, _ = train_batch(train_data, seq_len, batch_size, SEED, 0)
        model(xb.to(device))
        sync(device)
        result = {"config": args.config, "n_params": n_params, "device": device,
                  "peak_rss": rss_bytes(),
                  "peak_cuda": (torch.cuda.max_memory_allocated()
                                if device.startswith("cuda") else None)}
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"{args.config}.json"), "w") as f:
            json.dump(result, f)
        return

    try:
        wrapper, opt = build_wrapper_and_opt(kind, opts, model, args.sven_lr, device)
    except (NotImplementedError, ValueError) as err:
        print(f"[{args.config}] construction FAILED: {err}", flush=True)
        sys.exit(1)

    # --- training loop ----------------------------------------------------
    t_grad_all, t_step_all = [], []
    evals_log = []
    actual_fracs = []
    cum_opt_time = 0.0

    def log_eval(step: int) -> None:
        train_ce = evaluate(model, xt, yt)
        val_ce = evaluate(model, xv, yv)
        evals_log.append(dict(
            step=step, train_ce=train_ce, val_ce=val_ce,
            train_bpb=train_ce / math.log(2), val_bpb=val_ce / math.log(2),
            cum_opt_time=cum_opt_time,
        ))
        print(f"[{args.config}] step {step}/{args.steps} "
              f"train {train_ce:.4f} val {val_ce:.4f} ({val_ce / math.log(2):.3f} bpb) "
              f"opt-time {cum_opt_time:.1f}s", flush=True)

    log_eval(0)
    for s in range(args.steps):
        # Same seeded window sequence for every config
        xb, yb = train_batch(train_data, seq_len, batch_size, SEED, s)
        xb, yb = xb.to(device), yb.to(device)

        if kind in ("sven", "gram"):
            sync(device)
            t0 = time.perf_counter()
            wrapper.loss_and_grad((xb, yb))
            sync(device)
            t1 = time.perf_counter()
            opt.step()
            sync(device)
            t2 = time.perf_counter()
            if wrapper.param_mask is not None:
                actual_fracs.append(wrapper.actual_param_fraction)
        else:
            sync(device)
            t0 = time.perf_counter()
            loss = per_seq_ce(model(xb), yb).mean()
            opt.zero_grad()
            loss.backward()
            sync(device)
            t1 = time.perf_counter()
            opt.step()
            sync(device)
            t2 = time.perf_counter()

        t_grad_all.append(t1 - t0)
        t_step_all.append(t2 - t1)
        cum_opt_time += t2 - t0
        if (s + 1) % args.eval_every == 0:
            log_eval(s + 1)
    if args.steps % args.eval_every != 0:
        log_eval(args.steps)

    result = {
        "config": args.config,
        "kind": kind,
        "tier": TIER,
        "n_params": n_params,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "steps": args.steps,
        "device": device,
        "sven_lr": args.sven_lr if kind in ("sven", "gram") else None,
        "chunk_numel": opts.get("chunk_numel"),
        "evals": evals_log,
        "t_grad": t_grad_all,
        "t_step": t_step_all,
        "peak_rss": rss_bytes(),
        "peak_cuda": (torch.cuda.max_memory_allocated()
                      if device.startswith("cuda") else None),
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

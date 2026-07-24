"""Per-step peak-memory probe: one optimizer step per config, fresh process.

Peak RSS of a whole training run is dominated by the dataset-loading transient
and macOS malloc page retention, which drowns the optimizer working set (the
quantity of interest).  This probe isolates it: each config runs TWO optimizer
steps on synthetic data of the training shapes (no CIFAR load) in a fresh
subprocess, and reports ``ru_maxrss`` minus a model-only baseline.

Writes ``<out>/memory.json``; ``plot_comparison.py`` prefers it when present.

Usage:
    python mem_probe.py                # all configs -> results/memory.json
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def child(config: str) -> None:
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(4)
    torch.manual_seed(0)
    from train_one import BATCH_SIZE, CONFIGS, SVEN_KW, build_model, per_sample_ce

    model = build_model()
    x = torch.randn(BATCH_SIZE, 3072)
    y = torch.randint(0, 10, (BATCH_SIZE,))
    kind, opts = CONFIGS[config]

    if kind == "probe":
        model(x)
    else:
        opts = dict(opts)
        if kind == "sven":
            from sven.nn import SvenWrapper
            from sven.opt import Sven

            frac = opts.get("param_fraction", 1.0)
            wrapper = SvenWrapper(
                model, per_sample_ce, "cpu",
                param_fraction=frac,
                mask_by_block=frac < 1.0 and not opts.get("elementwise", False),
                jac_chunk_size=16 if opts.get("elementwise") else None,
            )
            opt = Sven(wrapper, **SVEN_KW, svd_mode=opts["svd_mode"])
        elif kind == "gram":
            from sven.nn import GramSvenWrapper
            from sven.opt import SvenGram

            wrapper = GramSvenWrapper(model, per_sample_ce, "cpu", capture="hooks")
            opt = SvenGram(wrapper, **SVEN_KW)
        elif kind == "torch":
            opt = getattr(torch.optim, opts.pop("cls"))(model.parameters(), **opts)
        else:
            opt = torch.optim.LBFGS(model.parameters(), line_search_fn="strong_wolfe", **opts)

        for _ in range(2):  # second step includes lazily-allocated optimizer state
            if kind in ("sven", "gram"):
                wrapper.loss_and_grad((x, y))
                opt.step()
            elif kind == "torch":
                loss = F.cross_entropy(model(x), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            else:
                def closure():
                    opt.zero_grad()
                    loss = F.cross_entropy(model(x), y)
                    loss.backward()
                    return loss
                opt.step(closure)

    print(json.dumps({"config": config,
                      "peak": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    sys.path.insert(0, HERE)
    from train_one import CONFIGS

    peaks: dict[str, int] = {}
    for cfg in CONFIGS:
        out = subprocess.run([sys.executable, __file__, "--child", cfg],
                             capture_output=True, text=True)
        if out.returncode != 0:
            print(f"{cfg}: FAILED\n{out.stderr[-500:]}")
            continue
        peaks[cfg] = json.loads(out.stdout.strip().splitlines()[-1])["peak"]
        print(f"{cfg:14s} {peaks[cfg]/1e6:8.1f} MB")

    base = peaks.get("probe", 0)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "memory.json"), "w") as f:
        json.dump({"baseline": base,
                   "per_step_delta": {k: v - base for k, v in peaks.items() if k != "probe"}},
                  f, indent=1)
    print("wrote", os.path.join(args.out, "memory.json"))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        sys.path.insert(0, HERE)
        child(sys.argv[2])
    else:
        main()

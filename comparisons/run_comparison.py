"""Run every optimizer config sequentially, each in a fresh subprocess.

Fresh subprocesses give attributable peak-RSS numbers and isolate failures.
Results land in ``<out>/<config>.json``; run ``plot_comparison.py`` afterwards.

Usage:
    python run_comparison.py                       # all configs, 20 epochs
    python run_comparison.py --configs sven_gram adam --epochs 5
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_ORDER = [
    "probe",
    "sven_gram",
    "adam",
    "adamw",
    "sgd",
    "lbfgs",
    "sven_struct",
    "sven_frac10",
    "sven_frac25",
    "sven_frac50",
    "sven_frac80",
    "sven_classic",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=DEFAULT_ORDER)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=4096)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    configs = list(args.configs)
    if "probe" not in configs:
        configs.insert(0, "probe")  # memory baseline is always needed

    for cfg in configs:
        t0 = time.time()
        print(f"=== {cfg} ===", flush=True)
        ret = subprocess.run(
            [sys.executable, os.path.join(HERE, "train_one.py"),
             "--config", cfg, "--epochs", str(args.epochs),
             "--n-train", str(args.n_train), "--out", args.out],
        )
        status = "ok" if ret.returncode == 0 else f"FAILED ({ret.returncode})"
        print(f"=== {cfg}: {status} in {time.time()-t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()

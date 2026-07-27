"""Run every transformer optimizer config sequentially, each in a fresh subprocess.

Fresh subprocesses give attributable peak-RSS numbers and isolate failures —
in particular the ``sven_gram_rows*`` configs, whose masked-chunked support
is being developed concurrently: if their construction raises, the config is
reported FAILED and the run carries on.

Results land in ``<out>/<config>.json``; run ``plot_comparison.py`` afterwards.

Usage:
    python run_comparison.py                              # all configs, 300 steps
    python run_comparison.py --configs sven_gram_c20 adamw --steps 30
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
    "sven_gram_c18",
    "sven_gram_c20",
    "sven_gram_c22",
    "sven_gram_rows10",
    "sven_gram_rows25",
    "sven_gram_rows50",
    "adam",
    "adamw",
    "sgd",
    "sven_classic",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=DEFAULT_ORDER)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=25)
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
             "--config", cfg, "--steps", str(args.steps),
             "--eval-every", str(args.eval_every), "--out", args.out],
        )
        status = "ok" if ret.returncode == 0 else f"FAILED ({ret.returncode})"
        print(f"=== {cfg}: {status} in {time.time()-t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()

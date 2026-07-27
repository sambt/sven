"""Per-step peak-memory probe: two optimizer steps per config, fresh process.

Peak RSS of a whole training run is dominated by the data-loading transient
and macOS malloc page retention, which drowns the optimizer working set (the
quantity of interest).  This probe isolates it: each config runs TWO
optimizer steps on synthetic batches of the tier's shapes (no parquet load)
in a fresh subprocess and reports ``ru_maxrss`` (bytes on macOS) minus a
model+forward baseline, plus the second step's t_grad/t_step wall times.

Every config is probed at BOTH tiers ("train" and "profile"), and the
profile-tier gram config is additionally swept over
``chunk_numel in {2**16, 2**18, 2**20, 2**22}`` — the chunk-size memory scan.

Because this machine is shared, each child runs under a watchdog that aborts
the process once its peak RSS crosses ``--cap-gb`` (default 2 GB); such
configs are recorded as over-cap rather than measured.  The profile tier's
M=64 row space makes every jacrev-based capture exceed the default cap (a
single uncapped reference run of the chunked capture at chunk_numel=2**16
peaked at ~4.8 GB), so the sweep is also runnable at the train tier.

Writes ``<out>/memory.json``; ``plot_comparison.py`` reads it when present.

Usage:
    python mem_probe.py                            # both tiers + profile sweep
    python mem_probe.py --sweep-tiers profile train --cap-gb 2
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP_CHUNKS = [2 ** 16, 2 ** 18, 2 ** 20, 2 ** 22]
SWEEP_CONFIG = "sven_gram_c20"
OVER_CAP_EXIT = 97


def rss_bytes() -> int:
    """Peak RSS in BYTES: ru_maxrss is bytes on macOS but kilobytes on Linux."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru * 1024 if sys.platform.startswith("linux") else ru


def _start_watchdog(cap_bytes: int) -> None:
    """Abort the process (exit 97) once peak RSS crosses ``cap_bytes``."""
    import threading

    def poll() -> None:
        while True:
            if rss_bytes() > cap_bytes:
                os._exit(OVER_CAP_EXIT)
            time.sleep(0.025)

    threading.Thread(target=poll, daemon=True).start()


def child(config: str, tier: str, chunk: int, cap_bytes: int, device_arg: str) -> None:
    if cap_bytes > 0:
        _start_watchdog(cap_bytes)
    import torch

    torch.set_num_threads(4)
    sys.path.insert(0, HERE)
    from model import TIERS, build_model, per_seq_ce
    from train_one import CONFIGS, build_wrapper_and_opt, resolve_device, sync, SEED

    device = resolve_device(device_arg)
    cfg = TIERS[tier]
    model = build_model(tier, SEED)
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, cfg["vocab"], (cfg["batch_size"], cfg["seq_len"]), generator=g).to(device)
    y = torch.randint(0, cfg["vocab"], (cfg["batch_size"], cfg["seq_len"]), generator=g).to(device)
    kind, opts = CONFIGS[config]

    t_grad = t_step = 0.0
    if kind == "probe":
        model.to(device)
        model(x)
        sync(device)
    else:
        opts = dict(opts)
        if chunk > 0:
            opts["chunk_numel"] = chunk
        wrapper, opt = build_wrapper_and_opt(
            kind, opts, model, sven_lr=0.3, device=device, track=False
        )
        for _ in range(2):  # second step includes lazily-allocated optimizer state
            if kind in ("sven", "gram"):
                sync(device)
                t0 = time.perf_counter()
                wrapper.loss_and_grad((x, y))
                sync(device)
                t1 = time.perf_counter()
                opt.step()
                sync(device)
                t2 = time.perf_counter()
            else:
                sync(device)
                t0 = time.perf_counter()
                loss = per_seq_ce(model(x), y).mean()
                opt.zero_grad()
                loss.backward()
                sync(device)
                t1 = time.perf_counter()
                opt.step()
                sync(device)
                t2 = time.perf_counter()
            t_grad, t_step = t1 - t0, t2 - t1

    print(json.dumps({
        "config": config, "tier": tier, "device": device,
        "peak": rss_bytes(),
        "peak_cuda": (torch.cuda.max_memory_allocated()
                      if device.startswith("cuda") else None),
        "t_grad": t_grad, "t_step": t_step,
    }))


def _run_child(config: str, tier: str, chunk: int, cap_bytes: int) -> dict | str:
    """Probe one config; returns the child's dict, "over_cap" or "failed"."""
    out = subprocess.run(
        [sys.executable, __file__, "--child", config, tier, str(chunk), str(cap_bytes),
         _DEVICE],
        capture_output=True, text=True,
    )
    if out.returncode == OVER_CAP_EXIT:
        return "over_cap"
    if out.returncode != 0:
        err = out.stderr.strip().splitlines()
        return "failed: " + (err[-1] if err else f"exit {out.returncode}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def _peak(res: dict) -> int:
    """Device-memory peak when on CUDA (the meaningful number there), else RSS."""
    return res["peak_cuda"] if res.get("peak_cuda") is not None else res["peak"]


def _probe_tier(tier: str, configs: list[str], cap_bytes: int) -> dict:
    section: dict = {"baseline": None, "per_step_delta": {}, "t_grad": {}, "t_step": {},
                     "over_cap": [], "failed": {}}
    res = _run_child("probe", tier, 0, cap_bytes)
    if not isinstance(res, dict):
        print(f"[{tier}] probe baseline {res}; skipping tier")
        return section
    section["baseline"] = _peak(res)
    section["device"] = res.get("device", "cpu")
    print(f"[{tier}] {'probe':16s} {_peak(res)/1e6:8.1f} MB (baseline, {section['device']})")
    for cfg in configs:
        if cfg == "probe":
            continue
        res = _run_child(cfg, tier, 0, cap_bytes)
        if res == "over_cap":
            section["over_cap"].append(cfg)
            print(f"[{tier}] {cfg:16s}    > cap")
        elif isinstance(res, str):
            section["failed"][cfg] = res
            print(f"[{tier}] {cfg:16s}    {res}")
        else:
            section["per_step_delta"][cfg] = _peak(res) - section["baseline"]
            section["t_grad"][cfg] = res["t_grad"]
            section["t_step"][cfg] = res["t_step"]
            print(f"[{tier}] {cfg:16s} {section['per_step_delta'][cfg]/1e6:8.1f} MB "
                  f"t_grad {res['t_grad']:.2f}s t_step {res['t_step']:.2f}s")
    return section


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="*", default=["train", "profile"])
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--sweep-tiers", nargs="*", default=["profile"])
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--cap-gb", type=float, default=2.0)
    ap.add_argument("--device", default="auto", help="cpu, cuda, cuda:N, or auto")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    sys.path.insert(0, HERE)
    from train_one import CONFIGS

    global _DEVICE
    _DEVICE = args.device
    cap_bytes = int(args.cap_gb * 1e9)
    configs = args.configs if args.configs else list(CONFIGS)
    result: dict = {"cap_gb": args.cap_gb}
    for tier in args.tiers:
        result[tier] = _probe_tier(tier, configs, cap_bytes)

    result["chunk_sweeps"] = {}
    for tier in [] if args.skip_sweep else args.sweep_tiers:
        sweep: dict = {"tier": tier, "config": SWEEP_CONFIG, "per_step_delta": {},
                       "t_grad": {}, "t_step": {}, "over_cap": [], "failed": {}}
        baseline = result.get(tier, {}).get("baseline")
        if baseline is None:
            res = _run_child("probe", tier, 0, cap_bytes)
            baseline = _peak(res) if isinstance(res, dict) else None
        sweep["baseline"] = baseline
        for chunk in SWEEP_CHUNKS:
            res = _run_child(SWEEP_CONFIG, tier, chunk, cap_bytes)
            key = str(chunk)
            if res == "over_cap":
                sweep["over_cap"].append(chunk)
                print(f"[sweep/{tier}] chunk 2**{chunk.bit_length()-1:2d}    > cap")
            elif isinstance(res, str):
                sweep["failed"][key] = res
                print(f"[sweep/{tier}] chunk 2**{chunk.bit_length()-1:2d}    {res}")
            else:
                sweep["per_step_delta"][key] = _peak(res) - baseline
                sweep["t_grad"][key] = res["t_grad"]
                sweep["t_step"][key] = res["t_step"]
                print(f"[sweep/{tier}] chunk 2**{chunk.bit_length()-1:2d} "
                      f"{sweep['per_step_delta'][key]/1e6:8.1f} MB "
                      f"t_grad {res['t_grad']:.2f}s t_step {res['t_step']:.2f}s")
        result["chunk_sweeps"][tier] = sweep

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "memory.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print("wrote", path)


_DEVICE = "auto"

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
    else:
        main()

"""Plots for the rtol/k/lr sensitivity scan (``results_scan/*.json``).

Two figures: final validation CE vs ``rtol`` (with the AdamW reference and
the no-truncation default marked), and the long-budget curve comparison
(AdamW vs the best Gram setting).  Follows the ``plot_comparison`` palette
and one-axis-per-chart conventions.

Usage:
    python plot_scan.py [--results results_scan]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_comparison import BASE, GRID, INK, INK2, MUTED, SURFACE

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results_scan"))
    args = ap.parse_args()
    plots = os.path.join(args.results, "plots")
    os.makedirs(plots, exist_ok=True)

    R = {}
    for p in glob.glob(os.path.join(args.results, "*.json")):
        r = json.load(open(p))
        R[r["config"]] = r

    def final_ce(name):
        return R[name]["evals"][-1]["val_ce"]

    # --- 1: final val CE vs rtol ------------------------------------------
    rtols = sorted(
        (r["sven_rtol"], name) for name, r in R.items()
        if r["kind"] == "gram" and r["sven_k"] == 32 and r.get("sven_lr") in (None, 0.1)
        and r["steps"] == min(x["steps"] for x in R.values())
    )
    if rtols:
        fig, ax = plt.subplots(figsize=(7.2, 4.8), facecolor=SURFACE)
        style_ax(ax)
        xs = [rt for rt, _ in rtols]
        ys = [final_ce(n) for _, n in rtols]
        ax.plot(xs, ys, color=BLUE, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, label="Sven Gram (hooks), k=32")
        if "scan_adamw" in R:
            ax.axhline(final_ce("scan_adamw"), color=AQUA, linewidth=2,
                       linestyle=(0, (4, 3)), label="AdamW reference")
        ax.set_xscale("log")
        ax.set_xlabel("rtol (log)", color=INK2, fontsize=10)
        ax.set_ylabel("final validation CE (nats)", color=INK2, fontsize=10)
        ax.set_title("Truncation tolerance scan\n", color=INK, fontsize=13,
                     loc="left", fontweight="bold")
        ax.text(0, 1.02, "rtol ≤ 1e-2 never truncates (full rank kept); "
                "the optimum sits near 0.1", transform=ax.transAxes,
                color=INK2, fontsize=9.5)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
        fig.tight_layout()
        fig.savefig(os.path.join(plots, "scan_rtol.png"), dpi=150, facecolor=SURFACE)
        plt.close(fig)
        print("wrote", os.path.join(plots, "scan_rtol.png"))

    # --- 2: long-budget curves --------------------------------------------
    long_steps = max(r["steps"] for r in R.values())
    pairs = [(n, r) for n, r in R.items() if r["steps"] == long_steps]
    if len(pairs) >= 2 and long_steps > min(r["steps"] for r in R.values()):
        fig, ax = plt.subplots(figsize=(8.0, 4.8), facecolor=SURFACE)
        style_ax(ax)
        for name, r in sorted(pairs):
            color = AQUA if r["kind"] != "gram" else ORANGE
            label = "AdamW" if r["kind"] != "gram" else (
                f"Sven Gram rtol={r['sven_rtol']:g}")
            ax.plot([e["step"] for e in r["evals"]],
                    [e["val_ce"] / math.log(2) for e in r["evals"]],
                    color=color, linewidth=2, label=label)
        ax.set_xlabel("optimizer step", color=INK2, fontsize=10)
        ax.set_ylabel("validation CE (bits/byte)", color=INK2, fontsize=10)
        ax.set_title("Long-budget comparison\n", color=INK, fontsize=13,
                     loc="left", fontweight="bold")
        ax.text(0, 1.02, "Sven plateaus after ~300 steps; AdamW keeps descending",
                transform=ax.transAxes, color=INK2, fontsize=9.5)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
        fig.tight_layout()
        fig.savefig(os.path.join(plots, "scan_long.png"), dpi=150, facecolor=SURFACE)
        plt.close(fig)
        print("wrote", os.path.join(plots, "scan_long.png"))

    if any(r.get("batch_size", 0) > 32 for r in R.values()):
        plot_m_scan(R, plots)


def plot_m_scan(R: dict, plots: str) -> None:
    """Val-CE curves vs steps for the M (batch-size) scan, both optimizers."""
    ramp = {32: "#86b6ef", 64: "#256abf", 128: "#0d366b"}
    fig, ax = plt.subplots(figsize=(8.0, 4.8), facecolor=SURFACE)
    style_ax(ax)
    for name, r in sorted(R.items()):
        b = r["batch_size"]
        if b not in ramp:
            continue
        is_gram = r["kind"] == "gram"
        if is_gram and not (r.get("sven_rtol") == 0.1 and r["sven_k"] == b
                            and r.get("sven_lr") in (None, 0.1)):
            continue
        if not is_gram and name not in (f"scan_adamw_M{b}", "scan_adamw"):
            continue
        ax.plot([e["step"] for e in r["evals"]],
                [e["val_ce"] / math.log(2) for e in r["evals"]],
                color=ramp[b], linewidth=2,
                linestyle="--" if is_gram else "-",
                label=("Sven Gram" if is_gram else "AdamW") + f" M={b}")
    ax.set_xlabel("optimizer step", color=INK2, fontsize=10)
    ax.set_ylabel("validation CE (bits/byte)", color=INK2, fontsize=10)
    ax.set_title("Row-count (M = batch) scan\n", color=INK, fontsize=13,
                 loc="left", fontweight="bold")
    ax.text(0, 1.02, "ramp = M; solid = AdamW, dashed = Sven Gram (rtol 0.1, k = M)",
            transform=ax.transAxes, color=INK2, fontsize=9.5)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "scan_M.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", os.path.join(plots, "scan_M.png"))


if __name__ == "__main__":
    main()

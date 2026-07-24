"""Produce comparison plots from ``results/*.json`` into ``results/plots/``.

Design notes: categorical colors follow the ENTITY (optimizer) in fixed slot
order and are never cycled; the masked-fraction family is an ordered magnitude
so it uses a single-hue sequential ramp instead of extra hues.  Palette
validated (CVD + contrast) — series whose hue sits below 3:1 on the surface
always carry direct labels.  One y-axis per chart, no dual axes.

Usage:
    python plot_comparison.py [--results results]
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- palette (validated) ---------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"

COLOR = {  # entity -> fixed categorical slot
    "sven_classic": "#2a78d6",  # blue
    "sven_gram": "#eb6834",     # orange
    "adam": "#1baf7a",          # aqua   (sub-3:1 -> direct label)
    "adamw": "#eda100",         # yellow (sub-3:1 -> direct label)
    "sgd": "#e87ba4",           # magenta(sub-3:1 -> direct label)
    "lbfgs": "#008300",         # green
    "sven_struct": "#4a3aa7",   # violet
    # masked fractions: ordinal steps of the sequential blue ramp
    "sven_frac10": "#86b6ef",
    "sven_frac25": "#5598e7",
    "sven_frac50": "#256abf",
    "sven_frac80": "#104281",
}
LABEL = {
    "sven_classic": "Sven (full)",
    "sven_gram": "Sven (Gram)",
    "sven_frac10": "Sven 10%",
    "sven_frac25": "Sven 25%",
    "sven_frac50": "Sven 50%",
    "sven_frac80": "Sven 80%",
    "sven_struct": "Sven struct 9%",
    "adam": "Adam",
    "adamw": "AdamW",
    "sgd": "SGD+mom",
    "lbfgs": "L-BFGS",
}
HEADLINE = ["sven_classic", "sven_gram", "adam", "adamw", "sgd", "lbfgs"]
FRACTIONS = ["sven_frac10", "sven_frac25", "sven_frac50", "sven_frac80"]
ALL_ORDER = (
    ["sven_classic", "sven_gram"] + FRACTIONS
    + ["sven_struct", "adam", "adamw", "sgd", "lbfgs"]
)


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def new_fig(w=8.0, h=5.0):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURFACE)
    style_ax(ax)
    return fig, ax


def finish(fig, ax, title, sub, xlabel, ylabel, out, legend=True, endlabels=None):
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.set_title(title + "\n", color=INK, fontsize=13, loc="left", fontweight="bold")
    ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK2, fontsize=9.5)
    if legend:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    if endlabels:  # direct labels at line ends for sub-contrast hues
        for name, (x, y) in endlabels.items():
            ax.annotate(LABEL[name], (x, y), xytext=(4, 0), textcoords="offset points",
                        color=INK2, fontsize=8.5, va="center")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out)


def series(r, key):
    return [e[key] for e in r["epochs"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = ap.parse_args()
    plots = os.path.join(args.results, "plots")
    os.makedirs(plots, exist_ok=True)

    R = {}
    for name in ALL_ORDER + ["probe"]:
        path = os.path.join(args.results, f"{name}.json")
        if os.path.exists(path):
            with open(path) as f:
                R[name] = json.load(f)
    present = [n for n in ALL_ORDER if n in R]
    probe_rss = R.get("probe", {}).get("peak_rss", 0)

    # Per-step working-set numbers from mem_probe.py (preferred: per-run RSS is
    # polluted by the data-load transient and malloc page retention)
    mem_path = os.path.join(args.results, "memory.json")
    mem = None
    if os.path.exists(mem_path):
        with open(mem_path) as f:
            mem = json.load(f)["per_step_delta"]

    def mem_mb(name):
        if mem and name in mem:
            return mem[name] / 1e6
        if R[name].get("peak_rss"):
            return (R[name]["peak_rss"] - probe_rss) / 1e6
        return None

    # --- 1+2: val loss / val acc vs epoch (headline set) -------------------
    for key, ylab, fname, logy in [
        ("val_loss", "validation loss (CE)", "val_loss_epoch.png", True),
        ("val_acc", "validation accuracy", "val_acc_epoch.png", False),
    ]:
        fig, ax = new_fig()
        ends = {}
        for name in [n for n in HEADLINE if n in R]:
            ep, ys = series(R[name], "epoch"), series(R[name], key)
            ax.plot(ep, ys, color=COLOR[name], linewidth=2, label=LABEL[name])
            ends[name] = (ep[-1], ys[-1])
        if logy:
            ax.set_yscale("log")
        finish(fig, ax, f"CIFAR-10 (4k train), 1M-param MLP — {ylab}",
               "equal epochs; per-config hyperparameters untuned defaults",
               "epoch", ylab, os.path.join(plots, fname), endlabels=ends)

    # --- 3: val acc vs cumulative optimizer wall time ----------------------
    fig, ax = new_fig()
    ends = {}
    for name in [n for n in HEADLINE if n in R]:
        xs, ys = series(R[name], "cum_opt_time"), series(R[name], "val_acc")
        ax.plot(xs, ys, color=COLOR[name], linewidth=2, label=LABEL[name])
        ends[name] = (xs[-1], ys[-1])
    ax.set_xscale("log")
    finish(fig, ax, "Validation accuracy vs optimizer wall time",
           "x = cumulative grad+step time (log scale); eval time excluded",
           "cumulative optimizer time (s)", "validation accuracy",
           os.path.join(plots, "val_acc_walltime.png"), endlabels=ends)

    # --- 4: masked-fraction panel (sequential ramp + classic reference) ----
    fig, ax = new_fig()
    for name in [n for n in FRACTIONS if n in R]:
        ax.plot(series(R[name], "epoch"), series(R[name], "val_acc"),
                color=COLOR[name], linewidth=2, label=LABEL[name])
    if "sven_struct" in R:
        ax.plot(series(R["sven_struct"], "epoch"), series(R["sven_struct"], "val_acc"),
                color=COLOR["sven_struct"], linewidth=2, linestyle=(0, (1.5, 1.5)),
                label=LABEL["sven_struct"])
    if "sven_classic" in R:
        ax.plot(series(R["sven_classic"], "epoch"), series(R["sven_classic"], "val_acc"),
                color=INK, linewidth=2, linestyle=(0, (4, 3)), label="Sven (full, ref)")
    finish(fig, ax, "Stochastic parameter fraction — validation accuracy",
           "elementwise masks resampled per step (ramp = fraction); struct = whole-tensor",
           "epoch", "validation accuracy", os.path.join(plots, "fractions_val_acc.png"))

    # --- 5: small multiples, train vs val loss per config ------------------
    n = len(present)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.7 * rows),
                             facecolor=SURFACE, sharex=True, sharey=True)
    for ax, name in zip(np.ravel(axes), present):
        style_ax(ax)
        r = R[name]
        ax.plot(series(r, "epoch"), series(r, "train_loss"), color=MUTED,
                linewidth=1.6, label="train")
        ax.plot(series(r, "epoch"), series(r, "val_loss"), color=COLOR[name],
                linewidth=2, label="val")
        ax.set_yscale("log")
        ax.set_title(LABEL[name], color=INK, fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    for ax in np.ravel(axes)[n:]:
        ax.set_visible(False)
    fig.suptitle("Train (gray) vs validation loss — overfitting view",
                 color=INK, fontsize=13, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(plots, "train_val_grid.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", os.path.join(plots, "train_val_grid.png"))

    # --- 6: wall time per batch (stacked grad | step) ----------------------
    fig, ax = new_fig(8.0, 4.6)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.grid(False, axis="y")
    names = [n for n in present if R[n].get("t_grad")]  # partial recoveries lack cost data
    names = sorted(names, key=lambda n: np.mean(R[n]["t_grad"]) + np.mean(R[n]["t_step"]))
    ypos = np.arange(len(names))
    for i, name in enumerate(names):
        g, s = float(np.mean(R[name]["t_grad"])), float(np.mean(R[name]["t_step"]))
        ax.barh(i, g, color=COLOR[name], height=0.62, edgecolor=SURFACE, linewidth=2)
        ax.barh(i, s, left=g, color=COLOR[name], alpha=0.45, height=0.62,
                edgecolor=SURFACE, linewidth=2)
        ax.text(g + s, i, f"  {g+s:.3g}s", color=INK2, fontsize=8.5, va="center")
    ax.set_yticks(ypos, [LABEL[n] for n in names], color=INK2, fontsize=9)
    ax.set_xscale("log")
    finish(fig, ax, "Mean wall time per batch",
           "solid = gradient/Jacobian phase, faded = optimizer step (log scale)",
           "seconds per batch (log)", "", os.path.join(plots, "cost_time.png"), legend=False)

    # --- 7: peak memory per config -----------------------------------------
    fig, ax = new_fig(8.0, 4.6)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.grid(False, axis="y")
    names = [n for n in present if mem_mb(n) is not None]
    names = sorted(names, key=mem_mb)
    for i, name in enumerate(names):
        mb = mem_mb(name)
        ax.barh(i, max(mb, 1.0), color="#2a78d6", height=0.62,
                edgecolor=SURFACE, linewidth=2)
        ax.text(max(mb, 1.0), i, f"  {mb:,.0f} MB", color=INK2, fontsize=8.5, va="center")
    ax.set_yticks(np.arange(len(names)), [LABEL[n] for n in names], color=INK2, fontsize=9)
    ax.set_xscale("log")
    finish(fig, ax, "Per-step peak memory over model+data baseline",
           "fresh-process probe, two optimizer steps on synthetic batch (mem_probe.py)",
           "peak RSS delta (MB, log)", "", os.path.join(plots, "cost_memory.png"), legend=False)

    # --- 8: efficiency frontier --------------------------------------------
    fig, ax = new_fig(7.2, 5.4)
    for name in [n for n in present if mem_mb(n) is not None]:
        mb = max(mem_mb(name), 1.0)
        acc = max(series(R[name], "val_acc"))
        ax.scatter(mb, acc, s=90, color=COLOR[name], edgecolor=SURFACE, linewidth=1.5,
                   zorder=3)
        ax.annotate(LABEL[name], (mb, acc), xytext=(6, 4), textcoords="offset points",
                    color=INK2, fontsize=8.5)
    ax.set_xscale("log")
    finish(fig, ax, "Best validation accuracy vs per-step peak memory",
           "up and to the left is better; every point direct-labeled",
           "per-step peak RSS delta (MB, log)", "best validation accuracy",
           os.path.join(plots, "frontier.png"), legend=False)

    # --- 9: Sven retained rank ---------------------------------------------
    sven_named = [n for n in present if R[n].get("num_nonzero_svs")]
    if sven_named:
        fig, ax = new_fig()
        for name in sven_named:
            ranks = R[name]["num_nonzero_svs"]
            ax.plot(np.arange(1, len(ranks) + 1), ranks, color=COLOR[name],
                    linewidth=1.6 if name.startswith("sven_frac") else 2,
                    label=LABEL[name])
        finish(fig, ax, "Retained SVD rank per step",
               "components surviving k=128 and rtol=1e-4 truncation",
               "optimizer step", "retained components",
               os.path.join(plots, "sven_rank.png"))


if __name__ == "__main__":
    main()

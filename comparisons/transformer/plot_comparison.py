"""Produce transformer-benchmark plots from ``results/*.json`` into ``results/plots/``.

Design notes: categorical colors follow the ENTITY (optimizer) in fixed slot
order and are never cycled; the chunk_numel scan and the masked-rows scan are
ordered magnitudes so each uses a single-hue sequential ramp (orange / blue)
instead of extra hues.  Palette validated (CVD + contrast) — series whose hue
sits below 3:1 on the surface always carry direct labels.  One y-axis per
chart, no dual axes; over-cap memory probes are drawn as faded bars at the
cap with an explicit annotation.

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
    "sven_classic": "#2a78d6",   # blue
    # chunk_numel scan: ordered family -> sequential orange ramp
    "sven_gram_c18": "#f59d6b",
    "sven_gram_c20": "#eb6834",
    "sven_gram_c22": "#a84314",
    # masked rows scan: ordered family -> sequential blue ramp (dash-dot)
    "sven_gram_rows10": "#86b6ef",
    "sven_gram_rows25": "#5598e7",
    "sven_gram_rows50": "#256abf",
    "adam": "#1baf7a",           # aqua   (sub-3:1 -> direct label)
    "adamw": "#eda100",          # yellow (sub-3:1 -> direct label)
    "sgd": "#e87ba4",            # magenta(sub-3:1 -> direct label)
}
LABEL = {
    "sven_classic": "Sven (full)",
    "sven_gram_c18": "Sven Gram 2^18",
    "sven_gram_c20": "Sven Gram 2^20",
    "sven_gram_c22": "Sven Gram 2^22",
    "sven_gram_rows10": "Sven Gram rows 10%",
    "sven_gram_rows25": "Sven Gram rows 25%",
    "sven_gram_rows50": "Sven Gram rows 50%",
    "adam": "Adam",
    "adamw": "AdamW",
    "sgd": "SGD+mom",
}
HEADLINE = ["sven_classic", "sven_gram_c20", "adam", "adamw", "sgd"]
CHUNKS = ["sven_gram_c18", "sven_gram_c20", "sven_gram_c22"]
ROWS = ["sven_gram_rows10", "sven_gram_rows25", "sven_gram_rows50"]
ALL_ORDER = ["sven_classic"] + CHUNKS + ROWS + ["adam", "adamw", "sgd"]


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
    return [e[key] for e in r["evals"]]


def memory_bars(section, names, cap_mb, title, sub, out):
    """Horizontal per-config memory bars; over-cap probes drawn faded at cap."""
    delta = section.get("per_step_delta", {})
    over = set(section.get("over_cap", []))
    rows = [(n, delta.get(n)) for n in names if n in delta or n in over]
    if not rows:
        return
    rows.sort(key=lambda nv: cap_mb * 1e6 if nv[1] is None else nv[1])
    fig, ax = new_fig(8.0, 4.6)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.grid(False, axis="y")
    for i, (name, v) in enumerate(rows):
        if v is None:  # over cap: watchdog aborted the probe
            ax.barh(i, cap_mb, color=COLOR.get(name, BASE), alpha=0.35, height=0.62,
                    edgecolor=SURFACE, linewidth=2)
            ax.text(cap_mb, i, "  > cap (aborted)", color=INK2, fontsize=8.5, va="center")
        else:
            mb = max(v / 1e6, 1.0)
            ax.barh(i, mb, color=COLOR.get(name, BASE), height=0.62,
                    edgecolor=SURFACE, linewidth=2)
            ax.text(mb, i, f"  {v/1e6:,.0f} MB", color=INK2, fontsize=8.5, va="center")
    ax.set_yticks(np.arange(len(rows)), [LABEL.get(n, n) for n, _ in rows],
                  color=INK2, fontsize=9)
    ax.set_xscale("log")
    finish(fig, ax, title, sub, "peak RSS delta (MB, log)", "", out, legend=False)


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
    present = [n for n in ALL_ORDER if n in R and R[n].get("evals")]

    mem = None
    mem_path = os.path.join(args.results, "memory.json")
    if os.path.exists(mem_path):
        with open(mem_path) as f:
            mem = json.load(f)

    # --- 1: val CE (bits/byte) vs steps (headline set) ---------------------
    fig, ax = new_fig()
    ends = {}
    for name in [n for n in HEADLINE if n in present]:
        xs, ys = series(R[name], "step"), series(R[name], "val_bpb")
        ax.plot(xs, ys, color=COLOR[name], linewidth=2, label=LABEL[name])
        ends[name] = (xs[-1], ys[-1])
    finish(fig, ax, "Byte-level OpenWebText, 0.9M-param GPT — validation CE",
           "equal steps and identical batch order; Sven lr from the 30-step scan",
           "optimizer step", "validation CE (bits/byte)",
           os.path.join(plots, "val_ce_steps.png"), endlabels=ends)

    # --- 2: val CE vs cumulative optimizer wall time -----------------------
    fig, ax = new_fig()
    ends = {}
    for name in [n for n in HEADLINE if n in present]:
        xs, ys = series(R[name], "cum_opt_time"), series(R[name], "val_bpb")
        xs, ys = [max(x, 1e-2) for x in xs], ys
        ax.plot(xs, ys, color=COLOR[name], linewidth=2, label=LABEL[name])
        ends[name] = (xs[-1], ys[-1])
    ax.set_xscale("log")
    finish(fig, ax, "Validation CE vs optimizer wall time",
           "x = cumulative grad+step time (log scale); eval time excluded",
           "cumulative optimizer time (s)", "validation CE (bits/byte)",
           os.path.join(plots, "val_ce_walltime.png"), endlabels=ends)

    # --- 3: chunked-capture panel (chunk scan = identical updates) ---------
    chunk_present = [n for n in CHUNKS if n in present]
    if len(chunk_present) > 1:
        fig, ax = new_fig()
        for name in chunk_present:
            ax.plot(series(R[name], "step"), series(R[name], "val_bpb"),
                    color=COLOR[name], linewidth=2, label=LABEL[name])
        if "sven_classic" in present:
            ax.plot(series(R["sven_classic"], "step"), series(R["sven_classic"], "val_bpb"),
                    color=INK, linewidth=2, linestyle=(0, (4, 3)), label="Sven (full, ref)")
        finish(fig, ax, "chunk_numel scan — validation CE",
               "the Gram is exact at every chunk size: curves should coincide "
               "(cost differs, see chunk_memory/chunk_time)",
               "optimizer step", "validation CE (bits/byte)",
               os.path.join(plots, "chunk_panel.png"))

    # --- 4: masked-rows panel (only if the rows configs ran) ---------------
    rows_present = [n for n in ROWS if n in present]
    if rows_present:
        fig, ax = new_fig()
        for name in rows_present:
            frac = R[name].get("actual_param_fraction")
            label = LABEL[name] + (f" (got {frac:.0%})" if frac else "")
            ax.plot(series(R[name], "step"), series(R[name], "val_bpb"),
                    color=COLOR[name], linewidth=2, linestyle="-.", label=label)
        if "sven_gram_c20" in present:
            ax.plot(series(R["sven_gram_c20"], "step"), series(R["sven_gram_c20"], "val_bpb"),
                    color=INK, linewidth=2, linestyle=(0, (4, 3)), label="Sven Gram (full, ref)")
        finish(fig, ax, "Stochastic parameter fraction (rows) — validation CE",
               "ramp = fraction; dash-dot = masked chunked capture",
               "optimizer step", "validation CE (bits/byte)",
               os.path.join(plots, "rows_panel.png"))

    # --- 5: wall time per batch (stacked grad | step) ----------------------
    names = [n for n in present if R[n].get("t_grad")]
    if names:
        names = sorted(names, key=lambda n: np.mean(R[n]["t_grad"]) + np.mean(R[n]["t_step"]))
        fig, ax = new_fig(8.0, 4.6)
        ax.grid(True, axis="x", color=GRID, linewidth=0.8)
        ax.grid(False, axis="y")
        for i, name in enumerate(names):
            g, s = float(np.mean(R[name]["t_grad"])), float(np.mean(R[name]["t_step"]))
            ax.barh(i, g, color=COLOR[name], height=0.62, edgecolor=SURFACE, linewidth=2)
            ax.barh(i, s, left=g, color=COLOR[name], alpha=0.45, height=0.62,
                    edgecolor=SURFACE, linewidth=2)
            ax.text(g + s, i, f"  {g+s:.3g}s", color=INK2, fontsize=8.5, va="center")
        ax.set_yticks(np.arange(len(names)), [LABEL[n] for n in names], color=INK2, fontsize=9)
        ax.set_xscale("log")
        finish(fig, ax, "Mean wall time per batch (train tier)",
               "solid = gradient/Gram phase, faded = optimizer step (log scale)",
               "seconds per batch (log)", "", os.path.join(plots, "cost_time.png"),
               legend=False)

    # --- 6+7: per-step peak memory per tier (mem_probe.py) -----------------
    if mem:
        cap_mb = mem.get("cap_gb", 2.0) * 1e3
        for tier in ("train", "profile"):
            if tier in mem and mem[tier].get("baseline") is not None:
                memory_bars(
                    mem[tier], ALL_ORDER, cap_mb,
                    f"Per-step peak memory over model baseline — {tier} tier",
                    "fresh-process probe, two optimizer steps on synthetic batch "
                    f"(mem_probe.py, cap {cap_mb/1e3:g} GB)",
                    os.path.join(plots, f"cost_memory_{tier}.png"),
                )

    # --- 8+9: chunk_numel tradeoff (memory and time vs chunk size) ---------
    for tier, sweep in ((mem or {}).get("chunk_sweeps", {})).items():
        if len(sweep.get("per_step_delta", {})) < 2:
            continue
        chunks = sorted(int(c) for c in sweep["per_step_delta"])
        for key, ylab, fname in [
            ("per_step_delta", "peak RSS delta (MB)", f"chunk_memory_{tier}.png"),
            ("t_grad", "Gram-capture time per step (s)", f"chunk_time_{tier}.png"),
        ]:
            ys = [sweep[key][str(c)] / (1e6 if key == "per_step_delta" else 1.0)
                  for c in chunks]
            fig, ax = new_fig(7.2, 4.8)
            ax.plot(chunks, ys, color=COLOR["sven_gram_c20"], linewidth=2, marker="o",
                    markersize=6, markeredgecolor=SURFACE)
            for c, v in zip(chunks, ys):
                ax.annotate(f"2^{c.bit_length()-1}", (c, v), xytext=(0, 8),
                            textcoords="offset points", color=INK2, fontsize=8.5,
                            ha="center")
            for c in sweep.get("over_cap", []):
                ax.axvline(c, color=MUTED, linewidth=1, linestyle=(0, (2, 2)))
                ax.annotate("> cap", (c, max(ys)), color=MUTED, fontsize=8.5,
                            ha="center", va="bottom")
            ax.set_xscale("log", base=2)
            finish(fig, ax,
                   f"chunk_numel tradeoff — {ylab.split(' (')[0]} ({tier} tier)",
                   "chunked Gram capture, 2-step fresh-process probe (mem_probe.py)",
                   "chunk_numel (log2)", ylab, os.path.join(plots, fname),
                   legend=False)


if __name__ == "__main__":
    main()

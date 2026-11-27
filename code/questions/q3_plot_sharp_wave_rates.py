#!/usr/bin/env python
"""Plot sharp-wave candidate rates by center row and by mouse."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator


DEFAULT_RESULTS = Path(__file__).resolve().parents[2] / "results" / "q3" / "psd_swr"
AREA_COLORS = {1: "#0072B2", 2: "#D55E00", 3: "#00856A"}
RATE = "sharp_waves_per_min"


def plot_mouse(mouse_id, mouse, output, *, rate=RATE, position="center_row",
               event_label="Sharp-wave", file_stem="sharp_wave"):
    fig = Figure(figsize=(11.3, 4.6), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3, sharex=True)
    fig.suptitle(f"Mouse {mouse_id:02d} · {event_label} candidates by channel",
                 fontsize=15, fontweight="bold")
    fig.supxlabel(f"{event_label} candidates per minute")
    row_label, unit_label = ("Center row", "centers") if position == "center_row" else (
        "Channel row", "channels")
    fig.supylabel(f"{row_label} (deepest to superficial)")
    xmax = max(1.0, mouse[rate].dropna().max()) * 1.08

    for ax, (area, color) in zip(axes, AREA_COLORS.items()):
        panel = mouse.loc[mouse.brain_area == area].sort_values(position)
        valid = panel.status.eq("ok")
        ax.set(title=f"Area {area} · {valid.sum()}/{len(panel)} {unit_label}",
               xlim=(0, xmax), ylim=(0.5, len(panel) + 0.5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis="x", color="#DFE4E9", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        # Shade rows with an unavailable flank or unusable signal.
        unavailable = panel.loc[~valid, position].to_numpy()
        blocks = np.split(unavailable, np.flatnonzero(np.diff(unavailable) > 1) + 1)
        for block in blocks:
            if block.size:
                ax.axhspan(block[0] - 0.5, block[-1] + 0.5, color="#EEF0F3", zorder=0)
        if valid.any():
            ax.plot(panel[rate], panel[position], color=color, marker="o", ms=3.3, lw=1.25)
        else:
            ax.text(0.5, 0.5, f"No eligible {unit_label}", transform=ax.transAxes,
                    ha="center", va="center", color="#4A5563")

    fig.savefig(output / f"q3_mouse_{mouse_id:02d}_{file_stem}_rate.png", dpi=165, facecolor="white")
    fig.clear()


def mouse_area_means(rates, *, rate=RATE):
    eligible = rates.loc[rates.status.eq("ok") & np.isfinite(rates[rate])]
    return eligible.groupby(["mouse_id", "brain_area"], as_index=False)[rate].mean()


def mouse_area_peaks(rates, *, rate=RATE, position="center_row"):
    eligible = rates.loc[rates.status.eq("ok") & np.isfinite(rates[rate])]
    sort_keys = ["mouse_id", "brain_area", "n_candidates", position]
    ordered = eligible.sort_values(sort_keys, ascending=[True, True, False, True])
    return ordered.drop_duplicates(["mouse_id", "brain_area"])


def plot_area_summary(means, path, *, title, ylabel, rate=RATE):
    fig = Figure(figsize=(6.8, 4.8), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.set_title(title, fontsize=15, fontweight="bold", pad=32)
    ax.text(0.5, 1.035, "One dot per mouse. Horizontal bars show means across mice.",
            transform=ax.transAxes, ha="center", fontsize=10, color="#465463")
    ax.set(ylabel=ylabel, xlim=(0.5, 3.5), ylim=(0, 1.12 * max(1, means[rate].max())))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E1E6EB")
    ax.spines[["top", "right"]].set_visible(False)

    mice = means.mouse_id.unique()
    rng = np.random.default_rng(31)
    offsets = dict(zip(mice, rng.uniform(-0.19, 0.19, size=len(mice))))
    labels = []
    for area, color in AREA_COLORS.items():
        area_means = means.loc[means.brain_area.eq(area)]
        labels.append(f"Area {area}\nn = {len(area_means)}")
        ax.scatter(area + area_means.mouse_id.map(offsets), area_means[rate],
                   s=41, color=color, alpha=0.81, edgecolor="white", linewidth=0.55, zorder=3)
        if len(area_means):
            ax.hlines(area_means[rate].mean(), area - 0.28, area + 0.28,
                      color="#233142", linewidth=2.3, zorder=4)
    ax.set_xticks(list(AREA_COLORS), labels)
    fig.savefig(path, dpi=180, facecolor="white")
    fig.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--summary-only", action="store_true",
                      help="Render only the across-channel mean summary.")
    mode.add_argument("--peak-only", action="store_true",
                      help="Render only the highest-count channel summary.")
    args = parser.parse_args()
    rates = pd.read_csv(args.results_dir / "q3_sharp_wave_channel_rates.csv")
    if not args.peak_only:
        summary_path = args.results_dir / "q3_sharp_wave_rate_area_summary.png"
        plot_area_summary(mouse_area_means(rates), summary_path,
                          title="Sharp-wave candidate rates by area",
                          ylabel="Mean candidates/min across channel centers")
        print(f"Saved the mouse-level area summary to {summary_path.resolve()}.")
    if not args.summary_only:
        peak_path = args.results_dir / "q3_sharp_wave_rate_peak_channel_summary.png"
        plot_area_summary(mouse_area_peaks(rates), peak_path,
                          title="Peak-channel sharp-wave rates by area",
                          ylabel="Candidates/min at highest-count channel center")
        print(f"Saved the highest-count channel summary to {peak_path.resolve()}.")
    if args.summary_only or args.peak_only:
        return
    output = args.results_dir / "rate_plots"
    output.mkdir(parents=True, exist_ok=True)
    for mouse_id, mouse in rates.groupby("mouse_id", sort=True):
        if set(mouse.brain_area) != set(AREA_COLORS):
            raise ValueError(f"Missing area data for mouse {mouse_id}")
        plot_mouse(mouse_id, mouse, output)
    print(f"Saved {rates.mouse_id.nunique()} rate figures to {output.resolve()}.")


if __name__ == "__main__":
    main()

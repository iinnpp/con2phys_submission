#!/usr/bin/env python
"""Align ripple candidates with sharp-wave windows two and three rows deeper."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from q3_plot_sharp_wave_rates import AREA_COLORS


Q3 = Path(__file__).resolve().parents[2] / "results" / "q3"
OFFSETS = (2, 3)
MODES = ("2_below", "3_below", "either", "both")
MODE_LABELS = {"2_below": "2 rows deeper", "3_below": "3 rows deeper",
               "either": "Either offset", "both": "Both offsets"}
PROFILE_STYLES = {
    "2_below": ("#0072B2", "-", "o"),
    "3_below": ("#D55E00", "--", "s"),
    "either": ("#3F4B59", ":", "^"),
}
OVERLAP_TOL_S = 1e-12
RIPPLE_KEYS = ["mouse_id", "brain_area", "channel_row"]
SPW_KEYS = ["mouse_id", "brain_area", "center_row"]
PAIR_COLUMNS = [
    "mouse_id", "brain_area", "ripple_channel_row", "spw_row_offset", "spw_center_row",
    "ripple_start_sample", "ripple_start_s", "ripple_stop_s_exclusive", "ripple_trough_s",
    "spw_peak_s", "spw_half_height_width_ms", "spw_start_s", "spw_stop_s",
    "overlap_ms", "trough_minus_spw_peak_ms",
]


def overlap_pairs(ripple_start, ripple_stop, spw_peak, spw_width_ms):
    """Return ripple and SPW row indices plus the pairwise overlap in seconds."""
    if not len(ripple_start) or not len(spw_peak):
        return np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0)
    radius = spw_width_ms / 2000.0
    max_radius = radius.max()
    lower = np.searchsorted(spw_peak, ripple_start - max_radius, side="left")
    upper = np.searchsorted(spw_peak, ripple_stop + max_radius, side="right")
    n = upper - lower
    ripple_idx = np.repeat(np.arange(len(n)), n)
    within_group = np.arange(n.sum()) - np.repeat(n.cumsum() - n, n)
    spw_idx = lower[ripple_idx] + within_group
    start = np.maximum(ripple_start[ripple_idx], spw_peak[spw_idx] - radius[spw_idx])
    stop = np.minimum(ripple_stop[ripple_idx], spw_peak[spw_idx] + radius[spw_idx])
    overlap = stop - start
    keep = overlap > OVERLAP_TOL_S
    return ripple_idx[keep], spw_idx[keep], overlap[keep]


def align(ripple_dir, spw_dir, output):
    ripple_channels = pd.read_csv(ripple_dir / "q3_ripple_channel_rates.csv").set_index(
        RIPPLE_KEYS)
    spw_channels = pd.read_csv(spw_dir / "q3_sharp_wave_channel_rates.csv").set_index(
        SPW_KEYS)
    if not ripple_channels.index.is_unique or not spw_channels.index.is_unique:
        raise ValueError("Expected one rate row per mouse, area, and channel.")
    ripples = pd.read_csv(ripple_dir / "q3_ripple_events.csv.gz", usecols=[
        *RIPPLE_KEYS, "start_sample", "start_s", "stop_s_exclusive", "lfp_trough_s"])
    spws = pd.read_csv(spw_dir / "q3_sharp_wave_events.csv.gz", usecols=[
        *SPW_KEYS, "peak_time_s", "half_height_width_ms"])
    spws = spws.sort_values([*SPW_KEYS, "peak_time_s"], ignore_index=True)
    ripple_groups = ripples.groupby(RIPPLE_KEYS, sort=False).indices
    spw_groups = spws.groupby(SPW_KEYS, sort=False).indices
    empty = np.empty(0, dtype=int)

    rows, pairs = [], []
    for key, channel in ripple_channels.iterrows():
        mouse, area, ripple_row = key
        waves = ripples.iloc[ripple_groups.get(key, empty)]
        valid_ripple = channel.status == "ok"
        if valid_ripple and len(waves) != channel.n_candidates:
            raise ValueError(f"Ripple count mismatch for {key}")
        row = dict(mouse_id=mouse, brain_area=area, ripple_channel_row=ripple_row,
                   duration_s=channel.duration_s, ripple_status=channel.status,
                   n_ripple_candidates=channel.n_candidates)
        matched = {}
        eligible = {}
        for offset in OFFSETS:
            mode = f"{offset}_below"
            target = (mouse, area, ripple_row - offset)
            exists = target in spw_channels.index
            eligible[mode] = valid_ripple and exists and spw_channels.at[target, "status"] == "ok"
            row[f"spw_center_row_{mode}"] = target[2] if exists else None
            matched[mode] = np.zeros(len(waves), dtype=bool)
            if not eligible[mode]:
                continue

            spw_channel = spw_channels.loc[target]
            if not np.isclose(channel.duration_s, spw_channel.duration_s, atol=0.001, rtol=0):
                raise ValueError(f"Recording duration mismatch for ripple {key} and SPW {target}")
            sharp = spws.iloc[spw_groups.get(target, empty)]
            if len(sharp) != spw_channel.n_candidates:
                raise ValueError(f"Sharp-wave count mismatch for {target}")
            r, s, overlap = overlap_pairs(waves.start_s.to_numpy(), waves.stop_s_exclusive.to_numpy(),
                                          sharp.peak_time_s.to_numpy(), sharp.half_height_width_ms.to_numpy())
            matched[mode][r] = True
            if len(r):
                rp = waves.iloc[r]
                sw = sharp.iloc[s]
                spw_peak = sw.peak_time_s.to_numpy()
                spw_width = sw.half_height_width_ms.to_numpy()
                trough = rp.lfp_trough_s.to_numpy()
                pairs.append(pd.DataFrame({
                    "mouse_id": mouse, "brain_area": area, "ripple_channel_row": ripple_row,
                    "spw_row_offset": offset, "spw_center_row": target[2],
                    "ripple_start_sample": rp.start_sample.to_numpy(),
                    "ripple_start_s": rp.start_s.to_numpy(),
                    "ripple_stop_s_exclusive": rp.stop_s_exclusive.to_numpy(),
                    "ripple_trough_s": trough, "spw_peak_s": spw_peak,
                    "spw_half_height_width_ms": spw_width, "spw_start_s": spw_peak - spw_width / 2000,
                    "spw_stop_s": spw_peak + spw_width / 2000, "overlap_ms": overlap * 1000,
                    "trough_minus_spw_peak_ms": (trough - spw_peak) * 1000,
                }))
        eligible["either"] = any(eligible.values())
        eligible["both"] = eligible["2_below"] and eligible["3_below"]
        matched["either"] = matched["2_below"] | matched["3_below"]
        matched["both"] = matched["2_below"] & matched["3_below"]
        for mode in MODES:
            count = int(matched[mode].sum()) if eligible[mode] else None
            row[f"eligible_{mode}"] = eligible[mode]
            row[f"n_matched_{mode}"] = count
            row[f"rate_per_min_{mode}"] = count * 60 / channel.duration_s if count is not None else None
        rows.append(row)

    output.mkdir(parents=True, exist_ok=True)
    rates = pd.DataFrame(rows)
    event_pairs = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame(columns=PAIR_COLUMNS)
    rates.to_csv(output / "q3_spwr_channel_rates.csv", index=False)
    event_pairs.to_csv(output / "q3_spwr_overlap_pairs.csv.gz", index=False,
                       compression={"method": "gzip", "compresslevel": 4, "mtime": 0})
    return rates, len(event_pairs)


def mouse_summaries(rates):
    summaries = []
    for mode in MODES:
        groups = rates.loc[rates[f"eligible_{mode}"]].groupby(["mouse_id", "brain_area"])
        for (mouse, area), group in groups:
            rate = f"rate_per_min_{mode}"
            peak = group.sort_values([rate, "ripple_channel_row"], ascending=[False, True]).iloc[0]
            summaries.append(dict(mouse_id=mouse, brain_area=area, matching_rule=mode,
                                  n_channels=len(group), mean_rate_per_min=group[rate].mean(),
                                  peak_rate_per_min=peak[rate],
                                  peak_ripple_channel_row=peak.ripple_channel_row))
    return pd.DataFrame(summaries)


def plot_mouse(mouse_id, rates, output):
    fig = Figure(figsize=(11.5, 4.9), layout="constrained")
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(2, 3, height_ratios=(0.08, 1))
    legend = fig.add_subplot(grid[0, :])
    legend.axis("off")
    left = fig.add_subplot(grid[1, 0])
    axes = [left, fig.add_subplot(grid[1, 1], sharex=left),
            fig.add_subplot(grid[1, 2], sharex=left)]
    fig.suptitle(f"Mouse {mouse_id:02d} · Sharp-wave–ripple overlaps by channel",
                 fontsize=15, fontweight="bold")
    fig.supxlabel("Matched ripple candidates per minute")
    fig.supylabel("Ripple channel row (deepest to superficial)")
    handles = [Line2D([], [], color=color, linestyle=style, marker=marker,
                      label=MODE_LABELS[mode], markersize=4)
               for mode, (color, style, marker) in PROFILE_STYLES.items()]
    handles.append(Patch(facecolor="#EEF0F3", edgecolor="none", label="Unavailable"))
    legend.legend(handles=handles, loc="center", ncol=4, frameon=False)
    xmax = max(1.0, rates.rate_per_min_either.max()) * 1.08
    for ax, area in zip(axes, AREA_COLORS):
        panel = rates.loc[rates.brain_area.eq(area)].sort_values("ripple_channel_row")
        available = panel.eligible_either
        ax.set(title=f"Area {area} · {available.sum()}/{len(panel)} channels",
               xlim=(0, xmax), ylim=(0.5, len(panel) + 0.5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.grid(axis="x", color="#E1E6EB")
        ax.spines[["top", "right"]].set_visible(False)
        unavailable = panel.loc[~available, "ripple_channel_row"].to_numpy()
        for block in np.split(unavailable, np.flatnonzero(np.diff(unavailable) > 1) + 1):
            if block.size:
                ax.axhspan(block[0] - 0.5, block[-1] + 0.5, color="#EEF0F3", zorder=0)
        for mode, (color, style, marker) in PROFILE_STYLES.items():
            ax.plot(panel[f"rate_per_min_{mode}"], panel.ripple_channel_row, color=color,
                    ls=style, marker=marker, lw=1.3, ms=3.1)
        if not available.any():
            ax.text(0.5, 0.5, "No eligible sharp-wave rows", transform=ax.transAxes,
                    ha="center", va="center", color="#4A5563")
    fig.savefig(output / f"q3_mouse_{mouse_id:02d}_spwr_rate.png", dpi=165, facecolor="white")


def plot_summary(summary, metric, title, path):
    fig = Figure(figsize=(10.4, 7.6), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2, sharey=True)
    fig.suptitle(title + "\nOne dot per mouse. Horizontal bars show means across mice.",
                 fontsize=13, linespacing=1.4)
    fig.supylabel("Matched ripple candidates per minute")
    mice = sorted(summary.mouse_id.unique())
    jitter = dict(zip(mice, np.random.default_rng(31).uniform(-0.18, 0.18, len(mice))))
    ymax = max(1.0, summary[metric].max()) * 1.10
    for ax, mode in zip(axes.flat, MODES):
        ax.set(title=MODE_LABELS[mode], xlim=(0.5, 3.5), ylim=(0, ymax))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E1E6EB")
        ax.spines[["top", "right"]].set_visible(False)
        labels = []
        for area, color in AREA_COLORS.items():
            group = summary.loc[summary.matching_rule.eq(mode) & summary.brain_area.eq(area)]
            labels.append(f"Area {area}\nn = {len(group)}")
            ax.scatter(area + group.mouse_id.map(jitter), group[metric], s=34,
                       color=color, alpha=0.83, edgecolor="white", lw=0.5, zorder=3)
            if len(group):
                ax.hlines(group[metric].mean(), area - 0.24, area + 0.24,
                          color="#233142", lw=2, zorder=4)
        ax.set_xticks(list(AREA_COLORS), labels)
    fig.savefig(path, dpi=165, facecolor="white")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ripple-dir", type=Path, default=Q3 / "psd_ripples")
    parser.add_argument("--spw-dir", type=Path, default=Q3 / "psd_swr")
    parser.add_argument("--output-dir", type=Path, default=Q3 / "psd_spwr")
    args = parser.parse_args()
    output = args.output_dir
    rates, n_pairs = align(args.ripple_dir, args.spw_dir, output)
    summary = mouse_summaries(rates)
    summary.to_csv(output / "q3_spwr_mouse_area_rates.csv", index=False)
    plot_dir = output / "rate_plots"
    plot_dir.mkdir(exist_ok=True)
    for mouse, group in rates.groupby("mouse_id", sort=True):
        plot_mouse(mouse, group, plot_dir)
    plot_summary(summary, "mean_rate_per_min", "Sharp-wave–ripple overlap rates · Mean across channels",
                 output / "q3_spwr_rate_area_summary.png")
    plot_summary(summary, "peak_rate_per_min", "Sharp-wave–ripple overlap rates · Highest-count channel",
                 output / "q3_spwr_rate_peak_channel_summary.png")
    config = dict(ripple_dir=str(args.ripple_dir.resolve()), spw_dir=str(args.spw_dir.resolve()),
                  channel_row_order="deepest to superficial",
                  spw_row_offsets=OFFSETS, spw_center_row="ripple channel row minus offset",
                  spw_interval="peak time ± half of the stored full half-height width",
                  ripple_interval="stored start and exclusive stop",
                  overlap="intersection longer than 1e-12 seconds",
                  counting="one ripple per offset; either is the union and both is the intersection",
                  eligibility="ripple and SPW channels valid; either needs one valid offset and both needs two",
                  rate_denominator="full recording duration in minutes",
                  event_label="overlap of pre-detected ripple and sharp-wave candidates")
    (output / "q3_spwr_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {n_pairs:,} overlap pairs, {len(rates):,} channel rows, "
          f"{rates.mouse_id.nunique()} mouse profiles, and two summaries to {output.resolve()}.")


if __name__ == "__main__":
    main()

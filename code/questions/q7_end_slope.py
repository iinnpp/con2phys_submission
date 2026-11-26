"""Plot Q7 waveform end-slope distributions from the saved unit profiles."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


Q7_RESULTS = Path(__file__).resolve().parents[2] / "results" / "q7"
# Sun et al. (2021), Fig. 2D, doi:10.1113/JP280844.
END_SLOPE_TIME_MS = 0.33
AREA_COLORS = {1: "#0072B2", 2: "#D55E00", 3: "#00856A"}


def load_waveform_profiles(results, extra_columns=()):
    keys = ["mouse_id", "cluster_id", "brain_area", *extra_columns]
    units = pd.read_csv(results / "q7_units.csv", usecols=[*keys, "qc_status"])
    with np.load(results / "q7_unit_profiles.npz") as profiles:
        for key in ("mouse_id", "cluster_id"):
            if not np.array_equal(units[key].to_numpy(), profiles[key]):
                raise ValueError(f"Unit CSV and waveform profiles disagree on {key}")
        grid_ms = profiles["waveform_grid_ms"]
        waveforms = profiles["waveform"]

    passed = units.qc_status.eq("pass")
    units = units.loc[passed, keys].copy()
    waveforms = waveforms[passed.to_numpy()]
    if not np.isfinite(waveforms).all():
        raise ValueError("Quality-passed units include missing waveform values")
    return units, grid_ms, waveforms


def slope_at_post_trough_time(waveform, grid_ms):
    """Measure the centered gradient of a trough-aligned, trough-normalized waveform."""
    sample_ms = np.diff(grid_ms)[0]
    left_ms = END_SLOPE_TIME_MS - sample_ms
    right_ms = END_SLOPE_TIME_MS + sample_ms
    left, right = np.interp([left_ms, right_ms], grid_ms, waveform, left=np.nan, right=np.nan)
    return (right - left) / (right_ms - left_ms)


def measure_end_slope(results):
    units, grid_ms, waveforms = load_waveform_profiles(results)
    units["end_slope_per_ms"] = [slope_at_post_trough_time(w, grid_ms) for w in waveforms]
    return units


def plot_distribution(units, path):
    slopes = units.end_slope_per_ms
    bin_width = 0.2
    first = int(np.floor(slopes.min() / bin_width))
    last = int(np.ceil(slopes.max() / bin_width))
    bins = np.arange(first, last + 1) * bin_width
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharex=True, sharey=True,
                             layout="constrained")
    for ax, (area, color) in zip(axes, AREA_COLORS.items()):
        values = units.loc[units.brain_area.eq(area), "end_slope_per_ms"]
        ax.hist(values, bins=bins, weights=np.full(len(values), 1 / len(values)),
                color=color, alpha=0.86, edgecolor="white", linewidth=0.6)
        ax.axvline(0, color="0.3", linestyle="--", linewidth=1)
        ax.set_title(f"Area {area}  (n = {len(values)})", fontsize=13)
        ax.set_xlim(bins[0], bins[-1])
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="0.9")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[0].set_ylabel("Percentage of units per bin")
    fig.suptitle("Waveform end-slope by brain area", fontsize=18)
    fig.supxlabel(f"Slope {END_SLOPE_TIME_MS:.2f} ms after trough (trough depths per ms)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Q7_RESULTS)
    args = parser.parse_args()

    units = measure_end_slope(args.results)
    output = args.results / "simple_approach"
    output.mkdir(parents=True, exist_ok=True)
    plot_distribution(units, output / "q7_end_slope_distribution.png")
    units.to_csv(output / "q7_end_slope_units.csv", index=False)


if __name__ == "__main__":
    main()

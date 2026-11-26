"""Compare the local slopes before the trough, after the trough, and after the peak."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np

from q7_end_slope import AREA_COLORS, Q7_RESULTS, load_waveform_profiles


PHASES = (
    ("pre_trough", "Before trough", "Depolarizing"),
    ("post_trough", "After trough", "Repolarizing"),
    ("post_peak", "After peak", "End-slope"),
)
OFFSET_SAMPLES = 2


def measure_slopes(results):
    units, grid_ms, waveforms = load_waveform_profiles(results, ("trough_to_peak_ms",))
    sample_ms = np.diff(grid_ms)[0]
    offset_ms = OFFSET_SAMPLES * sample_ms
    # Profiles are aligned to the trough, so trough-to-peak duration is the peak time.
    peak_ms = units.trough_to_peak_ms.to_numpy()
    if not np.isfinite(peak_ms).all() or np.any(peak_ms <= offset_ms + sample_ms):
        raise ValueError("Post-trough slope window must end before the positive peak")
    centers_ms = np.column_stack([
        np.full(len(units), -offset_ms),
        np.full(len(units), offset_ms),
        peak_ms + offset_ms,
    ])
    endpoints_ms = centers_ms[:, :, None] + np.array([-sample_ms, sample_ms])
    if endpoints_ms.min() < grid_ms[0] or endpoints_ms.max() > grid_ms[-1]:
        raise ValueError("Slope window would extend beyond the saved waveform")
    endpoints = np.array([np.interp(t, grid_ms, w) for t, w in zip(endpoints_ms, waveforms)])
    slopes = np.diff(endpoints, axis=2)[:, :, 0] / (2 * sample_ms)

    for idx, (phase, _, _) in enumerate(PHASES):
        units[f"{phase}_slope_time_ms_from_trough"] = centers_ms[:, idx]
        units[f"{phase}_slope_per_ms"] = slopes[:, idx]
    return units, offset_ms


def plot_distributions(units, offset_ms, path):
    fig, axes = plt.subplots(3, 3, figsize=(12, 8.8), sharex="row", sharey="row",
                             layout="constrained")
    for row, (phase, timing, name) in enumerate(PHASES):
        slopes = units[f"{phase}_slope_per_ms"]
        bin_width = 0.2 if phase == "post_peak" else 0.5
        first = int(np.floor(min(0, slopes.min()) / bin_width))
        last = int(np.ceil(max(0, slopes.max()) / bin_width))
        bins = np.arange(first, last + 1) * bin_width
        for col, (area, color) in enumerate(AREA_COLORS.items()):
            ax = axes[row, col]
            values = slopes[units.brain_area.eq(area)]
            ax.hist(values, bins=bins, weights=np.full(len(values), 1 / len(values)),
                    color=color, alpha=0.86, edgecolor="white", linewidth=0.6)
            if bins[0] < 0 < bins[-1]:
                ax.axvline(0, color="0.4", linestyle="--", linewidth=1)
            ax.set_xlim(bins[0], bins[-1])
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="0.9")
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(f"Area {area}  (n = {len(values)})", fontsize=12)
        axes[row, 0].set_ylabel(f"{timing}\n{name}")
        axes[row, 0].xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        axes[row, 0].yaxis.set_major_locator(MaxNLocator(nbins=6))
        axes[row, 0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    fig.suptitle(f"Waveform slopes by brain area\nLocal gradients {offset_ms:.3f} ms from the trough or peak",
                 fontsize=15)
    fig.supxlabel("Slope (trough depths per ms)")
    fig.supylabel("Percentage of units per bin")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Q7_RESULTS)
    args = parser.parse_args()

    units, offset_ms = measure_slopes(args.results)
    output = args.results / "simple_approach"
    output.mkdir(parents=True, exist_ok=True)
    plot_distributions(units, offset_ms, output / "q7_waveform_slope_distributions.png")
    units.to_csv(output / "q7_waveform_slope_units.csv", index=False)


if __name__ == "__main__":
    main()

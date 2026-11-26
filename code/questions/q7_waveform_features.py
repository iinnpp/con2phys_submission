"""Measure Q7 features on the original 30 kHz average waveforms.

Use the feature definitions supplied for Sun and Ibbotson (2026), Fig. 2A.
The post-peak inflection landmark is also illustrated by Trainito et al. (2019), Fig. 1A.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from q7_end_slope import END_SLOPE_TIME_MS, slope_at_post_trough_time
from q7_fast_spiking import DEFAULT_OUTPUT, SAMPLES_PER_MS, WAVEFORM_GRID_MS, crossing, vertex
from q7_waveform_pca import AREA_COLORS, load_raw_pca_waveforms


PEAK_SEARCH_SAMPLES = 39  # 1.3 ms after the trough, as in the earlier Q7 analysis.
SLOPE_OFFSET_SAMPLES = 2  # 0.067 ms before or after the corresponding extremum.
INFLECTION_WINDOW_SAMPLES = 7
FEATURES = (
    ("trough_to_peak_ms", "Trough-to-peak duration", "Duration (ms)", 1),
    ("peak_to_trough_ratio", "Peak-to-trough ratio", "Positive peak height / negative trough depth", 1),
    ("half_width_ms", "Trough half-width", "Width at half trough depth (ms)", 1),
    ("amplitude_native", "Waveform amplitude", "Maximum − minimum (×10⁻⁶ dataset units)", 1e6),
    ("depolarizing_slope_native_per_ms", "Depolarizing slope", "Slope before trough (×10⁻⁶ dataset units per ms)", 1e6),
    ("repolarizing_slope_native_per_ms", "Repolarizing slope", "Slope after trough (×10⁻⁶ dataset units per ms)", 1e6),
    ("end_slope_native_per_ms", "End-slope", "Slope after positive peak (×10⁻⁶ dataset units per ms)", 1e6),
    ("sun2021_end_slope_per_ms", "End-slope (Sun et al., 2021)",
     f"Slope {END_SLOPE_TIME_MS:.2f} ms after trough (trough depths per ms)", 1),
    ("repolarization_time_ms", "Repolarization time", "Time from positive peak to falling-phase inflection (ms)", 1),
)


def local_slope(waveform, center):
    """Centered difference over two samples in the original amplitude units per ms."""
    if center - 1 < 0 or center + 1 > len(waveform) - 1:
        return np.nan
    left, right = np.interp([center - 1, center + 1], np.arange(len(waveform)), waveform)
    return (right - left) * SAMPLES_PER_MS / 2


def falling_inflection(waveform, peak, peak_time):
    """Find the first concave-down to concave-up transition on the post-peak descent."""
    baseline_after_peak = np.flatnonzero(waveform[peak + 1:] <= 0) + peak + 1
    return_time = len(waveform) - 1
    if baseline_after_peak.size:
        right = baseline_after_peak[0]
        return_time = crossing(waveform, right - 1, right, 0)

    # Fit local cubics to estimate derivatives without differencing raw sample noise twice.
    velocity = savgol_filter(waveform, INFLECTION_WINDOW_SAMPLES, 3, deriv=1)
    curvature = savgol_filter(waveform, INFLECTION_WINDOW_SAMPLES, 3, deriv=2)
    half_window = INFLECTION_WINDOW_SAMPLES // 2
    for i in range(max(peak - 1, half_window), len(waveform) - half_window - 1):
        if i >= return_time:
            break
        if curvature[i] < 0 <= curvature[i + 1]:
            time = crossing(curvature, i, i + 1, 0)
            if peak_time < time <= return_time and np.interp(time, [i, i + 1], velocity[i:i + 2]) < 0:
                return time
    return np.nan


def measure_waveform(raw):
    baseline = np.median(np.r_[raw[:12], raw[-12:]])
    y = raw - baseline
    trough = int(np.argmin(y))
    features = {
        "baseline_native": baseline,
        "amplitude_native": np.ptp(raw),
        "positive_dominant": y.max() > -y[trough],
    }
    if y[trough] >= 0 or trough in (0, len(y) - 1):
        return features

    trough_time, trough_depth = vertex(y, trough)
    features["trough_time_ms"] = trough_time / SAMPLES_PER_MS
    features["depolarizing_slope_native_per_ms"] = local_slope(y, trough_time - SLOPE_OFFSET_SAMPLES)
    aligned = np.interp(trough_time + WAVEFORM_GRID_MS * SAMPLES_PER_MS, np.arange(len(y)), y,
                        left=np.nan, right=np.nan) / -trough_depth
    features["sun2021_end_slope_per_ms"] = slope_at_post_trough_time(aligned, WAVEFORM_GRID_MS)
    half_level = trough_depth / 2
    before = np.flatnonzero(y[:trough] >= half_level)
    after = np.flatnonzero(y[trough + 1:] >= half_level) + trough + 1
    if before.size and after.size:
        left = crossing(y, before[-1], before[-1] + 1, half_level)
        right = crossing(y, after[0] - 1, after[0], half_level)
        features["half_width_ms"] = (right - left) / SAMPLES_PER_MS

    peaks, _ = find_peaks(y)
    peaks = peaks[(peaks > trough) & (peaks <= trough + PEAK_SEARCH_SAMPLES) & (y[peaks] > 0)]
    rise_center = trough_time + SLOPE_OFFSET_SAMPLES
    if not peaks.size:
        features["repolarizing_slope_native_per_ms"] = local_slope(y, rise_center)
        return features

    peak = peaks[np.argmax(y[peaks])]
    peak_time, peak_height = vertex(y, peak)
    inflection = falling_inflection(y, peak, peak_time)
    features.update(
        peak_time_ms=peak_time / SAMPLES_PER_MS,
        post_peak_inflection_time_ms=inflection / SAMPLES_PER_MS,
        trough_to_peak_ms=(peak_time - trough_time) / SAMPLES_PER_MS,
        peak_to_trough_ratio=peak_height / -trough_depth,
        end_slope_native_per_ms=local_slope(y, peak_time + SLOPE_OFFSET_SAMPLES),
        repolarization_time_ms=(inflection - peak_time) / SAMPLES_PER_MS,
    )
    if rise_center + 1 <= peak_time:
        features["repolarizing_slope_native_per_ms"] = local_slope(y, rise_center)
    return features


def plot_feature(units, feature, title, xlabel, scale, path):
    values = units[feature].dropna() * scale
    low, high = min(0, values.min()), max(0, values.max())
    if low == high:
        high = low + 1
    bins = np.linspace(low, high, 37)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.15), sharex=True, sharey=True, layout="constrained")
    for ax, (area, color) in zip(axes, AREA_COLORS.items()):
        area_values = units.loc[units.brain_area.eq(area), feature].dropna() * scale
        if len(area_values):
            ax.hist(area_values, bins=bins, weights=np.full(len(area_values), 1 / len(area_values)),
                    color=color, alpha=.86, edgecolor="white", linewidth=.6)
        if low < 0 < high:
            ax.axvline(0, color="0.4", linestyle="--", linewidth=1)
        ax.set_xlim(low, high)
        ax.set_title(f"Area {area}  (n = {len(area_values)})", fontsize=12)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="0.9")
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    axes[0].set_ylabel("Percentage of measured units per bin")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    fig.supxlabel(xlabel)
    fig.suptitle(f"{title} by brain area", fontsize=17)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-pca-dir", type=Path, default=DEFAULT_OUTPUT / "pca")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "waveform_features")
    args = parser.parse_args()

    waveforms, metadata = load_raw_pca_waveforms(args.raw_pca_dir)
    columns = [feature for feature, *_ in FEATURES]
    measurements = pd.DataFrame([measure_waveform(waveform) for waveform in waveforms])
    landmarks = ["baseline_native", "trough_time_ms", "peak_time_ms", "post_peak_inflection_time_ms", "positive_dominant"]
    units = metadata.join(measurements.reindex(columns=[*columns, *landmarks]))
    args.output.mkdir(parents=True, exist_ok=True)
    units.to_csv(args.output / "q7_waveform_features.csv", index=False)
    summaries = []
    for feature, title, xlabel, scale in FEATURES:
        stats = units.groupby("brain_area")[feature].agg(n_total="size", n_measured="count", median="median")
        stats["n_missing"] = stats.n_total - stats.n_measured
        summaries.append(stats.reset_index().assign(feature=feature))
        plot_feature(units, feature, title, xlabel, scale, args.output / f"q7_{feature}_distribution.png")
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(args.output / "q7_waveform_feature_summary.csv", index=False)
    print(summary.pivot(index="feature", columns="brain_area", values="n_measured").to_string())


if __name__ == "__main__":
    main()

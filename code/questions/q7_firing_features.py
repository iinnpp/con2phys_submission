"""Calculate full-recording firing rates and per-unit interspike intervals for Q7."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from q6_glm import read_mice
from q7_fast_spiking import DEFAULT_OUTPUT
from q7_waveform_features import plot_feature


DEFAULT_DATASET = Path("E:/con2phys_submission/dataset")
KEYS = ["mouse_id", "cluster_id", "brain_area"]
FEATURES = (
    ("firing_rate_hz", "Mean firing rate", "Firing rate (spikes/s)"),
    ("isi_median_ms", "Median interspike interval", "Per-unit median interspike interval (ms)"),
)


def measure_units(dataset):
    metadata = pd.read_csv(dataset / "unit_metadata.csv", usecols=[*KEYS, "n_spikes", "recording_duration_sec"])
    if metadata.duplicated(["mouse_id", "cluster_id"]).any() or set(metadata.brain_area) != {1, 2, 3}:
        raise ValueError("Expected unique unit IDs and three brain areas in unit_metadata.csv")

    rows = []
    for mouse, areas, _, duration in read_mice(dataset / "full_sua_dataset.csv", selected=None):
        for area, units in areas.items():
            for cluster, spikes in units.items():
                intervals_ms = np.diff(spikes) * 1_000
                rows.append({
                    "mouse_id": mouse, "cluster_id": cluster, "brain_area": area,
                    "sua_n_spikes": spikes.size, "sua_duration_sec": duration, "n_isi": intervals_ms.size,
                    "isi_mean_ms": np.mean(intervals_ms) if intervals_ms.size else np.nan,
                    "isi_median_ms": np.median(intervals_ms) if intervals_ms.size else np.nan,
                })
    units = metadata.merge(pd.DataFrame(rows), on=KEYS, how="outer", validate="one_to_one", indicator=True)
    if (not units["_merge"].eq("both").all() or not units.n_spikes.eq(units.sua_n_spikes).all()
            or not np.allclose(units.recording_duration_sec, units.sua_duration_sec, rtol=0, atol=1e-9)):
        raise ValueError("Spike train IDs, spike counts, or recording durations disagree between the source CSVs")

    units["firing_rate_hz"] = units.n_spikes / units.recording_duration_sec
    return units[[*KEYS, "n_spikes", "recording_duration_sec", "firing_rate_hz", "n_isi", "isi_median_ms", "isi_mean_ms"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "firing_features")
    args = parser.parse_args()

    units = measure_units(args.dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    units.to_csv(args.output / "q7_firing_features.csv", index=False)
    summaries = []
    for feature, title, xlabel in FEATURES:
        stats = units.groupby("brain_area")[feature].agg(n_total="size", n_measured="count", median="median")
        summaries.append(stats.reset_index().assign(feature=feature))
        plot_feature(units, feature, title, xlabel, 1, args.output / f"q7_{feature}_distribution.png")
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(args.output / "q7_firing_feature_summary.csv", index=False)
    print(f"Calculated firing features for {len(units):,} units from {units.mouse_id.nunique()} mice.")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))


if __name__ == "__main__":
    main()

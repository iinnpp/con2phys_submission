#!/usr/bin/env python

"""Q15 population spiking-complexity analysis.

Streams each bin width independently, checkpoints after every resolution, and
writes mouse-level inputs for the companion R statistics script.
"""

import csv
import json
import os
import time
import zlib
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

project_folder = Path(os.environ.get("CON2PHYS_PROJECT_DIR", "E:/con2phys_submission"))
repository_folder = Path(os.environ.get(
    "CON2PHYS_REPOSITORY_DIR", "D:/git_repos/con2phys_submission"
))
sua_csv_path = project_folder / "dataset" / "full_sua_dataset.csv"
unit_metadata_csv_path = project_folder / "dataset" / "unit_metadata.csv"
trials_csv_path = repository_folder / "results" / "q11" / "trials_info_filtered.csv"
q_path = Path(os.environ.get("Q15_OUTPUT_DIR", repository_folder / "results" / "q15"))

N_NEURON_SUBSAMPLES = int(os.environ.get("Q15_N_SUBSAMPLES", 20))
N_RATE_SURROGATES = int(os.environ.get("Q15_N_SURROGATES", 1))
BIN_WIDTHS_MS = tuple(
    int(value.strip())
    for value in os.environ.get("Q15_BIN_WIDTHS_MS", "10,20,50").split(",")
    if value.strip()
)
PRIMARY_BIN_WIDTH_MS = 20
COMPRESSION_LEVEL = 1
RANDOM_SEED = 42
EXPECTED_N_MICE = 18
EXPECTED_N_UNITS = 1_449
EXPECTED_BRAIN_AREAS = (1, 2, 3)
MAX_MICE = int(os.environ["Q15_MAX_MICE"]) if os.environ.get("Q15_MAX_MICE") else None

if not BIN_WIDTHS_MS or any(value <= 0 for value in BIN_WIDTHS_MS):
    raise ValueError("BIN_WIDTHS_MS must contain positive integers")
if PRIMARY_BIN_WIDTH_MS not in BIN_WIDTHS_MS and MAX_MICE is None:
    raise ValueError("The full run must include the primary 20 ms bin width")
if N_NEURON_SUBSAMPLES <= 0 or N_RATE_SURROGATES <= 0:
    raise ValueError("Subsample and surrogate counts must be positive")

run_tag = "full" if MAX_MICE is None else f"first_{MAX_MICE}_mice"
file_suffix = "" if MAX_MICE is None else f"_{run_tag}"
unit_counts_csv_path = q_path / f"q15_unit_counts{file_suffix}.csv"
subsamples_csv_path = q_path / f"q15_lzc_subsamples{file_suffix}.csv"
mouse_area_bin_csv_path = q_path / f"q15_lzc_mouse_area_bin{file_suffix}.csv"
area_summary_csv_path = q_path / f"q15_lzc_area_summary{file_suffix}.csv"
figure_path = q_path / f"q15_lzc_by_area{file_suffix}.png"
config_path = q_path / f"q15_run_config{file_suffix}.json"

RUN_CONFIG = {
    "signal": "single-unit population spike raster",
    "spike_window": "[stim_start, trial_end) using complete bins only",
    "trial_filtering": "reuse results/q11/trials_info_filtered.csv from Q11-Q13",
    "primary_metric": "observed DEFLATE compressed bytes / mean rate-matched surrogate compressed bytes",
    "interpretation": "rate-normalized Lempel-Ziv population raster incompressibility",
    "raster_serialization": "time-major binary sequence; random unit order per subsample",
    "surrogate": "independent temporal redistribution preserving each selected unit's total binned spike count",
    "neuron_matching": "minimum area population within mouse",
    "bin_widths_ms": BIN_WIDTHS_MS,
    "primary_bin_width_ms": PRIMARY_BIN_WIDTH_MS,
    "n_neuron_subsamples": N_NEURON_SUBSAMPLES,
    "n_rate_surrogates_per_subsample": N_RATE_SURROGATES,
    "compression": f"zlib DEFLATE level {COMPRESSION_LEVEL}",
    "random_seed": RANDOM_SEED,
    "max_mice": MAX_MICE,
}
RUN_CONFIG

required_paths = [sua_csv_path, unit_metadata_csv_path, trials_csv_path]
missing_paths = [path for path in required_paths if not path.exists()]
if missing_paths:
    raise FileNotFoundError(f"Required paths not found: {missing_paths}")

trials_df = pd.read_csv(trials_csv_path)
expected_trial_columns = [
    "mouse_id", "trial_id", "variable_A", "variable_B", "variable_C",
    "trial_start", "stim_start", "outcome", "trial_end", "trial_duration",
]
if list(trials_df.columns) != expected_trial_columns:
    raise ValueError(f"Unexpected trial schema: {list(trials_df.columns)}")
if trials_df[expected_trial_columns].isna().any().any():
    raise ValueError("Filtered trial table contains missing values")
if trials_df[["mouse_id", "trial_id"]].duplicated().any():
    raise ValueError("Duplicate mouse/trial identifiers found")

correct_order = (
    trials_df["trial_start"].lt(trials_df["stim_start"])
    & trials_df["stim_start"].lt(trials_df["outcome"])
    & trials_df["outcome"].lt(trials_df["trial_end"])
)
if not correct_order.all():
    raise ValueError("Filtered trial table contains incorrectly ordered events")

all_mouse_ids = sorted(int(value) for value in trials_df["mouse_id"].unique())
if len(all_mouse_ids) != EXPECTED_N_MICE:
    raise ValueError(f"Expected {EXPECTED_N_MICE} mice, found {len(all_mouse_ids)}")
selected_mouse_ids = all_mouse_ids if MAX_MICE is None else all_mouse_ids[:MAX_MICE]
trials_df = (
    trials_df.loc[trials_df["mouse_id"].isin(selected_mouse_ids)]
    .sort_values(["mouse_id", "trial_id"])
    .reset_index(drop=True)
)
trials_by_mouse = {
    int(mouse_id): group.reset_index(drop=True)
    for mouse_id, group in trials_df.groupby("mouse_id", sort=True)
}
for mouse_id, mouse_trials in trials_by_mouse.items():
    starts = mouse_trials["stim_start"].to_numpy(dtype=float)
    ends = mouse_trials["trial_end"].to_numpy(dtype=float)
    if np.any(np.diff(starts) <= 0) or np.any(starts[1:] < ends[:-1]):
        raise ValueError(f"Stimulus-to-end intervals overlap or are unsorted for mouse {mouse_id}")

trial_counts = trials_df.groupby("mouse_id").size().rename("n_trials")
display(trial_counts.to_frame())
print(f"Validated and reused {len(trials_df):,} filtered trials from {len(selected_mouse_ids)} mice.")

bin_specs = {}
for mouse_id, mouse_trials in trials_by_mouse.items():
    starts = mouse_trials["stim_start"].to_numpy(dtype=np.float64)
    ends = mouse_trials["trial_end"].to_numpy(dtype=np.float64)
    durations = ends - starts
    for bin_width_ms in BIN_WIDTHS_MS:
        bin_width_sec = bin_width_ms / 1000.0
        n_bins_by_trial = np.floor(durations / bin_width_sec + 1e-10).astype(np.int64)
        if np.any(n_bins_by_trial <= 0):
            raise ValueError(f"Trial shorter than one {bin_width_ms} ms bin for mouse {mouse_id}")
        offsets = np.concatenate(([0], np.cumsum(n_bins_by_trial[:-1], dtype=np.int64)))
        bin_specs[(mouse_id, bin_width_ms)] = {
            "starts": starts,
            "n_bins_by_trial": n_bins_by_trial,
            "offsets": offsets,
            "n_time_bins": int(n_bins_by_trial.sum()),
            "bin_width_sec": bin_width_sec,
        }

def bin_unit_spikes(spike_times, spec):
    """Return one binary value per concatenated complete trial bin."""
    raster = np.zeros(spec["n_time_bins"], dtype=np.bool_)
    trial_index = np.searchsorted(spec["starts"], spike_times, side="right") - 1
    candidate_mask = trial_index >= 0
    if not np.any(candidate_mask):
        return raster
    candidate_spikes = spike_times[candidate_mask]
    candidate_trials = trial_index[candidate_mask]
    relative_time = candidate_spikes - spec["starts"][candidate_trials]
    local_bins = np.floor(relative_time / spec["bin_width_sec"] + 1e-12).astype(np.int64)
    valid = (local_bins >= 0) & (local_bins < spec["n_bins_by_trial"][candidate_trials])
    if np.any(valid):
        global_bins = spec["offsets"][candidate_trials[valid]] + local_bins[valid]
        raster[global_bins] = True
    return raster

expected_sua_columns = [
    "mouse_id", "cluster_id", "brain_area", "recording_duration_sec",
    "unit_last_spike_sec", "n_spikes", "spikes",
]
expected_metadata_columns = expected_sua_columns[:-1]
metadata_df = pd.read_csv(unit_metadata_csv_path)
if list(metadata_df.columns) != expected_metadata_columns:
    raise ValueError(f"Unexpected unit metadata schema: {list(metadata_df.columns)}")
if len(metadata_df) != EXPECTED_N_UNITS:
    raise ValueError(f"Expected {EXPECTED_N_UNITS} unit metadata rows, found {len(metadata_df)}")
if metadata_df[["mouse_id", "cluster_id"]].duplicated().any():
    raise ValueError("Duplicate unit keys in metadata")
metadata_df = metadata_df.loc[metadata_df["mouse_id"].isin(selected_mouse_ids)].copy()

expected_group_keys = {
    (mouse_id, brain_area)
    for mouse_id in selected_mouse_ids
    for brain_area in EXPECTED_BRAIN_AREAS
}
cluster_ids_by_group = {
    (int(mouse_id), int(brain_area)): group["cluster_id"].to_numpy(dtype=int)
    for (mouse_id, brain_area), group in metadata_df.groupby(
        ["mouse_id", "brain_area"], sort=True
    )
}
if set(cluster_ids_by_group) != expected_group_keys:
    raise ValueError(f"Missing mouse-area groups: {sorted(expected_group_keys - set(cluster_ids_by_group))}")

unit_count_rows = []
for mouse_id, brain_area in sorted(expected_group_keys):
    unit_count_rows.append({
        "mouse_id": mouse_id,
        "brain_area": brain_area,
        "n_units": len(cluster_ids_by_group[(mouse_id, brain_area)]),
        "n_trials": len(trials_by_mouse[mouse_id]),
    })
unit_counts_df = pd.DataFrame(unit_count_rows)
matched_units_by_mouse = unit_counts_df.groupby("mouse_id")["n_units"].min().astype(int)
unit_counts_df["n_matched_units"] = unit_counts_df["mouse_id"].map(matched_units_by_mouse)

# Precompute the same selected neurons and unit order for every bin width.
selection_rng = np.random.default_rng(RANDOM_SEED)
selected_columns_by_group_subsample = {}
for mouse_id, brain_area in sorted(expected_group_keys):
    n_available_units = len(cluster_ids_by_group[(mouse_id, brain_area)])
    n_matched_units = int(matched_units_by_mouse.loc[mouse_id])
    for subsample_id in range(1, N_NEURON_SUBSAMPLES + 1):
        selected_columns_by_group_subsample[(mouse_id, brain_area, subsample_id)] = (
            selection_rng.permutation(n_available_units)[:n_matched_units]
        )

q_path.mkdir(parents=True, exist_ok=True)
unit_counts_df.to_csv(unit_counts_csv_path, index=False)
display(unit_counts_df)

def compressed_size_bytes(binary_raster):
    packed = np.packbits(binary_raster, axis=None, bitorder="little")
    return len(zlib.compress(packed.tobytes(), level=COMPRESSION_LEVEL)), packed.size

def make_rate_matched_surrogate(binary_raster, rng):
    """Independently redistribute each unit's active bins, preserving its total."""
    n_time_bins, n_units = binary_raster.shape
    surrogate = np.zeros((n_time_bins, n_units), dtype=np.bool_)
    spike_bin_counts = binary_raster.sum(axis=0, dtype=np.int64)
    for unit_index, n_active_bins in enumerate(spike_bin_counts):
        if n_active_bins:
            chosen_bins = rng.choice(n_time_bins, size=int(n_active_bins), replace=False)
            surrogate[chosen_bins, unit_index] = True
    return surrogate

subsample_rows = []
analysis_started = time.perf_counter()
csv.field_size_limit(2**31 - 1)
for bin_width_ms in BIN_WIDTHS_MS:
    width_started = time.perf_counter()
    raster_columns_by_group = defaultdict(list)
    current_cluster_ids_by_group = defaultdict(list)
    n_source_units = 0
    with sua_csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_sua_columns:
            raise ValueError(f"Unexpected SUA schema: {reader.fieldnames}")
        for row in reader:
            n_source_units += 1
            mouse_id = int(row["mouse_id"])
            if mouse_id not in trials_by_mouse:
                continue
            cluster_id = int(row["cluster_id"])
            brain_area = int(row["brain_area"])
            group_key = (mouse_id, brain_area)
            spike_times = np.fromstring(row["spikes"], sep=";", dtype=np.float64)
            if spike_times.size != int(row["n_spikes"]) or spike_times.size == 0:
                raise ValueError(f"Spike-count mismatch for {(mouse_id, cluster_id)}")
            if not np.isfinite(spike_times).all() or np.any(np.diff(spike_times) < 0):
                raise ValueError(f"Invalid or unsorted spikes for {(mouse_id, cluster_id)}")
            raster_columns_by_group[group_key].append(
                bin_unit_spikes(spike_times, bin_specs[(mouse_id, bin_width_ms)])
            )
            current_cluster_ids_by_group[group_key].append(cluster_id)
    if MAX_MICE is None and n_source_units != EXPECTED_N_UNITS:
        raise ValueError(f"Expected {EXPECTED_N_UNITS} SUA units, found {n_source_units}")
    if set(current_cluster_ids_by_group) != expected_group_keys:
        raise ValueError("SUA file is missing expected mouse-area groups")

    rasters_by_group = {}
    for mouse_id, brain_area in sorted(expected_group_keys):
        group_key = (mouse_id, brain_area)
        current_ids = np.asarray(current_cluster_ids_by_group[group_key], dtype=int)
        if not np.array_equal(current_ids, cluster_ids_by_group[group_key]):
            raise ValueError(f"Unit metadata order differs from SUA order for {group_key}")
        matrix = np.column_stack(raster_columns_by_group[group_key])
        expected_shape = (
            bin_specs[(mouse_id, bin_width_ms)]["n_time_bins"],
            len(cluster_ids_by_group[group_key]),
        )
        if matrix.shape != expected_shape:
            raise ValueError(f"Unexpected raster shape for {group_key}: {matrix.shape} != {expected_shape}")
        rasters_by_group[group_key] = matrix
    del raster_columns_by_group, current_cluster_ids_by_group

    for mouse_id in selected_mouse_ids:
        n_matched_units = int(matched_units_by_mouse.loc[mouse_id])
        for brain_area in EXPECTED_BRAIN_AREAS:
            group_key = (mouse_id, brain_area)
            cluster_ids = cluster_ids_by_group[group_key]
            n_available_units = cluster_ids.size
            raster = rasters_by_group[group_key]
            for subsample_id in range(1, N_NEURON_SUBSAMPLES + 1):
                selected_columns = selected_columns_by_group_subsample[(
                    mouse_id, brain_area, subsample_id
                )]
                selected_cluster_ids = cluster_ids[selected_columns]
                selected_raster = raster[:, selected_columns]
                observed_bytes, packed_bytes = compressed_size_bytes(selected_raster)
                surrogate_bytes = np.empty(N_RATE_SURROGATES, dtype=float)
                surrogate_seed = np.random.SeedSequence([
                    RANDOM_SEED, mouse_id, brain_area, subsample_id, bin_width_ms
                ])
                surrogate_rng = np.random.default_rng(surrogate_seed)
                for surrogate_id in range(N_RATE_SURROGATES):
                    surrogate = make_rate_matched_surrogate(selected_raster, surrogate_rng)
                    surrogate_bytes[surrogate_id], surrogate_packed_bytes = compressed_size_bytes(surrogate)
                    if surrogate_packed_bytes != packed_bytes:
                        raise RuntimeError("Observed and surrogate packed lengths differ")
                surrogate_mean = float(surrogate_bytes.mean())
                surrogate_sd = float(surrogate_bytes.std(ddof=1)) if N_RATE_SURROGATES > 1 else np.nan
                normalized_lzc = observed_bytes / surrogate_mean
                z_score = (
                    (observed_bytes - surrogate_mean) / surrogate_sd
                    if np.isfinite(surrogate_sd) and surrogate_sd > 0 else np.nan
                )
                subsample_rows.append({
                    "mouse_id": mouse_id,
                    "brain_area": brain_area,
                    "subsample_id": subsample_id,
                    "bin_width_ms": bin_width_ms,
                    "n_trials": len(trials_by_mouse[mouse_id]),
                    "n_time_bins": selected_raster.shape[0],
                    "total_analyzed_time_sec": selected_raster.shape[0] * bin_width_ms / 1000.0,
                    "n_available_units": n_available_units,
                    "n_sampled_units": n_matched_units,
                    "selected_cluster_ids": ";".join(map(str, selected_cluster_ids)),
                    "n_active_unit_bins": int(selected_raster.sum()),
                    "spike_bin_density": float(selected_raster.mean()),
                    "packed_bytes": int(packed_bytes),
                    "observed_compressed_bytes": int(observed_bytes),
                    "observed_compression_ratio": observed_bytes / packed_bytes,
                    "mean_surrogate_compressed_bytes": surrogate_mean,
                    "sd_surrogate_compressed_bytes": surrogate_sd,
                    "normalized_lz_complexity": normalized_lzc,
                    "surrogate_z_score": z_score,
                    "n_rate_surrogates": N_RATE_SURROGATES,
                })
    del rasters_by_group
    pd.DataFrame(subsample_rows).to_csv(subsamples_csv_path, index=False)
    print(
        f"Completed {bin_width_ms} ms bins in "
        f"{time.perf_counter() - width_started:.1f} s; checkpoint saved",
        flush=True,
    )

subsamples_df = pd.DataFrame(subsample_rows)
expected_rows = (
    len(selected_mouse_ids) * len(EXPECTED_BRAIN_AREAS)
    * N_NEURON_SUBSAMPLES * len(BIN_WIDTHS_MS)
)
if len(subsamples_df) != expected_rows:
    raise RuntimeError(f"Expected {expected_rows} subsample rows, found {len(subsamples_df)}")
finite_columns = [
    "spike_bin_density", "observed_compression_ratio",
    "mean_surrogate_compressed_bytes", "normalized_lz_complexity",
]
if not np.isfinite(subsamples_df[finite_columns]).all().all():
    raise ValueError("Non-finite primary complexity result found")
if (subsamples_df["normalized_lz_complexity"] <= 0).any():
    raise ValueError("Normalized complexity must be positive")
subsamples_df.to_csv(subsamples_csv_path, index=False)
print(f"Calculated {len(subsamples_df):,} complexity estimates in {time.perf_counter() - analysis_started:.1f} s.")
display(subsamples_df.head())

mouse_area_bin_df = (
    subsamples_df.groupby(["mouse_id", "brain_area", "bin_width_ms"], as_index=False)
    .agg(
        mean_normalized_lz_complexity=("normalized_lz_complexity", "mean"),
        sd_normalized_lz_complexity=("normalized_lz_complexity", "std"),
        mean_observed_compression_ratio=("observed_compression_ratio", "mean"),
        mean_surrogate_compressed_bytes=("mean_surrogate_compressed_bytes", "mean"),
        mean_spike_bin_density=("spike_bin_density", "mean"),
        n_trials=("n_trials", "first"),
        n_time_bins=("n_time_bins", "first"),
        total_analyzed_time_sec=("total_analyzed_time_sec", "first"),
        n_available_units=("n_available_units", "first"),
        n_sampled_units=("n_sampled_units", "first"),
        n_subsamples=("subsample_id", "size"),
        n_rate_surrogates=("n_rate_surrogates", "first"),
    )
)
expected_mouse_rows = len(selected_mouse_ids) * len(EXPECTED_BRAIN_AREAS) * len(BIN_WIDTHS_MS)
if len(mouse_area_bin_df) != expected_mouse_rows:
    raise RuntimeError("Mouse-area-bin aggregation has an unexpected number of rows")
if not (mouse_area_bin_df.groupby(["mouse_id", "bin_width_ms"])["n_sampled_units"].nunique() == 1).all():
    raise RuntimeError("Neuron count was not matched across areas within mouse")
mouse_area_bin_df.to_csv(mouse_area_bin_csv_path, index=False)

area_summary_df = (
    mouse_area_bin_df.groupby(["brain_area", "bin_width_ms"], as_index=False)
    .agg(
        mean_normalized_lz_complexity=("mean_normalized_lz_complexity", "mean"),
        sd_normalized_lz_complexity=("mean_normalized_lz_complexity", "std"),
        mean_observed_compression_ratio=("mean_observed_compression_ratio", "mean"),
        mean_spike_bin_density=("mean_spike_bin_density", "mean"),
        n_mice=("mouse_id", "nunique"),
    )
)
area_summary_df["sem_normalized_lz_complexity"] = (
    area_summary_df["sd_normalized_lz_complexity"] / np.sqrt(area_summary_df["n_mice"])
)
area_summary_df["ci95_low"] = (
    area_summary_df["mean_normalized_lz_complexity"]
    - 1.96 * area_summary_df["sem_normalized_lz_complexity"]
)
area_summary_df["ci95_high"] = (
    area_summary_df["mean_normalized_lz_complexity"]
    + 1.96 * area_summary_df["sem_normalized_lz_complexity"]
)
area_summary_df.to_csv(area_summary_csv_path, index=False)
with config_path.open("w", encoding="utf-8") as handle:
    json.dump(RUN_CONFIG, handle, indent=2)

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, len(BIN_WIDTHS_MS), figsize=(5.0 * len(BIN_WIDTHS_MS), 4.8), sharey=True)
axes = np.atleast_1d(axes)
for ax, bin_width_ms in zip(axes, BIN_WIDTHS_MS):
    plot_df = mouse_area_bin_df.loc[mouse_area_bin_df["bin_width_ms"] == bin_width_ms]
    summary_df = area_summary_df.loc[area_summary_df["bin_width_ms"] == bin_width_ms]
    for _, mouse_df in plot_df.groupby("mouse_id", sort=True):
        mouse_df = mouse_df.sort_values("brain_area")
        ax.plot(
            mouse_df["brain_area"], mouse_df["mean_normalized_lz_complexity"],
            color="0.78", marker="o", markersize=3, linewidth=0.8, alpha=0.8,
        )
    ax.errorbar(
        summary_df["brain_area"], summary_df["mean_normalized_lz_complexity"],
        yerr=1.96 * summary_df["sem_normalized_lz_complexity"],
        fmt="o-", color="black", markersize=6, capsize=4, linewidth=1.6,
    )
    ax.axhline(1.0, color="0.45", linestyle="--", linewidth=1.0)
    ax.set_xticks(EXPECTED_BRAIN_AREAS, [f"Area {area}" for area in EXPECTED_BRAIN_AREAS])
    ax.set_xlabel("Brain area")
    ax.set_title(f"{bin_width_ms} ms bins" + (" (primary)" if bin_width_ms == PRIMARY_BIN_WIDTH_MS else ""))
    sns.despine(ax=ax)
axes[0].set_ylabel("Rate-normalized LZ compression complexity")
fig.suptitle("Stimulus-to-trial-end population spiking complexity", y=1.02)
fig.tight_layout()
fig.savefig(figure_path, dpi=200, bbox_inches="tight")
plt.close(fig)

display(mouse_area_bin_df)
display(area_summary_df)
print(f"Saved Q15 script results to {q_path}")


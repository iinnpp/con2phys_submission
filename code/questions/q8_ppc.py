#!/usr/bin/env python
"""Estimate 4–10 Hz pairwise phase consistency between units and same-area LFP.

Run in the con2phys environment. Pass --data-dir for the directory containing
the numbered mouse folders. q8_ppc_stats.R and q8_ppc_plots.ipynb read the output.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import neurodsp
import numpy as np
import pandas as pd
from neurodsp.filt import filter_signal
from neurodsp.timefrequency import phase_by_time
from scipy.fft import next_fast_len


ROOT = Path(__file__).resolve().parents[2]
AREAS = (1, 2, 3)
FS = 500
BAND = (4, 10)
FILTER_ORDER = 3
EDGE_SAMPLES = FS  # Exclude one second at both ends of each finite LFP segment.
MIN_SPIKES = 31


def median_lfp(lfp):
    """Take the channel median in time blocks to limit memory use."""
    median = np.empty(lfp.shape[1], dtype=np.float64)
    for start in range(0, median.size, 100_000):
        window = slice(start, start + 100_000)
        block = lfp[:, window]
        finite = np.isfinite(block)
        if finite.all():
            median[window] = np.median(block, axis=0)
        else:
            has_data = finite.any(axis=0)
            block_median = median[window]
            block_median[~has_data] = np.nan
            block_median[has_data] = np.nanmedian(
                np.where(finite[:, has_data], block[:, has_data], np.nan), axis=0)
    return median


def lfp_phase(trace, keep_filtered=False):
    """Return 4–10 Hz phase, leaving gaps and filter edges as NaN."""
    phase = np.full(trace.size, np.nan)
    filtered_trace = np.full(trace.size, np.nan) if keep_filtered else None
    finite = np.isfinite(trace)
    bounds = np.flatnonzero(np.diff(finite, prepend=False, append=False))
    for start, stop in bounds.reshape(-1, 2):
        segment = trace[start:stop]
        if segment.size <= 2 * EDGE_SAMPLES or np.ptp(segment) == 0:
            continue
        # The NeuroDSP response diagnostic takes log10(0) at an exact zero of the filter.
        with np.errstate(divide="ignore"):
            filtered = filter_signal(segment, FS, "bandpass", BAND, filter_type="iir",
                                     butterworth_order=FILTER_ORDER)
        # Reflect the filtered signal for a fast FFT and to soften the Hilbert boundary.
        fft_size = next_fast_len(segment.size + 2 * EDGE_SAMPLES)
        right_pad = fft_size - segment.size - EDGE_SAMPLES
        padded = np.pad(filtered, (EDGE_SAMPLES, right_pad), mode="reflect")
        padded_phase = phase_by_time(padded, FS)
        interior = slice(start + EDGE_SAMPLES, stop - EDGE_SAMPLES)
        phase[interior] = padded_phase[2 * EDGE_SAMPLES:segment.size]
        if keep_filtered:
            filtered_trace[interior] = filtered[EDGE_SAMPLES:-EDGE_SAMPLES]
    return phase, filtered_trace


def score_spike_phases(phases, unit_indices, n_units):
    """Toolbox PPC via the squared resultant, excluding the N self-pairs."""
    valid = np.isfinite(phases)
    phases = phases[valid]
    unit_indices = unit_indices[valid]
    counts = np.bincount(unit_indices, minlength=n_units)
    cos_sum = np.bincount(unit_indices, weights=np.cos(phases), minlength=n_units)
    sin_sum = np.bincount(unit_indices, weights=np.sin(phases), minlength=n_units)
    enough = counts >= MIN_SPIKES
    n = counts[enough].astype(float)
    ppc = np.full(n_units, np.nan)
    ppc[enough] = (cos_sum[enough]**2 + sin_sum[enough]**2 - n) / (n * (n - 1))
    return ppc, counts


def load_spikes(folder):
    """Join spike assignments to the per-unit metadata in cluster-ID order."""
    metadata = np.load(folder / "brain_area.npy", allow_pickle=True).item()
    order = np.argsort(metadata["cluster_id"])
    cluster_ids = np.asarray(metadata["cluster_id"])[order]
    areas = np.asarray(metadata["brain_area"])[order]
    times = np.load(folder / "spikes.npy", mmap_mode="r")
    assignments = np.load(folder / "clusters.npy", mmap_mode="r")
    if times.ndim != 1 or times.shape != assignments.shape or not np.isfinite(times).all():
        raise ValueError(f"Invalid spike arrays in {folder}")
    if np.any(np.diff(times) < 0):
        raise ValueError(f"Spike times must be sorted in {folder}")
    if len(np.unique(cluster_ids)) != len(cluster_ids) or not np.isin(areas, AREAS).all():
        raise ValueError(f"Invalid unit metadata in {folder}")
    if not np.array_equal(cluster_ids, np.unique(assignments)):
        raise ValueError(f"Spike assignments and metadata differ in {folder}")
    unit_indices = np.searchsorted(cluster_ids, assignments)
    return times, unit_indices, cluster_ids, areas


def alignment_example(mouse, area, unit, spike_times, sample_indices, filtered, phase):
    """Export a two-second example with at least a few aligned spikes when available."""
    ok = np.isfinite(phase[sample_indices])
    if not ok.any():
        return None
    window_size = 2 * FS
    # Pick a dense two-second window from the first two minutes of the trace.
    candidates = sample_indices[ok & (sample_indices < 120 * FS)]
    if not candidates.size:
        candidates = sample_indices[ok][:1]
    anchors = np.maximum(candidates - FS // 2, EDGE_SAMPLES)
    anchors = np.minimum(anchors, len(phase) - EDGE_SAMPLES - window_size)
    valid_indices = sample_indices[ok]
    counts = (np.searchsorted(valid_indices, anchors + window_size)
              - np.searchsorted(valid_indices, anchors))
    start = anchors[np.argmax(counts)]
    samples = np.arange(start, start + window_size)
    show_spike = ok & (sample_indices >= start) & (sample_indices < start + window_size)
    spike_samples = sample_indices[show_spike]
    labels = dict(mouse_id=mouse, brain_area=area, cluster_id=unit)
    return (
        pd.DataFrame(dict(**labels, sample_index=samples, time_s=samples / FS,
                          filtered_lfp=filtered[samples], phase_rad=phase[samples])),
        pd.DataFrame(dict(**labels, time_s=spike_times[show_spike], sample_index=spike_samples,
                          sample_time_s=spike_samples / FS, filtered_lfp=filtered[spike_samples],
                          phase_rad=phase[spike_samples])),
    )


def analyze_mouse(folder):
    mouse = int(folder.name)
    times, indices, cluster_ids, areas = load_spikes(folder)
    lfps = [np.load(folder / f"lfp_{area}.npy", mmap_mode="r") for area in AREAS]
    if any(lfp.ndim != 2 or not lfp.shape[0] or lfp.shape[1] <= 4 * FS for lfp in lfps):
        raise ValueError(f"Invalid LFP shape in {folder}")
    if len({lfp.shape[1] for lfp in lfps}) != 1:
        raise ValueError(f"Unequal LFP sample counts in {folder}")

    last_lfp_time = (lfps[0].shape[1] - 1) / FS
    before = times < 0
    after = times > last_lfp_time
    in_range = ~(before | after)
    n_units = cluster_ids.size
    spike_counts = np.bincount(indices, minlength=n_units)
    before_counts = np.bincount(indices[before], minlength=n_units)
    after_counts = np.bincount(indices[after], minlength=n_units)
    rows = []
    example = None
    for area, lfp in zip(AREAS, lfps):
        area_units = np.flatnonzero(areas == area)
        area_spikes = in_range & (areas[indices] == area)
        spike_times = times[area_spikes]
        area_indices = np.searchsorted(area_units, indices[area_spikes])
        # The LFP grid starts at time zero. Use nearest sample, with ties rounding up.
        spike_samples = np.floor(FS * spike_times + 0.5).astype(np.intp)
        keep_example = example is None
        phase, filtered = lfp_phase(median_lfp(lfp), keep_filtered=keep_example)
        primary, usable = score_spike_phases(phase[spike_samples], area_indices, area_units.size)
        if keep_example and np.isfinite(primary).any():
            unit_index = int(np.argmax(usable))
            selected = area_indices == unit_index
            example = alignment_example(mouse, area, cluster_ids[area_units[unit_index]],
                                        spike_times[selected], spike_samples[selected], filtered, phase)
        del phase, filtered

        channel_scores = np.empty((lfp.shape[0], area_units.size))
        for channel, voltage in enumerate(lfp):
            phase, _ = lfp_phase(voltage)
            channel_scores[channel], _ = score_spike_phases(
                phase[spike_samples], area_indices, area_units.size)

        for i, unit in enumerate(area_units):
            valid_channel_scores = channel_scores[:, i][np.isfinite(channel_scores[:, i])]
            rows.append(dict(
                mouse_id=mouse, brain_area=area, cluster_id=cluster_ids[unit],
                n_lfp_channels=lfp.shape[0], n_lfp_samples=lfp.shape[1],
                n_spikes=spike_counts[unit], n_before_lfp=before_counts[unit],
                n_after_lfp=after_counts[unit], n_median_trace_spikes=usable[i],
                n_valid_channel_scores=valid_channel_scores.size, ppc_median_trace=primary[i],
                ppc_median_channel=(np.median(valid_channel_scores)
                                    if valid_channel_scores.size else np.nan),
            ))
    return rows, example


def main():
    project = Path(os.environ.get("CON2PHYS_PROJECT_DIR", "E:/con2phys_submission"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project / "dataset_download/py-files")
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get(
        "Q8_OUTPUT_DIR", ROOT / "results/q8")))
    parser.add_argument("--mice", type=int, nargs="+", help="Optional subset for a pilot run")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Maximum number of recordings to process concurrently")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    folders = {int(path.name): path for path in args.data_dir.iterdir()
               if path.is_dir() and path.name.isdigit()}
    requested = set(args.mice) if args.mice else set(range(1, 19))
    if missing := requested - folders.keys():
        parser.error(f"Missing mouse folders under {args.data_dir}: {sorted(missing)}")

    started = time.perf_counter()
    results = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        jobs = {executor.submit(analyze_mouse, folders[mouse]): mouse for mouse in sorted(requested)}
        for job in as_completed(jobs):
            mouse = jobs[job]
            results[mouse] = job.result()
            elapsed = time.perf_counter() - started
            print(f"Mouse {mouse:2d}: {len(results[mouse][0]):3d} units ({elapsed:.0f} s elapsed)",
                  flush=True)
    rows = [row for mouse in sorted(results) for row in results[mouse][0]]
    example = next((results[mouse][1] for mouse in sorted(results) if results[mouse][1] is not None), None)
    if example is None:
        raise ValueError("No units with enough usable spikes for PPC")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in zip(("units", "example_lfp", "example_spikes"), (pd.DataFrame(rows), *example)):
        path = args.output_dir / f"q8_ppc_{name}.csv"
        staged = path.with_suffix(".csv.tmp")
        frame.to_csv(staged, index=False)
        staged.replace(path)
    config = dict(data_dir=str(args.data_dir.resolve()), mice=sorted(requested),
                  fs_hz=FS, lfp_start_s=0, band_hz=BAND, filter_order=FILTER_ORDER,
                  neurodsp_version=neurodsp.__version__, min_spikes=MIN_SPIKES,
                  edge_exclusion_s=EDGE_SAMPLES / FS,
                  phase_padding="reflect at least one second per edge, to a fast FFT length",
                  sample_alignment="nearest sample, after excluding times outside the LFP sample times",
                  primary="PPC against the median voltage trace of the unit's brain area",
                  sensitivity="median PPC across individual same-area channels with valid scores")
    (args.output_dir / "q8_ppc_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

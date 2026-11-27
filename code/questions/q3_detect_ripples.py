#!/usr/bin/env python
"""Detect per-channel ripple candidates using smoothed ripple-band power."""

import argparse
import csv
import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from threading import Lock

import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import butter, sosfiltfilt


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path("E:/con2phys_submission/dataset_download/py-files")
RESULTS_DIR = ROOT / "results" / "q3" / "psd_ripples"
MICE = tuple(range(1, 19))
AREAS = (1, 2, 3)
FS = 500
PREP_BAND = (0.1, 200.0)
RIPPLE_BAND = (100.0, 180.0)
FILTER_ORDER = 3
BOXCAR_SAMPLES = 5  # 10 ms at 500 Hz, closest odd-length match to MATLAB's 11 ms.
BOXCAR = np.full(BOXCAR_SAMPLES, 1.0 / BOXCAR_SAMPLES)
START_SD = 3.0
PEAK_SD = 6.0
MERGE_GAP_MS = 30
MIN_DURATION_MS = 20
MAX_DURATION_MS = 100
FILE_READ_LOCK = Lock()


def ripple_filter():
    """Combine both Butterworth bandpasses for one forward-backward pass."""
    return np.vstack([
        butter(FILTER_ORDER, band, btype="bandpass", fs=FS, output="sos")
        for band in (PREP_BAND, RIPPLE_BAND)
    ])


def intervals_over_threshold(power, threshold):
    """Return complete half-open runs, joining gaps strictly below 30 ms."""
    edges = np.flatnonzero(np.diff(power > threshold, prepend=False, append=False))
    starts, stops = edges[::2], edges[1::2]
    complete = (starts > 0) & (stops < power.size)
    starts, stops = starts[complete], stops[complete]
    if starts.size == 0:
        return starts, stops
    separate = (starts[1:] - stops[:-1]) * 1000 >= MERGE_GAP_MS * FS
    return starts[np.r_[True, separate]], stops[np.r_[separate, True]]


def detect_from_power(power, mean, sd):
    """Return (start, exclusive stop, peak, peak z) in sample coordinates."""
    starts, stops = intervals_over_threshold(power, mean + START_SD * sd)
    durations = (stops - starts) * 1000
    valid = (durations >= MIN_DURATION_MS * FS) & (durations <= MAX_DURATION_MS * FS)
    peak_threshold = mean + PEAK_SD * sd
    for start, stop in zip(starts[valid], stops[valid]):
        peak = start + np.argmax(power[start:stop])
        if power[peak] > peak_threshold:
            yield start, stop, peak, (power[peak] - mean) / sd


def detect_recording(job, sos):
    mouse, area, path = job
    # Sequential bulk reads from the recording drive while other workers filter.
    with FILE_READ_LOCK:
        lfp = np.load(path, allow_pickle=False)
    if lfp.ndim != 2 or lfp.shape[1] < FS:
        raise ValueError(f"Expected channel-by-time LFP of at least one second: {path}")
    duration_s = lfp.shape[1] / FS
    rates, events = [], []
    for channel, trace in enumerate(lfp, start=1):
        key = (mouse, area, channel)
        if not np.isfinite(trace).all() or np.ptp(trace) == 0:
            rates.append((*key, "invalid_signal", duration_s, "", "", "", ""))
            continue

        filtered = sosfiltfilt(sos, trace)
        # Match MATLAB's conv(signal.^2, window, 'same') with zero padding.
        filtered *= 1e6
        power = convolve1d(np.square(filtered), BOXCAR, mode="constant", cval=0.0)
        mean = float(np.mean(power))
        sd = float(np.std(power, ddof=1))
        if not np.isfinite(sd) or sd == 0:
            rates.append((*key, "invalid_power", duration_s, "", "", "", ""))
            continue
        count = 0
        for start, stop, peak, peak_sd in detect_from_power(power, mean, sd):
            trough = start + np.argmin(filtered[start:stop])
            events.append((*key, int(start), int(stop), int(peak), start / FS,
                           stop / FS, peak / FS, (stop - start) * 1000 / FS,
                           float(power[peak]), float(peak_sd), int(trough),
                           trough / FS, float(filtered[trough])))
            count += 1
        rates.append((*key, "ok", duration_s, mean, sd, count, count * 60 / duration_s))
    return rates, events


def run(data_dir, output_dir, mice, workers):
    jobs = [(mouse, area, data_dir / str(mouse) / f"lfp_{area}.npy")
            for mouse in mice for area in AREAS]
    for _, _, path in jobs:
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    rate_path = output_dir / "q3_ripple_channel_rates.csv"
    event_path = output_dir / "q3_ripple_events.csv.gz"
    rate_partial = rate_path.with_suffix(".csv.partial")
    event_partial = event_path.with_suffix(".gz.partial")
    with (rate_partial.open("w", newline="", encoding="utf-8") as rate_file,
          gzip.open(event_partial, "wt", newline="", encoding="utf-8", compresslevel=4) as event_file,
          ThreadPoolExecutor(max_workers=workers) as executor):
        rate_writer = csv.writer(rate_file)
        event_writer = csv.writer(event_file)
        rate_writer.writerow(["mouse_id", "brain_area", "channel_row", "status", "duration_s",
                              "mean_power_uv2", "sd_power_uv2", "n_candidates", "ripples_per_min"])
        event_writer.writerow(["mouse_id", "brain_area", "channel_row", "start_sample",
                               "stop_sample_exclusive", "power_peak_sample", "start_s",
                               "stop_s_exclusive", "power_peak_s", "duration_ms", "peak_power_uv2",
                               "peak_power_sd", "lfp_trough_sample", "lfp_trough_s",
                               "lfp_trough_uv"])
        results = executor.map(partial(detect_recording, sos=ripple_filter()), jobs)
        for index, ((mouse, area, _), (rates, events)) in enumerate(zip(jobs, results), 1):
            rate_writer.writerows(rates)
            rate_file.flush()
            event_writer.writerows(events)
            valid = sum(row[3] == "ok" for row in rates)
            print(f"[{index:02d}/{len(jobs)}] Mouse {mouse:02d}, area {area}: "
                  f"{valid}/{len(rates)} channels, {len(events)} candidates", flush=True)
    event_partial.replace(event_path)
    rate_partial.replace(rate_path)

    config = {
        "method_doi": "10.1038/s41467-024-44983-z",
        "method": "adapted first power-threshold detector",
        "historical_implementation": "bu_ripple_detection/get_ripples.m",
        "event_label": "putative ripple candidate",
        "input_data": str(data_dir.resolve()), "mouse_ids": list(mice), "areas": AREAS,
        "sampling_rate_hz": FS, "prep_band_hz": PREP_BAND, "ripple_band_hz": RIPPLE_BAND,
        "filter_order_per_bandpass": FILTER_ORDER,
        "filter_application": "cascaded SOS Butterworth, forward-backward, odd padding",
        "spatial_reference": "supplied reference, each channel independently",
        "power": "square the narrow-band trace in microvolts, then centered boxcar mean",
        "boxcar_samples": BOXCAR_SAMPLES, "boxcar_ms": BOXCAR_SAMPLES * 1000 / FS,
        "boxcar_padding": "zeros",
        "drop_threshold_runs_touching_recording_edges": True,
        "start_strictly_above_sd": START_SD, "peak_strictly_above_sd": PEAK_SD,
        "merge_gap_strictly_below_ms": MERGE_GAP_MS,
        "duration_inclusive_ms": [MIN_DURATION_MS, MAX_DURATION_MS],
        "duration_rule": "spontaneous, bounds adapted from MATLAB at the original sample rate",
        "baseline": "full-recording mean and sample SD of smoothed power within each channel",
        "rate_denominator": "full recording in minutes",
        "row_numbering": "one-based, deepest to superficial",
        "event_samples": "zero-based indices, stop is exclusive",
        "lfp_trough": "minimum of 100–180 Hz filtered trace within the detected interval",
    }
    (output_dir / "q3_ripple_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--mice", type=int, nargs="+", choices=MICE, default=MICE)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1 or len(set(args.mice)) != len(args.mice):
        parser.error("Use a positive worker count and unique mouse IDs.")
    run(args.data_dir, args.output_dir, args.mice, args.workers)


if __name__ == "__main__":
    main()

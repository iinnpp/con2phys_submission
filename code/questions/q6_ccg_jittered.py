#!/usr/bin/env python
"""Estimate jitter-corrected lead–lag asymmetry for the Q6 spike populations."""

import argparse
import json
import math
import os
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import irfft, next_fast_len, rfft

from q6_glm import AREAS, CANDIDATES, ROOT, matched_draws, population_counts, read_mice


BIN_MS = 5
PRIMARY = (200, 50)  # jitter window, maximum mirrored lag
SETTINGS = (PRIMARY, (200, 100), (100, 50), (400, 50))
MAX_LAG_BINS = max(lag for _, lag in SETTINGS) // BIN_MS
LAGS = np.arange(-MAX_LAG_BINS, MAX_LAG_BINS + 1)
JITTER_BINS = sorted({jitter // BIN_MS for jitter, _ in SETTINGS})
TRIM_MULTIPLE = math.lcm(*JITTER_BINS)


def observed_ccgs(counts, lags):
    """Count source(t) × target(t+lag) over valid bin pairs, with zero padding."""
    padding = next_fast_len(len(counts) + max(abs(lags)), real=True)
    spectra = rfft(counts, n=padding, axis=0)
    return {
        (a, b): np.rint(irfft(spectra[:, a].conj() * spectra[:, b], n=padding)[lags])
        for a, b in combinations(range(counts.shape[1]), 2)
    }


def interval_jitter_ccg(source, target, window_bins, lags):
    """Exact cross-area expectation under independent uniform interval jitter.

    Source and target are per-window expected counts in each fine bin. At an
    offset of k bins, each window contributes window_bins-k same-window pairs
    and each pair of adjacent windows contributes k cross-window pairs.
    """
    if np.max(np.abs(lags)) > window_bins:
        raise ValueError("Lag exceeds the jitter window")
    same = source @ target
    forward = source[:-1] @ target[1:]
    backward = source[1:] @ target[:-1]
    return (window_bins - np.abs(lags)) * same + np.abs(lags) * np.where(lags >= 0, forward, backward)


def score_draw(counts):
    observed = observed_ccgs(counts, LAGS)
    centered_norms = np.sqrt(counts.var(axis=0, dtype=float) * len(counts))
    if np.any(centered_norms == 0):
        raise ValueError("A selected population has no binned count variance")
    expectation = {
        window: counts.reshape(-1, window, len(AREAS)).mean(axis=1) for window in JITTER_BINS
    }
    for (a, b), observed_pairs in observed.items():
        norm = centered_norms[a] * centered_norms[b]
        for window, mean_counts in expectation.items():
            expected_pairs = interval_jitter_ccg(mean_counts[:, a], mean_counts[:, b], window, LAGS)
            corrected = (observed_pairs - expected_pairs) / norm
            for source, target, order in ((a, b, slice(None)), (b, a, slice(None, None, -1))):
                yield AREAS[source], AREAS[target], window * BIN_MS, {
                    "observed_pairs": observed_pairs[order],
                    "expected_pairs": expected_pairs[order],
                    "corrected_ccg": corrected[order],
                }


def save_results(output, scores, curves, units, args, elapsed_seconds):
    draws = pd.DataFrame(scores)
    keys = ["mouse_id", "bin_ms", "jitter_ms", "max_lag_ms", "primary", "source", "target"]
    if draws.duplicated(keys + ["subsample"]).any():
        raise RuntimeError("Duplicate subsample asymmetry scores")
    mice = draws.groupby(keys, as_index=False).agg(
        asymmetry=("asymmetry", "mean"), positive_mean=("positive_mean", "mean"),
        negative_mean=("negative_mean", "mean"), n_subsamples=("subsample", "size"),
    )
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in {"subsample_scores": draws, "mouse_scores": mice,
                        "mouse_correlograms": pd.concat(curves, ignore_index=True),
                        "unit_counts": pd.DataFrame(units)}.items():
        if name != "unit_counts":
            frame["candidate"] = [(a, b) in CANDIDATES for a, b in zip(frame.source, frame.target)]
        path = output / f"q6_ccg_{name}.csv"
        staged = path.with_suffix(".csv.tmp")
        frame.to_csv(staged, index=False)
        staged.replace(path)
    config = {
        "spikes_csv": str(args.spikes_csv.resolve()), "bin_ms": BIN_MS,
        "settings_jitter_ms_max_lag_ms": SETTINGS, "primary_jitter_ms_max_lag_ms": PRIMARY,
        "jitter": "exact mean over independent uniform within-interval jitter, conditional on each area's interval counts",
        "trim": "keep complete jitter windows at every tested width; common 400 ms multiple",
        "normalization": "sqrt(product of the centered sum of squares of both population bin-count series)",
        "primary_metric": "mean corrected CCG at +5 to +50 ms minus mean at mirrored negative lags",
        "lag_convention": "positive lag means the target follows the source; zero lag excluded from score",
        "n_subsamples": args.subsamples, "seed": args.seed,
        "mouse_ids": sorted(mice.mouse_id.unique().tolist()), "elapsed_seconds": round(elapsed_seconds, 1),
    }
    (output / "q6_ccg_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def main():
    project = Path(os.environ.get("CON2PHYS_PROJECT_DIR", "E:/con2phys_submission"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spikes-csv", type=Path, default=project / "dataset/full_sua_dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get(
        "Q6_CCG_OUTPUT_DIR", ROOT / "results/q6/ccg_jittered")))
    parser.add_argument("--mice", type=int, nargs="+", help="Optional mouse IDs for a pilot run")
    parser.add_argument("--subsamples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.subsamples < 1 or args.seed < 0:
        parser.error("Need at least one subsample and a nonnegative seed")

    scores, curves, unit_rows = [], [], []
    started = time.perf_counter()
    for mouse, units, start, duration in read_mice(args.spikes_csv, set(args.mice) if args.mice else None):
        draws = matched_draws(units, mouse, args.subsamples, args.seed)
        unit_rows.extend(dict(mouse_id=mouse, brain_area=area, available_units=len(units[area]),
                              matched_units=len(draws[0][area]), duration_sec=duration) for area in AREAS)
        curve_rows = []
        for sample, draw in enumerate(draws, start=1):
            counts = population_counts(units, draw, start, duration, BIN_MS)
            counts = counts[:len(counts) - len(counts) % TRIM_MULTIPLE]
            if len(counts) <= MAX_LAG_BINS:
                raise ValueError(f"Mouse {mouse} has insufficient recording duration")
            for source, target, jitter_ms, values in score_draw(counts):
                curve_rows.append(pd.DataFrame(dict(mouse_id=mouse, source=source, target=target,
                                                    jitter_ms=jitter_ms, lag_ms=BIN_MS * LAGS, **values)))
                for jitter, max_lag in SETTINGS:
                    if jitter != jitter_ms:
                        continue
                    lag_ms = BIN_MS * LAGS
                    corrected = values["corrected_ccg"]
                    positive = corrected[(lag_ms > 0) & (lag_ms <= max_lag)].mean()
                    negative = corrected[(lag_ms < 0) & (lag_ms >= -max_lag)].mean()
                    scores.append(dict(mouse_id=mouse, source=source, target=target, subsample=sample,
                                       jitter_ms=jitter_ms, bin_ms=BIN_MS, max_lag_ms=max_lag,
                                       primary=(jitter_ms, max_lag) == PRIMARY, n_recording_bins=len(counts),
                                       positive_mean=positive, negative_mean=negative,
                                       asymmetry=positive - negative))
        curves.append(pd.concat(curve_rows).groupby(
            ["mouse_id", "source", "target", "jitter_ms", "lag_ms"], as_index=False).mean())
        print(f"Mouse {mouse}: completed {len(draws)} draws ({time.perf_counter() - started:.0f} s elapsed)", flush=True)
    if not scores:
        raise RuntimeError("No mice were analyzed")
    save_results(args.output_dir, scores, curves, unit_rows, args, time.perf_counter() - started)
    print(f"Wrote jitter-corrected CCG results to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()

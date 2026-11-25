#!/usr/bin/env python
"""Estimate held-out conditional transfer entropy for Q6 population spikes."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from q6_glm import (
    AREAS, CANDIDATES, PRIMARY, ROOT, SETTINGS, SLOW_LAGS_MS, MIN_SHIFT_MS,
    SHIFT_BLOCK_MS, matched_draws, population_counts, read_mice, shifted_history,
)


RECENT_MS = 20
STATE_LEVELS = 3
AREA_STATES = STATE_LEVELS**2
CONTEXT_STATES = AREA_STATES**2 * STATE_LEVELS
TARGET_CAP = 5
TARGET_STATES = TARGET_CAP + 1
CONTEXT_PRIOR = 12.0
SOURCE_PRIOR = 24.0


def fold_counts(counts, bin_ms, history_ms, n_folds):
    """Sum separate 0–20 ms, 20–H ms and 1–5 s past windows within each fold."""
    edges = np.array([0, RECENT_MS, history_ms, *SLOW_LAGS_MS]) // bin_ms
    warmup = edges[-1]
    for block in np.array_split(counts, n_folds):
        if len(block) <= warmup:
            raise ValueError("Time block is too short for the slow conditioning window")
        cumulative = np.empty((len(block) + 1, len(AREAS)), dtype=np.int64)
        cumulative[0] = 0
        np.cumsum(block, axis=0, dtype=np.int64, out=cumulative[1:])
        rows = np.arange(warmup, len(block))
        histories = np.stack([cumulative[rows - edges[i]] - cumulative[rows - edges[j]]
                              for i, j in ((0, 1), (1, 2), (3, 4))], axis=2)
        yield block[warmup:], histories


def encode_history(histories, thresholds):
    """Threshold each window independently without splitting ties on integer counts."""
    low, high = thresholds
    states = (histories > low).astype(np.int32) + (histories > high)
    return STATE_LEVELS * states[:, :, 0] + states[:, :, 1], states[:, :, 2]


def frequency_table(response, context, source):
    indices = TARGET_STATES * (AREA_STATES * context + source) + response
    return np.bincount(indices, minlength=CONTEXT_STATES * AREA_STATES * TARGET_STATES).reshape(
        CONTEXT_STATES, AREA_STATES, TARGET_STATES)


def score_table(train, test):
    """Score the nested empirical conditional distributions on held-out bins."""
    context = train.sum(axis=1)
    marginal = context.sum(axis=0) + 0.5
    marginal /= marginal.sum()
    reduced = (context + CONTEXT_PRIOR * marginal) / (context.sum(axis=1, keepdims=True) + CONTEXT_PRIOR)
    source_counts = train.sum(axis=2, keepdims=True)
    full = (train + SOURCE_PRIOR * reduced[:, None, :]) / (source_counts + SOURCE_PRIOR)
    heldout_bits = np.sum(test * (np.log2(full) - np.log2(reduced[:, None, :])))
    seen_context_bins = test.sum(axis=2)[source_counts[:, :, 0] > 0].sum()
    return float(heldout_bits), int(seen_context_bins)


def score_setting(counts, bin_ms, history_ms, n_folds, rng, diagnostic):
    folds = list(fold_counts(counts, bin_ms, history_ms, n_folds))
    for held_out, (y_test, histories_test) in enumerate(folds):
        y_train = np.concatenate([y for i, (y, _) in enumerate(folds) if i != held_out])
        histories_train = np.concatenate([h for i, (_, h) in enumerate(folds) if i != held_out])
        thresholds = np.quantile(histories_train, [1 / 3, 2 / 3], axis=0, method="inverted_cdf")
        area_train, slow_train = encode_history(histories_train, thresholds)
        area_test, slow_test = encode_history(histories_test, thresholds)
        if diagnostic:
            # Roll within each original contiguous fold, independently of the other folds.
            lengths = [len(y) for i, (y, _) in enumerate(folds) if i != held_out]
            source_train = np.concatenate([
                shifted_history(part, bin_ms, rng)
                for part in np.split(area_train, np.cumsum(lengths)[:-1])
            ])
            source_test = shifted_history(area_test, bin_ms, rng)

        for source, target in ((a, b) for a in AREAS for b in AREAS if a != b):
            third = next(area for area in AREAS if area not in (source, target))
            a, b, c = source - 1, target - 1, third - 1
            context_train = (area_train[:, b] * AREA_STATES + area_train[:, c]) * STATE_LEVELS + slow_train[:, b]
            context_test = (area_test[:, b] * AREA_STATES + area_test[:, c]) * STATE_LEVELS + slow_test[:, b]
            response_train = np.minimum(y_train[:, b], TARGET_CAP)
            response_test = np.minimum(y_test[:, b], TARGET_CAP)
            train = frequency_table(response_train, context_train, area_train[:, a])
            test = frequency_table(response_test, context_test, area_test[:, a])
            bits, coverage = score_table(train, test)
            shifted_bits = np.nan
            if diagnostic:
                shifted_train = frequency_table(response_train, context_train, source_train[:, a])
                shifted_test = frequency_table(response_test, context_test, source_test[:, a])
                shifted_bits, _ = score_table(shifted_train, shifted_test)
            yield dict(source=source, target=target, fold=held_out + 1,
                       n_train_bins=len(y_train), n_test_bins=len(y_test),
                       target_spikes=int(y_test[:, b].sum()), seen_context_bins=coverage,
                       heldout_bits=bits, shifted_heldout_bits=shifted_bits)


def save_results(output, rows, unit_rows, args, settings):
    folds = pd.DataFrame(rows)
    draw_keys = ["mouse_id", "bin_ms", "history_ms", "primary", "subsample", "source", "target"]
    if folds.duplicated(draw_keys + ["fold"]).any():
        raise RuntimeError("Duplicate fold scores")
    draws = folds.groupby(draw_keys, as_index=False).agg(
        n_test_bins=("n_test_bins", "sum"), target_spikes=("target_spikes", "sum"),
        seen_context_bins=("seen_context_bins", "sum"), heldout_bits=("heldout_bits", "sum"),
        shifted_heldout_bits=("shifted_heldout_bits", lambda x: x.sum(min_count=1)),
    )
    if (draws.target_spikes <= 0).any():
        raise RuntimeError("A draw has no held-out target spikes")
    for prefix in ("", "shifted_"):
        draws[f"{prefix}bits_per_bin"] = draws[f"{prefix}heldout_bits"] / draws.n_test_bins
        draws[f"{prefix}bits_per_spike"] = draws[f"{prefix}heldout_bits"] / draws.target_spikes
    draws["seen_context_fraction"] = draws.seen_context_bins / draws.n_test_bins
    mice = draws.groupby([column for column in draw_keys if column != "subsample"], as_index=False).agg(
        bits_per_spike=("bits_per_spike", "mean"), bits_per_bin=("bits_per_bin", "mean"),
        shifted_bits_per_spike=("shifted_bits_per_spike", "mean"),
        shifted_bits_per_bin=("shifted_bits_per_bin", "mean"),
        seen_context_fraction=("seen_context_fraction", "mean"), n_subsamples=("subsample", "size"),
    )
    output.mkdir(parents=True, exist_ok=True)
    tables = {"fold_scores": folds, "subsample_scores": draws, "mouse_scores": mice,
              "unit_counts": pd.DataFrame(unit_rows)}
    for name, frame in tables.items():
        if name != "unit_counts":
            frame["candidate"] = [(a, b) in CANDIDATES for a, b in zip(frame.source, frame.target)]
        path = output / f"q6_cte_{name}.csv"
        staged = path.with_suffix(".csv.tmp")
        frame.to_csv(staged, index=False)
        staged.replace(path)
    config = {
        "spikes_csv": str(args.spikes_csv.resolve()), "settings_ms": settings, "primary_ms": PRIMARY,
        "n_folds": args.folds, "n_subsamples": args.subsamples, "seed": args.seed,
        "target_states": ["0", "1", "2", "3", "4", "5+"],
        "recent_lags_ms": [0, RECENT_MS], "older_lags_ms": [RECENT_MS, "history_ms"],
        "slow_target_lags_ms": SLOW_LAGS_MS,
        "history_encoding": "training-fold count tertiles; integer ties kept together",
        "conditioning": "recent and older target and third-area states plus slow target state",
        "context_prior_count": CONTEXT_PRIOR, "source_prior_count": SOURCE_PRIOR,
        "primary_metric": "held-out conditional log-probability gain / uncapped held-out target spikes",
        "secondary_metric": "held-out conditional log-probability gain per bin",
        "primary_shift_diagnostic": not args.no_shift,
        "shift_block_ms": SHIFT_BLOCK_MS, "minimum_shift_ms": MIN_SHIFT_MS,
        "mouse_ids": sorted(mice.mouse_id.unique().tolist()),
    }
    (output / "q6_cte_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def main():
    project = Path(os.environ.get("CON2PHYS_PROJECT_DIR", "E:/con2phys_submission"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spikes-csv", type=Path, default=project / "dataset/full_sua_dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get(
        "Q6_CTE_OUTPUT_DIR", ROOT / "results/q6/conditional_transfer_entropy")))
    parser.add_argument("--mice", type=int, nargs="+", help="Optional mouse IDs for a pilot run")
    parser.add_argument("--subsamples", type=int, default=5)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--primary-only", action="store_true")
    parser.add_argument("--no-shift", action="store_true")
    args = parser.parse_args()
    if args.subsamples < 1 or args.folds < 2 or args.seed < 0:
        parser.error("Need at least one subsample, two folds, and a nonnegative seed")
    settings = (PRIMARY,) if args.primary_only else SETTINGS
    rows, unit_rows = [], []
    start_time = time.perf_counter()
    for mouse, units, start, duration in read_mice(args.spikes_csv, set(args.mice) if args.mice else None):
        draws = matched_draws(units, mouse, args.subsamples, args.seed)
        unit_rows.extend(dict(mouse_id=mouse, brain_area=area, available_units=len(units[area]),
                              matched_units=len(draws[0][area]), duration_sec=duration) for area in AREAS)
        for sample, draw in enumerate(draws, start=1):
            counts = {b: population_counts(units, draw, start, duration, b) for b in {s[0] for s in settings}}
            for bin_ms, history_ms in settings:
                primary = (bin_ms, history_ms) == PRIMARY
                rng = np.random.default_rng([args.seed, mouse, sample, bin_ms, history_ms])
                results = score_setting(counts[bin_ms], bin_ms, history_ms, args.folds, rng,
                                        primary and not args.no_shift)
                rows.extend(dict(mouse_id=mouse, bin_ms=bin_ms, history_ms=history_ms,
                                 primary=primary, subsample=sample, **score) for score in results)
        print(f"Mouse {mouse}: finished {len(draws)} draws at {len(settings)} settings "
              f"({time.perf_counter() - start_time:.0f} s elapsed)", flush=True)
    if not rows:
        raise RuntimeError("No mice were analyzed")
    save_results(args.output_dir, rows, unit_rows, args, settings)
    print(f"Wrote conditional transfer entropy scores to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()

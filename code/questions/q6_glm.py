#!/usr/bin/env python
"""Estimate Q6 directed population-spiking connectivity by held-out Poisson GLMs."""

import argparse
import csv
import json
import math
import os
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[2]
AREAS = (1, 2, 3)
CANDIDATES = {(1, 2), (3, 2), (3, 1)}
PRIMARY = (10, 100)
SETTINGS = (PRIMARY, (5, 100), (20, 100), (10, 50), (10, 250))
RIDGE = 1e-4
SLOW_LAGS_MS = (1000, 5000)
SHIFT_BLOCK_MS = 20_000
MIN_SHIFT_MS = 2_000


def read_mice(path, selected):
    """Stream the mouse-grouped CSV written by load_spikes.ipynb."""
    csv.field_size_limit(2**31 - 1)
    seen = set()
    with path.open(newline="", encoding="utf-8-sig") as file:
        for mouse, rows in groupby(csv.DictReader(file), lambda row: int(row["mouse_id"])):
            if mouse in seen:
                raise ValueError(f"Mouse {mouse} has noncontiguous rows in {path}")
            seen.add(mouse)
            if selected is not None and mouse not in selected:
                continue
            units = {area: {} for area in AREAS}
            clusters = set()
            durations = set()
            for row in rows:
                area, cluster = int(row["brain_area"]), int(row["cluster_id"])
                spikes = np.fromstring(row["spikes"], sep=";", dtype=float)
                if (cluster in clusters or area not in AREAS or not spikes.size
                        or spikes.size != int(row["n_spikes"]) or not np.isfinite(spikes).all()
                        or np.any(np.diff(spikes) < 0)):
                    raise ValueError(f"Invalid unit in mouse {mouse}, cluster {cluster}")
                clusters.add(cluster)
                units[area][cluster] = spikes
                durations.add(float(row["recording_duration_sec"]))
            if len(durations) != 1 or not all(units.values()):
                raise ValueError(f"Inconsistent duration or missing area in mouse {mouse}")
            duration = durations.pop()
            start = min(spikes[0] for area_units in units.values() for spikes in area_units.values())
            stop = max(spikes[-1] for area_units in units.values() for spikes in area_units.values())
            if not np.isfinite(duration) or duration <= 0 or stop > start + duration + 1e-8:
                raise ValueError(f"Invalid recording window in mouse {mouse}")
            yield mouse, units, start, duration
            if selected is not None and selected <= seen:
                return
    if selected is not None and not selected <= seen:
        raise ValueError(f"Requested mice missing from dataset: {sorted(selected - seen)}")


def matched_draws(units, mouse, n_draws, seed):
    rng = np.random.default_rng([seed, mouse])
    n_matched = min(map(len, units.values()))
    return [
        {area: sorted(rng.choice(sorted(units[area]), n_matched, replace=False).tolist())
         for area in AREAS}
        for _ in range(n_draws)
    ]


def population_counts(units, draw, start, duration, bin_ms):
    bin_sec = bin_ms / 1000
    n_bins = math.floor(duration / bin_sec)
    counts = np.zeros((n_bins, len(AREAS)), dtype=np.int32)
    for col, area in enumerate(AREAS):
        for cluster in draw[area]:
            bins = np.floor((units[area][cluster] - start) / bin_sec).astype(np.int64)
            bins = bins[(bins >= 0) & (bins < n_bins)]
            counts[:, col] += np.bincount(bins, minlength=n_bins).astype(np.int32)
    return counts


def history_edges(bin_ms, history_ms):
    # Keep an immediate lag, then pool neighboring bins to capture slower activity.
    return sorted({0, history_ms, *(edge for edge in (bin_ms, 20, 40, 100)
                                    if edge < history_ms and edge % bin_ms == 0)})


def fold_histories(counts, bin_ms, history_ms, n_folds):
    """Construct only past-count features and restart histories in every time block."""
    edges = history_edges(bin_ms, history_ms)
    warmup = SLOW_LAGS_MS[1] // bin_ms
    for block in np.array_split(counts, n_folds):
        if len(block) <= warmup:
            raise ValueError("Recording too short for requested folds and slow baseline")
        cumulative = np.vstack([np.zeros((1, len(AREAS))), block.cumsum(axis=0, dtype=np.int64)])
        rows = np.arange(warmup, len(block))

        def window(near, far):
            return cumulative[rows - near // bin_ms] - cumulative[rows - far // bin_ms]

        # Column groups always follow area 1, area 2, area 3.
        short = np.stack([window(near, far) for near, far in zip(edges, edges[1:])], axis=2)
        slow = np.log1p(window(*SLOW_LAGS_MS))
        yield block[warmup:], short.reshape(len(rows), -1), slow


def fit_poisson(x_train, y_train, x_test):
    """Fit a ridge-Poisson log-link GLM and return held-out log rates."""
    if y_train.sum() == 0:
        raise ValueError("No target spikes in the training bins")
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0] = 1
    train = np.column_stack([np.ones(len(x_train)), (x_train - mean) / scale])
    observed = train.T @ y_train / len(train)

    def objective(beta):
        eta = train @ beta
        with np.errstate(over="ignore", invalid="ignore"):
            rate = np.exp(eta)
            loss = rate.mean() - observed @ beta + RIDGE * (beta[1:] @ beta[1:]) / 2
            gradient = train.T @ rate / len(train) - observed
        gradient[1:] += RIDGE * beta[1:]
        return loss, gradient

    initial = np.zeros(train.shape[1])
    initial[0] = np.log(y_train.mean())
    fit = minimize(objective, initial, jac=True, method="L-BFGS-B",
                   options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8})
    if not fit.success or not np.isfinite(fit.fun):
        raise RuntimeError(f"Poisson GLM fit failed: {fit.message}")
    weights = fit.x[1:] / scale
    return x_test @ weights + fit.x[0] - mean @ weights


def shifted_history(features, bin_ms, rng):
    """Keep local source history patterns while breaking their exact alignment."""
    n_blocks = max(1, math.ceil(len(features) * bin_ms / SHIFT_BLOCK_MS))
    min_shift = MIN_SHIFT_MS // bin_ms
    shifted = []
    for block in np.array_split(features, n_blocks):
        if len(block) < 2 * min_shift:
            raise ValueError("Time block too short for the circular-shift diagnostic")
        shift = rng.integers(min_shift, len(block) - min_shift + 1)
        shifted.append(np.roll(block, shift, axis=0))
    return np.concatenate(shifted)


def score_setting(counts, bin_ms, history_ms, n_folds, max_train_bins, rng, diagnostic):
    folds = list(fold_histories(counts, bin_ms, history_ms, n_folds))
    n_lags = folds[0][1].shape[1] // len(AREAS)
    grouping = np.repeat(np.array(AREAS), n_lags)
    shifted = [shifted_history(short, bin_ms, rng) for _, short, _ in folds] if diagnostic else None
    for fold, (y_test, short_test, slow_test) in enumerate(folds):
        others = [i for i in range(n_folds) if i != fold]
        lengths = [len(folds[i][0]) for i in others]
        total = sum(lengths)
        # Draw training bins uniformly without replacement. All held-out bins are scored.
        selection = (np.sort(rng.choice(total, max_train_bins, replace=False))
                     if total > max_train_bins else np.arange(total))
        offset = np.cumsum([0, *lengths])
        rows = [selection[(selection >= lo) & (selection < hi)] - lo
                for lo, hi in zip(offset, offset[1:])]
        y_train, short_train, slow_train = (
            np.concatenate([folds[i][index][row] for i, row in zip(others, rows)])
            for index in range(3)
        )
        if diagnostic:
            shifted_train = np.concatenate([shifted[i][row] for i, row in zip(others, rows)])

        for col, target in enumerate(AREAS):
            x_train = np.column_stack([short_train, slow_train[:, col]])
            x_test = np.column_stack([short_test, slow_test[:, col]])
            y0, y1 = y_train[:, col], y_test[:, col]
            full = fit_poisson(x_train, y0, x_test)
            for source in AREAS:
                if source == target:
                    continue
                source_cols = grouping == source
                keep = np.r_[~source_cols, True]
                reduced = fit_poisson(x_train[:, keep], y0, x_test[:, keep])
                gain = log_likelihood_gain(y1, full, reduced)
                null_gain = np.nan
                if diagnostic:
                    surrogate_train = x_train.copy()
                    surrogate_test = x_test.copy()
                    surrogate_train[:, :-1][:, source_cols] = shifted_train[:, source_cols]
                    surrogate_test[:, :-1][:, source_cols] = shifted[fold][:, source_cols]
                    surrogate = fit_poisson(surrogate_train, y0, surrogate_test)
                    null_gain = log_likelihood_gain(y1, surrogate, reduced)
                yield dict(source=source, target=target, fold=fold + 1,
                           n_train_bins=len(y0), n_test_bins=len(y1),
                           target_spikes=int(y1.sum()), gain_nats=gain,
                           shifted_gain_nats=null_gain)


def log_likelihood_gain(observed, full, reduced):
    # The log-factorial count term cancels for the two Poisson likelihoods.
    gain = np.sum(observed * (full - reduced) - (np.exp(full) - np.exp(reduced)))
    if not np.isfinite(gain):
        raise RuntimeError("Non-finite held-out likelihood gain")
    return gain


def save_results(output, rows, unit_rows, args, settings):
    folds = pd.DataFrame(rows)
    keys = ["mouse_id", "bin_ms", "history_ms", "primary", "subsample", "source", "target"]
    if folds.duplicated(keys + ["fold"]).any():
        raise RuntimeError("Duplicate fold scores")
    draws = (folds.groupby(keys, as_index=False)
             .agg(target_spikes=("target_spikes", "sum"), gain_nats=("gain_nats", "sum"),
                  shifted_gain_nats=("shifted_gain_nats", lambda x: x.sum(min_count=1))))
    if (draws["target_spikes"] <= 0).any():
        raise RuntimeError("A mouse-level score has no held-out target spikes")
    for prefix in ("", "shifted_"):
        draws[f"{prefix}bits_per_spike"] = draws[f"{prefix}gain_nats"] / (draws.target_spikes * np.log(2))
    mice = (draws.groupby([key for key in keys if key != "subsample"], as_index=False)
            .agg(bits_per_spike=("bits_per_spike", "mean"),
                 shifted_bits_per_spike=("shifted_bits_per_spike", "mean"),
                 n_subsamples=("subsample", "size")))
    for frame in (folds, draws, mice):
        frame["candidate"] = [(a, b) in CANDIDATES for a, b in zip(frame.source, frame.target)]
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in (("fold_scores", folds), ("subsample_scores", draws),
                        ("mouse_scores", mice), ("unit_counts", pd.DataFrame(unit_rows))):
        destination = output / f"q6_glm_{name}.csv"
        staged = destination.with_suffix(".csv.tmp")
        frame.to_csv(staged, index=False)
        staged.replace(destination)
    config = {"spikes_csv": str(args.spikes_csv.resolve()), "settings_ms": settings,
              "primary_ms": PRIMARY, "n_folds": args.folds, "n_subsamples": args.subsamples,
              "max_training_bins_per_fold": args.max_train_bins, "ridge_alpha": RIDGE,
              "slow_target_lags_ms": SLOW_LAGS_MS, "seed": args.seed,
              "primary_shift_diagnostic": not args.no_shift,
              "shift_block_ms": SHIFT_BLOCK_MS, "minimum_shift_ms": MIN_SHIFT_MS,
              "history_edges_ms": {f"{b}/{h}": history_edges(b, h) for b, h in settings},
              "mouse_ids": sorted(folds.mouse_id.unique().tolist())}
    (output / "q6_glm_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def main():
    project = Path(os.environ.get("CON2PHYS_PROJECT_DIR", "E:/con2phys_submission"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spikes-csv", type=Path, default=project / "dataset/full_sua_dataset.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(os.environ.get("Q6_OUTPUT_DIR", ROOT / "results/q6/glm")))
    parser.add_argument("--mice", nargs="+", type=int, help="Optional mouse IDs for a test run")
    parser.add_argument("--subsamples", type=int, default=5)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-train-bins", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--primary-only", action="store_true")
    parser.add_argument("--no-shift", action="store_true")
    args = parser.parse_args()
    if args.subsamples < 1 or args.folds < 2 or args.max_train_bins < 1 or args.seed < 0:
        parser.error("Need at least one subsample, two folds, a positive training cap, and a nonnegative seed")
    settings = (PRIMARY,) if args.primary_only else SETTINGS

    rows, unit_rows = [], []
    with threadpool_limits(limits=1, user_api="blas"):
        for mouse, units, start, duration in read_mice(args.spikes_csv, set(args.mice) if args.mice else None):
            draws = matched_draws(units, mouse, args.subsamples, args.seed)
            unit_rows.extend(dict(mouse_id=mouse, brain_area=area, available_units=len(units[area]),
                                  matched_units=len(draws[0][area]), duration_sec=duration)
                             for area in AREAS)
            for sample, draw in enumerate(draws, start=1):
                for bin_ms, history_ms in settings:
                    primary = (bin_ms, history_ms) == PRIMARY
                    rng = np.random.default_rng([args.seed, mouse, sample, bin_ms, history_ms])
                    counts = population_counts(units, draw, start, duration, bin_ms)
                    try:
                        result = list(score_setting(counts, bin_ms, history_ms, args.folds,
                                                    args.max_train_bins, rng, primary and not args.no_shift))
                    except (RuntimeError, ValueError) as exc:
                        raise RuntimeError(f"Mouse {mouse}, draw {sample}, {bin_ms}/{history_ms} ms: {exc}") from exc
                    rows.extend(dict(mouse_id=mouse, bin_ms=bin_ms, history_ms=history_ms,
                                     primary=primary, subsample=sample, **score) for score in result)
            print(f"Mouse {mouse}: completed {len(draws)} draws at {len(settings)} settings", flush=True)
    if not rows:
        raise ValueError("No mice were analyzed")
    save_results(args.output_dir, rows, unit_rows, args, settings)
    print(f"Wrote Q6 GLM scores to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()

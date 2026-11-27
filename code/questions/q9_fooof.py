#!/usr/bin/env python
"""Estimate the Q9 mouse-area aperiodic exponent with FOOOF.

The default path starts from the supplied channel x sample LFP arrays. It
calculates a Welch PSD for every valid channel, takes the pointwise median of
the linear channel PSDs within each mouse and brain area, and fits that single
mouse-area PSD with FOOOF. The primary 25-45 Hz fit and its quality threshold
follow Pochinok et al. (2024); the other ranges reproduce their Supplementary
Figure 2 sensitivity analysis.
"""

import argparse
import json
from pathlib import Path

import fooof
import numpy as np
import pandas as pd
import scipy
from fooof import FOOOF
from scipy.fft import rfft
from scipy.signal import get_window


ROOT = Path(__file__).resolve().parents[2]
AREAS = (1, 2, 3)
MICE = tuple(range(1, 19))
FS = 500
WINDOW_SECONDS = 1.0
OVERLAP_SECONDS = 0.25
NPERSEG = int(FS * WINDOW_SECONDS)
NOVERLAP = int(FS * OVERLAP_SECONDS)
PSD_RANGE = (1.0, 100.0)
PRIMARY_RANGE = (25.0, 45.0)
FIT_RANGES = ((15.0, 35.0), (20.0, 40.0), PRIMARY_RANGE, (30.0, 50.0))
MIN_R_SQUARED = 0.95
WELCH_BATCH_WINDOWS = 512


def range_label(frequency_range):
    return f"{frequency_range[0]:g}-{frequency_range[1]:g}"


def batched_welch(trace):
    """Mean Welch PSD with bounded memory and SciPy-compatible normalization."""
    step = NPERSEG - NOVERLAP
    starts = np.arange(0, trace.size - NPERSEG + 1, step, dtype=np.int64)
    if not starts.size:
        raise ValueError("Trace is shorter than one Welch window")
    window = get_window("hann", NPERSEG, fftbins=True)
    scale = 1.0 / (FS * np.sum(window ** 2))
    total = np.zeros(NPERSEG // 2 + 1, dtype=np.float64)
    for batch_start in range(0, starts.size, WELCH_BATCH_WINDOWS):
        batch_starts = starts[batch_start:batch_start + WELCH_BATCH_WINDOWS]
        segments = np.stack([trace[start:start + NPERSEG] for start in batch_starts])
        segments -= segments.mean(axis=1, keepdims=True)
        transform = rfft(segments * window, axis=1, workers=1)
        periodograms = np.abs(transform) ** 2 * scale
        # Match scipy.signal.welch's one-sided density scaling for an even FFT.
        periodograms[:, 1:-1] *= 2
        total += periodograms.sum(axis=0)
    frequencies = np.fft.rfftfreq(NPERSEG, d=1 / FS)
    return frequencies, total / starts.size


def channel_psds(path):
    """Return channel PSDs and QC without loading the full recording into RAM."""
    lfp = np.load(path, mmap_mode="r")
    if lfp.ndim != 2 or not lfp.shape[0] or lfp.shape[1] < NPERSEG:
        raise ValueError(f"Expected channels x samples with >= {NPERSEG} samples: {path}")

    spectra = []
    qc = []
    frequencies = None
    for channel_index, trace in enumerate(lfp, start=1):
        if not np.isfinite(trace).all():
            status = "nonfinite_signal"
        elif np.ptp(trace) == 0:
            status = "flat_signal"
        else:
            channel_frequencies, psd = batched_welch(trace)
            keep = ((channel_frequencies >= PSD_RANGE[0])
                    & (channel_frequencies <= PSD_RANGE[1]))
            channel_frequencies = channel_frequencies[keep]
            psd = psd[keep]
            if (np.isfinite(psd).all() and np.all(psd >= 0)
                    and np.any(psd > 0)):
                if frequencies is None:
                    frequencies = channel_frequencies
                elif not np.array_equal(frequencies, channel_frequencies):
                    raise ValueError(f"Inconsistent Welch frequency grid: {path}")
                spectra.append(psd)
                status = "included"
            else:
                status = "invalid_psd"
        qc.append((channel_index, status))

    if not spectra:
        raise ValueError(f"No valid channels: {path}")
    return frequencies, np.asarray(spectra), lfp.shape[1], qc


def fit_aperiodic(frequencies, psd, frequency_range):
    """Fit a fixed, peak-free FOOOF model and return parameters and curves."""
    model = FOOOF(aperiodic_mode="fixed", max_n_peaks=0, verbose=False)
    model.fit(frequencies, psd, frequency_range)
    offset, exponent = model.aperiodic_params_
    selected = ((frequencies >= frequency_range[0])
                & (frequencies <= frequency_range[1]))
    fit_frequencies = frequencies[selected]
    observed_psd = psd[selected]
    fitted_psd = 10 ** (offset - exponent * np.log10(fit_frequencies))
    return {
        "aperiodic_offset": float(offset),
        "aperiodic_exponent": float(exponent),
        # Larger values of this proxy correspond to stronger relative excitation.
        "relative_excitation_proxy": float(-exponent),
        "r_squared": float(model.r_squared_),
        "mean_absolute_error_log10": float(model.error_),
        "n_fit_frequencies": int(selected.sum()),
    }, fit_frequencies, observed_psd, fitted_psd


def atomic_csv(frame, path):
    staged = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(staged, index=False)
    staged.replace(path)


def run(data_dir, output_dir, mice, input_unit):
    files = [(mouse, area, data_dir / str(mouse) / f"lfp_{area}.npy")
             for mouse in mice for area in AREAS]
    missing = [path for _, _, path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing LFP file(s), starting with: {missing[0]}")

    psd_scale = 1e12 if input_unit == "V" else 1.0
    voltage_unit = "uV" if input_unit in ("V", "uV") else "stored_voltage_unit"
    psd_unit = f"{voltage_unit}^2/Hz"
    summaries = []
    spectra = []
    fits = []
    curves = []
    qc_rows = []
    expected_grid = None

    for mouse, area, path in files:
        frequencies, per_channel, n_samples, channel_qc = channel_psds(path)
        if expected_grid is None:
            expected_grid = frequencies
        elif not np.array_equal(frequencies, expected_grid):
            raise ValueError(f"Unexpected frequency grid: {path}")

        # This order is deliberate: PSD per channel first, then the channel median.
        median_psd = np.median(per_channel, axis=0) * psd_scale
        n_valid = per_channel.shape[0]
        summaries.append({
            "mouse_id": mouse,
            "brain_area": area,
            "n_channels": len(channel_qc),
            "n_valid_channels": n_valid,
            "n_samples": n_samples,
            "duration_s": n_samples / FS,
        })
        spectra.extend({
            "mouse_id": mouse,
            "brain_area": area,
            "frequency_hz": frequency,
            "psd": power,
            "psd_unit": psd_unit,
        } for frequency, power in zip(frequencies, median_psd))
        qc_rows.extend({
            "mouse_id": mouse,
            "brain_area": area,
            "channel_row_1based": channel,
            "status": status,
        } for channel, status in channel_qc)

        for frequency_range in FIT_RANGES:
            fit, fit_frequencies, observed, fitted = fit_aperiodic(
                frequencies, median_psd, frequency_range)
            label = range_label(frequency_range)
            passes = fit["r_squared"] >= MIN_R_SQUARED
            fits.append({
                "mouse_id": mouse,
                "brain_area": area,
                "fit_range": label,
                "fit_low_hz": frequency_range[0],
                "fit_high_hz": frequency_range[1],
                **fit,
                "passes_r_squared": passes,
                "is_primary": frequency_range == PRIMARY_RANGE,
                "included_primary": frequency_range == PRIMARY_RANGE and passes,
            })
            curves.extend({
                "mouse_id": mouse,
                "brain_area": area,
                "fit_range": label,
                "frequency_hz": frequency,
                "observed_psd": power,
                "aperiodic_fit_psd": fit_power,
                "observed_log10_psd_offset_normalized": np.log10(power)
                    - fit["aperiodic_offset"],
                "aperiodic_fit_log10_psd_offset_normalized": np.log10(fit_power)
                    - fit["aperiodic_offset"],
                "psd_unit": psd_unit,
            } for frequency, power, fit_power
              in zip(fit_frequencies, observed, fitted))

        print(f"Mouse {mouse:>2}, area {area}: {n_valid}/{len(channel_qc)} valid channels",
              flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(pd.DataFrame(summaries), output_dir / "q9_mouse_area_summary.csv")
    atomic_csv(pd.DataFrame(spectra), output_dir / "q9_mouse_area_psd.csv")
    atomic_csv(pd.DataFrame(fits), output_dir / "q9_fooof_fits.csv")
    atomic_csv(pd.DataFrame(curves), output_dir / "q9_fooof_curves.csv")
    atomic_csv(pd.DataFrame(qc_rows), output_dir / "q9_channel_qc.csv")

    config = {
        "paper": {
            "citation": "Pochinok et al., Nature Communications (2024)",
            "doi": "10.1038/s41467-024-44983-z",
            "adaptation": "PSD per channel, then pointwise median within mouse-area",
        },
        "data_dir": str(data_dir.resolve()),
        "mice": list(mice),
        "areas": list(AREAS),
        "fs_hz": FS,
        "welch": {
            "window": "hann",
            "window_seconds": WINDOW_SECONDS,
            "overlap_seconds": OVERLAP_SECONDS,
            "nperseg": NPERSEG,
            "noverlap": NOVERLAP,
            "detrend": "constant",
            "scaling": "density",
            "average_across_windows": "mean",
            "frequency_resolution_hz": FS / NPERSEG,
            "batch_windows": WELCH_BATCH_WINDOWS,
        },
        "channel_qc": "exclude nonfinite or flat traces and invalid PSDs",
        "channel_aggregation": "pointwise median of linear channel PSDs",
        "fooof": {
            "version": fooof.__version__,
            "aperiodic_mode": "fixed",
            "max_n_peaks": 0,
            "primary_range_hz": list(PRIMARY_RANGE),
            "sensitivity_ranges_hz": [list(item) for item in FIT_RANGES],
            "minimum_r_squared": MIN_R_SQUARED,
        },
        "interpretation": (
            "A smaller aperiodic exponent indicates stronger relative excitation; "
            "relative_excitation_proxy is negative exponent."
        ),
        "input_voltage_unit": input_unit,
        "psd_scale_from_stored_signal": psd_scale,
        "psd_unit": psd_unit,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
    }
    (output_dir / "q9_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path("E:/con2phys_submission/dataset_download/py-files"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "q9")
    parser.add_argument("--mice", type=int, nargs="+", choices=MICE, default=MICE)
    parser.add_argument("--input-unit", choices=("native", "V", "uV"), default="V",
                        help="The supplied arrays store volts; V exports PSD in uV^2/Hz.")
    args = parser.parse_args()
    if len(set(args.mice)) != len(args.mice):
        parser.error("--mice must contain unique IDs")
    run(args.data_dir, args.output_dir, args.mice, args.input_unit)


if __name__ == "__main__":
    main()

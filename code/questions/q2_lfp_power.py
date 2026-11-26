#!/usr/bin/env python
"""Calculate Q2 mouse–area broadband LFP power from the supplied LFP arrays."""

import argparse
import csv
import json
from pathlib import Path

import neurodsp
import numpy as np
from neurodsp.spectral import compute_spectrum
from scipy.integrate import trapezoid


ROOT = Path(__file__).resolve().parents[2]
AREAS = (1, 2, 3)
MICE = tuple(range(1, 19))
FS = 500
NPERSEG = FS * 4
NOVERLAP = FS * 2
F_RANGE = (0.1, 150.0)
POWER_BAND = (1.0, 100.0)


def mouse_area_spectrum(path):
    """Read one channel at a time and take the pointwise median of valid PSDs."""
    lfp = np.load(path, mmap_mode="r")
    if lfp.ndim != 2 or lfp.shape[1] < NPERSEG:
        raise ValueError(f"Expected channels × samples with at least {NPERSEG} samples: {path}")

    spectra = []
    channel_qc = []
    for index, trace in enumerate(lfp, start=1):
        if not np.isfinite(trace).all():
            status = "nonfinite_signal"
        elif np.ptp(trace) == 0:
            status = "flat_signal"
        else:
            frequencies, psd = compute_spectrum(
                trace, FS, method="welch", avg_type="median", window="hann",
                nperseg=NPERSEG, noverlap=NOVERLAP, f_range=F_RANGE,
            )
            if np.isfinite(psd).all() and np.all(psd >= 0) and np.any(psd > 0):
                spectra.append(psd)
                status = "included"
            else:
                status = "invalid_psd"
        channel_qc.append((index, status))

    if not spectra:
        raise ValueError(f"No valid channels: {path}")
    return frequencies, np.median(spectra, axis=0), lfp.shape[1], channel_qc


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def run(data_dir, output_dir, mice, input_unit):
    files = [(mouse, area, data_dir / str(mouse) / f"lfp_{area}.npy")
             for mouse in mice for area in AREAS]
    for _, _, path in files:
        if not path.is_file():
            raise FileNotFoundError(path)

    # Rescale densities after Welch so that the channel input stays memory-mapped.
    psd_scale = 1e12 if input_unit == "V" else 1.0
    voltage_unit = "uV" if input_unit in ("V", "uV") else "stored voltage unit"
    power_unit = f"{voltage_unit}^2"
    psd_unit = f"{power_unit}/Hz"
    summaries = []
    spectra = []
    qc = []
    area_spectra = {area: [] for area in AREAS}
    expected_grid = None
    sample_counts = {}
    for mouse, area, path in files:
        frequencies, psd, n_samples, channel_qc = mouse_area_spectrum(path)
        if sample_counts.setdefault(mouse, n_samples) != n_samples:
            raise ValueError(f"Area recordings for mouse {mouse} differ in sample count")
        if expected_grid is None:
            expected_grid = frequencies
        elif not np.array_equal(frequencies, expected_grid):
            raise ValueError(f"Unexpected frequency grid: {path}")

        psd *= psd_scale
        band = (frequencies >= POWER_BAND[0]) & (frequencies <= POWER_BAND[1])
        if tuple(frequencies[band][[0, -1]]) != POWER_BAND:
            raise ValueError("The frequency grid must contain both 1 and 100 Hz")
        band_power = float(trapezoid(psd[band], frequencies[band]))
        if not np.isfinite(band_power) or band_power <= 0:
            raise ValueError(f"Invalid 1–100 Hz power: {path}")
        n_valid = sum(status == "included" for _, status in channel_qc)

        summaries.append((mouse, area, len(channel_qc), n_valid, n_samples,
                          n_samples / FS, band_power, power_unit))
        spectra.extend((mouse, area, f, p, psd_unit) for f, p in zip(frequencies, psd))
        qc.extend((mouse, area, index, status) for index, status in channel_qc)
        area_spectra[area].append(psd)
        print(f"Mouse {mouse:>2}, area {area}: {n_valid}/{len(channel_qc)} valid channels", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "q2_mouse_area_power.csv",
              ("mouse_id", "brain_area", "n_channels", "n_valid_channels", "n_samples",
               "duration_s", "broadband_power", "power_unit"), summaries)
    write_csv(output_dir / "q2_mouse_area_psd.csv",
              ("mouse_id", "brain_area", "frequency_hz", "psd", "psd_unit"), spectra)
    write_csv(output_dir / "q2_channel_qc.csv",
              ("mouse_id", "brain_area", "channel_row_1based", "status"), qc)
    write_csv(output_dir / "q2_area_median_psd.csv",
              ("brain_area", "n_mice", "frequency_hz", "median_psd", "psd_unit"),
              ((area, len(mice), f, p, psd_unit)
               for area in AREAS
               for f, p in zip(expected_grid, np.median(area_spectra[area], axis=0))))

    config = {
        "data_dir": str(data_dir.resolve()), "mice": list(mice), "areas": list(AREAS),
        "fs_hz": FS, "neurodsp_version": neurodsp.__version__,
        "welch": {"window": "hann", "nperseg": NPERSEG, "noverlap": NOVERLAP,
                  "avg_type": "median", "f_range_hz": F_RANGE,
                  "first_frequency_hz": float(expected_grid[0]),
                  "last_frequency_hz": float(expected_grid[-1]),
                  "frequency_step_hz": FS / NPERSEG},
        "signal_preparation": "supplied full recordings, retaining supplied reference and preprocessing",
        "channel_qc": "exclude nonfinite or flat traces and invalid PSDs",
        "channel_aggregation": "pointwise median of linear PSDs",
        "across_mouse_spectrum": "pointwise median of mouse–area PSDs",
        "power_integration": "trapezoidal integration of the mouse–area PSD",
        "power_band_hz": POWER_BAND, "input_voltage_unit": input_unit,
        "psd_scale_from_stored_signal": psd_scale, "power_unit": power_unit, "psd_unit": psd_unit,
    }
    (output_dir / "q2_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path("E:/con2phys_submission/dataset_download/py-files"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "q2")
    parser.add_argument("--mice", type=int, nargs="+", choices=MICE, default=MICE)
    parser.add_argument("--input-unit", choices=("native", "V", "uV"), default="V",
                        help="The supplied CON2PHYS arrays store volts. Native retains stored scaling.")
    args = parser.parse_args()
    if len(set(args.mice)) != len(args.mice):
        parser.error("--mice must contain unique IDs")
    run(args.data_dir, args.output_dir, args.mice, args.input_unit)


if __name__ == "__main__":
    main()

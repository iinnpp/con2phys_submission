#!/usr/bin/env python
"""Plot channel-wise LFP PSDs before Q3 ripple analysis."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from neurodsp.spectral import compute_spectrum

from q2_lfp_power import AREAS, FS, MICE, NOVERLAP, NPERSEG, ROOT


DATA_DIR = Path("E:/con2phys_submission/dataset_download/py-files")
OUTPUT_DIR = ROOT / "results" / "q3" / "psd_pics"
F_RANGE = (0.1, 200.0)
RIPPLE_RANGE = (80.0, 200.0)


def channel_psds(path):
    """Return the frequency grid and one row per channel in stored depth order."""
    lfp = np.load(path, mmap_mode="r")
    if lfp.ndim != 2 or lfp.shape[1] < NPERSEG:
        raise ValueError(f"Expected channels by samples with at least {NPERSEG} samples: {path}")

    frequencies = None
    spectra = None
    valid_channels = 0
    for i, trace in enumerate(lfp):
        if not np.isfinite(trace).all() or np.ptp(trace) == 0:
            continue
        f, psd = compute_spectrum(
            trace, FS, method="welch", avg_type="median", window="hann",
            nperseg=NPERSEG, noverlap=NOVERLAP, f_range=F_RANGE,
        )
        if not np.isfinite(psd).all() or np.any(psd < 0) or not np.any(psd > 0):
            continue
        if frequencies is None:
            frequencies = f
            spectra = np.full((lfp.shape[0], f.size), np.nan)
        # Convert V²/Hz to dB relative to 1 µV²/Hz. Zero bins are masked.
        with np.errstate(divide="ignore"):
            spectra[i] = np.where(psd > 0, 10.0 * np.log10(psd) + 120.0, np.nan)
        valid_channels += 1

    if frequencies is None:
        raise ValueError(f"No valid channels in {path}")
    return frequencies, spectra, valid_channels


def color_limits(results, mask):
    """Choose fixed color limits per frequency view from all plotted recordings."""
    values = np.concatenate([psd[:, mask].ravel() for psd in results.values()])
    low, high = np.nanpercentile(values, [1, 99])
    return (low - 0.5, high + 0.5) if low == high else (low, high)


def plot_heatmap(path, mouse, area, frequencies, spectra, views):
    fig = Figure(figsize=(11, 4.5), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2, sharey=True)
    fig.suptitle(f"Mouse {mouse:02d} · Area {area}", fontsize=15, fontweight="semibold")
    step = FS / NPERSEG
    row_edges = np.arange(spectra.shape[0] + 1) + 0.5
    cmap = colormaps["magma"]
    cmap.set_bad("#B8BEC5")

    for ax, (title, bounds, mask, limits) in zip(axes, views):
        f = frequencies[mask]
        edges = np.r_[f - step / 2, f[-1] + step / 2]
        mesh = ax.pcolormesh(edges, row_edges, spectra[:, mask], cmap=cmap,
                             vmin=limits[0], vmax=limits[1], shading="flat")
        ax.set(xlim=bounds, ylim=(row_edges[0], row_edges[-1]),
               title=title, xlabel="Frequency (Hz)")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
        cbar = fig.colorbar(mesh, ax=ax, extend="both", pad=0.025)
        cbar.set_label("PSD (dB re 1 µV²/Hz)")
    axes[1].set_xticks(np.arange(80, 201, 20))
    axes[0].set_ylabel("Channel row (deepest to superficial)")
    fig.savefig(path, dpi=165, facecolor="white")
    fig.clear()


def run(data_dir, output_dir, mice, workers):
    paths = {(mouse, area): data_dir / str(mouse) / f"lfp_{area}.npy"
             for mouse in mice for area in AREAS}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    spectra = {}
    frequencies = None
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = {executor.submit(channel_psds, path): key for key, path in paths.items()}
        for completed in as_completed(jobs):
            mouse, area = jobs[completed]
            f, psd, valid = completed.result()
            if frequencies is None:
                frequencies = f
            elif not np.array_equal(f, frequencies):
                raise ValueError(f"Unexpected frequency grid in mouse {mouse}, area {area}")
            spectra[mouse, area] = psd
            print(f"[{len(spectra):02d}/{len(paths)}] Mouse {mouse:02d}, area {area}: "
                  f"{valid}/{psd.shape[0]} valid channels", flush=True)

    views = []
    for title, bounds in (("Full spectrum", F_RANGE), ("Ripple-band zoom", RIPPLE_RANGE)):
        mask = (frequencies >= bounds[0]) & (frequencies <= bounds[1])
        views.append((title, bounds, mask, color_limits(spectra, mask)))

    output_dir.mkdir(parents=True, exist_ok=True)
    for mouse, area in paths:
        plot_heatmap(output_dir / f"q3_mouse_{mouse:02d}_area_{area}_psd.png",
                     mouse, area, frequencies, spectra[mouse, area], views)
    print(f"Saved {len(paths)} heatmaps to {output_dir.resolve()}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--mice", type=int, nargs="+", choices=MICE, default=MICE)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1 or len(set(args.mice)) != len(args.mice):
        parser.error("Use a positive worker count and unique mouse IDs.")
    run(args.data_dir, args.output_dir, args.mice, args.workers)


if __name__ == "__main__":
    main()

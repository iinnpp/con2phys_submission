#!/usr/bin/env python
"""Calculate and plot channel-wise LFP PSDs before Q3 ripple analysis."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from neurodsp.spectral import compute_spectrum

from q2_lfp_power import AREAS, FS, MICE, NOVERLAP, NPERSEG, ROOT


DATA_DIR = Path("E:/con2phys_submission/dataset_download/py-files")
OUTPUT_DIR = ROOT / "results" / "q3" / "psd_pics"
LINE_OUTPUT_DIR = OUTPUT_DIR.with_name("psd_pics_line_plot")
PSD_ARCHIVE = OUTPUT_DIR.parent / "q3_channel_psd.npz"
F_RANGE = (0.1, 200.0)
RIPPLE_RANGE = (80.0, 200.0)
AREA_COLORS = {1: "#0072B2", 2: "#D55E00", 3: "#00856A"}
FILE_READ_LOCK = Lock()


def channel_psds(path, *, buffered=False):
    """Return the frequency grid and PSD in µV²/Hz in stored channel order."""
    if buffered:
        # Sequential file reads avoid contention on the recording drive while
        # the worker threads compute spectra for arrays already in memory.
        with FILE_READ_LOCK:
            lfp = np.load(path)
    else:
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
        spectra[i] = psd * 1e12
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
    fig.suptitle(f"Mouse {mouse:02d} · Area {area}", fontsize=15, fontweight="bold")
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


def plot_line_summary(path, mouse, frequencies, spectra):
    fig = Figure(figsize=(12.3, 6.6), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 3, sharex=True, sharey=True)
    fig.suptitle(f"Mouse {mouse:02d} · Channel PSDs\n"
                 "Thin lines show individual channels. Bold lines show each summary.",
                 fontsize=13, linespacing=1.45)
    fig.supxlabel("Frequency (Hz)")
    for col, area in enumerate(AREAS):
        psd = spectra[mouse, area]
        valid_count = np.count_nonzero(np.isfinite(psd).any(axis=1))
        axes[0, col].set_title(f"Area {area}  ·  {valid_count} channels", fontsize=11)
        for row, (label, aggregate) in enumerate(
            (("Mean", np.nanmean), ("Median", np.nanmedian))
        ):
            ax = axes[row, col]
            ax.set(yscale="log", xlim=F_RANGE)
            ax.grid(alpha=0.22)
            ax.spines[["top", "right"]].set_visible(False)
            ax.plot(frequencies, psd.T, color="#697D90", lw=0.55, alpha=0.28)
            ax.plot(frequencies, aggregate(psd, axis=0), color=AREA_COLORS[area], lw=2.1)
            if col == 0:
                ax.set_ylabel(f"{label} PSD (µV²/Hz)")
    axes[-1, 0].set_xticks([0.1, 50, 100, 150, 200], ["0.1", "50", "100", "150", "200"])
    fig.savefig(path, dpi=165, facecolor="white")
    fig.clear()


def save_channel_psds(path, frequencies, spectra):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"mouse_{mouse:02d}_area_{area}": psd
              for (mouse, area), psd in sorted(spectra.items())}
    with path.open("wb") as handle:
        np.savez_compressed(
            handle, frequency_hz=frequencies, sampling_rate_hz=FS,
            welch_nperseg=NPERSEG, welch_noverlap=NOVERLAP, welch_average="median",
            welch_window="hann", requested_frequency_range_hz=F_RANGE,
            psd_unit="uV^2/Hz", **arrays,
        )


def run(data_dir, output_dir, mice, workers, *, line_plots=False, save_psd=None):
    paths = {(mouse, area): data_dir / str(mouse) / f"lfp_{area}.npy"
             for mouse in mice for area in AREAS}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    spectra = {}
    frequencies = None
    output_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = {executor.submit(channel_psds, path, buffered=line_plots): key
                for key, path in paths.items()}
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
            if line_plots and all((mouse, a) in spectra for a in AREAS):
                plot_line_summary(output_dir / f"q3_mouse_{mouse:02d}_psd_mean_median.png",
                                  mouse, frequencies, spectra)
                print(f"Saved mouse {mouse:02d} line plot.", flush=True)

    if save_psd is not None:
        save_channel_psds(save_psd, frequencies, spectra)
        print(f"Saved channel PSDs to {save_psd.resolve()}.", flush=True)
    if line_plots:
        print(f"Saved {len(mice)} line plots to {output_dir.resolve()}.", flush=True)
        return

    # Heatmaps use dB relative to 1 µV²/Hz, with zero-power bins masked.
    with np.errstate(divide="ignore"):
        spectra = {key: np.where(psd > 0, 10.0 * np.log10(psd), np.nan)
                   for key, psd in spectra.items()}

    views = []
    for title, bounds in (("Full spectrum", F_RANGE), ("Ripple-band zoom", RIPPLE_RANGE)):
        mask = (frequencies >= bounds[0]) & (frequencies <= bounds[1])
        views.append((title, bounds, mask, color_limits(spectra, mask)))

    for mouse, area in paths:
        plot_heatmap(output_dir / f"q3_mouse_{mouse:02d}_area_{area}_psd.png",
                     mouse, area, frequencies, spectra[mouse, area], views)
    print(f"Saved {len(paths)} heatmaps to {output_dir.resolve()}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mice", type=int, nargs="+", choices=MICE, default=MICE)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--line-plots", action="store_true",
                        help="Plot six panels per mouse and save the per-channel PSD archive.")
    parser.add_argument("--save-psd", type=Path,
                        help="Override the PSD archive path or save PSDs with heatmaps.")
    args = parser.parse_args()
    if args.workers < 1 or len(set(args.mice)) != len(args.mice):
        parser.error("Use a positive worker count and unique mouse IDs.")
    output_dir = args.output_dir or (LINE_OUTPUT_DIR if args.line_plots else OUTPUT_DIR)
    psd_path = args.save_psd
    if psd_path is None and args.line_plots:
        full_default_run = args.output_dir is None and tuple(args.mice) == MICE
        psd_path = PSD_ARCHIVE if full_default_run else output_dir / PSD_ARCHIVE.name
    run(args.data_dir, output_dir, args.mice, args.workers,
        line_plots=args.line_plots, save_psd=psd_path)


if __name__ == "__main__":
    main()

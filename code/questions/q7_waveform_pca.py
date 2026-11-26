"""Fit PCA to all average waveforms pooled within each CON2PHYS brain area."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from q7_fast_spiking import DEFAULT_INPUT, DEFAULT_OUTPUT, SAMPLES_PER_MS


AREA_COLORS = {1: "#0072B2", 2: "#D55E00", 3: "#00856A"}
N_MICE = 18
N_SAMPLES = 128
RAW = "raw"
PEAK_TO_PEAK = "peak_to_peak"
Z_SCORE = "zscore"
WAVEFORM_LABELS = {
    RAW: "waveforms",
    PEAK_TO_PEAK: "amplitude-normalized waveforms",
    Z_SCORE: "column-wise z-scored waveforms",
}


def read_preprocessing(pca):
    if "preprocessing" in pca:
        return str(pca["preprocessing"].item())
    return PEAK_TO_PEAK if pca.get("amplitude_normalized", False) else RAW


def load_waveforms(root):
    """Stack original waveform rows and keep their identifiers in the same order."""
    waveforms, metadata = [], []
    for mouse in range(1, N_MICE + 1):
        folder = root / str(mouse)
        data = np.load(folder / "brain_area.npy", allow_pickle=True).item()
        cluster_ids = np.asarray(data["cluster_id"]).ravel()
        areas = np.asarray(data["brain_area"]).ravel()
        raw = np.load(folder / "waveforms.npy")
        if (raw.shape != (len(cluster_ids), N_SAMPLES) or areas.shape != cluster_ids.shape
                or set(areas) != set(AREA_COLORS) or len(np.unique(cluster_ids)) != len(cluster_ids)
                or not np.isfinite(raw).all()):
            raise ValueError(f"Invalid waveform data or metadata for mouse {mouse}")
        waveforms.append(raw)
        metadata.append(pd.DataFrame({"mouse_id": mouse, "cluster_id": cluster_ids, "brain_area": areas}))
    return np.concatenate(waveforms, axis=0), pd.concat(metadata, ignore_index=True)


def load_raw_pca_waveforms(root):
    """Read the original waveform matrices already saved by the first PCA run."""
    waveforms, metadata = [], []
    for area in AREA_COLORS:
        with np.load(root / f"q7_pca_area_{area}.npz") as pca:
            if read_preprocessing(pca) != RAW:
                raise ValueError(f"Expected original waveforms in Area {area}'s PCA archive")
            matrix = pca["waveforms"]
            units = pd.DataFrame({"mouse_id": pca["mouse_id"], "cluster_id": pca["cluster_id"], "brain_area": area})
        if (matrix.shape != (len(units), N_SAMPLES) or not np.isfinite(matrix).all()
                or units.duplicated(["mouse_id", "cluster_id"]).any()
                or set(units.mouse_id) != set(range(1, N_MICE + 1))):
            raise ValueError(f"Invalid original waveform matrix or unit IDs for Area {area}")
        waveforms.append(matrix)
        metadata.append(units)
    return np.concatenate(waveforms, axis=0), pd.concat(metadata, ignore_index=True)


def fit_areas(waveforms, metadata, output, *, preprocessing=RAW):
    variance_tables = []
    for area in AREA_COLORS:
        selected = metadata.brain_area.eq(area)
        units = metadata.loc[selected]
        matrix = waveforms[selected.to_numpy()].astype(np.float64, copy=False)
        amplitude = np.ptp(matrix, axis=1)
        scaler_parameters = {}
        if preprocessing == PEAK_TO_PEAK:
            if not (amplitude > 0).all():
                raise ValueError(f"Cannot normalize flat or invalid waveforms in Area {area}")
            edges = np.concatenate([matrix[:, :12], matrix[:, -12:]], axis=1)
            baseline = np.median(edges, axis=1)
            matrix = (matrix - baseline[:, None]) / amplitude[:, None]
        elif preprocessing == Z_SCORE:
            scaler = StandardScaler()
            matrix = scaler.fit_transform(matrix)
            scaler_parameters = {"zscore_mean": scaler.mean_, "zscore_scale": scaler.scale_}
        # PCA centers each time sample across all units in the area.
        pca = PCA(svd_solver="full")
        scores = pca.fit_transform(matrix)
        variance = pca.explained_variance_ratio_
        cumulative = np.cumsum(variance)
        variance_tables.append(pd.DataFrame({
            "brain_area": area,
            "n_units": len(units),
            "pc": np.arange(1, len(variance) + 1),
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": variance,
            "cumulative_explained_variance_ratio": cumulative,
        }))
        np.savez_compressed(
            output / f"q7_pca_area_{area}.npz",
            waveforms=matrix,
            preprocessing=preprocessing,
            amplitude_normalized=preprocessing == PEAK_TO_PEAK,
            original_peak_to_peak=amplitude,
            sample_time_ms=np.arange(matrix.shape[1]) / SAMPLES_PER_MS,
            mouse_id=units.mouse_id.to_numpy(),
            cluster_id=units.cluster_id.to_numpy(),
            mean_waveform=pca.mean_,
            components=pca.components_,
            scores=scores,
            explained_variance_ratio=variance,
            cumulative_explained_variance_ratio=cumulative,
            **scaler_parameters,
        )
        print(f"Area {area}: {matrix.shape[0]} units × {matrix.shape[1]} samples, "
              f"first three PCs explain {cumulative[2]:.1%}")
    return pd.concat(variance_tables, ignore_index=True)


def plot_variance(table, path, *, preprocessing=RAW):
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.8), sharey="row", layout="constrained")
    shown_pcs = 16
    for col, (area, color) in enumerate(AREA_COLORS.items()):
        data = table.loc[table.brain_area.eq(area)]
        leading = data.iloc[:shown_pcs]
        bar, cumulative = axes[:, col]
        bar.bar(leading.pc, leading.explained_variance_ratio, color=color)
        bar.set_title(f"Area {area}  (n = {data.n_units.iloc[0]})")
        bar.set_xlim(0.4, shown_pcs + 0.6)
        bar.set_xticks([1, 4, 8, 12, 16])
        bar.set_xlabel("Principal component")

        cumulative.plot(data.pc, data.cumulative_explained_variance_ratio, color=color, linewidth=2)
        cumulative.set_xscale("log", base=2)
        ticks = [1, 2, 4, 8, 16, 32, 64, 128]
        cumulative.set_xticks(ticks, labels=ticks)
        cumulative.set_xlim(1, N_SAMPLES)
        cumulative.set_ylim(0, 1.04)
        cumulative.set_xlabel("Number of PCs (log scale)")

        for ax in (bar, cumulative):
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="0.9")
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Variance explained per PC")
    axes[1, 0].set_ylabel("Cumulative variance explained")
    for ax in axes[:, 0]:
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    fig.suptitle(f"PCA of {WAVEFORM_LABELS[preprocessing]} by brain area", fontsize=16)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    source.add_argument("--raw-pca-dir", type=Path, help="Reuse the original waveforms stored by the first PCA run")
    scaling = parser.add_mutually_exclusive_group()
    scaling.add_argument("--normalize", dest="preprocessing", action="store_const", const=PEAK_TO_PEAK,
                         help="Baseline-correct and normalize each waveform by peak-to-peak amplitude")
    scaling.add_argument("--zscore", dest="preprocessing", action="store_const", const=Z_SCORE,
                         help="Standardize each time sample across units within each area")
    parser.set_defaults(preprocessing=RAW)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output_folders = {RAW: "pca", PEAK_TO_PEAK: "pca_normalized", Z_SCORE: "pca_zscored"}
    output = args.output or DEFAULT_OUTPUT / output_folders[args.preprocessing]
    if args.raw_pca_dir is None:
        waveforms, metadata = load_waveforms(args.input)
    else:
        waveforms, metadata = load_raw_pca_waveforms(args.raw_pca_dir)
    output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        table = fit_areas(waveforms, metadata, output, preprocessing=args.preprocessing)
    table.to_csv(output / "q7_pca_explained_variance.csv", index=False)
    plot_variance(table, output / "q7_pca_explained_variance.png", preprocessing=args.preprocessing)


if __name__ == "__main__":
    main()

"""Compare per-area DBSCAN fits using two or three unwhitened waveform PCs."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from q7_fast_spiking import DEFAULT_OUTPUT
from q7_waveform_pca import RAW, WAVEFORM_LABELS, read_preprocessing


AREAS = (1, 2, 3)
PC_COUNTS = (2, 3)
REFERENCE_MIN_SAMPLES = 8
MIN_SAMPLES_GRID = (5, REFERENCE_MIN_SAMPLES, 12)
EPS_MULTIPLIERS = (0.6, 0.8, 1.0, 1.2, 1.4)
KNEE_FIT_FRACTION = 0.98


def knee_radius(scores, min_samples):
    """Use the maximum gap between the sorted k-distance curve and its chord."""
    neighbors = NearestNeighbors(n_neighbors=min_samples).fit(scores)
    # Query with the same points so each point counts itself, as in DBSCAN.
    distances = np.sort(neighbors.kneighbors(scores)[0][:, -1])
    # Leave the largest distances out of the knee estimate to limit leverage from outliers.
    trimmed = distances[:int(np.ceil(len(distances) * KNEE_FIT_FRACTION))]
    spread = np.ptp(trimmed)
    if spread == 0:
        raise ValueError("No knee can be estimated from a constant k-distance curve")
    index = np.argmax(np.linspace(0, 1, len(trimmed)) - (trimmed - trimmed[0]) / spread)
    eps = trimmed[index]
    if eps <= 0:
        raise ValueError("The estimated DBSCAN radius must be positive")
    return eps, (index + 1) / len(distances)


def analyze_area(scores, identifiers, area):
    stats, assignments = [], []
    for n_pcs in PC_COUNTS:
        x = scores[:, :n_pcs]
        for min_samples in MIN_SAMPLES_GRID:
            radius, knee_fraction = knee_radius(x, min_samples)
            for multiplier in EPS_MULTIPLIERS:
                eps = radius * multiplier
                labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(x)
                noise = labels == -1
                sizes = np.sort(np.unique(labels[~noise], return_counts=True)[1])[::-1]
                reference = min_samples == REFERENCE_MIN_SAMPLES and multiplier == 1.0
                stats.append({
                    "brain_area": area,
                    "n_pcs": n_pcs,
                    "min_samples": min_samples,
                    "eps": eps,
                    "eps_multiplier": multiplier,
                    "knee_fraction": knee_fraction,
                    "reference_fit": reference,
                    "n_units": len(labels),
                    "n_clusters": len(sizes),
                    "n_noise": noise.sum(),
                    "noise_fraction": noise.mean(),
                    "largest_cluster_size": sizes[0] if len(sizes) else 0,
                    "second_largest_cluster_size": sizes[1] if len(sizes) >= 2 else 0,
                })
                if reference:
                    assignments.append(identifiers.assign(
                        n_pcs=n_pcs, pc1=scores[:, 0], pc2=scores[:, 1], pc3=scores[:, 2],
                        dbscan_label=labels,
                    ))
    return pd.DataFrame(stats), pd.concat(assignments, ignore_index=True)


def plot_clusters(assignments, summary, path, *, preprocessing=RAW):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex="col", sharey="col", layout="constrained")
    cmap = plt.get_cmap("tab10")
    display_scale = 1e6 if preprocessing == RAW else 1
    score_units = " (×10⁻⁶ dataset units)" if preprocessing == RAW else ""
    for ax, n_pcs, area in zip(axes.flat, np.repeat(PC_COUNTS, len(AREAS)), AREAS * len(PC_COUNTS)):
        rows = assignments.loc[assignments.brain_area.eq(area) & assignments.n_pcs.eq(n_pcs)]
        fit = summary.loc[summary.brain_area.eq(area) & summary.n_pcs.eq(n_pcs)].iloc[0]
        for label, group in rows.groupby("dbscan_label", sort=True):
            color = "#a3aab4" if label == -1 else cmap(label % cmap.N)
            ax.scatter(group.pc1 * display_scale, group.pc2 * display_scale, s=12,
                       color=color, alpha=.8, linewidths=0)
        cluster_word = "cluster" if fit.n_clusters == 1 else "clusters"
        ax.set_title(f"Area {area} · {fit.n_clusters} {cluster_word}, {fit.n_noise} noise", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(color="0.93")
    for row, n_pcs in enumerate(PC_COUNTS):
        axes[row, 0].set_ylabel(f"{n_pcs}-PC fit\nPC2 score{score_units}")
    fig.supxlabel(f"PC1 score{score_units}")
    source = WAVEFORM_LABELS[preprocessing]
    fig.suptitle(f"DBSCAN on PCA scores from {source}\nPC1–PC2 view of both fits · gray points are noise",
                 fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pca-dir", type=Path, default=DEFAULT_OUTPUT / "pca")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.pca_dir / "dbscan"

    stats, assignments = [], []
    preprocessing_modes = set()
    with threadpool_limits(limits=1):
        for area in AREAS:
            with np.load(args.pca_dir / f"q7_pca_area_{area}.npz") as pca:
                scores = pca["scores"][:, :max(PC_COUNTS)]
                ids = pd.DataFrame({"brain_area": area, "mouse_id": pca["mouse_id"],
                                    "cluster_id": pca["cluster_id"]})
                preprocessing_modes.add(read_preprocessing(pca))
            if (scores.shape != (len(ids), max(PC_COUNTS)) or not np.isfinite(scores).all()
                    or len(scores) < max(MIN_SAMPLES_GRID) or ids.duplicated().any()):
                raise ValueError(f"Invalid PCA scores or unit IDs for area {area}")
            area_stats, area_assignments = analyze_area(scores, ids, area)
            stats.append(area_stats)
            assignments.append(area_assignments)

    if len(preprocessing_modes) != 1:
        raise ValueError("PCA archives must use the same preprocessing in all areas")
    sensitivity = pd.concat(stats, ignore_index=True)
    summary = sensitivity.loc[sensitivity.reference_fit]
    units = pd.concat(assignments, ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    sensitivity.to_csv(output / "q7_dbscan_sensitivity.csv", index=False)
    summary.to_csv(output / "q7_dbscan_summary.csv", index=False)
    units.to_csv(output / "q7_dbscan_units.csv", index=False)
    plot_clusters(units, summary, output / "q7_dbscan_clusters.png", preprocessing=preprocessing_modes.pop())
    print(summary[["brain_area", "n_pcs", "eps", "n_clusters", "n_noise"]].to_string(index=False))


if __name__ == "__main__":
    main()

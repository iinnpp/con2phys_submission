"""Fit and inspect BIC-selected Gaussian mixtures of per-area waveform PCs."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from q7_fast_spiking import DEFAULT_OUTPUT, waveform_features
from q7_two_feature_clusters import COVARIANCES, fit_model
from q7_waveform_pca import PEAK_TO_PEAK, RAW, WAVEFORM_LABELS, Z_SCORE, read_preprocessing


AREAS = (1, 2, 3)
PC_COUNTS = (2, 3)
COMPONENT_COUNTS = range(1, 9)


def select_model(scores):
    models, bics = [], []
    for covariance in COVARIANCES:
        for count in COMPONENT_COUNTS:
            model = fit_model(scores, covariance, count)
            models.append(model)
            bics.append(model.bic(scores))
    best = int(np.argmin(bics))
    comparison = pd.DataFrame({
        "covariance": [model.covariance_type for model in models],
        "n_components": [model.n_components for model in models],
        "bic": bics,
        "delta_bic": np.asarray(bics) - bics[best],
        "selected": np.arange(len(models)) == best,
    })
    return models[best], comparison


def analyze_area(pca_path, area):
    with np.load(pca_path) as pca:
        waveforms = pca["waveforms"]
        times_ms = pca["sample_time_ms"]
        scores = pca["scores"][:, :max(PC_COUNTS)]
        ids = pd.DataFrame({"brain_area": area, "mouse_id": pca["mouse_id"], "cluster_id": pca["cluster_id"]})
        preprocessing = read_preprocessing(pca)
        # Measure duration and plot shape in the original sample amplitudes after column-wise z-scoring.
        if preprocessing == Z_SCORE:
            waveforms = waveforms * pca["zscore_scale"] + pca["zscore_mean"]
        amplitude = pca["original_peak_to_peak"] if "original_peak_to_peak" in pca else np.ptp(waveforms, axis=1)
    if (scores.shape != (len(ids), max(PC_COUNTS)) or waveforms.shape != (len(ids), len(times_ms))
            or not np.isfinite(scores).all() or not np.isfinite(waveforms).all() or ids.duplicated().any()):
        raise ValueError(f"Invalid PCA inputs for area {area}")

    # Apply one scale factor to every PC so covariance regularization is small in score units.
    # The variance ratios among PCs are preserved.
    scale = scores[:, 0].std()
    if scale <= 0:
        raise ValueError(f"Area {area} has zero variance along PC1")
    scaled = scores / scale
    shape = pd.DataFrame([waveform_features(waveform)[0] for waveform in waveforms])
    measurements = ids.assign(
        pc1=scores[:, 0], pc2=scores[:, 1], pc3=scores[:, 2],
        trough_to_peak_ms=shape.trough_to_peak_ms,
        half_width_ms=shape.half_width_ms,
        peak_to_peak_native=amplitude,
    )
    bic_tables, assignments, summaries = [], [], []
    for n_pcs in PC_COUNTS:
        x = scaled[:, :n_pcs]
        model, comparison = select_model(x)
        probabilities = model.predict_proba(x)
        rows = measurements.assign(n_pcs=n_pcs, component=probabilities.argmax(axis=1) + 1,
                                   posterior_probability=probabilities.max(axis=1))
        components = rows.groupby("component").agg(
            n_units=("cluster_id", "size"),
            n_measurable_width=("trough_to_peak_ms", "count"),
            median_trough_to_peak_ms=("trough_to_peak_ms", "median"),
            median_half_width_ms=("half_width_ms", "median"),
            median_peak_to_peak_native=("peak_to_peak_native", "median"),
            median_posterior_probability=("posterior_probability", "median"),
        ).reset_index()
        summaries.append(components.assign(brain_area=area, n_pcs=n_pcs,
                                           fraction=components.n_units / len(rows)))
        assignments.append(rows)
        bic_tables.append(comparison.assign(brain_area=area, n_pcs=n_pcs, pc1_score_scale=scale))
        print(f"Area {area}, {n_pcs} PCs: BIC selected {model.n_components} components "
              f"with {model.covariance_type} covariance", flush=True)
    panel = times_ms, waveforms, assignments[-1].component.to_numpy()
    return assignments, summaries, bic_tables, panel, preprocessing


def plot_fits(assignments, comparison, waveform_panels, path, *, preprocessing=RAW):
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 8.6), layout="constrained")
    colors = plt.get_cmap("tab10")
    score_scale = 1e6 if preprocessing == RAW else 1
    score_units = " (×10⁻⁶ dataset units)" if preprocessing == RAW else ""
    amplitude_scale = 1 if preprocessing == PEAK_TO_PEAK else 1e6
    amplitude_label = "Normalized amplitude" if preprocessing == PEAK_TO_PEAK else "Amplitude (×10⁻⁶ dataset units)"
    if preprocessing == Z_SCORE:
        amplitude_label = f"Original {amplitude_label.lower()}"
    for col, area in enumerate(AREAS):
        area_rows = assignments.loc[assignments.brain_area.eq(area)]
        for row, n_pcs in enumerate(PC_COUNTS):
            ax = axes[row, col]
            rows = area_rows.loc[area_rows.n_pcs.eq(n_pcs)]
            model = comparison.loc[comparison.brain_area.eq(area) & comparison.n_pcs.eq(n_pcs)
                                   & comparison.selected].iloc[0]
            for component, group in rows.groupby("component"):
                ax.scatter(group.pc1 * score_scale, group.pc2 * score_scale, s=11, alpha=.72, linewidths=0,
                           color=colors(component - 1))
            ax.set_title(f"Area {area} · {model.n_components} fitted components", fontsize=10)
            if row == 1:
                ax.set_xlabel(f"PC1 score{score_units}")
            else:
                ax.tick_params(labelbottom=False)

        times_ms, waveforms, components = waveform_panels[col]
        ax = axes[2, col]
        for component in np.unique(components):
            traces = waveforms[components == component]
            ax.plot(times_ms, np.median(traces, axis=0) * amplitude_scale, color=colors(component - 1),
                    label=f"C{component} ({len(traces)})", linewidth=1.5)
        ax.set_xlabel("Time from waveform start (ms)")
        ax.legend(fontsize=7, frameon=False, loc="upper right", ncols=2)

    for row, n_pcs in enumerate(PC_COUNTS):
        axes[row, 0].set_ylabel(f"{n_pcs}-PC fit\nPC2 score{score_units}")
    axes[2, 0].set_ylabel(f"3-PC component medians\n{amplitude_label}")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(color="0.93")
    source = WAVEFORM_LABELS[preprocessing]
    fig.suptitle(f"Gaussian mixtures on PCA scores from {source}", fontsize=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pca-dir", type=Path, default=DEFAULT_OUTPUT / "pca")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.pca_dir / "gmm"

    assignments, summaries, bics, panels = [], [], [], []
    preprocessing_modes = set()
    with threadpool_limits(limits=1):
        for area in AREAS:
            units, components, comparison, waveforms, preprocessing = analyze_area(
                args.pca_dir / f"q7_pca_area_{area}.npz", area,
            )
            assignments.extend(units)
            summaries.extend(components)
            bics.extend(comparison)
            panels.append(waveforms)
            preprocessing_modes.add(preprocessing)
    if len(preprocessing_modes) != 1:
        raise ValueError("PCA archives must use the same preprocessing in all areas")
    assignments = pd.concat(assignments, ignore_index=True)
    comparison = pd.concat(bics, ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output / "q7_pca_gmm_units.csv", index=False)
    comparison.to_csv(output / "q7_pca_gmm_bic.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(output / "q7_pca_gmm_components.csv", index=False)
    plot_fits(assignments, comparison, panels, output / "q7_pca_gmm_clusters.png",
              preprocessing=preprocessing_modes.pop())


if __name__ == "__main__":
    main()

"""Extract Q7 waveform and spike-train features, and fit candidate classifiers.

Run with Python in the con2phys environment, then run q7_fast_spiking_stats.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path("E:/con2phys_submission/dataset_download/py-files")
DEFAULT_OUTPUT = REPO / "results" / "q7"
SAMPLES_PER_MS = 30
ACG_BIN_MS = 0.5
ACG_CENTERS_MS = (np.arange(100) + 0.5) * ACG_BIN_MS
WAVEFORM_GRID_MS = np.arange(-24, 40) / SAMPLES_PER_MS
WIDTH_FEATURES = ["trough_to_peak_ms", "half_width_ms", "recovery50_ms"]
CUTOFFS_MS = (0.35, 0.40, 0.425, 0.45, 0.50)
SEED = 703


def vertex(values, i):
    """Subsample location and amplitude from the three points around an extremum."""
    left, center, right = values[i - 1:i + 2]
    curvature = left - 2 * center + right
    shift = np.clip(0.5 * (left - right) / curvature, -0.5, 0.5) if curvature else 0.0
    return i + shift, center + 0.25 * (right - left) * shift


def crossing(values, first, second, level):
    return first + (second - first) * (level - values[first]) / (values[second] - values[first])


def waveform_features(waveform):
    if not np.isfinite(waveform).all():
        return {"waveform_qc": "nonfinite"}, None
    baseline = np.median(np.r_[waveform[:12], waveform[-12:]])
    y = waveform.astype(float) - baseline
    trough = int(np.argmin(y))
    peak = int(np.argmax(y))
    result = {"trough_sample": trough, "peak_to_peak_native": np.ptp(y)}
    if -y[trough] <= 0 or y[peak] > -y[trough]:
        return {**result, "waveform_qc": "positive_or_flat"}, None
    if trough < 24 or trough + 41 >= y.size:
        return {**result, "waveform_qc": "trough_near_edge"}, None
    t_trough, v_trough = vertex(y, trough)
    before = np.flatnonzero(y[:trough] >= v_trough / 2)
    after = np.flatnonzero(y[trough + 1:] >= v_trough / 2) + trough + 1
    if not before.size or not after.size:
        return {**result, "waveform_qc": "missing_half_width"}, None
    left = crossing(y, before[-1], before[-1] + 1, v_trough / 2)
    right = crossing(y, after[0] - 1, after[0], v_trough / 2)
    # Use the dominant recovery peak within 1.3 ms. Flag edge or weak peaks.
    end = trough + 39
    peak = int(np.argmax(y[trough + 2:end + 1])) + trough + 2
    if peak == end or y[peak] < -0.05 * v_trough or y[peak] <= y[peak - 1] or y[peak] < y[peak + 1]:
        return {**result, "waveform_qc": "unclear_recovery_peak"}, None
    t_peak, v_peak = vertex(y, peak)
    aligned = np.interp(t_trough + WAVEFORM_GRID_MS * SAMPLES_PER_MS, np.arange(y.size), y) / -v_trough
    return {
        **result,
        "waveform_qc": "pass",
        "trough_to_peak_ms": (t_peak - t_trough) / SAMPLES_PER_MS,
        "half_width_ms": (right - left) / SAMPLES_PER_MS,
        "recovery50_ms": (right - t_trough) / SAMPLES_PER_MS,
        "postpeak_to_trough": v_peak / -v_trough,
    }, aligned


def spike_features(spikes, duration):
    isi = np.diff(spikes)
    rate = spikes.size / duration
    result = {"n_spikes": spikes.size, "recording_duration_sec": duration, "firing_rate_hz": rate}
    if not isi.size:
        return result, np.full(ACG_CENTERS_MS.size, np.nan)
    denominator = isi[1:] + isi[:-1]
    cv2 = 2 * np.abs(isi[1:] - isi[:-1])[denominator > 0] / denominator[denominator > 0]
    result.update(isi_median_ms=1_000 * np.median(isi), isi_cv2=np.mean(cv2) if cv2.size else np.nan,
                  refractory_violation_fraction=np.mean(isi < 0.001))
    if spikes.size < 100:
        return result, np.full(ACG_CENTERS_MS.size, np.nan)
    counts = np.zeros(ACG_CENTERS_MS.size, dtype=np.int64)
    for offset in range(1, spikes.size):
        distances = spikes[offset:] - spikes[:-offset]
        distances = distances[distances < 0.05]
        if not distances.size:
            break
        bins = (distances * (1_000 / ACG_BIN_MS)).astype(int)
        counts += np.bincount(bins, minlength=counts.size)[:counts.size]
    # Burst index uses the ratio of per-bin ACG densities at 3–5 and 200–300 ms.
    late = np.searchsorted(spikes, spikes + 0.3) - np.searchsorted(spikes, spikes + 0.2)
    early = counts[6:10].sum()
    result["acg_burst_index"] = (early / 4) / (late.sum() / 200) if late.sum() else np.nan
    result["acg_2_to_10ms_relative_rate"] = counts[4:20].sum() / (spikes.size * 0.008 * rate)
    # Conditional rate divided by each unit's own session mean firing rate.
    return result, counts / (spikes.size * (ACG_BIN_MS / 1_000) * rate)


def load_features(root):
    rows, waveforms, acgs = [], [], []
    for mouse in range(1, 19):
        folder = root / str(mouse)
        metadata = np.load(folder / "brain_area.npy", allow_pickle=True).item()
        ids, areas = metadata["cluster_id"], metadata["brain_area"]
        raw = np.load(folder / "waveforms.npy")
        spikes = np.load(folder / "spikes.npy")
        clusters = np.load(folder / "clusters.npy")
        shapes = [np.load(folder / f"lfp_{area}.npy", mmap_mode="r").shape for area in (1, 2, 3)]
        if len({shape[1] for shape in shapes}) != 1:
            raise ValueError(f"Mismatched recording lengths for mouse {mouse}")
        duration = shapes[0][1] / 500
        if (raw.shape != (len(ids), 128) or areas.shape != ids.shape or set(areas) != {1, 2, 3}
                or not np.array_equal(np.unique(clusters), np.sort(ids)) or spikes.shape != clusters.shape
                or not np.isfinite(spikes).all() or np.any(np.diff(spikes) < 0)):
            raise ValueError(f"Invalid waveform metadata or spike alignment for mouse {mouse}")
        order = np.argsort(clusters, kind="stable")
        sorted_spikes = spikes[order]
        sorted_ids, counts = np.unique(clusters, return_counts=True)
        groups = dict(zip(sorted_ids, np.split(sorted_spikes, counts.cumsum()[:-1])))
        for cluster, area, waveform in zip(ids, areas, raw):
            shape_features, aligned = waveform_features(waveform)
            timing_features, acg = spike_features(groups[cluster], duration)
            rows.append({"mouse_id": mouse, "cluster_id": int(cluster), "brain_area": int(area),
                         **shape_features, **timing_features})
            waveforms.append(aligned if aligned is not None else np.full(WAVEFORM_GRID_MS.size, np.nan))
            acgs.append(acg)
        print(f"Mouse {mouse:2}: extracted {len(ids)} units, spike times {spikes[0]:.3f} to "
              f"{spikes[-1]:.3f} s, LFP duration {duration:.3f} s", flush=True)
    units = pd.DataFrame(rows)
    if len(units) != 1449 or units[["mouse_id", "cluster_id"]].duplicated().any():
        raise ValueError("Expected exactly 1,449 distinct mouse/cluster IDs")
    return units, np.stack(waveforms), np.stack(acgs)


def fit_classifier(features, widths, *, n_init=12):
    scaled = StandardScaler().fit_transform(features)
    model = GaussianMixture(2, covariance_type="tied", n_init=n_init, random_state=SEED).fit(scaled)
    posterior = model.predict_proba(scaled)
    centers = np.average(np.broadcast_to(widths, (2, widths.size)), axis=1, weights=posterior.T)
    narrow = np.argmin(centers)
    return posterior[:, narrow], model, centers, scaled


def classify(units, waveforms):
    valid = units.waveform_qc.eq("pass") & units.refractory_violation_fraction.le(0.01)
    units["qc_status"] = units.waveform_qc.where(~units.waveform_qc.eq("pass"), "spike_refractory_violations")
    units.loc[valid, "qc_status"] = "pass"
    widths = units.loc[valid, "trough_to_peak_ms"].to_numpy()
    waveform_x = np.log(units.loc[valid, WIDTH_FEATURES].to_numpy())
    probability, model, centers, scaled = fit_classifier(waveform_x, widths)
    units.loc[valid, "p_fs_waveform"] = probability
    complete = valid & units[["acg_2_to_10ms_relative_rate", "isi_cv2"]].notna().all(axis=1)
    joint_x = np.column_stack([
        np.log(units.loc[complete, WIDTH_FEATURES].to_numpy()),
        np.log(units.loc[complete, "firing_rate_hz"]),
        np.log1p(units.loc[complete, "acg_2_to_10ms_relative_rate"]),
        units.loc[complete, "isi_cv2"],
    ])
    p_joint, _, joint_centers, _ = fit_classifier(joint_x, units.loc[complete, "trough_to_peak_ms"].to_numpy())
    units.loc[complete, "p_fs_joint"] = p_joint
    pca_x = PCA(n_components=4, svd_solver="full").fit_transform(waveforms[valid])
    p_pca, _, pca_centers, _ = fit_classifier(pca_x, widths)
    units.loc[valid, "p_fs_pca"] = p_pca
    variants = {"waveform": "p_fs_waveform", "joint": "p_fs_joint", "pca": "p_fs_pca"}
    for name, column in variants.items():
        p = units[column]
        units[f"fs_{name}"] = np.where(p.notna(), (p >= 0.5).astype(float), np.nan)
    for cutoff in CUTOFFS_MS:
        units[f"fs_cutoff_{int(round(cutoff * 1000))}us"] = np.where(valid, units.trough_to_peak_ms.le(cutoff), np.nan)
    bic = {str(k): GaussianMixture(k, covariance_type="tied", n_init=12, random_state=SEED).fit(scaled).bic(scaled)
           for k in (1, 3, 4)}
    bic["2"] = model.bic(scaled)
    diagnostics = {"bic_by_components": bic, "primary_component_mean_widths_ms": centers.tolist(),
                   "joint_component_mean_widths_ms": joint_centers.tolist(),
                   "pca_component_mean_widths_ms": pca_centers.tolist(),
                   "primary_ambiguous_p_025_to_075": int(((probability > .25) & (probability < .75)).sum()),
                   "primary_gmm_converged": bool(model.converged_)}
    return diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        units, waveforms, acgs = load_features(args.input)
        diagnostics = classify(units, waveforms)
    args.output.mkdir(parents=True, exist_ok=True)
    units.to_csv(args.output / "q7_units.csv", index=False)
    np.savez_compressed(args.output / "q7_unit_profiles.npz", waveform=waveforms, acg=acgs,
                        waveform_grid_ms=WAVEFORM_GRID_MS, acg_centers_ms=ACG_CENTERS_MS,
                        mouse_id=units.mouse_id.to_numpy(), cluster_id=units.cluster_id.to_numpy())
    config = {"input": str(args.input), "sampling_rate_hz": 30_000, "acg_bin_ms": ACG_BIN_MS,
              "primary_log_waveform_features": WIDTH_FEATURES, "maximum_sub_1ms_isi_fraction": 0.01,
              "cutoffs_ms": CUTOFFS_MS, "seed": SEED, **diagnostics}
    (args.output / "q7_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()

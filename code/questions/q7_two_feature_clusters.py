"""Explore waveform cluster separation independently within the three Q7 areas."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from q7_fast_spiking import DEFAULT_OUTPUT, SEED


TROUGH_TO_PEAK = "trough_to_peak_ms"
HALF_WIDTH = "half_width_ms"
END_SLOPE = "end_slope_per_ms"
HALF_WIDTH_FEATURES = (TROUGH_TO_PEAK, HALF_WIDTH)
END_SLOPE_FEATURES = (TROUGH_TO_PEAK, END_SLOPE)
THREE_FEATURES = (TROUGH_TO_PEAK, HALF_WIDTH, END_SLOPE)
LOG_FEATURES = {TROUGH_TO_PEAK, HALF_WIDTH, "firing_rate_hz", "isi_median_ms"}
END_SLOPE_UNITS = DEFAULT_OUTPUT / "simple_approach" / "q7_end_slope_units.csv"
COVARIANCES = {"tied": "Shared covariance", "full": "Separate covariances"}


def fit_model(x, covariance, n_components, n_init=12):
    model = GaussianMixture(n_components, covariance_type=covariance, n_init=n_init,
                            max_iter=300, tol=1e-5, random_state=SEED).fit(x)
    if not model.converged_:
        model.set_params(max_iter=1500).fit(x)
    if not model.converged_:
        raise RuntimeError(f"Gaussian mixture did not converge ({covariance}, {n_components} components)")
    return model


def narrow_probability(model, x):
    narrow = np.argmin(model.means_[:, 0])
    return model.predict_proba(x)[:, narrow]


def density_modes(model):
    """Locate density maxima along the Gaussian ridgeline in fitted feature space."""
    # Ray and Lindsay (2005), doi:10.1214/009053605000000417. All critical points
    # lie on this curve for a two-component Gaussian mixture.
    precision = model.precisions_
    if model.covariance_type == "tied":
        precision = np.stack([precision, precision])
    alpha = np.linspace(0, 1, 2001)
    weight = np.column_stack([1 - alpha, alpha])
    matrix = np.einsum("nk,kij->nij", weight, precision)
    rhs = np.einsum("nk,kij,kj->ni", weight, precision, model.means_)
    ridge = np.linalg.solve(matrix, rhs[..., None])[..., 0]
    log_density = model.score_samples(ridge)
    peaks, _ = find_peaks(log_density)
    if log_density[0] > log_density[1]:
        peaks = np.r_[0, peaks]
    if log_density[-1] > log_density[-2]:
        peaks = np.r_[peaks, len(ridge) - 1]
    return ridge[peaks]


def held_out_mice(x, mice, covariance):
    labels = np.empty(len(x), dtype=bool)
    score_gains = []
    for mouse in np.unique(mice):
        train = mice != mouse
        one, two = [fit_model(x[train], covariance, k, n_init=6) for k in (1, 2)]
        labels[~train] = narrow_probability(two, x[~train]) >= 0.5
        score_gains.append(two.score(x[~train]) - one.score(x[~train]))
    return labels, np.array(score_gains)


def bootstrap_checks(x, mice, covariance, full_labels, n_boot):
    rng = np.random.default_rng(SEED)
    blocks = [np.flatnonzero(mice == mouse) for mouse in np.unique(mice)]
    narrow_count, mode_count, overlaps = np.empty((3, n_boot))
    for i in range(n_boot):
        selected = rng.integers(len(blocks), size=len(blocks))
        indices = np.concatenate([blocks[j] for j in selected])
        model = fit_model(x[indices], covariance, 2, n_init=4)
        labels = narrow_probability(model, x) >= 0.5
        narrow_count[i] = labels.mean()
        mode_count[i] = len(density_modes(model))
        overlaps[i] = np.sum(labels & full_labels) / np.sum(labels | full_labels)
    return narrow_count, mode_count, overlaps


def transform_features(values, features, *, inverse=False):
    """Log positive timing and rate features and retain signed slopes on their original scale."""
    transformed = values.copy()
    logged = [index for index, feature in enumerate(features) if feature in LOG_FEATURES]
    transform = np.exp if inverse else np.log
    transformed[:, logged] = transform(transformed[:, logged])
    return transformed


def analyze(units, n_boot, features=HALF_WIDTH_FEATURES, *, max_components=4):
    summaries, scores, assignments, panels = [], [], [], []
    for area, rows in units.groupby("brain_area", sort=True):
        x = transform_features(rows[list(features)].to_numpy(), features)
        mice = rows.mouse_id.to_numpy()
        for covariance in COVARIANCES:
            models = [fit_model(x, covariance, k) for k in range(1, max_components + 1)]
            bic = np.array([model.bic(x) for model in models])
            scores.extend({"brain_area": area, "covariance": covariance, "components": k, "bic": value,
                           "delta_bic": value - bic.min()} for k, value in enumerate(bic, 1))
            two = models[1]
            p = narrow_probability(two, x)
            labels = p >= 0.5
            held_out, gain = held_out_mice(x, mice, covariance)
            modes = density_modes(two)
            narrow_fraction, boot_modes, overlap = bootstrap_checks(x, mice, covariance, labels, n_boot)
            centers = transform_features(two.means_[np.argsort(two.means_[:, 0])], features, inverse=True)
            extra_centers = {f"{label}_center_{feature}": centers[i, j]
                             for j, feature in enumerate(features[1:], start=1)
                             for i, label in enumerate(("short", "long"))}
            summaries.append({
                "brain_area": area, "covariance": covariance, "n_units": len(x),
                "n_mice": len(gain), "best_bic_components": bic.argmin() + 1,
                "bic_one_minus_two": bic[0] - bic[1], "modes_in_two_component_fit": len(modes),
                "bootstrap_two_mode_fraction": np.mean(boot_modes == 2),
                "held_out_mean_log_likelihood_gain": gain.mean(), "held_out_mice_favoring_two": np.sum(gain > 0),
                "held_out_narrow_jaccard": np.sum(held_out & labels) / np.sum(held_out | labels),
                "bootstrap_narrow_jaccard_median": np.median(overlap),
                "bootstrap_narrow_jaccard_p05": np.quantile(overlap, 0.05),
                "n_short_component": np.sum(labels), "short_fraction": labels.mean(),
                "bootstrap_model_short_fraction_p025": np.quantile(narrow_fraction, 0.025),
                "bootstrap_model_short_fraction_p975": np.quantile(narrow_fraction, 0.975),
                "n_with_p_between_0_25_and_0_75": np.sum((p > 0.25) & (p < 0.75)),
                "short_center_ttp_ms": centers[0, 0], "long_center_ttp_ms": centers[1, 0],
                **extra_centers,
            })
            assignments.append(rows[["mouse_id", "brain_area", "cluster_id", *features]].assign(
                covariance=covariance, p_short_component=p, short_component=labels, held_out_short_component=held_out))
            panels.append((area, covariance, rows, two, p, modes))
            print(f"Area {area}, {covariance}: {len(modes)} density modes, best BIC at {bic.argmin()+1} components", flush=True)
    return pd.DataFrame(summaries), pd.DataFrame(scores), pd.concat(assignments, ignore_index=True), panels


def save_plot(panels, output, features=HALF_WIDTH_FEATURES):
    y_feature = features[-1]
    projection = len(features) > 2
    x_limits = (.07, 1.2)
    y_limits = (.05, .56)
    y_label = "Half-width (ms)"
    title = "Two-feature waveform mixtures fitted separately by area"
    if y_feature != HALF_WIDTH:
        slopes = pd.concat([rows[y_feature] for _, _, rows, _, _, _ in panels])
        y_limits = (np.floor(slopes.min()), np.ceil(slopes.max()))
        max_duration = max(rows[TROUGH_TO_PEAK].max() for _, _, rows, _, _, _ in panels)
        x_limits = (x_limits[0], max(x_limits[1], np.ceil(max_duration * 10) / 10))
        y_label = "End-slope (trough depths per ms)"
        title = "Trough-to-peak and end-slope mixtures fitted separately by area"
    if projection:
        title = (f"{len(features)}-feature mixtures fitted separately by area\n"
                 "Trough-to-peak and end-slope projection")
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex=True, sharey=True, layout="constrained")
    if not projection:
        gx, gy = np.meshgrid(np.linspace(*x_limits, 300), np.linspace(*y_limits, 240))
        grid = transform_features(np.column_stack([gx.ravel(), gy.ravel()]), features)
    for area, covariance, rows, model, p, modes in panels:
        axis = axes[list(COVARIANCES).index(covariance), area - 1]
        points = axis.scatter(rows[TROUGH_TO_PEAK], rows[y_feature], c=p, cmap="viridis", s=12,
                              alpha=.76, vmin=0, vmax=1, linewidths=0)
        if not projection:
            probability = narrow_probability(model, grid).reshape(gx.shape)
            axis.contour(gx, gy, probability, levels=[.5], colors=["#334155"], linewidths=1, linestyles="--")
        if len(modes):
            observed_modes = transform_features(modes, features, inverse=True)
            axis.scatter(observed_modes[:, 0], observed_modes[:, -1], marker="X", c="#111827",
                         edgecolors="white", linewidths=.8,
                         s=70, zorder=4, label="Fitted density mode")
        axis.set(xlim=x_limits, ylim=y_limits)
        axis.set_title(f"Area {area} · {len(modes)} fitted {'mode' if len(modes) == 1 else 'modes'}", fontsize=10)
        axis.spines[["right", "top"]].set_visible(False)
    for i, label in enumerate(COVARIANCES.values()):
        axes[i, 0].set_ylabel(f"{label}\n{y_label}")
    axes[0, 2].legend(loc="upper right", fontsize=8, frameon=False)
    fig.supxlabel("Trough-to-peak duration (ms)")
    fig.suptitle(title)
    fig.colorbar(points, ax=axes, label="Probability of shorter component", shrink=.85)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, default=DEFAULT_OUTPUT / "q7_units.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "simple_approach")
    parser.add_argument("--end-slope-units", type=Path, nargs="?", const=END_SLOPE_UNITS,
                        help="Use the 0.33 ms post-trough end-slope measurements")
    parser.add_argument("--three-features", action="store_true",
                        help="Fit trough-to-peak duration, half-width, and end-slope together")
    parser.add_argument("--bootstrap", type=int, default=300)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap requires at least 100 draws")
    table = pd.read_csv(args.units)
    units = table.loc[table.qc_status.eq("pass")].copy()
    features = HALF_WIDTH_FEATURES
    prefix = "q7_two_feature_cluster"
    if args.three_features:
        features = THREE_FEATURES
        prefix = "q7_three_feature_cluster"
    elif args.end_slope_units is not None:
        features = END_SLOPE_FEATURES
        prefix = "q7_end_slope_cluster"
    if END_SLOPE in features:
        keys = ["mouse_id", "cluster_id", "brain_area"]
        slopes = pd.read_csv(args.end_slope_units or END_SLOPE_UNITS, usecols=[*keys, END_SLOPE])
        units = units.merge(slopes, on=keys, how="outer", validate="one_to_one", indicator=True)
        if not units["_merge"].eq("both").all():
            raise ValueError("End-slope measurements must match the quality-passed Q7 units")

    duration_features = [feature for feature in features if feature != END_SLOPE]
    if (set(units.brain_area) != {1, 2, 3} or units.duplicated(["mouse_id", "cluster_id"]).any()
            or not np.isfinite(units[list(features)]).all().all() or not units[duration_features].gt(0).all().all()):
        raise ValueError("Expected finite features, positive durations and unique unit IDs in three areas")
    with threadpool_limits(limits=1):
        summary, scores, assignments, panels = analyze(units, args.bootstrap, features)
    args.output.mkdir(parents=True, exist_ok=True)
    summary.assign(n_bootstrap=args.bootstrap).to_csv(args.output / f"{prefix}_summary.csv", index=False)
    scores.to_csv(args.output / f"{prefix}_bic.csv", index=False)
    assignments.to_csv(args.output / f"{prefix}_units.csv", index=False)
    save_plot(panels, args.output / f"{prefix}s.png", features)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()

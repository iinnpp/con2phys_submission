"""Cluster waveform duration and the Sun et al. post-trough slope by brain area."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from q7_fast_spiking import DEFAULT_OUTPUT
from q7_two_feature_clusters import COVARIANCES, TROUGH_TO_PEAK, analyze, fit_model, save_plot, transform_features


END_SLOPE = "sun2021_end_slope_per_ms"
FEATURES = (TROUGH_TO_PEAK, END_SLOPE)
KEYS = ["mouse_id", "brain_area", "cluster_id"]
N_BOOTSTRAP = 300
MAX_COMPONENTS = 8


def fit_bic_winners(units, winners, features=FEATURES):
    assignments, components = [], []
    for choice in winners.itertuples():
        rows = units.loc[units.brain_area.eq(choice.brain_area)].copy()
        scores = transform_features(rows[list(features)].to_numpy(), features)
        model = fit_model(scores, choice.covariance, choice.components)
        order = np.argsort(model.means_[:, 0])
        posteriors = model.predict_proba(scores)[:, order]
        rows = rows.assign(component=posteriors.argmax(axis=1) + 1, posterior_probability=posteriors.max(axis=1))
        assignments.append(rows)
        by_component = rows.groupby("component").agg(
            n_units=("cluster_id", "size"),
            n_positive_dominant=("positive_dominant", "sum"),
            **{f"median_{feature}": (feature, "median") for feature in features},
            median_posterior_probability=("posterior_probability", "median"),
        ).reset_index()
        components.append(by_component.assign(
            brain_area=choice.brain_area, covariance=choice.covariance,
            n_components=choice.components, fraction=by_component.n_units / len(rows),
        ))
    return pd.concat(assignments, ignore_index=True), pd.concat(components, ignore_index=True)


def plot_bic_winners(assignments, components, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), sharex=True, sharey=True, layout="constrained")
    colors = plt.get_cmap("tab10")
    for ax, (area, rows) in zip(axes, assignments.groupby("brain_area")):
        stats = components.loc[components.brain_area.eq(area)]
        for number, group in rows.groupby("component"):
            ax.scatter(group[TROUGH_TO_PEAK], group[END_SLOPE], s=13, alpha=.75, linewidths=0,
                       color=colors(number - 1), label=f"C{number} ({len(group)})")
        covariance = COVARIANCES[stats.covariance.iloc[0]].lower()
        ax.set_title(f"Area {area} · {stats.n_components.iloc[0]} components\n{covariance}", fontsize=11)
        ax.axhline(0, color="0.6", linewidth=.8, linestyle="--")
        ax.spines[["right", "top"]].set_visible(False)
        ax.legend(loc="upper right", fontsize=7.5, frameon=False, ncols=2)
    axes[0].set_ylabel("Sun et al. end-slope (trough depths per ms)")
    fig.supxlabel("Trough-to-peak duration (ms)")
    fig.suptitle("BIC-selected waveform mixtures by brain area")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, default=DEFAULT_OUTPUT / "waveform_features" / "q7_waveform_features.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "features_clustering_2features")
    args = parser.parse_args()

    table = pd.read_csv(args.units, usecols=[*KEYS, *FEATURES, "positive_dominant"])
    units = table.dropna(subset=list(FEATURES)).copy()
    if (set(units.brain_area) != {1, 2, 3} or table.duplicated(["mouse_id", "cluster_id"]).any()
            or not np.isfinite(units[list(FEATURES)]).all().all() or not units[TROUGH_TO_PEAK].gt(0).all()):
        raise ValueError("Expected unique unit IDs, finite features, and positive durations within each area")

    with threadpool_limits(limits=1):
        summary, bics, two_component, panels = analyze(units, N_BOOTSTRAP, FEATURES, max_components=MAX_COMPONENTS)
        bics["delta_bic_within_area"] = bics.bic - bics.groupby("brain_area").bic.transform("min")
        winner_indices = bics.groupby("brain_area").bic.idxmin()
        bics["selected"] = bics.index.isin(winner_indices)
        selected_units, selected_components = fit_bic_winners(units, bics.loc[winner_indices])

    args.output.mkdir(parents=True, exist_ok=True)
    prefix = "q7_sun_end_slope_gmm"
    summary.assign(n_bootstrap=N_BOOTSTRAP).to_csv(args.output / f"{prefix}_two_component_summary.csv", index=False)
    two_component.to_csv(args.output / f"{prefix}_two_component_units.csv", index=False)
    bics.to_csv(args.output / f"{prefix}_bic.csv", index=False)
    selected_units.to_csv(args.output / f"{prefix}_bic_selected_units.csv", index=False)
    selected_components.to_csv(args.output / f"{prefix}_bic_selected_components.csv", index=False)
    save_plot(panels, args.output / f"{prefix}_two_component.png", FEATURES)
    plot_bic_winners(selected_units, selected_components, args.output / f"{prefix}_bic_selected.png")
    excluded = len(table) - len(units)
    print(f"Fitted {len(units)} units across all areas. Missing either measure: {excluded}.")
    print(bics.loc[winner_indices, ["brain_area", "covariance", "components", "bic"]].to_string(index=False))


if __name__ == "__main__":
    main()

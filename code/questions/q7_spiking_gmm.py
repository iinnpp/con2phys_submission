"""Fit per-area Gaussian mixtures of firing rate and median interspike interval."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from q7_fast_spiking import DEFAULT_OUTPUT
from q7_two_feature_clusters import COVARIANCES, bootstrap_checks, density_modes, fit_model, narrow_probability


ISI = "isi_median_ms"
RATE = "firing_rate_hz"
FEATURES = [ISI, RATE]
KEYS = ["mouse_id", "cluster_id", "brain_area"]
COMPONENT_COUNTS = range(1, 9)
N_BOOTSTRAP = 300


def assign_components(rows, model, x):
    """Assign component IDs in increasing order of fitted log ISI."""
    order = np.argsort(model.means_[:, 0])
    posterior = model.predict_proba(x)[:, order]
    return rows.assign(component=posterior.argmax(axis=1) + 1, posterior_probability=posterior.max(axis=1))


def analyze(units):
    bics, selected_units, selected_components, two_units, two_summaries = [], [], [], [], []
    for area, rows in units.groupby("brain_area", sort=True):
        x = np.log(rows[FEATURES].to_numpy())
        mice = rows.mouse_id.to_numpy()
        models = {(cov, k): fit_model(x, cov, k) for cov in COVARIANCES for k in COMPONENT_COUNTS}
        bic = {key: model.bic(x) for key, model in models.items()}
        winner = min(bic, key=bic.get)
        bics.extend({"brain_area": area, "covariance": cov, "n_components": k, "bic": value,
                     "delta_bic": value - bic[winner], "selected": (cov, k) == winner}
                    for (cov, k), value in bic.items())

        selected = assign_components(rows, models[winner], x)
        component_stats = selected.groupby("component").agg(
            n_units=("cluster_id", "size"),
            median_firing_rate_hz=(RATE, "median"),
            median_isi_ms=(ISI, "median"),
            median_posterior_probability=("posterior_probability", "median"),
        ).reset_index()
        selected_units.append(selected.assign(covariance=winner[0], n_components=winner[1]))
        selected_components.append(component_stats.assign(
            brain_area=area, covariance=winner[0], n_components=winner[1], fraction=component_stats.n_units / len(rows),
        ))

        for covariance in COVARIANCES:
            model = models[covariance, 2]
            probability = narrow_probability(model, x)
            short = probability >= .5
            fraction, modes, overlap = bootstrap_checks(x, mice, covariance, short, N_BOOTSTRAP)
            two_units.append(rows.assign(covariance=covariance, short_isi_component=short,
                                         p_short_isi_component=probability))
            two_summaries.append({
                "brain_area": area, "covariance": covariance, "n_units": len(rows), "n_mice": len(np.unique(mice)),
                "n_short_isi": short.sum(), "short_isi_fraction": short.mean(),
                "short_component_median_isi_ms": rows.loc[short, ISI].median(),
                "long_component_median_isi_ms": rows.loc[~short, ISI].median(),
                "short_component_median_firing_rate_hz": rows.loc[short, RATE].median(),
                "long_component_median_firing_rate_hz": rows.loc[~short, RATE].median(),
                "modes_in_two_component_fit": len(density_modes(model)),
                "bootstrap_two_mode_fraction": np.mean(modes == 2),
                "bootstrap_short_fraction_p025": np.quantile(fraction, .025),
                "bootstrap_short_fraction_p975": np.quantile(fraction, .975),
                "bootstrap_short_jaccard_median": np.median(overlap),
                "n_bootstrap": N_BOOTSTRAP,
            })
        print(f"Area {area}: BIC selected {winner[1]} components with {COVARIANCES[winner[0]].lower()}", flush=True)

    return {
        "bic": pd.DataFrame(bics),
        "bic_selected_units": pd.concat(selected_units, ignore_index=True),
        "bic_selected_components": pd.concat(selected_components, ignore_index=True),
        "two_component_units": pd.concat(two_units, ignore_index=True),
        "two_component_summary": pd.DataFrame(two_summaries),
    }


def plot_selected(units, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.25), sharex=True, sharey=True, layout="constrained")
    colors = plt.get_cmap("tab10")
    for ax, (area, rows) in zip(axes, units.groupby("brain_area", sort=True)):
        for component, group in rows.groupby("component"):
            ax.scatter(group[RATE], group[ISI], s=14, color=colors(component - 1),
                       alpha=.75, linewidths=0, label=f"C{component} ({len(group)})")
        ax.set(xscale="log", yscale="log")
        count = rows.n_components.iloc[0]
        covariance = COVARIANCES[rows.covariance.iloc[0]].lower()
        ax.set_title(f"Area {area} · {count} {'component' if count == 1 else 'components'}\n{covariance}", fontsize=11)
        ax.grid(color=".92")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper right", fontsize=7.5, frameon=False, ncols=2)
    axes[0].set_ylabel("Per-unit median ISI (ms, log scale)")
    fig.supxlabel("Firing rate (Hz, log scale)")
    fig.suptitle("BIC-selected firing-pattern mixtures by brain area", fontsize=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, default=DEFAULT_OUTPUT / "firing_features" / "q7_firing_features.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "features_clustering_2features_spiking")
    args = parser.parse_args()

    units = pd.read_csv(args.units, usecols=[*KEYS, *FEATURES])
    if (units.duplicated(["mouse_id", "cluster_id"]).any() or set(units.brain_area) != {1, 2, 3}
            or not np.isfinite(units[FEATURES]).all().all() or not units[FEATURES].gt(0).all().all()):
        raise ValueError("Expected unique unit IDs and positive, finite firing rate and median ISI in three areas")

    with threadpool_limits(limits=1):
        tables = analyze(units)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.output / f"q7_spiking_gmm_{name}.csv", index=False)
    plot_selected(tables["bic_selected_units"], args.output / "q7_spiking_gmm_bic_selected.png")


if __name__ == "__main__":
    main()

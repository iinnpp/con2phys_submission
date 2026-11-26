"""Cluster waveform shape and firing patterns jointly within each Q7 brain area."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from q7_fast_spiking import DEFAULT_OUTPUT
from q7_spiking_gmm import ISI, RATE
from q7_sun_end_slope_gmm import END_SLOPE, KEYS, MAX_COMPONENTS, N_BOOTSTRAP, fit_bic_winners
from q7_two_feature_clusters import COVARIANCES, TROUGH_TO_PEAK, analyze, save_plot


FEATURES = (TROUGH_TO_PEAK, RATE, ISI, END_SLOPE)


def load_units(waveform_path, firing_path):
    waveforms = pd.read_csv(waveform_path, usecols=[*KEYS, TROUGH_TO_PEAK, END_SLOPE, "positive_dominant"])
    firing = pd.read_csv(firing_path, usecols=[*KEYS, RATE, ISI])
    table = waveforms.merge(firing, on=KEYS, how="outer", validate="one_to_one", indicator=True)
    if not table["_merge"].eq("both").all() or table.duplicated(["mouse_id", "cluster_id"]).any():
        raise ValueError("Waveform and firing feature tables disagree on unit IDs")
    units = table.drop(columns="_merge").dropna(subset=list(FEATURES))
    if (set(units.brain_area) != {1, 2, 3} or not np.isfinite(units[list(FEATURES)]).all().all()
            or not units[[TROUGH_TO_PEAK, RATE, ISI]].gt(0).all().all()):
        raise ValueError("Expected finite features and positive durations and rates in all three areas")
    return units, len(table) - len(units)


def plot_projections(assignments, components, path):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), sharex="row", sharey="row", layout="constrained")
    colors = plt.get_cmap("tab10")
    for col, (area, rows) in enumerate(assignments.groupby("brain_area", sort=True)):
        waveform, firing = axes[:, col]
        stats = components.loc[components.brain_area.eq(area)].iloc[0]
        for number, group in rows.groupby("component"):
            color = colors(number - 1)
            waveform.scatter(group[TROUGH_TO_PEAK], group[END_SLOPE], s=13, alpha=.75, linewidths=0,
                             color=color, label=f"C{number} ({len(group)})")
            firing.scatter(group[RATE], group[ISI], s=13, alpha=.75, linewidths=0, color=color)
        covariance = COVARIANCES[stats.covariance].lower()
        waveform.set_title(f"Area {area} · {stats.n_components} components\n{covariance}", fontsize=11)
        waveform.axhline(0, color="0.6", linewidth=.8, linestyle="--")
        waveform.set_xlabel("Trough-to-peak duration (ms)")
        waveform.legend(loc="upper right", fontsize=7.5, frameon=False, ncols=2)
        firing.set(xscale="log", yscale="log", xlabel="Firing rate (Hz, log scale)")
        for ax in (waveform, firing):
            ax.spines[["right", "top"]].set_visible(False)
            ax.grid(color="0.93")
            ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("Sun et al. end-slope (trough depths per ms)")
    axes[1, 0].set_ylabel("Per-unit median ISI (ms, log scale)")
    fig.suptitle("BIC-selected four-feature GMMs by brain area\nColors show the same joint assignments in both views",
                 fontsize=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waveform-features", type=Path, default=DEFAULT_OUTPUT / "waveform_features" / "q7_waveform_features.csv")
    parser.add_argument("--firing-features", type=Path, default=DEFAULT_OUTPUT / "firing_features" / "q7_firing_features.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "features_clustering_4features")
    args = parser.parse_args()

    units, excluded = load_units(args.waveform_features, args.firing_features)
    with threadpool_limits(limits=1):
        two_summary, bics, two_units, panels = analyze(units, N_BOOTSTRAP, FEATURES, max_components=MAX_COMPONENTS)
        bics["delta_bic_within_area"] = bics.bic - bics.groupby("brain_area").bic.transform("min")
        winners = bics.groupby("brain_area").bic.idxmin()
        bics["selected"] = bics.index.isin(winners)
        selected_units, selected_components = fit_bic_winners(units, bics.loc[winners], FEATURES)

    args.output.mkdir(parents=True, exist_ok=True)
    prefix = "q7_four_feature_gmm"
    two_summary.assign(n_bootstrap=N_BOOTSTRAP).to_csv(args.output / f"{prefix}_two_component_summary.csv", index=False)
    two_units.to_csv(args.output / f"{prefix}_two_component_units.csv", index=False)
    bics.to_csv(args.output / f"{prefix}_bic.csv", index=False)
    selected_units.to_csv(args.output / f"{prefix}_bic_selected_units.csv", index=False)
    selected_components.to_csv(args.output / f"{prefix}_bic_selected_components.csv", index=False)
    plot_projections(selected_units, selected_components, args.output / f"{prefix}_bic_selected.png")
    save_plot(panels, args.output / f"{prefix}_two_component.png", FEATURES)
    print(f"Fitted {len(units):,} units. Missing one or more features: {excluded}.")
    print(bics.loc[winners, ["brain_area", "covariance", "components", "bic"]].to_string(index=False))


if __name__ == "__main__":
    main()

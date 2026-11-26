"""Compare Q7 prevalence across areas and summarize classification checks."""

import argparse
import json
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, roots_hermitenorm
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from q7_fast_spiking import DEFAULT_OUTPUT, SEED, WIDTH_FEATURES, fit_classifier


AREAS = [1, 2, 3]
PAIRS = np.array(list(combinations(range(3), 2)))
COLORS = ['#0072B2', '#D55E00', '#00856A']
METHODS = {
    "fs_waveform": "Waveform mixture (primary)",
    "p_fs_waveform": "Waveform probabilities",
    "fs_joint": "Waveform + spike train",
    "fs_pca": "Whole-waveform PCA",
    "fs_mixture4_narrowest": "Shortest of four waveform components",
    "fs_cutoff_350us": "Width ≤ 0.350 ms",
    "fs_cutoff_400us": "Width ≤ 0.400 ms",
    "fs_cutoff_425us": "Width ≤ 0.425 ms",
    "fs_cutoff_450us": "Width ≤ 0.450 ms",
    "fs_cutoff_500us": "Width ≤ 0.500 ms",
}
CORE_METHODS = ["p_fs_waveform", "fs_joint", "fs_pca", "fs_cutoff_400us", "fs_cutoff_425us", "fs_cutoff_450us"]
SPIKE_METRICS = {
    "firing_rate_hz": "Firing rate (Hz)",
    "isi_median_ms": "Median ISI (ms)",
    "isi_cv2": "ISI CV2",
    "acg_burst_index": "ACG burst index (3–5 ms / 200–300 ms)",
    "acg_2_to_10ms_relative_rate": "Rate-normalized 2–10 ms ACG",
}


@lru_cache
def normal_quadrature(quadrature):
    nodes, weights = roots_hermitenorm(quadrature)
    return nodes, weights / np.sqrt(2 * np.pi)


def fit_prevalence(counts, totals, *, quadrature=64):
    """Binomial area fixed effects plus a normally distributed mouse intercept."""
    nodes, weights = normal_quadrature(quadrature)

    def objective(params):
        sigma = np.sqrt(params[3])
        logits = params[:3] + sigma * nodes[:, None]
        probs = expit(logits)
        log_joint = counts @ logits.T - totals @ np.logaddexp(0, logits).T + np.log(weights)
        log_integral = logsumexp(log_joint, axis=1, keepdims=True)
        node_probs = np.exp(log_joint - log_integral)
        residual = (counts[:, None, :] - totals[:, None, :] * probs) * node_probs[:, :, None]
        if sigma > 1e-4:
            variance_gradient = np.sum(residual * nodes[None, :, None]) / (2 * sigma)
        else:
            # Right derivative at variance zero from the second derivative of the likelihood.
            p = expit(params[:3])
            variance_gradient = 0.5 * np.sum(((counts - totals * p).sum(axis=1)) ** 2 - totals @ (p * (1 - p)))
        gradient = np.r_[residual.sum(axis=(0, 1)), variance_gradient]
        return -log_integral.sum(), -gradient

    initial = np.log((counts.sum(axis=0) + 0.5) / (totals.sum(axis=0) - counts.sum(axis=0) + 0.5))
    bounds = [(-15, 15)] * 3 + [(0, 16)]
    fit = minimize(objective, np.r_[initial, 0.25], jac=True, method="L-BFGS-B",
                   bounds=bounds, options={"ftol": 1e-10, "gtol": 1e-6, "maxls": 40})
    if not fit.success:
        fit = minimize(objective, fit.x, jac=True, method="SLSQP", bounds=bounds,
                       options={"ftol": 1e-9, "maxiter": 150})
    if not fit.success:
        raise RuntimeError(f"Prevalence model failed to converge: {fit.message}")
    params = np.r_[fit.x[:3], np.sqrt(fit.x[3])]
    prevalence = weights @ expit(params[:3] + params[3] * nodes[:, None])
    return prevalence, params


def area_differences(values):
    return values[..., PAIRS[:, 0]] - values[..., PAIRS[:, 1]]


def inference(units, n_boot, rng):
    mice, areas = np.unique(units.mouse_id, return_inverse=True)[1], units.brain_area.to_numpy() - 1
    n_mice = len(np.unique(mice))
    grouping = 3 * mice + areas
    valid = units.fs_waveform.notna().to_numpy()
    y = units.loc[valid, "fs_waveform"].to_numpy()
    totals = np.bincount(grouping[valid], minlength=n_mice * 3).reshape(n_mice, 3)
    counts = np.bincount(grouping[valid], weights=y, minlength=n_mice * 3).reshape(n_mice, 3)
    if np.any(totals == 0):
        raise ValueError("Expected classified units in every mouse-area group")
    prevalence, params = fit_prevalence(counts, totals)
    check, _ = fit_prevalence(counts, totals, quadrature=128)
    if not np.allclose(prevalence, check, atol=1e-5, rtol=0):
        raise RuntimeError("Random-intercept quadrature did not stabilize at 64 nodes")
    data_by_mouse = [np.flatnonzero(mice[valid] == mouse) for mouse in range(n_mice)]
    grouped_areas = areas[valid]
    waveform_x = np.log(units.loc[valid, WIDTH_FEATURES].to_numpy())
    widths = units.loc[valid, "trough_to_peak_ms"].to_numpy()
    samples = rng.integers(n_mice, size=(n_boot, n_mice))
    fixed, refitted, log_odds = np.empty((3, n_boot, 3))
    for i, selected in enumerate(samples):
        fixed[i], beta = fit_prevalence(counts[selected], totals[selected])
        log_odds[i] = beta[:3]
        indices = np.concatenate([data_by_mouse[mouse] for mouse in selected])
        p, _, _, _ = fit_classifier(waveform_x[indices], widths[indices], n_init=3)
        # A duplicate mouse in one resample retains the same realization of its unit labels.
        unique_idx, first, inverse = np.unique(indices, return_index=True, return_inverse=True)
        labels = (rng.random(len(unique_idx)) < p[first])[inverse]
        boot_mouse = np.repeat(np.arange(n_mice), totals[selected].sum(axis=1))
        assigned = np.bincount(3 * boot_mouse + grouped_areas[indices], weights=labels,
                               minlength=n_mice * 3).reshape(n_mice, 3)
        refitted[i], _ = fit_prevalence(assigned, totals[selected])
        if (i + 1) % 500 == 0:
            print(f"Completed {i + 1} of {n_boot} mouse bootstrap fits", flush=True)
    paired = pd.DataFrame({"mouse_id": np.repeat(np.unique(units.mouse_id), 3), "brain_area": np.tile(AREAS, n_mice),
                           "n_fs": counts.ravel().astype(int), "n_valid": totals.ravel(),
                           "prevalence": (counts / totals).ravel()})
    means = pd.DataFrame({"brain_area": AREAS, "n_mice": n_mice, "n_fs": counts.sum(axis=0).astype(int),
                          "n_valid": totals.sum(axis=0), "mean_mouse_prevalence": (counts / totals).mean(axis=0),
                          "model_prevalence": prevalence,
                          "lower_95": np.quantile(fixed, .025, axis=0), "upper_95": np.quantile(fixed, .975, axis=0),
                          "lower_95_refit_draw": np.quantile(refitted, .025, axis=0),
                          "upper_95_refit_draw": np.quantile(refitted, .975, axis=0)})
    difference = area_differences(prevalence)
    log_or = area_differences(params[:3])
    boot_diff, boot_log_or = area_differences(fixed), area_differences(log_odds)
    diff_se = boot_diff.std(axis=0, ddof=1)
    or_se = boot_log_or.std(axis=0, ddof=1)
    max_t = np.abs((boot_diff - difference) / diff_se).max(axis=1)
    critical = np.quantile(max_t, .95)
    or_critical = np.quantile(np.abs((boot_log_or - log_or) / or_se).max(axis=1), .95)
    p_adjusted = (1 + (max_t[:, None] >= np.abs(difference) / diff_se).sum(axis=0)) / (n_boot + 1)
    refit_diff = area_differences(refitted)
    contrasts = pd.DataFrame({"area_a": PAIRS[:, 0] + 1, "area_b": PAIRS[:, 1] + 1,
                              "difference_pp": 100 * difference, "simultaneous_lower_95_pp": 100 * (difference - critical * diff_se),
                              "simultaneous_upper_95_pp": 100 * (difference + critical * diff_se), "p_adjusted": p_adjusted,
                              "conditional_odds_ratio": np.exp(log_or),
                              "or_simultaneous_lower_95": np.exp(log_or - or_critical * or_se),
                              "or_simultaneous_upper_95": np.exp(log_or + or_critical * or_se),
                              "refit_draw_lower_95_pp": 100 * np.quantile(refit_diff, .025, axis=0),
                              "refit_draw_upper_95_pp": 100 * np.quantile(refit_diff, .975, axis=0)})
    boot_ranks = np.bincount(refitted.argmax(axis=1), minlength=3) / n_boot
    return paired, means, contrasts, params[3], boot_ranks


def sensitivity(units, waveforms):
    eligible = units.fs_waveform.notna()
    x = StandardScaler().fit_transform(np.log(units.loc[eligible, WIDTH_FEATURES]))
    widths_for_mixture = units.loc[eligible, "trough_to_peak_ms"].to_numpy()
    mixture4 = GaussianMixture(4, covariance_type="tied", n_init=12, random_state=SEED).fit(x)
    p = mixture4.predict_proba(x)
    centers4 = (p.T @ widths_for_mixture) / p.sum(axis=0)
    units = units.copy()
    units.loc[eligible, "fs_mixture4_narrowest"] = (p.argmax(axis=1) == centers4.argmin()).astype(float)
    rows = []
    for method, label in METHODS.items():
        valid = units[units[method].notna()]
        data = valid.groupby(["mouse_id", "brain_area"])[method].agg(["sum", "count"]).unstack()
        values, _ = fit_prevalence(data["sum"].to_numpy(), data["count"].to_numpy())
        for i, area in enumerate(AREAS):
            rows.append({"method": method, "description": label, "brain_area": area,
                         "model_prevalence": values[i], "n_units": data["count"][area].sum(),
                         "rank": int(1 + np.sum(values > values[i]))})
    valid = units.fs_waveform.notna().to_numpy()
    waveform_x = np.log(units.loc[valid, WIDTH_FEATURES].to_numpy())
    waveform_samples = waveforms[valid]
    widths = units.loc[valid, "trough_to_peak_ms"].to_numpy()
    ids = units.loc[valid, "mouse_id"].to_numpy()
    # Fit the scaler and mixture on training mice, then classify held-out units.
    lomo = []
    for method in ("waveform", "pca"):
        for mouse in np.unique(ids):
            train = ids != mouse
            x = waveform_x
            if method == "pca":
                pca = PCA(n_components=4, svd_solver="full").fit(waveform_samples[train])
                x = pca.transform(waveform_samples)
            _, model, centers, _ = fit_classifier(x[train], widths[train])
            scaler = StandardScaler().fit(x[train])
            posterior = model.predict_proba(scaler.transform(x[~train]))[:, centers.argmin()]
            held_out = units.loc[valid].loc[~train, ["mouse_id", "brain_area", "cluster_id"]].copy()
            held_out["method"] = method
            held_out["fs_probability"] = posterior
            held_out["fs"] = posterior >= .5
            held_out["agrees_with_full_fit"] = held_out.fs.to_numpy() == units.loc[valid, f"fs_{method}"].to_numpy()[~train]
            lomo.append(held_out)
    lomo = pd.concat(lomo, ignore_index=True)
    for method, grouped in lomo.groupby("method"):
        data = grouped.groupby(["mouse_id", "brain_area"]).fs.agg(["sum", "count"]).unstack()
        values, _ = fit_prevalence(data["sum"].to_numpy(), data["count"].to_numpy())
        for i, area in enumerate(AREAS):
            rows.append({"method": f"lomo_{method}", "description": f"Held-out mouse ({method})", "brain_area": area,
                         "model_prevalence": values[i], "n_units": data["count"][area].sum(),
                         "rank": int(1 + np.sum(values > values[i]))})
    return pd.DataFrame(rows), lomo, np.sort(centers4)


def timing_checks(units, rng):
    eligible = units[units.fs_waveform.notna()]
    results = []
    for name, label in SPIKE_METRICS.items():
        measured = eligible.dropna(subset=[name])
        fs, other = (measured.loc[measured.fs_waveform.eq(group), name].to_numpy() for group in (1, 0))
        strata = measured.groupby(["mouse_id", "brain_area", "fs_waveform"])[name].median().unstack()
        differences = (strata[1] - strata[0]).dropna()
        by_mouse = differences.groupby(level="mouse_id").mean().to_numpy()
        boot = rng.choice(by_mouse, size=(2000, len(by_mouse)), replace=True).mean(axis=1)
        results.append({"metric": name, "description": label, "n_fs": len(fs), "n_other": len(other),
                        "median_fs": np.median(fs), "median_other": np.median(other),
                        "paired_mouse_mean_difference": by_mouse.mean(), "n_paired_mice": len(by_mouse),
                        "n_paired_mouse_areas": len(differences), "lower_95": np.quantile(boot, .025),
                        "upper_95": np.quantile(boot, .975)})
    return pd.DataFrame(results)


def save_figures(output, units, profiles, paired, means, variants):
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 4.4), layout="constrained", sharey=True)
    for _, mouse in paired.groupby("mouse_id"):
        ax[0].plot(mouse.brain_area, 100 * mouse.prevalence, ".-", color="#64748B", alpha=.42)
    ax[0].set(title="Observed paired mouse proportions", ylabel="Putative fast-spiking units (%)")
    for j, row in means.iterrows():
        area, center = row.brain_area, 100 * row.model_prevalence
        ax[1].vlines(area, 100 * row.lower_95_refit_draw, 100 * row.upper_95_refit_draw, color="#A0AEC0", linewidth=6)
        ax[1].vlines(area, 100 * row.lower_95, 100 * row.upper_95, color=COLORS[j], linewidth=2.5)
        ax[1].scatter(area, center, color=COLORS[j], zorder=3)
    ax[1].plot([], [], color="#334155", linewidth=2.5, label="Mouse bootstrap")
    ax[1].plot([], [], color="#A0AEC0", linewidth=6, label="With classifier refit and label draws")
    ax[1].legend(frameon=False, loc="best", fontsize=8)
    ax[1].set_title("Mixed-model estimates and 95% intervals")
    for axis in ax:
        axis.set(xticks=AREAS, xlabel="Brain area")
        axis.grid(axis="y", alpha=.2)
    fig.savefig(output / "q7_prevalence.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(9.4, 7.1), layout="constrained")
    retained = units.fs_waveform.notna()
    shapes = units.loc[retained]
    scatter = ax[0, 0].scatter(shapes.trough_to_peak_ms, shapes.half_width_ms, c=shapes.p_fs_waveform,
                               cmap="viridis", s=9, alpha=.75, vmin=0, vmax=1, linewidths=0)
    ax[0, 0].set(xlabel="Trough-to-peak (ms)", ylabel="Trough half-width (ms)", title="Waveform-based classification")
    fig.colorbar(scatter, ax=ax[0, 0], label="Narrow-component probability")
    plot_specs = [(ax[0, 1], "waveform", "waveform_grid_ms", "Time from trough (ms)", "Normalized voltage"),
                  (ax[1, 0], "acg", "acg_centers_ms", "Positive lag (ms)", "Conditional rate / session rate")]
    for axis, key, grid, xlabel, ylabel in plot_specs:
        for group, name, color in [(0, "Other units", "#788385"), (1, "Putative fast-spiking", COLORS[0])]:
            block = profiles[key][units.fs_waveform.eq(group).to_numpy()]
            block = block[np.isfinite(block).all(axis=1)]
            lower, median, upper = np.quantile(block, [.25, .5, .75], axis=0)
            axis.fill_between(profiles[grid], lower, upper, color=color, alpha=.20)
            axis.plot(profiles[grid], median, color=color, label=name)
        axis.set(xlabel=xlabel, ylabel=ylabel)
    ax[0, 1].set_title("Waveforms, median and interquartile range")
    ax[0, 1].legend(frameon=False, fontsize=8)
    ax[1, 0].set_title("Autocorrelograms, median and interquartile range")
    rate = [units.loc[units.fs_waveform.eq(group), "firing_rate_hz"] for group in [0, 1]]
    ax[1, 1].boxplot(rate, showfliers=False)
    ax[1, 1].set(xticks=[1, 2], xticklabels=["Other units", "Putative fast-spiking"], yscale="log",
                 ylabel="Session firing rate (Hz)", title="Spike-train check")
    fig.savefig(output / "q7_classification.png", dpi=180)
    plt.close(fig)

    table = variants.pivot(index="description", columns="brain_area", values="model_prevalence")
    order = variants.description.drop_duplicates()
    fig, axis = plt.subplots(figsize=(8.3, 5.0), layout="constrained")
    for area, color in zip(AREAS, COLORS):
        y = np.arange(len(order)) + (area - 2) * .20
        axis.scatter(100 * table.loc[order, area], y, label=f"Area {area}", color=color)
    axis.set(yticks=np.arange(len(order)), yticklabels=order, xlabel="Model-estimated prevalence (%)")
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=.24)
    axis.legend(frameon=False)
    fig.savefig(output / "q7_sensitivity.png", dpi=180)
    plt.close(fig)


def write_answer(output, units, means, contrasts, variants, lomo, timings, sigma, boot_ranks, n_boot, centers4):
    highest = int(means.loc[means.model_prevalence.idxmax(), "brain_area"])
    wins = ((contrasts.area_a.eq(highest) & contrasts.simultaneous_lower_95_pp.gt(0)) |
            (contrasts.area_b.eq(highest) & contrasts.simultaneous_upper_95_pp.lt(0)))
    tops = variants[variants["rank"].eq(1)].set_index("method").brain_area
    core = CORE_METHODS + ["lomo_waveform", "lomo_pca"]
    consistent = all((tops.loc[[method]] == highest).all() for method in core)
    robust = wins.sum() == 2 and consistent
    answer = f"Brain area {highest}" if robust else "Not enough data / no differences"
    lines = [f"Q7 answer: {answer}", "",
             "The estimand is the fraction of eligible recorded units assigned to the narrow waveform component.",
             "Sampling volume and independently verified cell identities would be needed for anatomical density.",
             f"The area-blind model retained {units.fs_waveform.notna().sum()} of {len(units)} units from 18 mice.",
             "Classification used a tied-covariance two-component mixture of log trough-to-peak duration, "
             "half-width, and time to 50% recovery. The maximum posterior determined each primary label.",
             "The same waveform and refractory-violation quality rules were applied in all areas.", "",
             f"Binomial mixed model with mouse random intercept (SD={sigma:.3f} log odds). "
             f"The {n_boot:,} resamples each retain all three areas of every resampled mouse.",
             "Marginal estimates integrate over mouse effects. Pairwise tests use simultaneous max-t bootstrap "
             "intervals and adjusted p-values. An additional bootstrap refits the classifier and samples labels."]
    for row in means.itertuples():
        lines.append(f"Area {row.brain_area}: {100 * row.model_prevalence:.2f}% (95% CI "
                     f"{100 * row.lower_95:.2f}–{100 * row.upper_95:.2f}%), {row.n_fs}/{row.n_valid} units.")
    for row in contrasts.itertuples():
        lines.append(f"Area {row.area_a} minus area {row.area_b}: {row.difference_pp:+.2f} percentage points "
                     f"(simultaneous 95% CI {row.simultaneous_lower_95_pp:+.2f} to {row.simultaneous_upper_95_pp:+.2f}, "
                     f"adjusted p={row.p_adjusted:.4f}), conditional OR={row.conditional_odds_ratio:.2f}.")
    lines.append(f"Area {highest} had the largest primary point estimate and led {100 * boot_ranks[highest - 1]:.1f}% "
                 "of bootstrap fits with classifier refitting and label sampling.")
    lines.append(f"Primary adjusted intervals separated that area from {int(wins.sum())} of 2 alternatives. "
                 f"The core sensitivity analyses {'agreed' if consistent else 'did not agree'} on the same highest area.")
    lines.append("Leading areas by method: " + "; ".join(f"{name}: {area}" for name, area in tops.items()) + ".")
    diagnostics = json.loads((output / "q7_config.json").read_text(encoding="utf-8"))
    widths = sorted(diagnostics["primary_component_mean_widths_ms"])
    bic = diagnostics["bic_by_components"]
    lines.append(f"The primary mean widths were {widths[0]:.3f} and {widths[1]:.3f} ms. "
                 f"The four-component BIC was {bic['4']:.0f} vs {bic['2']:.0f} for two components. "
                 "Its mean widths were " + ", ".join(f"{value:.3f}" for value in centers4) + " ms. "
                 "The primary two-way split is an operational definition of a narrow waveform group.")
    for method, group in lomo.groupby("method"):
        lines.append(f"Held-out-mouse {method} label agreement with the full fit: {100 * group.agrees_with_full_fit.mean():.1f}%.")
    lines += ["", "Spike-train checks compare medians within mouse and area before averaging differences across mice."]
    for row in timings.itertuples():
        lines.append(f"{row.description}: class difference {row.paired_mouse_mean_difference:+.3g} "
                     f"(95% CI {row.lower_95:+.3g} to {row.upper_95:+.3g}, {row.n_paired_mice} mice).")
    lines += ["", "References: Barthó et al. (2004), doi:10.1152/jn.01170.2003. "
              "Petersen et al. (2021), doi:10.1016/j.neuron.2021.09.002. "
              "Wei et al. (2023), doi:10.1038/s41467-023-37844-8."]
    (output / "answer.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap requires at least 100 resamples")
    output = args.output
    units = pd.read_csv(output / "q7_units.csv")
    with np.load(output / "q7_unit_profiles.npz") as data:
        profiles = dict(data)
    for key in ("mouse_id", "cluster_id"):
        if not np.array_equal(units[key].to_numpy(), profiles[key]):
            raise ValueError("Unit CSV and profile archive are misaligned")
    rng = np.random.default_rng(SEED)
    with threadpool_limits(limits=1):
        paired, means, contrasts, sigma, boot_ranks = inference(units, args.bootstrap, rng)
        variants, lomo, centers4 = sensitivity(units, profiles["waveform"])
    timings = timing_checks(units, rng)
    qc = units.groupby(["brain_area", "qc_status"]).size().rename("n_units").reset_index()
    tables = {"mouse_prevalence": paired, "prevalence": means, "contrasts": contrasts, "sensitivity": variants,
              "held_out_mice": lomo, "spike_train_checks": timings, "qc": qc}
    for name, table in tables.items():
        table.to_csv(output / f"q7_{name}.csv", index=False)
    save_figures(output, units, profiles, paired, means, variants)
    write_answer(output, units, means, contrasts, variants, lomo, timings, sigma, boot_ranks, args.bootstrap, centers4)


if __name__ == "__main__":
    main()

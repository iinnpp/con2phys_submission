"""Apply the published CellExplorer classification rules to CON2PHYS Q7.

Uses the Q7 waveform QC and trough-to-peak measurements, then rebuilds the
CellExplorer-style ACGs from the original spike times. See Petersen et al.
(2021), doi:10.1016/j.neuron.2021.09.002, and cellexplorer.org/pipeline/acg-fit/.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

from q7_fast_spiking import ACG_BIN_MS, DEFAULT_INPUT, DEFAULT_OUTPUT, SEED
from q7_fast_spiking_stats import AREAS, COLORS, PAIRS, area_differences, fit_prevalence, plt, timing_checks


CUTOFFS = {"paper_450us": 0.450, "docs_425us": 0.425}
RISE_THRESHOLD_MS = 6.0
LAGS_MS = np.arange(1, 101) * ACG_BIN_MS
PARAM_NAMES = ["tau_decay_ms", "tau_rise_ms", "decay_amplitude_hz", "rise_scale",
               "asymptote_hz", "refractory_shift_ms", "tau_burst_ms", "burst_amplitude_hz"]
# Numerical settings from CellExplorer's fit_ACG.m. Amplitudes assume a rate ACG in Hz.
START = np.array([20, 1, 30, 2, 0.5, 5, 1.5, 2])
LOWER = np.array([1, 0.1, 0, 0, -30, 0, 0.1, 0])
UPPER = np.array([500, 50, 500, 15, 50, 20, 5, 100])
CLASSES = ["narrow_interneuron", "wide_interneuron", "pyramidal", "unresolved_acg", "excluded_qc"]


def autocorrelogram(spikes):
    """Conditional rate at 0.5, 1, ..., 50 ms. Distinct pairs only."""
    n_bins = len(LAGS_MS)
    bin_s = ACG_BIN_MS / 1_000
    max_lag_s = (n_bins + 0.5) * bin_s
    counts = np.zeros(n_bins + 1, dtype=np.int64)
    for offset in range(1, len(spikes)):
        lags = spikes[offset:] - spikes[:-offset]
        lags = lags[lags < max_lag_s]
        if not lags.size:
            break
        indices = np.floor(lags / bin_s + 0.5).astype(int)
        counts += np.bincount(indices, minlength=n_bins + 1)[:n_bins + 1]
    # fit_ACG.m zeros the central and neighboring bins and fits positive lags.
    counts[1] = 0
    return counts[1:] / (len(spikes) * bin_s)


def model_terms(params):
    decay, rise, amplitude, scale, asymptote, shift, burst, burst_amplitude = params
    elapsed = LAGS_MS - shift
    slow, fast, transient = [np.exp(-elapsed / tau) for tau in (decay, rise, burst)]
    value = amplitude * (slow - scale * fast) + burst_amplitude * transient + asymptote
    return value, elapsed, slow, fast, transient


def fitted_curve(params):
    return np.maximum(model_terms(params)[0], 0)


def fit_jacobian(params):
    decay, rise, amplitude, scale, _, _, burst, burst_amplitude = params
    value, elapsed, slow, fast, transient = model_terms(params)
    gradient = np.column_stack([
        amplitude * slow * elapsed / decay**2,
        -amplitude * scale * fast * elapsed / rise**2,
        slow - scale * fast,
        -amplitude * fast,
        np.ones_like(LAGS_MS),
        amplitude * (slow / decay - scale * fast / rise) + burst_amplitude * transient / burst,
        burst_amplitude * transient * elapsed / burst**2,
        transient,
    ])
    gradient[value <= 0] = 0
    return gradient


def fit_acg(acg):
    total_ss = np.sum((acg - acg.mean()) ** 2)
    if total_ss == 0:
        return np.full(len(START), np.nan), np.nan, "constant_acg", 0
    residual = lambda params: fitted_curve(params) - acg
    fit = least_squares(residual, START, jac=fit_jacobian, bounds=(LOWER, UPPER), max_nfev=400)
    if not fit.success or 2 * fit.cost >= total_ss:
        # Retry stalled fits with a shorter initial shift and amplitudes near the observed rate.
        rate = np.clip(np.median(acg[-20:]), 0.1, 45)
        start = [30, 4, rate, 1.5, rate / 2, 1.5, 1.5, rate / 2]
        retry = least_squares(residual, start, jac=fit_jacobian, bounds=(LOWER, UPPER), max_nfev=1500)
        if retry.success and (not fit.success or retry.cost < fit.cost):
            fit = retry
    r2 = 1 - 2 * fit.cost / total_ss
    status = "converged" if fit.success and r2 > 0 else "unresolved_fit"
    return fit.x, r2, status, fit.nfev


def rebuild_acgs(units, source):
    acgs = np.full((len(units), len(LAGS_MS)), np.nan)
    for mouse, rows in units.groupby("mouse_id", sort=True):
        folder = source / str(mouse)
        times = np.load(folder / "spikes.npy", mmap_mode="r")
        clusters = np.load(folder / "clusters.npy", mmap_mode="r")
        if times.shape != clusters.shape or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
            raise ValueError(f"Invalid spike arrays for mouse {mouse}")
        order = np.argsort(clusters, kind="stable")
        sorted_times = times[order]
        ids, counts = np.unique(clusters, return_counts=True)
        groups = dict(zip(ids, np.split(sorted_times, counts.cumsum()[:-1])))
        if set(ids) != set(rows.cluster_id):
            raise ValueError(f"Cluster IDs disagree with the Q7 unit table for mouse {mouse}")
        for row in rows.itertuples():
            train = groups[row.cluster_id]
            if len(train) != row.n_spikes:
                raise ValueError(f"Spike count mismatch for mouse {mouse}, cluster {row.cluster_id}")
            if row.qc_status == "pass":
                acgs[row.Index] = autocorrelogram(train)
        print(f"Mouse {mouse:2}: built {rows.qc_status.eq('pass').sum()} eligible autocorrelograms", flush=True)
    return acgs


def classify(units, acgs):
    eligible = units.qc_status.eq("pass")
    units = units.copy()
    fits = np.full(acgs.shape, np.nan)
    units["acg_status"] = "excluded_qc"
    total = int(eligible.sum())
    for completed, index in enumerate(units.index[eligible], start=1):
        params, r2, status, nfev = fit_acg(acgs[index])
        units.loc[index, PARAM_NAMES] = params
        units.loc[index, "acg_r2"] = r2
        units.loc[index, "acg_fit_evaluations"] = nfev
        units.loc[index, "acg_status"] = status
        if np.isfinite(params).all():
            fits[index] = fitted_curve(params)
        if completed % 250 == 0 or completed == total:
            print(f"Processed {completed:,} of {total:,} ACG fits", flush=True)
    width = units.trough_to_peak_ms
    rise = units.tau_rise_ms
    converged = units.acg_status.eq("converged")
    for name, threshold in CUTOFFS.items():
        units[name] = np.select(
            [~eligible, width.le(threshold), ~converged, rise.gt(RISE_THRESHOLD_MS)],
            ["excluded_qc", "narrow_interneuron", "unresolved_acg", "wide_interneuron"],
            default="pyramidal",
        )
    return units, fits


def compare_areas(mouse_counts, n_boot):
    """The Q7 mixed model and paired mouse bootstrap, applied to narrow labels."""
    counts = mouse_counts.narrow_interneuron.unstack().to_numpy()
    totals = mouse_counts.n_eligible.unstack().to_numpy()
    estimates, _ = fit_prevalence(counts, totals)
    fine, _ = fit_prevalence(counts, totals, quadrature=128)
    np.testing.assert_allclose(estimates, fine, rtol=0, atol=1e-5)
    idx = np.random.default_rng(SEED).integers(len(counts), size=(n_boot, len(counts)))
    bootstrap = np.array([fit_prevalence(counts[i], totals[i])[0] for i in idx])
    lo, hi = np.quantile(bootstrap, [.025, .975], axis=0)
    prevalence = pd.DataFrame({"brain_area": AREAS, "n_narrow": counts.sum(axis=0),
                               "n_eligible": totals.sum(axis=0), "estimate": estimates, "lower_95": lo, "upper_95": hi})
    diff = area_differences(estimates)
    boot_diff = area_differences(bootstrap)
    se = boot_diff.std(axis=0, ddof=1)
    max_t = np.abs((boot_diff - diff) / se).max(axis=1)
    critical = np.quantile(max_t, .95)
    adjusted_p = (1 + (max_t[:, None] >= np.abs(diff) / se).sum(axis=0)) / (n_boot + 1)
    contrasts = pd.DataFrame({"area_a": PAIRS[:, 0] + 1, "area_b": PAIRS[:, 1] + 1, "difference_pp": 100 * diff,
                              "simultaneous_lower_95_pp": 100 * (diff - critical * se),
                              "simultaneous_upper_95_pp": 100 * (diff + critical * se), "p_adjusted": adjusted_p})
    return prevalence, contrasts


def save_plot(output, units, estimates):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    selected = units[units.acg_status.eq("converged")]
    labels = {"narrow_interneuron": "Narrow interneuron", "wide_interneuron": "Wide interneuron", "pyramidal": "Pyramidal"}
    for color, (name, label) in zip(COLORS, labels.items()):
        group = selected[selected.paper_450us.eq(name)]
        axes[0].scatter(group.trough_to_peak_ms, group.tau_rise_ms, s=9, color=color, alpha=.55, label=label)
    axes[0].axvline(CUTOFFS["paper_450us"], color="0.35", linewidth=1, linestyle="--")
    axes[0].hlines(RISE_THRESHOLD_MS, CUTOFFS["paper_450us"], selected.trough_to_peak_ms.max(),
                   color="0.35", linewidth=1, linestyle="--")
    axes[0].set(xlabel="Trough-to-peak duration (ms)", ylabel="ACG rise time (ms)", yscale="log",
                title="Published CellExplorer classification")
    axes[0].legend(loc="upper right", frameon=False, fontsize=8)
    for index, (name, table) in enumerate(estimates.items()):
        x = table.brain_area.to_numpy() + (index - 0.5) * .18
        axes[1].errorbar(x, 100 * table.estimate, yerr=100 * np.stack([table.estimate - table.lower_95,
                         table.upper_95 - table.estimate]), marker="o", linestyle="None", capsize=4,
                         label=f"{CUTOFFS[name]:.3f} ms", color=COLORS[index])
    axes[1].set(xticks=AREAS, xlabel="Brain area", ylabel="Narrow-waveform units (%)",
                title="Marginal prevalence and 95% intervals")
    axes[1].grid(axis="y", alpha=.25)
    axes[1].legend(frameon=False, title="Width cutoff")
    fig.savefig(output / "q7_cellexplorer.png", dpi=180)
    plt.close(fig)


def save_results(output, units, acgs, fits, n_boot):
    estimates, comparisons, summaries, mouse_summaries = {}, {}, [], []
    for name in CUTOFFS:
        by_mouse = units.groupby(["mouse_id", "brain_area", name]).size().unstack(fill_value=0)
        by_mouse = by_mouse.reindex(columns=CLASSES, fill_value=0)
        by_mouse["n_eligible"] = by_mouse[CLASSES[:-1]].sum(axis=1)
        by_mouse["narrow_fraction"] = by_mouse.narrow_interneuron / by_mouse.n_eligible
        estimates[name], comparisons[name] = compare_areas(by_mouse, n_boot)
        summary = by_mouse.drop(columns="narrow_fraction").groupby(level="brain_area").sum().reset_index()
        summaries.append(summary.assign(rule=name))
        mouse_summaries.append(by_mouse.reset_index().assign(rule=name))
    match = units.qc_status.eq("pass")
    checks = units.assign(fs_waveform=np.where(match, units.paper_450us.eq("narrow_interneuron"), np.nan))
    timing = timing_checks(checks, np.random.default_rng(SEED))

    output.mkdir(parents=True, exist_ok=True)
    units.to_csv(output / "q7_cellexplorer_units.csv", index=False)
    np.savez_compressed(output / "q7_cellexplorer_acgs.npz", lag_ms=LAGS_MS, observed_rate_hz=acgs,
                        fitted_rate_hz=fits, mouse_id=units.mouse_id, cluster_id=units.cluster_id)
    pd.concat(summaries).to_csv(output / "q7_cellexplorer_area_counts.csv", index=False)
    pd.concat(mouse_summaries).to_csv(output / "q7_cellexplorer_mouse_counts.csv", index=False)
    pd.concat(table.assign(rule=name) for name, table in estimates.items()).to_csv(output / "q7_cellexplorer_prevalence.csv", index=False)
    pd.concat(table.assign(rule=name) for name, table in comparisons.items()).to_csv(output / "q7_cellexplorer_contrasts.csv", index=False)
    timing.to_csv(output / "q7_cellexplorer_spike_train_checks.csv", index=False)
    save_plot(output, units, estimates)
    return estimates, comparisons


def write_answer(output, units, estimates, contrasts, n_boot):
    primary = estimates["paper_450us"]
    paired = contrasts["paper_450us"]
    top = int(primary.loc[primary.estimate.idxmax(), "brain_area"])
    wins = ((paired.area_a.eq(top) & paired.simultaneous_lower_95_pp.gt(0)) |
            (paired.area_b.eq(top) & paired.simultaneous_upper_95_pp.lt(0)))
    answer = f"Brain area {top}" if wins.sum() == 2 else "Not enough data / no differences"
    lines = [f"Q7 answer using the published CellExplorer classifier: {answer}", "",
             "Paper rule: narrow waveform at trough-to-peak <= 0.450 ms. Above that cutoff, "
             "an ACG rise time > 6 ms gives a wide-waveform interneuron label and a rise time <= 6 ms "
             "gives a putative pyramidal label. Failed fits leave wide-waveform units unresolved.",
             "For Q7, the narrow-waveform class is the proxy for putative fast-spiking interneurons. "
             "The denominator is all eligible recorded units in each mouse and area.",
             f"We reused CON2PHYS Q7 waveform measurements and quality checks ({units.qc_status.eq('pass').sum()} eligible "
             f"of {len(units)} units), then rebuilt 0.5 ms rate ACGs through 50 ms. "
             "We used the CellExplorer model, parameter bounds and initial values with SciPy least squares.",
             f"The binomial mixed model includes a mouse random intercept. Intervals use {n_boot:,} mouse "
             "bootstrap draws, with simultaneous max-t adjustment for the three pairwise comparisons."]
    for name, table in estimates.items():
        lines += ["", f"Trough-to-peak cutoff: {CUTOFFS[name]:.3f} ms"]
        for row in table.itertuples():
            lines.append(f"Area {row.brain_area}: {row.n_narrow}/{row.n_eligible} narrow, "
                         f"model estimate {100 * row.estimate:.1f}% (95% CI {100 * row.lower_95:.1f}–{100 * row.upper_95:.1f}%).")
        for row in contrasts[name].itertuples():
            lines.append(f"Area {row.area_a} minus area {row.area_b}: {row.difference_pp:+.2f} percentage points "
                         f"(simultaneous 95% CI {row.simultaneous_lower_95_pp:+.2f} to {row.simultaneous_upper_95_pp:+.2f}, "
                         f"adjusted p={row.p_adjusted:.4f}).")
    lines += ["", "ACG fit status counts among eligible units:",
              units.loc[units.qc_status.eq("pass"), "acg_status"].value_counts().to_string(), "",
              "Petersen et al. (2021), Neuron, doi:10.1016/j.neuron.2021.09.002",
              "https://cellexplorer.org/pipeline/acg-fit/",
              "https://cellexplorer.org/pipeline/cell-type-classification/"]
    report = "\n".join(lines) + "\n"
    (output / "answer.txt").write_text(report, encoding="utf-8")
    print(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--units", type=Path, default=DEFAULT_OUTPUT / "q7_units.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "cellexplorer")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap requires at least 100 draws")
    units = pd.read_csv(args.units)
    if (units.duplicated(["mouse_id", "cluster_id"]).any() or set(units.brain_area) != set(AREAS)
            or not np.isfinite(units.loc[units.qc_status.eq("pass"), "trough_to_peak_ms"]).all()):
        raise ValueError("Invalid Q7 unit table")
    columns = ["mouse_id", "brain_area", "cluster_id", "qc_status", "trough_to_peak_ms", "half_width_ms",
               "n_spikes", "firing_rate_hz", "isi_median_ms", "isi_cv2", "acg_burst_index", "acg_2_to_10ms_relative_rate"]
    with threadpool_limits(limits=1):
        acgs = rebuild_acgs(units, args.input)
        print("Fitting triple exponentials to eligible ACGs", flush=True)
        classified, fitted = classify(units[columns], acgs)
        estimates, contrasts = save_results(args.output, classified, acgs, fitted, args.bootstrap)
    write_answer(args.output, classified, estimates, contrasts, args.bootstrap)


if __name__ == "__main__":
    main()

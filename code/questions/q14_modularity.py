#!/usr/bin/env python

"""Q14 whole-recording spike-network modularity calculation.

The script reuses the Q4 pairwise STTC table so that Q14 has exactly the same
whole-recording observation windows and STTC definition. It matches neuron
counts across areas within mouse, estimates weighted Leiden modularity, and
subtracts the optimized modularity of degree-preserving null graphs.

All inferential inputs are mouse-area summaries. Neuron subsamples and null
graphs quantify calculation uncertainty and are never treated as independent
biological observations.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd


REPOSITORY = Path(os.environ.get(
    "CON2PHYS_REPOSITORY_DIR", Path(__file__).resolve().parents[2]
))
INPUT_PATH = REPOSITORY / "results" / "q4" / "sttc_pairs_100ms.csv"
OUTPUT_DIR = Path(os.environ.get(
    "Q14_OUTPUT_DIR", REPOSITORY / "results" / "q14"
))

RANDOM_SEED = 14042
EXPECTED_MICE = 18
EXPECTED_AREAS = (1, 2, 3)
EXPECTED_PAIRS = 25_299
STTC_DT_SEC = 0.1
MIN_MATCHED_UNITS = 10

PRIMARY_DENSITY = 0.20
PRIMARY_GAMMA = 1.0
N_SUBSAMPLES = int(os.environ.get("Q14_N_SUBSAMPLES", 100))
N_NULLS = int(os.environ.get("Q14_N_NULLS", 20))
N_OBSERVED_RESTARTS = int(os.environ.get("Q14_N_OBSERVED_RESTARTS", 25))
N_NULL_RESTARTS = int(os.environ.get("Q14_N_NULL_RESTARTS", 10))

SENSITIVITY_SUBSAMPLES = int(os.environ.get("Q14_SENSITIVITY_SUBSAMPLES", 20))
SENSITIVITY_NULLS = int(os.environ.get("Q14_SENSITIVITY_NULLS", 10))
SENSITIVITY_OBSERVED_RESTARTS = int(os.environ.get(
    "Q14_SENSITIVITY_OBSERVED_RESTARTS", 15
))
SENSITIVITY_NULL_RESTARTS = int(os.environ.get(
    "Q14_SENSITIVITY_NULL_RESTARTS", 5
))
SENSITIVITY_SETTINGS = (
    (0.10, 1.00, "density"),
    (0.15, 1.00, "density"),
    (0.25, 1.00, "density"),
    (0.30, 1.00, "density"),
    (0.20, 0.50, "resolution"),
    (0.20, 0.75, "resolution"),
    (0.20, 1.25, "resolution"),
    (0.20, 1.50, "resolution"),
    (0.20, 2.00, "resolution"),
)

for name, value in {
    "Q14_N_SUBSAMPLES": N_SUBSAMPLES,
    "Q14_N_NULLS": N_NULLS,
    "Q14_N_OBSERVED_RESTARTS": N_OBSERVED_RESTARTS,
    "Q14_N_NULL_RESTARTS": N_NULL_RESTARTS,
    "Q14_SENSITIVITY_SUBSAMPLES": SENSITIVITY_SUBSAMPLES,
    "Q14_SENSITIVITY_NULLS": SENSITIVITY_NULLS,
    "Q14_SENSITIVITY_OBSERVED_RESTARTS": SENSITIVITY_OBSERVED_RESTARTS,
    "Q14_SENSITIVITY_NULL_RESTARTS": SENSITIVITY_NULL_RESTARTS,
}.items():
    if value <= 0:
        raise ValueError(f"{name} must be positive")
if SENSITIVITY_SUBSAMPLES > N_SUBSAMPLES:
    raise ValueError("Sensitivity subsamples cannot exceed primary subsamples")

UNIT_COUNTS_PATH = OUTPUT_DIR / "q14_unit_counts.csv"
SUBSAMPLES_PATH = OUTPUT_DIR / "q14_modularity_subsamples.csv"
MOUSE_AREA_PATH = OUTPUT_DIR / "q14_modularity_mouse_area.csv"
SENSITIVITY_SUBSAMPLES_PATH = OUTPUT_DIR / "q14_modularity_sensitivity_subsamples.csv"
SENSITIVITY_MOUSE_AREA_PATH = OUTPUT_DIR / "q14_modularity_sensitivity_mouse_area.csv"
CONFIG_PATH = OUTPUT_DIR / "q14_run_config.json"


@dataclass(frozen=True)
class GroupNetwork:
    mouse_id: int
    brain_area: int
    unit_ids: np.ndarray
    sttc: np.ndarray
    n_spikes: np.ndarray
    recording_duration_sec: float
    recording_start_sec: float
    recording_stop_sec: float


def stable_seed(*values: int) -> int:
    """Return a reproducible 31-bit seed without Python hash randomization."""
    sequence = np.random.SeedSequence([RANDOM_SEED, *(int(value) for value in values)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0] % (2**31 - 1))


def density_code(value: float) -> int:
    return int(round(value * 10_000))


def gamma_code(value: float) -> int:
    return int(round(value * 10_000))


def maximize_modularity(
    n_nodes: int,
    edges: list[tuple[int, int]],
    weights: np.ndarray,
    gamma: float,
    n_restarts: int,
    seed_prefix: tuple[int, ...],
) -> tuple[float, int]:
    """Maximize weighted RB-configuration modularity with repeated Leiden runs."""
    if len(edges) == 0 or len(edges) != len(weights):
        raise ValueError("A nonempty edge list and one weight per edge are required")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Leiden edge weights must be finite and strictly positive")

    graph = ig.Graph(n=n_nodes, edges=edges, directed=False)
    graph.es["weight"] = weights.tolist()
    best_q = -np.inf
    best_n_communities = 0
    for restart in range(n_restarts):
        partition = la.find_partition(
            graph,
            la.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=gamma,
            n_iterations=-1,
            seed=stable_seed(*seed_prefix, restart),
        )
        q_value = float(graph.modularity(
            partition.membership,
            weights="weight",
            resolution=gamma,
            directed=False,
        ))
        if q_value > best_q:
            best_q = q_value
            best_n_communities = len(set(partition.membership))
    return best_q, best_n_communities


def degree_preserving_rewire(
    edges: list[tuple[int, int]],
    n_nodes: int,
    seed: int,
    requested_swaps: int,
) -> tuple[list[tuple[int, int]], int]:
    """Rewire a simple graph in compiled igraph while preserving every degree."""
    graph = ig.Graph(n=n_nodes, edges=edges, directed=False)
    original_edges = {tuple(sorted(edge)) for edge in edges}
    # igraph uses the configured Python RNG for this operation. A dedicated
    # seeded instance makes every null graph reproducible without touching
    # NumPy's independent generator used for edge-weight permutation.
    ig.set_random_number_generator(random.Random(seed))
    graph.rewire(n=requested_swaps, mode="simple")
    rewired = [tuple(sorted(edge.tuple)) for edge in graph.es]
    changed_edges = len(original_edges - set(rewired))
    return rewired, changed_edges


def select_graph_edges(
    network: GroupNetwork,
    selected_indices: np.ndarray,
    density: float,
) -> tuple[list[tuple[int, int]], np.ndarray, int, float]:
    """Retain the strongest fixed proportion of positive STTC edges."""
    n_nodes = len(selected_indices)
    local_i, local_j = np.triu_indices(n_nodes, k=1)
    sttc_values = network.sttc[
        selected_indices[local_i], selected_indices[local_j]
    ]
    positive = sttc_values > 0
    possible_edges = n_nodes * (n_nodes - 1) // 2
    target_edges = int(math.ceil(density * possible_edges))
    if int(positive.sum()) < target_edges:
        raise ValueError(
            f"Mouse {network.mouse_id}, area {network.brain_area}: only "
            f"{int(positive.sum())} positive STTC edges for target {target_edges}"
        )
    positive_indices = np.flatnonzero(positive)
    order = np.argsort(-sttc_values[positive_indices], kind="mergesort")
    keep = positive_indices[order[:target_edges]]
    edges = [
        (int(local_i[index]), int(local_j[index]))
        for index in keep
    ]
    weights = sttc_values[keep].astype(float, copy=True)
    actual_density = len(edges) / possible_edges
    return edges, weights, possible_edges, actual_density


def calculate_subsample(
    network: GroupNetwork,
    matched_units: int,
    subsample_id: int,
    density: float,
    gamma: float,
    n_nulls: int,
    n_observed_restarts: int,
    n_null_restarts: int,
    analysis_code: int,
) -> dict[str, object]:
    sample_rng = np.random.default_rng(stable_seed(
        1, network.mouse_id, network.brain_area, subsample_id
    ))
    selected_indices = np.sort(sample_rng.choice(
        len(network.unit_ids), size=matched_units, replace=False
    ))
    selected_unit_ids = network.unit_ids[selected_indices]
    edges, weights, possible_edges, actual_density = select_graph_edges(
        network, selected_indices, density
    )

    seed_base = (
        analysis_code,
        network.mouse_id,
        network.brain_area,
        subsample_id,
        density_code(density),
        gamma_code(gamma),
    )
    q_observed, n_communities = maximize_modularity(
        matched_units,
        edges,
        weights,
        gamma,
        n_observed_restarts,
        (*seed_base, 1),
    )

    null_q = np.empty(n_nulls, dtype=float)
    changed_edges = np.empty(n_nulls, dtype=int)
    requested_swaps = max(10 * len(edges), 50)
    for null_id in range(n_nulls):
        null_rng = np.random.default_rng(stable_seed(*seed_base, 2, null_id))
        null_edges, n_changed = degree_preserving_rewire(
            edges,
            matched_units,
            stable_seed(*seed_base, 4, null_id),
            requested_swaps,
        )
        null_weights = null_rng.permutation(weights)
        null_q[null_id], _ = maximize_modularity(
            matched_units,
            null_edges,
            null_weights,
            gamma,
            n_null_restarts,
            (*seed_base, 3, null_id),
        )
        changed_edges[null_id] = n_changed

    null_mean = float(null_q.mean())
    null_sd = float(null_q.std(ddof=1)) if n_nulls > 1 else np.nan
    degrees = np.zeros(matched_units, dtype=int)
    for left, right in edges:
        degrees[left] += 1
        degrees[right] += 1

    return {
        "mouse_id": network.mouse_id,
        "brain_area": network.brain_area,
        "subsample_id": subsample_id,
        "density": density,
        "gamma": gamma,
        "n_available_units": len(network.unit_ids),
        "n_sampled_units": matched_units,
        "selected_cluster_ids": ";".join(str(value) for value in selected_unit_ids),
        "n_possible_edges": possible_edges,
        "n_retained_edges": len(edges),
        "actual_density": actual_density,
        "n_isolates": int((degrees == 0).sum()),
        "q_observed": q_observed,
        "n_communities": n_communities,
        "mean_q_null": null_mean,
        "sd_q_null": null_sd,
        "delta_q": q_observed - null_mean,
        "z_q": (q_observed - null_mean) / null_sd if null_sd > 0 else np.nan,
        "n_nulls": n_nulls,
        "mean_changed_edges_in_null": float(changed_edges.mean()),
        "min_changed_edges_in_null": int(changed_edges.min()),
    }


def summarize_subsamples(rows: pd.DataFrame) -> pd.DataFrame:
    grouping = ["mouse_id", "brain_area", "density", "gamma"]
    summary = (
        rows.groupby(grouping, as_index=False, sort=True)
        .agg(
            n_available_units=("n_available_units", "first"),
            n_sampled_units=("n_sampled_units", "first"),
            n_subsamples=("subsample_id", "nunique"),
            n_nulls_per_subsample=("n_nulls", "first"),
            mean_q_observed=("q_observed", "mean"),
            sd_q_observed=("q_observed", "std"),
            mean_q_null=("mean_q_null", "mean"),
            sd_mean_q_null=("mean_q_null", "std"),
            mean_delta_q=("delta_q", "mean"),
            sd_delta_q=("delta_q", "std"),
            mean_z_q=("z_q", "mean"),
            mean_n_communities=("n_communities", "mean"),
            mean_n_isolates=("n_isolates", "mean"),
            mean_actual_density=("actual_density", "mean"),
            min_changed_edges_in_null=("min_changed_edges_in_null", "min"),
        )
    )
    return summary


def validate_and_build_networks(sttc_pairs: pd.DataFrame) -> dict[tuple[int, int], GroupNetwork]:
    required = [
        "mouse_id", "brain_area", "cluster_id_a", "cluster_id_b",
        "n_spikes_a", "n_spikes_b", "recording_duration_sec",
        "recording_start_sec", "recording_stop_sec", "sttc_dt_sec", "sttc",
    ]
    if list(sttc_pairs.columns) != required:
        raise ValueError(f"Unexpected Q4 STTC schema: {list(sttc_pairs.columns)}")
    if len(sttc_pairs) != EXPECTED_PAIRS:
        raise ValueError(f"Expected {EXPECTED_PAIRS:,} Q4 pairs, found {len(sttc_pairs):,}")
    if sttc_pairs[required].isna().any().any():
        raise ValueError("Q4 STTC table contains missing values")
    if not np.isfinite(sttc_pairs["sttc"]).all() or not sttc_pairs["sttc"].between(-1, 1).all():
        raise ValueError("STTC values must be finite and within [-1, 1]")
    if not np.allclose(sttc_pairs["sttc_dt_sec"], STTC_DT_SEC):
        raise ValueError("Q14 requires the Q4 dt = 100 ms STTC calculation")
    if not (sttc_pairs["cluster_id_a"] < sttc_pairs["cluster_id_b"]).all():
        raise ValueError("Expected cluster_id_a < cluster_id_b for every pair")
    if sttc_pairs[["mouse_id", "brain_area", "cluster_id_a", "cluster_id_b"]].duplicated().any():
        raise ValueError("Duplicate Q4 unit pairs found")
    if sorted(sttc_pairs["mouse_id"].unique()) != list(range(1, EXPECTED_MICE + 1)):
        raise ValueError("Expected mouse IDs 1 through 18")
    if tuple(sorted(sttc_pairs["brain_area"].unique())) != EXPECTED_AREAS:
        raise ValueError("Expected brain areas 1, 2, and 3")

    networks: dict[tuple[int, int], GroupNetwork] = {}
    for (mouse_id, brain_area), group in sttc_pairs.groupby(
        ["mouse_id", "brain_area"], sort=True
    ):
        unit_ids = np.unique(np.concatenate((
            group["cluster_id_a"].to_numpy(dtype=int),
            group["cluster_id_b"].to_numpy(dtype=int),
        )))
        n_units = len(unit_ids)
        expected_group_pairs = n_units * (n_units - 1) // 2
        if len(group) != expected_group_pairs:
            raise ValueError(
                f"Mouse {mouse_id}, area {brain_area} is not a complete pair table"
            )
        index = {unit_id: position for position, unit_id in enumerate(unit_ids)}
        matrix = np.full((n_units, n_units), np.nan, dtype=float)
        np.fill_diagonal(matrix, 0.0)
        spike_counts: dict[int, int] = {}
        for row in group.itertuples(index=False):
            left = index[int(row.cluster_id_a)]
            right = index[int(row.cluster_id_b)]
            matrix[left, right] = matrix[right, left] = float(row.sttc)
            for unit_id, n_spikes in (
                (int(row.cluster_id_a), int(row.n_spikes_a)),
                (int(row.cluster_id_b), int(row.n_spikes_b)),
            ):
                previous = spike_counts.setdefault(unit_id, n_spikes)
                if previous != n_spikes:
                    raise ValueError(f"Inconsistent spike count for unit {unit_id}")
        if np.isnan(matrix).any():
            raise ValueError(f"Incomplete STTC matrix for mouse {mouse_id}, area {brain_area}")
        for column in (
            "recording_duration_sec", "recording_start_sec", "recording_stop_sec"
        ):
            if group[column].nunique() != 1:
                raise ValueError(f"{column} varies within mouse-area group")
        duration = float(group["recording_duration_sec"].iloc[0])
        start = float(group["recording_start_sec"].iloc[0])
        stop = float(group["recording_stop_sec"].iloc[0])
        if not np.isclose(stop - start, duration):
            raise ValueError("Recording window length does not equal recording duration")
        networks[(int(mouse_id), int(brain_area))] = GroupNetwork(
            mouse_id=int(mouse_id),
            brain_area=int(brain_area),
            unit_ids=unit_ids,
            sttc=matrix,
            n_spikes=np.array([spike_counts[value] for value in unit_ids], dtype=int),
            recording_duration_sec=duration,
            recording_start_sec=start,
            recording_stop_sec=stop,
        )
    if len(networks) != EXPECTED_MICE * len(EXPECTED_AREAS):
        raise ValueError("Expected all 54 mouse-area networks")
    return networks


def main() -> None:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Q4 STTC input not found: {INPUT_PATH}")

    print(f"Reading whole-recording Q4 STTC edges from {INPUT_PATH}")
    sttc_pairs = pd.read_csv(INPUT_PATH)
    networks = validate_and_build_networks(sttc_pairs)

    count_rows = []
    eligible_mice = []
    for mouse_id in range(1, EXPECTED_MICE + 1):
        counts = {
            area: len(networks[(mouse_id, area)].unit_ids)
            for area in EXPECTED_AREAS
        }
        matched = min(counts.values())
        eligible = matched >= MIN_MATCHED_UNITS
        if eligible:
            eligible_mice.append(mouse_id)
        for area in EXPECTED_AREAS:
            count_rows.append({
                "mouse_id": mouse_id,
                "brain_area": area,
                "n_available_units": counts[area],
                "n_matched_units": matched,
                "eligible": eligible,
                "exclusion_reason": "" if eligible else (
                    f"within-mouse minimum < {MIN_MATCHED_UNITS} units"
                ),
            })
    unit_counts = pd.DataFrame(count_rows)
    unit_counts.to_csv(UNIT_COUNTS_PATH, index=False)
    excluded_mice = sorted(set(range(1, EXPECTED_MICE + 1)) - set(eligible_mice))
    print(
        f"Eligible complete mice: {len(eligible_mice)}; excluded: {excluded_mice} "
        f"(minimum matched units = {MIN_MATCHED_UNITS})"
    )

    primary_rows: list[dict[str, object]] = []
    for mouse_position, mouse_id in enumerate(eligible_mice, start=1):
        matched = int(unit_counts.loc[
            unit_counts["mouse_id"].eq(mouse_id), "n_matched_units"
        ].iloc[0])
        print(
            f"Primary analysis: mouse {mouse_id} ({mouse_position}/{len(eligible_mice)}), "
            f"{matched} matched units per area"
        )
        for area in EXPECTED_AREAS:
            network = networks[(mouse_id, area)]
            for subsample_id in range(1, N_SUBSAMPLES + 1):
                primary_rows.append(calculate_subsample(
                    network=network,
                    matched_units=matched,
                    subsample_id=subsample_id,
                    density=PRIMARY_DENSITY,
                    gamma=PRIMARY_GAMMA,
                    n_nulls=N_NULLS,
                    n_observed_restarts=N_OBSERVED_RESTARTS,
                    n_null_restarts=N_NULL_RESTARTS,
                    analysis_code=10,
                ))

    primary = pd.DataFrame(primary_rows)
    primary.to_csv(SUBSAMPLES_PATH, index=False)
    primary_summary = summarize_subsamples(primary)
    primary_summary.to_csv(MOUSE_AREA_PATH, index=False)

    sensitivity_rows: list[dict[str, object]] = []
    for setting_index, (density, gamma, setting_type) in enumerate(
        SENSITIVITY_SETTINGS, start=1
    ):
        print(
            f"Sensitivity {setting_index}/{len(SENSITIVITY_SETTINGS)}: "
            f"density={density:.2f}, gamma={gamma:.2f}"
        )
        for mouse_id in eligible_mice:
            matched = int(unit_counts.loc[
                unit_counts["mouse_id"].eq(mouse_id), "n_matched_units"
            ].iloc[0])
            for area in EXPECTED_AREAS:
                network = networks[(mouse_id, area)]
                for subsample_id in range(1, SENSITIVITY_SUBSAMPLES + 1):
                    result = calculate_subsample(
                        network=network,
                        matched_units=matched,
                        subsample_id=subsample_id,
                        density=density,
                        gamma=gamma,
                        n_nulls=SENSITIVITY_NULLS,
                        n_observed_restarts=SENSITIVITY_OBSERVED_RESTARTS,
                        n_null_restarts=SENSITIVITY_NULL_RESTARTS,
                        analysis_code=20 + setting_index,
                    )
                    result["setting_type"] = setting_type
                    sensitivity_rows.append(result)

    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(SENSITIVITY_SUBSAMPLES_PATH, index=False)
    sensitivity_summary = summarize_subsamples(sensitivity)
    setting_lookup = sensitivity[["density", "gamma", "setting_type"]].drop_duplicates()
    sensitivity_summary = sensitivity_summary.merge(
        setting_lookup, on=["density", "gamma"], how="left", validate="many_to_one"
    )
    primary_for_sensitivity = primary_summary.copy()
    primary_for_sensitivity["setting_type"] = "primary"
    sensitivity_summary = pd.concat(
        [primary_for_sensitivity, sensitivity_summary], ignore_index=True
    ).sort_values(["setting_type", "density", "gamma", "mouse_id", "brain_area"])
    sensitivity_summary.to_csv(SENSITIVITY_MOUSE_AREA_PATH, index=False)

    run_config = {
        "question": "Q14: brain area with the lowest whole-recording modularity",
        "input": str(INPUT_PATH),
        "output_directory": str(OUTPUT_DIR),
        "signal": "single-unit spike trains",
        "connectivity": (
            "Q4 whole-recording pairwise STTC, dt = 0.1 s, identical recording windows"
        ),
        "network_unit": "one undirected weighted network per mouse and brain area",
        "edge_rule": "strongest fixed proportion of positive STTC edges",
        "community_detection": "weighted Leiden RB configuration partition",
        "primary_density": PRIMARY_DENSITY,
        "primary_resolution_gamma": PRIMARY_GAMMA,
        "chance_correction": (
            "delta_Q = optimized observed Q - mean optimized Q from degree-preserving "
            "double-edge-swap nulls with shuffled observed edge weights"
        ),
        "minimum_matched_units": MIN_MATCHED_UNITS,
        "eligible_mice": eligible_mice,
        "excluded_mice": excluded_mice,
        "n_primary_subsamples": N_SUBSAMPLES,
        "n_primary_nulls_per_subsample": N_NULLS,
        "n_primary_observed_leiden_restarts": N_OBSERVED_RESTARTS,
        "n_primary_null_leiden_restarts": N_NULL_RESTARTS,
        "n_sensitivity_subsamples": SENSITIVITY_SUBSAMPLES,
        "n_sensitivity_nulls_per_subsample": SENSITIVITY_NULLS,
        "sensitivity_settings": [
            {"density": density, "gamma": gamma, "type": setting_type}
            for density, gamma, setting_type in SENSITIVITY_SETTINGS
        ],
        "random_seed": RANDOM_SEED,
        "elapsed_seconds": time.perf_counter() - started,
        "software": {
            "igraph": ig.__version__,
            "leidenalg": getattr(la, "__version__", "not exposed"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    CONFIG_PATH.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print(f"Saved unit counts to {UNIT_COUNTS_PATH}")
    print(f"Saved primary subsamples to {SUBSAMPLES_PATH}")
    print(f"Saved primary mouse-area scores to {MOUSE_AREA_PATH}")
    print(f"Saved sensitivity scores to {SENSITIVITY_MOUSE_AREA_PATH}")
    print(f"Completed in {(time.perf_counter() - started) / 60:.1f} minutes")


if __name__ == "__main__":
    main()

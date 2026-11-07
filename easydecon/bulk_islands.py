"""Detect connected spatial regions supported by a mapped bulk program."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .bulk import BulkSignatureResult, _finite_number, _positive_integer


_SUMMARY_COLUMNS = [
    "island_id",
    "sample",
    "program",
    "n_observations",
    "mean_score",
    "median_score",
    "mean_balance_score",
    "mean_detected_genes",
    "centroid_x",
    "centroid_y",
    "min_x",
    "max_x",
    "min_y",
    "max_y",
]
_NICHE_COLUMNS = ["island_id", "sample", "program", "niche", "count", "fraction"]


@dataclass
class BulkIslandResult:
    """Observation assignments and summaries for descriptive bulk-program islands."""

    observations: pd.DataFrame
    island_summary: pd.DataFrame
    niche_composition: pd.DataFrame
    parameters: dict
    diagnostics: dict


def _aligned_coordinates(coordinates, index):
    if isinstance(coordinates, pd.DataFrame):
        if not coordinates.index.is_unique:
            raise ValueError("Coordinate observation identifiers must be unique.")
        if not index.isin(coordinates.index).all() or not coordinates.index.isin(index).all():
            raise ValueError(
                "coordinates must contain exactly the observations in bulk_result.scores."
            )
        values = coordinates.reindex(index).to_numpy()
    else:
        values = np.asarray(coordinates)
    if values.shape != (len(index), 2):
        raise ValueError("coordinates must have shape (n_observations, 2).")
    try:
        values = values.astype(float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("coordinates must contain finite numeric values.") from exc
    if not np.isfinite(values).all():
        raise ValueError("coordinates must contain finite numeric values.")
    return values


def _aligned_labels(labels, index, name, *, allow_missing):
    if isinstance(labels, pd.DataFrame):
        if labels.shape[1] != 1:
            raise ValueError(f"{name} must be a Series, one-column DataFrame, or array.")
        labels = labels.iloc[:, 0]
    if isinstance(labels, pd.Series):
        if not labels.index.is_unique:
            raise ValueError(f"{name} observation identifiers must be unique.")
        if not index.isin(labels.index).all() or not labels.index.isin(index).all():
            raise ValueError(
                f"{name} must contain exactly the observations in bulk_result.scores."
            )
        result = labels.reindex(index).copy()
    else:
        values = np.asarray(labels, dtype=object)
        if values.ndim != 1 or len(values) != len(index):
            raise ValueError(f"{name} must contain one value per observation.")
        result = pd.Series(values.copy(), index=index, name=name)
    if not allow_missing and result.isna().any():
        raise ValueError(f"{name} must not contain missing values.")
    return result


def _spatial_graph(coordinates, sample_labels, n_neighbors, max_distance):
    n_observations = len(coordinates)
    row_parts = []
    column_parts = []
    for sample in pd.unique(sample_labels):
        positions = np.flatnonzero(sample_labels == sample)
        if len(positions) < 2:
            continue
        k = min(n_neighbors + 1, len(positions))
        distances, neighbors = cKDTree(coordinates[positions]).query(
            coordinates[positions], k=k
        )
        if k == 1:
            distances = distances[:, None]
            neighbors = neighbors[:, None]
        local_rows = np.broadcast_to(
            np.arange(len(positions))[:, None], neighbors.shape
        )
        valid = (neighbors != local_rows) & (distances <= max_distance)
        valid &= np.cumsum(valid, axis=1) <= n_neighbors
        row_parts.append(positions[local_rows[valid]])
        column_parts.append(positions[neighbors[valid]])
    if not row_parts:
        return sparse.csr_matrix((n_observations, n_observations), dtype=np.uint8)
    rows = np.concatenate(row_parts)
    columns = np.concatenate(column_parts)
    directed = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
        shape=(n_observations, n_observations),
    )
    graph = directed.maximum(directed.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph.data[:] = 1
    return graph


def _numeric_score_column(scores, column, *, allow_nan):
    values = pd.to_numeric(scores[column], errors="coerce")
    invalid = scores[column].notna() & values.isna()
    if invalid.any() or np.isinf(values.to_numpy(dtype=float)).any():
        qualifier = "finite numeric values and NaN" if allow_nan else "finite numeric values"
        raise ValueError(f"{column!r} must contain {qualifier}.")
    return values.astype(float)


def detect_bulk_islands(
    bulk_result,
    *,
    coordinates,
    sample_labels,
    niche_labels=None,
    program="up",
    score_threshold,
    max_neighbor_distance,
    min_detected_genes=3,
    n_neighbors=6,
    min_candidate_neighbors=2,
    min_candidate_neighbor_fraction=0.5,
    min_island_cells=20,
):
    """Detect connected, locally supported regions of one mapped bulk program.

    Pandas inputs are aligned by observation ID. Array-like inputs are aligned
    positionally. Neighbor support is a single pass against the original
    candidate mask; expression scores are not smoothed or iteratively filtered.
    """
    if not isinstance(bulk_result, BulkSignatureResult):
        raise TypeError("bulk_result must be a BulkSignatureResult.")
    if program not in {"up", "down"}:
        raise ValueError("program must be one of: 'down', 'up'.")
    score_threshold = _finite_number(score_threshold, "score_threshold")
    max_neighbor_distance = _finite_number(
        max_neighbor_distance, "max_neighbor_distance", minimum=0
    )
    if max_neighbor_distance == 0:
        raise ValueError("max_neighbor_distance must be greater than 0.")
    min_detected_genes = _positive_integer(min_detected_genes, "min_detected_genes")
    n_neighbors = _positive_integer(n_neighbors, "n_neighbors")
    min_candidate_neighbors = _positive_integer(
        min_candidate_neighbors, "min_candidate_neighbors"
    )
    min_island_cells = _positive_integer(min_island_cells, "min_island_cells")
    min_candidate_neighbor_fraction = _finite_number(
        min_candidate_neighbor_fraction,
        "min_candidate_neighbor_fraction",
        minimum=0,
        maximum=1,
    )

    scores = bulk_result.scores
    if not isinstance(scores, pd.DataFrame):
        raise TypeError("bulk_result.scores must be a pandas DataFrame.")
    index = scores.index.copy()
    if not index.is_unique:
        raise ValueError("bulk_result.scores observation identifiers must be unique.")
    if pd.isna(index.to_numpy()).any():
        raise ValueError("bulk_result.scores observation identifiers must not be missing.")
    score_column = f"{program}_score"
    detected_column = f"{program}_detected_genes"
    required = [score_column, detected_column, "all_zero"]
    missing = [column for column in required if column not in scores]
    if missing:
        raise ValueError(
            "bulk_result.scores is missing required column(s): "
            f"{', '.join(map(repr, missing))}."
        )

    xy = _aligned_coordinates(coordinates, index)
    samples = _aligned_labels(
        sample_labels, index, "sample_labels", allow_missing=False
    )
    niches = (
        _aligned_labels(niche_labels, index, "niche_labels", allow_missing=True)
        if niche_labels is not None
        else None
    )
    program_scores = _numeric_score_column(scores, score_column, allow_nan=True)
    detected = _numeric_score_column(scores, detected_column, allow_nan=False)
    if detected.isna().any() or (detected < 0).any() or (detected % 1 != 0).any():
        raise ValueError(f"{detected_column!r} must contain nonnegative integer counts.")
    all_zero = scores["all_zero"]
    if all_zero.isna().any() or not all_zero.isin([True, False, 0, 1]).all():
        raise ValueError("'all_zero' must contain nonmissing boolean values.")
    all_zero = all_zero.astype(bool)
    if "balance_score" in scores:
        balance = _numeric_score_column(scores, "balance_score", allow_nan=True)
    else:
        balance = pd.Series(np.nan, index=index, dtype=float)

    graph = _spatial_graph(
        xy, samples.to_numpy(), n_neighbors, max_neighbor_distance
    )
    finite_score = np.isfinite(program_scores.to_numpy())
    candidate = (
        finite_score
        & program_scores.ge(score_threshold).to_numpy()
        & detected.ge(min_detected_genes).to_numpy()
        & ~all_zero.to_numpy()
    )
    degrees = np.asarray(graph.getnnz(axis=1)).ravel().astype(int)
    candidate_neighbors = np.asarray(graph @ candidate.astype(np.int64)).ravel().astype(int)
    candidate_fraction = np.divide(
        candidate_neighbors,
        degrees,
        out=np.zeros(len(index), dtype=float),
        where=degrees > 0,
    )
    neighbor_supported = candidate & (
        candidate_neighbors >= min_candidate_neighbors
    ) & (candidate_fraction >= min_candidate_neighbor_fraction)

    status = np.full(len(index), "insufficient_neighbor_support", dtype=object)
    status[~finite_score | all_zero.to_numpy()] = "unscored"
    status[finite_score & ~all_zero.to_numpy() & program_scores.lt(score_threshold).to_numpy()] = (
        "below_score_threshold"
    )
    status[
        finite_score
        & ~all_zero.to_numpy()
        & program_scores.ge(score_threshold).to_numpy()
        & detected.lt(min_detected_genes).to_numpy()
    ] = "insufficient_detection"
    status[candidate & ~neighbor_supported] = "insufficient_neighbor_support"

    island_ids = pd.array([pd.NA] * len(index), dtype="string")
    supported_positions = np.flatnonzero(neighbor_supported)
    component_sizes = np.array([], dtype=int)
    retained_components = []
    if len(supported_positions):
        _, labels = connected_components(
            graph[supported_positions][:, supported_positions],
            directed=False,
        )
        component_sizes = np.bincount(labels)
        components = [
            supported_positions[labels == label] for label in range(len(component_sizes))
        ]
        components.sort(key=lambda positions: int(positions.min()))
        for positions in components:
            if len(positions) < min_island_cells:
                status[positions] = "too_small_component"
                continue
            retained_components.append(positions)
            island_id = f"bulk_island_{len(retained_components):04d}"
            island_ids[positions] = island_id
            status[positions] = "assigned"

    observations = pd.DataFrame(
        {
            "sample": samples.to_numpy(copy=True),
            "x": xy[:, 0],
            "y": xy[:, 1],
            "program": program,
            "program_score": program_scores.to_numpy(copy=True),
            "balance_score": balance.to_numpy(copy=True),
            "detected_genes": detected.to_numpy(dtype=int),
            "all_zero": all_zero.to_numpy(copy=True),
            "candidate": candidate,
            "neighbor_count": degrees,
            "candidate_neighbor_count": candidate_neighbors,
            "candidate_neighbor_fraction": candidate_fraction,
            "neighbor_supported": neighbor_supported,
            "island_id": island_ids,
            "assignment_status": status,
        },
        index=index,
    )
    if niches is not None:
        observations.insert(3, "niche", niches.to_numpy(copy=True))

    summaries = []
    for positions in retained_components:
        group = observations.iloc[positions]
        summaries.append(
            {
                "island_id": str(group["island_id"].iloc[0]),
                "sample": group["sample"].iloc[0],
                "program": program,
                "n_observations": int(len(group)),
                "mean_score": float(group["program_score"].mean()),
                "median_score": float(group["program_score"].median()),
                "mean_balance_score": float(group["balance_score"].mean()),
                "mean_detected_genes": float(group["detected_genes"].mean()),
                "centroid_x": float(group["x"].mean()),
                "centroid_y": float(group["y"].mean()),
                "min_x": float(group["x"].min()),
                "max_x": float(group["x"].max()),
                "min_y": float(group["y"].min()),
                "max_y": float(group["y"].max()),
            }
        )
    island_summary = pd.DataFrame(summaries, columns=_SUMMARY_COLUMNS)

    niche_composition = pd.DataFrame(columns=_NICHE_COLUMNS)
    if niches is not None and retained_components:
        assigned = observations.loc[observations["island_id"].notna()].copy()
        assigned["niche"] = assigned["niche"].astype(object).where(
            assigned["niche"].notna(), "<missing>"
        )
        niche_composition = (
            assigned.groupby(
                ["island_id", "sample", "program", "niche"],
                observed=True,
                sort=False,
            )
            .size()
            .rename("count")
            .reset_index()
        )
        niche_composition["fraction"] = niche_composition["count"] / niche_composition.groupby(
            "island_id", observed=True
        )["count"].transform("sum")
        niche_composition = niche_composition[_NICHE_COLUMNS]

    per_specimen = []
    for sample in pd.unique(samples):
        mask = samples.eq(sample).to_numpy()
        specimen_status = pd.Series(status[mask]).value_counts()
        per_specimen.append(
            {
                "sample": sample,
                "n_observations": int(mask.sum()),
                "n_unscored": int(specimen_status.get("unscored", 0)),
                "n_below_score_threshold": int(
                    specimen_status.get("below_score_threshold", 0)
                ),
                "n_insufficient_detection": int(
                    specimen_status.get("insufficient_detection", 0)
                ),
                "n_candidates": int(candidate[mask].sum()),
                "n_neighbor_supported": int(neighbor_supported[mask].sum()),
                "n_insufficient_neighbor_support": int(
                    specimen_status.get("insufficient_neighbor_support", 0)
                ),
                "n_too_small_component": int(
                    specimen_status.get("too_small_component", 0)
                ),
                "n_assigned": int(specimen_status.get("assigned", 0)),
                "n_islands": int(observations.loc[mask, "island_id"].nunique()),
            }
        )
    status_counts = pd.Series(status).value_counts().to_dict()
    warnings = []
    if not candidate.any():
        warnings.append("No observations met the bulk-program candidate criteria.")
    if not retained_components:
        warnings.append("No bulk-program islands passed min_island_cells.")
    if status_counts.get("unscored", 0):
        warnings.append(f"{status_counts['unscored']} observations were unscored.")
    diagnostics = {
        "n_observations": int(len(index)),
        "n_unscored": int(status_counts.get("unscored", 0)),
        "n_below_score_threshold": int(
            status_counts.get("below_score_threshold", 0)
        ),
        "n_insufficient_detection": int(
            status_counts.get("insufficient_detection", 0)
        ),
        "n_candidates": int(candidate.sum()),
        "n_neighbor_supported": int(neighbor_supported.sum()),
        "n_insufficient_neighbor_support": int(
            status_counts.get("insufficient_neighbor_support", 0)
        ),
        "n_too_small_components": int(
            (component_sizes < min_island_cells).sum()
        ),
        "n_too_small_component_observations": int(
            status_counts.get("too_small_component", 0)
        ),
        "n_retained_islands": int(len(retained_components)),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "per_specimen": per_specimen,
        "warnings": warnings,
    }
    parameters = {
        "program": program,
        "score_threshold": score_threshold,
        "max_neighbor_distance": max_neighbor_distance,
        "min_detected_genes": min_detected_genes,
        "n_neighbors": n_neighbors,
        "min_candidate_neighbors": min_candidate_neighbors,
        "min_candidate_neighbor_fraction": min_candidate_neighbor_fraction,
        "min_island_cells": min_island_cells,
        "neighbor_support_rule": "one pass against the original candidate mask",
        "bulk_signature_parameters": deepcopy(bulk_result.parameters),
    }
    return BulkIslandResult(
        observations=observations,
        island_summary=island_summary,
        niche_composition=niche_composition,
        parameters=parameters,
        diagnostics=diagnostics,
    )


def plot_bulk_islands(result, *, specimen=None, point_size=8, figsize=None):
    """Plot retained bulk-program islands separately for each specimen."""
    import matplotlib.pyplot as plt

    if not isinstance(result, BulkIslandResult):
        raise TypeError("result must be a BulkIslandResult.")
    observations = result.observations
    samples = list(pd.unique(observations["sample"])) if len(observations) else []
    if specimen is not None:
        if specimen not in samples:
            raise ValueError(f"Specimen {specimen!r} was not found in result.observations.")
        samples = [specimen]
    if not samples:
        fig, axis = plt.subplots(figsize=figsize or (5, 4))
        axis.text(0.5, 0.5, "No observations", ha="center", va="center")
        axis.set_title("Bulk-program islands")
        axis.set_aspect("equal")
        return fig, np.asarray([axis])

    fig, axes = plt.subplots(
        1,
        len(samples),
        figsize=figsize or (5 * len(samples), 4),
        squeeze=False,
    )
    axes = axes[0]
    island_ids = observations["island_id"].dropna().drop_duplicates().tolist()
    colors = {
        island_id: plt.get_cmap("tab20")(position % 20)
        for position, island_id in enumerate(island_ids)
    }
    for axis, sample in zip(axes, samples):
        frame = observations.loc[observations["sample"].eq(sample)]
        unscored = frame["assignment_status"].eq("unscored")
        background = frame["island_id"].isna() & ~unscored
        axis.scatter(
            frame.loc[background, "x"],
            frame.loc[background, "y"],
            c="#d0d0d0",
            s=point_size,
            label="not assigned",
        )
        axis.scatter(
            frame.loc[unscored, "x"],
            frame.loc[unscored, "y"],
            c="#707070",
            marker="x",
            s=point_size,
            label="unscored",
        )
        for island_id, group in frame.loc[frame["island_id"].notna()].groupby(
            "island_id", observed=True, sort=False
        ):
            axis.scatter(
                group["x"],
                group["y"],
                c=[colors[island_id]],
                s=point_size,
                label=str(island_id),
            )
        if frame["island_id"].isna().all():
            axis.text(
                0.5,
                0.95,
                "No retained islands",
                transform=axis.transAxes,
                ha="center",
                va="top",
            )
        axis.set_title(f"Bulk-program islands — {sample}")
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.legend(loc="best", fontsize="small")
    fig.tight_layout()
    return fig, axes


__all__ = ["BulkIslandResult", "detect_bulk_islands", "plot_bulk_islands"]

"""Deterministic synthetic validation helpers for easydecon benchmarks.

These utilities intentionally live outside the public ``easydecon`` package
API. They create controlled synthetic data and reports for implementation
validation; they do not establish biological superiority of any method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import platform
import time
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

import easydecon as ed
from easydecon.config import config as easydecon_config
from easydecon.config import set_n_jobs


@dataclass
class SyntheticValidationDataset:
    spatial: ad.AnnData
    reference: ad.AnnData
    true_labels: pd.Series
    true_scores: pd.DataFrame
    marker_groups: dict
    scenario: str
    metadata: dict


@dataclass(frozen=True)
class ValidationConfiguration:
    name: str
    marker_source: str
    marker_method: str | None
    reference_contrast: str | None
    marker_roles: str
    phase2_method: str
    candidate_pruning: bool
    candidate_threshold: float
    evidence_to_likelihood: str
    extra_kwargs: dict = field(default_factory=dict)


SCENARIO_OVERRIDES = {
    "clean": {},
    "dropout": {"dropout_rate": 0.35},
    "shared_markers": {"shared_marker_multiplier": 3, "marker_mean": 2.5},
    "library_shift": {"library_shift": True},
    "contamination": {"contamination_rate": 0.25},
    "mixed": {"mixture_fraction": 0.30},
    "difficult": {
        "dropout_rate": 0.35,
        "contamination_rate": 0.25,
        "mixture_fraction": 0.20,
        "marker_mean": 1.5,
        "library_shift": True,
        "shared_marker_multiplier": 2,
    },
}


def _as_dense(matrix) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def _maybe_sparse(matrix: np.ndarray, sparse_output: bool):
    return sparse.csr_matrix(matrix) if sparse_output else matrix


def _log_normalize_counts(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    library = counts.sum(axis=1, keepdims=True)
    library[library <= 0] = 1.0
    return np.log1p(counts / library * 1e4)


def _sample_counts(rng, expected, library_size, dropout_rate):
    expected = np.asarray(expected, dtype=float)
    expected = np.clip(expected, 0.0, None)
    total = expected.sum()
    if total <= 0:
        probs = np.full(expected.shape, 1.0 / len(expected))
    else:
        probs = expected / total
    lam = probs * float(library_size)
    counts = rng.poisson(lam).astype(np.int64)
    if dropout_rate > 0:
        dropout = rng.random(counts.shape) < dropout_rate
        counts = counts.copy()
        counts[dropout] = 0
    return counts


def _scenario_parameters(
    scenario,
    *,
    shared_markers,
    dropout_rate,
    contamination_rate,
    mixture_fraction,
    marker_mean,
):
    if scenario not in SCENARIO_OVERRIDES:
        raise ValueError(
            f"scenario must be one of {sorted(SCENARIO_OVERRIDES)}. Got {scenario!r}."
        )
    params = {
        "dropout_rate": dropout_rate,
        "contamination_rate": contamination_rate,
        "mixture_fraction": mixture_fraction,
        "marker_mean": marker_mean,
        "library_shift": False,
        "shared_markers": shared_markers,
    }
    overrides = SCENARIO_OVERRIDES[scenario]
    params.update({k: v for k, v in overrides.items() if k != "shared_marker_multiplier"})
    if "shared_marker_multiplier" in overrides:
        params["shared_markers"] = int(shared_markers * overrides["shared_marker_multiplier"])
    return params


def make_synthetic_validation_dataset(
    *,
    scenario="clean",
    n_groups=4,
    n_reference_cells_per_group=80,
    n_spots_per_group=50,
    n_genes=400,
    markers_per_group=20,
    shared_markers=5,
    negative_markers_per_group=5,
    background_mean=0.05,
    marker_mean=3.0,
    library_size_mean=2000,
    dropout_rate=0.0,
    contamination_rate=0.0,
    mixture_fraction=0.0,
    random_state=0,
    sparse=True,
) -> SyntheticValidationDataset:
    """Create a deterministic synthetic reference and spatial dataset."""
    rng = np.random.default_rng(random_state)
    params = _scenario_parameters(
        scenario,
        shared_markers=shared_markers,
        dropout_rate=dropout_rate,
        contamination_rate=contamination_rate,
        mixture_fraction=mixture_fraction,
        marker_mean=marker_mean,
    )
    dropout_rate = float(params["dropout_rate"])
    contamination_rate = float(params["contamination_rate"])
    mixture_fraction = float(params["mixture_fraction"])
    marker_mean = float(params["marker_mean"])
    shared_markers = int(params["shared_markers"])
    library_shift = bool(params["library_shift"])

    groups = [f"Type_{idx}" for idx in range(n_groups)]
    genes = [f"Gene_{idx:04d}" for idx in range(n_genes)]
    required = n_groups * markers_per_group + shared_markers + n_groups * negative_markers_per_group
    if required > n_genes:
        raise ValueError(
            "n_genes is too small for the requested marker architecture. "
            f"Need at least {required}, got {n_genes}."
        )

    cursor = 0
    positives = {}
    for group in groups:
        positives[group] = genes[cursor : cursor + markers_per_group]
        cursor += markers_per_group
    shared = genes[cursor : cursor + shared_markers]
    cursor += shared_markers
    negatives = {}
    negative_targets = {}
    for group_idx, group in enumerate(groups):
        negatives[group] = genes[cursor : cursor + negative_markers_per_group]
        negative_targets[group] = groups[(group_idx + 1) % n_groups]
        cursor += negative_markers_per_group

    profiles = pd.DataFrame(background_mean, index=groups, columns=genes, dtype=float)
    for group in groups:
        profiles.loc[group, positives[group]] = marker_mean
        profiles.loc[group, shared] = max(background_mean, marker_mean * 0.65)
    for group, anti_genes in negatives.items():
        profiles.loc[group, anti_genes] = background_mean
        profiles.loc[negative_targets[group], anti_genes] = marker_mean

    reference_counts = []
    reference_labels = []
    sample_ids = []
    for group in groups:
        for cell_idx in range(n_reference_cells_per_group):
            lib = rng.lognormal(np.log(library_size_mean), 0.15)
            counts = _sample_counts(
                rng,
                profiles.loc[group].to_numpy(),
                lib,
                dropout_rate=0.0,
            )
            reference_counts.append(counts)
            reference_labels.append(group)
            sample_ids.append(f"{group}_rep{cell_idx % 2}")
    reference_counts = np.vstack(reference_counts).astype(np.int64)

    spatial_counts = []
    labels = []
    score_rows = []
    coordinates = []
    for group_idx, group in enumerate(groups):
        for spot_idx in range(n_spots_per_group):
            weights = pd.Series(0.0, index=groups)
            weights[group] = 1.0
            if mixture_fraction > 0:
                secondary = groups[(group_idx + 1 + spot_idx) % n_groups]
                if secondary == group:
                    secondary = groups[(group_idx + 1) % n_groups]
                weights[group] = 1.0 - mixture_fraction
                weights[secondary] += mixture_fraction
            expected = weights.to_numpy() @ profiles.loc[groups].to_numpy()
            if contamination_rate > 0:
                contaminant = groups[(group_idx + 1) % n_groups]
                expected = (
                    (1.0 - contamination_rate) * expected
                    + contamination_rate * profiles.loc[contaminant].to_numpy()
                )
            lib = (
                rng.lognormal(np.log(library_size_mean), 0.8)
                if library_shift
                else rng.lognormal(np.log(library_size_mean), 0.15)
            )
            counts = _sample_counts(rng, expected, lib, dropout_rate=dropout_rate)
            spatial_counts.append(counts)
            labels.append(group)
            score_rows.append(weights)
            coordinates.append([group_idx * 10.0, float(spot_idx)])
    spatial_counts = np.vstack(spatial_counts).astype(np.int64)
    true_index = [f"spot_{idx:04d}" for idx in range(spatial_counts.shape[0])]
    true_labels = pd.Series(labels, index=true_index, name="true_label")
    true_scores = pd.DataFrame(score_rows, index=true_index, columns=groups)
    true_scores = true_scores.div(true_scores.sum(axis=1), axis=0)

    reference = ad.AnnData(
        X=_maybe_sparse(_log_normalize_counts(reference_counts), sparse),
        obs=pd.DataFrame(
            {"cell_type": reference_labels, "sample_id": sample_ids},
            index=[f"ref_{idx:05d}" for idx in range(reference_counts.shape[0])],
        ),
        var=pd.DataFrame(index=genes),
    )
    reference.layers["counts"] = _maybe_sparse(reference_counts, sparse)

    spatial_adata = ad.AnnData(
        X=_maybe_sparse(_log_normalize_counts(spatial_counts), sparse),
        obs=pd.DataFrame(index=true_index),
        var=pd.DataFrame(index=genes),
    )
    spatial_adata.layers["counts"] = _maybe_sparse(spatial_counts, sparse)
    spatial_adata.obsm["spatial"] = np.asarray(coordinates, dtype=float)

    marker_groups = {
        "positive": positives,
        "shared": shared,
        "negative": negatives,
        "negative_targets": negative_targets,
        "background": genes[cursor:],
    }
    metadata = {
        "scenario": scenario,
        "random_state": int(random_state),
        "n_groups": int(n_groups),
        "n_reference_cells_per_group": int(n_reference_cells_per_group),
        "n_spots_per_group": int(n_spots_per_group),
        "n_genes": int(n_genes),
        "markers_per_group": int(markers_per_group),
        "shared_markers": int(shared_markers),
        "negative_markers_per_group": int(negative_markers_per_group),
        "background_mean": float(background_mean),
        "marker_mean": float(marker_mean),
        "library_size_mean": float(library_size_mean),
        "dropout_rate": float(dropout_rate),
        "contamination_rate": float(contamination_rate),
        "mixture_fraction": float(mixture_fraction),
        "library_shift": bool(library_shift),
        "sparse": bool(sparse),
    }
    return SyntheticValidationDataset(
        spatial=spatial_adata,
        reference=reference,
        true_labels=true_labels,
        true_scores=true_scores,
        marker_groups=marker_groups,
        scenario=scenario,
        metadata=metadata,
    )


def make_known_marker_table(
    dataset: SyntheticValidationDataset,
    marker_roles="shared",
) -> pd.DataFrame:
    """Build deterministic oracle marker definitions for benchmark isolation."""
    rows = []
    groups = list(dataset.marker_groups["positive"])
    for group in groups:
        for rank, gene in enumerate(dataset.marker_groups["positive"][group]):
            base = {
                "group": group,
                "names": gene,
                "logfoldchanges": float(5.0 - rank * 0.01),
                "scores": float(100.0 - rank),
                "marker_source": "known",
            }
            if marker_roles == "phase_specific":
                rows.append({**base, "marker_role": "presence"})
                rows.append({**base, "marker_role": "identity"})
            elif marker_roles == "shared":
                rows.append(base)
            else:
                raise ValueError("marker_roles must be 'shared' or 'phase_specific'.")
        if marker_roles == "shared":
            for rank, gene in enumerate(dataset.marker_groups["shared"]):
                rows.append(
                    {
                        "group": group,
                        "names": gene,
                        "logfoldchanges": float(1.0 - rank * 0.01),
                        "scores": float(20.0 - rank),
                        "marker_source": "known",
                    }
                )
        else:
            for rank, gene in enumerate(dataset.marker_groups["negative"][group]):
                rows.append(
                    {
                        "group": group,
                        "names": gene,
                        "logfoldchanges": float(-2.0 - rank * 0.01),
                        "scores": float(50.0 - rank),
                        "marker_role": "negative",
                        "marker_source": "known",
                    }
                )
    return pd.DataFrame(rows)


def default_validation_configurations() -> list[ValidationConfiguration]:
    return [
        ValidationConfiguration("known_shared_wjaccard", "known", None, None, "shared", "wjaccard", False, 0.0, "softmax", {}),
        ValidationConfiguration("known_shared_auc", "known", None, None, "shared", "auc", False, 0.0, "softmax", {"min_markers": 2}),
        ValidationConfiguration("known_shared_ucell", "known", None, None, "shared", "ucell", False, 0.0, "softmax", {"min_markers": 2}),
        ValidationConfiguration("known_phase_specific_ucell", "known", None, None, "phase_specific", "ucell", False, 0.0, "softmax", {"min_markers": 2}),
        ValidationConfiguration("reference_max_other_wjaccard", "reference", "reference", "max_other", "shared", "wjaccard", False, 0.0, "softmax", {}),
        ValidationConfiguration("reference_max_other_ucell", "reference", "reference", "max_other", "shared", "ucell", False, 0.0, "softmax", {"min_markers": 2}),
        ValidationConfiguration("reference_mean_other_ucell", "reference", "reference", "mean_other", "shared", "ucell", False, 0.0, "softmax", {"min_markers": 2}),
        ValidationConfiguration("reference_phase_specific_ucell", "reference", "reference", "max_other", "phase_specific", "ucell", False, 0.0, "softmax", {"min_markers": 2}),
        ValidationConfiguration("reference_ucell_unpruned", "reference", "reference", "max_other", "shared", "ucell", False, 0.0, "softmax", {"min_markers": 2}),
        ValidationConfiguration("reference_ucell_pruned_zero", "reference", "reference", "max_other", "shared", "ucell", True, 0.0, "softmax", {"min_markers": 2}),
        ValidationConfiguration("reference_ucell_pruned_threshold", "reference", "reference", "max_other", "shared", "ucell", True, 0.05, "softmax", {"min_markers": 2}),
    ]


def _configuration_by_name(configurations=None):
    configs = default_validation_configurations() if configurations is None else list(configurations)
    return {config.name: config for config in configs}


def _base_workflow_kwargs(configuration: ValidationConfiguration) -> dict:
    kwargs = {
        "filtering_algorithm": "quantile",
        "quantile": 0.5,
        "assign_method": "max",
        "minimum_evidence": 0.0,
        "tie_tolerance": 1e-12,
        "return_result_object": True,
        "verbose": False,
        "marker_roles": configuration.marker_roles,
        "method": configuration.phase2_method,
        "evidence_to_likelihood": configuration.evidence_to_likelihood,
        "phase2_candidate_pruning": configuration.candidate_pruning,
        "phase2_candidate_threshold": configuration.candidate_threshold,
        "log2fc_min": -np.inf,
        "pval_cutoff": 1.0,
        "drop_ribosomal": False,
        "drop_mitochondrial": False,
        "reference_min_cells": 5,
        "reference_min_mean": 0.0,
        "reference_min_log2fc": 0.25,
        "reference_min_detection": 0.05,
        "reference_min_detection_delta": 0.0,
        "reference_presence_min_log2fc": 0.25,
        "reference_presence_min_detection_delta": 0.0,
        "reference_negative_min_log2fc": 0.25,
        "reference_negative_min_detection": 0.05,
        "reference_negative_min_detection_delta": 0.0,
        "top_n_genes": None,
        "copy_adata": True,
    }
    kwargs.update(configuration.extra_kwargs)
    return kwargs


def _markers_for_configuration(dataset, configuration):
    if configuration.marker_source == "known":
        return make_known_marker_table(dataset, marker_roles=configuration.marker_roles), None
    if configuration.marker_source == "reference":
        return None, dataset.reference
    raise ValueError(f"Unsupported marker_source={configuration.marker_source!r}.")


def _prediction_series(result, true_index) -> pd.Series:
    assigned = result.assigned_labels.reindex(true_index)
    if assigned.shape[1] == 0:
        return pd.Series(np.nan, index=true_index)
    return assigned.iloc[:, 0].replace("", np.nan)


def assignment_metrics(true_labels, predicted_labels) -> dict:
    true_labels = pd.Series(true_labels).astype(str)
    predicted = pd.Series(predicted_labels, index=true_labels.index)
    assigned_mask = predicted.notna()
    n_total = int(len(true_labels))
    n_assigned = int(assigned_mask.sum())
    correct = (predicted.astype("object") == true_labels.astype("object")) & assigned_mask
    n_correct = int(correct.sum())
    labels = sorted(true_labels.unique().tolist())
    precisions = []
    recalls = []
    f1s = []
    for label in labels:
        tp = int(((true_labels == label) & (predicted == label)).sum())
        fp = int(((true_labels != label) & (predicted == label)).sum())
        fn = int(((true_labels == label) & (predicted != label)).sum())
        precision = np.nan if tp + fp == 0 else tp / (tp + fp)
        recall = np.nan if tp + fn == 0 else tp / (tp + fn)
        f1 = (
            np.nan
            if not np.isfinite(precision) or not np.isfinite(recall) or precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    def _safe_nanmean(values):
        values = np.asarray(values, dtype=float)
        return np.nan if values.size == 0 or np.isnan(values).all() else float(np.nanmean(values))

    return {
        "n_locations": n_total,
        "n_assigned": n_assigned,
        "n_unassigned": int(n_total - n_assigned),
        "coverage": 0.0 if n_total == 0 else n_assigned / n_total,
        "assigned_accuracy": np.nan if n_assigned == 0 else n_correct / n_assigned,
        "overall_accuracy": 0.0 if n_total == 0 else n_correct / n_total,
        "macro_recall": _safe_nanmean(recalls),
        "macro_precision": _safe_nanmean(precisions),
        "macro_f1": _safe_nanmean(f1s),
    }


def confusion_counts(true_labels, predicted_labels) -> pd.DataFrame:
    true_labels = pd.Series(true_labels).astype(str)
    predicted = pd.Series(predicted_labels, index=true_labels.index).astype("object")
    predicted = predicted.where(predicted.notna(), "<unassigned>").astype(str)
    return (
        pd.DataFrame({"true_label": true_labels, "predicted_label": predicted})
        .groupby(["true_label", "predicted_label"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["true_label", "predicted_label"], kind="stable")
        .reset_index(drop=True)
    )


def _normalized_score_matrix(score_df: pd.DataFrame) -> pd.DataFrame:
    scores = score_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scores = scores.clip(lower=0.0)
    row_sum = scores.sum(axis=1).replace(0, np.nan)
    return scores.div(row_sum, axis=0).fillna(0.0)


def _score_matrix_from_result(result) -> pd.DataFrame:
    if result.posterior_df is not None:
        return result.posterior_df.copy()
    return _normalized_score_matrix(result.assignment_df)


def score_ranking_metrics(score_df, true_labels, true_scores=None) -> dict:
    true_labels = pd.Series(true_labels).astype(str)
    scores = _normalized_score_matrix(score_df.reindex(true_labels.index).copy())
    labels = list(scores.columns)
    n_labels = len(labels)
    true_group_scores = []
    max_scores = []
    ranks = []
    reciprocal = []
    top1_correct = []
    entropies = []
    for idx, true_label in true_labels.items():
        row = scores.loc[idx] if idx in scores.index else pd.Series(dtype=float)
        total = float(row.sum()) if len(row) else 0.0
        if total <= 0 or n_labels == 0:
            true_group_scores.append(0.0)
            max_scores.append(0.0)
            ranks.append(n_labels + 1)
            reciprocal.append(0.0)
            top1_correct.append(False)
            entropies.append(np.nan)
            continue
        p_true = float(row.get(true_label, 0.0))
        true_group_scores.append(p_true)
        max_scores.append(float(row.max()))
        sorted_values = row.sort_values(ascending=False, kind="mergesort")
        if true_label in sorted_values.index:
            rank = int(np.flatnonzero(sorted_values.index == true_label)[0] + 1)
        else:
            rank = n_labels + 1
        ranks.append(rank)
        reciprocal.append(0.0 if rank > n_labels else 1.0 / rank)
        top1_correct.append(sorted_values.index[0] == true_label)
        positive = row[row > 0]
        entropies.append(float(-(positive * np.log(positive)).sum()))
    metrics = {
        "top1_score_accuracy": float(np.mean(top1_correct)) if top1_correct else np.nan,
        "mean_true_group_rank": float(np.mean(ranks)) if ranks else np.nan,
        "mean_reciprocal_rank": float(np.mean(reciprocal)) if reciprocal else np.nan,
        "mean_true_group_score": float(np.mean(true_group_scores)) if true_group_scores else np.nan,
        "median_true_group_score": float(np.median(true_group_scores)) if true_group_scores else np.nan,
        "mean_max_score": float(np.mean(max_scores)) if max_scores else np.nan,
        "median_max_score": float(np.median(max_scores)) if max_scores else np.nan,
        "mean_score_entropy": float(np.nanmean(entropies)) if entropies else np.nan,
    }
    if np.allclose(scores.sum(axis=1).to_numpy(), 1.0, atol=1e-6):
        metrics.update(probability_metrics(scores, true_labels, true_scores=true_scores))
    return metrics


def probability_metrics(probabilities, true_labels, true_scores=None, epsilon=1e-12) -> dict:
    true_labels = pd.Series(true_labels).astype(str)
    probs = _normalized_score_matrix(probabilities.reindex(true_labels.index).copy())
    all_columns = sorted(set(probs.columns).union(set(true_labels.unique())))
    probs = probs.reindex(columns=all_columns, fill_value=0.0)
    targets = pd.DataFrame(0.0, index=true_labels.index, columns=all_columns)
    for idx, label in true_labels.items():
        targets.loc[idx, label] = 1.0
    clipped_true = np.array([probs.loc[idx, label] for idx, label in true_labels.items()])
    clipped_true = np.clip(clipped_true, epsilon, 1.0)
    out = {
        "multiclass_brier": float(((probs - targets) ** 2).sum(axis=1).mean()),
        "negative_log_likelihood": float((-np.log(clipped_true)).mean()),
    }
    if true_scores is not None:
        aligned_true = true_scores.reindex(index=true_labels.index, columns=all_columns, fill_value=0.0)
        diff = probs - aligned_true
        out["composition_mae"] = float(diff.abs().to_numpy().mean())
        out["composition_rmse"] = float(np.sqrt((diff.to_numpy() ** 2).mean()))
    else:
        out["composition_mae"] = np.nan
        out["composition_rmse"] = np.nan
    return out


def abstention_diagnostics(true_labels, predicted_labels, score_df=None) -> dict:
    true_labels = pd.Series(true_labels).astype(str)
    predicted = pd.Series(predicted_labels, index=true_labels.index)
    assigned = predicted.notna()
    correct = (predicted == true_labels) & assigned
    incorrect = assigned & ~correct
    out = {
        "correct_assigned": int(correct.sum()),
        "incorrect_assigned": int(incorrect.sum()),
        "correct_fraction_of_all": float(correct.mean()) if len(correct) else np.nan,
        "error_fraction_of_assigned": (
            np.nan if int(assigned.sum()) == 0 else float(incorrect.sum() / assigned.sum())
        ),
        "unassigned_fraction": float((~assigned).mean()) if len(assigned) else np.nan,
    }
    if score_df is not None:
        scores = _normalized_score_matrix(score_df.reindex(true_labels.index).copy())
        max_score = scores.max(axis=1)
        out["median_max_posterior_correct"] = float(max_score[correct].median()) if correct.any() else np.nan
        out["median_max_posterior_incorrect"] = float(max_score[incorrect].median()) if incorrect.any() else np.nan
        out["median_max_posterior_unassigned"] = float(max_score[~assigned].median()) if (~assigned).any() else np.nan
    return out


def marker_diagnostics(marker_table, marker_diagnostics_dict=None) -> dict:
    if marker_table is None or marker_table.empty:
        out = {
            "n_markers_total": 0,
            "n_marker_groups": 0,
            "min_markers_per_group": 0,
            "median_markers_per_group": 0.0,
            "max_markers_per_group": 0,
        }
    else:
        marker_table = marker_table.reset_index(drop=True)
        counts = marker_table.groupby("group").size()
        out = {
            "n_markers_total": int(marker_table.shape[0]),
            "n_marker_groups": int(counts.shape[0]),
            "min_markers_per_group": int(counts.min()),
            "median_markers_per_group": float(counts.median()),
            "max_markers_per_group": int(counts.max()),
        }
        if "marker_role" in marker_table.columns:
            roles = marker_table["marker_role"].astype(str).str.casefold()
            out.update(
                {
                    "n_presence_markers": int((roles == "presence").sum()),
                    "n_identity_markers": int((roles == "identity").sum()),
                    "n_positive_markers": int(roles.isin(["positive", "identity"]).sum()),
                    "n_negative_markers": int((roles == "negative").sum()),
                }
            )
    if marker_diagnostics_dict:
        out["reference_contrast"] = marker_diagnostics_dict.get("reference_contrast")
        ref_diag = marker_diagnostics_dict.get("reference_profile")
        if isinstance(ref_diag, dict):
            out["reference_n_groups_retained"] = ref_diag.get("n_groups_retained")
            out["reference_n_markers"] = ref_diag.get("n_markers")
    return out


def candidate_pruning_metrics(result) -> dict:
    perf = result.diagnostics.get("phase2", {}).get("performance", {})
    enabled = bool(perf.get("candidate_pruning_enabled", False))
    fraction = float(perf.get("candidate_fraction", 1.0 if not enabled else np.nan))
    return {
        "candidate_pruning_enabled": enabled,
        "candidate_threshold": float(perf.get("candidate_threshold", 0.0)),
        "exact_candidate_pruning": bool(perf.get("exact_candidate_pruning", False)),
        "n_total_location_group_pairs": perf.get(
            "n_total_location_group_pairs",
            int(result.phase2_result.shape[0] * result.phase2_result.shape[1]),
        ),
        "n_candidate_pairs": perf.get(
            "n_candidate_pairs",
            int(result.phase2_result.shape[0] * result.phase2_result.shape[1]),
        ),
        "candidate_fraction": fraction,
        "n_rows_without_candidates": int(perf.get("n_rows_without_candidates", 0)),
        "evaluated_pair_reduction": 0.0 if not np.isfinite(fraction) else 1.0 - fraction,
    }


def compare_candidate_pruning(unpruned_result, pruned_result) -> dict:
    if unpruned_result.posterior_df is None or pruned_result.posterior_df is None:
        posterior_max = np.nan
        posterior_mean = np.nan
    else:
        left, right = unpruned_result.posterior_df.align(
            pruned_result.posterior_df, join="outer", axis=None, fill_value=0.0
        )
        diff = (left - right).abs()
        posterior_max = float(diff.to_numpy().max()) if diff.size else 0.0
        posterior_mean = float(diff.to_numpy().mean()) if diff.size else 0.0
    pred_left = _prediction_series(
        unpruned_result, unpruned_result.assigned_labels.index
    ).astype("object")
    pred_right = _prediction_series(pruned_result, pred_left.index).astype("object")
    equal = (pred_left.fillna("<NA>") == pred_right.fillna("<NA>"))
    phase2_values = pruned_result.phase2_result.to_numpy(dtype=float)
    return {
        "posterior_max_abs_difference": posterior_max,
        "posterior_mean_abs_difference": posterior_mean,
        "assignments_equal": bool(equal.all()),
        "n_assignment_differences": int((~equal).sum()),
        "candidate_phase2_zero_fraction": float((phase2_values == 0).mean()) if phase2_values.size else np.nan,
    }


def run_validation_configuration(
    dataset: SyntheticValidationDataset,
    configuration: ValidationConfiguration,
    *,
    verbose=False,
) -> dict:
    marker_table, reference = _markers_for_configuration(dataset, configuration)
    spatial_copy = dataset.spatial.copy()
    kwargs = _base_workflow_kwargs(configuration)
    if marker_table is not None:
        kwargs["markers_df"] = marker_table.copy()
    if reference is not None:
        kwargs.update(
            {
                "adata": reference,
                "marker_method": configuration.marker_method or "reference",
                "groupby": "cell_type",
                "sample_col": "sample_id",
                "layer": "counts",
                "reference_contrast": configuration.reference_contrast or "max_other",
            }
        )
    kwargs["verbose"] = verbose
    old_n_jobs = easydecon_config.n_jobs
    set_n_jobs(1)
    start = time.perf_counter()
    try:
        result = ed.run_easydecon(spatial_copy, **kwargs)
    finally:
        set_n_jobs(old_n_jobs)
    elapsed = time.perf_counter() - start

    true_labels = dataset.true_labels.reindex(spatial_copy.obs.index)
    predictions = _prediction_series(result, true_labels.index)
    scores = _score_matrix_from_result(result).reindex(true_labels.index)
    metrics = {
        "scenario": dataset.scenario,
        "configuration": configuration.name,
        "status": "ok",
        "marker_source": configuration.marker_source,
        "marker_method": configuration.marker_method,
        "reference_contrast": configuration.reference_contrast,
        "marker_roles": configuration.marker_roles,
        "phase2_method": configuration.phase2_method,
        "candidate_pruning": configuration.candidate_pruning,
        "candidate_threshold": configuration.candidate_threshold,
        "evidence_to_likelihood": configuration.evidence_to_likelihood,
    }
    metrics.update(assignment_metrics(true_labels, predictions))
    metrics.update(score_ranking_metrics(scores, true_labels, true_scores=dataset.true_scores))
    metrics.update(abstention_diagnostics(true_labels, predictions, score_df=scores))
    metrics.update(marker_diagnostics(result.markers_df, result.diagnostics.get("markers")))
    metrics.update(candidate_pruning_metrics(result))
    n_locations = int(len(true_labels))
    n_pairs = int(n_locations * max(1, result.phase2_result.shape[1]))
    n_candidate_pairs = metrics.get("n_candidate_pairs", n_pairs)
    metrics.update(
        {
            "elapsed_seconds": float(elapsed),
            "locations_per_second": float(n_locations / elapsed) if elapsed > 0 else np.nan,
            "location_group_pairs_per_second": float(n_pairs / elapsed) if elapsed > 0 else np.nan,
            "evaluated_candidate_pairs_per_second": (
                float(n_candidate_pairs / elapsed) if elapsed > 0 else np.nan
            ),
        }
    )
    return {
        "configuration": configuration,
        "result": result,
        "metrics": metrics,
        "confusion": confusion_counts(true_labels, predictions),
        "markers": result.markers_df.copy(),
    }


def _failed_metrics_row(dataset, configuration, random_state, repetition, exc):
    return {
        "scenario": dataset.scenario,
        "configuration": configuration.name,
        "random_state": random_state,
        "repetition": repetition,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "marker_source": configuration.marker_source,
        "marker_method": configuration.marker_method,
        "reference_contrast": configuration.reference_contrast,
        "marker_roles": configuration.marker_roles,
        "phase2_method": configuration.phase2_method,
        "candidate_pruning": configuration.candidate_pruning,
        "candidate_threshold": configuration.candidate_threshold,
    }


def run_validation_suite(
    *,
    scenarios=None,
    configurations=None,
    random_states=(0,),
    repeat=1,
    sparse=True,
    verbose=False,
    retain_results=False,
    dataset_kwargs=None,
) -> tuple[pd.DataFrame, dict]:
    scenarios = list(scenarios or ["clean", "dropout", "shared_markers"])
    if configurations is None:
        config_objects = default_validation_configurations()
    else:
        lookup = _configuration_by_name()
        config_objects = [lookup[c] if isinstance(c, str) else c for c in configurations]
    dataset_kwargs = dict(dataset_kwargs or {})
    rows = []
    details = {"confusion": [], "pruning_comparisons": [], "marker_summaries": []}
    if retain_results:
        details["results"] = []

    for scenario in scenarios:
        for random_state in random_states:
            dataset = make_synthetic_validation_dataset(
                scenario=scenario,
                random_state=int(random_state),
                sparse=sparse,
                **dataset_kwargs,
            )
            completed = {}
            for repetition in range(int(repeat)):
                for config in config_objects:
                    try:
                        run = run_validation_configuration(dataset, config, verbose=verbose)
                        row = dict(run["metrics"])
                        row.update({"random_state": int(random_state), "repetition": int(repetition)})
                        rows.append(row)
                        confusion = run["confusion"].copy()
                        confusion.insert(0, "repetition", int(repetition))
                        confusion.insert(0, "random_state", int(random_state))
                        confusion.insert(0, "configuration", config.name)
                        confusion.insert(0, "scenario", scenario)
                        details["confusion"].append(confusion)
                        details["marker_summaries"].append(
                            {
                                key: row.get(key)
                                for key in row
                                if key.startswith("n_marker")
                                or key.endswith("markers")
                                or key.startswith("reference_")
                            }
                            | {
                                "scenario": scenario,
                                "configuration": config.name,
                                "random_state": int(random_state),
                                "repetition": int(repetition),
                            }
                        )
                        if retain_results:
                            details["results"].append(run)
                        completed[config.name] = run["result"]
                    except Exception as exc:  # suite-level continuation is intentional
                        rows.append(_failed_metrics_row(dataset, config, int(random_state), int(repetition), exc))
                unpruned = completed.get("reference_ucell_unpruned")
                for pruned_name in ("reference_ucell_pruned_zero", "reference_ucell_pruned_threshold"):
                    if unpruned is not None and pruned_name in completed:
                        comparison = compare_candidate_pruning(unpruned, completed[pruned_name])
                        comparison.update(
                            {
                                "scenario": scenario,
                                "configuration": pruned_name,
                                "random_state": int(random_state),
                                "repetition": int(repetition),
                            }
                        )
                        details["pruning_comparisons"].append(comparison)

    metrics_df = pd.DataFrame(rows)
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(
            ["scenario", "configuration", "random_state", "repetition"],
            kind="stable",
        ).reset_index(drop=True)
    return metrics_df, details


def summarize_validation_results(metrics_df) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    ok = metrics_df[metrics_df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame(
            columns=[
                "scenario",
                "configuration",
                "phase2_method",
                "marker_source",
                "marker_roles",
                "reference_contrast",
                "candidate_pruning",
                "overall_accuracy_mean",
                "coverage_mean",
                "assigned_accuracy_mean",
                "macro_f1_mean",
                "reciprocal_rank_mean",
                "brier_mean",
                "composition_mae_mean",
                "elapsed_median",
                "candidate_fraction_mean",
            ]
        )
    keys = [
        "overall_accuracy",
        "coverage",
        "assigned_accuracy",
        "macro_f1",
        "mean_reciprocal_rank",
        "multiclass_brier",
        "composition_mae",
        "elapsed_seconds",
        "candidate_fraction",
    ]
    present = [key for key in keys if key in ok.columns]
    grouped = ok.groupby(["scenario", "configuration"], dropna=False)
    frames = []
    for key in present:
        frames.append(grouped[key].mean().rename(f"{key}_mean"))
        frames.append(grouped[key].std(ddof=0).rename(f"{key}_std"))
    summary = pd.concat(frames, axis=1).reset_index() if frames else grouped.size().reset_index(name="n")
    metadata_cols = [
        "phase2_method",
        "marker_source",
        "marker_roles",
        "reference_contrast",
        "candidate_pruning",
    ]
    first_meta = grouped[[c for c in metadata_cols if c in ok.columns]].first().reset_index()
    summary = summary.merge(first_meta, on=["scenario", "configuration"], how="left")
    rename = {
        "mean_reciprocal_rank_mean": "reciprocal_rank_mean",
        "multiclass_brier_mean": "brier_mean",
        "elapsed_seconds_mean": "elapsed_mean",
        "elapsed_seconds_std": "elapsed_std",
    }
    summary = summary.rename(columns=rename)
    elapsed = grouped["elapsed_seconds"].agg(
        elapsed_median="median",
        elapsed_min="min",
        elapsed_max="max",
    ).reset_index()
    summary = summary.merge(elapsed, on=["scenario", "configuration"], how="left")
    preferred = [
        "scenario",
        "configuration",
        "phase2_method",
        "marker_source",
        "marker_roles",
        "reference_contrast",
        "candidate_pruning",
        "overall_accuracy_mean",
        "coverage_mean",
        "assigned_accuracy_mean",
        "macro_f1_mean",
        "reciprocal_rank_mean",
        "brier_mean",
        "composition_mae_mean",
        "elapsed_median",
        "candidate_fraction_mean",
    ]
    return summary[[c for c in preferred if c in summary.columns] + [c for c in summary.columns if c not in preferred]]


def validation_metadata(args_dict, scenarios, configurations, seeds, repeat, sparse_mode):
    import anndata
    import scipy

    return {
        "easydecon_version": getattr(ed, "__version__", "unknown"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "anndata_version": anndata.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command_arguments": args_dict,
        "scenarios": list(scenarios),
        "configurations": [c.name if isinstance(c, ValidationConfiguration) else str(c) for c in configurations],
        "seeds": [int(seed) for seed in seeds],
        "repeat": int(repeat),
        "sparse": bool(sparse_mode),
    }


def plot_accuracy_by_scenario(summary_df):
    import matplotlib.pyplot as plt

    pivot = summary_df.pivot(index="scenario", columns="configuration", values="overall_accuracy_mean")
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(pivot.columns)), 4))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Overall accuracy")
    ax.set_xlabel("Scenario")
    ax.legend(title="Configuration", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    return fig, ax


def plot_coverage_vs_accuracy(metrics_df):
    import matplotlib.pyplot as plt

    ok = metrics_df[metrics_df["status"] == "ok"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(ok["coverage"], ok["assigned_accuracy"])
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Assigned accuracy")
    fig.tight_layout()
    return fig, ax


def plot_runtime_by_configuration(summary_df):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(summary_df)), 4))
    labels = summary_df["scenario"].astype(str) + "\n" + summary_df["configuration"].astype(str)
    ax.bar(np.arange(len(summary_df)), summary_df["elapsed_median"])
    ax.set_xticks(np.arange(len(summary_df)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel("Median elapsed seconds")
    fig.tight_layout()
    return fig, ax


def plot_candidate_reduction(metrics_df):
    import matplotlib.pyplot as plt

    ok = metrics_df[(metrics_df["status"] == "ok") & (metrics_df["candidate_pruning_enabled"] == True)]
    fig, ax = plt.subplots(figsize=(6, 4))
    if not ok.empty:
        labels = ok["scenario"].astype(str) + "\n" + ok["configuration"].astype(str)
        ax.bar(np.arange(len(ok)), ok["evaluated_pair_reduction"])
        ax.set_xticks(np.arange(len(ok)))
        ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel("Evaluated pair reduction")
    fig.tight_layout()
    return fig, ax

"""Simple hierarchical group refinement helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._schema import get_table
from ._validation import validate_choice
from .easydecon import (
    _validate_finite_nonnegative,
    assign_clusters_from_df,
    get_clusters_by_similarity_on_tissue,
    read_markers_dataframe,
)
from .extra import EasyDeconResult, _evidence_to_likelihood, easydecon_workflow
from .markers import resolve_phase_marker_tables


REFINEMENT_MODES = frozenset({"full", "phase2"})
PARENT_SOURCES = frozenset({"priors", "posterior"})


@dataclass
class RefinedGroupResult:
    parent_group: str
    mode: str
    parent_scores: pd.Series
    eligible_mask: pd.Series
    conditional_df: pd.DataFrame
    absolute_df: pd.DataFrame
    assigned_labels: pd.DataFrame
    phase2_result: pd.DataFrame
    child_result: EasyDeconResult | None
    diagnostics: dict


def _resolve_parent_scores(
    parent_result,
    parent_group,
    parent_source,
    spatial_index,
) -> pd.Series:
    validate_choice(parent_source, PARENT_SOURCES, "parent_source")
    if parent_source == "priors":
        matrix_name = "priors_df"
        matrix = getattr(parent_result, matrix_name, None)
    else:
        matrix_name = "posterior_df"
        matrix = getattr(parent_result, matrix_name, None)
        if matrix is None:
            raise ValueError(
                "parent_result.posterior_df is None. Use parent_source='priors' "
                "or run a workflow that produces posterior_df."
            )

    if not isinstance(matrix, pd.DataFrame):
        raise ValueError(f"parent_result must provide a pandas {matrix_name}.")

    if parent_group not in matrix.columns:
        available = ", ".join(map(str, matrix.columns))
        raise ValueError(
            f"parent_group={parent_group!r} was not found. "
            f"Available groups: {available}."
        )

    values = pd.to_numeric(matrix[parent_group], errors="coerce")
    values = values.reindex(spatial_index)
    values = values.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    values = values.clip(lower=0.0)
    values.name = str(parent_group)
    return values


def _pop_blocked_workflow_kwargs(workflow_kwargs):
    blocked = {"return_result_object", "return_diagnostics", "results_column"}
    duplicated = blocked.intersection(workflow_kwargs)
    if duplicated:
        raise ValueError(
            "Do not pass these workflow_kwargs to refine_group because they are "
            f"controlled internally: {sorted(duplicated)}."
        )


def _read_marker_kwargs(workflow_kwargs):
    mapping = {
        "marker_method": "marker_method",
        "groupby": "groupby",
        "sample_col": "sample_col",
        "marker_key": "key",
        "top_n_genes": "top_n_genes",
        "sort_by_column": "sort_by_column",
        "ascending": "ascending",
        "log2fc_min": "log2fc_min",
        "pval_cutoff": "pval_cutoff",
        "drop_ribosomal": "drop_ribosomal",
        "drop_mitochondrial": "drop_mitochondrial",
        "scanpy_method": "scanpy_method",
        "layer": "layer",
        "use_raw": "use_raw",
        "reference": "reference",
        "copy_adata": "copy_adata",
        "rank_genes_groups_kwargs": "rank_genes_groups_kwargs",
        "min_cells_per_group": "min_cells_per_group",
        "min_replicates_per_condition": "min_replicates_per_condition",
        "deseq_alpha": "deseq_alpha",
        "deseq_n_cpus": "deseq_n_cpus",
        "deseq_quiet": "deseq_quiet",
        "deseq_kwargs": "deseq_kwargs",
        "deseq_stats_kwargs": "deseq_stats_kwargs",
        "reference_min_cells": "reference_min_cells",
        "reference_min_mean": "reference_min_mean",
        "reference_min_log2fc": "reference_min_log2fc",
        "reference_min_detection": "reference_min_detection",
        "reference_min_detection_delta": "reference_min_detection_delta",
        "reference_pseudocount": "reference_pseudocount",
        "reference_contrast": "reference_contrast",
        "marker_roles": "marker_roles",
        "reference_presence_min_log2fc": "reference_presence_min_log2fc",
        "reference_presence_min_detection_delta": "reference_presence_min_detection_delta",
        "reference_negative_min_log2fc": "reference_negative_min_log2fc",
        "reference_negative_min_detection": "reference_negative_min_detection",
        "reference_negative_min_detection_delta": "reference_negative_min_detection_delta",
        "celltype": "celltype",
        "gene_id_column": "gene_id_column",
    }
    return {
        target: workflow_kwargs[source]
        for source, target in mapping.items()
        if source in workflow_kwargs
    }


def _phase2_kwargs(workflow_kwargs):
    keys = (
        "similarity_by_column",
        "lambda_param",
        "weight_column",
        "min_markers",
        "fallback_auc",
        "expression_threshold",
        "top_n_markers",
        "recovery_power",
        "drop_shared_markers",
        "center_auc",
        "ucell_max_rank",
        "ucell_negative_weight",
        "ucell_marker_role_column",
    )
    return {key: workflow_kwargs[key] for key in keys if key in workflow_kwargs}


def refine_group(
    sdata,
    parent_result,
    parent_group,
    markers_df=None,
    prepared_markers=None,
    filename=None,
    adata=None,
    mode="phase2",
    parent_source="priors",
    parent_threshold=0.0,
    results_column=None,
    bin_size=8,
    table_key=None,
    preferred_table_keys=None,
    marker_roles="shared",
    reference_presence_min_log2fc: float = 0.5,
    reference_presence_min_detection_delta: float = 0.0,
    reference_negative_min_log2fc: float = 1.0,
    reference_negative_min_detection: float = 0.10,
    reference_negative_min_detection_delta: float = 0.05,
    evidence_to_likelihood="softmax",
    softmax_tau=1.0,
    assign_method="max",
    allow_multiple=False,
    fold_change_threshold=2.0,
    minimum_evidence=0.0,
    tie_tolerance=1e-12,
    verbose=True,
    **workflow_kwargs,
) -> RefinedGroupResult:
    """Refine one broad parent group into marker-defined subclusters."""
    validate_choice(mode, REFINEMENT_MODES, "mode")
    validate_choice(parent_source, PARENT_SOURCES, "parent_source")
    validate_choice(evidence_to_likelihood, {"row_normalize", "softmax"}, "evidence_to_likelihood")
    if parent_threshold < 0:
        raise ValueError("parent_threshold must be non-negative.")
    _validate_finite_nonnegative(minimum_evidence, "minimum_evidence")
    _validate_finite_nonnegative(tie_tolerance, "tie_tolerance")
    _pop_blocked_workflow_kwargs(workflow_kwargs)
    workflow_kwargs.setdefault("marker_roles", marker_roles)
    workflow_kwargs.setdefault(
        "reference_presence_min_log2fc", reference_presence_min_log2fc
    )
    workflow_kwargs.setdefault(
        "reference_presence_min_detection_delta",
        reference_presence_min_detection_delta,
    )
    workflow_kwargs.setdefault(
        "reference_negative_min_log2fc", reference_negative_min_log2fc
    )
    workflow_kwargs.setdefault(
        "reference_negative_min_detection", reference_negative_min_detection
    )
    workflow_kwargs.setdefault(
        "reference_negative_min_detection_delta",
        reference_negative_min_detection_delta,
    )

    table = get_table(
        sdata,
        bin_size=bin_size,
        table_key=table_key,
        preferred_table_keys=preferred_table_keys,
    )
    parent_scores = _resolve_parent_scores(
        parent_result,
        parent_group,
        parent_source,
        table.obs.index,
    )
    eligible_mask = (parent_scores > parent_threshold).astype(bool)
    if not bool(eligible_mask.any()):
        raise ValueError(
            f"No spatial locations passed parent_group={parent_group!r} "
            f"with parent_threshold={parent_threshold}."
        )

    child_table = table[eligible_mask.to_numpy()].copy()
    child_results_column = results_column or f"{parent_group}_subcluster"

    if mode == "full":
        child_result = easydecon_workflow(
            sdata=child_table,
            markers_df=markers_df,
            prepared_markers=prepared_markers,
            filename=filename,
            adata=adata,
            bin_size=bin_size,
            return_result_object=True,
            results_column=child_results_column,
            evidence_to_likelihood=evidence_to_likelihood,
            softmax_tau=softmax_tau,
            assign_method=assign_method,
            allow_multiple=allow_multiple,
            fold_change_threshold=fold_change_threshold,
            minimum_evidence=minimum_evidence,
            tie_tolerance=tie_tolerance,
            verbose=verbose,
            **workflow_kwargs,
        )
        if child_result.posterior_df is None:
            raise ValueError(
                "Full refinement expected child_result.posterior_df, but it "
                "was None. Avoid list-style marker_genes for full refinement."
            )
        conditional_child = child_result.posterior_df
        phase2_child = child_result.phase2_result
        marker_diagnostics = child_result.diagnostics.get("markers")

    else:
        child_result = None
        read_kwargs = _read_marker_kwargs(workflow_kwargs)
        if read_kwargs.get("marker_roles") == "phase_specific":
            read_kwargs["top_n_genes"] = None
        child_markers, marker_diagnostics = read_markers_dataframe(
            child_table,
            markers_df=markers_df,
            prepared_markers=prepared_markers,
            filename=filename,
            adata=adata,
            bin_size=bin_size,
            return_diagnostics=True,
            verbose=verbose,
            **read_kwargs,
        )
        _, phase2_markers, marker_role_diagnostics = resolve_phase_marker_tables(
            child_markers,
            marker_roles=workflow_kwargs.get("marker_roles", "shared"),
            method=workflow_kwargs.get("method", "wjaccard"),
            marker_role_column=workflow_kwargs.get(
                "ucell_marker_role_column", "marker_role"
            ),
            top_n_genes=(
                workflow_kwargs.get("top_n_genes")
                if workflow_kwargs.get("marker_roles", "shared") == "phase_specific"
                else None
            ),
            require_phase1=False,
        )
        phase2_child = get_clusters_by_similarity_on_tissue(
            child_table,
            phase2_markers,
            common_group_name=None,
            bin_size=bin_size,
            gene_id_column="names",
            celltype="group",
            method=workflow_kwargs.get("method", "wjaccard"),
            add_to_obs=False,
            verbose=verbose,
            **_phase2_kwargs(workflow_kwargs),
        )
        conditional_child = _evidence_to_likelihood(
            phase2_child,
            method=evidence_to_likelihood,
            softmax_tau=softmax_tau,
        )

    conditional_df = conditional_child.reindex(table.obs.index, fill_value=0.0)
    phase2_result = phase2_child.reindex(table.obs.index, fill_value=0.0)
    absolute_df = conditional_df.mul(parent_scores, axis=0)

    assigned_labels = assign_clusters_from_df(
        table,
        absolute_df,
        bin_size=bin_size,
        results_column=child_results_column,
        method=assign_method,
        allow_multiple=allow_multiple,
        fold_change_threshold=fold_change_threshold,
        minimum_evidence=minimum_evidence,
        tie_tolerance=tie_tolerance,
        add_to_obs=True,
        verbose=verbose,
    )

    diagnostics = {
        "mode": mode,
        "parent_group": parent_group,
        "parent_source": parent_source,
        "parent_threshold": parent_threshold,
        "n_spatial_locations": int(table.n_obs),
        "n_eligible_locations": int(eligible_mask.sum()),
        "eligible_fraction": float(eligible_mask.mean()),
        "n_subclusters": int(conditional_df.shape[1]),
        "results_column": child_results_column,
        "child_phase1_ran": mode == "full",
        "child_phase2_ran": True,
        "marker_diagnostics": marker_diagnostics,
        "marker_roles": (
            marker_role_diagnostics
            if mode == "phase2"
            else child_result.diagnostics.get("marker_roles")
        ),
        "phase2_roles": (
            marker_role_diagnostics.get("phase2_roles")
            if mode == "phase2"
            else child_result.diagnostics.get("marker_roles", {}).get("phase2_roles")
        ),
        "phase2_marker_counts_by_group": (
            marker_role_diagnostics.get("phase2_marker_counts_by_group")
            if mode == "phase2"
            else child_result.diagnostics.get("marker_roles", {}).get(
                "phase2_marker_counts_by_group"
            )
        ),
        "phase2_method": workflow_kwargs.get("method", "wjaccard"),
        "minimum_evidence": minimum_evidence,
        "tie_tolerance": tie_tolerance,
    }

    return RefinedGroupResult(
        parent_group=str(parent_group),
        mode=mode,
        parent_scores=parent_scores,
        eligible_mask=eligible_mask,
        conditional_df=conditional_df,
        absolute_df=absolute_df,
        assigned_labels=assigned_labels,
        phase2_result=phase2_result,
        child_result=child_result,
        diagnostics=diagnostics,
    )


__all__ = ["RefinedGroupResult", "refine_group"]

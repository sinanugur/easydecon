"""Notebook-friendly summaries for easydecon marker tables and results."""

import pandas as pd

from ._schema import get_table


def _safe_numeric_df(df):
    if not isinstance(df, pd.DataFrame):
        return None
    return df.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _shape_or_none(df):
    if not isinstance(df, pd.DataFrame):
        return None
    return (int(df.shape[0]), int(df.shape[1]))


def _nonzero_row_count(df):
    numeric = _safe_numeric_df(df)
    if numeric is None:
        return None
    return int(numeric.ne(0).any(axis=1).sum())


def _zero_row_count(df):
    numeric = _safe_numeric_df(df)
    if numeric is None:
        return None
    return int(numeric.eq(0).all(axis=1).sum())


def _row_sum_stats(df):
    numeric = _safe_numeric_df(df)
    if numeric is None or numeric.shape[0] == 0:
        return {"row_sum_min": None, "row_sum_median": None, "row_sum_max": None}
    row_sums = numeric.sum(axis=1)
    return {
        "row_sum_min": float(row_sums.min()),
        "row_sum_median": float(row_sums.median()),
        "row_sum_max": float(row_sums.max()),
    }


def _max_value_stats(df):
    numeric = _safe_numeric_df(df)
    if numeric is None or numeric.shape[0] == 0 or numeric.shape[1] == 0:
        return {"max_value_median": None}
    return {"max_value_median": float(numeric.max(axis=1).median())}


def summarize_marker_table(
    markers_df,
    group_col: str = "group",
    gene_col: str = "names",
    source_col: str = "marker_source",
    top_genes: int = 5,
) -> pd.DataFrame:
    """Return one compact marker-count row per cell type or marker group."""
    if not isinstance(markers_df, pd.DataFrame):
        raise TypeError("markers_df must be a pandas DataFrame.")
    missing = [column for column in (group_col, gene_col) if column not in markers_df]
    if missing:
        raise ValueError(
            f"markers_df is missing required columns: {', '.join(missing)}."
        )

    rows = []
    grouped = markers_df.groupby(markers_df[group_col], sort=False, dropna=False)
    for group, group_df in grouped:
        genes = group_df[gene_col].dropna().astype(str)
        sources = None
        if source_col in group_df.columns:
            unique_sources = (
                group_df[source_col].dropna().astype(str).drop_duplicates().tolist()
            )
            sources = ", ".join(unique_sources) if unique_sources else None
        row = {
            "group": group,
            "n_markers": int(group_df.shape[0]),
            "n_unique_genes": int(genes.nunique()),
            "top_genes": ", ".join(genes.iloc[: max(0, top_genes)].tolist()),
            "marker_sources": sources,
        }
        if "marker_role" in group_df.columns:
            role_counts = group_df["marker_role"].dropna().astype(str).value_counts()
            row.update(
                {
                    "marker_roles": ", ".join(role_counts.index.tolist()),
                    "n_presence": int(role_counts.get("presence", 0)),
                    "n_identity": int(role_counts.get("identity", 0)),
                    "n_positive": int(role_counts.get("positive", 0)),
                    "n_negative": int(role_counts.get("negative", 0)),
                }
            )
        rows.append(row)
    columns = [
        "group",
        "n_markers",
        "n_unique_genes",
        "top_genes",
        "marker_sources",
    ]
    if any("marker_roles" in row for row in rows):
        columns.extend(
            ["marker_roles", "n_presence", "n_identity", "n_positive", "n_negative"]
        )
    return pd.DataFrame(rows, columns=columns)


def _matrix_summary(df):
    shape = _shape_or_none(df)
    summary = {
        "shape": shape,
        "n_rows": shape[0] if shape is not None else None,
        "n_columns": shape[1] if shape is not None else None,
        "nonzero_rows": _nonzero_row_count(df),
        "zero_rows": _zero_row_count(df),
    }
    summary.update(_row_sum_stats(df))
    summary.update(_max_value_stats(df))
    return summary


def _plain_label_counts(series):
    counts = series.dropna().value_counts()
    return {
        label.item() if hasattr(label, "item") else label: int(count)
        for label, count in counts.items()
    }


def _diagnostic_int(diagnostics, key, fallback):
    value = diagnostics.get(key)
    return int(fallback if value is None else value)


def _overlap_count(spatial_index, df):
    if not isinstance(df, pd.DataFrame):
        return None
    return int(spatial_index.isin(df.index).sum())


def _flatten_summary(summary):
    rows = []

    def visit(section, value, prefix=""):
        if isinstance(value, dict):
            for key, child in value.items():
                metric = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(child, dict) and key != "label_counts":
                    visit(section, child, metric)
                else:
                    rows.append({"section": section, "metric": metric, "value": child})
        else:
            rows.append({"section": section, "metric": prefix or "value", "value": value})

    for section, value in summary.items():
        visit(section, value)
    return pd.DataFrame(rows, columns=["section", "metric", "value"])


def summarize_easydecon_result(
    result,
    sdata=None,
    bin_size: int = 8,
    table_key=None,
    preferred_table_keys=None,
    assignment_column=None,
    as_dataframe: bool = True,
):
    """Summarize marker, matrix, posterior, assignment, and alignment QC."""
    required_attributes = (
        "markers_df",
        "phase1_result",
        "phase2_result",
        "assigned_labels",
        "priors_df",
        "likelihoods_df",
        "assignment_df",
        "diagnostics",
    )
    if any(not hasattr(result, attribute) for attribute in required_attributes):
        raise TypeError(
            "result must be an EasyDeconResult-like object returned by "
            "easydecon_workflow(..., return_result_object=True)."
        )

    markers_df = result.markers_df
    marker_table = summarize_marker_table(markers_df)
    marker_counts = marker_table["n_markers"]
    diagnostics = result.diagnostics if isinstance(result.diagnostics, dict) else {}
    marker_diagnostics = diagnostics.get("markers", {})
    if not isinstance(marker_diagnostics, dict):
        marker_diagnostics = {}

    groups = markers_df["group"].dropna().drop_duplicates().tolist()
    markers_summary = {
        "n_markers": int(markers_df.shape[0]),
        "n_celltypes": int(markers_df["group"].nunique()),
        "groups": groups,
        "min_markers_per_celltype": (
            int(marker_counts.min()) if not marker_counts.empty else None
        ),
        "median_markers_per_celltype": (
            float(marker_counts.median()) if not marker_counts.empty else None
        ),
        "max_markers_per_celltype": (
            int(marker_counts.max()) if not marker_counts.empty else None
        ),
    }
    if "marker_source" in markers_df.columns:
        markers_summary["marker_sources"] = (
            markers_df["marker_source"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
    if marker_diagnostics.get("marker_method") is not None:
        markers_summary["marker_method"] = marker_diagnostics["marker_method"]
    if marker_diagnostics.get("source") is not None:
        markers_summary["marker_source"] = marker_diagnostics["source"]
    if marker_diagnostics.get("reference_contrast") is not None:
        markers_summary["reference_contrast"] = marker_diagnostics["reference_contrast"]
    if marker_diagnostics.get("input_kind") is not None:
        markers_summary["preparation_input_kind"] = marker_diagnostics["input_kind"]
    selection_diagnostics = marker_diagnostics.get("selection")
    if isinstance(selection_diagnostics, dict):
        markers_summary["n_spatial_selected_markers"] = selection_diagnostics.get(
            "n_selected_markers"
        )
    prepared = getattr(result, "prepared_markers", None)
    if prepared is not None:
        markers_summary["prepared_marker_method"] = getattr(
            prepared, "marker_method", None
        )
        markers_summary["prepared_marker_signature"] = getattr(
            prepared, "signature", None
        )
        raw_markers = getattr(prepared, "raw_markers_df", None)
        if isinstance(raw_markers, pd.DataFrame):
            markers_summary["n_raw_prepared_markers"] = int(raw_markers.shape[0])
    inference = marker_diagnostics.get("marker_role_inference")
    if isinstance(inference, dict) and (
        inference.get("requested") or inference.get("applied")
    ):
        markers_summary["marker_role_inference"] = {
            key: inference.get(key)
            for key in (
                "mode",
                "applied",
                "n_positive_inferred",
                "n_negative_inferred",
                "n_ambiguous_dropped",
                "n_score_sign_discordant",
            )
            if inference.get(key) is not None
        }

    posterior_df = getattr(result, "posterior_df", None)
    workflow_summary = {
        "posterior_available": diagnostics.get(
            "posterior_available", posterior_df is not None
        ),
        "assignment_matrix": diagnostics.get(
            "assignment_matrix",
            "posterior_df" if posterior_df is not None else "phase2_result",
        ),
        "results_column": diagnostics.get("results_column"),
        "mask_col": diagnostics.get("mask_col"),
        "n_phase1_spots": _diagnostic_int(
            diagnostics, "n_phase1_spots", result.phase1_result.shape[0]
        ),
        "n_phase1_celltypes": _diagnostic_int(
            diagnostics, "n_phase1_celltypes", result.phase1_result.shape[1]
        ),
        "n_phase2_spots": _diagnostic_int(
            diagnostics, "n_phase2_spots", result.phase2_result.shape[0]
        ),
        "n_phase2_celltypes": _diagnostic_int(
            diagnostics, "n_phase2_celltypes", result.phase2_result.shape[1]
        ),
    }
    phase2_diagnostics = diagnostics.get("phase2", {})
    if isinstance(phase2_diagnostics, dict) and phase2_diagnostics.get("method") == "ucell":
        workflow_summary["phase2"] = {
            key: phase2_diagnostics.get(key)
            for key in (
                "method",
                "ucell_max_rank",
                "ucell_negative_weight",
                "min_markers",
                "expression_threshold",
                "recovery_power",
                "drop_shared_markers",
                "n_informative_rows",
                "n_uninformative_rows",
            )
        }
    role_diagnostics = diagnostics.get("marker_roles", {})
    if isinstance(role_diagnostics, dict) and role_diagnostics:
        workflow_summary["marker_roles"] = {
            key: role_diagnostics.get(key)
            for key in (
                "mode",
                "phase1_n_markers",
                "phase2_n_markers",
                "combined_marker_counts_by_role",
                "phase1_roles",
                "phase2_roles",
            )
            if role_diagnostics.get(key) is not None
        }

    matrices = {
        "phase1_result": _matrix_summary(result.phase1_result),
        "phase2_result": _matrix_summary(result.phase2_result),
        "priors_df": _matrix_summary(result.priors_df),
        "likelihoods_df": _matrix_summary(result.likelihoods_df),
        "assignment_df": _matrix_summary(result.assignment_df),
    }
    if posterior_df is not None:
        matrices["posterior_df"] = _matrix_summary(posterior_df)

    posterior_summary = {
        "available": posterior_df is not None,
        "reason_if_missing": (
            None
            if posterior_df is not None
            else "marker_genes list-style mask workflow or no posterior generated"
        ),
    }
    if posterior_df is not None:
        numeric_posterior = _safe_numeric_df(posterior_df)
        row_max = numeric_posterior.max(axis=1)
        posterior_summary.update(
            {
                "n_nonzero_rows": _nonzero_row_count(posterior_df),
                "median_max_probability": (
                    float(row_max.median()) if not row_max.empty else None
                ),
                "n_high_confidence_0_5": int(row_max.ge(0.5).sum()),
                "n_high_confidence_0_8": int(row_max.ge(0.8).sum()),
                "n_high_confidence_0_9": int(row_max.ge(0.9).sum()),
            }
        )

    assigned_labels = result.assigned_labels
    if not isinstance(assigned_labels, pd.DataFrame):
        raise TypeError("result.assigned_labels must be a pandas DataFrame.")
    resolved_assignment_column = assignment_column
    if resolved_assignment_column is None:
        resolved_assignment_column = diagnostics.get("results_column")
    if resolved_assignment_column not in assigned_labels.columns:
        resolved_assignment_column = (
            assigned_labels.columns[0] if len(assigned_labels.columns) else None
        )
    if resolved_assignment_column is None:
        assignment_series = pd.Series(index=assigned_labels.index, dtype=object)
    else:
        assignment_series = assigned_labels[resolved_assignment_column]
    n_assigned = int(assignment_series.notna().sum())
    n_total = int(assignment_series.shape[0])
    assignments_summary = {
        "assignment_column": resolved_assignment_column,
        "n_assigned": n_assigned,
        "n_unassigned": int(n_total - n_assigned),
        "assigned_fraction": float(n_assigned / n_total) if n_total else 0.0,
        "label_counts": _plain_label_counts(assignment_series),
    }

    spatial_alignment = None
    if sdata is not None:
        table = get_table(
            sdata,
            bin_size=bin_size,
            table_key=table_key,
            preferred_table_keys=preferred_table_keys,
        )
        spatial_index = table.obs.index
        n_spatial = int(len(spatial_index))
        phase1_overlap = _overlap_count(spatial_index, result.phase1_result)
        phase2_overlap = _overlap_count(spatial_index, result.phase2_result)
        assignment_overlap = _overlap_count(spatial_index, result.assignment_df)
        spatial_alignment = {
            "n_spatial_spots": n_spatial,
            "n_markers_spots_overlap_phase1": phase1_overlap,
            "n_markers_spots_overlap_phase2": phase2_overlap,
            "n_markers_spots_overlap_assignment": assignment_overlap,
            "phase1_fraction_aligned": (
                float(phase1_overlap / n_spatial) if n_spatial else 0.0
            ),
            "phase2_fraction_aligned": (
                float(phase2_overlap / n_spatial) if n_spatial else 0.0
            ),
            "assignment_fraction_aligned": (
                float(assignment_overlap / n_spatial) if n_spatial else 0.0
            ),
        }
        if posterior_df is not None:
            posterior_overlap = _overlap_count(spatial_index, posterior_df)
            spatial_alignment.update(
                {
                    "n_markers_spots_overlap_posterior": posterior_overlap,
                    "posterior_fraction_aligned": (
                        float(posterior_overlap / n_spatial) if n_spatial else 0.0
                    ),
                }
            )

    summary = {
        "markers": markers_summary,
        "workflow": workflow_summary,
        "matrices": matrices,
        "posterior": posterior_summary,
        "assignments": assignments_summary,
        "spatial_alignment": spatial_alignment,
    }
    return _flatten_summary(summary) if as_dataframe else summary


__all__ = ["summarize_marker_table", "summarize_easydecon_result"]

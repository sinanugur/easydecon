"""Reusable marker preparation helpers.

This module separates expensive marker generation from spatial-gene-universe
selection. ``PreparedMarkers`` intentionally stores a standardized but
spatial-unfiltered marker table so the same generated markers can be reused for
multiple spatial datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Integral, Real

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix, diags, issparse

from ._schema import (
    MarkerSchema,
    normalize_marker_roles,
    resolve_marker_columns,
    standardize_marker_dataframe,
)
from ._validation import (
    MARKER_ROLE_INFERENCE_MODES,
    MARKER_ROLE_MODES,
    MARKER_METHODS,
    PYDESEQ2_MARKER_METHODS,
    REFERENCE_CONTRASTS,
    REFERENCE_MARKER_METHODS,
    validate_choice,
    validate_positive,
    validate_probability_range,
)


_RUNTIME_PARAMETER_KEYS = {
    "verbose",
    "n_cpus",
    "quiet",
    "copy_adata",
    "deseq_n_cpus",
    "deseq_quiet",
}


def _normalize_marker_method(marker_method) -> str:
    method = str(marker_method).casefold()
    validate_choice(method, MARKER_METHODS, "marker_method")
    if method in {"deseq2", "pseudobulk_deseq2"}:
        return "pydeseq2"
    if method == "rctd_like":
        return "reference"
    return method


def _normalize_marker_parameters(parameters) -> dict:
    """Return a deterministic JSON-serializable representation of parameters."""

    def normalize(value):
        if isinstance(value, dict):
            normalized = {}
            for key in sorted(value, key=lambda item: str(item)):
                key_text = str(key)
                if key_text in _RUNTIME_PARAMETER_KEYS:
                    continue
                normalized[key_text] = normalize(value[key])
            return normalized
        if isinstance(value, tuple):
            return [normalize(item) for item in value]
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, set):
            return [normalize(item) for item in sorted(value, key=repr)]
        if value is None or isinstance(value, (str, bool)):
            return value
        if isinstance(value, Integral) and not isinstance(value, bool):
            return int(value)
        if isinstance(value, Real) and not isinstance(value, bool):
            value_float = float(value)
            if math.isfinite(value_float):
                return value_float
            return repr(value)
        return repr(value)

    normalized = normalize(parameters or {})
    if not isinstance(normalized, dict):
        return {"value": normalized}
    return normalized


def _validate_reference_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError(f"{name} must be an integer greater than or equal to 1.")
    return int(value)


def _validate_reference_float_nonnegative(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{name} must be a finite number greater than or equal to 0.")
    return float(value)


def _validate_reference_probability(value, name):
    return validate_probability_range(value, name)


def _validate_optional_top_n(value, name="top_n_genes"):
    if value is None:
        return None
    return _validate_reference_integer(value, name)


def _validate_bool(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool.")
    return value


def infer_scanpy_signed_marker_roles(
    markers_df,
    *,
    schema=None,
    log2fc_min=0.25,
) -> tuple[pd.DataFrame, dict]:
    """Infer positive/negative roles from signed Scanpy-style marker rows."""
    if not isinstance(markers_df, pd.DataFrame):
        raise TypeError("markers_df must be a pandas DataFrame.")
    log2fc_min = _validate_reference_float_nonnegative(log2fc_min, "log2fc_min")
    work = markers_df.copy()
    roles, role_column = normalize_marker_roles(work)
    if roles is not None:
        if role_column != "marker_role":
            work = work.rename(columns={role_column: "marker_role"})
        work["marker_role"] = roles
        resolved = resolve_marker_columns(work, schema=schema or MarkerSchema())
        group_column = resolved.get("group", "group")
        diagnostics = {
            "mode": "scanpy_signed",
            "inference_applied": False,
            "existing_roles_preserved": True,
            "n_input_rows": int(markers_df.shape[0]),
            "n_output_rows": int(work.shape[0]),
            "n_positive_inferred": 0,
            "n_negative_inferred": 0,
            "n_ambiguous_dropped": 0,
            "n_nonfinite_logfoldchange": 0,
            "n_below_effect_threshold": 0,
            "n_score_sign_discordant": 0,
            "n_zero_score": 0,
            "groups_with_positive": (
                work.loc[
                    work["marker_role"].isin(["positive", "presence", "identity"]),
                    group_column,
                ]
                .drop_duplicates()
                .astype(str)
                .tolist()
                if group_column in work
                else []
            ),
            "groups_with_negative": (
                work.loc[work["marker_role"] == "negative", group_column]
                .drop_duplicates()
                .astype(str)
                .tolist()
                if group_column in work
                else []
            ),
            "groups_without_positive": [],
            "groups_without_negative": [],
            "log2fc_min": float(log2fc_min),
        }
        return work, diagnostics

    schema = schema or MarkerSchema()
    resolved = resolve_marker_columns(work, schema=schema)
    if "logfoldchanges" not in resolved:
        raise ValueError(
            "marker_role_inference='scanpy_signed' requires a signed "
            "logfoldchanges column. Roles are not inferred from scores alone."
        )
    missing = {"group", "names"}.difference(resolved)
    if missing:
        raise ValueError(
            "marker_role_inference='scanpy_signed' requires resolvable group "
            f"and gene columns. Missing: {sorted(missing)}."
        )

    lfc = pd.to_numeric(work[resolved["logfoldchanges"]], errors="coerce")
    finite_lfc = np.isfinite(lfc)
    positive = finite_lfc & (lfc > 0) & (lfc >= log2fc_min)
    negative = finite_lfc & (lfc < 0) & (lfc <= -log2fc_min)
    directional = positive | negative

    score_discordant = pd.Series(False, index=work.index)
    zero_score = pd.Series(False, index=work.index)
    if "scores" in resolved:
        score = pd.to_numeric(work[resolved["scores"]], errors="coerce")
        finite_score = np.isfinite(score)
        zero_score = finite_score & (score == 0)
        score_discordant = finite_score & (
            (positive & (score < 0)) | (negative & (score > 0))
        )
    keep = directional & ~score_discordant & ~zero_score
    retained = work.loc[keep].copy()
    retained["marker_role"] = np.where(positive.loc[keep], "positive", "negative")

    groups = work[resolved["group"]].dropna().astype(str).drop_duplicates().tolist()
    positive_groups = retained.loc[retained["marker_role"] == "positive", resolved["group"]].astype(str).drop_duplicates().tolist()
    negative_groups = retained.loc[retained["marker_role"] == "negative", resolved["group"]].astype(str).drop_duplicates().tolist()
    diagnostics = {
        "mode": "scanpy_signed",
        "inference_applied": True,
        "existing_roles_preserved": False,
        "n_input_rows": int(work.shape[0]),
        "n_output_rows": int(retained.shape[0]),
        "n_positive_inferred": int((retained["marker_role"] == "positive").sum()) if "marker_role" in retained else 0,
        "n_negative_inferred": int((retained["marker_role"] == "negative").sum()) if "marker_role" in retained else 0,
        "n_ambiguous_dropped": int((~keep).sum()),
        "n_nonfinite_logfoldchange": int((~finite_lfc).sum()),
        "n_below_effect_threshold": int((finite_lfc & ~directional).sum()),
        "n_score_sign_discordant": int(score_discordant.sum()),
        "n_zero_score": int(zero_score.sum()),
        "groups_with_positive": positive_groups,
        "groups_with_negative": negative_groups,
        "groups_without_positive": [group for group in groups if group not in set(positive_groups)],
        "groups_without_negative": [group for group in groups if group not in set(negative_groups)],
        "log2fc_min": float(log2fc_min),
    }
    return retained, diagnostics


def _validate_marker_role_inference_for_method(marker_role_inference, marker_method):
    validate_choice(
        marker_role_inference,
        MARKER_ROLE_INFERENCE_MODES,
        "marker_role_inference",
    )
    normalized_method = _normalize_marker_method(marker_method)
    if (
        marker_role_inference == "scanpy_signed"
        and normalized_method in REFERENCE_MARKER_METHODS | PYDESEQ2_MARKER_METHODS
    ):
        raise ValueError(
            "marker_role_inference='scanpy_signed' is intended for Scanpy-style "
            "signed marker results. It is not applied to reference-profile or "
            "PyDESeq2 marker generation."
        )
    return marker_role_inference


def _get_reference_expression_matrix(adata, layer):
    """Return a validated abundance matrix for reference-profile markers."""
    missing = [name for name in ("obs", "var_names", "X") if not hasattr(adata, name)]
    if missing:
        raise TypeError(
            "adata must be AnnData-like with obs, var_names, and X. "
            f"Missing: {', '.join(missing)}."
        )

    if isinstance(layer, str):
        if not hasattr(adata, "layers") or layer not in adata.layers:
            raise ValueError(f"layer={layer!r} was not found in adata.layers.")
        matrix = adata.layers[layer]
        source = f"layer:{layer}"
    elif layer is None:
        if hasattr(adata, "layers") and "counts" in adata.layers:
            matrix = adata.layers["counts"]
            source = "layer:counts"
        else:
            matrix = adata.X
            source = "X"
    else:
        raise ValueError("layer must be None or a string layer name.")

    expected_shape = (int(getattr(adata, "n_obs", 0)), int(getattr(adata, "n_vars", 0)))
    if tuple(getattr(matrix, "shape", ())) != expected_shape:
        raise ValueError(
            "Reference expression matrix shape does not match adata dimensions. "
            f"Expected {expected_shape}, got {getattr(matrix, 'shape', None)}."
        )

    if issparse(matrix):
        matrix = matrix.tocsr(copy=True)
        matrix.eliminate_zeros()
        values = np.asarray(matrix.data)
        if not np.isfinite(values).all():
            raise ValueError("Reference expression matrix contains NaN or infinite values.")
        if (values < 0).any():
            raise ValueError("Reference expression matrix contains negative values.")
    else:
        matrix = np.asarray(matrix)
        if not np.isfinite(matrix).all():
            raise ValueError("Reference expression matrix contains NaN or infinite values.")
        if (matrix < 0).any():
            raise ValueError("Reference expression matrix contains negative values.")

    return matrix, source


def _row_sums(matrix):
    if issparse(matrix):
        return np.asarray(matrix.sum(axis=1)).reshape(-1)
    return np.asarray(matrix.sum(axis=1)).reshape(-1)


def _group_mean(matrix):
    values = matrix.mean(axis=0)
    return np.asarray(values).reshape(-1)


def _group_detection(matrix):
    if issparse(matrix):
        matrix = matrix.tocsr(copy=False)
        detected = np.asarray(matrix.getnnz(axis=0), dtype=float)
    else:
        detected = np.count_nonzero(np.asarray(matrix) > 0, axis=0).astype(float)
    return detected / matrix.shape[0]


def compute_reference_profile_markers(
    adata,
    groupby,
    layer=None,
    min_cells_per_group=25,
    min_mean_expression=2e-4,
    min_log2fc=1.0,
    min_detection=0.10,
    min_detection_delta=0.05,
    contrast="max_other",
    top_n_genes=None,
    pseudocount=1e-9,
    drop_ribosomal=False,
    drop_mitochondrial=False,
    marker_roles: str = "shared",
    reference_presence_min_log2fc: float = 0.5,
    reference_presence_min_detection_delta: float = 0.0,
    reference_negative_min_log2fc: float = 1.0,
    reference_negative_min_detection: float = 0.10,
    reference_negative_min_detection_delta: float = 0.05,
) -> tuple[pd.DataFrame, dict]:
    """Select marker genes from library-size-normalized reference profiles."""
    if groupby is None:
        raise ValueError("groupby is required for reference-profile markers.")
    if not hasattr(adata, "obs") or groupby not in adata.obs.columns:
        raise ValueError(f"groupby={groupby!r} was not found in adata.obs.columns.")
    if not getattr(adata, "var_names", pd.Index([])).is_unique:
        raise ValueError("adata.var_names must be unique for reference-profile markers.")

    min_cells_per_group = _validate_reference_integer(
        min_cells_per_group, "min_cells_per_group"
    )
    min_mean_expression = validate_probability_range(
        min_mean_expression, "min_mean_expression"
    )
    min_log2fc = _validate_reference_float_nonnegative(min_log2fc, "min_log2fc")
    min_detection = validate_probability_range(min_detection, "min_detection")
    min_detection_delta = validate_probability_range(
        min_detection_delta, "min_detection_delta"
    )
    validate_choice(contrast, REFERENCE_CONTRASTS, "contrast")
    validate_choice(marker_roles, MARKER_ROLE_MODES, "marker_roles")
    reference_presence_min_log2fc = _validate_reference_float_nonnegative(
        reference_presence_min_log2fc, "reference_presence_min_log2fc"
    )
    reference_presence_min_detection_delta = _validate_reference_probability(
        reference_presence_min_detection_delta,
        "reference_presence_min_detection_delta",
    )
    reference_negative_min_log2fc = _validate_reference_float_nonnegative(
        reference_negative_min_log2fc, "reference_negative_min_log2fc"
    )
    reference_negative_min_detection = _validate_reference_probability(
        reference_negative_min_detection, "reference_negative_min_detection"
    )
    reference_negative_min_detection_delta = _validate_reference_probability(
        reference_negative_min_detection_delta,
        "reference_negative_min_detection_delta",
    )
    top_n_genes = _validate_optional_top_n(top_n_genes)
    pseudocount = validate_positive(pseudocount, "pseudocount")
    _validate_bool(drop_ribosomal, "drop_ribosomal")
    _validate_bool(drop_mitochondrial, "drop_mitochondrial")

    matrix, source = _get_reference_expression_matrix(adata, layer)
    n_input_cells = int(adata.n_obs)
    n_input_genes = int(adata.n_vars)
    group_values = adata.obs[groupby]
    missing_group_mask = group_values.isna().to_numpy()
    library_sizes = _row_sums(matrix)
    empty_mask = library_sizes <= 0
    keep_mask = ~missing_group_mask & ~empty_mask

    n_missing_group_cells_excluded = int(missing_group_mask.sum())
    n_empty_cells_excluded = int((empty_mask & ~missing_group_mask).sum())
    if not keep_mask.any():
        raise ValueError(
            "Reference-profile marker selection requires at least two groups with "
            f"at least {min_cells_per_group} non-empty cells each."
        )

    kept_groups = group_values.loc[keep_mask].astype(str)
    counts = kept_groups.value_counts().sort_index()
    groups_attempted = sorted(group_values.dropna().astype(str).unique().tolist())
    retained_groups = counts[counts >= min_cells_per_group].index.tolist()
    skipped_groups = {
        group: {
            "reason": "too_few_cells",
            "n_cells": int(counts.get(group, 0)),
            "min_cells_per_group": int(min_cells_per_group),
        }
        for group in counts.index
        if counts[group] < min_cells_per_group
    }
    if len(retained_groups) < 2:
        raise ValueError(
            "Reference-profile marker selection requires at least two groups with "
            f"at least {min_cells_per_group} non-empty cells each."
        )

    keep_positions = np.flatnonzero(keep_mask)
    retained_mask_within_kept = kept_groups.isin(retained_groups).to_numpy()
    retained_positions = keep_positions[retained_mask_within_kept]
    retained_group_values = kept_groups.iloc[
        np.flatnonzero(retained_mask_within_kept)
    ].to_numpy()
    retained_library_sizes = library_sizes[retained_positions].astype(float)
    retained_matrix = matrix[retained_positions, :]
    if issparse(retained_matrix):
        retained_matrix = retained_matrix.tocsr(copy=True)
        normalized = diags(1.0 / retained_library_sizes) @ retained_matrix.astype(float)
        normalized = normalized.tocsr()
    else:
        retained_matrix = np.asarray(retained_matrix)
        normalized = retained_matrix.astype(float) / retained_library_sizes[:, None]

    mean_profiles = []
    detection_profiles = []
    cell_counts_per_group = {}
    for group in retained_groups:
        mask = retained_group_values == group
        cell_counts_per_group[group] = int(mask.sum())
        mean_profiles.append(_group_mean(normalized[mask, :]))
        detection_profiles.append(_group_detection(retained_matrix[mask, :]))
    mean_profiles = np.vstack(mean_profiles)
    detection_profiles = np.vstack(detection_profiles)

    n_groups = len(retained_groups)
    n_celltypes_expressing_gene = (mean_profiles > 0).sum(axis=0).astype(int)
    shared_marker_weight = (
        np.log((n_groups + 1) / (n_celltypes_expressing_gene + 1)) + 1
    )
    gene_names = np.asarray([str(name) for name in adata.var_names])
    gene_upper = pd.Series(gene_names).str.upper()
    gene_allowed = np.ones(n_input_genes, dtype=bool)
    if drop_ribosomal:
        gene_allowed &= ~gene_upper.str.startswith(("RPS", "RPL")).to_numpy()
    if drop_mitochondrial:
        gene_allowed &= ~gene_upper.str.startswith("MT-").to_numpy()

    rows = []
    groups_without_markers = []
    markers_per_group = {}
    presence_markers_per_group = {}
    identity_markers_per_group = {}
    negative_markers_per_group = {}
    role_order = {"presence": 0, "identity": 1, "negative": 2}
    for group_index, group in enumerate(retained_groups):
        other_mask = np.arange(n_groups) != group_index
        mean_target = mean_profiles[group_index]
        detection_target = detection_profiles[group_index]
        mean_other = mean_profiles[other_mask].mean(axis=0)
        max_other = mean_profiles[other_mask].max(axis=0)
        detection_other_max = detection_profiles[other_mask].max(axis=0)
        log2fc_mean = np.log2((mean_target + pseudocount) / (mean_other + pseudocount))
        log2fc_max = np.log2((mean_target + pseudocount) / (max_other + pseudocount))
        group_rows = []

        if marker_roles == "shared":
            selected_log2fc = log2fc_mean if contrast == "mean_other" else log2fc_max
            scores = (
                np.maximum(selected_log2fc, 0)
                * np.sqrt(detection_target)
                * shared_marker_weight
            )
            valid = (
                gene_allowed
                & np.isfinite(selected_log2fc)
                & np.isfinite(scores)
                & (scores >= 0)
                & (mean_target >= min_mean_expression)
                & (selected_log2fc >= min_log2fc)
                & (detection_target >= min_detection)
                & ((detection_target - detection_other_max) >= min_detection_delta)
            )
            selected_indices = np.flatnonzero(valid)
            for gene_index in selected_indices:
                group_rows.append(
                    {
                        "group": group,
                        "names": gene_names[gene_index],
                        "logfoldchanges": float(selected_log2fc[gene_index]),
                        "scores": float(scores[gene_index]),
                        "mean_target": float(mean_target[gene_index]),
                        "mean_other": float(mean_other[gene_index]),
                        "max_other": float(max_other[gene_index]),
                        "detection_target": float(detection_target[gene_index]),
                        "detection_other_max": float(detection_other_max[gene_index]),
                        "log2fc_mean": float(log2fc_mean[gene_index]),
                        "log2fc_max": float(log2fc_max[gene_index]),
                        "shared_marker_weight": float(shared_marker_weight[gene_index]),
                        "n_celltypes_expressing_gene": int(
                            n_celltypes_expressing_gene[gene_index]
                        ),
                        "marker_source": "reference_profile",
                    }
                )
            group_df = pd.DataFrame(group_rows)
            if not group_df.empty:
                group_df = group_df.sort_values(
                    ["scores", "logfoldchanges", "names"],
                    ascending=[False, False, True],
                    kind="stable",
                )
                if top_n_genes is not None:
                    group_df = group_df.head(top_n_genes)
                group_df["marker_rank"] = np.arange(1, len(group_df) + 1)
                rows.append(group_df)
                markers_per_group[group] = int(group_df.shape[0])
            else:
                groups_without_markers.append(group)
                markers_per_group[group] = 0
        else:
            presence_scores = (
                np.maximum(log2fc_mean, 0)
                * np.sqrt(detection_target)
                * shared_marker_weight
            )
            identity_scores = (
                np.maximum(log2fc_max, 0)
                * np.sqrt(detection_target)
                * shared_marker_weight
            )
            negative_log2fc = np.log2(
                (max_other + pseudocount) / (mean_target + pseudocount)
            )
            negative_scores = (
                np.maximum(negative_log2fc, 0)
                * np.sqrt(detection_other_max)
                * shared_marker_weight
            )
            role_specs = [
                (
                    "presence",
                    log2fc_mean,
                    presence_scores,
                    (
                        gene_allowed
                        & np.isfinite(log2fc_mean)
                        & np.isfinite(presence_scores)
                        & (presence_scores >= 0)
                        & (mean_target >= min_mean_expression)
                        & (log2fc_mean >= reference_presence_min_log2fc)
                        & (detection_target >= min_detection)
                        & (
                            (detection_target - detection_other_max)
                            >= reference_presence_min_detection_delta
                        )
                    ),
                    None,
                ),
                (
                    "identity",
                    log2fc_max,
                    identity_scores,
                    (
                        gene_allowed
                        & np.isfinite(log2fc_max)
                        & np.isfinite(identity_scores)
                        & (identity_scores >= 0)
                        & (mean_target >= min_mean_expression)
                        & (log2fc_max >= min_log2fc)
                        & (detection_target >= min_detection)
                        & ((detection_target - detection_other_max) >= min_detection_delta)
                    ),
                    None,
                ),
                (
                    "negative",
                    negative_log2fc,
                    negative_scores,
                    (
                        gene_allowed
                        & np.isfinite(negative_log2fc)
                        & np.isfinite(negative_scores)
                        & (negative_scores >= 0)
                        & (max_other >= min_mean_expression)
                        & (negative_log2fc >= reference_negative_min_log2fc)
                        & (detection_other_max >= reference_negative_min_detection)
                        & (
                            (detection_other_max - detection_target)
                            >= reference_negative_min_detection_delta
                        )
                    ),
                    negative_log2fc,
                ),
            ]
            for role, selected_log2fc, scores, valid, negative_values in role_specs:
                for gene_index in np.flatnonzero(valid):
                    group_rows.append(
                        {
                            "group": group,
                            "names": gene_names[gene_index],
                            "marker_role": role,
                            "logfoldchanges": float(selected_log2fc[gene_index]),
                            "scores": float(scores[gene_index]),
                            "mean_target": float(mean_target[gene_index]),
                            "mean_other": float(mean_other[gene_index]),
                            "max_other": float(max_other[gene_index]),
                            "detection_target": float(detection_target[gene_index]),
                            "detection_other_max": float(detection_other_max[gene_index]),
                            "log2fc_mean": float(log2fc_mean[gene_index]),
                            "log2fc_max": float(log2fc_max[gene_index]),
                            "negative_log2fc": (
                                float(negative_values[gene_index])
                                if negative_values is not None
                                else np.nan
                            ),
                            "shared_marker_weight": float(shared_marker_weight[gene_index]),
                            "n_celltypes_expressing_gene": int(
                                n_celltypes_expressing_gene[gene_index]
                            ),
                            "marker_source": "reference_profile",
                        }
                    )
            group_df = pd.DataFrame(group_rows)
            if not group_df.empty:
                group_df["_role_order"] = group_df["marker_role"].map(role_order)
                group_df = group_df.sort_values(
                    ["_role_order", "scores", "logfoldchanges", "names"],
                    ascending=[True, False, False, True],
                    kind="stable",
                ).drop(columns="_role_order")
                if top_n_genes is not None:
                    group_df = group_df.groupby(
                        ["group", "marker_role"], sort=False, group_keys=False
                    ).head(top_n_genes)
                group_df["marker_rank"] = (
                    group_df.groupby(["group", "marker_role"], sort=False).cumcount() + 1
                )
                rows.append(group_df)
            else:
                groups_without_markers.append(group)
            role_counts = (
                group_df["marker_role"].value_counts().to_dict()
                if not group_df.empty
                else {}
            )
            presence_markers_per_group[group] = int(role_counts.get("presence", 0))
            identity_markers_per_group[group] = int(role_counts.get("identity", 0))
            negative_markers_per_group[group] = int(role_counts.get("negative", 0))
            markers_per_group[group] = int(group_df.shape[0]) if not group_df.empty else 0

    if not rows:
        raise ValueError(
            "Reference-profile marker selection produced no markers. "
            "Consider relaxing reference thresholds."
        )
    markers = pd.concat(rows, ignore_index=True)
    groups_with_markers = markers["group"].drop_duplicates().tolist()
    diagnostics = {
        "method": "reference_profile",
        "groupby": groupby,
        "layer": source,
        "contrast": contrast,
        "groups_attempted": groups_attempted,
        "groups_retained_for_profiles": list(retained_groups),
        "groups_with_markers": groups_with_markers,
        "groups_skipped": skipped_groups,
        "groups_without_markers": groups_without_markers,
        "cell_counts_per_group": cell_counts_per_group,
        "markers_per_group": markers_per_group,
        "n_input_cells": n_input_cells,
        "n_cells_used": int(len(retained_positions)),
        "n_input_genes": n_input_genes,
        "n_selected_markers": int(markers.shape[0]),
        "n_missing_group_cells_excluded": n_missing_group_cells_excluded,
        "n_empty_cells_excluded": n_empty_cells_excluded,
        "thresholds": {
            "min_cells_per_group": int(min_cells_per_group),
            "min_mean_expression": float(min_mean_expression),
            "min_log2fc": float(min_log2fc),
            "min_detection": float(min_detection),
            "min_detection_delta": float(min_detection_delta),
            "pseudocount": float(pseudocount),
        },
    }
    if marker_roles == "phase_specific":
        diagnostics.update(
            {
                "presence_markers_per_group": presence_markers_per_group,
                "marker_roles_mode": marker_roles,
                "identity_markers_per_group": identity_markers_per_group,
                "negative_markers_per_group": negative_markers_per_group,
                "groups_without_presence_markers": [
                    group
                    for group in retained_groups
                    if presence_markers_per_group.get(group, 0) == 0
                ],
                "groups_without_identity_markers": [
                    group
                    for group in retained_groups
                    if identity_markers_per_group.get(group, 0) == 0
                ],
                "groups_without_negative_markers": [
                    group
                    for group in retained_groups
                    if negative_markers_per_group.get(group, 0) == 0
                ],
            }
        )
    return markers, diagnostics


def _adata_has_rank_genes_groups(adata, key):
    return hasattr(adata, "uns") and key in adata.uns


def _generate_scanpy_rank_genes_groups(
    adata,
    groupby,
    key,
    scanpy_method="wilcoxon",
    layer=None,
    use_raw=None,
    reference="rest",
    copy_adata=True,
    rank_genes_groups_kwargs=None,
):
    if groupby is None:
        raise ValueError(
            "groupby is required to generate markers from AnnData. Provide the "
            "obs column containing cell-type or cluster labels."
        )
    if not hasattr(adata, "obs") or groupby not in adata.obs.columns:
        raise ValueError(f"groupby={groupby!r} was not found in adata.obs.columns.")

    kwargs = dict(rank_genes_groups_kwargs or {})
    try:
        work_adata = adata.copy() if copy_adata else adata
        if str(work_adata.obs[groupby].dtype) != "category":
            work_adata.obs[groupby] = (
                work_adata.obs[groupby].astype(str).astype("category")
            )
        sc.tl.rank_genes_groups(
            work_adata,
            groupby=groupby,
            method=scanpy_method,
            key_added=key,
            layer=layer,
            use_raw=use_raw,
            reference=reference,
            **kwargs,
        )
    except Exception as exc:
        raise ValueError(
            "Could not generate markers with sc.tl.rank_genes_groups. Ensure "
            "adata contains normalized/log-transformed expression or provide a "
            "precomputed markers_df/filename."
        ) from exc
    return work_adata


def _pydeseq2_counts_error():
    return ValueError(
        "PyDESeq2 marker generation requires raw non-negative integer counts. "
        "Provide layer='counts' or another raw-count layer."
    )


def _get_adata_count_matrix(adata, layer="counts"):
    """Return a cells-by-genes raw-count DataFrame for pseudobulk analysis."""
    try:
        if layer is not None:
            if not hasattr(adata, "layers") or layer not in adata.layers:
                raise _pydeseq2_counts_error()
            matrix = adata.layers[layer]
        else:
            matrix = adata.X
    except ValueError:
        raise
    except Exception as exc:
        raise _pydeseq2_counts_error() from exc

    if matrix is None:
        raise _pydeseq2_counts_error()

    if issparse(matrix):
        values_to_check = np.asarray(matrix.data)
    else:
        values_to_check = np.asarray(matrix)

    try:
        finite = np.isfinite(values_to_check)
        if not finite.all():
            raise _pydeseq2_counts_error()
        if np.any(values_to_check < 0):
            raise _pydeseq2_counts_error()
        nonzero_values = values_to_check[values_to_check != 0]
        if nonzero_values.size and not np.allclose(
            nonzero_values, np.round(nonzero_values)
        ):
            raise _pydeseq2_counts_error()
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(
            "PyDESeq2 marker generation requires"
        ):
            raise
        raise _pydeseq2_counts_error() from exc

    dense_counts = matrix.toarray() if issparse(matrix) else np.asarray(matrix)
    return pd.DataFrame(
        dense_counts,
        index=adata.obs_names,
        columns=adata.var_names,
    )


def _build_one_vs_rest_pseudobulk(
    adata,
    target_group,
    groupby,
    sample_col,
    counts_df,
    min_cells_per_group=20,
    min_replicates_per_condition=2,
):
    """Aggregate raw counts by biological sample for one-vs-rest testing."""
    group_values = adata.obs[groupby].astype(str)
    sample_values = adata.obs[sample_col].astype(str)
    target_group = str(target_group)
    pseudobulk_rows = []
    metadata_rows = []

    for sample in sorted(sample_values.drop_duplicates().tolist()):
        sample_mask = sample_values == sample
        condition_masks = {
            "target": sample_mask & (group_values == target_group),
            "rest": sample_mask & (group_values != target_group),
        }
        for condition, mask in condition_masks.items():
            cell_ids = adata.obs_names[np.asarray(mask)]
            if len(cell_ids) < min_cells_per_group:
                continue
            row_name = f"{sample}__{target_group}__{condition}"
            summed = counts_df.loc[cell_ids].sum(axis=0)
            summed.name = row_name
            pseudobulk_rows.append(summed)
            metadata_rows.append((row_name, condition))

    if pseudobulk_rows:
        counts_pb = pd.DataFrame(pseudobulk_rows, columns=counts_df.columns)
        metadata_pb = pd.DataFrame(
            metadata_rows, columns=["pseudobulk_sample", "condition"]
        ).set_index("pseudobulk_sample")
        metadata_pb.index.name = counts_pb.index.name
    else:
        counts_pb = pd.DataFrame(columns=counts_df.columns)
        metadata_pb = pd.DataFrame(columns=["condition"])

    condition_counts = (
        metadata_pb["condition"].value_counts() if not metadata_pb.empty else {}
    )
    n_target = int(condition_counts.get("target", 0))
    n_rest = int(condition_counts.get("rest", 0))
    stats = {
        "n_target_replicates": n_target,
        "n_rest_replicates": n_rest,
        "skipped": False,
    }
    if (
        n_target < min_replicates_per_condition
        or n_rest < min_replicates_per_condition
    ):
        stats["skipped"] = True
        stats["reason"] = (
            "Insufficient pseudobulk replicates after cell-count filtering: "
            f"target={n_target}, rest={n_rest}, required="
            f"{min_replicates_per_condition} per condition."
        )
        return (
            pd.DataFrame(columns=counts_df.columns),
            pd.DataFrame(columns=["condition"]),
            stats,
        )

    return counts_pb, metadata_pb, stats


def _instantiate_pydeseq2_with_fallback(
    constructor,
    *args,
    quiet=True,
    n_cpus=None,
    **kwargs,
):
    attempts = [
        {"quiet": quiet, "n_cpus": n_cpus},
        {"n_cpus": n_cpus},
        {"quiet": quiet},
        {},
    ]
    last_error = None
    for optional_kwargs in attempts:
        call_kwargs = dict(kwargs)
        for name, value in optional_kwargs.items():
            if name not in call_kwargs:
                call_kwargs[name] = value
        try:
            return constructor(*args, **call_kwargs)
        except TypeError as exc:
            last_error = exc
    raise last_error


def _run_pydeseq2_one_vs_rest(
    counts_pb,
    metadata_pb,
    condition_col="condition",
    tested_level="target",
    reference_level="rest",
    alpha=0.05,
    n_cpus=None,
    quiet=True,
    deseq_kwargs=None,
    deseq_stats_kwargs=None,
):
    """Fit one PyDESeq2 target-vs-rest pseudobulk contrast."""
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError as exc:
        raise ImportError(
            "marker_method='pydeseq2' requires pydeseq2. Install it with "
            "`pip install pydeseq2`."
        ) from exc

    dds = _instantiate_pydeseq2_with_fallback(
        DeseqDataSet,
        counts=counts_pb,
        metadata=metadata_pb,
        design_factors=condition_col,
        quiet=quiet,
        n_cpus=n_cpus,
        **dict(deseq_kwargs or {}),
    )
    dds.deseq2()
    stat_res = _instantiate_pydeseq2_with_fallback(
        DeseqStats,
        dds,
        contrast=[condition_col, tested_level, reference_level],
        alpha=alpha,
        quiet=quiet,
        n_cpus=n_cpus,
        **dict(deseq_stats_kwargs or {}),
    )
    stat_res.summary()
    results_df = getattr(stat_res, "results_df", None)
    if results_df is None:
        raise ValueError("PyDESeq2 did not provide a results_df table.")
    return results_df


def compute_pseudobulk_deseq_markers(
    adata,
    groupby,
    sample_col,
    layer="counts",
    min_cells_per_group=20,
    min_replicates_per_condition=2,
    alpha=0.05,
    n_cpus=None,
    quiet=True,
    deseq_kwargs=None,
    deseq_stats_kwargs=None,
):
    """Generate one-vs-rest pseudobulk marker tables with PyDESeq2."""
    if groupby is None:
        raise ValueError("groupby is required for pseudobulk PyDESeq2 markers.")
    if not hasattr(adata, "obs") or groupby not in adata.obs.columns:
        raise ValueError(f"groupby={groupby!r} was not found in adata.obs.columns.")
    if sample_col is None:
        raise ValueError("sample_col is required for pseudobulk PyDESeq2 markers.")
    if sample_col not in adata.obs.columns:
        raise ValueError(f"sample_col={sample_col!r} was not found in adata.obs.columns.")

    counts_df = _get_adata_count_matrix(adata, layer=layer)
    groups = sorted(adata.obs[groupby].astype(str).unique().tolist())
    diagnostics = {
        "method": "pydeseq2_pseudobulk",
        "groupby": groupby,
        "sample_col": sample_col,
        "layer": layer,
        "groups_attempted": groups,
        "groups_completed": [],
        "groups_skipped": {},
        "min_cells_per_group": min_cells_per_group,
        "min_replicates_per_condition": min_replicates_per_condition,
    }
    marker_tables = []

    for target_group in groups:
        counts_pb, metadata_pb, group_stats = _build_one_vs_rest_pseudobulk(
            adata,
            target_group=target_group,
            groupby=groupby,
            sample_col=sample_col,
            counts_df=counts_df,
            min_cells_per_group=min_cells_per_group,
            min_replicates_per_condition=min_replicates_per_condition,
        )
        if group_stats["skipped"]:
            diagnostics["groups_skipped"][target_group] = group_stats
            continue

        results = _run_pydeseq2_one_vs_rest(
            counts_pb,
            metadata_pb,
            alpha=alpha,
            n_cpus=n_cpus,
            quiet=quiet,
            deseq_kwargs=deseq_kwargs,
            deseq_stats_kwargs=deseq_stats_kwargs,
        ).copy()
        required_results = {"log2FoldChange", "padj"}
        if not required_results.issubset(results.columns):
            missing = sorted(required_results.difference(results.columns))
            raise ValueError(
                f"PyDESeq2 results are missing required columns: {missing}."
            )
        results["group"] = target_group
        results["names"] = results.index.astype(str)
        results["logfoldchanges"] = results["log2FoldChange"]
        results["pvals_adj"] = results["padj"]
        if "stat" in results.columns:
            results["scores"] = pd.to_numeric(
                results["stat"], errors="coerce"
            ).abs()
        else:
            adjusted = pd.to_numeric(results["padj"], errors="coerce")
            results["scores"] = -np.log10(
                adjusted.clip(lower=np.finfo(float).tiny)
            )
        marker_tables.append(results.reset_index(drop=True))
        diagnostics["groups_completed"].append(target_group)

    if not marker_tables:
        raise ValueError(
            "No groups produced pseudobulk PyDESeq2 markers. Skipped group "
            f"diagnostics: {diagnostics['groups_skipped']}"
        )
    return pd.concat(marker_tables, ignore_index=True), diagnostics


def _obs_values_for_signature(adata, column):
    if column is None or not hasattr(adata, "obs") or column not in adata.obs:
        return None
    values = adata.obs[column]
    return [None if pd.isna(value) else str(value) for value in values.tolist()]


def _select_expression_matrix_for_signature(adata, parameters, marker_method=None):
    layer = parameters.get("layer")
    use_raw = parameters.get("use_raw")

    if layer is not None:
        if hasattr(adata, "layers") and layer in adata.layers:
            return str(layer), adata.layers[layer]
        return str(layer), None
    if marker_method == "reference":
        if hasattr(adata, "layers") and "counts" in adata.layers:
            return "layer:counts", adata.layers["counts"]
        return "X", getattr(adata, "X", None)
    if use_raw is True or (use_raw is None and getattr(adata, "raw", None) is not None):
        raw = getattr(adata, "raw", None)
        return "raw", getattr(raw, "X", None)
    return "X", getattr(adata, "X", None)


def _matrix_summary_for_signature(matrix):
    if matrix is None:
        return {"shape": None, "dtype": None}

    summary = {
        "shape": tuple(int(value) for value in getattr(matrix, "shape", ())),
        "dtype": str(getattr(matrix, "dtype", None)),
    }
    try:
        if issparse(matrix):
            summary["sum"] = float(np.asarray(matrix.sum()).reshape(-1)[0])
            summary["nnz"] = int(matrix.nnz)
        else:
            size = int(np.prod(summary["shape"])) if summary["shape"] else 0
            if size <= 10_000_000:
                array = np.asarray(matrix)
                summary["sum"] = float(np.sum(array))
                summary["nnz"] = int(np.count_nonzero(array))
    except Exception:
        # The signature is a practical stale-cache guard. If a particular matrix
        # object cannot cheaply summarize itself, keep the structural fields.
        pass
    return summary


def make_marker_signature(
    adata,
    marker_method,
    parameters,
) -> str:
    """Build a deterministic practical signature for generated markers."""
    normalized_parameters = _normalize_marker_parameters(parameters)
    normalized_method = _normalize_marker_method(marker_method)
    expression_source, matrix = _select_expression_matrix_for_signature(
        adata, normalized_parameters, normalized_method
    )
    payload = {
        "marker_method": normalized_method,
        "parameters": normalized_parameters,
        "n_obs": int(getattr(adata, "n_obs", 0)),
        "n_vars": int(getattr(adata, "n_vars", 0)),
        "var_names": [str(name) for name in getattr(adata, "var_names", [])],
        "groupby_values": _obs_values_for_signature(
            adata, normalized_parameters.get("groupby")
        ),
        "sample_col_values": _obs_values_for_signature(
            adata, normalized_parameters.get("sample_col")
        ),
        "expression_source": expression_source,
        "matrix": _matrix_summary_for_signature(matrix),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _marker_table_content_hash(markers_df) -> str:
    """Return a deterministic hash for marker table content."""
    if not isinstance(markers_df, pd.DataFrame):
        raise TypeError("markers_df must be a pandas DataFrame.")
    table = markers_df.reset_index(drop=True).copy()
    payload = {
        "columns": [str(column) for column in table.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in table.dtypes.items()},
        "row_count": int(table.shape[0]),
        "column_count": int(table.shape[1]),
        "hash_values": [
            int(value)
            for value in pd.util.hash_pandas_object(table, index=True).to_numpy()
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def make_marker_table_signature(
    markers_df,
    marker_method,
    parameters,
) -> str:
    """Build a deterministic signature for existing marker tables."""
    payload = {
        "marker_method": _normalize_marker_method(marker_method),
        "parameters": _normalize_marker_parameters(parameters),
        "table_content_hash": _marker_table_content_hash(markers_df),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _input_kind_from_sources(prepared_markers, markers_df, filename, adata):
    if prepared_markers is not None:
        return "prepared_markers"
    if markers_df is not None:
        return "dataframe"
    if filename is not None:
        return "file"
    if adata is not None:
        return "anndata"
    return None


def _ignored_marker_inputs(chosen_kind, prepared_markers, markers_df, filename, adata):
    priority = [
        ("prepared_markers", prepared_markers),
        ("dataframe", markers_df),
        ("file", filename),
        ("anndata", adata),
    ]
    seen_chosen = False
    ignored = []
    for kind, value in priority:
        if kind == chosen_kind:
            seen_chosen = True
            continue
        if seen_chosen and value is not None:
            ignored.append(kind)
    return ignored


@dataclass
class PreparedMarkers:
    raw_markers_df: pd.DataFrame
    marker_method: str
    source: str
    parameters: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    signature: str = ""

    def __post_init__(self):
        if not isinstance(self.raw_markers_df, pd.DataFrame):
            raise TypeError("raw_markers_df must be a pandas DataFrame.")
        resolved = resolve_marker_columns(self.raw_markers_df)
        missing = {"group", "names"}.difference(resolved)
        if missing:
            available = ", ".join(map(str, self.raw_markers_df.columns))
            raise ValueError(
                "PreparedMarkers.raw_markers_df must contain resolvable group "
                f"and gene columns. Missing: {sorted(missing)}. "
                f"Available columns: {available}."
            )
        self.raw_markers_df = self.raw_markers_df.copy(deep=True)
        self.marker_method = _normalize_marker_method(self.marker_method)
        self.parameters = _normalize_marker_parameters(self.parameters)
        self.diagnostics = dict(self.diagnostics or {})
        self.source = str(self.source)
        self.signature = str(self.signature)


def prepare_markers(
    adata=None,
    marker_method="auto",
    *,
    prepared_markers=None,
    markers_df=None,
    filename=None,
    source=None,
    celltype: str = "group",
    gene_id_column: str = "names",
    groupby=None,
    marker_key="rank_genes_groups",
    scanpy_method="wilcoxon",
    layer=None,
    use_raw=None,
    reference="rest",
    copy_adata=True,
    rank_genes_groups_kwargs=None,
    sample_col=None,
    min_cells_per_group=20,
    min_replicates_per_condition=2,
    deseq_alpha=0.05,
    deseq_n_cpus=None,
    deseq_quiet=True,
    deseq_kwargs=None,
    deseq_stats_kwargs=None,
    reference_min_cells: int = 25,
    reference_min_mean: float = 2e-4,
    reference_min_log2fc: float = 1.0,
    reference_min_detection: float = 0.10,
    reference_min_detection_delta: float = 0.05,
    reference_pseudocount: float = 1e-9,
    reference_contrast: str = "max_other",
    marker_roles: str = "shared",
    reference_presence_min_log2fc: float = 0.5,
    reference_presence_min_detection_delta: float = 0.0,
    reference_negative_min_log2fc: float = 1.0,
    reference_negative_min_detection: float = 0.10,
    reference_negative_min_detection_delta: float = 0.05,
    marker_role_inference: str = "none",
    marker_role_inference_log2fc_min: float = 0.25,
    verbose=True,
) -> PreparedMarkers:
    """Load, generate, or reuse a spatial-unfiltered marker preparation."""
    if prepared_markers is not None:
        if not isinstance(prepared_markers, PreparedMarkers):
            raise TypeError("prepared_markers must be a PreparedMarkers object.")
        return prepared_markers

    chosen_kind = _input_kind_from_sources(
        prepared_markers, markers_df, filename, adata
    )
    if chosen_kind is None:
        raise ValueError(
            "Please provide prepared_markers, markers_df, filename, or an adata object."
        )

    requested_method = _normalize_marker_method(marker_method)
    normalized_method = "existing" if chosen_kind in {"dataframe", "file"} else requested_method
    validate_choice(marker_roles, MARKER_ROLE_MODES, "marker_roles")
    _validate_marker_role_inference_for_method(
        marker_role_inference, normalized_method
    )
    if (
        chosen_kind == "anndata"
        and marker_roles == "phase_specific"
        and normalized_method not in REFERENCE_MARKER_METHODS
        and marker_role_inference != "scanpy_signed"
    ):
        raise ValueError(
            "Automatic phase-specific role generation is currently supported only for "
            "marker_method='reference'. Provide a marker table with marker_role for "
            "Scanpy or DESeq-derived markers."
        )
    if marker_role_inference == "scanpy_signed":
        marker_role_inference_log2fc_min = _validate_reference_float_nonnegative(
            marker_role_inference_log2fc_min,
            "marker_role_inference_log2fc_min",
        )
    else:
        marker_role_inference_log2fc_min = 0.25
    parameters = {
        "marker_method": normalized_method,
        "groupby": groupby,
        "marker_key": marker_key,
        "scanpy_method": scanpy_method,
        "layer": layer,
        "use_raw": use_raw,
        "reference": reference,
        "copy_adata": copy_adata,
        "rank_genes_groups_kwargs": dict(rank_genes_groups_kwargs or {}),
        "sample_col": sample_col,
        "min_cells_per_group": min_cells_per_group,
        "min_replicates_per_condition": min_replicates_per_condition,
        "deseq_alpha": deseq_alpha,
        "deseq_n_cpus": deseq_n_cpus,
        "deseq_quiet": deseq_quiet,
        "deseq_kwargs": dict(deseq_kwargs or {}),
        "deseq_stats_kwargs": dict(deseq_stats_kwargs or {}),
        "reference_min_cells": reference_min_cells,
        "reference_min_mean": reference_min_mean,
        "reference_min_log2fc": reference_min_log2fc,
        "reference_min_detection": reference_min_detection,
        "reference_min_detection_delta": reference_min_detection_delta,
        "reference_pseudocount": reference_pseudocount,
        "reference_contrast": reference_contrast,
        "marker_roles": marker_roles,
        "reference_presence_min_log2fc": reference_presence_min_log2fc,
        "reference_presence_min_detection_delta": reference_presence_min_detection_delta,
        "reference_negative_min_log2fc": reference_negative_min_log2fc,
        "reference_negative_min_detection": reference_negative_min_detection,
        "reference_negative_min_detection_delta": reference_negative_min_detection_delta,
        "marker_role_inference": marker_role_inference,
        "marker_role_inference_log2fc_min": marker_role_inference_log2fc_min,
        "celltype": celltype,
        "gene_id_column": gene_id_column,
        "verbose": verbose,
    }
    normalized_parameters = _normalize_marker_parameters(parameters)
    diagnostics = {
        "marker_method": normalized_method,
        "requested_marker_method": requested_method,
        "input_kind": chosen_kind,
        "ignored_input_kinds": _ignored_marker_inputs(
            chosen_kind, prepared_markers, markers_df, filename, adata
        ),
        "groupby": groupby,
        "generated_rank_genes_groups": False,
        "generated_pseudobulk_deseq": False,
        "generated_reference_profile": False,
        "pseudobulk_deseq": None,
        "reference_profile": None,
        "reference_contrast": None,
        "marker_role_inference": {
            "mode": marker_role_inference,
            "requested": marker_role_inference != "none",
            "applied": False,
            "existing_roles_preserved": False,
            "input_source": None,
        },
    }

    schema = MarkerSchema(group_col=celltype, gene_col=gene_id_column)
    table_signature = False

    if chosen_kind == "dataframe":
        if not isinstance(markers_df, pd.DataFrame):
            raise TypeError("markers_df must be a pandas DataFrame.")
        raw_df = markers_df
        resolved_source = source if source is not None else "dataframe"
        diagnostics["source"] = resolved_source
        table_signature = True

    elif chosen_kind == "file":
        try:
            raw_df = pd.read_csv(filename)
        except Exception:
            try:
                raw_df = pd.read_excel(filename)
            except Exception as exc:
                raise ValueError(
                    f"Could not read marker file {filename!r} as CSV or Excel."
                ) from exc
        resolved_source = source if source is not None else "file"
        diagnostics["source"] = resolved_source
        table_signature = True

    elif normalized_method in REFERENCE_MARKER_METHODS:
        raw_df, reference_diagnostics = compute_reference_profile_markers(
            adata,
            groupby=groupby,
            layer=layer,
            min_cells_per_group=reference_min_cells,
            min_mean_expression=reference_min_mean,
            min_log2fc=reference_min_log2fc,
            min_detection=reference_min_detection,
            min_detection_delta=reference_min_detection_delta,
            contrast=reference_contrast,
            top_n_genes=None,
            pseudocount=reference_pseudocount,
            drop_ribosomal=False,
            drop_mitochondrial=False,
            marker_roles=marker_roles,
            reference_presence_min_log2fc=reference_presence_min_log2fc,
            reference_presence_min_detection_delta=reference_presence_min_detection_delta,
            reference_negative_min_log2fc=reference_negative_min_log2fc,
            reference_negative_min_detection=reference_negative_min_detection,
            reference_negative_min_detection_delta=reference_negative_min_detection_delta,
        )
        diagnostics["generated_reference_profile"] = True
        diagnostics["reference_profile"] = reference_diagnostics
        diagnostics["reference_contrast"] = reference_contrast
        resolved_source = source if source is not None else "reference_profile"
        diagnostics["input_kind"] = "anndata_reference"
        diagnostics["source"] = resolved_source

    elif normalized_method in PYDESEQ2_MARKER_METHODS:
        raw_df, deseq_diagnostics = compute_pseudobulk_deseq_markers(
            adata,
            groupby=groupby,
            sample_col=sample_col,
            layer=layer,
            min_cells_per_group=min_cells_per_group,
            min_replicates_per_condition=min_replicates_per_condition,
            alpha=deseq_alpha,
            n_cpus=deseq_n_cpus,
            quiet=deseq_quiet,
            deseq_kwargs=deseq_kwargs,
            deseq_stats_kwargs=deseq_stats_kwargs,
        )
        diagnostics["generated_pseudobulk_deseq"] = True
        diagnostics["pseudobulk_deseq"] = deseq_diagnostics
        resolved_source = source if source is not None else "pydeseq2_pseudobulk"
        diagnostics["input_kind"] = "anndata_pydeseq2"
        diagnostics["source"] = resolved_source
    else:
        if _adata_has_rank_genes_groups(adata, marker_key):
            marker_adata = adata
            resolved_source = source if source is not None else f"adata.uns[{marker_key!r}]"
            diagnostics["input_kind"] = "anndata_existing_scanpy"
        else:
            if normalized_method == "existing":
                raise ValueError(
                    f"Could not read markers from adata.uns[{marker_key!r}]. Run "
                    "sc.tl.rank_genes_groups first, set marker_method='scanpy' "
                    "with groupby=..., or provide markers_df/filename."
                )
            marker_adata = _generate_scanpy_rank_genes_groups(
                adata,
                groupby=groupby,
                key=marker_key,
                scanpy_method=scanpy_method,
                layer=layer,
                use_raw=use_raw,
                reference=reference,
                copy_adata=copy_adata,
                rank_genes_groups_kwargs=rank_genes_groups_kwargs,
            )
            diagnostics["generated_rank_genes_groups"] = True
            resolved_source = source if source is not None else f"scanpy_generated[{marker_key!r}]"
            diagnostics["input_kind"] = "anndata_generated_scanpy"

        try:
            raw_df = sc.get.rank_genes_groups_df(
                marker_adata,
                group=None,
                key=marker_key,
                pval_cutoff=None,
                log2fc_min=None,
            )
        except Exception as exc:
            raise ValueError(
                f"Could not read markers from adata.uns[{marker_key!r}]. "
                "Run sc.tl.rank_genes_groups first or provide markers_df/filename."
            ) from exc
        diagnostics["source"] = resolved_source

    if marker_role_inference == "scanpy_signed":
        raw_df, role_inference_diagnostics = infer_scanpy_signed_marker_roles(
            raw_df,
            schema=schema,
            log2fc_min=marker_role_inference_log2fc_min,
        )
        diagnostics["marker_role_inference"] = {
            **role_inference_diagnostics,
            "requested": True,
            "applied": role_inference_diagnostics.get("inference_applied", False),
            "input_source": resolved_source,
        }
        if (
            role_inference_diagnostics.get("inference_applied")
            and marker_roles == "phase_specific"
        ):
            raise ValueError(
                "Signed Scanpy role inference creates positive and negative roles only. "
                "Use marker_roles='shared', provide a manually annotated marker table "
                "with presence/identity roles, or use marker_method='reference'."
            )

    standardized = standardize_marker_dataframe(
        raw_df,
        schema=schema,
        gene_universe=None,
        exclude_celltype=None,
        top_n_genes=None,
        sort_by_column=None,
        ascending=False,
        log2fc_min=-np.inf,
        pval_cutoff=1.0,
        drop_ribosomal=False,
        drop_mitochondrial=False,
        source=None,
    )
    if table_signature:
        table_content_hash = _marker_table_content_hash(standardized)
        signature = make_marker_table_signature(
            standardized,
            normalized_method,
            normalized_parameters,
        )
        diagnostics["table_content_hash"] = table_content_hash
    else:
        signature = make_marker_signature(
            adata,
            normalized_method,
            normalized_parameters,
        )
    diagnostics.update(
        {
            "source": resolved_source,
            "n_raw_markers": int(standardized.shape[0]),
            "n_celltypes": int(standardized["group"].nunique()),
            "signature": signature,
        }
    )
    if "marker_role" in standardized.columns:
        diagnostics["marker_role_counts"] = (
            standardized["marker_role"].value_counts().astype(int).to_dict()
        )
    if verbose:
        print(f"Prepared markers from {resolved_source}.")

    return PreparedMarkers(
        raw_markers_df=standardized,
        marker_method=normalized_method,
        source=resolved_source,
        parameters=normalized_parameters,
        diagnostics=diagnostics,
        signature=signature,
    )


def select_prepared_markers(
    prepared,
    gene_universe,
    exclude_celltype=None,
    top_n_genes=60,
    sort_by_column="scores",
    ascending=False,
    log2fc_min=0.25,
    pval_cutoff=0.05,
    drop_ribosomal=False,
    drop_mitochondrial=False,
    source=None,
    return_diagnostics=False,
) -> pd.DataFrame:
    """Select spatial-specific markers from a reusable marker preparation."""
    if not isinstance(prepared, PreparedMarkers):
        raise TypeError("prepared must be a PreparedMarkers object.")

    effective_sort_column = sort_by_column
    resolved_columns = resolve_marker_columns(prepared.raw_markers_df)
    if sort_by_column == "scores" and "scores" not in resolved_columns:
        effective_sort_column = None

    raw_df = prepared.raw_markers_df
    selected = standardize_marker_dataframe(
        raw_df,
        gene_universe=gene_universe,
        exclude_celltype=exclude_celltype,
        top_n_genes=top_n_genes,
        sort_by_column=effective_sort_column,
        ascending=ascending,
        log2fc_min=log2fc_min,
        pval_cutoff=pval_cutoff,
        drop_ribosomal=drop_ribosomal,
        drop_mitochondrial=drop_mitochondrial,
        source=prepared.source if source is None else source,
        copy=True,
    )
    if not return_diagnostics:
        return selected

    raw_genes = (
        raw_df["names"].astype(str)
        if "names" in raw_df.columns
        else pd.Series(dtype="object")
    )
    allowed_genes = {str(gene) for gene in gene_universe} if gene_universe is not None else None
    selected_names = set(selected["names"].astype(str)) if "names" in selected else set()
    diagnostics = {
        "n_raw_markers": int(raw_df.shape[0]),
        "n_selected_markers": int(selected.shape[0]),
        "n_raw_groups": int(raw_df["group"].nunique()) if "group" in raw_df else None,
        "n_selected_groups": int(selected["group"].nunique()) if "group" in selected else None,
        "n_spatial_genes": int(len(gene_universe)) if gene_universe is not None else None,
        "n_markers_removed_by_gene_universe": (
            int((~raw_genes.isin(allowed_genes)).sum())
            if allowed_genes is not None and not raw_genes.empty
            else 0
        ),
        "n_markers_removed_total": int(raw_df.shape[0] - selected.shape[0]),
        "marker_counts_per_group": (
            selected.groupby(selected["group"], sort=False).size().astype(int).to_dict()
            if "group" in selected
            else {}
        ),
        "marker_role_counts": (
            selected["marker_role"].value_counts().astype(int).to_dict()
            if "marker_role" in selected
            else {}
        ),
        "top_n_genes": top_n_genes,
        "sort_by_column": effective_sort_column,
        "ascending": bool(ascending),
        "log2fc_min": float(log2fc_min),
        "pval_cutoff": float(pval_cutoff),
        "drop_ribosomal": bool(drop_ribosomal),
        "drop_mitochondrial": bool(drop_mitochondrial),
        "source": prepared.source if source is None else source,
        "selected_genes": sorted(selected_names),
    }
    return selected, diagnostics


def _top_n_per_group(df, top_n_genes):
    if top_n_genes is None:
        return df.copy()
    top_n_genes = _validate_optional_top_n(top_n_genes)
    work = df.copy()
    pieces = []
    for _, group_df in work.groupby(work["group"], sort=False, group_keys=False):
        group_df = group_df.copy()
        if "marker_rank" in group_df.columns:
            group_df = group_df.sort_values("marker_rank", ascending=True, kind="stable")
        pieces.append(group_df.head(top_n_genes))
    work = pd.concat(pieces, ignore_index=False) if pieces else work.iloc[0:0].copy()
    work["marker_rank"] = work.groupby(work["group"], sort=False).cumcount() + 1
    return work


def _top_n_per_role(df, top_n_genes):
    if top_n_genes is None:
        return df.copy()
    top_n_genes = _validate_optional_top_n(top_n_genes)
    if "marker_role" not in df.columns:
        return _top_n_per_group(df, top_n_genes)
    work = df.copy()
    pieces = []
    for _, group_df in work.groupby(
        [work["group"], work["marker_role"]], sort=False, group_keys=False
    ):
        group_df = group_df.copy()
        if "marker_rank" in group_df.columns:
            group_df = group_df.sort_values("marker_rank", ascending=True, kind="stable")
        pieces.append(group_df.head(top_n_genes))
    work = pd.concat(pieces, ignore_index=False) if pieces else work.iloc[0:0].copy()
    work["marker_rank"] = (
        work.groupby([work["group"], work["marker_role"]], sort=False).cumcount() + 1
    )
    return work


def _marker_counts_by_group(df):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {}
    return df.groupby(df["group"], sort=False).size().astype(int).to_dict()


def resolve_phase_marker_tables(
    markers_df,
    marker_roles="shared",
    method="wjaccard",
    marker_role_column="marker_role",
    top_n_genes=None,
    *,
    require_phase1=True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Resolve marker subsets for Phase 1 and Phase 2 without mutating input."""
    validate_choice(marker_roles, MARKER_ROLE_MODES, "marker_roles")
    if not isinstance(markers_df, pd.DataFrame):
        raise TypeError("markers_df must be a pandas DataFrame.")
    missing = {"group", "names"}.difference(markers_df.columns)
    if missing:
        raise ValueError(
            "markers_df must contain canonical columns 'group' and 'names'. "
            f"Missing: {sorted(missing)}."
        )

    work = markers_df.copy()
    if work.index.name is not None and work.index.name in work.columns:
        work = work.reset_index(drop=True)
    roles, role_column = normalize_marker_roles(
        work,
        marker_role_column=marker_role_column,
    )
    has_roles = roles is not None
    if has_roles:
        if role_column != "marker_role":
            work.rename(columns={role_column: "marker_role"}, inplace=True)
        work["marker_role"] = roles

    if marker_roles == "shared" and not has_roles:
        phase1 = _top_n_per_group(work, top_n_genes)
        phase2 = _top_n_per_group(work, top_n_genes)
        diagnostics = {
            "mode": marker_roles,
            "role_column": marker_role_column,
            "phase1_roles": [],
            "phase2_roles": [],
            "combined_marker_counts_by_role": {},
            "phase1_marker_counts_by_group": _marker_counts_by_group(phase1),
            "phase2_marker_counts_by_group": _marker_counts_by_group(phase2),
            "phase1_n_markers": int(phase1.shape[0]),
            "phase2_n_markers": int(phase2.shape[0]),
        }
        return phase1, phase2, diagnostics

    if not has_roles:
        work["marker_role"] = "positive"

    if marker_roles == "shared":
        phase1_roles = ["positive", "presence", "identity"]
        phase2_roles = (
            ["positive", "identity", "negative"]
            if method == "ucell"
            else ["positive", "presence", "identity"]
        )
    else:
        phase1_roles = ["presence"]
        phase2_roles = (
            ["positive", "identity", "negative"]
            if method == "ucell"
            else ["positive", "identity"]
        )

    phase1 = work.loc[work["marker_role"].isin(phase1_roles)].copy()
    phase2 = work.loc[work["marker_role"].isin(phase2_roles)].copy()
    if top_n_genes is not None:
        phase1 = _top_n_per_role(phase1, top_n_genes)
        phase2 = _top_n_per_role(phase2, top_n_genes)

    if marker_roles == "phase_specific":
        if require_phase1 and phase1.empty:
            raise ValueError(
                "No Phase 1 presence markers remain after filtering. "
                "Relax reference presence thresholds or provide marker_genes."
            )
        identity_compatible = phase2["marker_role"].isin(["positive", "identity"])
        if phase2.loc[identity_compatible].empty:
            raise ValueError("No Phase 2 identity markers remain after filtering.")

    combined = pd.concat([phase1, phase2], ignore_index=False)
    if not combined.empty:
        combined = combined.drop_duplicates(
            subset=["group", "names", "marker_role"], keep="first"
        )
        role_order = {"positive": 0, "presence": 1, "identity": 2, "negative": 3}
        combined["_role_order"] = combined["marker_role"].map(role_order).fillna(99)
        combined = combined.sort_values(
            ["group", "_role_order", "marker_rank", "names"],
            ascending=[True, True, True, True],
            kind="stable",
        ).drop(columns="_role_order")
        combined.set_index("group", drop=False, inplace=True)

    diagnostics = {
        "mode": marker_roles,
        "role_column": marker_role_column,
        "phase1_roles": [
            role for role in phase1_roles if role in set(phase1.get("marker_role", []))
        ],
        "phase2_roles": [
            role for role in phase2_roles if role in set(phase2.get("marker_role", []))
        ],
        "combined_marker_counts_by_role": (
            combined["marker_role"].value_counts().astype(int).to_dict()
            if "marker_role" in combined
            else {}
        ),
        "phase1_marker_counts_by_group": _marker_counts_by_group(phase1),
        "phase2_marker_counts_by_group": _marker_counts_by_group(phase2),
        "phase1_n_markers": int(phase1.shape[0]),
        "phase2_n_markers": int(phase2.shape[0]),
    }
    return phase1, phase2, diagnostics


__all__ = [
    "PreparedMarkers",
    "compute_reference_profile_markers",
    "compute_pseudobulk_deseq_markers",
    "infer_scanpy_signed_marker_roles",
    "make_marker_table_signature",
    "prepare_markers",
    "select_prepared_markers",
    "resolve_phase_marker_tables",
    "make_marker_signature",
]

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
from scipy.sparse import issparse

from ._schema import resolve_marker_columns, standardize_marker_dataframe
from ._validation import MARKER_METHODS, PYDESEQ2_MARKER_METHODS, validate_choice


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


def _obs_values_for_signature(adata, column):
    if column is None or not hasattr(adata, "obs") or column not in adata.obs:
        return None
    values = adata.obs[column]
    return [None if pd.isna(value) else str(value) for value in values.tolist()]


def _select_expression_matrix_for_signature(adata, parameters):
    layer = parameters.get("layer")
    use_raw = parameters.get("use_raw")

    if layer is not None:
        if hasattr(adata, "layers") and layer in adata.layers:
            return str(layer), adata.layers[layer]
        return str(layer), None
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
    expression_source, matrix = _select_expression_matrix_for_signature(
        adata, normalized_parameters
    )
    payload = {
        "marker_method": _normalize_marker_method(marker_method),
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
    adata,
    marker_method="auto",
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
    verbose=True,
) -> PreparedMarkers:
    """Generate or extract reusable, spatial-unfiltered marker tables."""
    normalized_method = _normalize_marker_method(marker_method)
    parameters = {
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
        "verbose": verbose,
    }
    normalized_parameters = _normalize_marker_parameters(parameters)
    diagnostics = {
        "marker_method": normalized_method,
        "groupby": groupby,
        "generated_rank_genes_groups": False,
        "generated_pseudobulk_deseq": False,
    }

    if normalized_method in PYDESEQ2_MARKER_METHODS:
        from .easydecon import compute_pseudobulk_deseq_markers

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
        source = "pydeseq2_pseudobulk"
    else:
        from .easydecon import _adata_has_rank_genes_groups
        from .easydecon import _generate_scanpy_rank_genes_groups

        if _adata_has_rank_genes_groups(adata, marker_key):
            marker_adata = adata
            source = f"adata.uns[{marker_key!r}]"
        else:
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
            source = f"scanpy_generated[{marker_key!r}]"

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

    standardized = standardize_marker_dataframe(
        raw_df,
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
    signature = make_marker_signature(
        adata,
        normalized_method,
        normalized_parameters,
    )
    diagnostics.update(
        {
            "source": source,
            "n_raw_markers": int(standardized.shape[0]),
            "n_celltypes": int(standardized["group"].nunique()),
            "signature": signature,
        }
    )
    if verbose:
        print(f"Prepared markers from {source}.")

    return PreparedMarkers(
        raw_markers_df=standardized,
        marker_method=normalized_method,
        source=source,
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
) -> pd.DataFrame:
    """Select spatial-specific markers from a reusable marker preparation."""
    if not isinstance(prepared, PreparedMarkers):
        raise TypeError("prepared must be a PreparedMarkers object.")

    effective_sort_column = sort_by_column
    resolved_columns = resolve_marker_columns(prepared.raw_markers_df)
    if sort_by_column == "scores" and "scores" not in resolved_columns:
        effective_sort_column = None

    return standardize_marker_dataframe(
        prepared.raw_markers_df,
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


__all__ = [
    "PreparedMarkers",
    "prepare_markers",
    "select_prepared_markers",
    "make_marker_signature",
]

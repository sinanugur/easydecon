"""Shared schema helpers for marker tables and AnnData-like objects."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MarkerSchema:
    """Column names used by an easydecon marker DataFrame."""

    group_col: str = "group"
    gene_col: str = "names"
    lfc_col: str = "logfoldchanges"
    padj_col: str = "pvals_adj"
    score_col: str = "scores"


GROUP_ALIASES = [
    "group", "groups", "celltype", "cell_type", "cell type",
    "cluster", "clusters", "annotation", "label", "labels",
]

GENE_ALIASES = [
    "names", "name", "gene", "genes", "gene_id", "gene_ids",
    "gene_symbol", "symbol", "feature", "features",
]

LFC_ALIASES = [
    "logfoldchanges", "logfoldchange", "log2FoldChange",
    "log2foldchange", "log2fc", "avg_log2FC", "avg_log2fc",
    "avg_logFC", "lfc",
]

PADJ_ALIASES = [
    "pvals_adj", "pval_adj", "p_val_adj", "padj",
    "FDR", "fdr", "qval", "q_value", "qvalue",
]

SCORE_ALIASES = [
    "scores", "score", "stat", "wald_stat", "statistics",
    "baseMean", "basemean",
]


_CANONICAL_ALIASES = {
    "group": GROUP_ALIASES,
    "names": GENE_ALIASES,
    "logfoldchanges": LFC_ALIASES,
    "pvals_adj": PADJ_ALIASES,
    "scores": SCORE_ALIASES,
}


def _schema_columns(schema):
    schema = schema or MarkerSchema()
    return {
        "group": schema.group_col,
        "names": schema.gene_col,
        "logfoldchanges": schema.lfc_col,
        "pvals_adj": schema.padj_col,
        "scores": schema.score_col,
    }


def resolve_marker_columns(df, schema=None):
    """Map easydecon's canonical marker columns to columns present in ``df``.

    Matching is case-insensitive, while the returned values retain the exact
    column labels from the input DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    columns = list(df.columns)
    schema_columns = _schema_columns(schema)
    resolved = {}

    for canonical, aliases in _CANONICAL_ALIASES.items():
        # A canonical spelling has priority over every alias or schema hint.
        if canonical in columns:
            resolved[canonical] = canonical
            continue

        candidates = [schema_columns[canonical], canonical, *aliases]
        for candidate in candidates:
            candidate_folded = str(candidate).casefold()
            match = next(
                (
                    column
                    for column in columns
                    if str(column).casefold() == candidate_folded
                ),
                None,
            )
            if match is not None:
                resolved[canonical] = match
                break

    return resolved


def _resolve_sort_column(sort_by_column, df, resolved, schema):
    if sort_by_column is None:
        for canonical in ("scores", "logfoldchanges", "pvals_adj"):
            if canonical in df.columns:
                return canonical
        return None

    if sort_by_column in df.columns:
        return sort_by_column

    requested = str(sort_by_column).casefold()
    for column in df.columns:
        if str(column).casefold() == requested:
            return column

    for canonical, original in resolved.items():
        if str(original).casefold() == requested and canonical in df.columns:
            return canonical

    schema_columns = _schema_columns(schema)
    for canonical, aliases in _CANONICAL_ALIASES.items():
        candidates = [canonical, schema_columns[canonical], *aliases]
        if any(str(candidate).casefold() == requested for candidate in candidates):
            if canonical in df.columns:
                return canonical

    available = ", ".join(map(str, df.columns))
    raise ValueError(
        f"Could not resolve sort column {sort_by_column!r}. "
        f"Available columns: {available}."
    )


def _as_exclusion_set(values):
    if isinstance(values, str):
        return {values}
    return set(values)


def standardize_marker_dataframe(
    df,
    schema=None,
    gene_universe=None,
    exclude_celltype=None,
    top_n_genes=60,
    sort_by_column=None,
    ascending=False,
    log2fc_min=0.25,
    pval_cutoff=0.05,
    drop_ribosomal=False,
    drop_mitochondrial=False,
    source=None,
    require_group=True,
    require_gene=True,
    copy=True,
):
    """Return a consistently named and filtered marker DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    result = df.copy() if copy else df
    resolved = resolve_marker_columns(result, schema=schema)
    available = ", ".join(map(str, result.columns))

    if require_group and "group" not in resolved:
        raise ValueError(
            "Could not resolve a marker group column. "
            f"Available columns: {available}."
        )
    if require_gene and "names" not in resolved:
        raise ValueError(
            "Could not resolve a marker gene column. "
            f"Available columns: {available}."
        )

    rename_columns = {
        original: canonical
        for canonical, original in resolved.items()
        if original != canonical
    }
    result.rename(columns=rename_columns, inplace=True)

    required_present = [column for column in ("group", "names") if column in result]
    if required_present:
        result.dropna(subset=required_present, inplace=True)
    for column in required_present:
        result[column] = result[column].astype(str)

    if "logfoldchanges" in result.columns:
        lfc_values = pd.to_numeric(result["logfoldchanges"], errors="coerce")
        result = result.loc[lfc_values >= log2fc_min]
    if "pvals_adj" in result.columns:
        padj_values = pd.to_numeric(result["pvals_adj"], errors="coerce")
        result = result.loc[padj_values <= pval_cutoff]

    if "names" in result.columns:
        gene_names = result["names"].str.upper()
        if drop_ribosomal:
            result = result.loc[~gene_names.str.startswith(("RPS", "RPL"))]
            gene_names = result["names"].str.upper()
        if drop_mitochondrial:
            result = result.loc[~gene_names.str.startswith("MT-")]

        if gene_universe is not None:
            allowed_genes = {str(gene) for gene in gene_universe}
            result = result.loc[result["names"].isin(allowed_genes)]

    if exclude_celltype is not None and "group" in result.columns:
        excluded_groups = {str(group) for group in _as_exclusion_set(exclude_celltype)}
        result = result.loc[~result["group"].isin(excluded_groups)]

    sort_column = _resolve_sort_column(sort_by_column, result, resolved, schema)
    if sort_column is not None:
        sort_ascending = (
            True
            if sort_by_column is None and sort_column == "pvals_adj"
            else ascending
        )
        result = result.sort_values(sort_column, ascending=sort_ascending, kind="stable")

    if {"group", "names"}.issubset(result.columns):
        result = result.drop_duplicates(subset=["group", "names"], keep="first")
        if top_n_genes is not None:
            result = result.groupby(
                result["group"], sort=False, group_keys=False
            ).head(top_n_genes)
        result["marker_rank"] = (
            result.groupby(result["group"], sort=False).cumcount() + 1
        )
        result.set_index("group", drop=False, inplace=True)
    else:
        result["marker_rank"] = range(1, len(result) + 1)

    if source is not None:
        result["marker_source"] = source

    return result


def get_table(sdata, bin_size=8, table_key=None, preferred_table_keys=None):
    """Resolve and validate an AnnData-like table from a container or table."""
    if table_key is not None:
        try:
            table = sdata.tables[table_key]
        except (AttributeError, KeyError, TypeError) as error:
            raise KeyError(f"Table {table_key!r} was not found in sdata.tables.") from error
    else:
        tables = getattr(sdata, "tables", None)
        table = None
        table_found = False
        if tables is not None:
            if isinstance(preferred_table_keys, str):
                keys = [preferred_table_keys]
            else:
                keys = list(preferred_table_keys or [])
            keys.extend(["cell_segmentations", f"square_{bin_size:03}um", "table"])
            for key in dict.fromkeys(keys):
                try:
                    table = tables[key]
                    table_found = True
                    break
                except (KeyError, TypeError):
                    continue
        if not table_found:
            table = sdata

    missing = [name for name in ("obs", "var_names", "X") if not hasattr(table, name)]
    if missing:
        missing_text = ", ".join(missing)
        raise TypeError(
            "Resolved table is not AnnData-like; expected attributes obs, "
            f"var_names, and X. Missing: {missing_text}."
        )
    return table


__all__ = [
    "MarkerSchema",
    "GROUP_ALIASES",
    "GENE_ALIASES",
    "LFC_ALIASES",
    "PADJ_ALIASES",
    "SCORE_ALIASES",
    "resolve_marker_columns",
    "standardize_marker_dataframe",
    "get_table",
]

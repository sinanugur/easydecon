# Marker inputs and table schema

easydecon accepts markers through several routes. `prepare_markers` is the
shared source controller used by the workflow and by the backward-compatible
`read_markers_dataframe` wrapper. Inputs are resolved in this priority order:

1. `prepared_markers`
2. `markers_df`
3. `filename`
4. `adata`

If none of these is provided, marker preparation raises a `ValueError`.

## Input routes

`markers_df`
: A pandas DataFrame passed directly. It is copied before standardization and
  prepared as `marker_method="existing"`.

`filename`
: A CSV or Excel file read into a marker DataFrame and prepared as
  `marker_method="existing"`.

`adata` with existing Scanpy results
: If `adata.uns[marker_key]` exists, easydecon reads it with
  `scanpy.get.rank_genes_groups_df`.

Scanpy-generated markers
: With `marker_method="scanpy"` and `groupby=...`, easydecon runs
  `scanpy.tl.rank_genes_groups` when the marker key is missing.

pseudobulk PyDESeq2
: With `marker_method="pydeseq2"` and `sample_col=...`, easydecon builds
  one-vs-rest pseudobulk count tables and runs PyDESeq2.

reference-profile markers
: With `marker_method="reference"`, easydecon selects markers from normalized
  reference profiles without calculating p-values.

`PreparedMarkers`
: A reusable, spatial-unfiltered marker preparation produced by
  `ed.prepare_markers`.

Aliases `marker_method="deseq2"` and `"pseudobulk_deseq2"` normalize to
`"pydeseq2"`. Alias `marker_method="rctd_like"` normalizes to `"reference"` for
compatibility; examples use `"reference"`.

## Generated-marker workflow examples

Scanpy-generated markers:

```python
result = ed.run_easydecon(
    sdata=sdata,
    adata=sc_adata,
    groupby="cell_type",
    marker_method="scanpy",
    filtering_algorithm="permutation",
    method="wjaccard",
    return_result_object=True,
)
```

## Prepare once, select per spatial dataset

`PreparedMarkers` is independent of a spatial gene universe. Use
`prepare_markers` when you want to prepare markers once and reuse them, or let
`run_easydecon` do this automatically and retrieve `result.prepared_markers`.

```python
prepared = ed.prepare_markers(
    markers_df=deseq_df,
    source="deseq_table",
)

result = ed.run_easydecon(
    sdata,
    prepared_markers=prepared,
    return_result_object=True,
)

selected = ed.select_prepared_markers(
    prepared,
    gene_universe=sdata.tables["table"].var_names,
)
```

`read_markers_dataframe` remains supported when you want the selected marker
DataFrame directly for one spatial dataset; internally it delegates to
`prepare_markers` and `select_prepared_markers`.

PyDESeq2 markers:

```python
result = ed.run_easydecon(
    sdata=sdata,
    adata=sc_adata,
    groupby="cell_type",
    sample_col="donor",
    marker_method="pydeseq2",
    filtering_algorithm="permutation",
    method="wjaccard",
    return_result_object=True,
)
```

Reference-profile markers:

```python
result = ed.run_easydecon(
    sdata=sdata,
    adata=sc_adata,
    groupby="cell_type",
    marker_method="reference",
    filtering_algorithm="permutation",
    method="wjaccard",
    return_result_object=True,
)
```

## Canonical columns

Required after standardization:

`group`
: Marker group or cell type.

`names`
: Gene identifier.

Optional but commonly used:

`logfoldchanges`
: Effect-size or marker-weight column.

`pvals_adj`
: Adjusted p-value column.

`scores`
: Ranking or scoring column.

`marker_rank`
: Rank assigned within group, or within group and role.

`marker_source`
: Source label such as `dataframe`, `file`, `reference_profile`, or
  `pydeseq2_pseudobulk`.

`marker_role`
: One of `positive`, `negative`, `presence`, or `identity`.

`MarkerSchema` and `resolve_marker_columns` recognize common aliases such as
`cell_type`, `cluster`, `gene`, `gene_symbol`, `log2FoldChange`, `avg_log2FC`,
`padj`, and `score`. Matching is case-insensitive, but returned DataFrames use
canonical column names.

## Manual marker table

```python
import pandas as pd

markers_df = pd.DataFrame(
    {
        "group": ["T cell", "T cell", "B cell"],
        "names": ["CD3D", "CD3E", "MS4A1"],
        "logfoldchanges": [2.1, 1.8, 2.4],
        "pvals_adj": [0.001, 0.004, 0.001],
        "scores": [8.0, 7.5, 9.0],
    }
)
```

Gene identifiers in `names` must match `table.var_names` after any upstream
normalization. No marker genes remain if the table uses different identifiers,
for example Ensembl IDs in one table and gene symbols in the other.

## Top-N behavior

Direct `ed.read_markers_dataframe(...)` applies `top_n_genes` during marker
selection. In `ed.run_easydecon(...)`, markers are selected with
`top_n_genes=None`, then `top_n_genes` is applied during Phase 1 and Phase 2
role routing. With marker roles, the limit is per group and role.

`top_n_markers` is separate. It limits markers inside selected Phase 2 scoring
methods such as AUC and UCell-like scoring, after marker-table routing.

## Marker roles and signed inference

Manual marker tables may include `marker_role`. Missing or blank roles are
treated as `positive`. Unknown roles raise an error.

`marker_role_inference="signed"` works with signed Scanpy, DESeq, and PyDESeq2
marker rows. It infers `positive` and `negative` roles from signed
`logfoldchanges`, optionally checking finite directional Scanpy scores or
DESeq statistics. It is opt-in and intended for `marker_roles="shared"` when
negative-marker-capable `method="ucell"` scoring is desired.
`"scanpy_signed"` remains a backward-compatible alias.

Reference-profile marker generation can create `presence`, `identity`, and
`negative` roles with `marker_roles="phase_specific"`. Scanpy and PyDESeq2 do
not automatically create phase-specific presence/identity roles.

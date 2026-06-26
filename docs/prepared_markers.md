# Reusing marker preparation

`PreparedMarkers` separates marker preparation from spatial gene-universe
filtering. It stores a standardized but spatial-unfiltered marker table so one
single-cell reference, one DESeq-style marker table, or one marker file can be
reused across multiple spatial datasets.

## What PreparedMarkers stores

`PreparedMarkers` has these fields:

`raw_markers_df`
: Standardized marker rows before filtering to a spatial gene universe.

`marker_method`
: Normalized method name such as `existing`, `scanpy`, `pydeseq2`, or
  `reference`.

`source`
: Source label, for example `scanpy_generated['rank_genes_groups']` or
  `reference_profile`.

`parameters`
: Normalized marker-generation parameters used for the practical signature.

`diagnostics`
: Marker-generation diagnostics.

`signature`
: A deterministic practical signature. AnnData preparations include marker
  method, parameters, reference annotations, gene names, sample labels, and an
  expression summary. Table preparations include canonical table content,
  dtypes, marker roles, and preparation parameters.

## Function responsibilities

`prepare_markers`
: Source loading, marker generation, alias resolution, canonicalization, and
  optional signed Scanpy role inference. It accepts AnnData, marker DataFrames,
  marker files, or existing `PreparedMarkers` objects. It does not filter to a
  spatial gene universe.

`select_prepared_markers`
: Spatial-specific marker selection from a `PreparedMarkers` object. It applies
  spatial gene intersection, generic log-fold-change and p-value filters,
  mitochondrial/ribosomal filtering, excluded cell types, and optional direct
  top-N selection.

`resolve_phase_marker_tables`
: Internal Phase 1/Phase 2 role routing and workflow top-N selection.

`read_markers_dataframe`
: Supported backward-compatible convenience wrapper that returns a selected
  DataFrame for one spatial dataset. It delegates to
  `prepare_markers` and `select_prepared_markers`; it is not deprecated.

## Example: reuse one reference

```python
import easydecon as ed

prepared = ed.prepare_markers(
    sc_adata,
    marker_method="scanpy",
    groupby="cell_type",
    scanpy_method="wilcoxon",
    verbose=False,
)

result_a = ed.run_easydecon(
    spatial_a,
    prepared_markers=prepared,
    filtering_algorithm="permutation",
    return_result_object=True,
    verbose=False,
)

result_b = ed.run_easydecon(
    spatial_b,
    prepared_markers=prepared,
    filtering_algorithm="permutation",
    return_result_object=True,
    verbose=False,
)
```

Marker generation runs once. For each spatial dataset, easydecon filters the
prepared marker table to the dataset's `var_names` and applies marker
thresholds, ribosomal/mitochondrial filters, and top-N selection.

## Example: reuse an existing DESeq-style table

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
```

Common DESeq-style aliases such as `cell_type`, `gene`, `log2FoldChange`,
`padj`, and `stat` are canonicalized to `group`, `names`, `logfoldchanges`,
`pvals_adj`, and `scores`.

## When to regenerate

Regenerate `PreparedMarkers` when any input that affects marker generation
changes:

* the reference expression matrix;
* `adata.obs` annotations;
* `groupby`;
* biological sample labels used by PyDESeq2;
* `layer` or `use_raw`;
* Scanpy, PyDESeq2, or reference-profile parameters;
* marker role settings or signed role inference; or
* the intended marker method.

`PreparedMarkers` does not mutate the single-cell AnnData object. If you use
`marker_role_inference="scanpy_signed"`, recreate the preparation when you want
those inferred roles stored for later reuse.

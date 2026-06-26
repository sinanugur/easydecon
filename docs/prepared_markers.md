# Reusing marker preparation

`PreparedMarkers` separates expensive marker generation from spatial
gene-universe filtering. It stores a standardized but spatial-unfiltered marker
table so one single-cell reference can be reused across multiple spatial
datasets.

## What PreparedMarkers stores

`PreparedMarkers` has these fields:

`raw_markers_df`
: Standardized marker rows before filtering to a spatial gene universe.

`marker_method`
: Normalized method name such as `scanpy`, `pydeseq2`, or `reference`.

`source`
: Source label, for example `scanpy_generated['rank_genes_groups']` or
  `reference_profile`.

`parameters`
: Normalized marker-generation parameters used for the practical signature.

`diagnostics`
: Marker-generation diagnostics.

`signature`
: A deterministic practical signature based on marker method, parameters,
  reference annotations, gene names, sample labels, and expression summary.

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

## Function responsibilities

`prepare_markers`
: Generate or extract a reusable marker table from `adata`. It does not filter
  to a spatial gene universe.

`select_prepared_markers`
: Filter a `PreparedMarkers` object to a spatial gene universe.

`read_markers_dataframe`
: Resolve one marker source for one spatial table. If `prepared_markers` is
  passed, it delegates to `select_prepared_markers`.

`run_easydecon`
: Reads markers with deferred top-N behavior, routes markers by phase, then
  runs Phase 1 and Phase 2.

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

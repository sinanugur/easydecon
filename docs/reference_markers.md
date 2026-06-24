# Reference-profile markers

`marker_method="reference"` selects markers from single-cell reference profiles
without running a differential-expression test. It is useful when you want a
lightweight, reusable marker table based on normalized abundance and detection
specificity.

## Input requirements

- An AnnData-like object with `obs`, `var_names`, and `X`.
- A `groupby` column in `adata.obs`.
- Unique `adata.var_names`.
- Non-negative finite abundance values.

When `layer` is provided, `adata.layers[layer]` is used. When `layer=None`,
reference marker generation prefers `adata.layers["counts"]` if available and
otherwise uses `adata.X`. Raw integer counts are preferred but not required.

## Library-size normalization

Cells with missing group labels and cells with total abundance `<= 0` are
excluded. Each retained cell is normalized by its own total abundance, so rows
sum to approximately one. Mean profiles are then calculated per cell type from
the normalized matrix.

Detection fractions are calculated from the original abundance matrix as the
fraction of cells in a group where a gene is greater than zero.

## Competitor contrasts

For each target group, easydecon compares the target mean profile against all
other retained groups:

- `reference_contrast="max_other"` compares against the strongest competing
  group for each gene.
- `reference_contrast="mean_other"` compares against the unweighted average of
  competing group profiles.

`max_other` is stricter and favors markers that distinguish the target from its
closest competitor. `mean_other` is more permissive.

## Detection thresholds

Markers must pass minimum mean expression, log2 fold-change, target detection,
and target-minus-competitor detection thresholds. Ribosomal and mitochondrial
gene filters are available for direct calls to
`compute_reference_profile_markers`; generic marker filtering is also applied
later by `read_markers_dataframe` or `select_prepared_markers`.

## Output columns

The output marker table includes canonical columns `group`, `names`,
`logfoldchanges`, `scores`, `marker_rank`, and `marker_source`, plus reference
profile metadata such as `mean_target`, `mean_other`, `max_other`,
`detection_target`, `detection_other_max`, `log2fc_mean`, `log2fc_max`,
`shared_marker_weight`, and `n_celltypes_expressing_gene`.

Reference-profile markers do not fabricate `pvals_adj` or other p-values.

## PreparedMarkers reuse

```python
prepared = ed.prepare_markers(
    sc_adata,
    marker_method="reference",
    groupby="cell_type",
    layer="counts",
)
```

`PreparedMarkers` stores the reference marker table before spatial gene-universe
filtering, so the same reference profiles can be reused across spatial
datasets with different genes.

## Comparison with other marker methods

- Scanpy marker generation runs rank-based or test-based comparisons through
  `sc.tl.rank_genes_groups`.
- Pseudobulk PyDESeq2 uses sample-level count aggregation and a negative
  binomial model.
- Reference-profile markers use normalized cell-type profiles and detection
  thresholds, not statistical p-values.
- This method is inspired by reference-profile marker-selection concepts, but
  it is not an implementation of the complete RCTD model or its likelihood.

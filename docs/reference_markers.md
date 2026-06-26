# Reference-profile markers

`marker_method="reference"` selects markers from single-cell reference profiles
without a differential-expression test. It uses per-cell library-size
normalization, group mean profiles, detection fractions, and competitor
contrasts. It is not a full RCTD implementation and does not calculate
statistical p-values.

Alias `marker_method="rctd_like"` is accepted for compatibility, but examples
use `reference`.

## Input requirements

* An AnnData-like object with `obs`, `var_names`, and `X`.
* A `groupby` column in `adata.obs`.
* Unique `adata.var_names`.
* Non-negative finite abundance values.
* At least two retained groups with at least `reference_min_cells` cells each.

When `layer` is a string, `adata.layers[layer]` is used. When `layer=None`, the
implementation prefers `adata.layers["counts"]` if present and otherwise uses
`adata.X`. Raw counts are preferred but not required for reference-profile
markers.

## Profile construction

Cells with missing group labels or total abundance `<= 0` are excluded. Each
retained cell is normalized by its total abundance. Mean profiles are computed
per group from the normalized matrix. Detection fractions are computed from the
original abundance matrix as the fraction of cells in the group where a gene is
greater than zero.

`shared_marker_weight` downweights genes expressed across many groups using the
number of groups with nonzero mean expression.

## Competitor contrasts

`reference_contrast="max_other"`
: Compare the target mean profile with the strongest competing group for each
  gene. This is stricter and is the default.

`reference_contrast="mean_other"`
: Compare the target mean profile with the average of competing groups.

## Thresholds

Shared reference markers must pass mean expression, log2 fold-change, target
detection, and target-minus-competitor detection thresholds. Ribosomal and
mitochondrial filters are available in direct calls to
`compute_reference_profile_markers`; workflow-level marker filtering is applied
later by `read_markers_dataframe` or `select_prepared_markers`.

## Output columns

Shared-mode output includes canonical columns such as `group`, `names`,
`logfoldchanges`, `scores`, `marker_rank`, and `marker_source`, plus reference
metadata including `mean_target`, `mean_other`, `max_other`,
`detection_target`, `detection_other_max`, `log2fc_mean`, `log2fc_max`,
`shared_marker_weight`, and `n_celltypes_expressing_gene`.

Reference-profile markers do not fabricate `pvals_adj`.

## Phase-specific roles

With `marker_roles="phase_specific"`, reference-profile generation can emit:

`presence`
: Phase 1 markers based on target expression and permissive presence
  thresholds.

`identity`
: Phase 2 positive markers based on target-vs-competitor specificity.

`negative`
: Anti-markers based on genes stronger in competitors than the target.
  Negative rows store a positive penalty magnitude in `logfoldchanges` and
  `negative_log2fc`; UCell-like scoring decides direction from `marker_role`.

Presence and identity rows may overlap for the same group and gene. Marker
ranks are assigned per group and role.

## PreparedMarkers reuse

```python
prepared = ed.prepare_markers(
    sc_adata,
    marker_method="reference",
    marker_roles="phase_specific",
    groupby="cell_type",
    layer="counts",
    verbose=False,
)
```

`PreparedMarkers` stores the spatial-unfiltered marker table, so the same
reference profiles can be reused across spatial datasets with different gene
universes.

## Comparison with other marker methods

* Scanpy markers use `scanpy.tl.rank_genes_groups` and expect suitable
  normalized/log-transformed input.
* Pseudobulk PyDESeq2 uses raw integer counts and biological replicate labels.
* Reference-profile markers use normalized profiles and detection thresholds,
  not p-values.
* Reference-profile markers are inspired by reference-profile marker-selection
  ideas, but they are not a complete RCTD likelihood model.

# Troubleshooting

## No marker genes remain after filtering

Likely cause: thresholds are too strict or marker genes do not overlap
`table.var_names`.

Inspect: `ed.summarize_marker_table(result.markers_df)` or
`read_markers_dataframe(..., return_diagnostics=True)`.

Action: relax `log2fc_min`, `pval_cutoff`, `top_n_genes`, ribosomal/mitochondrial
filters, or fix gene identifiers.

## Gene names do not overlap

Likely cause: one table uses symbols and the other uses Ensembl IDs.

Inspect: compare `set(markers_df["names"])` with `set(table.var_names)`.

Action: map identifiers upstream and rerun marker loading.

## posterior_df is None

Likely cause: list-style `marker_genes` was used, so Phase 1 acts as a row
mask.

Inspect: `result.diagnostics["assignment_matrix"]`.

Action: use a grouped marker table for cell-type-specific priors, or use
`assignment_df`/`phase2_result` for this workflow.

## All rows are unassigned

Likely cause: all score rows are zero, tied, or below `minimum_evidence`.

Inspect: row maxima in `result.assignment_df` and `result.assigned_labels`.

Action: inspect marker overlap, relax assignment settings, or use a different
Phase 2 method.

## Too many ties

Likely cause: equal score rows or insufficient marker specificity.

Inspect: `result.assignment_df.loc[row].sort_values(ascending=False)`.

Action: use more specific markers, increase marker counts, or adjust
`tie_tolerance` carefully.

## Phase 1 priors are all zero

Likely cause: no marker signal passed Phase 1 filtering.

Inspect: `result.phase1_result.sum(axis=1)` and marker overlap.

Action: use `filtering_algorithm="quantile"` as a fast exploratory shortcut,
relax `quantile`, or check gene identifiers. The standard Phase 1 workflow
uses permutation filtering.

## UCell scores are all zero

Likely cause: too few detected positive markers, constant rows, all-zero rows,
or overly strict `expression_threshold`.

Inspect: `result.diagnostics["phase2"]` and UCell marker counts.

Action: lower `min_markers`, lower `expression_threshold`, or review marker
roles.

## No negative markers are present

Likely cause: marker table has no `negative` role rows or signed Scanpy
inference was not requested.

Inspect: `result.markers_df["marker_role"].value_counts()`.

Action: provide manual negative markers, use reference phase-specific markers,
or opt into `marker_role_inference="signed"` for signed DE-style markers.

## Reference marker generation returns no markers

Likely cause: reference thresholds are too strict or too few cells remain per
group.

Inspect: diagnostics from `prepare_markers` or `read_markers_dataframe`.

Action: relax reference thresholds or check group labels and expression layers.

## PyDESeq2 raw-count validation fails

Likely cause: selected layer is not raw non-negative integer counts.

Inspect: `adata.layers[layer]` values and dtype.

Action: provide `layer="counts"` or another raw-count layer.

## Insufficient pseudobulk replicates

Likely cause: `sample_col` does not contain enough biological replicates for
target/rest comparisons.

Inspect: counts by `groupby` and `sample_col`.

Action: use true replicate labels, lower minimums only if scientifically
justified, or use a different marker method.

## Missing spatial coordinates

Likely cause: visualization or niche detection expected `obsm["spatial"]`.

Inspect: `table.obsm.keys()` and `table.obs.columns`.

Action: add coordinates to `table.obsm["spatial"]` or provide `x`/`y` columns
for niche detection.

## Candidate pruning incompatibility

Likely cause: `prior_weight <= 0`, list-style `marker_genes`, or
`refine_group(mode="phase2")`.

Inspect: workflow parameters.

Action: disable candidate pruning, use grouped markers, or switch refinement to
`mode="full"`.

## refine_group finds no eligible locations

Likely cause: `parent_threshold` is too high or the parent group has no support.

Inspect: `parent.priors_df[parent_group]` or `parent.posterior_df[parent_group]`.

Action: lower `parent_threshold`, use the other `parent_source`, or review
parent markers.

## Optional SpatialData dependency missing

Likely cause: SpatialData extras are not installed.

Inspect: import errors mentioning `spatialdata`.

Action: install `easydecon[spatial]` or pass an AnnData table directly.

## Documentation build import failures

Likely cause: missing core dependencies or environment-specific Numba/Scanpy
cache behavior.

Inspect: the first Sphinx traceback.

Action: install `python -m pip install -e ".[docs]"`; if Scanpy import fails in
the local environment, run with `NUMBA_DISABLE_JIT=1`.

## High runtime for permutation Phase 1

Likely cause: large `num_permutations`, `subsample_size`, or many marker
groups.

Inspect: Phase 1 progress bars and parameter values.

Action: use `filtering_algorithm="quantile"` for iteration when you need a
fast exploratory shortcut, then return to permutation filtering for standard
analyses.

## Memory concerns

Likely cause: large spatial matrices, dense conversions in selected workflows,
or high parallelism.

Inspect: matrix sparsity, selected Phase 2 method, and `config.n_jobs`.

Action: reduce `n_jobs`, use smaller marker sets, preserve sparse inputs, and
avoid unnecessary full-data copies.

# Phase 2: marker-profile similarity

Weighted Jaccard is the default Phase 2 method, but easydecon supports several
method families. No method should be assumed universally superior.

Phase 2 is implemented by `get_clusters_by_similarity_on_tissue`. It scores
each processed spatial location against marker groups and returns
`phase2_result`. The workflow then maps that evidence to `likelihoods_df`.

## Balanced method comparison

Supported values are defined in `SIMILARITY_METHODS`.

| Method | Family | Uses marker weights | Uses full expression row or marker union | Supports negative markers | Typical use | Main caveat |
| --- | --- | --- | --- | --- | --- | --- |
| `wjaccard` | weighted marker overlap | yes | full expression row | no | default starting method for weighted marker tables | depends on marker-weight quality and denominator includes non-marker expressed genes |
| `jaccard` | set overlap | no | full expression row | no | binary expressed-gene and marker-set comparisons | non-marker expressed genes affect the denominator |
| `overlap` | set overlap | no | full expression row | no | marker recovery as an overlap coefficient | ignores marker weights and denominator differs from Jaccard |
| `cosine` | vector profile | marker reference values | marker union | no | profile-vector comparisons | sensitive to marker scaling and sparsity |
| `correlation` | vector profile | marker reference values | marker union | no | rank-style profile comparisons | implemented with Spearman correlation and a detected-marker fraction factor |
| `euclidean` | vector profile | marker reference values | marker union | no | distance-style profile comparisons | distance is converted to `1 / (1 + distance)` |
| `auc` | rank based | rank/order only | marker union | no | within-location positive-marker rank evidence | does not interpret negative marker roles |
| `ucell` | rank based | role-aware ordering | marker union | yes | rank evidence with anti-marker subtraction | not the official UCell implementation and not the default |
| `sum` | simple aggregation | no | marker union | no | transparent marker-expression baseline | sensitive to marker number and expression scale |
| `mean` | simple aggregation | no | marker union | no | transparent marker-expression baseline | sensitive to scaling and marker selection |
| `median` | simple aggregation | no | marker union | no | robust marker-expression baseline | often zero for sparse marker sets |
| `diagnostic` | diagnostic | no | marker union | no | inspect expressed marker intersections | returns sets, not a normal numeric score matrix |

Only UCell-like scoring interprets `negative` marker roles. Non-UCell methods
exclude negative rows through marker-role routing.

## Weighted marker-overlap family

`method="wjaccard"` is the package default. It compares the positive target
expression row with weighted marker membership. When `weight_column` is
available, marker weights are normalized within each group. When explicit
weights are absent, ranked fallback weights are created with `lambda_param`.
The denominator uses the union of expressed target genes and marker genes, so
non-marker expressed genes can lower the score. Negative marker roles are not
interpreted.

Common parameters:

`weight_column`
: Marker-table column used for weights, usually `logfoldchanges`.

`lambda_param`
: Exponential decay used for fallback ranked weights.

`expression_threshold`
: Controls detection in other methods; weighted Jaccard currently uses
  positive expression from the row.

Normal workflow example:

```python
result = ed.run_easydecon(
    sdata,
    markers_df=markers_df,
    filtering_algorithm="permutation",
    method="wjaccard",
    return_result_object=True,
)
```

## Set-overlap methods

`method="jaccard"` and `method="overlap"` compare the expressed-gene set in a
location with each marker set.

Jaccard uses the intersection divided by the union of expressed genes and
marker genes. Overlap uses the intersection divided by the smaller set size.
Both methods use the full expression row, both can be affected by non-marker
expressed genes, and neither interprets negative marker roles. Use them when a
set-membership interpretation is intended.

## Vector-profile methods

`method="cosine"`, `method="correlation"`, and `method="euclidean"` compare
spatial marker expression against marker-side reference values from
`similarity_by_column`, usually `logfoldchanges`.

These methods extract the marker union, align each group's reference vector,
and ignore rows without enough valid marker values. Correlation is implemented
with Spearman correlation and multiplied by the fraction of detected markers in
the group. Cosine measures vector orientation after marker-side min-max
scaling. Euclidean distance is converted to a score with `1 / (1 + distance)`.
Results depend on marker scaling, marker sparsity, and the selected reference
column.

## Rank-based methods

`method="auc"` and `method="ucell"` rank genes within each location over the
marker union. They reduce dependence on absolute expression scale and penalize
missing or unrecovered markers according to the configured parameters.

AUC uses positive-marker rank enrichment. It supports `center_auc`,
`fallback_auc`, `recovery_power`, `expression_threshold`, `top_n_markers`, and
`drop_shared_markers`.

UCell-like scoring uses positive and identity markers as positive evidence and
subtracts evidence from detected negative markers. Presence markers are ignored
in UCell-like Phase 2. It supports `ucell_max_rank`,
`ucell_negative_weight`, `min_markers`, `recovery_power`,
`expression_threshold`, `top_n_markers`, and `drop_shared_markers`.

UCell-like scoring is useful when rank robustness or anti-marker evidence is
desired. It is not the general default and is not guaranteed to outperform
weighted Jaccard, AUC, cosine, or other supported methods. See
[UCell-like Phase 2 scoring](ucell.md) for the technical subguide.

## Simple aggregation methods

`method="sum"`, `method="mean"`, and `method="median"` directly summarize
expression across each group's marker genes. They are transparent baselines, do
not require a reference-profile vector, and do not interpret negative marker
roles. They are sensitive to marker number, expression scale, and marker
quality.

## Diagnostic method

`method="diagnostic"` returns the expressed marker intersection for each group.
It is useful for debugging and inspection, not ordinary final assignments or
posterior interpretation.

## Choosing a Phase 2 method

* Start with weighted Jaccard for ordinary weighted DE marker tables.
* Use cosine or correlation for profile-vector comparisons.
* Use AUC or UCell-like scoring for within-location rank evidence.
* Use UCell-like scoring when negative markers are available.
* Use Jaccard or overlap when set membership is the intended interpretation.
* Use sum, mean, or median for transparent expression summaries.
* Compare methods on annotated or simulated validation data when possible.

Method choice should be validated on the user's dataset.

## Common Phase 2 parameters

`expression_threshold`
: Values at or below this threshold are treated as not detected by methods
  that use detection.

`min_markers`
: Minimum available or detected markers required by rank-based methods.

`fallback_auc`
: Default is `0.0` in the workflow. AUC and UCell-like uninformative rows can
  return zero evidence.

`top_n_markers`
: Phase 2 scorer-specific marker limit. This is separate from `top_n_genes`,
  which limits marker-table selection/routing.

`center_auc`
: For `method="auc"`, centered scores use `max(0, 2 * (auc - 0.5))`.

`ucell_max_rank`, `ucell_negative_weight`, `ucell_marker_role_column`
: UCell-like scoring controls described in [UCell-like Phase 2 scoring](ucell.md).

## Evidence to likelihood

`evidence_to_likelihood="softmax"`
: Applies a row-wise softmax to finite Phase 2 evidence. `softmax_tau` controls
  temperature.

`evidence_to_likelihood="row_normalize"`
: Clips or shifts evidence to non-negative values and normalizes rows.

When candidate pruning is active, likelihood normalization is candidate-aware:
noncandidate groups remain zero.

## Posterior and assignment parameters

`prior_weight` and `likelihood_weight` control posterior combination after
Phase 2 scoring. `minimum_evidence`, `tie_tolerance`,
`fold_change_threshold`, and `allow_multiple` are hard-assignment settings, not
Phase 2 scoring parameters.

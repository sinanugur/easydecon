# Phase 2: marker-profile similarity

Phase 2 is implemented by `get_clusters_by_similarity_on_tissue`. It scores
each processed spatial location against marker groups and returns
`phase2_result`. The workflow then maps that evidence to `likelihoods_df`.

## Supported methods

Supported values are defined in `SIMILARITY_METHODS`.

| Method | Output type | Row semantics | Weights | Negative markers | Main caveat |
| --- | --- | --- | --- | --- | --- |
| `correlation` | Spearman-like correlation | marker union | optional marker reference values | no | undefined rows fall back to zero-like values |
| `cosine` | cosine similarity | marker union | optional marker reference values | no | scale and sparsity affect interpretation |
| `euclidean` | inverse distance-like score | marker union | optional marker reference values | no | distance is transformed to a score |
| `sum` | expression summary | marker union | no | no | favors groups with highly expressed marker sets |
| `mean` | expression summary | marker union | no | no | sensitive to marker selection |
| `median` | expression summary | marker union | no | no | can be zero for sparse markers |
| `diagnostic` | expressed marker set | full row set | no | no | diagnostic output, not a normal score matrix |
| `jaccard` | set overlap | full expressed row | no | no | denominator includes non-marker expressed genes |
| `overlap` | marker recovery fraction | full expressed row | no | no | ignores marker weights |
| `wjaccard` | weighted set overlap | full expressed row | yes | no | depends on marker-weight quality |
| `auc` | rank-based marker enrichment | marker union | rank/order only | no | does not interpret negative roles |
| `ucell` | UCell-like rank score | marker union | role-aware ordering | yes | not the official UCell implementation |

Jaccard, overlap, and weighted Jaccard intentionally preserve full-gene row
semantics where implemented. UCell-like and AUC scoring rank marker-union genes.
Only UCell-like scoring interprets `negative` marker roles. Non-UCell methods
exclude negative rows through marker-role routing.

## Method selection guide

* Existing weighted marker table: `wjaccard` is a reasonable starting option.
* Rank-based robustness to expression scale: consider `ucell` or `auc`.
* Negative marker support: use `ucell`.
* Simple marker-expression summaries: use `sum`, `mean`, or `median`.
* Set-overlap interpretation: use `jaccard` or `overlap`.
* Profile-vector comparison: use `cosine` or `correlation`.

Validate method choice on your own data; no method is universally best.

## Common Phase 2 parameters

`expression_threshold`
: Values at or below this threshold are treated as not detected by methods that
  use detection.

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

`prior_weight` and `likelihood_weight` control posterior combination after Phase
2 scoring. `minimum_evidence`, `tie_tolerance`, `fold_change_threshold`, and
`allow_multiple` are hard-assignment settings, not Phase 2 scoring parameters.

# Optional Phase 2 candidate pruning

Phase 1 and Phase 2 use two different gates:

- row gating decides which spatial locations are processed by Phase 2;
- candidate pruning decides which cell-type groups are scored within each
  processed location.

Candidate pruning is disabled by default. Enable it only when Phase 1 produces
cell-type priors:

```python
result = ed.run_easydecon(
    sdata,
    markers_df=markers_df,
    phase2_candidate_pruning=True,
    phase2_candidate_threshold=0.0,
    return_result_object=True,
)
```

With `phase2_candidate_threshold=0.0`, a group is a candidate when its Phase 1
prior is positive. When `prior_weight > 0`, this normally preserves
`posterior_df` and hard assignments, because zero-prior groups could not win the
posterior anyway. `phase2_result` and `likelihoods_df` can still differ:
noncandidate group entries are zero in the pruned run.

A positive `phase2_candidate_threshold` is stricter: groups must have
`prior > threshold`. This can set low-prior groups to zero in `phase2_result`,
`likelihoods_df`, and `posterior_df`, and can change hard assignments or leave a
row unassigned if no groups pass.

Candidate pruning requires `prior_weight > 0` because candidates are derived
from Phase 1 priors. It is not available for list-style `marker_genes`
workflows, where Phase 1 is only a location mask and does not produce
cell-type-specific priors.

In hierarchical refinement, `refine_group(mode="full")` can use child Phase 1
priors to prune child subtype Phase 2 scoring. `refine_group(mode="phase2")`
cannot use child candidate pruning because it does not run child Phase 1; use
`parent_threshold` to restrict parent-positive locations or switch to
`mode="full"`.

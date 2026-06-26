# Optional Phase 2 candidate pruning

Phase 1 and Phase 2 use two different gates:

* row gating decides which spatial locations are processed by Phase 2;
* candidate pruning decides which marker groups are scored within each
  processed location.

Candidate pruning is disabled by default and is controlled by
`phase2_candidate_pruning` and `phase2_candidate_threshold`.

```python
result = ed.run_easydecon(
    sdata,
    markers_df=markers_df,
    filtering_algorithm="quantile",
    phase2_candidate_pruning=True,
    phase2_candidate_threshold=0.0,
    return_result_object=True,
)
```

## Zero-threshold pruning

With `phase2_candidate_threshold=0.0`, a group is a candidate when its Phase 1
prior is positive. When `prior_weight > 0`, zero-prior groups could not win the
posterior, so zero-threshold pruning normally preserves `posterior_df` and hard
assignments for non-negative methods.

`phase2_result` and `likelihoods_df` can still differ from an unpruned run
because noncandidate entries are zero.

## Positive-threshold pruning

A positive threshold requires `prior > threshold`. This is stricter and can
change `phase2_result`, candidate-aware `likelihoods_df`, `posterior_df`, and
`assigned_labels`. Rows with no candidate groups receive zero Phase 2 evidence.

## Restrictions

Candidate pruning requires `prior_weight > 0` because candidates are derived
from Phase 1 priors. It is not available for list-style `marker_genes`
workflows, where Phase 1 is only a row mask and does not produce
cell-type-specific priors.

`refine_group(mode="full")` can use child Phase 1 priors for child candidate
pruning. `refine_group(mode="phase2")` cannot use candidate pruning because it
does not run child Phase 1.

## Diagnostics and performance

Diagnostics are stored in:

```python
result.diagnostics["phase2"]["performance"]
```

Fields include `candidate_pruning_enabled`, `candidate_threshold`,
`exact_candidate_pruning`, `n_total_location_group_pairs`,
`n_candidate_pairs`, `candidate_fraction`, and candidate-count summaries.

Candidate pruning can reduce work when the candidate matrix is sparse. Runtime
benefit depends on candidate sparsity, scoring method overhead, and data shape;
it is not guaranteed for every dataset.

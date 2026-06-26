# How easydecon works

The main workflow is `ed.run_easydecon`, an alias for
`easydecon_workflow`. It turns a marker source and an AnnData-like spatial
table into evidence matrices, posterior support, and hard assignments.

```text
marker preparation
    -> Phase 1 permutation filtering
    -> Phase 1 priors
    -> Phase 2 similarity method
    -> Phase 2 likelihoods
    -> posterior support
    -> hard assignment
```

## Marker preparation

Markers can come from a DataFrame, file, existing Scanpy
`rank_genes_groups`, generated Scanpy markers, pseudobulk PyDESeq2,
reference-profile markers, or `PreparedMarkers`. The output is standardized to
canonical columns such as `group` and `names`, filtered to the spatial gene
universe, and optionally routed by marker role.

See [marker inputs](marker_inputs.md) and [prepared markers](prepared_markers.md).

## Phase 1

Phase 1 asks whether a marker group is plausible at each spatial location. It
aggregates expression for the Phase 1 markers and, by default, compares the
signal with a random-gene permutation null. Quantile filtering is available for
fast exploratory runs, and NB filtering is available for specialized raw-count
workflows. The workflow row-normalizes non-negative Phase 1 output into
`priors_df`.

See [Phase 1](phase1.md).

## Phase 2

Phase 2 measures marker-profile similarity or marker-rank evidence at spatial
locations selected by the Phase 1 row mask. It produces `phase2_result`, then
maps the evidence into `likelihoods_df` using `softmax` or `row_normalize`.
Weighted Jaccard is the default Phase 2 implementation choice; other supported
methods are useful for vector-profile, set-overlap, rank-based, simple
aggregation, and diagnostic workflows.

See [Phase 2](phase2.md).

## Posterior support

When `marker_genes` is not a plain list, easydecon aligns Phase 1 priors and
Phase 2 likelihoods by location and marker group, then combines them as:

```text
posterior_unnormalized = priors_df ** prior_weight * likelihoods_df ** likelihood_weight
```

Rows are normalized after combination. A zero Phase 1 prior usually prevents a
group from receiving posterior support when `prior_weight > 0`.

Posterior rows are relative support among the tested groups. They are not
automatically absolute cell fractions.

## Hard assignment

Final hard labels are assigned from `assignment_df`, which is usually
`posterior_df`. With list-style `marker_genes`, Phase 1 is only a row mask,
`posterior_df` is `None`, and assignment is made from `phase2_result`.

Hard assignment discards uncertainty. Use `posterior_df`, `priors_df`, and
diagnostics when ambiguity matters.

## Candidate pruning

Candidate pruning is a compute optimization, not a third inference phase. It
uses Phase 1 priors to decide which marker groups are scored in Phase 2 for
each spatial location. Noncandidate entries are zeroed before likelihood
normalization.

See [candidate pruning](candidate_pruning.md) and [results](results.md).

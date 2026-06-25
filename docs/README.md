# easydecon documentation

- Getting started: see the root [README](../README.md).
- Understanding results: [results.md](results.md).
- Visualization recipes: [visualization.md](visualization.md).
- Refining broad groups into subclusters: [refinement.md](refinement.md).
- Reference-profile markers: [reference_markers.md](reference_markers.md).
- UCell-like Phase 2 scoring: [ucell.md](ucell.md).
- Marker roles and phase-specific routing: [marker_roles.md](marker_roles.md).
- Optional Phase 2 candidate pruning: [candidate_pruning.md](candidate_pruning.md).
- Synthetic validation benchmarks: [validation.md](validation.md).

## Workflow overview

```text
Marker table
    |
    v
Phase 1: marker-expression filtering
    |
    v
Priors
    |
    +-------------------+
                        |
Spatial expression     |
    |                   |
    v                   |
Phase 2: similarity    |
    |                   |
    v                   |
Likelihoods -----------+
    |
    v
Posterior
    |
    v
Hard assignment
```

Phase 1 determines which cell types are plausible at each spatial location.
Phase 2 evaluates marker-profile similarity between the selected markers and
spatial expression. Priors and likelihoods are combined into posterior
probabilities. When `prior_weight > 0`, a zero Phase 1 prior normally prevents
that cell type from receiving posterior probability.

List-style `marker_genes` workflows are different: Phase 1 is used as a
location mask and final assignment is made directly from Phase 2, so
`posterior_df` is `None`.

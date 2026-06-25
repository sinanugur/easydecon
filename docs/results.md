# Understanding EasyDeconResult

`EasyDeconResult` is returned when you call `ed.run_easydecon(...,
return_result_object=True)`.

| Field | Shape/type | Meaning | Typical use |
| --- | --- | --- | --- |
| `markers_df` | marker rows × metadata columns | Spatial-compatible selected markers | Marker QC |
| `phase1_result` | locations × cell types | Thresholded Phase 1 marker-expression evidence | Presence detection |
| `priors_df` | locations × cell types | Row-normalized Phase 1 values | Prior gating and hierarchical analyses |
| `phase2_result` | locations × cell types | Raw marker-profile similarity | Similarity inspection |
| `likelihoods_df` | locations × cell types | Normalized Phase 2 evidence | Comparing Phase 2 support |
| `posterior_df` | locations × cell types or `None` | Combined priors and likelihoods | Preferred probabilistic output |
| `assignment_df` | locations × cell types | Matrix used for hard assignment | Reassignment and diagnostics |
| `assigned_labels` | locations × assignment columns | Final categorical labels | Plotting and annotation |
| `diagnostics` | `dict` | Workflow and marker-generation metadata | QC and reproducibility |
| `prepared_markers` | `PreparedMarkers` or `None` | Reusable marker preparation | Reusing DE results |

`PreparedMarkers` stores reusable marker-generation output before spatial gene
filtering. This lets the same marker preparation be filtered cheaply for
different spatial gene universes.

## Which matrix should I use?

- Spatial probability or composition plots: `result.posterior_df`
- Presence or gating: `result.priors_df`
- Inspecting marker similarity independently of Phase 1: `result.phase2_result`
- Hard cell-type map: `result.assigned_labels`
- Reassigning labels without recomputing scores: `result.assignment_df`
- List-style marker mask workflow: `result.assignment_df` or `result.phase2_result`

`posterior_df` is the preferred probabilistic output when available. It is
`None` for list-style `marker_genes` mask workflows because those assign
directly from Phase 2.

## Important interpretation note

Posterior rows are relative probabilities among the tested cell types. They are
not automatically absolute cell fractions. Hard labels discard uncertainty, and
mixed spatial locations can have meaningful support for multiple cell types.
Inspect posterior maxima and diagnostics before interpreting assignments.

## Phase 2 expression handling

UCell-like and AUC Phase 2 scoring rank only the resolved marker-union genes.
Other marker-only similarity methods also extract only the marker union when
that is mathematically equivalent.

Some set-based methods intentionally use the full expression row: Jaccard,
overlap, and weighted Jaccard consider non-marker expressed genes in their
denominators, so easydecon preserves the full-gene row for those methods.

Sparse spatial matrices are processed row-wise during Phase 2 without
densifying the complete spatial matrix. Individual selected rows are converted
as needed for scoring.

## Candidate-pruned Phase 2 results

When `phase2_candidate_pruning=True`, Phase 1 priors define which cell types are
scored at each spatial location. Noncandidate entries are zero in
`phase2_result` and `likelihoods_df`. With the default pruning threshold of
zero and `prior_weight > 0`, the final `posterior_df` is expected to match the
unpruned posterior for non-negative similarity methods. Positive pruning
thresholds are intentionally more aggressive and can change posterior
probabilities or hard labels.

```python
summary = ed.summarize_easydecon_result(
    result,
    sdata=sdata,
    as_dataframe=False,
)

print(summary["posterior"])
print(summary["assignments"])
```

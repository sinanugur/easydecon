# Understanding EasyDeconResult

`EasyDeconResult` is returned by:

```python
result = ed.run_easydecon(..., return_result_object=True)
```

| Field | Shape/type | Meaning | Typical use |
| --- | --- | --- | --- |
| `markers_df` | marker rows x metadata columns | Spatial-compatible markers after role routing and final de-duplication | Marker QC |
| `phase1_result` | locations x groups | Thresholded Phase 1 expression evidence | Presence detection |
| `priors_df` | locations x groups | Row-normalized Phase 1 evidence | Prior gating and refinement |
| `phase2_result` | locations x groups | Raw Phase 2 similarity or rank evidence | Method diagnostics |
| `likelihoods_df` | locations x groups | Normalized Phase 2 evidence | Compare Phase 2 support |
| `posterior_df` | locations x groups or `None` | Combined prior and likelihood support | Preferred probabilistic matrix |
| `assignment_df` | locations x groups | Matrix used for hard assignment | Reassignment and inspection |
| `assigned_labels` | locations x assignment columns | Final categorical labels | Plotting and annotation |
| `diagnostics` | `dict` | Marker, Phase 2, assignment, and workflow metadata | QC and reproducibility |
| `prepared_markers` | `PreparedMarkers` or `None` | Reusable marker preparation passed into the workflow | Marker reuse |

## Which matrix should I use?

* Spatial support plots: `posterior_df` when available.
* Phase 1 presence maps: `priors_df`.
* Phase 2 method debugging: `phase2_result`.
* Hard cell-type maps: `assigned_labels`.
* Reassignment without rescoring: `assignment_df`.
* List-style marker mask workflow: `assignment_df` or `phase2_result`.

## List-style marker_genes exception

When `marker_genes` is a plain list, Phase 1 is used as a row mask rather than
a cell-type-specific prior matrix. In that workflow `posterior_df` is `None`
and `assignment_df` is `phase2_result`.

## Interpretation warnings

Posterior rows are relative support among tested marker groups. They are not
automatically absolute biological cell fractions. Spatial locations may contain
mixtures, and hard labels discard uncertainty.

Assignment counts are counts of spatial units, not biological cell counts.
Depending on the data, a unit may be a spot, bin, or segmented cell.

## Candidate-pruned matrices

With `phase2_candidate_pruning=True`, noncandidate group entries are zero in
`phase2_result` and `likelihoods_df`. With `phase2_candidate_threshold=0.0` and
`prior_weight > 0`, the posterior is normally preserved for non-negative
methods, but positive thresholds can change `posterior_df` and
`assigned_labels`.

Candidate pruning diagnostics live under:

```python
result.diagnostics["phase2"]["performance"]
```

## Marker-role diagnostics

Role routing diagnostics live under:

```python
result.diagnostics["marker_roles"]
```

They include the routing mode, roles used in Phase 1 and Phase 2, marker counts
by group, and marker counts by role when available.

## Summary helpers

```python
summary = ed.summarize_easydecon_result(
    result,
    sdata=sdata,
    as_dataframe=False,
)

marker_summary = ed.summarize_marker_table(result.markers_df)
```

`summarize_easydecon_result` reports marker counts, workflow settings, matrix
shapes, row-sum statistics, assignment counts, and optional spatial alignment
checks. `summarize_marker_table` returns compact per-group marker counts and
top genes.

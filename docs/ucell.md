# UCell-like Phase 2 scoring

`method="ucell"` scores each spatial location from the ranks of marker genes
inside that location. Highly expressed positive markers increase evidence for a
group, while detected negative markers can reduce it.

```python
result = ed.run_easydecon(
    sdata,
    markers_df=markers_df,
    filtering_algorithm="quantile",
    method="ucell",
    min_markers=3,
    top_n_markers=50,
    return_result_object=True,
    verbose=False,
)
```

This is a lightweight, UCell-inspired rank score. It is intended as another
Phase 2 evidence option inside easydecon; it is not a drop-in implementation of
the R package.

## Marker roles

If the marker table contains `marker_role`, roles are interpreted as follows:

- `positive` or `identity`: positive signature genes.
- `negative`: genes that subtract evidence when detected.
- `presence`: ignored by UCell-like Phase 2 scoring.

Missing or blank roles are treated as `positive`. If the role column is absent,
all markers are treated as positive. Unknown roles raise an error so typos are
visible.

```python
markers_df = pd.DataFrame(
    {
        "group": ["T cell", "T cell", "B cell"],
        "names": ["CD3D", "MS4A1", "MS4A1"],
        "marker_role": ["positive", "negative", "positive"],
    }
)

result = ed.run_easydecon(
    sdata,
    markers_df=markers_df,
    method="ucell",
    ucell_negative_weight=0.5,
    filtering_algorithm="quantile",
    return_result_object=True,
)
```

## Useful parameters

- `min_markers`: minimum available and detected positive markers required to
  score a group at a location.
- `expression_threshold`: values at or below this threshold are treated as not
  detected.
- `top_n_markers`: keeps the strongest positive and strongest negative markers
  separately for each group before scoring.
- `drop_shared_markers`: removes positive markers shared by more than one
  group, while leaving negative markers unchanged.
- `recovery_power`: controls how strongly missing positive markers reduce the
  final score.
- `ucell_max_rank`: caps the rank window used by the normalized rank score.
- `ucell_negative_weight`: controls how strongly negative-marker evidence is
  subtracted.
- `ucell_marker_role_column`: changes the role column name from the default
  `marker_role`.

When every value in the marker union is zero or uninformative for a location,
the UCell-like scorer returns zeros for all groups. The normal assignment logic
then leaves tied or all-zero rows unassigned unless you explicitly relax the
assignment settings.

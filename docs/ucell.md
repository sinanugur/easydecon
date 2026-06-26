# UCell-like Phase 2 scoring

`method="ucell"` scores each spatial location from ranks within that location.
Highly expressed positive markers increase evidence for a group. Detected
negative markers can subtract evidence. This is UCell-like scoring inside
easydecon; it is not the official UCell implementation and should not be
treated as a calibrated probability model.

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

## Role behavior

If the marker table has a `marker_role` column:

* `positive` and `identity` are positive signature genes;
* `negative` genes subtract evidence when detected;
* `presence` is ignored by UCell-like Phase 2; and
* a gene cannot be both positive/identity and negative for the same group.

If the role column is absent, all markers are treated as positive. Unknown
roles raise an error.

## Parameters

`min_markers`
: Minimum available and detected positive markers needed to score a group.

`expression_threshold`
: Values at or below this threshold are treated as not detected.

`top_n_markers`
: Keeps the strongest positive and strongest negative markers separately for
  each group.

`recovery_power`
: Controls the penalty for missing positive markers.

`drop_shared_markers`
: Removes positive markers shared by more than one group. Shared negative
  markers are not removed.

`ucell_max_rank`
: Truncates the rank window used by the normalized rank score.

`ucell_negative_weight`
: Controls how strongly negative-marker evidence is subtracted.

`ucell_marker_role_column`
: Changes the role-column name from `marker_role`.

## Uninformative rows

All-zero rows, constant rows, rows with too few available markers, and rows with
too few detected positive markers return zero evidence for affected groups.
Tie-safe assignment then leaves all-zero or tied rows unassigned unless you
relax assignment settings.

## Marker sources

UCell-like scoring works with ordinary marker tables, manual negative markers,
reference-profile markers, signed Scanpy markers, `PreparedMarkers`, and
`refine_group`. Signed Scanpy inference is opt-in:

```python
result = ed.run_easydecon(
    sdata=sdata,
    adata=sc_adata,
    groupby="cell_type",
    marker_method="scanpy",
    marker_role_inference="scanpy_signed",
    marker_roles="shared",
    method="ucell",
    filtering_algorithm="quantile",
    return_result_object=True,
)
```

## Difference from AUC

`method="auc"` is also rank-based and uses marker-union genes, but it does not
interpret negative marker roles. Use UCell-like scoring when anti-markers are a
core part of the marker design.

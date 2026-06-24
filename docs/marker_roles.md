# Marker roles and phase-specific routing

easydecon supports two marker-role modes:

- `marker_roles="shared"`: the default. Existing marker workflows keep using
  the same selected marker table for Phase 1 and Phase 2.
- `marker_roles="phase_specific"`: Phase 1 and Phase 2 use role-selected
  marker subsets.

## Roles

- `presence`: sensitive markers used for Phase 1 marker-expression priors.
- `identity`: specific markers used for Phase 2 similarity.
- `positive`: accepted as identity-compatible for manually annotated marker
  tables.
- `negative`: anti-markers used only by UCell-like Phase 2 scoring.

The same group/gene can appear twice with different roles, especially as both
`presence` and `identity`. This is intentional: a gene can help detect whether
a group is plausible in Phase 1 and also help distinguish it in Phase 2.

## Reference-profile role generation

Automatic role generation is currently limited to reference-profile markers:

```python
prepared = ed.prepare_markers(
    sc_adata,
    marker_method="reference",
    marker_roles="phase_specific",
    groupby="cell_type",
    layer="counts",
)
```

Reference-profile generation emits spatial-unfiltered rows with
`marker_role`, profile metrics, `scores`, `logfoldchanges`, and
`marker_source="reference_profile"`. Presence and identity rows use target
expression enrichment. Negative rows use a positive penalty magnitude in
`logfoldchanges` and `negative_log2fc`; UCell decides direction from
`marker_role`, not from the sign.

## Manual marker tables

Manually supplied `markers_df`, files, and `PreparedMarkers` may include a
`marker_role` column. Missing or blank roles are treated as `positive`.
Unknown roles raise an error.

Scanpy and PyDESeq2 marker generation do not infer phase-specific roles. To use
`marker_roles="phase_specific"` with those marker sources, provide a marker
table that already contains `marker_role`.

## Routing behavior

In shared mode without a role column, Phase 1 and Phase 2 receive the same
table.

In shared mode with a role column:

- Phase 1 uses `positive`, `presence`, and `identity`.
- UCell Phase 2 uses `positive`, `identity`, and `negative`.
- Non-UCell Phase 2 uses `positive`, `presence`, and `identity`.

In phase-specific mode:

- Phase 1 uses `presence`.
- UCell Phase 2 uses `positive`, `identity`, and `negative`.
- Non-UCell Phase 2 uses `positive` and `identity`.

If `marker_genes` is provided to `run_easydecon`, it remains a Phase 1
override. Role routing still applies to Phase 2.

## Refinement

`refine_group(..., mode="full", marker_roles="phase_specific")` runs child
Phase 1 with presence markers and child Phase 2 with identity/negative markers.

`mode="phase2"` does not run child Phase 1, so presence markers are ignored and
only the Phase 2 role subset is used.

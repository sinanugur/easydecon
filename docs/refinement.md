# Refining broad groups into subclusters

`ed.refine_group` lets you take a parent result, such as a coarse Myeloid
assignment, and run a second marker-based analysis only where that parent group
is supported.

## Parent analysis

First run a normal easydecon workflow with broad cell types:

```python
parent = ed.run_easydecon(
    sdata,
    markers_df=broad_markers,
    return_result_object=True,
)
```

The parent score can come from `parent.priors_df` or `parent.posterior_df`.
`parent_source="priors"` is useful when you want Phase 1 presence gating.
`parent_source="posterior"` is stricter because it also includes Phase 2
similarity support. If `parent.posterior_df` is `None`, use
`parent_source="priors"`.

## Phase-2-only child refinement

This is the default and fastest mode:

```python
refined = ed.refine_group(
    sdata,
    parent_result=parent,
    parent_group="Myeloid",
    markers_df=myeloid_subcluster_markers,
    mode="phase2",
    parent_source="priors",
    parent_threshold=0.0,
)
```

`mode="phase2"` uses the previous Myeloid score as a parent gate and runs only
subtype marker-profile similarity inside the eligible locations. It does not
run child Phase 1.

## Full child refinement

Use full refinement when you want subtype-specific Phase 1 priors as well as
Phase 2 evidence:

```python
refined = ed.refine_group(
    sdata,
    parent_result=parent,
    parent_group="Myeloid",
    markers_df=myeloid_subcluster_markers,
    mode="full",
    parent_source="priors",
    parent_threshold=0.0,
)
```

`mode="full"` subsets to parent-positive locations, then runs
`run_easydecon(..., return_result_object=True)` on that child table.

## Conditional versus absolute subtype values

`refined.conditional_df` contains relative subtype probabilities within the
parent group. These rows usually sum to one for eligible locations with subtype
evidence.

`refined.absolute_df` multiplies each conditional subtype value by the parent
score:

```python
refined.absolute_df = refined.conditional_df * refined.parent_scores
```

When conditional subtype probabilities sum to one, absolute subtype values sum
to the parent score. Locations outside the parent gate remain zero and are left
unassigned in `refined.assigned_labels`.

## Choosing priors versus posterior as the parent source

- `parent_source="priors"`: use the Phase 1 parent presence map as the gate.
- `parent_source="posterior"`: use the combined parent posterior as the gate.

Use a positive `parent_threshold` to restrict refinement to locations with
stronger parent support. If no locations pass the threshold, `refine_group`
raises a clear `ValueError`.

This helper does not recursively refine multiple hierarchy levels
automatically, infer parent-child marker relationships, or cache results on
disk.

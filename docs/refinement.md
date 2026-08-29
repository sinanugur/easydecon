# Refining broad groups into subclusters

`ed.refine_group` refines one parent marker group, such as a broad myeloid
group, into subtype scores only where that parent group is supported.

First run a parent workflow:

```python
parent = ed.run_easydecon(
    sdata,
    markers_df=broad_markers,
    return_result_object=True,
    verbose=False,
)
```

## Parent gate

`parent_source="priors"`
: Use `parent.priors_df[parent_group]` as the gate.

`parent_source="posterior"`
: Use `parent.posterior_df[parent_group]` as the gate. This raises if
  `posterior_df` is `None`.

`parent_threshold` selects eligible spatial locations with
`parent_scores > parent_threshold`. If none pass, `refine_group` raises a
clear `ValueError`.

## Fast child refinement: mode="phase2"

```python
refined = ed.refine_group(
    sdata,
    parent_result=parent,
    parent_group="Myeloid",
    markers_df=myeloid_subcluster_markers,
    mode="phase2",
    method="wjaccard",
    parent_source="priors",
    parent_threshold=0.0,
)
```

This mode:

* uses the parent gate;
* reads/routs child markers;
* runs child Phase 2 only;
* maps child Phase 2 evidence to `conditional_df`; and
* does not run child Phase 1.

Because it does not calculate child priors, `phase2_candidate_pruning=True` is
not available in `mode="phase2"`. Use `parent_threshold` to narrow locations or
switch to `mode="full"`.

## Full child refinement: mode="full"

```python
refined = ed.refine_group(
    sdata,
    parent_result=parent,
    parent_group="Myeloid",
    markers_df=myeloid_subcluster_markers,
    mode="full",
    parent_source="priors",
    parent_threshold=0.0,
    filtering_algorithm="permutation",
    method="wjaccard",
    phase2_candidate_pruning=True,
)
```

This mode:

* uses the parent gate;
* runs a complete child `run_easydecon(..., return_result_object=True)`;
* calculates child Phase 1 priors;
* calculates child Phase 2 likelihoods;
* returns child posterior values in `conditional_df`; and
* can use child candidate pruning.

Full mode rejects list-style child `marker_genes` because it expects a child
`posterior_df`.

## Phase-specific refinement with UCell-like negative markers

UCell-like scoring can be used for child refinement when the child marker table
contains informative negative markers.

```python
refined = ed.refine_group(
    sdata,
    parent_result=parent,
    parent_group="Myeloid",
    markers_df=myeloid_role_markers,
    mode="full",
    marker_roles="phase_specific",
    filtering_algorithm="permutation",
    method="ucell",
    parent_source="priors",
    parent_threshold=0.0,
)
```

## Result fields

`RefinedGroupResult` contains:

`parent_scores`
: Parent prior or posterior scores aligned to the full spatial table.

`eligible_mask`
: Boolean mask of locations passing the parent threshold.

`conditional_df`
: Relative subtype support within eligible parent-positive locations.

`absolute_df`
: Conditional subtype values scaled by parent support.

```text
absolute_df = conditional_df * parent_scores
```

`assigned_labels`
: Hard subtype labels assigned from `absolute_df`.

`phase2_result`
: Child Phase 2 evidence aligned to the full spatial table.

`child_result`
: Full child `EasyDeconResult` in `mode="full"`, otherwise `None`.

`diagnostics`
: Parent gate, marker, role-routing, Phase 2, and assignment metadata.

The helper does not infer a recursive hierarchy or cache multi-level results on
disk.

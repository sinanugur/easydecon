# Refining broad groups into subclusters

`ed.refine_group` performs hierarchical refinement of one parent group into
subtypes. Its defaults run a complete child workflow using the final parent
posterior as the gate.

```python
prepared_myeloid_markers = ed.prepare_markers(
    filename="pelka_midlevel_myeloid.csv",
    source="midlevel_deseq",
    marker_role_inference="signed",
    marker_role_inference_log2fc_min=0.25,
    marker_roles="shared",
)

refined_myeloid = ed.refine_group(
    sdata=sdata,
    parent_result=result,
    parent_group="Myeloid",
    prepared_markers=prepared_myeloid_markers,
    bin_size=bin_size,
    table_key=f"square_00{bin_size}um",
)
```

By default, refinement:

1. uses the final parent posterior;
2. keeps locations with parent posterior greater than zero;
3. runs child Phase 1;
4. runs child UCell-like Phase 2;
5. calculates conditional child posterior probabilities;
6. scales conditional child probabilities by the parent signal for absolute
   subtype evidence; and
7. performs max assignment.

`absolute_df` is therefore:

```text
absolute_df = conditional_df * parent_scores
```

## Refinement defaults

The refinement profile intentionally uses a smaller adaptive marker range and
a stricter Phase 1 alpha than the top-level workflow:

```python
refined_myeloid = ed.refine_group(
    sdata=sdata,
    parent_result=result,
    parent_group="Myeloid",
    prepared_markers=prepared_myeloid_markers,
    top_n_genes="auto",
    log2fc_min=0.25,
    pval_cutoff=0.05,
    auto_marker_min=10,
    auto_marker_max=60,
    bin_size=bin_size,
    table_key=f"square_00{bin_size}um",
    mode="full",
    parent_source="posterior",
    parent_threshold=0.0,
    filtering_algorithm="permutation",
    phase1_output_stat="minus_log10_p",
    aggregation_method="coverage",
    alpha=0.01,
    permutation_gene_pool_fraction="auto",
    method="ucell",
    prior_weight=1.0,
    likelihood_weight=3.0,
    assign_method="max",
)
```

Every value above can be overridden for a specific child analysis.

## Optional fast child refinement

`mode="phase2"` remains available for fast child scoring without child Phase
1 priors. It defaults to the same refinement marker-selection and UCell
profile, but cannot use `phase2_candidate_pruning=True` because it does not
calculate child priors.

```python
refined = ed.refine_group(
    sdata=sdata,
    parent_result=result,
    parent_group="Myeloid",
    prepared_markers=prepared_myeloid_markers,
    mode="phase2",
)
```

Use `parent_source="priors"` or another supported Phase 2 method such as
`method="wjaccard"` only when that is an intentional alternative analysis.

## Result fields

`RefinedGroupResult` contains `parent_scores`, `eligible_mask`,
`conditional_df`, `absolute_df`, `assigned_labels`, `phase2_result`,
`child_result`, and `diagnostics`. `child_result` is the child
`EasyDeconResult` in `mode="full"` and is `None` in `mode="phase2"`.

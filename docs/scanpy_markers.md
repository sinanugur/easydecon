# Scanpy markers

easydecon can reuse existing Scanpy `rank_genes_groups` results or generate
them from an AnnData reference.

## Existing rank_genes_groups

If `adata.uns[marker_key]` exists, `read_markers_dataframe` reads it with
`scanpy.get.rank_genes_groups_df`. This is the `marker_method="auto"` behavior
when `adata` is supplied and the marker key already exists.

```python
markers_df = ed.read_markers_dataframe(
    sdata,
    adata=sc_adata,
    marker_key="rank_genes_groups",
    verbose=False,
)
```

## Generated Scanpy markers

Use `marker_method="scanpy"` to run `scanpy.tl.rank_genes_groups` when the
marker key is missing.

```python
result = ed.run_easydecon(
    sdata=sdata,
    adata=sc_adata,
    groupby="cell_type",
    marker_method="scanpy",
    scanpy_method="wilcoxon",
    filtering_algorithm="quantile",
    return_result_object=True,
    verbose=False,
)
```

Important parameters:

`groupby`
: Column in `adata.obs` used for marker groups.

`marker_key`
: Key in `adata.uns`; passed as `key` to direct `read_markers_dataframe`.

`scanpy_method`
: Differential-expression method passed to Scanpy.

`layer`, `use_raw`, `reference`
: Passed through to `scanpy.tl.rank_genes_groups`.

`copy_adata`
: When `True`, Scanpy marker generation runs on a copy. When `False`, the
  generated result is written into the input AnnData.

`rank_genes_groups_kwargs`
: Extra keyword arguments passed to Scanpy.

Scanpy marker generation expects expression values that are appropriate for the
chosen Scanpy method, commonly normalized and log-transformed data. easydecon
does not perform reference preprocessing for you.

## Signed Scanpy markers for UCell

`marker_role_inference="scanpy_signed"` is implemented and opt-in. It is useful
when Scanpy results contain signed `logfoldchanges` and you want UCell-like
Phase 2 scoring to use both positive markers and anti-markers.

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
    verbose=False,
)
```

Behavior:

* positive log fold changes become `positive` markers;
* negative log fold changes become `negative` markers;
* finite score signs, when present, must agree with fold-change direction;
* zero scores, zero or small effects, non-finite fold changes, and discordant
  rows are dropped;
* roles are not inferred from scores alone;
* existing explicit `marker_role` values are preserved;
* inference creates only `positive` and `negative` roles; and
* it does not create phase-specific `presence` or `identity` roles.

Use `marker_roles="shared"` for signed Scanpy inference. If you need
phase-specific `presence` and `identity` roles, provide a manual role table or
use `marker_method="reference"`.

## Top-N behavior

Direct `read_markers_dataframe` applies `top_n_genes` while standardizing the
Scanpy table. In `run_easydecon`, marker reading defers top-N selection and the
phase resolver applies `top_n_genes` per group, or per group and role when
roles exist.

## Reuse with PreparedMarkers

```python
prepared = ed.prepare_markers(
    sc_adata,
    marker_method="scanpy",
    groupby="cell_type",
    marker_role_inference="scanpy_signed",
    verbose=False,
)

result = ed.run_easydecon(
    sdata,
    prepared_markers=prepared,
    method="ucell",
    filtering_algorithm="quantile",
    return_result_object=True,
)
```

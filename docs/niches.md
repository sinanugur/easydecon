# Spatial niche detection

easydecon includes simple spatial niche utilities that cluster local
composition profiles derived from posterior-like matrices.

Public functions:

* `detect_spatial_niches_from_posteriors`
* `detect_niches_from_easydecon_result`
* `summarize_niche_compositions`
* `plot_niche_compositions`

## Inputs

`detect_spatial_niches_from_posteriors` accepts either a pandas DataFrame or an
EasyDeconResult-like object. If a result object has `posterior_df=None`, pass
`use_assignment_if_no_posterior=True` to use `assignment_df` instead.

Rows are aligned to `table.obs.index`. Spatial coordinates are read from
`table.obsm["spatial"]`, with a fallback to `table.obs[["x", "y"]]` when those
columns exist.

## Smoothing and clustering

`n_neighbors`
: Number of spatial nearest neighbors for optional local averaging.

`smooth`
: If `True`, cluster neighborhood-smoothed profiles.

`n_niches`
: Fixed number of niche clusters when `auto_n_niches=False`.

`auto_n_niches`, `n_niches_min`, `n_niches_max`, `selection_metric`
: Optional k selection using `silhouette` or `inertia`.

`random_state`
: Seed passed to k-means.

`add_to_obs`
: If `True`, writes labels to `table.obs[niches_column]`.

## Example

```python
niches, smoothed, diagnostics = ed.detect_niches_from_easydecon_result(
    sdata,
    result,
    n_neighbors=6,
    auto_n_niches=True,
    n_niches_min=2,
    n_niches_max=8,
    return_diagnostics=True,
)

composition = ed.summarize_niche_compositions(smoothed, niches)
fig, ax = ed.plot_niche_compositions(smoothed, niches)
```

`niches` is a DataFrame with one categorical label column. `smoothed` is the
matrix used for clustering. Niche IDs are categorical labels and do not imply
an ordering.

Spatial niche clustering is exploratory. It depends on posterior quality,
coordinate quality, neighborhood size, and the chosen number of clusters.

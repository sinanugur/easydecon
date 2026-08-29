# Visualizing easydecon results

These recipes use Matplotlib only. Every example aligns values to
`table.obs.index` before plotting.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import easydecon as ed

table = ed.get_table(sdata)
coords = np.asarray(table.obsm["spatial"])
```

If your platform's y-axis convention is image-like, call `ax.invert_yaxis()`
after plotting.

## Hard assignment map

```python
assignment_column = result.diagnostics.get(
    "results_column",
    result.assigned_labels.columns[0],
)
labels = result.assigned_labels[assignment_column].reindex(table.obs.index)

fig, ax = plt.subplots(figsize=(6, 6))
for label in labels.dropna().astype(str).unique():
    mask = labels.astype(str).eq(label).to_numpy()
    ax.scatter(coords[mask, 0], coords[mask, 1], s=8, label=label)

ax.set_title("easydecon assignments")
ax.set_aspect("equal")
ax.set_xlabel("Spatial x")
ax.set_ylabel("Spatial y")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
fig.tight_layout()
```

Unassigned locations are omitted here. Assignment counts are counts of spatial
units, not cell counts.

## Single-cell-type prior

```python
cell_type = "Myeloid"
values = result.priors_df[cell_type].reindex(table.obs.index).fillna(0.0)

fig, ax = plt.subplots(figsize=(6, 6))
points = ax.scatter(coords[:, 0], coords[:, 1], c=values.to_numpy(), s=8)
fig.colorbar(points, ax=ax, label="Phase 1 prior")
ax.set_title(f"{cell_type} Phase 1 prior")
ax.set_aspect("equal")
fig.tight_layout()
```

## Single-cell-type posterior

```python
cell_type = "Myeloid"

if result.posterior_df is None:
    raise ValueError("This workflow has no posterior_df; use assignment_df instead.")

values = result.posterior_df[cell_type].reindex(table.obs.index).fillna(0.0)

fig, ax = plt.subplots(figsize=(6, 6))
points = ax.scatter(coords[:, 0], coords[:, 1], c=values.to_numpy(), s=8)
fig.colorbar(points, ax=ax, label="Posterior support")
ax.set_title(f"{cell_type} posterior support")
ax.set_aspect("equal")
fig.tight_layout()
```

## Phase 1 versus posterior comparison

```python
def plot_spatial_values(table, values, title, colorbar_label):
    coords = np.asarray(table.obsm["spatial"])
    values = values.reindex(table.obs.index).fillna(0.0)

    fig, ax = plt.subplots(figsize=(6, 6))
    points = ax.scatter(coords[:, 0], coords[:, 1], c=values.to_numpy(), s=8)
    fig.colorbar(points, ax=ax, label=colorbar_label)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("Spatial x")
    ax.set_ylabel("Spatial y")
    fig.tight_layout()
    return fig, ax
```

```python
plot_spatial_values(table, result.priors_df["Myeloid"], "Myeloid prior", "Prior")
plot_spatial_values(
    table,
    result.posterior_df["Myeloid"],
    "Myeloid posterior",
    "Posterior support",
)
```

## Assignment counts

```python
assignment_column = result.diagnostics.get(
    "results_column",
    result.assigned_labels.columns[0],
)
counts = (
    result.assigned_labels[assignment_column]
    .dropna()
    .value_counts()
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(7, 4))
counts.plot.bar(ax=ax)
ax.set_title("Assigned spatial locations per group")
ax.set_xlabel("Marker group")
ax.set_ylabel("Number of spatial locations")
fig.tight_layout()
```

## Posterior heatmap

```python
if result.posterior_df is None:
    raise ValueError("This workflow has no posterior_df; use assignment_df instead.")

matrix = result.posterior_df.copy()
matrix = matrix.loc[matrix.max(axis=1).sort_values(ascending=False).index]
matrix = matrix.iloc[:100]

fig, ax = plt.subplots(figsize=(8, 5))
image = ax.imshow(matrix.to_numpy(), aspect="auto", interpolation="nearest")
ax.set_xticks(range(matrix.shape[1]))
ax.set_xticklabels(matrix.columns, rotation=90)
ax.set_ylabel("Spatial locations")
ax.set_title("Posterior support")
fig.colorbar(image, ax=ax, label="Posterior support")
fig.tight_layout()
```

## Maximum-posterior confidence map

```python
if result.posterior_df is None:
    raise ValueError("This workflow has no posterior_df; use assignment_df instead.")

confidence = result.posterior_df.max(axis=1).reindex(table.obs.index).fillna(0.0)

fig, ax = plt.subplots(figsize=(6, 6))
points = ax.scatter(coords[:, 0], coords[:, 1], c=confidence.to_numpy(), s=8)
fig.colorbar(points, ax=ax, label="Maximum posterior support")
ax.set_title("Assignment confidence")
ax.set_aspect("equal")
fig.tight_layout()
```

## Unassigned-location map

```python
assignment_column = result.diagnostics.get(
    "results_column",
    result.assigned_labels.columns[0],
)
labels = result.assigned_labels[assignment_column].reindex(table.obs.index)
unassigned = labels.isna().to_numpy()

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(coords[~unassigned, 0], coords[~unassigned, 1], s=6, alpha=0.3)
ax.scatter(coords[unassigned, 0], coords[unassigned, 1], s=10)
ax.set_title("Unassigned spatial locations")
ax.set_aspect("equal")
fig.tight_layout()
```

## Niche composition plot

```python
niches, smoothed = ed.detect_niches_from_easydecon_result(
    sdata,
    result,
    n_neighbors=6,
    n_niches=5,
)

fig, ax = ed.plot_niche_compositions(smoothed, niches)
```

Niche IDs are categorical labels and do not imply an ordering.

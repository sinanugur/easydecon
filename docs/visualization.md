# Visualizing easydecon results

These recipes use only Matplotlib and work with an AnnData table directly or a
table resolved from a SpatialData object.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import easydecon as ed

table = ed.get_table(sdata)
coords = np.asarray(table.obsm["spatial"])
```

The examples require spatial coordinates in `table.obsm["spatial"]`.

## Hard assignments

```python
table = ed.get_table(sdata)
coords = np.asarray(table.obsm["spatial"])

assignment_column = result.diagnostics.get(
    "results_column",
    result.assigned_labels.columns[0],
)
labels = result.assigned_labels[assignment_column].reindex(table.obs.index)

fig, ax = plt.subplots(figsize=(6, 6))

for label in labels.dropna().astype(str).unique():
    mask = labels.astype(str).eq(label).to_numpy()
    ax.scatter(
        coords[mask, 0],
        coords[mask, 1],
        s=8,
        label=label,
    )

ax.set_title("easydecon assignments")
ax.set_aspect("equal")
ax.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=False,
)
ax.set_xlabel("Spatial x")
ax.set_ylabel("Spatial y")
fig.tight_layout()
```

Use `ax.invert_yaxis()` only when required by the platform's coordinate
convention. Unassigned locations are omitted in this basic example.

## Posterior for one cell type

```python
cell_type = "Myeloid"

if result.posterior_df is None:
    raise ValueError(
        "This workflow has no posterior_df; use assignment_df instead."
    )

values = (
    result.posterior_df[cell_type]
    .reindex(table.obs.index)
    .fillna(0.0)
)

fig, ax = plt.subplots(figsize=(6, 6))
points = ax.scatter(
    coords[:, 0],
    coords[:, 1],
    c=values.to_numpy(),
    s=8,
)
fig.colorbar(
    points,
    ax=ax,
    label=f"{cell_type} posterior",
)
ax.set_title(f"{cell_type} spatial posterior")
ax.set_aspect("equal")
ax.set_xlabel("Spatial x")
ax.set_ylabel("Spatial y")
fig.tight_layout()
```

This retains uncertainty better than a hard assignment map. The same recipe can
visualize `priors_df`, `likelihoods_df`, or `phase2_result`.

## Compare Phase 1 and posterior

```python
def plot_spatial_values(table, values, title, colorbar_label):
    coords = np.asarray(table.obsm["spatial"])
    values = values.reindex(table.obs.index).fillna(0.0)

    fig, ax = plt.subplots(figsize=(6, 6))
    points = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=values.to_numpy(),
        s=8,
    )
    fig.colorbar(points, ax=ax, label=colorbar_label)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("Spatial x")
    ax.set_ylabel("Spatial y")
    fig.tight_layout()
    return fig, ax
```

```python
plot_spatial_values(
    table,
    result.priors_df["Myeloid"],
    "Myeloid Phase 1 prior",
    "Prior",
)

plot_spatial_values(
    table,
    result.posterior_df["Myeloid"],
    "Myeloid posterior",
    "Posterior",
)
```

The prior map shows where Myeloid is permitted by Phase 1. The posterior map
additionally includes Phase 2 marker similarity. A zero prior generally remains
zero in the posterior when `prior_weight > 0`.

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
ax.set_title("Assigned spatial locations per cell type")
ax.set_xlabel("Cell type")
ax.set_ylabel("Number of locations")
fig.tight_layout()
```

Counts are numbers of assigned spots, bins, or cells in the spatial table. They
are not estimated biological cell counts.

## Posterior heatmap

```python
matrix = result.posterior_df.copy()
matrix = matrix.loc[matrix.max(axis=1).sort_values(ascending=False).index]
matrix = matrix.iloc[:100]

fig, ax = plt.subplots(figsize=(8, 5))
image = ax.imshow(
    matrix.to_numpy(),
    aspect="auto",
    interpolation="nearest",
)
ax.set_xticks(range(matrix.shape[1]))
ax.set_xticklabels(matrix.columns, rotation=90)
ax.set_ylabel("Spatial locations")
ax.set_title("Posterior probabilities")
fig.colorbar(image, ax=ax, label="Posterior")
fig.tight_layout()
```

Limit the number of rows for readability. This is useful for comparing
ambiguous versus confident assignments.

## Niche compositions

```python
niches, smoothed = ed.detect_niches_from_easydecon_result(
    sdata,
    result,
    n_neighbors=6,
    n_niches=5,
)

fig, ax = ed.plot_niche_compositions(
    smoothed,
    niches,
)
```

Each stacked bar represents the mean cell-type composition of a niche. The
niche labels are categorical and do not imply an ordering.

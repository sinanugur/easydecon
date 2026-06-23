"""Visualize easydecon outputs with plain Matplotlib."""

import matplotlib.pyplot as plt
import numpy as np

import easydecon as ed

try:
    from ._synthetic import make_synthetic_spatial_and_markers, temporary_n_jobs
except ImportError:  # Support direct execution: python examples/...
    from _synthetic import make_synthetic_spatial_and_markers, temporary_n_jobs


def _table_and_coords(sdata, bin_size=8):
    table = ed.get_table(sdata, bin_size=bin_size)
    if "spatial" not in table.obsm:
        raise ValueError("Visualization requires spatial coordinates in table.obsm['spatial'].")
    return table, np.asarray(table.obsm["spatial"])


def _assignment_column(result):
    return result.diagnostics.get(
        "results_column",
        result.assigned_labels.columns[0],
    )


def plot_assignments(sdata, result, *, bin_size=8, assignment_column=None):
    """Plot hard assignment labels and omit unassigned locations."""
    table, coords = _table_and_coords(sdata, bin_size=bin_size)
    assignment_column = assignment_column or _assignment_column(result)
    labels = result.assigned_labels[assignment_column].reindex(table.obs.index)

    fig, ax = plt.subplots(figsize=(6, 6))
    for label in labels.dropna().astype(str).unique():
        mask = labels.astype(str).eq(label).to_numpy()
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=12,
            label=label,
        )

    ax.set_title("easydecon assignments")
    ax.set_aspect("equal")
    if labels.notna().any():
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False,
        )
    ax.set_xlabel("Spatial x")
    ax.set_ylabel("Spatial y")
    fig.tight_layout()
    return fig, ax


def plot_celltype_posterior(sdata, result, cell_type, *, bin_size=8):
    """Plot posterior support for one cell type."""
    if result.posterior_df is None:
        raise ValueError(
            "This workflow has no posterior_df; use assignment_df instead."
        )
    if cell_type not in result.posterior_df.columns:
        available = ", ".join(map(str, result.posterior_df.columns))
        raise ValueError(
            f"cell_type={cell_type!r} was not found in result.posterior_df. "
            f"Available cell types: {available}."
        )

    table, coords = _table_and_coords(sdata, bin_size=bin_size)
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
        s=12,
    )
    fig.colorbar(points, ax=ax, label=f"{cell_type} posterior")
    ax.set_title(f"{cell_type} spatial posterior")
    ax.set_aspect("equal")
    ax.set_xlabel("Spatial x")
    ax.set_ylabel("Spatial y")
    fig.tight_layout()
    return fig, ax


def plot_assignment_counts(result, *, assignment_column=None):
    """Plot numbers of assigned spatial units per label."""
    assignment_column = assignment_column or _assignment_column(result)
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
    return fig, ax


def main(
    show=True,
    return_outputs=False,
    *,
    n_spots=120,
    n_genes=40,
    n_celltypes=3,
):
    sdata, markers_df = make_synthetic_spatial_and_markers(
        n_spots=n_spots,
        n_genes=n_genes,
        n_celltypes=n_celltypes,
    )
    if "spatial" not in sdata.obsm:
        spot_number = np.arange(sdata.n_obs)
        width = int(np.ceil(np.sqrt(sdata.n_obs)))
        sdata.obsm["spatial"] = np.column_stack(
            (spot_number % width, spot_number // width)
        )

    with temporary_n_jobs(1):
        result = ed.run_easydecon(
            sdata=sdata,
            markers_df=markers_df,
            filtering_algorithm="quantile",
            method="wjaccard",
            return_result_object=True,
            verbose=False,
        )

    assignment_fig, _ = plot_assignments(sdata, result)
    cell_type = result.posterior_df.columns[0]
    posterior_fig, _ = plot_celltype_posterior(sdata, result, cell_type)
    counts_fig, _ = plot_assignment_counts(result)

    if show:
        plt.show()

    if return_outputs:
        return result, assignment_fig, posterior_fig, counts_fig


if __name__ == "__main__":
    main()

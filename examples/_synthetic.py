"""Deterministic synthetic inputs shared by the example and benchmark scripts."""

from contextlib import contextmanager
import math

import anndata as ad
import numpy as np
import pandas as pd


@contextmanager
def temporary_n_jobs(n_jobs=1):
    """Temporarily use a predictable worker count for lightweight examples."""
    import easydecon as ed
    from easydecon.config import config

    previous_n_jobs = config.n_jobs
    ed.set_n_jobs(n_jobs)
    try:
        yield
    finally:
        ed.set_n_jobs(previous_n_jobs)


def _validate_dimensions(n_spots, n_genes, n_celltypes):
    if n_spots < n_celltypes:
        raise ValueError("n_spots must be at least n_celltypes.")
    if n_genes < 2 * n_celltypes:
        raise ValueError("n_genes must provide at least two genes per cell type.")


def _marker_layout(n_genes, n_celltypes):
    markers_per_celltype = min(5, n_genes // n_celltypes)
    return [
        np.arange(group * markers_per_celltype, (group + 1) * markers_per_celltype)
        for group in range(n_celltypes)
    ]


def make_synthetic_spatial_and_markers(
    n_spots=200,
    n_genes=50,
    n_celltypes=3,
    seed=7,
):
    """Create a spatial AnnData and matching canonical marker DataFrame."""
    _validate_dimensions(n_spots, n_genes, n_celltypes)
    rng = np.random.default_rng(seed)
    marker_indices = _marker_layout(n_genes, n_celltypes)
    spot_groups = np.minimum(
        np.arange(n_spots) * n_celltypes // n_spots,
        n_celltypes - 1,
    )

    counts = rng.poisson(1.0, size=(n_spots, n_genes)).astype(float)
    for group, genes in enumerate(marker_indices):
        group_mask = spot_groups == group
        counts[np.ix_(group_mask, genes)] += rng.poisson(
            8.0, size=(int(group_mask.sum()), len(genes))
        ) + 2.0

    width = int(math.ceil(math.sqrt(n_spots)))
    spot_number = np.arange(n_spots)
    coordinates = np.column_stack((spot_number % width, spot_number // width))
    spatial = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(
            {"synthetic_cell_type": [f"CellType_{group}" for group in spot_groups]},
            index=[f"spot_{index}" for index in range(n_spots)],
        ),
        var=pd.DataFrame(index=[f"G{index}" for index in range(n_genes)]),
    )
    spatial.layers["counts"] = counts.copy()
    spatial.obsm["spatial"] = coordinates.astype(float)

    marker_rows = []
    for group, genes in enumerate(marker_indices):
        for rank, gene_index in enumerate(genes, start=1):
            marker_rows.append(
                {
                    "group": f"CellType_{group}",
                    "names": f"G{gene_index}",
                    "scores": float(len(genes) - rank + 1),
                    "logfoldchanges": float(3.0 - 0.1 * (rank - 1)),
                    "pvals_adj": float(1e-4 * rank),
                }
            )
    markers = pd.DataFrame(marker_rows)
    return spatial, markers


def make_synthetic_single_cell(
    n_cells=90,
    n_genes=50,
    n_celltypes=3,
    seed=17,
):
    """Create a normalized/log-transformed single-cell AnnData reference."""
    _validate_dimensions(n_cells, n_genes, n_celltypes)
    rng = np.random.default_rng(seed)
    marker_indices = _marker_layout(n_genes, n_celltypes)
    cell_groups = np.minimum(
        np.arange(n_cells) * n_celltypes // n_cells,
        n_celltypes - 1,
    )
    counts = rng.poisson(1.0, size=(n_cells, n_genes)).astype(float)
    for group, genes in enumerate(marker_indices):
        group_mask = cell_groups == group
        counts[np.ix_(group_mask, genes)] += rng.poisson(
            10.0, size=(int(group_mask.sum()), len(genes))
        ) + 3.0

    reference = ad.AnnData(
        X=np.log1p(counts),
        obs=pd.DataFrame(
            {
                "cell_type": pd.Categorical(
                    [f"CellType_{group}" for group in cell_groups]
                )
            },
            index=[f"cell_{index}" for index in range(n_cells)],
        ),
        var=pd.DataFrame(index=[f"G{index}" for index in range(n_genes)]),
    )
    reference.layers["counts"] = counts
    return reference


__all__ = [
    "make_synthetic_spatial_and_markers",
    "make_synthetic_single_cell",
    "temporary_n_jobs",
]

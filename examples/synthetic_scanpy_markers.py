"""Generate synthetic markers with Scanpy, then run easydecon."""

import easydecon as ed

try:
    from ._synthetic import (
        make_synthetic_single_cell,
        make_synthetic_spatial_and_markers,
        temporary_n_jobs,
    )
except ImportError:  # Support direct execution: python examples/...
    from _synthetic import (
        make_synthetic_single_cell,
        make_synthetic_spatial_and_markers,
        temporary_n_jobs,
    )


def main(
    return_outputs=False,
    *,
    n_spots=120,
    n_genes=50,
    n_celltypes=3,
    n_cells=90,
):
    sdata, _ = make_synthetic_spatial_and_markers(
        n_spots=n_spots,
        n_genes=n_genes,
        n_celltypes=n_celltypes,
    )
    sc_adata = make_synthetic_single_cell(
        n_cells=n_cells,
        n_genes=n_genes,
        n_celltypes=n_celltypes,
    )

    # Scanpy marker generation expects normalized/log-transformed expression.
    # For real DESeq-style pseudobulk markers, use marker_method="pydeseq2",
    # sample_col=..., and layer="counts".
    with temporary_n_jobs(1):
        result = ed.run_easydecon(
            sdata=sdata,
            adata=sc_adata,
            groupby="cell_type",
            marker_method="scanpy",
            filtering_algorithm="quantile",
            method="auc",
            return_result_object=True,
            verbose=False,
        )

    if return_outputs:
        return result
    print(ed.summarize_marker_table(result.markers_df))
    print(result.assigned_labels.head())


if __name__ == "__main__":
    main()

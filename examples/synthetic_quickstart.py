"""Run easydecon with a synthetic spatial AnnData and marker table."""

import easydecon as ed

try:
    from ._synthetic import make_synthetic_spatial_and_markers, temporary_n_jobs
except ImportError:  # Support direct execution: python examples/...
    from _synthetic import make_synthetic_spatial_and_markers, temporary_n_jobs


def main(
    return_outputs=False,
    *,
    n_spots=200,
    n_genes=50,
    n_celltypes=3,
):
    sdata, markers_df = make_synthetic_spatial_and_markers(
        n_spots=n_spots,
        n_genes=n_genes,
        n_celltypes=n_celltypes,
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
    summary = ed.summarize_easydecon_result(result, sdata=sdata)
    marker_summary = ed.summarize_marker_table(result.markers_df)

    if return_outputs:
        return result, summary, marker_summary
    print(summary)
    print(marker_summary)
    print(result.assigned_labels.head())


if __name__ == "__main__":
    main()

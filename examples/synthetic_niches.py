"""Detect spatial niches from a synthetic easydecon result."""

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
    niches, smoothed = ed.detect_niches_from_easydecon_result(
        sdata,
        result,
        n_neighbors=6,
        n_niches=3,
        smooth=True,
        add_to_obs=False,
    )
    composition = ed.summarize_niche_compositions(smoothed, niches)

    if return_outputs:
        return result, niches, smoothed, composition
    print(niches.head())
    print(composition)


if __name__ == "__main__":
    main()

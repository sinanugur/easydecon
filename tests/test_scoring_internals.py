import anndata as ad
import numpy as np
import pandas as pd
import pytest

from easydecon.config import config, set_batch_size, set_n_jobs
from easydecon.easydecon import (
    assign_clusters_from_df,
    common_markers_gene_expression_and_filter,
    function_row_mean,
    function_row_median,
    function_row_spearman,
    function_row_sum,
    function_row_weighted_jaccard,
    get_clusters_by_similarity_on_tissue,
)


@pytest.fixture(autouse=True)
def _single_thread_config():
    old_n_jobs = config.n_jobs
    old_batch_size = config.batch_size
    set_n_jobs(1)
    set_batch_size(64)
    try:
        yield
    finally:
        set_n_jobs(old_n_jobs)
        set_batch_size(old_batch_size)


@pytest.fixture
def spatial_table():
    return ad.AnnData(
        X=np.array(
            [
                [4.0, 0.0, 1.0],
                [0.0, 5.0, 1.0],
                [2.0, 2.0, 0.0],
            ]
        ),
        obs=pd.DataFrame(index=["spot_b", "spot_a", "spot_c"]),
        var=pd.DataFrame(index=["G1", "G2", "G3"]),
    )


@pytest.fixture
def alias_markers():
    return pd.DataFrame(
        {
            "cell_type": ["A", "A", "B"],
            "gene": ["G1", "G3", "G2"],
        }
    )


def test_get_clusters_by_similarity_preserves_obs_order(
    spatial_table, alias_markers
):
    spatial_table.obs["mask"] = [1, 0, 1]

    result = get_clusters_by_similarity_on_tissue(
        spatial_table,
        alias_markers,
        common_group_name="mask",
        celltype="cell_type",
        gene_id_column="gene",
        method="sum",
        verbose=False,
    )

    assert result.index.equals(spatial_table.obs.index)


def test_get_clusters_by_similarity_empty_mask_returns_zero_df(
    spatial_table, alias_markers
):
    spatial_table.obs["mask"] = 0

    result = get_clusters_by_similarity_on_tissue(
        spatial_table,
        alias_markers,
        common_group_name="mask",
        celltype="cell_type",
        gene_id_column="gene",
        method="sum",
        verbose=False,
    )

    assert result.index.equals(spatial_table.obs.index)
    assert result.columns.tolist() == ["A", "B"]
    assert (result == 0).all().all()


def test_get_clusters_by_similarity_accepts_alias_marker_columns(
    spatial_table, alias_markers
):
    result = get_clusters_by_similarity_on_tissue(
        spatial_table,
        alias_markers,
        celltype="cell_type",
        gene_id_column="gene",
        method="sum",
        verbose=False,
    )

    assert result.columns.tolist() == ["A", "B"]


def test_common_markers_gene_expression_accepts_alias_marker_columns(
    spatial_table, alias_markers
):
    result = common_markers_gene_expression_and_filter(
        spatial_table,
        alias_markers,
        celltype="cell_type",
        gene_id_column="gene",
        filtering_algorithm="quantile",
        add_to_obs=False,
        verbose=False,
    )

    assert result.columns.tolist() == ["A", "B"]


def test_function_row_sum_mean_median_missing_genes_do_not_raise():
    row = pd.Series({"G1": 2.0, "G2": 4.0})
    markers = pd.DataFrame(
        {"group": ["A", "A"], "names": ["G1", "MISSING"]}
    ).set_index("group", drop=False)

    summed = function_row_sum(row, markers, gene_id_column="names")["A"]
    mean = function_row_mean(row, markers, gene_id_column="names")["A"]
    median = function_row_median(row, markers, gene_id_column="names")["A"]

    assert summed == 2.0
    assert mean == 1.0
    assert median == 1.0
    assert np.isfinite([summed, mean, median]).all()


def test_assign_clusters_from_df_reindexes_missing_spots(spatial_table):
    scores = pd.DataFrame(
        {"A": [1.0, 0.0], "B": [0.0, 1.0]},
        index=["spot_b", "spot_a"],
    )

    result = assign_clusters_from_df(
        spatial_table,
        scores,
        results_column="assigned",
        add_to_obs=False,
    )

    assert result.index.equals(spatial_table.obs.index)
    assert pd.isna(result.loc["spot_c", "assigned"])


def test_function_row_spearman_nan_returns_zero():
    row = pd.Series({"G1": 1.0, "G2": 1.0})
    markers = pd.DataFrame(
        {
            "group": ["A", "A"],
            "names": ["G1", "G2"],
            "logfoldchanges": [1.0, 1.0],
        }
    ).set_index("group", drop=False)

    result = function_row_spearman(
        row,
        markers,
        gene_id_column="names",
        similarity_by_column="logfoldchanges",
    )

    assert result["A"] == 0.0


def test_weighted_jaccard_duplicate_marker_genes_does_not_raise():
    row = pd.Series({"G1": 3.0, "G2": 1.0})
    markers = pd.DataFrame(
        {
            "group": ["A", "A", "A"],
            "names": ["G1", "G1", "G2"],
            "logfoldchanges": [1.0, 2.0, 0.5],
        }
    ).set_index("group", drop=False)

    result = function_row_weighted_jaccard(
        row,
        markers,
        gene_id_column="names",
        weight_column="logfoldchanges",
    )

    assert np.isfinite(result["A"])

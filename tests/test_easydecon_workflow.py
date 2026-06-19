import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

import easydecon.extra as extra_module
from easydecon.config import config, set_batch_size, set_n_jobs


@pytest.fixture(autouse=True)
def _single_thread_config():
    old_n_jobs = config.n_jobs
    old_batch_size = config.batch_size
    set_n_jobs(1)
    set_batch_size(256)
    try:
        yield
    finally:
        set_n_jobs(old_n_jobs)
        set_batch_size(old_batch_size)


@pytest.fixture
def table_subset():
    sd = pytest.importorskip("spatialdata")
    sdata = sd.read_zarr("tests/data/sdata_test.zarr")
    return sdata.tables["square_008um"][:500, :].copy()


@pytest.fixture
def workflow_markers():
    markers_df = pd.read_csv("tests/data/test_workflow_markers.csv", index_col=0)
    if "group" not in markers_df.columns:
        if "group.1" in markers_df.columns:
            markers_df["group"] = markers_df["group.1"]
        else:
            markers_df["group"] = markers_df.index
    markers_df["group"] = markers_df["group"].astype(str)
    markers_df.index = markers_df["group"]
    return markers_df


def _sorted_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_index().sort_index(axis=1)


@pytest.fixture
def small_spatial_table():
    expression = np.array(
        [[8.0, 1.0, 4.0, 1.0]] * 6
        + [[1.0, 8.0, 1.0, 4.0]] * 6,
        dtype=float,
    )
    return ad.AnnData(
        X=expression,
        obs=pd.DataFrame(index=[f"spot_{index}" for index in range(12)]),
        var=pd.DataFrame(index=["G1", "G2", "G3", "G4"]),
    )


@pytest.fixture
def small_markers():
    return pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "names": ["G1", "G3", "G2", "G4"],
            "scores": [8.0, 4.0, 8.0, 4.0],
            "logfoldchanges": [2.0, 1.0, 2.0, 1.0],
            "pvals_adj": [0.001, 0.01, 0.001, 0.01],
        }
    )


def _small_single_cell_reference():
    expression = np.log1p(
        np.array(
            [[20.0, 1.0, 8.0, 1.0]] * 6
            + [[1.0, 20.0, 1.0, 8.0]] * 6,
            dtype=float,
        )
    )
    return ad.AnnData(
        X=expression,
        obs=pd.DataFrame(
            {"cell_type": pd.Categorical(["A"] * 6 + ["B"] * 6)},
            index=[f"cell_{index}" for index in range(12)],
        ),
        var=pd.DataFrame(index=["G1", "G2", "G3", "G4"]),
    )


def _small_pseudobulk_reference():
    rng = np.random.default_rng(7)
    rows = []
    cell_types = []
    sample_ids = []
    for sample_index, sample in enumerate(["S1", "S2", "S3"]):
        for cell_type in ["A", "B"]:
            for _ in range(5):
                means = (
                    [35 + sample_index, 2, 8, 3]
                    if cell_type == "A"
                    else [2, 35 + sample_index, 3, 8]
                )
                rows.append(rng.poisson(means))
                cell_types.append(cell_type)
                sample_ids.append(sample)
    counts = np.asarray(rows, dtype=int)
    reference = ad.AnnData(
        X=np.log1p(counts.astype(float)),
        obs=pd.DataFrame(
            {"cell_type": cell_types, "sample_id": sample_ids},
            index=[f"pb_cell_{index}" for index in range(len(rows))],
        ),
        var=pd.DataFrame(index=["G1", "G2", "G3", "G4"]),
    )
    reference.layers["counts"] = counts
    return reference


def test_easydecon_workflow_basic(table_subset, workflow_markers, monkeypatch):
    expected_phase1 = _sorted_df(pd.read_csv("tests/data/test_workflow_phase1.csv", index_col=0))
    expected_phase2 = _sorted_df(pd.read_csv("tests/data/test_workflow_phase2.csv", index_col=0))
    expected_priors = _sorted_df(pd.read_csv("tests/data/test_workflow_priors.csv", index_col=0))
    expected_posterior = _sorted_df(pd.read_csv("tests/data/test_workflow_posterior.csv", index_col=0))
    expected_assigned = pd.read_csv("tests/data/test_workflow_assigned_labels.csv", index_col=0)

    marker_genes = workflow_markers["names"].dropna().astype(str).unique()[:15].tolist()

    calls = {"phase1": 0, "phase2": 0}
    original_phase1 = extra_module.common_markers_gene_expression_and_filter
    original_phase2 = extra_module.get_clusters_by_similarity_on_tissue

    def wrapped_phase1(*args, **kwargs):
        calls["phase1"] += 1
        return original_phase1(*args, **kwargs)

    def wrapped_phase2(*args, **kwargs):
        calls["phase2"] += 1
        return original_phase2(*args, **kwargs)

    monkeypatch.setattr(extra_module, "common_markers_gene_expression_and_filter", wrapped_phase1)
    monkeypatch.setattr(extra_module, "get_clusters_by_similarity_on_tissue", wrapped_phase2)

    phase1_result, phase2_result, assigned_labels, priors_df, posterior_df = extra_module.easydecon_workflow(
        table_subset,
        markers_df=workflow_markers,
        marker_genes=marker_genes,
        filtering_algorithm="quantile",
        quantile=0.85,
        method="jaccard",
        results_column="easydecon_test",
        assign_method="max",
        top_n_genes=None,
        log2fc_min=-np.inf,
        pval_cutoff=1.0,
    )

    assert calls["phase1"] == 1
    assert calls["phase2"] == 1

    phase1_result = _sorted_df(phase1_result)
    phase2_result = _sorted_df(phase2_result)
    priors_df = _sorted_df(priors_df)
    posterior_df = _sorted_df(posterior_df)

    pd.testing.assert_frame_equal(phase1_result, expected_phase1, check_dtype=False, atol=1e-5, rtol=0)
    pd.testing.assert_frame_equal(phase2_result, expected_phase2, check_dtype=False, atol=1e-5, rtol=0)
    pd.testing.assert_frame_equal(priors_df, expected_priors, check_dtype=False, atol=1e-5, rtol=0)
    pd.testing.assert_frame_equal(posterior_df, expected_posterior, check_dtype=False, atol=1e-5, rtol=0)

    assigned = pd.to_numeric(assigned_labels.sort_index()["easydecon_test"], errors="coerce")
    expected = pd.to_numeric(expected_assigned.sort_index()["easydecon_test"], errors="coerce")
    pd.testing.assert_series_equal(assigned, expected, check_names=False, check_dtype=False)

    assert "MarkerGroup" in phase1_result.columns
    assert assigned_labels["easydecon_test"].notna().sum() > 0

    row_sums = priors_df.sum(axis=1)
    assert np.isclose(row_sums[row_sums > 0], 1.0).all()


def test_easydecon_workflow_old_tuple_return_with_markers_df(
    small_spatial_table, small_markers
):
    result = extra_module.easydecon_workflow(
        small_spatial_table,
        small_markers,
        filtering_algorithm="quantile",
        method="jaccard",
        verbose=False,
    )

    assert len(result) == 5
    assert isinstance(result[2], pd.DataFrame)


def test_easydecon_workflow_return_result_object(
    small_spatial_table, small_markers
):
    result = extra_module.easydecon_workflow(
        small_spatial_table,
        small_markers,
        filtering_algorithm="quantile",
        method="jaccard",
        return_result_object=True,
        verbose=False,
    )

    assert isinstance(result, extra_module.EasyDeconResult)
    for attribute in (
        "markers_df",
        "phase1_result",
        "phase2_result",
        "priors_df",
        "likelihoods_df",
        "assignment_df",
    ):
        assert isinstance(getattr(result, attribute), pd.DataFrame)
    assert {"group", "names"}.issubset(result.markers_df.columns)


def test_easydecon_workflow_with_adata_scanpy_markers(small_spatial_table):
    result = extra_module.easydecon_workflow(
        sdata=small_spatial_table,
        adata=_small_single_cell_reference(),
        groupby="cell_type",
        marker_method="scanpy",
        filtering_algorithm="quantile",
        method="jaccard",
        return_result_object=True,
        verbose=False,
    )

    assert not result.markers_df.empty
    assert result.posterior_df is not None
    assert set(result.posterior_df.columns) == set(result.markers_df["group"])


def test_easydecon_workflow_with_marker_genes_list_keeps_mask_workflow(
    small_spatial_table, small_markers
):
    result = extra_module.easydecon_workflow(
        small_spatial_table,
        small_markers,
        marker_genes=["G1", "G2"],
        filtering_algorithm="quantile",
        method="jaccard",
        return_result_object=True,
        verbose=False,
    )

    assert result.posterior_df is None
    assert result.assignment_df is result.phase2_result


def test_easydecon_workflow_return_diagnostics_tuple(
    small_spatial_table, small_markers
):
    result = extra_module.easydecon_workflow(
        small_spatial_table,
        small_markers,
        filtering_algorithm="quantile",
        method="jaccard",
        return_diagnostics=True,
        verbose=False,
    )

    assert len(result) == 6
    diagnostics = result[-1]
    assert "markers" in diagnostics
    assert "posterior_available" in diagnostics


def test_easydecon_workflow_with_pydeseq2_markers(small_spatial_table):
    pytest.importorskip("pydeseq2")
    result = extra_module.easydecon_workflow(
        sdata=small_spatial_table,
        adata=_small_pseudobulk_reference(),
        marker_method="pydeseq2",
        groupby="cell_type",
        sample_col="sample_id",
        layer="counts",
        min_cells_per_group=5,
        min_replicates_per_condition=2,
        deseq_n_cpus=1,
        filtering_algorithm="quantile",
        method="jaccard",
        return_result_object=True,
        verbose=False,
    )

    assert not result.markers_df.empty
    assert result.markers_df["marker_source"].eq("pydeseq2_pseudobulk").all()

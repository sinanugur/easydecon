import numpy as np
import pandas as pd
import pytest
import spatialdata as sd

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

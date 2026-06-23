import anndata as ad
import numpy as np
import pandas as pd
import pytest

import easydecon as ed
import easydecon.easydecon as easydecon_module
import easydecon.extra as extra_module
from easydecon.config import config, set_batch_size, set_n_jobs
from easydecon.markers import PreparedMarkers, prepare_markers, select_prepared_markers


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


def _single_cell_reference():
    counts = np.array(
        [[30, 2, 4, 1]] * 8
        + [[2, 30, 1, 4]] * 8,
        dtype=float,
    )
    reference = ad.AnnData(
        X=np.log1p(counts),
        obs=pd.DataFrame(
            {"cell_type": pd.Categorical(["A"] * 8 + ["B"] * 8)},
            index=[f"cell_{index}" for index in range(16)],
        ),
        var=pd.DataFrame(index=["G1", "G2", "G3", "G4"]),
    )
    reference.layers["counts"] = counts.copy()
    return reference


def _spatial_table(var_names=("G1", "G2", "G3", "G4")):
    var_names = list(var_names)
    gene_index = {gene: index for index, gene in enumerate(var_names)}
    expression = np.ones((8, len(var_names)), dtype=float)
    if "G1" in gene_index:
        expression[:4, gene_index["G1"]] = 12.0
    if "G2" in gene_index:
        expression[:4, gene_index["G2"]] = 8.0
    if "G3" in gene_index:
        expression[4:, gene_index["G3"]] = 12.0
    if "G4" in gene_index:
        expression[4:, gene_index["G4"]] = 8.0
    return ad.AnnData(
        X=expression,
        obs=pd.DataFrame(index=[f"spot_{index}" for index in range(8)]),
        var=pd.DataFrame(index=var_names),
    )


def _prepared_raw_markers():
    return pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "names": ["G1", "G2", "G3", "G4"],
            "scores": [9.0, 6.0, 9.0, 6.0],
            "logfoldchanges": [2.0, 1.5, 2.0, 1.5],
            "pvals_adj": [0.001, 0.01, 0.001, 0.01],
        }
    )


def _prepared_object():
    return PreparedMarkers(
        raw_markers_df=_prepared_raw_markers(),
        marker_method="scanpy",
        source="test_prepared",
        parameters={"groupby": "cell_type"},
        diagnostics={"source": "test"},
        signature="test-signature",
    )


def test_prepare_scanpy_markers_returns_unfiltered_markers():
    prepared = prepare_markers(
        _single_cell_reference(),
        marker_method="scanpy",
        groupby="cell_type",
        scanpy_method="t-test",
        verbose=False,
    )

    assert isinstance(prepared, PreparedMarkers)
    assert {"group", "names"}.issubset(prepared.raw_markers_df.columns)
    assert "G4" in set(prepared.raw_markers_df["names"])


def test_prepare_markers_signature_is_deterministic():
    adata = _single_cell_reference()

    first = prepare_markers(
        adata,
        marker_method="scanpy",
        groupby="cell_type",
        scanpy_method="t-test",
        verbose=False,
    )
    second = prepare_markers(
        adata,
        marker_method="scanpy",
        groupby="cell_type",
        scanpy_method="t-test",
        verbose=False,
    )

    assert first.signature == second.signature


def test_prepare_markers_signature_changes_with_groupby():
    adata = _single_cell_reference()
    changed = adata.copy()
    changed.obs["cell_type"] = pd.Categorical(
        ["A"] * 7 + ["B"] + ["B"] * 8
    )

    first = prepare_markers(
        adata,
        marker_method="scanpy",
        groupby="cell_type",
        scanpy_method="t-test",
        verbose=False,
    )
    second = prepare_markers(
        changed,
        marker_method="scanpy",
        groupby="cell_type",
        scanpy_method="t-test",
        verbose=False,
    )

    assert first.signature != second.signature


def test_prepare_markers_signature_changes_with_method_parameters():
    adata = _single_cell_reference()

    first = prepare_markers(
        adata,
        marker_method="scanpy",
        groupby="cell_type",
        scanpy_method="t-test",
        verbose=False,
    )
    second = prepare_markers(
        adata,
        marker_method="scanpy",
        groupby="cell_type",
        scanpy_method="wilcoxon",
        verbose=False,
    )

    assert first.signature != second.signature


def test_select_prepared_markers_filters_for_each_spatial_gene_universe():
    prepared = PreparedMarkers(
        raw_markers_df=pd.DataFrame(
            {
                "group": ["A", "A", "A"],
                "names": ["G1", "G2", "G3"],
                "scores": [3.0, 2.0, 1.0],
                "logfoldchanges": [1.0, 1.0, 1.0],
                "pvals_adj": [0.01, 0.01, 0.01],
            }
        ),
        marker_method="scanpy",
        source="prepared",
        parameters={},
        diagnostics={},
        signature="abc",
    )
    original = prepared.raw_markers_df.copy(deep=True)

    first = select_prepared_markers(prepared, ["G1", "G2"])
    second = select_prepared_markers(prepared, ["G2", "G3"])

    assert first["names"].tolist() == ["G1", "G2"]
    assert second["names"].tolist() == ["G2", "G3"]
    pd.testing.assert_frame_equal(prepared.raw_markers_df, original)


def test_read_markers_dataframe_uses_prepared_markers_without_generation(monkeypatch):
    prepared = _prepared_object()

    def fail_generation(*args, **kwargs):
        raise AssertionError("marker generation should not be called")

    monkeypatch.setattr(
        easydecon_module,
        "_generate_scanpy_rank_genes_groups",
        fail_generation,
    )
    monkeypatch.setattr(
        easydecon_module,
        "compute_pseudobulk_deseq_markers",
        fail_generation,
    )

    markers, diagnostics = easydecon_module.read_markers_dataframe(
        _spatial_table(),
        prepared_markers=prepared,
        return_diagnostics=True,
        verbose=False,
    )

    assert not markers.empty
    assert diagnostics["prepared_markers_used"] is True
    assert diagnostics["marker_signature"] == "test-signature"
    assert diagnostics["marker_generation_reused"] is True


def test_workflow_accepts_prepared_markers():
    prepared = _prepared_object()

    result = extra_module.easydecon_workflow(
        _spatial_table(),
        prepared_markers=prepared,
        filtering_algorithm="quantile",
        method="jaccard",
        min_markers=1,
        return_result_object=True,
        verbose=False,
    )

    assert not result.markers_df.empty
    assert result.prepared_markers is prepared
    assert result.diagnostics["markers"]["prepared_markers_used"] is True
    assert result.diagnostics["markers"]["marker_generation_reused"] is True


def test_same_prepared_markers_can_be_used_for_two_spatial_tables(monkeypatch):
    prepared = _prepared_object()

    def fail_generation(*args, **kwargs):
        raise AssertionError("marker generation should not be called")

    monkeypatch.setattr(
        easydecon_module,
        "_generate_scanpy_rank_genes_groups",
        fail_generation,
    )
    monkeypatch.setattr(
        easydecon_module,
        "compute_pseudobulk_deseq_markers",
        fail_generation,
    )

    first = ed.run_easydecon(
        _spatial_table(("G1", "G3")),
        prepared_markers=prepared,
        filtering_algorithm="quantile",
        method="jaccard",
        min_markers=1,
        return_result_object=True,
        verbose=False,
    )
    second = ed.run_easydecon(
        _spatial_table(("G2", "G4")),
        prepared_markers=prepared,
        filtering_algorithm="quantile",
        method="jaccard",
        min_markers=1,
        return_result_object=True,
        verbose=False,
    )

    assert first.prepared_markers is prepared
    assert second.prepared_markers is prepared
    assert first.markers_df["names"].tolist() == ["G1", "G3"]
    assert second.markers_df["names"].tolist() == ["G2", "G4"]

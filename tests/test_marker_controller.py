import importlib

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import easydecon as ed
import easydecon.extra as extra_module
import easydecon.refinement as refinement_module
from easydecon.config import config
from easydecon.markers import (
    PreparedMarkers,
    make_marker_table_signature,
    prepare_markers,
    select_prepared_markers,
)


def _markers():
    return pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "names": ["G1", "G2", "G3", "NOT_SPATIAL"],
            "logfoldchanges": [2.0, 1.0, 2.0, 3.0],
            "scores": [4.0, 2.0, 4.0, 5.0],
            "pvals_adj": [0.001, 0.01, 0.001, 0.001],
            "extra": ["a", "b", "c", "d"],
        }
    )


def _deseq_markers():
    return pd.DataFrame(
        {
            "cell_type": ["A", "A", "B"],
            "gene": ["G1", "G2", "G3"],
            "log2FoldChange": [2.0, -1.5, 3.0],
            "padj": [0.001, 0.01, 0.001],
            "stat": [6.0, -3.0, 7.0],
            "baseMean": [100.0, 50.0, 90.0],
        }
    )


def _spatial_table(var_names=("G1", "G2", "G3")):
    return ad.AnnData(
        X=np.ones((3, len(var_names))),
        obs=pd.DataFrame(index=["s1", "s2", "s3"]),
        var=pd.DataFrame(index=list(var_names)),
    )


def test_prepare_markers_accepts_dataframe_without_sdata():
    prepared = prepare_markers(markers_df=_markers(), verbose=False)

    assert isinstance(prepared, PreparedMarkers)
    assert prepared.diagnostics["input_kind"] == "dataframe"
    assert "NOT_SPATIAL" in prepared.raw_markers_df["names"].tolist()


def test_prepare_markers_accepts_deseq_alias_columns_and_preserves_values():
    prepared = prepare_markers(
        markers_df=_deseq_markers(),
        celltype="cell_type",
        gene_id_column="gene",
        source="deseq_table",
        verbose=False,
    )

    assert {"group", "names", "logfoldchanges", "pvals_adj", "scores"}.issubset(
        prepared.raw_markers_df.columns
    )
    assert prepared.source == "deseq_table"
    assert prepared.marker_method == "existing"
    assert -1.5 in prepared.raw_markers_df["logfoldchanges"].tolist()
    assert "baseMean" in prepared.raw_markers_df.columns


def test_prepare_markers_accepts_csv_and_excel(tmp_path):
    pytest.importorskip("openpyxl")
    csv_path = tmp_path / "markers.csv"
    xlsx_path = tmp_path / "markers.xlsx"
    _markers().to_csv(csv_path, index=False)
    _markers().to_excel(xlsx_path, index=False)

    csv_prepared = prepare_markers(filename=csv_path, verbose=False)
    xlsx_prepared = prepare_markers(filename=xlsx_path, verbose=False)

    assert csv_prepared.diagnostics["input_kind"] == "file"
    assert xlsx_prepared.diagnostics["input_kind"] == "file"
    assert csv_prepared.raw_markers_df["names"].tolist() == xlsx_prepared.raw_markers_df["names"].tolist()


def test_prepare_markers_prepared_input_is_idempotent(monkeypatch):
    prepared = prepare_markers(markers_df=_markers(), verbose=False)

    def fail(*args, **kwargs):
        raise AssertionError("lower-priority marker input should be ignored")

    monkeypatch.setattr("easydecon.markers._generate_scanpy_rank_genes_groups", fail)
    reused = prepare_markers(
        prepared_markers=prepared,
        markers_df=_deseq_markers(),
        adata=_spatial_table(),
        verbose=False,
    )

    assert reused is prepared
    assert reused.signature == prepared.signature


def test_prepare_markers_input_priority(tmp_path):
    csv_path = tmp_path / "markers.csv"
    _markers().assign(names=["G4", "G4", "G4", "G4"]).to_csv(csv_path, index=False)
    prepared = prepare_markers(markers_df=_markers(), verbose=False)

    assert prepare_markers(prepared_markers=prepared, markers_df=_deseq_markers(), filename=csv_path, verbose=False) is prepared
    dataframe_first = prepare_markers(markers_df=_deseq_markers(), filename=csv_path, celltype="cell_type", gene_id_column="gene", verbose=False)
    file_first = prepare_markers(filename=csv_path, adata=_spatial_table(), verbose=False)

    assert dataframe_first.diagnostics["input_kind"] == "dataframe"
    assert file_first.diagnostics["input_kind"] == "file"
    assert "G4" in file_first.raw_markers_df["names"].tolist()


def test_marker_table_signature_is_deterministic_and_value_sensitive():
    first = prepare_markers(markers_df=_markers(), verbose=False)
    second = prepare_markers(markers_df=_markers(), verbose=False)
    changed = _markers()
    changed.loc[0, "scores"] = 99.0
    third = prepare_markers(markers_df=changed, verbose=False)

    assert first.signature == second.signature
    assert first.signature != third.signature


def test_equivalent_file_and_dataframe_have_same_content_hash(tmp_path):
    path = tmp_path / "markers.csv"
    _markers().to_csv(path, index=False)

    from_df = prepare_markers(markers_df=_markers(), verbose=False)
    from_file = prepare_markers(filename=path, verbose=False)

    assert from_df.diagnostics["table_content_hash"] == from_file.diagnostics["table_content_hash"]
    assert make_marker_table_signature(
        from_df.raw_markers_df,
        from_df.marker_method,
        from_df.parameters,
    ) == from_df.signature


def test_select_prepared_markers_filters_spatial_genes_without_mutation():
    prepared = prepare_markers(markers_df=_markers(), verbose=False)
    original = prepared.raw_markers_df.copy(deep=True)

    selected = select_prepared_markers(
        prepared,
        gene_universe=_spatial_table().var_names,
        top_n_genes=None,
    )

    assert "NOT_SPATIAL" not in selected["names"].tolist()
    pd.testing.assert_frame_equal(prepared.raw_markers_df, original)


def test_select_prepared_markers_top_n_per_role_and_preserves_negative_sign():
    markers = _markers().iloc[:2].copy()
    markers["marker_role"] = ["negative", "negative"]
    markers["logfoldchanges"] = [-2.0, -1.0]
    prepared = prepare_markers(markers_df=markers, verbose=False)

    selected = select_prepared_markers(
        prepared,
        gene_universe=["G1", "G2"],
        top_n_genes=1,
        log2fc_min=0.5,
        pval_cutoff=1.0,
        sort_by_column="logfoldchanges",
    )

    assert selected["names"].tolist() == ["G1"]
    assert selected["logfoldchanges"].iloc[0] < 0


def test_workflow_does_not_call_read_and_returns_prepared(monkeypatch):
    monkeypatch.setattr(config, "n_jobs", 1)

    def fail_read(*args, **kwargs):
        raise AssertionError("workflow should not call read_markers_dataframe")

    monkeypatch.setattr("easydecon.easydecon.read_markers_dataframe", fail_read)

    result = ed.run_easydecon(
        _spatial_table(),
        markers_df=_markers(),
        filtering_algorithm="quantile",
        method="jaccard",
        min_markers=1,
        return_result_object=True,
        verbose=False,
    )

    assert isinstance(result.prepared_markers, PreparedMarkers)
    assert "NOT_SPATIAL" not in result.markers_df["names"].tolist()
    assert "NOT_SPATIAL" in result.prepared_markers.raw_markers_df["names"].tolist()


def test_phase2_refinement_does_not_call_read(monkeypatch):
    monkeypatch.setattr(config, "n_jobs", 1)

    def fail_read(*args, **kwargs):
        raise AssertionError("phase2 refinement should not call read_markers_dataframe")

    monkeypatch.setattr("easydecon.easydecon.read_markers_dataframe", fail_read)
    parent = extra_module.EasyDeconResult(
        markers_df=pd.DataFrame(),
        phase1_result=pd.DataFrame({"Parent": [1.0, 0.0, 1.0]}, index=["s1", "s2", "s3"]),
        phase2_result=pd.DataFrame({"Parent": [1.0, 0.0, 1.0]}, index=["s1", "s2", "s3"]),
        assigned_labels=pd.DataFrame({"easydecon": ["Parent", None, "Parent"]}, index=["s1", "s2", "s3"]),
        priors_df=pd.DataFrame({"Parent": [1.0, 0.0, 1.0]}, index=["s1", "s2", "s3"]),
        likelihoods_df=pd.DataFrame({"Parent": [1.0, 0.0, 1.0]}, index=["s1", "s2", "s3"]),
        posterior_df=pd.DataFrame({"Parent": [1.0, 0.0, 1.0]}, index=["s1", "s2", "s3"]),
        assignment_df=pd.DataFrame({"Parent": [1.0, 0.0, 1.0]}, index=["s1", "s2", "s3"]),
        diagnostics={"results_column": "easydecon"},
    )

    refined = ed.refine_group(
        _spatial_table(),
        parent,
        "Parent",
        markers_df=_markers(),
        method="jaccard",
        min_markers=1,
        verbose=False,
    )

    assert refined.diagnostics["child_phase2_ran"] is True


def test_imports_and_public_pseudobulk_reexport_are_stable():
    markers_module = importlib.import_module("easydecon.markers")
    easydecon_module = importlib.import_module("easydecon.easydecon")
    importlib.import_module("easydecon.extra")
    importlib.import_module("easydecon.refinement")

    assert ed.compute_pseudobulk_deseq_markers is markers_module.compute_pseudobulk_deseq_markers
    assert easydecon_module.compute_pseudobulk_deseq_markers is markers_module.compute_pseudobulk_deseq_markers
    assert not hasattr(extra_module, "read_markers_dataframe")
    assert not hasattr(refinement_module, "read_markers_dataframe")

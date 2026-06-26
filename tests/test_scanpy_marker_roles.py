import anndata as ad
import numpy as np
import pandas as pd
import pytest

import easydecon as ed
import easydecon.easydecon as edmod
import easydecon.extra as extra_module
import easydecon.refinement as refinement_module
from easydecon._schema import standardize_marker_dataframe
from easydecon.config import config
from easydecon.markers import (
    PreparedMarkers,
    infer_scanpy_signed_marker_roles,
    prepare_markers,
    resolve_phase_marker_tables,
)


def _spatial_table():
    return ad.AnnData(
        X=np.array(
            [
                [8.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 7.0, 1.0],
                [4.0, 0.0, 3.0, 0.0],
            ]
        ),
        obs=pd.DataFrame(index=["s1", "s2", "s3"]),
        var=pd.DataFrame(index=["G1", "G2", "G3", "G4"]),
    )


def _signed_markers():
    return pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "names": ["G1", "G2", "G3", "G4"],
            "logfoldchanges": [2.0, -2.0, 2.0, -2.0],
            "scores": [5.0, -5.0, 4.0, -4.0],
            "pvals_adj": [0.001, 0.001, 0.001, 0.001],
        }
    )


def _phase_specific_markers():
    markers = _signed_markers().copy()
    markers["logfoldchanges"] = markers["logfoldchanges"].abs()
    markers["scores"] = markers["scores"].abs()
    markers["marker_role"] = ["presence", "identity", "presence", "identity"]
    return markers


def test_signed_role_inference_positive_and_negative_preserves_signs():
    inferred, diagnostics = infer_scanpy_signed_marker_roles(_signed_markers())

    assert inferred["marker_role"].tolist() == ["positive", "negative", "positive", "negative"]
    assert inferred.loc[inferred["marker_role"] == "negative", "logfoldchanges"].lt(0).all()
    assert inferred.loc[inferred["marker_role"] == "negative", "scores"].lt(0).all()
    assert diagnostics["n_positive_inferred"] == 2
    assert diagnostics["n_negative_inferred"] == 2


def test_missing_score_uses_foldchange_direction():
    markers = _signed_markers().drop(columns="scores")

    inferred, _ = infer_scanpy_signed_marker_roles(markers)

    assert inferred["marker_role"].tolist() == ["positive", "negative", "positive", "negative"]


def test_score_sign_disagreement_zero_score_zero_foldchange_small_effect_and_nonfinite_are_dropped():
    markers = pd.DataFrame(
        {
            "group": ["A"] * 6,
            "names": ["G1", "G2", "G3", "G4", "G5", "G6"],
            "logfoldchanges": [1.0, -1.0, 1.0, 0.0, 0.1, np.inf],
            "scores": [-1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        }
    )

    inferred, diagnostics = infer_scanpy_signed_marker_roles(markers, log2fc_min=0.25)

    assert inferred.empty
    assert diagnostics["n_score_sign_discordant"] == 2
    assert diagnostics["n_zero_score"] == 1
    assert diagnostics["n_below_effect_threshold"] == 2
    assert diagnostics["n_nonfinite_logfoldchange"] == 1


def test_missing_logfoldchanges_raises():
    with pytest.raises(ValueError, match="requires a signed logfoldchanges column"):
        infer_scanpy_signed_marker_roles(
            pd.DataFrame({"group": ["A"], "names": ["G1"], "scores": [1.0]})
        )


def test_existing_marker_roles_are_preserved():
    markers = _signed_markers()
    markers["marker_role"] = ["presence", "identity", "negative", "positive"]

    inferred, diagnostics = infer_scanpy_signed_marker_roles(markers)

    assert inferred["marker_role"].tolist() == ["presence", "identity", "negative", "positive"]
    assert diagnostics["inference_applied"] is False
    assert diagnostics["existing_roles_preserved"] is True


def test_negative_role_logfoldchange_filter_and_score_sort_use_absolute_value():
    markers = pd.DataFrame(
        {
            "group": ["A", "A", "A"],
            "names": ["weak_neg", "strong_neg", "pos"],
            "logfoldchanges": [-0.5, -3.0, 1.0],
            "scores": [-2.0, -10.0, 1.0],
            "marker_role": ["negative", "negative", "positive"],
        }
    )

    result = standardize_marker_dataframe(
        markers,
        log2fc_min=1.0,
        pval_cutoff=1.0,
        sort_by_column="scores",
        top_n_genes=None,
    )

    assert "weak_neg" not in result["names"].tolist()
    assert result[result["marker_role"] == "negative"].iloc[0]["names"] == "strong_neg"
    assert result.loc["A", "scores"].min() < 0


def test_role_free_negative_scores_do_not_use_absolute_sort():
    markers = pd.DataFrame(
        {
            "group": ["A", "A"],
            "names": ["minus_ten", "minus_two"],
            "logfoldchanges": [1.0, 1.0],
            "scores": [-10.0, -2.0],
        }
    )

    result = standardize_marker_dataframe(
        markers,
        log2fc_min=0.0,
        pval_cutoff=1.0,
        sort_by_column="scores",
        ascending=False,
    )

    assert result["names"].tolist() == ["minus_two", "minus_ten"]


def test_top_n_with_roles_is_per_group_and_role():
    markers = pd.DataFrame(
        {
            "group": ["A"] * 4,
            "names": ["p1", "p2", "n1", "n2"],
            "logfoldchanges": [2.0, 1.5, -2.0, -1.5],
            "scores": [4.0, 3.0, -4.0, -3.0],
            "marker_role": ["positive", "positive", "negative", "negative"],
        }
    )

    phase1, phase2, _ = resolve_phase_marker_tables(
        standardize_marker_dataframe(markers, log2fc_min=0.0, pval_cutoff=1.0, top_n_genes=None),
        marker_roles="shared",
        method="ucell",
        top_n_genes=1,
    )

    assert phase1["names"].tolist() == ["p1"]
    assert phase2["names"].tolist() == ["p1", "n1"]


def test_shared_role_free_top_n_is_applied_in_phase_resolver():
    markers = standardize_marker_dataframe(
        pd.DataFrame(
            {
                "group": ["B", "B", "A", "A"],
                "names": ["B1", "B2", "A1", "A2"],
                "logfoldchanges": [4.0, 3.0, 4.0, 3.0],
                "scores": [4.0, 3.0, 4.0, 3.0],
            }
        ),
        top_n_genes=None,
        log2fc_min=0.0,
        pval_cutoff=1.0,
    )

    phase1, phase2, _ = resolve_phase_marker_tables(markers, top_n_genes=1)

    assert phase1["names"].tolist() == ["B1", "A1"]
    assert phase2["names"].tolist() == ["B1", "A1"]
    assert "marker_role" not in phase1.columns


def test_direct_scanpy_like_dataframe_inference_and_pval_after_inference():
    markers = _signed_markers()
    markers.loc[1, "pvals_adj"] = 0.9

    result = edmod.read_markers_dataframe(
        _spatial_table(),
        markers_df=markers,
        marker_role_inference="scanpy_signed",
        pval_cutoff=0.05,
        log2fc_min=0.25,
        top_n_genes=None,
        verbose=False,
    )

    assert "marker_role" in result.columns
    assert "G2" not in result["names"].tolist()
    assert set(result["marker_role"]) == {"positive", "negative"}


def test_scanpy_file_inference(tmp_path):
    path = tmp_path / "markers.csv"
    _signed_markers().to_csv(path, index=False)

    result = edmod.read_markers_dataframe(
        _spatial_table(),
        filename=path,
        marker_role_inference="scanpy_signed",
        top_n_genes=None,
        verbose=False,
    )

    assert result["marker_role"].value_counts().to_dict() == {"positive": 2, "negative": 2}


def test_existing_scanpy_anndata_inference(monkeypatch):
    adata = ad.AnnData(
        X=np.ones((4, 4)),
        obs=pd.DataFrame({"cell_type": ["A", "A", "B", "B"]}),
        var=pd.DataFrame(index=["G1", "G2", "G3", "G4"]),
    )
    adata.uns["rank_genes_groups"] = {"mock": True}
    monkeypatch.setattr(edmod.sc.get, "rank_genes_groups_df", lambda *args, **kwargs: _signed_markers())

    result = edmod.read_markers_dataframe(
        _spatial_table(),
        adata=adata,
        marker_method="existing",
        marker_role_inference="scanpy_signed",
        top_n_genes=None,
        verbose=False,
    )

    assert set(result["marker_role"]) == {"positive", "negative"}


def test_reference_and_pydeseq_generation_reject_scanpy_inference(monkeypatch):
    adata = ad.AnnData(
        X=np.ones((4, 4)),
        obs=pd.DataFrame({"cell_type": ["A", "A", "B", "B"], "sample": ["s1", "s2", "s1", "s2"]}),
        var=pd.DataFrame(index=["G1", "G2", "G3", "G4"]),
    )

    with pytest.raises(ValueError, match="intended for Scanpy-style"):
        edmod.read_markers_dataframe(
            _spatial_table(),
            adata=adata,
            marker_method="reference",
            marker_role_inference="scanpy_signed",
            groupby="cell_type",
            verbose=False,
        )
    with pytest.raises(ValueError, match="intended for Scanpy-style"):
        edmod.read_markers_dataframe(
            _spatial_table(),
            adata=adata,
            marker_method="pydeseq2",
            marker_role_inference="scanpy_signed",
            groupby="cell_type",
            sample_col="sample",
            verbose=False,
        )


def test_direct_read_top_n_behavior_is_preserved_and_inferred_top_n_is_per_role():
    result = edmod.read_markers_dataframe(
        _spatial_table(),
        markers_df=_signed_markers(),
        marker_role_inference="scanpy_signed",
        top_n_genes=1,
        verbose=False,
    )

    assert result.groupby([result["group"], result["marker_role"]]).size().max() == 1


def test_prepare_scanpy_markers_with_signed_roles_and_signature(monkeypatch):
    adata = ad.AnnData(
        X=np.ones((4, 4)),
        obs=pd.DataFrame({"cell_type": ["A", "A", "B", "B"]}),
        var=pd.DataFrame(index=["G1", "G2", "G3", "G4"]),
    )
    adata.uns["rank_genes_groups"] = {"mock": True}
    monkeypatch.setattr(edmod.sc.get, "rank_genes_groups_df", lambda *args, **kwargs: _signed_markers())

    plain = prepare_markers(adata, marker_method="existing", groupby="cell_type", verbose=False)
    inferred = prepare_markers(
        adata,
        marker_method="existing",
        groupby="cell_type",
        marker_role_inference="scanpy_signed",
        verbose=False,
    )

    assert "marker_role" in inferred.raw_markers_df.columns
    assert inferred.signature != plain.signature
    assert inferred.diagnostics["marker_role_counts"] == {"positive": 2, "negative": 2}


def test_prepared_without_roles_rejects_late_inference_and_with_roles_accepts(monkeypatch):
    adata = ad.AnnData(
        X=np.ones((4, 4)),
        obs=pd.DataFrame({"cell_type": ["A", "A", "B", "B"]}),
        var=pd.DataFrame(index=["G1", "G2", "G3", "G4"]),
    )
    adata.uns["rank_genes_groups"] = {"mock": True}
    monkeypatch.setattr(edmod.sc.get, "rank_genes_groups_df", lambda *args, **kwargs: _signed_markers())
    plain = prepare_markers(adata, marker_method="existing", groupby="cell_type", verbose=False)
    inferred = prepare_markers(
        adata,
        marker_method="existing",
        groupby="cell_type",
        marker_role_inference="scanpy_signed",
        verbose=False,
    )

    with pytest.raises(ValueError, match="Recreate it with marker_role_inference"):
        edmod.read_markers_dataframe(
            _spatial_table(),
            prepared_markers=plain,
            marker_role_inference="scanpy_signed",
            verbose=False,
        )
    accepted = edmod.read_markers_dataframe(
        _spatial_table(),
        prepared_markers=inferred,
        marker_role_inference="scanpy_signed",
        verbose=False,
    )
    assert "marker_role" in accepted.columns


def test_workflow_always_reads_with_top_n_none(monkeypatch):
    select_calls = []
    prepare_calls = []

    def fake_prepare(*args, **kwargs):
        prepare_calls.append(kwargs)
        return PreparedMarkers(
            raw_markers_df=standardize_marker_dataframe(
                _phase_specific_markers(),
                log2fc_min=0.0,
                pval_cutoff=1.0,
                top_n_genes=None,
            ),
            marker_method="existing",
            source="fake",
            diagnostics={
                "source": "fake",
                "input_kind": "dataframe",
                "marker_method": "existing",
                "marker_role_inference": {"mode": "none", "requested": False, "applied": False},
            },
            signature="fake",
        )

    def fake_select(prepared, *args, **kwargs):
        select_calls.append(kwargs["top_n_genes"])
        selected = prepared.raw_markers_df.copy()
        if kwargs.get("return_diagnostics"):
            return selected, {
                "source": prepared.source,
                "n_selected_markers": int(selected.shape[0]),
                "marker_counts_per_group": selected.groupby(selected["group"]).size().to_dict(),
            }
        return selected

    monkeypatch.setattr(extra_module, "prepare_markers", fake_prepare)
    monkeypatch.setattr(extra_module, "select_prepared_markers", fake_select)
    monkeypatch.setattr(
        extra_module,
        "common_markers_gene_expression_and_filter",
        lambda *args, **kwargs: pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]}, index=["s1", "s2"]),
    )
    monkeypatch.setattr(
        extra_module,
        "get_clusters_by_similarity_on_tissue",
        lambda *args, **kwargs: pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]}, index=["s1", "s2"]),
    )
    table = ad.AnnData(
        X=np.ones((2, 4)),
        obs=pd.DataFrame(index=["s1", "s2"]),
        var=pd.DataFrame(index=["G1", "G2", "G3", "G4"]),
    )

    extra_module.easydecon_workflow(table, markers_df=_signed_markers(), marker_roles="shared", verbose=False)
    extra_module.easydecon_workflow(table, markers_df=_signed_markers(), marker_roles="phase_specific", verbose=False)

    assert len(prepare_calls) == 2
    assert select_calls == [None, None]


def test_scanpy_signed_inference_phase_specific_raises_helpful_error():
    with pytest.raises(ValueError, match="positive and negative roles only"):
        ed.run_easydecon(
            _spatial_table(),
            markers_df=_signed_markers(),
            marker_role_inference="scanpy_signed",
            marker_roles="phase_specific",
            filtering_algorithm="quantile",
            verbose=False,
        )


def test_scanpy_ucell_signed_inference_routes_negative_markers(monkeypatch):
    monkeypatch.setattr(config, "n_jobs", 1)
    result = ed.run_easydecon(
        _spatial_table(),
        markers_df=_signed_markers(),
        marker_role_inference="scanpy_signed",
        marker_roles="shared",
        method="ucell",
        filtering_algorithm="quantile",
        min_markers=1,
        log2fc_min=0.25,
        pval_cutoff=1.0,
        return_result_object=True,
        verbose=False,
    )

    assert set(result.markers_df["marker_role"]) == {"positive", "negative"}
    assert "negative" in result.diagnostics["marker_roles"]["phase2_roles"]
    assert result.diagnostics["markers"]["marker_role_inference"]["applied"] is True


def test_scanpy_signed_default_none_excludes_negative_rows():
    result = edmod.read_markers_dataframe(
        _spatial_table(),
        markers_df=_signed_markers(),
        marker_role_inference="none",
        top_n_genes=None,
        pval_cutoff=1.0,
        verbose=False,
    )

    assert "marker_role" not in result.columns
    assert result["logfoldchanges"].gt(0).all()


def test_phase2_refinement_forwards_marker_role_inference(monkeypatch):
    captured = {}

    def fake_prepare(*args, **kwargs):
        captured.update(kwargs)
        return PreparedMarkers(
            raw_markers_df=standardize_marker_dataframe(
                _signed_markers(),
                log2fc_min=-np.inf,
                pval_cutoff=1.0,
                top_n_genes=None,
            ),
            marker_method="existing",
            source="fake",
            diagnostics={
                "source": "fake",
                "input_kind": "dataframe",
                "marker_method": "existing",
                "marker_role_inference": {"mode": "scanpy_signed", "requested": True, "applied": True},
            },
            signature="fake",
        )

    def fake_select(prepared, *args, **kwargs):
        captured["selection_top_n_genes"] = kwargs["top_n_genes"]
        selected = prepared.raw_markers_df.copy()
        if kwargs.get("return_diagnostics"):
            return selected, {
                "source": prepared.source,
                "n_selected_markers": int(selected.shape[0]),
                "marker_counts_per_group": selected.groupby(selected["group"]).size().to_dict(),
            }
        return selected

    monkeypatch.setattr(refinement_module, "prepare_markers", fake_prepare)
    monkeypatch.setattr(refinement_module, "select_prepared_markers", fake_select)
    monkeypatch.setattr(
        refinement_module,
        "get_clusters_by_similarity_on_tissue",
        lambda *args, **kwargs: pd.DataFrame({"A": [1.0]}, index=["s1"]),
    )
    parent = extra_module.EasyDeconResult(
        markers_df=pd.DataFrame(),
        phase1_result=pd.DataFrame({"Parent": [1.0]}, index=["s1"]),
        phase2_result=pd.DataFrame({"Parent": [1.0]}, index=["s1"]),
        assigned_labels=pd.DataFrame({"easydecon": ["Parent"]}, index=["s1"]),
        priors_df=pd.DataFrame({"Parent": [1.0]}, index=["s1"]),
        likelihoods_df=pd.DataFrame({"Parent": [1.0]}, index=["s1"]),
        posterior_df=pd.DataFrame({"Parent": [1.0]}, index=["s1"]),
        assignment_df=pd.DataFrame({"Parent": [1.0]}, index=["s1"]),
        diagnostics={"results_column": "easydecon"},
    )
    table = ad.AnnData(
        X=np.ones((1, 4)),
        obs=pd.DataFrame(index=["s1"]),
        var=pd.DataFrame(index=["G1", "G2", "G3", "G4"]),
    )

    ed.refine_group(
        table,
        parent,
        "Parent",
        markers_df=_signed_markers(),
        marker_role_inference="scanpy_signed",
        method="ucell",
        verbose=False,
    )

    assert captured["marker_role_inference"] == "scanpy_signed"
    assert captured["selection_top_n_genes"] is None

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import easydecon as ed
from easydecon.config import config
from easydecon.markers import (
    _normalize_marker_role_inference,
    infer_signed_marker_roles,
    prepare_markers,
)


def _deseq_table():
    return pd.DataFrame(
        {
            "cell_type": ["A", "A", "B", "B"],
            "gene": ["A_POS", "A_NEG", "B_POS", "B_NEG"],
            "log2FoldChange": [1.5, -1.5, 2.0, -2.0],
            "padj": [0.01, 0.01, 0.02, 0.02],
            "stat": [4.0, -4.0, 3.0, -3.0],
        }
    )


def _pseudobulk_adata():
    return ad.AnnData(
        X=np.ones((4, 4)),
        obs=pd.DataFrame(
            {
                "cell_type": ["A", "A", "B", "B"],
                "sample_id": ["s1", "s2", "s1", "s2"],
            }
        ),
        var=pd.DataFrame(index=["A_POS", "A_NEG", "B_POS", "B_NEG"]),
    )


def test_signed_mode_and_legacy_alias_normalize_to_signed():
    assert _normalize_marker_role_inference("none") == "none"
    assert _normalize_marker_role_inference("signed") == "signed"
    assert _normalize_marker_role_inference("scanpy_signed") == "signed"
    with pytest.raises(ValueError, match="marker_role_inference must be one of"):
        _normalize_marker_role_inference("not-a-mode")


def test_direct_deseq_dataframe_infers_roles_and_preserves_signed_stat():
    prepared = prepare_markers(
        markers_df=_deseq_table(),
        marker_role_inference="signed",
        marker_role_inference_log2fc_min=0.25,
        verbose=False,
    )

    markers = prepared.raw_markers_df
    assert {"group", "names", "logfoldchanges", "pvals_adj", "scores", "stat", "marker_role"}.issubset(markers.columns)
    assert markers.loc[markers["names"] == "A_POS", "marker_role"].item() == "positive"
    assert markers.loc[markers["names"] == "A_NEG", "marker_role"].item() == "negative"
    assert markers.loc[markers["names"] == "A_NEG", "logfoldchanges"].item() == -1.5
    assert markers.loc[markers["names"] == "A_NEG", "stat"].item() == -4.0
    inference = prepared.diagnostics["marker_role_inference"]
    assert inference["mode"] == "signed"
    assert inference["direction_source"] == "logfoldchanges"
    assert inference["directional_score_column"] == "stat"
    assert inference["directional_score_used"] is True


def test_signed_lfc_rules_and_directional_score_concordance():
    markers = pd.DataFrame(
        {
            "group": ["A"] * 7,
            "names": ["positive", "negative", "zero", "small", "nan", "discordant", "zero_stat"],
            "logfoldchanges": [1.0, -1.0, 0.0, 0.2, np.inf, 1.0, -1.0],
            "stat": [2.0, -2.0, 1.0, 1.0, 1.0, -2.0, 0.0],
        }
    )

    inferred, diagnostics = infer_signed_marker_roles(markers, log2fc_min=0.25)

    assert inferred["names"].tolist() == ["positive", "negative"]
    assert inferred["marker_role"].tolist() == ["positive", "negative"]
    assert diagnostics["n_nonfinite_logfoldchange"] == 1
    assert diagnostics["n_below_effect_threshold"] == 2
    assert diagnostics["n_score_sign_discordant"] == 1
    assert diagnostics["n_zero_directional_score"] == 1


def test_missing_or_nonfinite_directional_scores_use_lfc_only_and_basemean_is_unsigned():
    no_stat = _deseq_table().drop(columns="stat")
    inferred, diagnostics = infer_signed_marker_roles(no_stat)
    assert set(inferred["marker_role"]) == {"positive", "negative"}
    assert diagnostics["directional_score_column"] is None
    assert diagnostics["directional_score_used"] is False

    nonfinite_stat = _deseq_table()
    nonfinite_stat["stat"] = np.nan
    inferred, _ = infer_signed_marker_roles(nonfinite_stat)
    assert set(inferred["marker_role"]) == {"positive", "negative"}

    unsigned = _deseq_table().drop(columns="stat")
    unsigned["baseMean"] = [-10.0, 10.0, -10.0, 10.0]
    inferred, diagnostics = infer_signed_marker_roles(unsigned)
    assert set(inferred["marker_role"]) == {"positive", "negative"}
    assert diagnostics["directional_score_column"] is None


def test_existing_roles_are_preserved_without_reinference():
    markers = _deseq_table()
    markers["marker_role"] = ["presence", "identity", "negative", "positive"]

    inferred, diagnostics = infer_signed_marker_roles(markers)

    assert inferred["marker_role"].tolist() == markers["marker_role"].tolist()
    assert diagnostics["inference_applied"] is False
    assert diagnostics["existing_roles_preserved"] is True


def test_deseq_file_matches_dataframe(tmp_path):
    table = _deseq_table()
    path = tmp_path / "markers.csv"
    table.to_csv(path, index=False)

    dataframe_prepared = prepare_markers(
        markers_df=table, marker_role_inference="signed", verbose=False
    )
    file_prepared = prepare_markers(
        filename=path, marker_role_inference="signed", verbose=False
    )

    pd.testing.assert_frame_equal(
        dataframe_prepared.raw_markers_df.reset_index(drop=True),
        file_prepared.raw_markers_df.reset_index(drop=True),
    )


def test_deseq_excel_file_matches_dataframe(tmp_path):
    pytest.importorskip("openpyxl")
    table = _deseq_table()
    path = tmp_path / "markers.xlsx"
    table.to_excel(path, index=False)

    expected = prepare_markers(
        markers_df=table, marker_role_inference="signed", verbose=False
    )
    actual = prepare_markers(
        filename=path, marker_role_inference="signed", verbose=False
    )

    pd.testing.assert_frame_equal(
        expected.raw_markers_df.reset_index(drop=True),
        actual.raw_markers_df.reset_index(drop=True),
    )


def test_alias_inputs_have_equivalent_prepared_signatures():
    signed = prepare_markers(
        markers_df=_deseq_table(), marker_role_inference="signed", verbose=False
    )
    alias = prepare_markers(
        markers_df=_deseq_table(),
        marker_role_inference="scanpy_signed",
        verbose=False,
    )

    pd.testing.assert_frame_equal(signed.raw_markers_df, alias.raw_markers_df)
    assert signed.signature == alias.signature
    assert alias.parameters["marker_role_inference"] == "signed"
    assert alias.diagnostics["marker_role_inference"]["requested_mode"] == "scanpy_signed"
    assert alias.diagnostics["marker_role_inference"]["mode"] == "signed"


def test_generated_pydeseq2_supports_signed_inference_without_changing_scores(monkeypatch):
    generated = _deseq_table().rename(columns={"cell_type": "group", "gene": "names"})
    generated["scores"] = generated["stat"].abs()

    def fake_generation(*args, **kwargs):
        return generated.copy(), {"mocked": True}

    monkeypatch.setattr(
        "easydecon.markers.compute_pseudobulk_deseq_markers", fake_generation
    )
    prepared = prepare_markers(
        adata=_pseudobulk_adata(),
        marker_method="pydeseq2",
        groupby="cell_type",
        sample_col="sample_id",
        marker_role_inference="signed",
        verbose=False,
    )

    markers = prepared.raw_markers_df
    assert set(markers["marker_role"]) == {"positive", "negative"}
    assert markers.loc[markers["names"] == "A_NEG", "logfoldchanges"].item() == -1.5
    assert markers.loc[markers["names"] == "A_NEG", "stat"].item() == -4.0
    assert markers.loc[markers["names"] == "A_NEG", "scores"].item() == 4.0
    assert prepared.diagnostics["marker_role_inference"]["directional_score_column"] == "stat"


def test_generated_pydeseq2_without_stat_uses_lfc_only(monkeypatch):
    generated = _deseq_table().drop(columns="stat").rename(
        columns={"cell_type": "group", "gene": "names"}
    )
    generated["scores"] = -np.log10(generated["padj"])

    monkeypatch.setattr(
        "easydecon.markers.compute_pseudobulk_deseq_markers",
        lambda *args, **kwargs: (generated.copy(), {"mocked": True}),
    )
    prepared = prepare_markers(
        adata=_pseudobulk_adata(),
        marker_method="pydeseq2",
        groupby="cell_type",
        sample_col="sample_id",
        marker_role_inference="signed",
        verbose=False,
    )

    inference = prepared.diagnostics["marker_role_inference"]
    assert set(prepared.raw_markers_df["marker_role"]) == {"positive", "negative"}
    assert inference["directional_score_column"] is None
    assert inference["directional_score_used"] is False


def test_generated_pydeseq2_without_inference_keeps_historical_scores(monkeypatch):
    generated = _deseq_table().rename(columns={"cell_type": "group", "gene": "names"})
    generated["scores"] = generated["stat"].abs()
    monkeypatch.setattr(
        "easydecon.markers.compute_pseudobulk_deseq_markers",
        lambda *args, **kwargs: (generated.copy(), {"mocked": True}),
    )

    prepared = prepare_markers(
        adata=_pseudobulk_adata(),
        marker_method="pydeseq2",
        groupby="cell_type",
        sample_col="sample_id",
        marker_role_inference="none",
        verbose=False,
    )

    assert "marker_role" not in prepared.raw_markers_df.columns
    assert prepared.raw_markers_df["scores"].tolist() == generated["scores"].tolist()


def test_signed_inference_rejects_phase_specific_only_when_it_is_applied():
    with pytest.raises(ValueError, match="positive and negative roles only"):
        prepare_markers(
            markers_df=_deseq_table(),
            marker_role_inference="signed",
            marker_roles="phase_specific",
            verbose=False,
        )


def test_signed_deseq_roles_route_to_ucell_and_non_ucell_workflows(monkeypatch):
    monkeypatch.setattr(config, "n_jobs", 1)
    spatial = ad.AnnData(
        X=np.array(
            [
                [8.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 8.0, 1.0],
                [4.0, 0.0, 4.0, 0.0],
            ]
        ),
        obs=pd.DataFrame(index=["s1", "s2", "s3"]),
        var=pd.DataFrame(index=["A_POS", "A_NEG", "B_POS", "B_NEG"]),
    )
    common_kwargs = {
        "markers_df": _deseq_table(),
        "marker_role_inference": "signed",
        "marker_roles": "shared",
        "filtering_algorithm": "quantile",
        "min_markers": 1,
        "top_n_genes": None,
        "pval_cutoff": 1.0,
        "return_result_object": True,
        "verbose": False,
    }

    ucell = ed.run_easydecon(spatial.copy(), method="ucell", **common_kwargs)
    weighted = ed.run_easydecon(spatial.copy(), method="wjaccard", **common_kwargs)

    assert set(ucell.prepared_markers.raw_markers_df["marker_role"]) == {
        "positive",
        "negative",
    }
    assert "negative" in ucell.diagnostics["marker_roles"]["phase2_roles"]
    assert set(weighted.markers_df["marker_role"]) == {"positive"}

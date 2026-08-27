from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

import easydecon as ed
import easydecon.extra as extra_module
import easydecon.refinement as refinement_module
from easydecon.config import config, set_n_jobs
from easydecon.markers import PreparedMarkers, select_prepared_markers


@pytest.fixture(autouse=True)
def _single_thread_config():
    previous = config.n_jobs
    set_n_jobs(1)
    try:
        yield
    finally:
        set_n_jobs(previous)


def _prepared(markers_df):
    return PreparedMarkers(
        raw_markers_df=markers_df,
        marker_method="existing",
        source="test",
        parameters={},
        diagnostics={},
        signature="auto-test",
    )


def _spatial(var_names, *, detected=None, sparse=False):
    var_names = list(var_names)
    detected = set(var_names if detected is None else detected)
    matrix = np.zeros((4, len(var_names)), dtype=float)
    for index, gene in enumerate(var_names):
        if gene in detected:
            matrix[index % 4, index] = 1.0
    if sparse:
        matrix = csr_matrix(matrix)
    return ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(index=[f"spot_{index}" for index in range(4)]),
        var=pd.DataFrame(index=var_names),
    )


def _de_markers(groups):
    rows = []
    for group, qualities in groups.items():
        for index, quality in enumerate(qualities):
            rows.append(
                {
                    "group": group,
                    "names": f"{group}_{index}",
                    "logfoldchanges": float(quality),
                    "pvals_adj": 0.01,
                    "scores": float(quality),
                }
            )
    return pd.DataFrame(rows)


def test_fixed_and_none_top_n_keep_existing_behavior():
    raw = _de_markers({"A": [5, 4, 3], "B": [5, 4, 3]})
    prepared = _prepared(raw)

    fixed = select_prepared_markers(
        prepared, raw["names"], top_n_genes=2, log2fc_min=0, pval_cutoff=1
    )
    unlimited = select_prepared_markers(
        prepared, raw["names"], top_n_genes=None, log2fc_min=0, pval_cutoff=1
    )

    assert fixed.groupby(fixed["group"]).size().to_dict() == {"A": 2, "B": 2}
    assert unlimited.groupby(unlimited["group"]).size().to_dict() == {"A": 3, "B": 3}


def test_fixed_top_n_still_uses_requested_score_order_and_role_ranks():
    raw = pd.DataFrame(
        {
            "group": ["A"] * 6,
            "marker_role": ["positive"] * 3 + ["negative"] * 3,
            "names": ["P1", "P2", "P3", "N1", "N2", "N3"],
            "logfoldchanges": [3, 2, 1, -3, -2, -1],
            "pvals_adj": [0.01] * 6,
            "scores": [1, 3, 2, -1, -3, -2],
        }
    )

    selected = select_prepared_markers(
        _prepared(raw),
        raw["names"],
        top_n_genes=2,
        sort_by_column="scores",
        log2fc_min=0,
        pval_cutoff=1,
    )

    assert selected["names"].tolist() == ["P2", "N2", "P3", "N3"]
    assert selected.groupby(
        [selected["group"], selected["marker_role"]]
    )["marker_rank"].apply(list).to_dict() == {
        ("A", "negative"): [1, 2],
        ("A", "positive"): [1, 2],
    }


def test_auto_selection_truncates_weak_tails_and_allows_different_counts():
    raw = _de_markers(
        {
            "A": [10, 9, 0.1, 0.09, 0.08],
            "B": [10, 9, 8, 7, 6],
        }
    )
    prepared = _prepared(raw)
    table = _spatial(raw["names"])

    selected, diagnostics = select_prepared_markers(
        prepared,
        table.var_names,
        top_n_genes="auto",
        spatial_table=table,
        auto_marker_min=1,
        auto_marker_max=4,
        auto_marker_cumulative_fraction=0.9,
        auto_marker_relative_strength=0.15,
        log2fc_min=0,
        pval_cutoff=1,
        return_diagnostics=True,
    )

    assert selected.groupby(selected["group"]).size().to_dict() == {"A": 2, "B": 4}
    assert selected.loc["A", "names"].tolist() == ["A_0", "A_1"]
    auto = diagnostics["auto_marker_selection"]
    assert auto["enabled"] is True
    assert auto["groups"]["A"]["k_relative"] == 2
    assert auto["groups"]["A"]["fallback_used"] is False
    assert not any(column.startswith("_auto") for column in selected.columns)


def _ranking_disagreement_markers(include_scores=True):
    data = {
        "group": ["A"] * 5,
        "names": ["A", "B", "C", "D", "E"],
        "logfoldchanges": [1.0, 1.1, 4.0, 3.5, 0.1],
        "pvals_adj": [1e-5, 1e-5, 1e-20, 1e-18, 0.1],
    }
    if include_scores:
        data["scores"] = [10, 9, 8, 7, 6]
    return pd.DataFrame(data)


def test_auto_preserves_score_ranking_while_adaptive_quality_estimates_n():
    raw = _ranking_disagreement_markers()
    table = _spatial(raw["names"])

    selected, diagnostics = select_prepared_markers(
        ed.prepare_markers(markers_df=raw, verbose=False),
        table.var_names,
        top_n_genes="auto",
        sort_by_column="scores",
        spatial_table=table,
        auto_marker_min=1,
        auto_marker_max=5,
        log2fc_min=0,
        pval_cutoff=1,
        return_diagnostics=True,
    )

    assert selected["names"].tolist() == ["A", "B"]
    assert selected["marker_rank"].tolist() == [1, 2]
    group = diagnostics["auto_marker_selection"]["groups"]["A"]
    assert group["ranking_source"] == "scores"
    assert group["ranking_fallback_used"] is False
    assert group["size_estimation_source"] == (
        "abs_logfoldchanges_x_capped_neg_log10_pvals_adj"
    )
    assert group["size_cutoff_quality"] == group["cutoff_quality"]
    assert group["selected_last_quality"] != group["size_cutoff_quality"]


def test_auto_uses_adaptive_quality_for_ranking_when_scores_are_missing():
    raw = _ranking_disagreement_markers(include_scores=False)
    table = _spatial(raw["names"])

    selected, diagnostics = select_prepared_markers(
        ed.prepare_markers(markers_df=raw, verbose=False),
        table.var_names,
        top_n_genes="auto",
        sort_by_column="scores",
        spatial_table=table,
        auto_marker_min=1,
        auto_marker_max=5,
        log2fc_min=0,
        pval_cutoff=1,
        return_diagnostics=True,
    )

    assert selected["names"].tolist() == ["C", "D"]
    group = diagnostics["auto_marker_selection"]["groups"]["A"]
    assert group["ranking_source"] == group["size_estimation_source"]
    assert group["ranking_fallback_used"] is True


def test_auto_default_uses_deseq_stat_and_never_basemean_for_ranking():
    raw = pd.DataFrame(
        {
            "group": ["A"] * 5,
            "names": ["G1", "G2", "G3", "G4", "G5"],
            "log2FoldChange": [1.0, 1.1, 4.0, 3.5, 0.1],
            "padj": [1e-5, 1e-5, 1e-20, 1e-18, 0.1],
            "stat": [10, 9, 8, 7, 6],
            "baseMean": [1, 2, 3, 4, 1000],
        }
    )
    table = _spatial(raw["names"])

    selected, diagnostics = select_prepared_markers(
        ed.prepare_markers(markers_df=raw, verbose=False),
        table.var_names,
        top_n_genes="auto",
        sort_by_column="scores",
        spatial_table=table,
        auto_marker_min=1,
        auto_marker_max=5,
        log2fc_min=0,
        pval_cutoff=1,
        return_diagnostics=True,
    )

    assert selected["names"].tolist() == ["G1", "G2"]
    group = diagnostics["auto_marker_selection"]["groups"]["A"]
    assert group["ranking_source"] == "stat"
    assert group["ranking_source"].casefold() != "basemean"


def test_auto_basemean_only_table_falls_back_to_adaptive_ranking():
    raw = _ranking_disagreement_markers(include_scores=False).assign(
        baseMean=[1000, 4, 3, 2, 1]
    )
    table = _spatial(raw["names"])

    selected, diagnostics = select_prepared_markers(
        ed.prepare_markers(markers_df=raw, verbose=False),
        table.var_names,
        top_n_genes="auto",
        sort_by_column="scores",
        spatial_table=table,
        auto_marker_min=1,
        auto_marker_max=5,
        log2fc_min=0,
        pval_cutoff=1,
        return_diagnostics=True,
    )

    assert selected["names"].tolist() == ["C", "D"]
    group = diagnostics["auto_marker_selection"]["groups"]["A"]
    assert group["ranking_fallback_used"] is True
    assert group["ranking_source"].casefold() != "basemean"


def test_auto_basemean_without_de_statistics_uses_stable_minimum_fallback():
    raw = pd.DataFrame(
        {
            "group": ["A"] * 3,
            "names": ["G1", "G2", "G3"],
            "baseMean": [1, 1000, 500],
        }
    )
    table = _spatial(raw["names"])

    selected, diagnostics = select_prepared_markers(
        _prepared(raw),
        table.var_names,
        top_n_genes="auto",
        sort_by_column="scores",
        spatial_table=table,
        auto_marker_min=2,
        auto_marker_max=3,
        return_diagnostics=True,
    )

    assert selected["names"].tolist() == ["G1", "G2"]
    group = diagnostics["auto_marker_selection"]["groups"]["A"]
    assert group["ranking_source"] == "marker_rank_stable_order"
    assert group["size_estimation_source"] == "none"
    assert group["fallback_used"] is True


def test_auto_respects_explicit_custom_numeric_ranking():
    raw = _ranking_disagreement_markers(include_scores=False).assign(
        AUC=[0.99, 0.95, 0.80, 0.70, 0.60]
    )
    table = _spatial(raw["names"])

    selected, diagnostics = select_prepared_markers(
        _prepared(raw),
        table.var_names,
        top_n_genes="auto",
        sort_by_column="auc",
        spatial_table=table,
        auto_marker_min=1,
        auto_marker_max=5,
        log2fc_min=0,
        pval_cutoff=1,
        return_diagnostics=True,
    )

    assert selected["names"].tolist() == ["A", "B"]
    group = diagnostics["auto_marker_selection"]["groups"]["A"]
    assert group["ranking_source"] == "AUC"
    assert group["ranking_fallback_used"] is False


def test_auto_ranking_usability_falls_back_per_signature():
    first = _ranking_disagreement_markers().assign(group="A")
    second = _ranking_disagreement_markers().assign(
        group="B",
        names=["BA", "BB", "BC", "BD", "BE"],
        scores=np.nan,
    )
    raw = pd.concat([first, second], ignore_index=True)
    table = _spatial(raw["names"])

    selected, diagnostics = select_prepared_markers(
        _prepared(raw),
        table.var_names,
        top_n_genes="auto",
        sort_by_column="scores",
        spatial_table=table,
        auto_marker_min=1,
        auto_marker_max=5,
        log2fc_min=0,
        pval_cutoff=1,
        return_diagnostics=True,
    )

    assert selected.loc["A", "names"].tolist() == ["A", "B"]
    assert selected.loc["B", "names"].tolist() == ["BC", "BD"]
    groups = diagnostics["auto_marker_selection"]["groups"]
    assert groups["A"]["ranking_fallback_used"] is False
    assert groups["B"]["ranking_fallback_used"] is True


def test_auto_selection_is_independent_per_signed_role():
    raw = pd.DataFrame(
        {
            "group": ["A"] * 7,
            "marker_role": ["positive"] * 3 + ["negative"] * 4,
            "names": [f"G{index}" for index in range(7)],
            "logfoldchanges": [10, 9, 0.1, -10, -8, -7, -6],
            "pvals_adj": [0.01] * 7,
            "scores": [1, 2, 3, -1, -2, -3, -4],
        }
    )
    table = _spatial(raw["names"])

    selected = select_prepared_markers(
        _prepared(raw),
        table.var_names,
        top_n_genes="auto",
        spatial_table=table,
        auto_marker_min=1,
        auto_marker_max=10,
        auto_marker_cumulative_fraction=0.8,
        auto_marker_relative_strength=0.15,
        log2fc_min=0,
        pval_cutoff=1,
    )

    counts = selected.groupby(
        [selected["group"], selected["marker_role"]]
    ).size().to_dict()
    assert counts == {("A", "negative"): 3, ("A", "positive"): 2}
    assert selected.loc[selected["marker_role"] == "positive", "names"].tolist() == [
        "G2",
        "G1",
    ]
    assert selected.loc[selected["marker_role"] == "negative", "names"].tolist() == [
        "G6",
        "G5",
        "G4",
    ]
    ranks = selected.groupby(
        [selected["group"], selected["marker_role"]]
    )["marker_rank"].apply(list).to_dict()
    assert ranks == {
        ("A", "negative"): [1, 2, 3],
        ("A", "positive"): [1, 2],
    }


def test_auto_spatial_detection_is_sparse_safe_and_can_be_disabled():
    raw = _de_markers({"A": [10, 8, 6]})
    dense = _spatial(raw["names"], detected={"A_0", "A_1"}).X

    class NoDensifyCSR(csr_matrix):
        def toarray(self, *args, **kwargs):
            raise AssertionError("auto marker detection must not densify sparse X")

    table = SimpleNamespace(
        X=NoDensifyCSR(dense),
        var_names=pd.Index(raw["names"]),
    )

    filtered, diagnostics = select_prepared_markers(
        _prepared(raw),
        table.var_names,
        top_n_genes="auto",
        spatial_table=table,
        auto_marker_min=3,
        auto_marker_max=3,
        log2fc_min=0,
        pval_cutoff=1,
        return_diagnostics=True,
    )
    unfiltered = select_prepared_markers(
        _prepared(raw),
        table.var_names,
        top_n_genes="auto",
        spatial_table=table,
        auto_marker_min=3,
        auto_marker_max=3,
        auto_marker_min_detected_spots=0,
        log2fc_min=0,
        pval_cutoff=1,
    )

    assert filtered["names"].tolist() == ["A_0", "A_1"]
    assert diagnostics["auto_marker_selection"][
        "n_removed_by_spatial_detection"
    ] == 1
    assert unfiltered["names"].tolist() == ["A_0", "A_1", "A_2"]


def test_auto_fallback_uses_stable_order_and_keeps_small_sets():
    raw = pd.DataFrame(
        {"group": ["A", "A", "A"], "names": ["G2", "G1", "G3"]}
    )
    table = _spatial(raw["names"])

    selected, diagnostics = select_prepared_markers(
        _prepared(raw),
        table.var_names,
        top_n_genes="auto",
        spatial_table=table,
        auto_marker_min=2,
        auto_marker_max=5,
        return_diagnostics=True,
    )

    assert selected["names"].tolist() == ["G2", "G1"]
    group = diagnostics["auto_marker_selection"]["groups"]["A"]
    assert group["quality_source"] == "none"
    assert group["ranking_source"] == "marker_rank_stable_order"
    assert group["size_estimation_source"] == "none"
    assert group["fallback_used"] is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"auto_marker_min": 0}, "auto_marker_min"),
        ({"auto_marker_min": 3, "auto_marker_max": 2}, "auto_marker_max"),
        ({"auto_marker_cumulative_fraction": 0}, "cumulative_fraction"),
        ({"auto_marker_cumulative_fraction": 1.1}, "cumulative_fraction"),
        ({"auto_marker_relative_strength": 0}, "relative_strength"),
        ({"auto_marker_relative_strength": 1.1}, "relative_strength"),
        ({"auto_marker_padj_cap": 0}, "padj_cap"),
        ({"auto_marker_padj_cap": np.inf}, "padj_cap"),
        ({"auto_marker_min_detected_spots": -1}, "min_detected_spots"),
        ({"auto_marker_min_detected_spots": 1.5}, "min_detected_spots"),
    ],
)
def test_auto_parameter_validation(kwargs, message):
    raw = _de_markers({"A": [1]})
    table = _spatial(raw["names"])
    with pytest.raises(ValueError, match=message):
        select_prepared_markers(
            _prepared(raw),
            table.var_names,
            top_n_genes="auto",
            spatial_table=table,
            log2fc_min=0,
            pval_cutoff=1,
            **kwargs,
        )


def test_auto_requires_spatial_table_and_rejects_unknown_string():
    raw = _de_markers({"A": [1]})
    prepared = _prepared(raw)
    with pytest.raises(ValueError, match="requires spatial_table"):
        select_prepared_markers(prepared, raw["names"], top_n_genes="auto")
    with pytest.raises(ValueError, match="None, or 'auto'"):
        select_prepared_markers(prepared, raw["names"], top_n_genes="adaptive123")


def test_prepared_markers_can_auto_select_differently_without_mutation():
    raw = _de_markers({"A": [5, 4, 3]})
    prepared = _prepared(raw)
    original = prepared.raw_markers_df.copy(deep=True)
    first_table = _spatial(raw["names"], detected={"A_0", "A_1"}, sparse=True)
    second_table = _spatial(raw["names"], detected={"A_1", "A_2"}, sparse=True)

    first = select_prepared_markers(
        prepared,
        first_table.var_names,
        top_n_genes="auto",
        spatial_table=first_table,
        auto_marker_min=3,
        auto_marker_max=3,
        log2fc_min=0,
        pval_cutoff=1,
    )
    second = select_prepared_markers(
        prepared,
        second_table.var_names,
        top_n_genes="auto",
        spatial_table=second_table,
        auto_marker_min=3,
        auto_marker_max=3,
        log2fc_min=0,
        pval_cutoff=1,
    )

    assert set(first["names"]) == {"A_0", "A_1"}
    assert set(second["names"]) == {"A_1", "A_2"}
    pd.testing.assert_frame_equal(prepared.raw_markers_df, original)


def test_workflow_auto_selects_before_phase_resolution(monkeypatch):
    markers = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "marker_role": ["positive", "negative", "positive", "negative"],
            "names": ["A_POS", "A_NEG", "B_POS", "B_NEG"],
            "logfoldchanges": [2, -2, 2, -2],
            "pvals_adj": [0.01] * 4,
            "scores": [2, -2, 2, -2],
        }
    )
    expression = np.array(
        [
            [5, 0, 0, 3],
            [4, 0, 0, 2],
            [0, 3, 5, 0],
            [0, 2, 4, 0],
        ],
        dtype=float,
    )
    table = ad.AnnData(
        X=expression,
        obs=pd.DataFrame(index=[f"spot_{index}" for index in range(4)]),
        var=pd.DataFrame(index=markers["names"]),
    )
    captured = {}
    original_resolver = extra_module.resolve_phase_marker_tables

    def recording_resolver(*args, **kwargs):
        captured["top_n_genes"] = kwargs.get("top_n_genes")
        resolved = original_resolver(*args, **kwargs)
        captured["phase1_markers"] = resolved[0]
        captured["phase2_markers"] = resolved[1]
        return resolved

    monkeypatch.setattr(extra_module, "resolve_phase_marker_tables", recording_resolver)
    result = ed.run_easydecon(
        table,
        markers_df=markers,
        top_n_genes="auto",
        auto_marker_min=1,
        auto_marker_max=2,
        log2fc_min=0,
        pval_cutoff=1,
        drop_ribosomal=False,
        drop_mitochondrial=False,
        marker_roles="shared",
        filtering_algorithm="quantile",
        method="ucell",
        min_markers=1,
        return_result_object=True,
        verbose=False,
    )

    assert captured["top_n_genes"] is None
    assert set(captured["phase1_markers"]["marker_role"]) == {"positive"}
    assert set(captured["phase2_markers"]["marker_role"]) == {
        "positive",
        "negative",
    }
    assert set(result.markers_df["marker_role"]) == {"positive", "negative"}
    assert result.diagnostics["markers"]["top_n_applied_by"] == (
        "auto_spatial_marker_selector"
    )
    assert result.diagnostics["markers"]["selection"]["auto_marker_selection"][
        "enabled"
    ] is True
    summary = ed.summarize_easydecon_result(result, as_dataframe=False)
    assert summary["markers"]["auto_marker_selection"] == {
        "enabled": True,
        "min_selected_per_signature": 1,
        "max_selected_per_signature": 1,
        "ranking_sources": ["scores"],
        "size_estimation_sources": [
            "abs_logfoldchanges_x_capped_neg_log10_pvals_adj"
        ],
    }


def test_refinement_phase2_auto_uses_child_spatial_table(monkeypatch):
    markers = _ranking_disagreement_markers().assign(group="ChildA")
    table = ad.AnnData(
        X=csr_matrix(np.ones((4, markers.shape[0]))),
        obs=pd.DataFrame(index=[f"spot_{index}" for index in range(4)]),
        var=pd.DataFrame(index=markers["names"]),
    )
    parent = SimpleNamespace(
        priors_df=pd.DataFrame(
            {"Parent": [1.0, 1.0, 0.0, 0.0]}, index=table.obs.index
        )
    )
    captured = {}
    original_selector = refinement_module.select_prepared_markers
    original_resolver = refinement_module.resolve_phase_marker_tables

    def recording_selector(*args, **kwargs):
        captured["spatial_table"] = kwargs.get("spatial_table")
        result = original_selector(*args, **kwargs)
        captured["selected_markers"] = result[0] if isinstance(result, tuple) else result
        return result

    def recording_resolver(*args, **kwargs):
        captured["phase_top_n"] = kwargs.get("top_n_genes")
        return original_resolver(*args, **kwargs)

    monkeypatch.setattr(refinement_module, "select_prepared_markers", recording_selector)
    monkeypatch.setattr(refinement_module, "resolve_phase_marker_tables", recording_resolver)
    refined = ed.refine_group(
        table,
        parent_result=parent,
        parent_group="Parent",
        markers_df=markers,
        mode="phase2",
        top_n_genes="auto",
        auto_marker_min=1,
        auto_marker_max=5,
        log2fc_min=0,
        pval_cutoff=1,
        drop_ribosomal=False,
        drop_mitochondrial=False,
        method="jaccard",
        min_markers=1,
        verbose=False,
    )

    assert captured["spatial_table"].n_obs == 2
    assert captured["phase_top_n"] is None
    assert captured["selected_markers"]["names"].tolist() == ["A", "B"]
    assert refined.diagnostics["marker_diagnostics"]["top_n_applied_by"] == (
        "auto_spatial_marker_selector"
    )

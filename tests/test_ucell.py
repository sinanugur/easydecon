import anndata as ad
import numpy as np
import pandas as pd
import pytest

import easydecon as ed
from easydecon.config import config, set_batch_size, set_n_jobs
from easydecon.easydecon import (
    _build_ucell_signatures,
    assign_clusters_from_df,
    function_row_ucell,
    get_clusters_by_similarity_on_tissue,
)
from easydecon.extra import EasyDeconResult
from easydecon.diagnostics import summarize_easydecon_result
import easydecon.refinement as refinement_module


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


def _table():
    return ad.AnnData(
        X=np.array(
            [
                [10.0, 8.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 10.0, 8.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [5.0, 5.0, 5.0, 5.0, 5.0],
            ]
        ),
        obs=pd.DataFrame(index=["spot_a", "spot_b", "spot_zero", "spot_const"]),
        var=pd.DataFrame(index=["G1", "G2", "G3", "G4", "G5"]),
    )


def _markers(include_roles=False):
    markers = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "names": ["G1", "G2", "G3", "G4"],
            "logfoldchanges": [4.0, 3.0, 4.0, 3.0],
            "scores": [4.0, 3.0, 4.0, 3.0],
        }
    )
    if include_roles:
        markers["marker_role"] = "positive"
    return markers


def test_ucell_positive_signature_scores_highest():
    row = pd.Series({"G1": 10.0, "G2": 8.0, "G3": 1.0, "G4": 0.0})

    scores = function_row_ucell(row, markers_df=_markers(), min_markers=1)

    assert scores["A"] > scores["B"]


def test_ucell_negative_markers_reduce_score():
    markers = pd.DataFrame(
        {
            "group": ["A", "A"],
            "names": ["G1", "G2"],
            "marker_role": ["positive", "negative"],
        }
    )
    clean = pd.Series({"G1": 10.0, "G2": 0.0})
    contaminated = pd.Series({"G1": 10.0, "G2": 9.0})

    clean_score = function_row_ucell(clean, markers_df=markers, min_markers=1)["A"]
    contaminated_score = function_row_ucell(
        contaminated, markers_df=markers, min_markers=1
    )["A"]

    assert contaminated_score < clean_score


def test_ucell_missing_markers_reduce_recovery_score():
    full = pd.Series({"G1": 10.0, "G2": 8.0})
    partial = pd.Series({"G1": 10.0, "G2": 0.0})

    full_score = function_row_ucell(full, markers_df=_markers().iloc[:2], min_markers=1)[
        "A"
    ]
    partial_score = function_row_ucell(
        partial,
        markers_df=_markers().iloc[:2],
        min_markers=1,
        recovery_power=1.0,
    )["A"]

    assert partial_score < full_score


def test_ucell_all_zero_row_returns_zero():
    row = pd.Series(0.0, index=["G1", "G2", "G3", "G4"])

    scores = function_row_ucell(row, markers_df=_markers(), min_markers=1)

    assert scores == {"A": 0.0, "B": 0.0}


def test_ucell_constant_row_returns_zero():
    row = pd.Series(5.0, index=["G1", "G2", "G3", "G4"])

    scores = function_row_ucell(row, markers_df=_markers(), min_markers=1)

    assert scores == {"A": 0.0, "B": 0.0}


def test_ucell_too_few_available_markers_returns_zero():
    row = pd.Series({"G1": 10.0})

    scores = function_row_ucell(row, markers_df=_markers().iloc[:2], min_markers=2)

    assert scores["A"] == 0.0


def test_ucell_too_few_detected_markers_returns_zero():
    row = pd.Series({"G1": 10.0, "G2": 0.0})

    scores = function_row_ucell(row, markers_df=_markers().iloc[:2], min_markers=2)

    assert scores["A"] == 0.0


def test_ucell_scores_are_finite_and_bounded():
    row = pd.Series({"G1": 10.0, "G2": 8.0, "G3": 1.0, "G4": 0.0})

    scores = function_row_ucell(row, markers_df=_markers(), min_markers=1)

    assert all(np.isfinite(list(scores.values())))
    assert all(0.0 <= value <= 1.0 for value in scores.values())


def test_ucell_role_column_absent_means_all_positive():
    row = pd.Series({"G1": 10.0, "G2": 8.0})

    scores = function_row_ucell(row, markers_df=_markers().iloc[:2], min_markers=1)

    assert scores["A"] > 0


def test_ucell_identity_role_is_positive():
    markers = pd.DataFrame({"group": ["A"], "names": ["G1"], "marker_role": ["identity"]})
    row = pd.Series({"G1": 10.0})

    scores = function_row_ucell(row, markers_df=markers, min_markers=1)

    assert scores["A"] > 0


def test_ucell_presence_role_is_ignored():
    markers = pd.DataFrame(
        {
            "group": ["A", "A"],
            "names": ["G1", "G2"],
            "marker_role": ["presence", "identity"],
        }
    )

    signatures = _build_ucell_signatures(markers, ["G1", "G2"])

    assert signatures["positive"]["A"].tolist() == ["G2"]


def test_ucell_unknown_role_raises():
    markers = pd.DataFrame({"group": ["A"], "names": ["G1"], "marker_role": ["anti"]})

    with pytest.raises(ValueError, match="Allowed values"):
        _build_ucell_signatures(markers, ["G1"])


def test_ucell_conflicting_roles_raise():
    markers = pd.DataFrame(
        {
            "group": ["A", "A"],
            "names": ["G1", "G1"],
            "marker_role": ["positive", "negative"],
        }
    )

    with pytest.raises(ValueError, match="both positive and negative"):
        _build_ucell_signatures(markers, ["G1"])


def test_ucell_drop_shared_markers():
    markers = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "names": ["shared", "A1", "shared", "B1"],
        }
    )

    signatures = _build_ucell_signatures(
        markers,
        ["shared", "A1", "B1"],
        drop_shared_markers=True,
    )

    assert signatures["positive"]["A"].tolist() == ["A1"]
    assert signatures["positive"]["B"].tolist() == ["B1"]


def test_ucell_drop_shared_markers_does_not_remove_negatives():
    markers = pd.DataFrame(
        {
            "group": ["A", "B"],
            "names": ["shared_negative", "shared_negative"],
            "marker_role": ["negative", "negative"],
        }
    )

    signatures = _build_ucell_signatures(
        markers,
        ["shared_negative"],
        drop_shared_markers=True,
    )

    assert signatures["negative"]["A"].tolist() == ["shared_negative"]
    assert signatures["negative"]["B"].tolist() == ["shared_negative"]


def test_ucell_top_n_uses_marker_strength():
    markers = pd.DataFrame(
        {
            "group": ["A", "A", "A"],
            "names": ["weak", "strong", "middle"],
            "logfoldchanges": [1.0, 5.0, 3.0],
        }
    )

    signatures = _build_ucell_signatures(markers, ["weak", "strong", "middle"], top_n_markers=2)

    assert signatures["positive"]["A"].tolist() == ["strong", "middle"]


def test_ucell_top_n_is_applied_per_role():
    markers = pd.DataFrame(
        {
            "group": ["A", "A", "A", "A"],
            "names": ["p1", "p2", "n1", "n2"],
            "marker_role": ["positive", "positive", "negative", "negative"],
            "logfoldchanges": [5.0, 1.0, 4.0, 2.0],
        }
    )

    signatures = _build_ucell_signatures(markers, ["p1", "p2", "n1", "n2"], top_n_markers=1)

    assert signatures["positive"]["A"].tolist() == ["p1"]
    assert signatures["negative"]["A"].tolist() == ["n1"]


def test_ucell_marker_union_preserves_spatial_gene_order():
    signatures = _build_ucell_signatures(_markers(), ["G3", "G1", "G4", "G2"])

    assert signatures["marker_union"].tolist() == ["G3", "G1", "G4", "G2"]


def test_ucell_max_rank_changes_rank_truncation():
    row = pd.Series({"G1": 10.0, "G2": 1.0, "G3": 9.0, "G4": 8.0})
    markers = pd.DataFrame({"group": ["A", "A"], "names": ["G1", "G2"]})

    full = function_row_ucell(row, markers_df=markers, min_markers=1)["A"]
    truncated = function_row_ucell(
        row,
        markers_df=markers,
        min_markers=1,
        ucell_max_rank=1,
    )["A"]

    assert truncated < full


def test_get_clusters_by_similarity_ucell():
    result = get_clusters_by_similarity_on_tissue(
        _table(),
        _markers(),
        method="ucell",
        min_markers=1,
        verbose=False,
    )

    assert result.shape == (4, 2)
    assert result.loc["spot_a", "A"] > result.loc["spot_a", "B"]


def test_run_easydecon_ucell_with_marker_dataframe():
    result = ed.run_easydecon(
        _table(),
        markers_df=_markers(),
        filtering_algorithm="quantile",
        method="ucell",
        min_markers=1,
        return_result_object=True,
        verbose=False,
    )

    assert isinstance(result, EasyDeconResult)
    assert np.isfinite(result.phase2_result.to_numpy()).all()
    assert result.diagnostics["phase2"]["method"] == "ucell"
    assert "n_informative_rows" in result.diagnostics["phase2"]
    assert "n_uninformative_rows" in result.diagnostics["phase2"]


def test_summarize_easydecon_result_includes_ucell_phase2_fields():
    result = ed.run_easydecon(
        _table(),
        markers_df=_markers(),
        filtering_algorithm="quantile",
        method="ucell",
        min_markers=1,
        ucell_max_rank=3,
        ucell_negative_weight=0.5,
        return_result_object=True,
        verbose=False,
    )

    summary = summarize_easydecon_result(result, as_dataframe=False)

    assert summary["workflow"]["phase2"]["method"] == "ucell"
    assert summary["workflow"]["phase2"]["ucell_max_rank"] == 3
    assert summary["workflow"]["phase2"]["ucell_negative_weight"] == 0.5
    assert "n_informative_rows" in summary["workflow"]["phase2"]


def _reference_adata():
    counts = np.array(
        [[10, 8, 1, 0, 0], [10, 8, 1, 0, 0], [1, 0, 10, 8, 0], [1, 0, 10, 8, 0]],
        dtype=float,
    )
    adata = ad.AnnData(
        X=counts,
        obs=pd.DataFrame({"cell_type": ["A", "A", "B", "B"]}),
        var=pd.DataFrame(index=["G1", "G2", "G3", "G4", "G5"]),
    )
    adata.layers["counts"] = counts.copy()
    return adata


def test_run_easydecon_ucell_with_reference_markers():
    result = ed.run_easydecon(
        _table(),
        adata=_reference_adata(),
        marker_method="reference",
        groupby="cell_type",
        layer="counts",
        reference_min_cells=2,
        reference_min_mean=0,
        reference_min_log2fc=0,
        reference_min_detection=0,
        reference_min_detection_delta=0,
        filtering_algorithm="quantile",
        method="ucell",
        min_markers=1,
        log2fc_min=0,
        pval_cutoff=1.0,
        return_result_object=True,
        verbose=False,
    )

    assert not result.markers_df.empty
    assert result.diagnostics["phase2"]["method"] == "ucell"


def test_run_easydecon_ucell_with_prepared_markers():
    prepared = ed.prepare_markers(
        _reference_adata(),
        marker_method="reference",
        groupby="cell_type",
        layer="counts",
        reference_min_cells=2,
        reference_min_mean=0,
        reference_min_log2fc=0,
        reference_min_detection=0,
        reference_min_detection_delta=0,
        verbose=False,
    )

    result = ed.run_easydecon(
        _table(),
        prepared_markers=prepared,
        filtering_algorithm="quantile",
        method="ucell",
        min_markers=1,
        log2fc_min=0,
        pval_cutoff=1.0,
        return_result_object=True,
        verbose=False,
    )

    assert result.diagnostics["markers"]["prepared_markers_used"] is True
    assert result.diagnostics["phase2"]["method"] == "ucell"


def test_ucell_skipped_locations_are_zero():
    table = _table()
    table.obs["mask"] = [1, 0, 1, 0]

    result = get_clusters_by_similarity_on_tissue(
        table,
        _markers(),
        common_group_name="mask",
        method="ucell",
        min_markers=1,
        verbose=False,
    )

    assert (result.loc[table.obs["mask"] == 0] == 0).all().all()


def test_ucell_constant_scores_remain_unassigned():
    table = _table()
    scores = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=["spot_a"])

    assigned = assign_clusters_from_df(
        table[:1].copy(),
        scores,
        results_column="assigned",
        add_to_obs=False,
        verbose=False,
    )

    assert pd.isna(assigned.loc["spot_a", "assigned"])


def test_ucell_clear_winner_is_assigned():
    table = _table()
    scores = get_clusters_by_similarity_on_tissue(
        table,
        _markers(),
        method="ucell",
        min_markers=1,
        verbose=False,
    )

    assigned = assign_clusters_from_df(
        table,
        scores,
        results_column="assigned",
        add_to_obs=False,
        verbose=False,
    )

    assert assigned.loc["spot_a", "assigned"] == "A"


def test_ucell_zero_row_is_unassigned():
    table = _table()
    scores = get_clusters_by_similarity_on_tissue(
        table,
        _markers(),
        method="ucell",
        min_markers=1,
        verbose=False,
    )

    assigned = assign_clusters_from_df(
        table,
        scores,
        results_column="assigned",
        add_to_obs=False,
        verbose=False,
    )

    assert pd.isna(assigned.loc["spot_zero", "assigned"])


def _parent_result(table):
    priors = pd.DataFrame({"Parent": [1.0] * table.n_obs}, index=table.obs.index)
    return EasyDeconResult(
        markers_df=pd.DataFrame(),
        phase1_result=priors.copy(),
        phase2_result=priors.copy(),
        assigned_labels=pd.DataFrame({"easydecon": ["Parent"] * table.n_obs}, index=table.obs.index),
        priors_df=priors,
        likelihoods_df=priors.copy(),
        posterior_df=priors.copy(),
        assignment_df=priors.copy(),
        diagnostics={"results_column": "easydecon"},
    )


def test_refine_group_phase2_ucell():
    table = _table()

    refined = ed.refine_group(
        table,
        parent_result=_parent_result(table),
        parent_group="Parent",
        markers_df=_markers(),
        mode="phase2",
        method="ucell",
        min_markers=1,
        verbose=False,
    )

    assert refined.child_result is None
    assert np.isfinite(refined.phase2_result.to_numpy()).all()
    assert refined.diagnostics["child_phase1_ran"] is False


def test_refine_group_full_ucell():
    table = _table()

    refined = ed.refine_group(
        table,
        parent_result=_parent_result(table),
        parent_group="Parent",
        markers_df=_markers(),
        mode="full",
        filtering_algorithm="quantile",
        method="ucell",
        min_markers=1,
        verbose=False,
    )

    assert isinstance(refined.child_result, EasyDeconResult)
    assert refined.child_result.diagnostics["phase2"]["method"] == "ucell"


def test_refine_group_phase2_forwards_ucell_parameters(monkeypatch):
    captured = {}

    def fake_phase2(*args, **kwargs):
        captured.update(kwargs)
        child_table = args[0] if args else kwargs["sdata"]
        return pd.DataFrame(
            {"A": [1.0] * child_table.n_obs, "B": [0.0] * child_table.n_obs},
            index=child_table.obs.index,
        )

    monkeypatch.setattr(refinement_module, "get_clusters_by_similarity_on_tissue", fake_phase2)

    table = _table()
    ed.refine_group(
        table,
        parent_result=_parent_result(table),
        parent_group="Parent",
        markers_df=_markers(),
        mode="phase2",
        method="ucell",
        min_markers=1,
        ucell_max_rank=3,
        ucell_negative_weight=0.5,
        ucell_marker_role_column="marker_role",
        top_n_markers=2,
        recovery_power=2.0,
        drop_shared_markers=True,
        verbose=False,
    )

    assert captured["ucell_max_rank"] == 3
    assert captured["ucell_negative_weight"] == 0.5
    assert captured["ucell_marker_role_column"] == "marker_role"
    assert captured["top_n_markers"] == 2
    assert captured["recovery_power"] == 2.0
    assert captured["drop_shared_markers"] is True


def test_existing_methods_still_return_expected_shape_and_values():
    table = _table()
    methods = ["auc", "wjaccard", "jaccard", "cosine", "correlation", "sum", "mean", "median", "euclidean"]

    for method in methods:
        result = get_clusters_by_similarity_on_tissue(
            table,
            _markers(),
            method=method,
            min_markers=1,
            verbose=False,
        )
        assert result.shape == (table.n_obs, 2)
        assert np.isfinite(result.to_numpy()).all()

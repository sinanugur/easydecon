import anndata as ad
import numpy as np
import pandas as pd

import easydecon.extra as extra_module
import easydecon.refinement as refinement_module
from easydecon.extra import EasyDeconResult


def _spatial_table():
    return ad.AnnData(
        X=np.array([[8.0, 1.0, 1.0], [1.0, 8.0, 1.0], [4.0, 4.0, 1.0]]),
        obs=pd.DataFrame(index=["spot_0", "spot_1", "spot_2"]),
        var=pd.DataFrame(index=["G1", "G2", "G3"]),
    )


def _markers():
    return pd.DataFrame(
        {
            "group": ["A", "B"],
            "names": ["G1", "G2"],
            "scores": [5.0, 5.0],
            "logfoldchanges": [2.0, 2.0],
            "pvals_adj": [0.001, 0.001],
        }
    )


def _parent_result(table):
    index = table.obs.index
    priors = pd.DataFrame({"Parent": [1.0, 1.0, 1.0]}, index=index)
    return EasyDeconResult(
        markers_df=pd.DataFrame(),
        phase1_result=priors.copy(),
        phase2_result=priors.copy(),
        assigned_labels=pd.DataFrame({"easydecon": ["Parent"] * len(index)}, index=index),
        priors_df=priors,
        likelihoods_df=priors.copy(),
        posterior_df=priors.copy(),
        assignment_df=priors.copy(),
        diagnostics={"results_column": "easydecon"},
    )


def test_workflow_forwards_rank_parameters(monkeypatch):
    captured = {}

    def fake_phase2(*args, **kwargs):
        captured.update(kwargs)
        table = args[0] if args else kwargs["sdata"]
        return pd.DataFrame({"A": [1.0, 0.0, 0.5], "B": [0.0, 1.0, 0.0]}, index=table.obs.index)

    monkeypatch.setattr(extra_module, "get_clusters_by_similarity_on_tissue", fake_phase2)

    result = extra_module.easydecon_workflow(
        _spatial_table(),
        markers_df=_markers(),
        filtering_algorithm="quantile",
        method="auc",
        min_markers=1,
        top_n_markers=10,
        recovery_power=2.0,
        drop_shared_markers=True,
        center_auc=False,
        verbose=False,
        return_result_object=True,
    )

    assert captured["top_n_markers"] == 10
    assert captured["recovery_power"] == 2.0
    assert captured["drop_shared_markers"] is True
    assert captured["center_auc"] is False
    assert result.diagnostics["phase2"]["top_n_markers"] == 10
    assert result.diagnostics["phase2"]["fallback_auc"] == 0.0
    assert result.diagnostics["assignment"]["minimum_evidence"] == 0.0
    assert result.diagnostics["assignment"]["tie_tolerance"] == 1e-12


def test_refine_group_phase2_forwards_rank_parameters(monkeypatch):
    table = _spatial_table()
    captured = {}

    def fake_phase2(*args, **kwargs):
        captured.update(kwargs)
        child_table = args[0] if args else kwargs["sdata"]
        return pd.DataFrame({"A": [1.0] * child_table.n_obs, "B": [0.0] * child_table.n_obs}, index=child_table.obs.index)

    monkeypatch.setattr(refinement_module, "get_clusters_by_similarity_on_tissue", fake_phase2)

    refinement_module.refine_group(
        table,
        parent_result=_parent_result(table),
        parent_group="Parent",
        markers_df=_markers(),
        mode="phase2",
        method="auc",
        min_markers=1,
        top_n_markers=10,
        recovery_power=2.0,
        drop_shared_markers=True,
        center_auc=False,
        verbose=False,
    )

    assert captured["top_n_markers"] == 10
    assert captured["recovery_power"] == 2.0
    assert captured["drop_shared_markers"] is True
    assert captured["center_auc"] is False


def test_refine_group_forwards_assignment_safety_parameters(monkeypatch):
    table = _spatial_table()
    captured = {}

    def fake_phase2(*args, **kwargs):
        child_table = args[0] if args else kwargs["sdata"]
        return pd.DataFrame(
            {"A": [1.0] * child_table.n_obs, "B": [0.0] * child_table.n_obs},
            index=child_table.obs.index,
        )

    def fake_assign(*args, **kwargs):
        captured.update(kwargs)
        return pd.DataFrame({"Parent_subcluster": ["A"] * table.n_obs}, index=table.obs.index)

    monkeypatch.setattr(refinement_module, "get_clusters_by_similarity_on_tissue", fake_phase2)
    monkeypatch.setattr(refinement_module, "assign_clusters_from_df", fake_assign)

    refinement_module.refine_group(
        table,
        parent_result=_parent_result(table),
        parent_group="Parent",
        markers_df=_markers(),
        mode="phase2",
        method="jaccard",
        min_markers=1,
        minimum_evidence=0.05,
        tie_tolerance=1e-6,
        verbose=False,
    )

    assert captured["minimum_evidence"] == 0.05
    assert captured["tie_tolerance"] == 1e-6


def test_existing_phase2_parameters_still_forward(monkeypatch):
    table = _spatial_table()
    captured = {}

    def fake_phase2(*args, **kwargs):
        captured.update(kwargs)
        child_table = args[0] if args else kwargs["sdata"]
        return pd.DataFrame({"A": [1.0] * child_table.n_obs, "B": [0.0] * child_table.n_obs}, index=child_table.obs.index)

    monkeypatch.setattr(refinement_module, "get_clusters_by_similarity_on_tissue", fake_phase2)

    refinement_module.refine_group(
        table,
        parent_result=_parent_result(table),
        parent_group="Parent",
        markers_df=_markers(),
        mode="phase2",
        method="auc",
        min_markers=5,
        fallback_auc=0.2,
        expression_threshold=0.3,
        weight_column="scores",
        lambda_param=0.7,
        verbose=False,
    )

    assert captured["min_markers"] == 5
    assert captured["fallback_auc"] == 0.2
    assert captured["expression_threshold"] == 0.3
    assert captured["weight_column"] == "scores"
    assert captured["lambda_param"] == 0.7

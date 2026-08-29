import anndata as ad
import numpy as np
import pandas as pd

import easydecon.extra as extra_module
from easydecon.config import config, set_batch_size, set_n_jobs
from easydecon.easydecon import assign_clusters_from_df, function_row_auc_specific_v2


def _markers():
    return pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "names": ["G1", "G2", "G3", "G4"],
            "scores": [4.0, 3.0, 4.0, 3.0],
            "logfoldchanges": [2.0, 1.5, 2.0, 1.5],
            "pvals_adj": [0.001, 0.001, 0.001, 0.001],
        }
    ).set_index("group", drop=False)


def _spatial_table():
    return ad.AnnData(
        X=np.array([[8.0, 7.0, 1.0, 0.0], [0.0, 1.0, 8.0, 7.0]]),
        obs=pd.DataFrame(index=["spot_0", "spot_1"]),
        var=pd.DataFrame(index=["G1", "G2", "G3", "G4"]),
    )


def test_auc_uninformative_row_returns_zero():
    row = pd.Series(0.0, index=["G1", "G2", "G3", "G4"])

    result = function_row_auc_specific_v2(
        row,
        _markers(),
        gene_id_column="names",
        min_markers=2,
    )

    assert result == {"A": 0.0, "B": 0.0}


def test_auc_too_few_markers_returns_zero():
    row = pd.Series({"G1": 5.0, "G2": 4.0, "G3": 1.0, "G4": 0.0})

    result = function_row_auc_specific_v2(
        row,
        _markers(),
        gene_id_column="names",
        min_markers=3,
    )

    assert result == {"A": 0.0, "B": 0.0}


def test_workflow_auc_default_fallback_is_zero(monkeypatch):
    captured = {}

    def fake_phase2(*args, **kwargs):
        captured.update(kwargs)
        table = args[0] if args else kwargs["sdata"]
        return pd.DataFrame(
            {"A": [1.0, 0.0], "B": [0.0, 1.0]},
            index=table.obs.index,
        )

    monkeypatch.setattr(extra_module, "get_clusters_by_similarity_on_tissue", fake_phase2)

    extra_module.easydecon_workflow(
        _spatial_table(),
        markers_df=_markers().reset_index(drop=True),
        filtering_algorithm="quantile",
        method="auc",
        min_markers=1,
        return_result_object=True,
        verbose=False,
    )

    assert captured["fallback_auc"] == 0.0


def test_explicit_auc_fallback_is_preserved(monkeypatch):
    captured = {}

    def fake_phase2(*args, **kwargs):
        captured.update(kwargs)
        table = args[0] if args else kwargs["sdata"]
        return pd.DataFrame(
            {"A": [1.0, 0.0], "B": [0.0, 1.0]},
            index=table.obs.index,
        )

    monkeypatch.setattr(extra_module, "get_clusters_by_similarity_on_tissue", fake_phase2)

    extra_module.easydecon_workflow(
        _spatial_table(),
        markers_df=_markers().reset_index(drop=True),
        filtering_algorithm="quantile",
        method="auc",
        min_markers=1,
        fallback_auc=0.2,
        return_result_object=True,
        verbose=False,
    )

    assert captured["fallback_auc"] == 0.2


def test_auc_clear_signal_still_assigns_expected_group():
    row = pd.Series({"G1": 8.0, "G2": 7.0, "G3": 1.0, "G4": 0.0})
    scores = function_row_auc_specific_v2(
        row,
        _markers(),
        gene_id_column="names",
        min_markers=2,
    )
    table = ad.AnnData(
        X=np.ones((1, 1)),
        obs=pd.DataFrame(index=["spot_0"]),
        var=pd.DataFrame(index=["dummy"]),
    )
    score_df = pd.DataFrame([scores], index=table.obs.index)

    assigned = assign_clusters_from_df(
        table,
        score_df,
        results_column="assigned",
        add_to_obs=False,
        verbose=False,
    )

    assert scores["A"] > scores["B"]
    assert assigned.loc["spot_0", "assigned"] == "A"

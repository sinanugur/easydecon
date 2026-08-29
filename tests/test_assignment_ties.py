import anndata as ad
import numpy as np
import pandas as pd

from easydecon.easydecon import assign_clusters_from_df


def _table(n_obs=1):
    return ad.AnnData(
        X=np.ones((n_obs, 1)),
        obs=pd.DataFrame(index=[f"spot_{index}" for index in range(n_obs)]),
        var=pd.DataFrame(index=["G1"]),
    )


def _assign(scores, **kwargs):
    table = _table(scores.shape[0])
    scores = scores.copy()
    scores.index = table.obs.index
    return assign_clusters_from_df(
        table,
        scores,
        results_column="assigned",
        add_to_obs=False,
        verbose=False,
        **kwargs,
    )["assigned"]


def test_max_all_zero_row_is_unassigned():
    scores = pd.DataFrame({"A": [0.0], "B": [0.0]})

    result = _assign(scores)

    assert pd.isna(result.iloc[0])


def test_max_exact_top_tie_is_unassigned():
    scores = pd.DataFrame({"A": [1.0], "B": [1.0], "C": [0.0]})

    result = _assign(scores)

    assert pd.isna(result.iloc[0])


def test_max_near_tie_within_tolerance_is_unassigned():
    scores = pd.DataFrame({"A": [1.0], "B": [1.0 - 1e-13]})

    result = _assign(scores, tie_tolerance=1e-12)

    assert pd.isna(result.iloc[0])


def test_max_clear_winner_is_assigned():
    scores = pd.DataFrame({"A": [1.0], "B": [0.8]})

    result = _assign(scores)

    assert result.iloc[0] == "A"


def test_single_positive_column_is_assigned():
    scores = pd.DataFrame({"A": [0.7]})

    result = _assign(scores)

    assert result.iloc[0] == "A"


def test_single_zero_column_is_unassigned():
    scores = pd.DataFrame({"A": [0.0]})

    result = _assign(scores)

    assert pd.isna(result.iloc[0])


def test_minimum_evidence_rejects_weak_winner():
    scores = pd.DataFrame({"A": [0.01], "B": [0.0]})

    result = _assign(scores, minimum_evidence=0.05)

    assert pd.isna(result.iloc[0])


def test_zmax_constant_or_tied_row_is_unassigned():
    scores = pd.DataFrame(
        {
            "A": [1.0, 2.0],
            "B": [1.0, 2.0],
            "C": [0.0, 0.0],
        },
        index=["spot_0", "spot_1"],
    )

    result = _assign(scores, method="zmax")

    assert pd.isna(result.loc["spot_0"])


def test_add_to_obs_false_does_not_mutate_obs():
    table = _table()
    scores = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=table.obs.index)

    assign_clusters_from_df(
        table,
        scores,
        results_column="assigned",
        add_to_obs=False,
        verbose=False,
    )

    assert "assigned" not in table.obs.columns

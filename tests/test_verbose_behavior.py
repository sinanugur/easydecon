import anndata as ad
import numpy as np
import pandas as pd

from easydecon.easydecon import add_df_to_spatialdata, assign_clusters_from_df


def _spatial_table(index=("s1", "s2", "s3")):
    return ad.AnnData(
        X=np.ones((len(index), 1)),
        obs=pd.DataFrame(index=list(index)),
        var=pd.DataFrame(index=["G1"]),
    )


def _assignment_scores():
    return pd.DataFrame(
        {
            "A": [0.9, 0.2, 0.7],
            "B": [0.1, 0.8, 0.3],
        },
        index=["s1", "s2", "s3"],
    )


def test_assign_clusters_from_df_verbose_false_suppresses_output(capsys):
    spatial = _spatial_table()

    assign_clusters_from_df(
        spatial,
        _assignment_scores(),
        results_column="assigned",
        add_to_obs=True,
        verbose=False,
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_assign_clusters_from_df_verbose_true_keeps_output(capsys):
    spatial = _spatial_table()

    assign_clusters_from_df(
        spatial,
        _assignment_scores(),
        results_column="assigned",
        add_to_obs=True,
        verbose=True,
    )

    assert "Adding results" in capsys.readouterr().out


def test_assign_clusters_hybrid_progress_suppressed_when_verbose_false(capsys):
    spatial = _spatial_table()
    hybrid_scores = pd.DataFrame(
        {
            "A": [0.90, 0.10, 0.85],
            "B": [0.85, 0.90, 0.80],
            "C": [0.80, 0.85, 0.90],
            "D": [0.10, 0.80, 0.10],
        },
        index=["s1", "s2", "s3"],
    )

    assign_clusters_from_df(
        spatial,
        hybrid_scores,
        method="hybrid",
        allow_multiple=True,
        add_to_obs=False,
        verbose=False,
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_add_df_to_spatialdata_verbose_false_suppresses_output(capsys):
    spatial = _spatial_table()
    values = pd.DataFrame({"score": [1.0, 2.0]}, index=["s1", "s2"])

    add_df_to_spatialdata(spatial, values, verbose=False)

    assert capsys.readouterr().out == ""
    assert "score" in spatial.obs.columns


def test_add_df_to_spatialdata_reindexes_to_obs_order():
    spatial = _spatial_table(index=("s2", "s1", "s3"))
    values = pd.DataFrame({"score": [1.0, 2.0]}, index=["s1", "s2"])

    add_df_to_spatialdata(spatial, values, verbose=False)

    assert spatial.obs.index.tolist() == ["s2", "s1", "s3"]
    assert spatial.obs.loc["s2", "score"] == 2.0
    assert spatial.obs.loc["s1", "score"] == 1.0
    assert pd.isna(spatial.obs.loc["s3", "score"])


def test_add_df_to_spatialdata_requires_dataframe():
    spatial = _spatial_table()

    try:
        add_df_to_spatialdata(spatial, {"score": [1.0]}, verbose=False)
    except TypeError as exc:
        assert str(exc) == "df must be a pandas DataFrame."
    else:
        raise AssertionError("Expected a TypeError for non-DataFrame input.")

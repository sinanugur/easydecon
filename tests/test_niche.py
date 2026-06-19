from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from easydecon.niche import (
    detect_niches_from_easydecon_result,
    detect_spatial_niches_from_posteriors,
    summarize_niche_compositions,
)


@pytest.fixture
def spatial_table():
    table = ad.AnnData(
        X=np.ones((6, 2)),
        obs=pd.DataFrame(
            index=["spot_c", "spot_a", "spot_f", "spot_b", "spot_e", "spot_d"]
        ),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    table.obsm["spatial"] = np.array(
        [[0, 0], [1, 0], [10, 0], [0, 1], [10, 1], [11, 0]], dtype=float
    )
    return table


@pytest.fixture
def posterior_df():
    return pd.DataFrame(
        {
            "A": [0.1, 0.9, 0.2, 0.8, 0.15, 0.85],
            "B": [0.9, 0.1, 0.8, 0.2, 0.85, 0.15],
        },
        index=["spot_f", "spot_c", "spot_e", "spot_a", "spot_d", "spot_b"],
    )


def _detect(table, posterior_like, **kwargs):
    return detect_spatial_niches_from_posteriors(
        table,
        posterior_like,
        n_neighbors=2,
        n_niches=2,
        smooth=False,
        add_to_obs=False,
        **kwargs,
    )


def test_detect_spatial_niches_accepts_dataframe(spatial_table, posterior_df):
    niches, smoothed = _detect(spatial_table, posterior_df)

    assert niches.shape == (6, 1)
    assert smoothed.shape == (6, 2)


def test_detect_spatial_niches_accepts_easydecon_result_like_object(
    spatial_table, posterior_df
):
    result = SimpleNamespace(posterior_df=posterior_df, assignment_df=None)

    niches, smoothed = _detect(spatial_table, result)

    assert niches.shape[0] == smoothed.shape[0] == 6


def test_detect_spatial_niches_result_without_posterior_raises_helpful_error(
    spatial_table, posterior_df
):
    result = SimpleNamespace(posterior_df=None, assignment_df=posterior_df)

    with pytest.raises(
        ValueError,
        match="list-style mask workflow.*use_assignment_if_no_posterior",
    ):
        _detect(spatial_table, result)


def test_detect_spatial_niches_can_use_assignment_if_no_posterior(
    spatial_table, posterior_df
):
    result = SimpleNamespace(posterior_df=None, assignment_df=posterior_df)

    niches, smoothed = _detect(
        spatial_table,
        result,
        use_assignment_if_no_posterior=True,
    )

    assert niches.shape[0] == smoothed.shape[0] == 6


def test_detect_spatial_niches_preserves_table_obs_order(
    spatial_table, posterior_df
):
    partial = posterior_df.drop(index="spot_e")

    _, smoothed = _detect(spatial_table, partial)

    expected = spatial_table.obs.index[spatial_table.obs.index.isin(partial.index)]
    assert smoothed.index.equals(expected)


def test_detect_spatial_niches_rejects_all_zero_input(spatial_table):
    posterior = pd.DataFrame(
        0.0,
        index=spatial_table.obs.index,
        columns=["A", "B"],
    )

    with pytest.raises(ValueError, match="contains only zero values"):
        _detect(spatial_table, posterior)


def test_detect_niches_from_easydecon_result_delegates(
    spatial_table, posterior_df
):
    result = SimpleNamespace(posterior_df=posterior_df, assignment_df=None)
    direct_niches, direct_smoothed = _detect(spatial_table, result)

    wrapped_niches, wrapped_smoothed = detect_niches_from_easydecon_result(
        spatial_table,
        result,
        n_neighbors=2,
        n_niches=2,
        smooth=False,
        add_to_obs=False,
    )

    assert wrapped_niches.shape == direct_niches.shape
    assert wrapped_smoothed.shape == direct_smoothed.shape


def test_summarize_niche_compositions_numeric_conversion():
    smoothed = pd.DataFrame(
        {"A": ["3", "1", "0", "2"], "B": ["1", "3", "2", "0"]},
        index=["s1", "s2", "s3", "s4"],
    )
    niches = pd.DataFrame(
        {"niche": pd.Categorical(["0", "0", "1", "1"])},
        index=smoothed.index,
    )

    summary = summarize_niche_compositions(
        smoothed,
        niches,
        normalize_rows=True,
    )

    assert all(np.issubdtype(dtype, np.number) for dtype in summary.dtypes)
    assert np.allclose(summary.sum(axis=1), 1.0)

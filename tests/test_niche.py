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


def _multi_sample_data():
    samples = np.array(["A"] * 100 + ["B"] * 40)
    n_locations = len(samples)
    table = ad.AnnData(
        X=np.ones((n_locations, 1)),
        obs=pd.DataFrame(
            {"sample": samples},
            index=[f"spot_{index}" for index in range(n_locations)],
        ),
        var=pd.DataFrame(index=["G1"]),
    )
    table.obsm["spatial"] = np.column_stack(
        [np.arange(n_locations, dtype=float), np.zeros(n_locations)]
    )
    first = np.where(np.arange(n_locations) % 2 == 0, 0.9, 0.1)
    posterior = pd.DataFrame(
        {"T": first, "B": 1.0 - first},
        index=table.obs.index,
    )
    return table, posterior


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


def test_sample_aware_smoothing_never_crosses_sample_boundaries():
    table = ad.AnnData(
        X=np.ones((3, 1)),
        obs=pd.DataFrame(
            {"sample": ["A", "A", "B"]},
            index=["A1", "A2", "B1"],
        ),
        var=pd.DataFrame(index=["G1"]),
    )
    table.obsm["spatial"] = np.array(
        [[0, 0], [1, 0], [0, 0]], dtype=float
    )
    posterior = pd.DataFrame(
        {"type_a": [1.0, 1.0, 0.0], "type_b": [0.0, 0.0, 1.0]},
        index=table.obs.index,
    )

    niches, smoothed, diagnostics = detect_spatial_niches_from_posteriors(
        table,
        posterior,
        sample_column="sample",
        n_neighbors=2,
        n_niches=2,
        add_to_obs=False,
        return_diagnostics=True,
    )

    pd.testing.assert_frame_equal(smoothed, posterior.astype(np.float32))
    assert niches.index.equals(table.obs.index)
    assert diagnostics["effective_neighbors_by_sample"] == {"A": 2, "B": 1}

    _, chunked = detect_spatial_niches_from_posteriors(
        table,
        posterior,
        sample_column="sample",
        n_neighbors=2,
        n_niches=2,
        smoothing_chunk_size=1,
        add_to_obs=False,
    )
    pd.testing.assert_frame_equal(chunked, smoothed)


def test_posterior_conversion_is_float32_ordered_and_non_mutating(spatial_table):
    posterior = pd.DataFrame(
        {
            "B": ["0.9", "0.1", "0.8", "0.2", "0.7", "0.3"],
            "invalid": ["x"] * 6,
            "A": [0.1, 0.9, np.inf, 0.8, np.nan, 0.7],
            "zero": [0.0] * 6,
        },
        index=spatial_table.obs.index,
    )
    original = posterior.copy(deep=True)

    _, smoothed = _detect(spatial_table, posterior)

    pd.testing.assert_frame_equal(posterior, original)
    assert smoothed.columns.tolist() == ["B", "A"]
    assert all(dtype == np.dtype("float32") for dtype in smoothed.dtypes)
    assert np.isfinite(smoothed.to_numpy()).all()


@pytest.mark.parametrize(
    ("balance_samples", "max_fit_per_sample", "expected"),
    [
        (True, None, {"A": 40, "B": 40}),
        (True, 25, {"A": 25, "B": 25}),
        (False, 50, {"A": 50, "B": 40}),
    ],
)
def test_multi_sample_fit_selection_predicts_all_locations(
    balance_samples,
    max_fit_per_sample,
    expected,
):
    table, posterior = _multi_sample_data()

    niches, smoothed, diagnostics = detect_spatial_niches_from_posteriors(
        table,
        posterior,
        sample_column="sample",
        balance_samples=balance_samples,
        max_fit_per_sample=max_fit_per_sample,
        smooth=False,
        n_niches=2,
        add_to_obs=False,
        return_diagnostics=True,
        random_state=7,
    )

    assert diagnostics["sample_counts"] == {"A": 100, "B": 40}
    assert diagnostics["fit_counts_by_sample"] == expected
    assert diagnostics["n_fit_locations"] == sum(expected.values())
    assert len(niches) == len(smoothed) == 140


def test_balanced_fit_is_deterministic():
    table, posterior = _multi_sample_data()
    kwargs = {
        "sample_column": "sample",
        "balance_samples": True,
        "max_fit_per_sample": 25,
        "smooth": False,
        "n_niches": 2,
        "add_to_obs": False,
        "return_model": True,
        "random_state": 11,
    }

    niches_a, _, model_a = detect_spatial_niches_from_posteriors(
        table, posterior, **kwargs
    )
    niches_b, _, model_b = detect_spatial_niches_from_posteriors(
        table, posterior, **kwargs
    )

    pd.testing.assert_frame_equal(niches_a, niches_b)
    assert np.allclose(model_a.cluster_centers_, model_b.cluster_centers_)


@pytest.mark.parametrize(
    ("clustering_method", "expected_model"),
    [("kmeans", "KMeans"), ("minibatch", "MiniBatchKMeans"), ("auto", "KMeans")],
)
def test_clustering_methods(clustering_method, expected_model):
    table, posterior = _multi_sample_data()

    niches, smoothed, diagnostics, model = detect_spatial_niches_from_posteriors(
        table,
        posterior,
        smooth=False,
        n_niches=2,
        clustering_method=clustering_method,
        prediction_chunk_size=11,
        add_to_obs=False,
        return_diagnostics=True,
        return_model=True,
        random_state=5,
    )

    assert type(model).__name__ == expected_model
    assert diagnostics["selected_clustering_method"] == (
        "minibatch" if expected_model == "MiniBatchKMeans" else "kmeans"
    )
    assert diagnostics["matrix_dtype"] == "float32"
    assert len(niches) == len(smoothed) == 140
    assert np.array_equal(
        model.predict(smoothed).astype(str),
        niches["niche"].astype(str).to_numpy(),
    )


def test_auto_clustering_uses_minibatch_at_threshold():
    n_locations = 50_000
    table = ad.AnnData(
        X=np.ones((n_locations, 1), dtype=np.float32),
        obs=pd.DataFrame(index=[f"spot_{index}" for index in range(n_locations)]),
        var=pd.DataFrame(index=["G1"]),
    )
    table.obsm["spatial"] = np.column_stack(
        [np.arange(n_locations, dtype=np.float32), np.zeros(n_locations)]
    )
    first = np.linspace(0.05, 0.95, n_locations, dtype=np.float32)
    posterior = pd.DataFrame(
        {"A": first, "B": 1.0 - first}, index=table.obs.index
    )

    niches, smoothed, diagnostics, model = detect_spatial_niches_from_posteriors(
        table,
        posterior,
        smooth=False,
        n_niches=3,
        clustering_method="auto",
        prediction_chunk_size=7_000,
        add_to_obs=False,
        return_diagnostics=True,
        return_model=True,
    )

    assert type(model).__name__ == "MiniBatchKMeans"
    assert diagnostics["selected_clustering_method"] == "minibatch"
    assert len(niches) == len(smoothed) == n_locations


def test_chunked_prediction_matches_single_batch():
    table, posterior = _multi_sample_data()
    common = {
        "smooth": False,
        "n_niches": 2,
        "add_to_obs": False,
        "random_state": 9,
    }

    chunked, _ = detect_spatial_niches_from_posteriors(
        table, posterior, prediction_chunk_size=7, **common
    )
    single_batch, _ = detect_spatial_niches_from_posteriors(
        table, posterior, prediction_chunk_size=10_000, **common
    )

    pd.testing.assert_frame_equal(chunked, single_batch)


def test_minibatch_supports_automatic_niche_selection():
    table, posterior = _multi_sample_data()

    niches, smoothed, diagnostics, model = detect_spatial_niches_from_posteriors(
        table,
        posterior,
        smooth=False,
        auto_n_niches=True,
        n_niches_min=2,
        n_niches_max=4,
        selection_metric="inertia",
        clustering_method="minibatch",
        add_to_obs=False,
        return_diagnostics=True,
        return_model=True,
        random_state=2,
    )

    assert type(model).__name__ == "MiniBatchKMeans"
    assert all(np.isnan(diagnostics["silhouette"]))
    assert len(niches) == len(smoothed) == 140


def test_fixed_niches_never_calculates_silhouette(
    spatial_table, posterior_df, monkeypatch
):
    import sklearn.metrics

    def fail_if_called(*args, **kwargs):
        raise AssertionError("silhouette_score should not be called")

    monkeypatch.setattr(sklearn.metrics, "silhouette_score", fail_if_called)

    _, _, diagnostics = _detect(
        spatial_table,
        posterior_df,
        return_diagnostics=True,
    )

    assert np.isnan(diagnostics["silhouette"][0])


def test_verbose_reports_memory_sensitive_stages(spatial_table, posterior_df, capsys):
    _detect(spatial_table, posterior_df, verbose=True, prediction_chunk_size=2)

    output = capsys.readouterr().out
    assert "Aligning and converting posterior matrix" in output
    assert "Final model fitting completed" in output
    assert "Full-data prediction completed" in output
    assert "Constructing output DataFrames" in output


def test_obs_assignment_preserves_table_without_merge(spatial_table, posterior_df):
    original_index = spatial_table.obs.index.copy()
    partial = posterior_df.drop(index="spot_e")

    niches, _ = detect_spatial_niches_from_posteriors(
        spatial_table,
        partial,
        smooth=False,
        n_niches=2,
    )

    assert spatial_table.obs.index.equals(original_index)
    assert spatial_table.obs.loc[niches.index, "niche"].notna().all()
    assert pd.isna(spatial_table.obs.loc["spot_e", "niche"])


@pytest.mark.parametrize(
    ("return_diagnostics", "return_model", "expected_length"),
    [(False, False, 2), (True, False, 3), (False, True, 3), (True, True, 4)],
)
def test_return_model_signatures_and_prediction(
    spatial_table,
    posterior_df,
    return_diagnostics,
    return_model,
    expected_length,
):
    result = _detect(
        spatial_table,
        posterior_df,
        return_diagnostics=return_diagnostics,
        return_model=return_model,
    )

    assert len(result) == expected_length
    if return_model:
        niches, smoothed, model = result[0], result[1], result[-1]
        predicted = model.predict(smoothed).astype(str)
        assert np.array_equal(predicted, niches["niche"].astype(str).to_numpy())
        assert model.feature_names_in_.tolist() == smoothed.columns.tolist()


def test_auto_niches_uses_fit_subset_and_predicts_full_data():
    table, posterior = _multi_sample_data()

    niches, smoothed, diagnostics = detect_spatial_niches_from_posteriors(
        table,
        posterior,
        sample_column="sample",
        balance_samples=True,
        max_fit_per_sample=10,
        smooth=False,
        auto_n_niches=True,
        n_niches_min=2,
        n_niches_max=5,
        silhouette_sample_size=10,
        add_to_obs=False,
        return_diagnostics=True,
        random_state=3,
    )

    assert diagnostics["n_fit_locations"] == 20
    assert diagnostics["silhouette_sample_size"] == 10
    assert max(diagnostics["candidate_k"]) <= 20
    assert len(niches) == len(smoothed) == 140


def test_inertia_selection_uses_an_elbow_not_the_largest_candidate():
    rng = np.random.default_rng(4)
    values = np.vstack(
        [
            rng.normal(center, 0.015, size=(30, 2))
            for center in ([0.9, 0.1], [0.5, 0.5], [0.1, 0.9])
        ]
    )
    table = ad.AnnData(
        X=np.ones((len(values), 1)),
        obs=pd.DataFrame(index=[f"spot_{index}" for index in range(len(values))]),
        var=pd.DataFrame(index=["G1"]),
    )
    table.obsm["spatial"] = np.column_stack(
        [np.arange(len(values), dtype=float), np.zeros(len(values))]
    )
    posterior = pd.DataFrame(values, index=table.obs.index, columns=["A", "B"])

    _, _, diagnostics = detect_spatial_niches_from_posteriors(
        table,
        posterior,
        smooth=False,
        auto_n_niches=True,
        n_niches_min=2,
        n_niches_max=6,
        selection_metric="inertia",
        add_to_obs=False,
        return_diagnostics=True,
        random_state=4,
    )

    assert diagnostics["chosen_k"] < 6


def test_detect_spatial_niches_rejects_all_zero_input(spatial_table):
    posterior = pd.DataFrame(
        0.0,
        index=spatial_table.obs.index,
        columns=["A", "B"],
    )

    with pytest.raises(ValueError, match="contains only zero values"):
        _detect(spatial_table, posterior)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"sample_column": "missing"}, ValueError, "was not found"),
        ({"balance_samples": 1}, TypeError, "must be a bool"),
        ({"max_fit_per_sample": 0, "sample_column": "sample"}, ValueError, ">= 1"),
        ({"silhouette_sample_size": 1}, ValueError, ">= 2"),
        ({"selection_metric": "invalid"}, ValueError, "selection_metric"),
        ({"clustering_method": "invalid"}, ValueError, "clustering_method"),
        ({"smoothing_chunk_size": 0}, ValueError, "smoothing_chunk_size"),
        ({"prediction_chunk_size": 0}, ValueError, "prediction_chunk_size"),
        ({"balance_samples": True}, ValueError, "requires sample_column"),
        ({"max_fit_per_sample": 10}, ValueError, "requires sample_column"),
    ],
)
def test_multi_sample_parameter_validation(spatial_table, posterior_df, kwargs, error, match):
    if kwargs.get("sample_column") == "sample":
        spatial_table.obs["sample"] = "A"
    with pytest.raises(error, match=match):
        _detect(spatial_table, posterior_df, **kwargs)


def test_multi_sample_rejects_missing_sample_identifiers(spatial_table, posterior_df):
    spatial_table.obs["sample"] = ["A", "A", np.nan, "B", "B", "B"]

    with pytest.raises(ValueError, match="missing sample identifiers"):
        _detect(spatial_table, posterior_df, sample_column="sample")


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

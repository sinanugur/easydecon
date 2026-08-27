import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import easydecon as ed
from easydecon.config import config
from easydecon._validation import AGGREGATION_METHODS
from easydecon.easydecon import (
    _aggregate_marker_expression,
    _build_permutation_gene_pool,
    _get_detected_gene_mask,
    _resolve_permutation_gene_pool_size,
    _sample_permutation_genes,
    common_markers_gene_expression_and_filter,
)


def _permutation_table():
    rng = np.random.default_rng(42)
    expression = rng.gamma(shape=2.0, scale=2.0, size=(30, 16))
    expression[:15, 0] += 7.0
    expression[:15, 1] += 4.0
    expression[15:, 2] += 7.0
    expression[15:, 3] += 4.0
    return ad.AnnData(
        X=expression,
        obs=pd.DataFrame(index=[f"spot_{index}" for index in range(30)]),
        var=pd.DataFrame(index=[f"G{index}" for index in range(16)]),
    )


def _run_permutation(
    table,
    random_state,
    pool_fraction="auto",
    aggregation_method="sum",
    coverage_power=0.5,
):
    diagnostics = {}
    result = common_markers_gene_expression_and_filter(
        table,
        {"A": ["G0", "G1"], "B": ["G2", "G3"]},
        filtering_algorithm="permutation",
        num_permutations=30,
        n_subs=3,
        subsample_size=18,
        subsample_signal_quantile=0.0,
        permutation_gene_pool_fraction=pool_fraction,
        aggregation_method=aggregation_method,
        coverage_power=coverage_power,
        parametric=False,
        add_to_obs=False,
        random_state=random_state,
        verbose=False,
        _diagnostics_out=diagnostics,
    )
    return result, diagnostics


def test_vectorized_coverage_score_math_and_index():
    expression = pd.DataFrame(
        [
            [4.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [4.0, 2.0, 1.0, 3.0],
        ],
        index=["partial", "zero", "full"],
    )

    result = _aggregate_marker_expression(expression, "coverage")

    assert result.index.equals(expression.index)
    assert result.shape == (3,)
    np.testing.assert_allclose(
        result.to_numpy(),
        [3.0 * np.sqrt(0.5), 0.0, 2.5],
    )
    assert _aggregate_marker_expression(
        expression.iloc[[0]], "coverage", coverage_power=0.0
    ).iloc[0] == pytest.approx(3.0)
    assert _aggregate_marker_expression(
        expression.iloc[[0]], "coverage", coverage_power=1.0
    ).iloc[0] == pytest.approx(1.5)


def test_vectorized_coverage_matches_reference_implementation():
    expression = pd.DataFrame(
        [
            [4.0, 2.0, 0.0, 0.0],
            [0.0, 3.0, 5.0, 0.0],
            [1.0, -2.0, 0.0, 4.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        index=["a", "b", "negative", "zero"],
    )

    def reference_coverage(row, power):
        positive = row[row > 0]
        if len(row) == 0 or len(positive) == 0:
            return 0.0
        return positive.mean() * (len(positive) / len(row)) ** power

    for power in (0.0, 0.25, 0.5, 1.0):
        expected = expression.apply(
            reference_coverage,
            axis=1,
            power=power,
        )
        actual = _aggregate_marker_expression(
            expression,
            "coverage",
            coverage_power=power,
        )
        np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy())
        assert actual.index.equals(expected.index)


@pytest.mark.parametrize("method", ["sum", "mean", "median"])
def test_vectorized_helper_preserves_builtin_aggregations(method):
    expression = pd.DataFrame(
        [[1.0, 2.0, 3.0], [0.0, 4.0, 8.0]],
        index=["first", "second"],
    )

    expected = expression.agg(method, axis=1)
    actual = _aggregate_marker_expression(expression, method)

    pd.testing.assert_series_equal(actual, expected)


@pytest.mark.parametrize("coverage_power", [-0.1, 1.1, True])
def test_invalid_coverage_power_raises(coverage_power):
    with pytest.raises(ValueError, match="coverage_power must be between 0 and 1"):
        common_markers_gene_expression_and_filter(
            None,
            [],
            coverage_power=coverage_power,
        )


def test_coverage_replaces_cs_in_allowed_aggregation_methods():
    assert "coverage" in AGGREGATION_METHODS
    assert "cs" not in AGGREGATION_METHODS

    with pytest.raises(ValueError, match="aggregation_method must be one of"):
        common_markers_gene_expression_and_filter(
            None,
            [],
            aggregation_method="cs",
        )


def test_permutation_is_reproducible_for_same_random_state():
    table = _permutation_table()

    first_result, first_diagnostics = _run_permutation(table, 10)
    second_result, second_diagnostics = _run_permutation(table, 10)

    pd.testing.assert_frame_equal(first_result, second_result)
    assert first_diagnostics == second_diagnostics


def test_permutation_null_diagnostics_can_change_with_seed():
    table = _permutation_table()

    _, first_diagnostics = _run_permutation(table, 10)
    _, second_diagnostics = _run_permutation(table, 11)

    first_group = first_diagnostics["groups"]["A"]
    second_group = second_diagnostics["groups"]["A"]
    first_summary = (
        first_group["null_median"],
        first_group["null_q95"],
        first_group["null_q99"],
        first_group["threshold"],
    )
    second_summary = (
        second_group["null_median"],
        second_group["null_q95"],
        second_group["null_q99"],
        second_group["threshold"],
    )
    assert first_summary != second_summary


@pytest.mark.parametrize("pool_fraction", ["auto", 0.3])
def test_auto_and_numeric_permutation_pool_settings_run(pool_fraction):
    result, diagnostics = _run_permutation(
        _permutation_table(),
        random_state=10,
        pool_fraction=pool_fraction,
    )

    assert result.shape == (30, 2)
    expected_mode = "auto" if pool_fraction == "auto" else "numeric"
    assert diagnostics["groups"]["A"]["gene_pool_mode"] == expected_mode


def test_coverage_aggregation_runs_with_permutation_filtering():
    first_result, first_diagnostics = _run_permutation(
        _permutation_table(),
        random_state=10,
        aggregation_method="coverage",
        coverage_power=0.25,
    )
    second_result, second_diagnostics = _run_permutation(
        _permutation_table(),
        random_state=10,
        aggregation_method="coverage",
        coverage_power=0.25,
    )

    assert first_result.shape == (30, 2)
    assert np.isfinite(first_result.to_numpy()).all()
    pd.testing.assert_frame_equal(first_result, second_result)
    assert first_diagnostics == second_diagnostics
    assert first_diagnostics["aggregation_method"] == "coverage"
    assert first_diagnostics["coverage_power"] == 0.25


def test_coverage_aggregation_is_rejected_by_nb_filtering():
    table = _permutation_table()
    table.layers["counts"] = np.asarray(table.X).copy()

    with pytest.raises(
        ValueError,
        match="NB filtering currently supports aggregation_method='sum' only",
    ):
        common_markers_gene_expression_and_filter(
            table,
            {"A": ["G0", "G1"]},
            filtering_algorithm="nb",
            aggregation_method="coverage",
            add_to_obs=False,
            verbose=False,
        )


@pytest.mark.parametrize("pool_fraction", ["foobar", 0, -0.1, 1.1])
def test_invalid_permutation_pool_settings_raise(pool_fraction):
    with pytest.raises(
        ValueError,
        match="permutation_gene_pool_fraction must be 'auto' or a numeric value",
    ):
        common_markers_gene_expression_and_filter(
            None,
            [],
            permutation_gene_pool_fraction=pool_fraction,
        )


def test_group_pool_excludes_target_markers_and_undetected_genes():
    var_names = pd.Index(["M1", "M2", "ZERO", "N1", "N2"])
    variability = np.array([100.0, 90.0, 80.0, 2.0, 1.0])
    detected = np.array([True, True, False, True, True])

    gene_pool, pool_info = _build_permutation_gene_pool(
        var_names,
        variability,
        detected,
        ["M1", "M2"],
        "auto",
        group_name="A",
    )

    assert set(gene_pool) == {"N1", "N2"}
    assert "M1" not in gene_pool
    assert "M2" not in gene_pool
    assert "ZERO" not in gene_pool
    assert pool_info["n_detected_genes"] == 4
    assert pool_info["n_eligible_null_genes"] == 2


def test_detected_gene_mask_is_sparse_safe():
    matrix = sparse.csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        )
    )

    assert _get_detected_gene_mask(matrix).tolist() == [False, True, True]


def test_auto_pool_scales_with_marker_set_size():
    small_marker_set = _resolve_permutation_gene_pool_size(10_000, 20, "auto")
    large_marker_set = _resolve_permutation_gene_pool_size(10_000, 60, "auto")

    assert small_marker_set["pool_size"] <= large_marker_set["pool_size"]
    assert small_marker_set["pool_size"] == 500
    assert large_marker_set["pool_size"] == 1_200


def test_auto_pool_respects_available_panel_size():
    small_panel = _resolve_permutation_gene_pool_size(1_000, 20, "auto")
    large_panel = _resolve_permutation_gene_pool_size(20_000, 20, "auto")

    assert small_panel["pool_size"] == 500
    assert large_panel["pool_size"] == 1_000
    assert small_panel["effective_fraction"] == 0.5
    assert large_panel["effective_fraction"] == 0.05


def test_rng_progresses_between_null_gene_draws():
    rng = np.random.default_rng(10)
    gene_pool = np.arange(100)

    first = _sample_permutation_genes(rng, gene_pool, 10)
    second = _sample_permutation_genes(rng, gene_pool, 10)

    assert not np.array_equal(first, second)


def test_workflow_exposes_phase1_permutation_diagnostics(monkeypatch):
    monkeypatch.setattr(config, "n_jobs", 1)
    table = _permutation_table()
    markers = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "names": ["G0", "G1", "G2", "G3"],
            "scores": [4.0, 3.0, 4.0, 3.0],
            "logfoldchanges": [2.0, 1.5, 2.0, 1.5],
            "pvals_adj": [0.001, 0.002, 0.001, 0.002],
        }
    )

    result = ed.easydecon_workflow(
        table,
        markers_df=markers,
        aggregation_method="coverage",
        coverage_power=0.25,
        filtering_algorithm="permutation",
        permutation_gene_pool_fraction="auto",
        random_state=10,
        num_permutations=12,
        n_subs=2,
        subsample_size=12,
        parametric=False,
        method="sum",
        min_markers=1,
        return_result_object=True,
        verbose=False,
    )

    phase1 = result.diagnostics["phase1"]
    assert phase1["aggregation_method"] == "coverage"
    assert phase1["coverage_power"] == 0.25
    assert phase1["permutation_gene_pool_fraction"] == "auto"
    assert phase1["random_state"] == 10
    assert phase1["performance"]["filtering_algorithm"] == "permutation"
    assert phase1["performance"]["aggregation_method"] == "coverage"
    assert phase1["performance"]["coverage_power"] == 0.25
    assert set(phase1["performance"]["groups"]) == {"A", "B"}
    assert "threshold" in phase1["performance"]["groups"]["A"]

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import easydecon as ed
from easydecon.config import config
from easydecon.easydecon import (
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


def _run_permutation(table, random_state, pool_fraction="auto"):
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
        parametric=False,
        add_to_obs=False,
        random_state=random_state,
        verbose=False,
        _diagnostics_out=diagnostics,
    )
    return result, diagnostics


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
    assert phase1["permutation_gene_pool_fraction"] == "auto"
    assert phase1["random_state"] == 10
    assert phase1["performance"]["filtering_algorithm"] == "permutation"
    assert set(phase1["performance"]["groups"]) == {"A", "B"}
    assert "threshold" in phase1["performance"]["groups"]["A"]

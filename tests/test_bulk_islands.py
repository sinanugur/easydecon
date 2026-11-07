import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from easydecon.bulk import BulkSignatureResult
from easydecon.bulk_islands import (
    BulkIslandResult,
    detect_bulk_islands,
    plot_bulk_islands,
)


def _bulk_result(index, up=1.0, down=0.0, detected=4):
    n = len(index)
    up = np.broadcast_to(up, n).astype(float).copy()
    down = np.broadcast_to(down, n).astype(float).copy()
    detected = np.broadcast_to(detected, n).astype(int).copy()
    scores = pd.DataFrame(
        {
            "up_score": up,
            "down_score": down,
            "balance_score": up - down,
            "up_detected_genes": detected,
            "down_detected_genes": detected,
            "up_detected_fraction": detected / 4,
            "down_detected_fraction": detected / 4,
            "all_zero": False,
        },
        index=pd.Index(index),
    )
    return BulkSignatureResult(
        scores=scores,
        selected_genes=pd.DataFrame(),
        diagnostics={},
        parameters={
            "signature_name": "synthetic",
            "rank_cap": 100,
            "scoring_definition": "test ranks",
        },
    )


def _detect(result, coordinates, samples, **kwargs):
    settings = {
        "score_threshold": 0.5,
        "max_neighbor_distance": 1.5,
        "min_detected_genes": 3,
        "n_neighbors": 3,
        "min_candidate_neighbors": 1,
        "min_candidate_neighbor_fraction": 0.3,
        "min_island_cells": 2,
    }
    settings.update(kwargs)
    return detect_bulk_islands(
        result,
        coordinates=coordinates,
        sample_labels=samples,
        **settings,
    )


def test_separate_islands_and_long_edges_are_not_connected():
    index = [f"cell_{i}" for i in range(8)]
    coordinates = np.array(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
            [10, 0],
            [11, 0],
            [10, 1],
            [11, 1],
        ],
        dtype=float,
    )
    result = _detect(_bulk_result(index), coordinates, ["S1"] * 8)

    assert result.island_summary["n_observations"].tolist() == [4, 4]
    assert result.observations["island_id"].nunique() == 2
    assert result.diagnostics["n_retained_islands"] == 2
    assert set(result.observations["assignment_status"]) == {"assigned"}


def test_identical_coordinates_in_different_specimens_never_connect():
    index = ["a1", "a2", "b1", "b2"]
    result = _detect(
        _bulk_result(index),
        np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=float),
        ["A", "A", "B", "B"],
        n_neighbors=1,
    )

    assert result.island_summary.groupby("sample")["n_observations"].sum().to_dict() == {
        "A": 2,
        "B": 2,
    }
    assert result.observations["island_id"].nunique() == 2


def test_low_scoring_bridge_does_not_merge_islands():
    index = [f"p{i}" for i in range(7)]
    scores = [1, 1, 1, 0.1, 1, 1, 1]
    result = _detect(
        _bulk_result(index, up=scores),
        np.column_stack([np.arange(7), np.zeros(7)]),
        ["S"] * 7,
        max_neighbor_distance=1.1,
        n_neighbors=2,
    )

    assert result.observations.loc["p3", "assignment_status"] == "below_score_threshold"
    assert result.observations["island_id"].nunique() == 2
    assert result.island_summary["n_observations"].tolist() == [3, 3]


def test_detection_neighbor_support_and_small_component_statuses():
    index = ["low_detection", "neighbor", "isolated", "small_a", "small_b"]
    result = _bulk_result(index, detected=[1, 4, 4, 4, 4])
    detected = _detect(
        result,
        np.array([[0, 0], [1, 0], [20, 0], [40, 0], [41, 0]], dtype=float),
        ["S"] * 5,
        min_island_cells=3,
    )

    status = detected.observations["assignment_status"]
    assert status["low_detection"] == "insufficient_detection"
    assert status["neighbor"] == "insufficient_neighbor_support"
    assert status["isolated"] == "insufficient_neighbor_support"
    assert status["small_a"] == "too_small_component"
    assert status["small_b"] == "too_small_component"
    assert detected.island_summary.empty


def test_nan_duplicate_coordinates_and_singleton_are_safe():
    index = ["dup1", "dup2", "dup3", "nan", "singleton"]
    result = _bulk_result(index, up=[1, 1, 1, np.nan, 1])
    detected = _detect(
        result,
        np.array([[0, 0], [0, 0], [0, 0], [1, 0], [50, 0]], dtype=float),
        ["A", "A", "A", "A", "B"],
        n_neighbors=2,
    )

    assert detected.observations.loc["nan", "assignment_status"] == "unscored"
    assert detected.observations.loc["singleton", "neighbor_count"] == 0
    assert detected.observations.loc["singleton", "assignment_status"] == (
        "insufficient_neighbor_support"
    )
    assert detected.island_summary["n_observations"].tolist() == [3]


def test_empty_all_ineligible_result_has_stable_tables_and_diagnostics():
    result = _bulk_result(["a", "b"], up=[np.nan, 0.1])
    detected = _detect(result, [[0, 0], [1, 0]], ["S", "S"])

    assert isinstance(detected, BulkIslandResult)
    assert detected.island_summary.columns.tolist() == [
        "island_id",
        "sample",
        "program",
        "n_observations",
        "mean_score",
        "median_score",
        "mean_balance_score",
        "mean_detected_genes",
        "centroid_x",
        "centroid_y",
        "min_x",
        "max_x",
        "min_y",
        "max_y",
    ]
    assert detected.island_summary.empty
    assert detected.niche_composition.empty
    assert detected.diagnostics["n_candidates"] == 0
    assert detected.diagnostics["n_retained_islands"] == 0


def test_pandas_inputs_align_by_id_and_bad_ids_are_rejected():
    index = pd.Index(["a", "b", "c"])
    result = _bulk_result(index)
    coordinates = pd.DataFrame(
        {"x": [2.0, 0.0, 1.0], "y": [0.0, 0.0, 0.0]},
        index=["c", "a", "b"],
    )
    samples = pd.Series(["S", "S", "S"], index=["b", "c", "a"])
    detected = _detect(result, coordinates, samples)

    assert detected.observations["x"].to_dict() == {"a": 0, "b": 1, "c": 2}
    duplicate = coordinates.copy()
    duplicate.index = ["a", "a", "c"]
    with pytest.raises(ValueError, match="unique"):
        _detect(result, duplicate, samples)
    with pytest.raises(ValueError, match="exactly"):
        _detect(result, coordinates.drop(index="c"), samples)
    bad_result = _bulk_result(["a", "a"])
    with pytest.raises(ValueError, match="unique"):
        _detect(bad_result, [[0, 0], [1, 0]], ["S", "S"])


def test_inputs_unchanged_results_deterministic_and_parameters_snapshotted():
    index = ["a", "b", "c"]
    bulk = _bulk_result(index)
    coordinates = pd.DataFrame([[0, 0], [1, 0], [2, 0]], index=index)
    samples = pd.Series("S", index=index)
    scores_before = bulk.scores.copy(deep=True)
    coordinates_before = coordinates.copy(deep=True)
    samples_before = samples.copy(deep=True)

    first = _detect(bulk, coordinates, samples)
    second = _detect(bulk, coordinates, samples)

    pd.testing.assert_frame_equal(first.observations, second.observations)
    pd.testing.assert_frame_equal(first.island_summary, second.island_summary)
    pd.testing.assert_frame_equal(bulk.scores, scores_before)
    pd.testing.assert_frame_equal(coordinates, coordinates_before)
    pd.testing.assert_series_equal(samples, samples_before)
    bulk.parameters["signature_name"] = "changed"
    assert first.parameters["bulk_signature_parameters"]["signature_name"] == "synthetic"


def test_niche_labels_change_only_composition_and_include_missing_category():
    index = ["a", "b", "c"]
    bulk = _bulk_result(index)
    coordinates = [[0, 0], [1, 0], [2, 0]]
    plain = _detect(bulk, coordinates, ["S"] * 3)
    annotated = detect_bulk_islands(
        bulk,
        coordinates=coordinates,
        sample_labels=["S"] * 3,
        niche_labels=pd.Series(["N1", np.nan, "N2"], index=index),
        score_threshold=0.5,
        max_neighbor_distance=1.5,
        min_detected_genes=3,
        n_neighbors=2,
        min_candidate_neighbors=1,
        min_candidate_neighbor_fraction=0.3,
        min_island_cells=2,
    )

    pd.testing.assert_series_equal(
        plain.observations["island_id"], annotated.observations["island_id"]
    )
    assert set(annotated.niche_composition["niche"]) == {"N1", "N2", "<missing>"}
    assert annotated.niche_composition["fraction"].sum() == pytest.approx(1)


def test_down_program_uses_down_scores_and_detection():
    index = ["a", "b", "c"]
    bulk = _bulk_result(index, up=0.0, down=[1, 1, 1], detected=4)
    detected = detect_bulk_islands(
        bulk,
        coordinates=[[0, 0], [1, 0], [2, 0]],
        sample_labels=["S"] * 3,
        program="down",
        score_threshold=0.5,
        max_neighbor_distance=1.5,
        min_detected_genes=3,
        n_neighbors=2,
        min_candidate_neighbors=1,
        min_candidate_neighbor_fraction=0.3,
        min_island_cells=2,
    )

    assert detected.parameters["program"] == "down"
    assert detected.observations["program_score"].eq(1).all()
    assert detected.observations["assignment_status"].eq("assigned").all()


def test_plotting_ordinary_and_empty_results():
    ordinary = _detect(
        _bulk_result(["a", "b", "c"]),
        [[0, 0], [1, 0], [2, 0]],
        ["S"] * 3,
    )
    fig, axes = plot_bulk_islands(ordinary)
    assert len(axes) == 1
    assert "Bulk-program islands" in axes[0].get_title()
    plt.close(fig)

    empty = _detect(
        _bulk_result(["a", "b"], up=0.0),
        [[0, 0], [1, 0]],
        ["S"] * 2,
    )
    fig, axes = plot_bulk_islands(empty, specimen="S")
    assert any(text.get_text() == "No retained islands" for text in axes[0].texts)
    plt.close(fig)


def test_validation_rejects_missing_columns_labels_and_bad_parameters():
    bulk = _bulk_result(["a", "b"])
    bulk.scores = bulk.scores.drop(columns="up_detected_genes")
    with pytest.raises(ValueError, match="missing required"):
        _detect(bulk, [[0, 0], [1, 0]], ["S", "S"])

    bulk = _bulk_result(["a", "b"])
    with pytest.raises(ValueError, match="missing values"):
        _detect(bulk, [[0, 0], [1, 0]], ["S", None])
    with pytest.raises(ValueError, match="greater than 0"):
        detect_bulk_islands(
            bulk,
            coordinates=[[0, 0], [1, 0]],
            sample_labels=["S", "S"],
            score_threshold=0.5,
            max_neighbor_distance=0,
        )

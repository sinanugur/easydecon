from types import SimpleNamespace

import numpy as np
import pandas as pd

from benchmarks import run_synthetic_validation as validation_cli
from benchmarks.synthetic_validation import (
    ValidationConfiguration,
    assignment_metrics,
    compare_candidate_pruning,
    confusion_counts,
    default_validation_configurations,
    make_known_marker_table,
    make_synthetic_validation_dataset,
    probability_metrics,
    run_validation_configuration,
    run_validation_suite,
    score_ranking_metrics,
    summarize_validation_results,
)


def _dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _tiny_dataset(**kwargs):
    params = {
        "scenario": "clean",
        "n_groups": 3,
        "n_reference_cells_per_group": 8,
        "n_spots_per_group": 5,
        "n_genes": 80,
        "markers_per_group": 5,
        "negative_markers_per_group": 3,
        "random_state": 0,
    }
    params.update(kwargs)
    return make_synthetic_validation_dataset(**params)


def _config(name):
    return {config.name: config for config in default_validation_configurations()}[name]


def test_generator_is_deterministic():
    first = _tiny_dataset(random_state=7)
    second = _tiny_dataset(random_state=7)

    np.testing.assert_array_equal(_dense(first.spatial.layers["counts"]), _dense(second.spatial.layers["counts"]))
    np.testing.assert_array_equal(_dense(first.reference.layers["counts"]), _dense(second.reference.layers["counts"]))
    pd.testing.assert_series_equal(first.true_labels, second.true_labels)
    pd.testing.assert_frame_equal(first.true_scores, second.true_scores)
    assert first.marker_groups == second.marker_groups


def test_generator_changes_with_seed():
    first = _tiny_dataset(random_state=7)
    second = _tiny_dataset(random_state=8)

    assert not np.array_equal(_dense(first.spatial.layers["counts"]), _dense(second.spatial.layers["counts"]))


def test_true_scores_sum_to_one_and_counts_are_nonnegative_integers():
    dataset = _tiny_dataset()

    np.testing.assert_allclose(dataset.true_scores.sum(axis=1).to_numpy(), 1.0)
    for matrix in (dataset.spatial.layers["counts"], dataset.reference.layers["counts"]):
        values = _dense(matrix)
        assert np.issubdtype(values.dtype, np.integer)
        assert (values >= 0).all()


def test_sparse_and_dense_generators_match():
    sparse_ds = _tiny_dataset(sparse=True, random_state=4)
    dense_ds = _tiny_dataset(sparse=False, random_state=4)

    np.testing.assert_array_equal(_dense(sparse_ds.spatial.layers["counts"]), _dense(dense_ds.spatial.layers["counts"]))
    np.testing.assert_array_equal(_dense(sparse_ds.reference.layers["counts"]), _dense(dense_ds.reference.layers["counts"]))


def test_marker_groups_do_not_overlap_illegally():
    dataset = _tiny_dataset()

    for group, positives in dataset.marker_groups["positive"].items():
        assert set(positives).isdisjoint(dataset.marker_groups["negative"][group])


def test_known_marker_configuration_runs_and_does_not_mutate_spatial():
    dataset = _tiny_dataset()
    before_columns = dataset.spatial.obs.columns.tolist()

    output = run_validation_configuration(dataset, _config("known_shared_ucell"))

    assert output["metrics"]["status"] == "ok"
    assert output["metrics"]["n_locations"] == dataset.spatial.n_obs
    assert dataset.spatial.obs.columns.tolist() == before_columns


def test_reference_and_phase_specific_configurations_run():
    dataset = _tiny_dataset()

    reference = run_validation_configuration(dataset, _config("reference_max_other_ucell"))
    phase_specific = run_validation_configuration(dataset, _config("known_phase_specific_ucell"))

    assert reference["metrics"]["status"] == "ok"
    assert phase_specific["metrics"]["status"] == "ok"
    assert "marker_role" in make_known_marker_table(dataset, marker_roles="phase_specific").columns


def test_candidate_pruned_configuration_runs_with_metrics():
    dataset = _tiny_dataset()

    output = run_validation_configuration(dataset, _config("reference_ucell_pruned_zero"))

    assert output["metrics"]["candidate_pruning_enabled"] is True
    assert 0 <= output["metrics"]["candidate_fraction"] <= 1


def test_metrics_have_expected_columns_and_confusion_counts_sum():
    dataset = _tiny_dataset()
    output = run_validation_configuration(dataset, _config("known_shared_ucell"))
    metrics = output["metrics"]

    for column in ("overall_accuracy", "coverage", "macro_f1", "mean_reciprocal_rank"):
        assert column in metrics
    assert output["confusion"]["count"].sum() == metrics["n_locations"]


def test_assignment_metrics_with_all_correct_unassigned_and_none_assigned():
    true = pd.Series(["A", "B", "A"], index=["s1", "s2", "s3"])

    all_correct = assignment_metrics(true, pd.Series(["A", "B", "A"], index=true.index))
    with_unassigned = assignment_metrics(true, pd.Series(["A", np.nan, "B"], index=true.index))
    none_assigned = assignment_metrics(true, pd.Series([np.nan, np.nan, np.nan], index=true.index))

    assert all_correct["overall_accuracy"] == 1.0
    assert with_unassigned["coverage"] == 2 / 3
    assert with_unassigned["overall_accuracy"] == 1 / 3
    assert none_assigned["coverage"] == 0.0
    assert none_assigned["overall_accuracy"] == 0.0
    assert np.isnan(none_assigned["assigned_accuracy"])


def test_confusion_counts_marks_unassigned():
    true = pd.Series(["A", "B"], index=["s1", "s2"])
    pred = pd.Series(["A", np.nan], index=true.index)

    counts = confusion_counts(true, pred)

    assert counts["count"].sum() == 2
    assert "<unassigned>" in counts["predicted_label"].tolist()


def test_reciprocal_rank_entropy_brier_and_log_loss():
    labels = pd.Series(["A", "B"], index=["s1", "s2"])
    scores = pd.DataFrame({"A": [0.8, 0.1], "B": [0.2, 0.9]}, index=labels.index)

    ranking = score_ranking_metrics(scores, labels)
    probs = probability_metrics(scores, labels, epsilon=1e-6)

    assert ranking["mean_reciprocal_rank"] == 1.0
    assert ranking["top1_score_accuracy"] == 1.0
    assert probs["multiclass_brier"] < 0.1
    assert probs["negative_log_likelihood"] < 0.2


def test_mixed_composition_mae_and_worst_rank_for_zero_rows():
    labels = pd.Series(["A", "B"], index=["s1", "s2"])
    scores = pd.DataFrame({"A": [0.0, 0.25], "B": [0.0, 0.75]}, index=labels.index)
    true_scores = pd.DataFrame({"A": [0.5, 0.25], "B": [0.5, 0.75]}, index=labels.index)

    ranking = score_ranking_metrics(scores, labels, true_scores=true_scores)
    composition = probability_metrics(true_scores, labels, true_scores=true_scores)

    assert ranking["mean_reciprocal_rank"] == 0.5
    assert composition["composition_mae"] == 0.0


def test_pruning_comparison_nan_assignments_compare_equal():
    posterior = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 0.0]}, index=["s1", "s2"])
    assigned = pd.DataFrame({"easydecon": ["A", np.nan]}, index=posterior.index)
    left = SimpleNamespace(posterior_df=posterior, assigned_labels=assigned, phase2_result=posterior)
    right = SimpleNamespace(posterior_df=posterior.copy(), assigned_labels=assigned.copy(), phase2_result=posterior.copy())

    comparison = compare_candidate_pruning(left, right)

    assert comparison["assignments_equal"] is True
    assert comparison["posterior_max_abs_difference"] == 0.0


def test_suite_runner_and_summary_continue_on_failures():
    bad = ValidationConfiguration(
        name="bad_reference",
        marker_source="reference",
        marker_method="reference",
        reference_contrast="max_other",
        marker_roles="shared",
        phase2_method="ucell",
        candidate_pruning=False,
        candidate_threshold=0.0,
        evidence_to_likelihood="softmax",
        extra_kwargs={"reference_min_cells": 999},
    )

    metrics, details = run_validation_suite(
        scenarios=["clean"],
        configurations=[_config("known_shared_ucell"), bad],
        random_states=(0,),
        repeat=1,
        dataset_kwargs={
            "n_groups": 3,
            "n_reference_cells_per_group": 8,
            "n_spots_per_group": 5,
            "n_genes": 80,
            "markers_per_group": 5,
        },
    )
    summary = summarize_validation_results(metrics)

    assert set(metrics["status"]) == {"ok", "failed"}
    assert not summary.empty
    assert details["confusion"]


def test_suite_records_zero_threshold_pruning_equivalence():
    metrics, details = run_validation_suite(
        scenarios=["clean"],
        configurations=["reference_ucell_unpruned", "reference_ucell_pruned_zero"],
        random_states=(0,),
        repeat=1,
        dataset_kwargs={
            "n_groups": 3,
            "n_reference_cells_per_group": 8,
            "n_spots_per_group": 5,
            "n_genes": 80,
            "markers_per_group": 5,
        },
    )

    assert set(metrics["status"]) == {"ok"}
    comparison = details["pruning_comparisons"][0]
    assert comparison["assignments_equal"] is True
    assert comparison["posterior_max_abs_difference"] < 1e-12


def test_clean_scenario_has_better_than_random_recovery():
    dataset = _tiny_dataset()
    output = run_validation_configuration(dataset, _config("known_shared_ucell"))

    assert output["metrics"]["overall_accuracy"] > 1 / dataset.metadata["n_groups"]


def test_cli_smoke_creates_outputs(tmp_path):
    validation_cli.main(
        [
            "--scenarios",
            "clean",
            "--configurations",
            "known_shared_ucell",
            "--seeds",
            "0",
            "--repeat",
            "1",
            "--n-groups",
            "3",
            "--n-reference-cells-per-group",
            "8",
            "--n-spots-per-group",
            "5",
            "--n-genes",
            "80",
            "--markers-per-group",
            "5",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert (tmp_path / "validation_metrics.csv").is_file()
    assert (tmp_path / "validation_summary.csv").is_file()
    assert (tmp_path / "validation_metadata.json").is_file()

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import easydecon as ed
import easydecon.extra as extra_module
from easydecon.config import config, set_batch_size, set_n_jobs
from easydecon.extra import EasyDeconResult
from easydecon.refinement import RefinedGroupResult


@pytest.fixture(autouse=True)
def _single_thread_config():
    old_n_jobs = config.n_jobs
    old_batch_size = config.batch_size
    set_n_jobs(1)
    set_batch_size(256)
    try:
        yield
    finally:
        set_n_jobs(old_n_jobs)
        set_batch_size(old_batch_size)


@pytest.fixture
def spatial_table():
    expression = np.array(
        [
            [10.0, 1.0, 1.0, 1.0],
            [1.0, 10.0, 1.0, 1.0],
            [8.0, 8.0, 1.0, 1.0],
            [1.0, 1.0, 10.0, 1.0],
            [1.0, 1.0, 8.0, 1.0],
            [1.0, 1.0, 1.0, 10.0],
        ],
        dtype=float,
    )
    return ad.AnnData(
        X=expression,
        obs=pd.DataFrame(index=[f"spot_{index}" for index in range(6)]),
        var=pd.DataFrame(index=["G1", "G2", "G3", "G4"]),
    )


@pytest.fixture
def subtype_markers():
    return pd.DataFrame(
        {
            "group": ["Monocyte", "Macrophage"],
            "names": ["G1", "G2"],
            "scores": [5.0, 5.0],
            "logfoldchanges": [2.0, 2.0],
            "pvals_adj": [0.001, 0.001],
        }
    )


@pytest.fixture
def parent_result(spatial_table):
    index = spatial_table.obs.index
    priors = pd.DataFrame(
        {
            "Myeloid": [1.0, 0.8, 0.6, 0.0, 0.0, 0.0],
            "Other": [0.0, 0.2, 0.4, 1.0, 1.0, 1.0],
        },
        index=index,
    )
    posterior = pd.DataFrame(
        {
            "Myeloid": [0.7, 0.5, 0.3, 0.0, 0.0, 0.0],
            "Other": [0.3, 0.5, 0.7, 1.0, 1.0, 1.0],
        },
        index=index,
    )
    assigned = pd.DataFrame(
        {"easydecon": ["Myeloid", "Myeloid", "Other", "Other", "Other", "Other"]},
        index=index,
    )
    return EasyDeconResult(
        markers_df=pd.DataFrame(),
        phase1_result=priors.copy(),
        phase2_result=posterior.copy(),
        assigned_labels=assigned,
        priors_df=priors,
        likelihoods_df=posterior.copy(),
        posterior_df=posterior,
        assignment_df=posterior.copy(),
        diagnostics={"results_column": "easydecon"},
    )


def test_refine_group_phase2_runs_only_phase2(
    spatial_table, parent_result, subtype_markers, monkeypatch
):
    def fail_phase1(*args, **kwargs):
        raise AssertionError("child Phase 1 should not run in phase2 mode")

    monkeypatch.setattr(
        extra_module,
        "common_markers_gene_expression_and_filter",
        fail_phase1,
    )

    refined = ed.refine_group(
        spatial_table,
        parent_result=parent_result,
        parent_group="Myeloid",
        markers_df=subtype_markers,
        mode="phase2",
        method="jaccard",
        min_markers=1,
        verbose=False,
    )

    assert isinstance(refined, RefinedGroupResult)
    assert refined.child_result is None
    assert refined.diagnostics["child_phase1_ran"] is False
    assert refined.diagnostics["child_phase2_ran"] is True


def test_refine_group_full_runs_phase1_and_phase2(
    spatial_table, parent_result, subtype_markers
):
    refined = ed.refine_group(
        spatial_table,
        parent_result=parent_result,
        parent_group="Myeloid",
        markers_df=subtype_markers,
        mode="full",
        filtering_algorithm="quantile",
        method="jaccard",
        min_markers=1,
        log2fc_min=-np.inf,
        pval_cutoff=1.0,
        verbose=False,
    )

    assert isinstance(refined.child_result, EasyDeconResult)
    assert refined.diagnostics["child_phase1_ran"] is True
    assert refined.child_result.posterior_df is not None


def test_refinement_is_restricted_to_parent_positive_locations(
    spatial_table, parent_result, subtype_markers
):
    refined = ed.refine_group(
        spatial_table,
        parent_result=parent_result,
        parent_group="Myeloid",
        markers_df=subtype_markers,
        mode="phase2",
        method="jaccard",
        min_markers=1,
        verbose=False,
    )

    outside = ~refined.eligible_mask
    assert (refined.conditional_df.loc[outside] == 0).all().all()
    assert (refined.absolute_df.loc[outside] == 0).all().all()
    column = refined.diagnostics["results_column"]
    assert refined.assigned_labels.loc[outside, column].isna().all()


def test_absolute_values_sum_to_parent_scores(
    spatial_table, parent_result, subtype_markers
):
    refined = ed.refine_group(
        spatial_table,
        parent_result=parent_result,
        parent_group="Myeloid",
        markers_df=subtype_markers,
        mode="phase2",
        method="jaccard",
        min_markers=1,
        verbose=False,
    )

    row_sums = refined.conditional_df.sum(axis=1)
    informative = row_sums > 0
    np.testing.assert_allclose(
        refined.absolute_df.loc[informative].sum(axis=1).to_numpy(),
        refined.parent_scores.loc[informative].to_numpy(),
    )


def test_parent_source_priors(spatial_table, parent_result, subtype_markers):
    refined = ed.refine_group(
        spatial_table,
        parent_result=parent_result,
        parent_group="Myeloid",
        markers_df=subtype_markers,
        parent_source="priors",
        method="jaccard",
        min_markers=1,
        verbose=False,
    )

    pd.testing.assert_series_equal(
        refined.parent_scores,
        parent_result.priors_df["Myeloid"],
        check_names=False,
    )


def test_parent_source_posterior(spatial_table, parent_result, subtype_markers):
    refined = ed.refine_group(
        spatial_table,
        parent_result=parent_result,
        parent_group="Myeloid",
        markers_df=subtype_markers,
        parent_source="posterior",
        method="jaccard",
        min_markers=1,
        verbose=False,
    )

    pd.testing.assert_series_equal(
        refined.parent_scores,
        parent_result.posterior_df["Myeloid"],
        check_names=False,
    )


def test_missing_parent_group_error(spatial_table, parent_result, subtype_markers):
    with pytest.raises(ValueError, match="Available groups: Myeloid, Other"):
        ed.refine_group(
            spatial_table,
            parent_result=parent_result,
            parent_group="Lymphoid",
            markers_df=subtype_markers,
            verbose=False,
        )


def test_no_eligible_locations_error(spatial_table, parent_result, subtype_markers):
    with pytest.raises(ValueError, match="No spatial locations passed"):
        ed.refine_group(
            spatial_table,
            parent_result=parent_result,
            parent_group="Myeloid",
            markers_df=subtype_markers,
            parent_threshold=2.0,
            verbose=False,
        )


def test_same_output_shape_for_both_modes(
    spatial_table, parent_result, subtype_markers
):
    phase2 = ed.refine_group(
        spatial_table,
        parent_result=parent_result,
        parent_group="Myeloid",
        markers_df=subtype_markers,
        mode="phase2",
        method="jaccard",
        min_markers=1,
        verbose=False,
    )
    full = ed.refine_group(
        spatial_table,
        parent_result=parent_result,
        parent_group="Myeloid",
        markers_df=subtype_markers,
        mode="full",
        filtering_algorithm="quantile",
        method="jaccard",
        min_markers=1,
        log2fc_min=-np.inf,
        pval_cutoff=1.0,
        verbose=False,
    )

    for result in (phase2, full):
        assert result.conditional_df.index.equals(spatial_table.obs.index)
        assert result.absolute_df.index.equals(spatial_table.obs.index)
        assert result.assigned_labels.index.equals(spatial_table.obs.index)
        assert set(result.diagnostics).issuperset(
            {
                "mode",
                "parent_group",
                "parent_source",
                "n_eligible_locations",
                "child_phase1_ran",
                "child_phase2_ran",
            }
        )


def test_phase2_mode_does_not_mutate_parent_result(
    spatial_table, parent_result, subtype_markers
):
    priors_before = parent_result.priors_df.copy(deep=True)
    posterior_before = parent_result.posterior_df.copy(deep=True)

    ed.refine_group(
        spatial_table,
        parent_result=parent_result,
        parent_group="Myeloid",
        markers_df=subtype_markers,
        mode="phase2",
        method="jaccard",
        min_markers=1,
        verbose=False,
    )

    pd.testing.assert_frame_equal(parent_result.priors_df, priors_before)
    pd.testing.assert_frame_equal(parent_result.posterior_df, posterior_before)

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import easydecon as ed
import easydecon.easydecon as edmod
import easydecon.extra as extra_module
import easydecon.refinement as refinement_module
from easydecon.config import config, set_batch_size, set_n_jobs
from easydecon.easydecon import get_clusters_by_similarity_on_tissue
from easydecon.extra import (
    _build_phase2_candidate_mask,
    _evidence_to_likelihood,
    easydecon_workflow,
)


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


def _table(sparse_x=False):
    x = np.array(
        [
            [8.0, 6.0, 0.0, 0.0],
            [0.0, 0.0, 7.0, 5.0],
            [4.0, 0.0, 3.0, 0.0],
        ],
        dtype=float,
    )
    if sparse_x:
        x = sparse.csr_matrix(x)
    return ad.AnnData(
        X=x,
        obs=pd.DataFrame(index=["spot_a", "spot_b", "spot_c"]),
        var=pd.DataFrame(index=["G1", "G2", "G3", "G4"]),
    )


def _markers(order=("A", "B")):
    genes = {"A": ["G1", "G2"], "B": ["G3", "G4"]}
    rows = []
    for group in order:
        for rank, gene in enumerate(genes[group]):
            rows.append(
                {
                    "group": group,
                    "names": gene,
                    "logfoldchanges": 3.0 - rank,
                    "scores": 3.0 - rank,
                    "pvals_adj": 0.001,
                }
            )
    return pd.DataFrame(rows)


def _priors(index=("spot_a", "spot_b", "spot_c")):
    return pd.DataFrame(
        {
            "A": [1.0, 0.0, 0.4],
            "B": [0.0, 1.0, 0.6],
        },
        index=index,
    )


def test_candidate_mask_is_built_from_positive_priors_and_preserves_order():
    mask = _build_phase2_candidate_mask(
        _priors(),
        phase2_groups=["B", "A"],
        spatial_index=["spot_a", "spot_b", "spot_c", "missing"],
    )

    assert mask.columns.tolist() == ["B", "A"]
    assert mask.loc["spot_a"].to_dict() == {"B": False, "A": True}
    assert mask.loc["spot_b"].to_dict() == {"B": True, "A": False}
    assert mask.loc["missing"].to_dict() == {"B": False, "A": False}


def test_candidate_threshold_is_strict_and_missing_groups_are_false():
    priors = pd.DataFrame({"A": [0.5, 0.51], "Extra": [1.0, 1.0]}, index=["s1", "s2"])
    mask = _build_phase2_candidate_mask(
        priors,
        phase2_groups=["A", "B"],
        spatial_index=["s1", "s2"],
        threshold=0.5,
    )

    assert mask.loc["s1", "A"] is np.False_
    assert mask.loc["s2", "A"] is np.True_
    assert mask["B"].tolist() == [False, False]


def test_noncandidate_groups_are_not_scored_and_shape_is_unchanged(monkeypatch):
    calls = []
    original = edmod.function_row_sum

    def spy(row, markers_df=None, **kwargs):
        calls.append(tuple(kwargs.get("candidate_group_positions") or ()))
        return original(row, markers_df=markers_df, **kwargs)

    monkeypatch.setattr(edmod, "function_row_sum", spy)
    mask = pd.DataFrame(
        {"A": [True, False, False], "B": [False, True, False]},
        index=_table().obs.index,
    )

    result = get_clusters_by_similarity_on_tissue(
        _table(),
        _markers(),
        method="sum",
        add_to_obs=False,
        verbose=False,
        _candidate_mask=mask,
    )

    assert calls == [(0,), (1,)]
    assert result.index.tolist() == _table().obs.index.tolist()
    assert result.columns.tolist() == ["A", "B"]
    assert result.loc["spot_a", "B"] == 0.0
    assert result.loc["spot_b", "A"] == 0.0
    assert (result.loc["spot_c"] == 0.0).all()


@pytest.mark.parametrize("method", ["cosine", "sum", "auc", "ucell", "jaccard", "overlap", "wjaccard"])
def test_candidate_entries_match_unpruned_scores_and_noncandidates_zero(method):
    kwargs = {"min_markers": 1, "expression_threshold": 0.0}
    mask = pd.DataFrame(
        {"A": [True, False, True], "B": [False, True, False]},
        index=_table().obs.index,
    )
    unpruned = get_clusters_by_similarity_on_tissue(
        _table(), _markers(), method=method, add_to_obs=False, verbose=False, **kwargs
    )
    pruned = get_clusters_by_similarity_on_tissue(
        _table(),
        _markers(),
        method=method,
        add_to_obs=False,
        verbose=False,
        _candidate_mask=mask,
        **kwargs,
    )

    for row in mask.index:
        for column in mask.columns:
            if mask.loc[row, column]:
                assert np.isclose(pruned.loc[row, column], unpruned.loc[row, column])
            else:
                assert pruned.loc[row, column] == 0.0


@pytest.mark.parametrize("sparse_x", [False, True])
def test_candidate_pruning_dense_and_sparse_equivalent(sparse_x):
    mask = pd.DataFrame(
        {"A": [True, False, True], "B": [False, True, False]},
        index=_table().obs.index,
    )
    dense = get_clusters_by_similarity_on_tissue(
        _table(False),
        _markers(),
        method="wjaccard",
        add_to_obs=False,
        verbose=False,
        _candidate_mask=mask,
    )
    maybe_sparse = get_clusters_by_similarity_on_tissue(
        _table(sparse_x),
        _markers(),
        method="wjaccard",
        add_to_obs=False,
        verbose=False,
        _candidate_mask=mask,
    )

    pd.testing.assert_frame_equal(dense, maybe_sparse)


def test_softmax_and_row_normalize_only_candidates():
    evidence = pd.DataFrame(
        {"A": [2.0, 10.0], "B": [100.0, 0.0], "C": [4.0, -5.0]},
        index=["s1", "s2"],
    )
    mask = pd.DataFrame(
        {"A": [True, False], "B": [False, False], "C": [True, False]},
        index=evidence.index,
    )

    softmax = _evidence_to_likelihood(evidence, method="softmax", candidate_mask=mask)
    row_norm = _evidence_to_likelihood(
        evidence, method="row_normalize", candidate_mask=mask
    )

    assert softmax.loc["s1", "B"] == 0.0
    assert np.isclose(softmax.loc["s1", ["A", "C"]].sum(), 1.0)
    assert (softmax.loc["s2"] == 0.0).all()
    assert row_norm.loc["s1", "B"] == 0.0
    assert np.isclose(row_norm.loc["s1", ["A", "C"]].sum(), 1.0)
    assert (row_norm.loc["s2"] == 0.0).all()


def test_candidate_mask_none_preserves_historical_likelihoods():
    evidence = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})

    old = _evidence_to_likelihood(evidence, method="softmax")
    all_candidates = _evidence_to_likelihood(
        evidence,
        method="softmax",
        candidate_mask=pd.DataFrame(True, index=evidence.index, columns=evidence.columns),
    )

    pd.testing.assert_frame_equal(old, all_candidates)


@pytest.mark.parametrize("likelihood_method", ["softmax", "row_normalize"])
def test_zero_threshold_pruning_preserves_posterior_and_assignments(
    monkeypatch, likelihood_method
):
    def fake_phase1(*args, **kwargs):
        return _priors()

    monkeypatch.setattr(
        extra_module, "common_markers_gene_expression_and_filter", fake_phase1
    )
    common = dict(
        markers_df=_markers(),
        filtering_algorithm="quantile",
        method="sum",
        evidence_to_likelihood=likelihood_method,
        min_markers=1,
        log2fc_min=-np.inf,
        pval_cutoff=1.0,
        return_result_object=True,
        verbose=False,
    )

    unpruned = easydecon_workflow(sdata=_table(), **common)
    pruned = easydecon_workflow(sdata=_table(), **common, phase2_candidate_pruning=True)

    pd.testing.assert_frame_equal(
        unpruned.posterior_df,
        pruned.posterior_df,
        check_exact=False,
        atol=1e-12,
    )
    pd.testing.assert_frame_equal(unpruned.assigned_labels, pruned.assigned_labels)


def test_positive_threshold_removes_low_prior_group_and_updates_diagnostics(monkeypatch):
    def fake_phase1(*args, **kwargs):
        return _priors()

    monkeypatch.setattr(
        extra_module, "common_markers_gene_expression_and_filter", fake_phase1
    )

    result = easydecon_workflow(
        _table(),
        markers_df=_markers(),
        filtering_algorithm="quantile",
        method="sum",
        min_markers=1,
        log2fc_min=-np.inf,
        pval_cutoff=1.0,
        phase2_candidate_pruning=True,
        phase2_candidate_threshold=0.5,
        return_result_object=True,
        verbose=False,
    )

    assert result.phase2_result.loc["spot_c", "A"] == 0.0
    assert result.likelihoods_df.loc["spot_c", "A"] == 0.0
    assert result.posterior_df.loc["spot_c", "A"] == 0.0
    assert result.posterior_df.loc["spot_c", "B"] == 1.0
    perf = result.diagnostics["phase2"]["performance"]
    assert perf["candidate_pruning_enabled"] is True
    assert perf["candidate_threshold"] == 0.5
    assert perf["exact_candidate_pruning"] is False


def test_candidate_pruning_rejects_prior_weight_zero():
    with pytest.raises(ValueError, match="requires prior_weight > 0"):
        easydecon_workflow(
            _table(),
            markers_df=_markers(),
            marker_method="dataframe",
            prior_weight=0.0,
            phase2_candidate_pruning=True,
            verbose=False,
        )


def test_candidate_pruning_rejects_list_style_marker_genes():
    with pytest.raises(ValueError, match="list-style marker_genes"):
        easydecon_workflow(
            _table(),
            markers_df=_markers(),
            marker_genes=["G1", "G2"],
            phase2_candidate_pruning=True,
            verbose=False,
        )


def test_refine_group_phase2_rejects_candidate_pruning():
    parent = extra_module.EasyDeconResult(
        markers_df=pd.DataFrame(),
        phase1_result=pd.DataFrame({"Parent": [1.0]}, index=["spot_a"]),
        phase2_result=pd.DataFrame({"Parent": [1.0]}, index=["spot_a"]),
        assigned_labels=pd.DataFrame({"easydecon": ["Parent"]}, index=["spot_a"]),
        priors_df=pd.DataFrame({"Parent": [1.0]}, index=["spot_a"]),
        likelihoods_df=pd.DataFrame({"Parent": [1.0]}, index=["spot_a"]),
        posterior_df=pd.DataFrame({"Parent": [1.0]}, index=["spot_a"]),
        assignment_df=pd.DataFrame({"Parent": [1.0]}, index=["spot_a"]),
        diagnostics={"results_column": "easydecon"},
    )
    table = ad.AnnData(
        X=np.array([[1.0, 0.0]]),
        obs=pd.DataFrame(index=["spot_a"]),
        var=pd.DataFrame(index=["G1", "G2"]),
    )

    with pytest.raises(ValueError, match="unavailable for refine_group\\(mode='phase2'\\)"):
        ed.refine_group(
            table,
            parent,
            "Parent",
            markers_df=pd.DataFrame(
                {"group": ["Child"], "names": ["G1"], "logfoldchanges": [1.0]}
            ),
            mode="phase2",
            phase2_candidate_pruning=True,
            verbose=False,
        )


def test_refine_group_full_forwards_candidate_pruning(monkeypatch):
    parent = extra_module.EasyDeconResult(
        markers_df=pd.DataFrame(),
        phase1_result=pd.DataFrame({"Parent": [1.0]}, index=["spot_a"]),
        phase2_result=pd.DataFrame({"Parent": [1.0]}, index=["spot_a"]),
        assigned_labels=pd.DataFrame({"easydecon": ["Parent"]}, index=["spot_a"]),
        priors_df=pd.DataFrame({"Parent": [1.0]}, index=["spot_a"]),
        likelihoods_df=pd.DataFrame({"Parent": [1.0]}, index=["spot_a"]),
        posterior_df=pd.DataFrame({"Parent": [1.0]}, index=["spot_a"]),
        assignment_df=pd.DataFrame({"Parent": [1.0]}, index=["spot_a"]),
        diagnostics={"results_column": "easydecon"},
    )
    table = ad.AnnData(
        X=np.array([[1.0, 0.0]]),
        obs=pd.DataFrame(index=["spot_a"]),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    captured = {}

    def fake_workflow(**kwargs):
        captured.update(kwargs)
        posterior = pd.DataFrame({"Child": [1.0]}, index=["spot_a"])
        return extra_module.EasyDeconResult(
            markers_df=pd.DataFrame(),
            phase1_result=posterior.copy(),
            phase2_result=posterior.copy(),
            assigned_labels=pd.DataFrame({"Parent_subcluster": ["Child"]}, index=["spot_a"]),
            priors_df=posterior.copy(),
            likelihoods_df=posterior.copy(),
            posterior_df=posterior.copy(),
            assignment_df=posterior.copy(),
            diagnostics={
                "markers": {},
                "marker_roles": {},
                "phase2": {
                    "performance": {
                        "candidate_pruning_enabled": True,
                        "candidate_threshold": 0.2,
                    }
                },
            },
        )

    monkeypatch.setattr(refinement_module, "easydecon_workflow", fake_workflow)

    refined = ed.refine_group(
        table,
        parent,
        "Parent",
        markers_df=pd.DataFrame(
            {"group": ["Child"], "names": ["G1"], "logfoldchanges": [1.0]}
        ),
        mode="full",
        phase2_candidate_pruning=True,
        phase2_candidate_threshold=0.2,
        verbose=False,
    )

    assert captured["phase2_candidate_pruning"] is True
    assert captured["phase2_candidate_threshold"] == 0.2
    assert refined.diagnostics["phase2_candidate_pruning"]["candidate_pruning_enabled"] is True

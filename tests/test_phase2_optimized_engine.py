import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import easydecon as ed
import easydecon.easydecon as edmod
from easydecon.config import config, set_batch_size, set_n_jobs
from easydecon.easydecon import (
    _build_phase2_cache,
    function_row_auc_specific_v2,
    function_row_ucell,
    function_row_weighted_jaccard,
    get_clusters_by_similarity_on_tissue,
)
from easydecon.markers import resolve_phase_marker_tables


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
            [10.0, 8.0, 1.0, 0.0, 30.0],
            [1.0, 0.0, 10.0, 8.0, 40.0],
            [0.0, 0.0, 0.0, 0.0, 50.0],
        ]
    )
    if sparse_x:
        x = sparse.csr_matrix(x)
    return ad.AnnData(
        X=x,
        obs=pd.DataFrame(index=["spot_a", "spot_b", "spot_noise"]),
        var=pd.DataFrame(index=["G1", "G2", "G3", "G4", "noise"]),
    )


def _markers(include_missing_group=False):
    data = {
        "group": ["A", "A", "B", "B"],
        "names": ["G1", "G2", "G3", "G4"],
        "logfoldchanges": [4.0, 3.0, 4.0, 3.0],
        "scores": [4.0, 3.0, 4.0, 3.0],
    }
    if include_missing_group:
        data["group"].append("Missing")
        data["names"].append("not_in_spatial")
        data["logfoldchanges"].append(0.5)
        data["scores"].append(0.5)
    return pd.DataFrame(data)


def _legacy_scores(table, markers, method, **kwargs):
    standardized = edmod.standardize_marker_dataframe(
        markers,
        gene_universe=table.var_names,
        top_n_genes=None,
        sort_by_column=None,
        log2fc_min=-np.inf,
        pval_cutoff=1.0,
    )
    func = {
        "correlation": edmod.function_row_spearman,
        "cosine": edmod.function_row_cosine,
        "euclidean": edmod.function_row_euclidean,
        "jaccard": edmod.function_row_jaccard,
        "overlap": edmod.function_row_overlap,
        "wjaccard": edmod.function_row_weighted_jaccard,
        "diagnostic": edmod.function_row_diagnostic,
        "sum": edmod.function_row_sum,
        "mean": edmod.function_row_mean,
        "median": edmod.function_row_median,
        "auc": edmod.function_row_auc_specific_v2,
        "ucell": edmod.function_row_ucell,
    }[method]
    rows = table.to_df()
    expected = {}
    for idx, row in rows.iterrows():
        expected[idx] = func(
            row,
            markers_df=standardized,
            gene_id_column="names",
            weight_column=kwargs.get("weight_column", "logfoldchanges"),
            similarity_by_column=kwargs.get("similarity_by_column", "logfoldchanges"),
            **kwargs,
        )
    return pd.DataFrame.from_dict(expected, orient="index")


@pytest.mark.parametrize(
    "method",
    [
        "correlation",
        "cosine",
        "euclidean",
        "jaccard",
        "overlap",
        "wjaccard",
        "sum",
        "mean",
        "median",
        "auc",
        "ucell",
    ],
)
def test_optimized_outputs_match_legacy_row_scorers(method):
    kwargs = {"min_markers": 1, "expression_threshold": 0.0}

    optimized = get_clusters_by_similarity_on_tissue(
        _table(), _markers(), method=method, add_to_obs=False, verbose=False, **kwargs
    )
    expected = _legacy_scores(_table(), _markers(), method, **kwargs)

    pd.testing.assert_frame_equal(
        optimized.reindex(columns=expected.columns),
        expected,
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
    )


def test_diagnostic_output_matches_legacy_intersections():
    optimized = get_clusters_by_similarity_on_tissue(
        _table(), _markers(), method="diagnostic", add_to_obs=False, verbose=False
    )
    expected = _legacy_scores(_table(), _markers(), "diagnostic")

    assert optimized.to_dict() == expected.to_dict()


def test_jaccard_non_marker_expression_affects_score():
    result = get_clusters_by_similarity_on_tissue(
        _table(), _markers().iloc[:2], method="jaccard", add_to_obs=False, verbose=False
    )

    assert np.isclose(result.loc["spot_a", "A"], 2 / 4)


def test_overlap_non_marker_expression_affects_score():
    result = get_clusters_by_similarity_on_tissue(
        _table(), _markers().iloc[:2], method="overlap", add_to_obs=False, verbose=False
    )

    assert np.isclose(result.loc["spot_a", "A"], 1.0)
    assert result.loc["spot_noise", "A"] == 0.0


def test_wjaccard_non_marker_expression_penalizes_score():
    clean_table = ad.AnnData(
        X=np.array([[10.0, 8.0, 0.0]]),
        obs=pd.DataFrame(index=["clean"]),
        var=pd.DataFrame(index=["G1", "G2", "noise"]),
    )
    noisy_table = ad.AnnData(
        X=np.array([[10.0, 8.0, 100.0]]),
        obs=pd.DataFrame(index=["noisy"]),
        var=pd.DataFrame(index=["G1", "G2", "noise"]),
    )
    markers = pd.DataFrame(
        {"group": ["A", "A"], "names": ["G1", "G2"], "logfoldchanges": [1.0, 1.0]}
    )

    clean = get_clusters_by_similarity_on_tissue(
        clean_table, markers, method="wjaccard", add_to_obs=False, verbose=False
    ).iloc[0, 0]
    noisy = get_clusters_by_similarity_on_tissue(
        noisy_table, markers, method="wjaccard", add_to_obs=False, verbose=False
    ).iloc[0, 0]

    assert noisy < clean


@pytest.mark.parametrize("method", ["correlation", "cosine", "sum", "auc", "ucell"])
def test_marker_union_safe_methods_ignore_non_marker_expression(method):
    base = _table()
    changed = _table()
    changed.X = np.asarray(changed.X).copy()
    changed.X[:, changed.var_names.get_loc("noise")] = [1000.0, 2000.0, 3000.0]
    kwargs = {"min_markers": 1, "expression_threshold": 0.0}

    base_result = get_clusters_by_similarity_on_tissue(
        base, _markers(), method=method, add_to_obs=False, verbose=False, **kwargs
    )
    changed_result = get_clusters_by_similarity_on_tissue(
        changed, _markers(), method=method, add_to_obs=False, verbose=False, **kwargs
    )
    cache = _build_phase2_cache(_markers(), changed.var_names, method, **kwargs)

    pd.testing.assert_frame_equal(base_result, changed_result)
    assert "noise" not in cache.expression_genes


@pytest.mark.parametrize("method", ["cosine", "sum", "jaccard", "wjaccard", "auc", "ucell"])
def test_dense_and_sparse_outputs_match(method):
    kwargs = {"min_markers": 1, "expression_threshold": 0.0}

    dense = get_clusters_by_similarity_on_tissue(
        _table(False), _markers(), method=method, add_to_obs=False, verbose=False, **kwargs
    )
    sparse_result = get_clusters_by_similarity_on_tissue(
        _table(True), _markers(), method=method, add_to_obs=False, verbose=False, **kwargs
    )

    pd.testing.assert_frame_equal(dense, sparse_result, check_exact=False, atol=1e-12)


@pytest.mark.parametrize("method", ["sum", "wjaccard", "ucell"])
def test_phase2_does_not_call_anndata_to_df(monkeypatch, method):
    def explode(*args, **kwargs):
        raise AssertionError("AnnData.to_df should not be called in Phase 2")

    monkeypatch.setattr(ad.AnnData, "to_df", explode)

    get_clusters_by_similarity_on_tissue(
        _table(), _markers(), method=method, min_markers=1, add_to_obs=False, verbose=False
    )


def test_cache_builders_run_once(monkeypatch):
    counts = {"cache": 0, "ucell": 0, "auc": 0, "wjaccard": 0}
    original_cache = edmod._build_phase2_cache
    original_ucell = edmod._build_ucell_signatures
    original_auc = edmod._build_auc_signatures
    original_wj = edmod._build_wjaccard_weights

    def cache_wrapper(*args, **kwargs):
        counts["cache"] += 1
        return original_cache(*args, **kwargs)

    def ucell_wrapper(*args, **kwargs):
        counts["ucell"] += 1
        return original_ucell(*args, **kwargs)

    def auc_wrapper(*args, **kwargs):
        counts["auc"] += 1
        return original_auc(*args, **kwargs)

    def wj_wrapper(*args, **kwargs):
        counts["wjaccard"] += 1
        return original_wj(*args, **kwargs)

    monkeypatch.setattr(edmod, "_build_phase2_cache", cache_wrapper)
    monkeypatch.setattr(edmod, "_build_ucell_signatures", ucell_wrapper)
    get_clusters_by_similarity_on_tissue(
        _table(), _markers(), method="ucell", min_markers=1, add_to_obs=False, verbose=False
    )
    assert counts["cache"] == 1
    assert counts["ucell"] == 1

    counts.update({"cache": 0, "ucell": 0, "auc": 0, "wjaccard": 0})
    monkeypatch.setattr(edmod, "_build_auc_signatures", auc_wrapper)
    get_clusters_by_similarity_on_tissue(
        _table(), _markers(), method="auc", min_markers=1, add_to_obs=False, verbose=False
    )
    assert counts["cache"] == 1
    assert counts["auc"] == 1

    counts.update({"cache": 0, "ucell": 0, "auc": 0, "wjaccard": 0})
    monkeypatch.setattr(edmod, "_build_wjaccard_weights", wj_wrapper)
    get_clusters_by_similarity_on_tissue(
        _table(), _markers(), method="wjaccard", add_to_obs=False, verbose=False
    )
    assert counts["cache"] == 1
    assert counts["wjaccard"] == 1


def test_output_alignment_and_missing_group_zero_column():
    result = get_clusters_by_similarity_on_tissue(
        _table(), _markers(include_missing_group=True), method="sum", add_to_obs=False, verbose=False
    )

    assert result.index.tolist() == _table().obs.index.tolist()
    assert result.columns.tolist() == ["A", "B", "Missing"]
    assert (result["Missing"] == 0).all()


def test_skipped_rows_are_zero():
    table = _table()
    table.obs["mask"] = [1, 0, 1]

    result = get_clusters_by_similarity_on_tissue(
        table,
        _markers(),
        common_group_name="mask",
        method="sum",
        add_to_obs=False,
        verbose=False,
    )

    assert (result.loc[table.obs["mask"] == 0] == 0).all().all()


def test_phase_specific_ucell_cache_excludes_presence():
    markers = edmod.standardize_marker_dataframe(
        pd.DataFrame(
            {
                "group": ["A", "A", "A"],
                "names": ["G1", "G2", "G3"],
                "marker_role": ["presence", "identity", "negative"],
                "logfoldchanges": [3, 2, -2],
            }
        ),
        log2fc_min=0,
        top_n_genes=None,
    )

    _, phase2, _ = resolve_phase_marker_tables(
        markers, marker_roles="phase_specific", method="ucell", require_phase1=False
    )
    cache = _build_phase2_cache(phase2, _table().var_names, method="ucell")

    assert "G1" not in cache.marker_union
    assert set(cache.marker_union) == {"G2", "G3"}


def test_phase_specific_non_ucell_cache_excludes_presence_and_negative():
    markers = edmod.standardize_marker_dataframe(
        pd.DataFrame(
            {
                "group": ["A", "A", "A"],
                "names": ["G1", "G2", "G3"],
                "marker_role": ["presence", "identity", "negative"],
                "logfoldchanges": [3, 2, -2],
            }
        ),
        log2fc_min=0,
        top_n_genes=None,
    )

    _, phase2, _ = resolve_phase_marker_tables(
        markers, marker_roles="phase_specific", method="sum", require_phase1=False
    )
    cache = _build_phase2_cache(phase2, _table().var_names, method="sum")

    assert cache.marker_union == ("G2",)


def test_direct_scorer_compatibility_without_cache():
    row = _table().to_df().iloc[0]
    markers = edmod.standardize_marker_dataframe(
        _markers(), gene_universe=_table().var_names, log2fc_min=-np.inf, pval_cutoff=1.0
    )

    assert function_row_weighted_jaccard(row, markers, gene_id_column="names")
    assert function_row_auc_specific_v2(row, markers, gene_id_column="names", min_markers=1)
    assert function_row_ucell(row, markers, gene_id_column="names", min_markers=1)


def test_workflow_phase2_performance_diagnostics():
    result = ed.run_easydecon(
        _table(),
        markers_df=_markers(),
        filtering_algorithm="quantile",
        method="sum",
        return_result_object=True,
        verbose=False,
    )

    performance = result.diagnostics["phase2"]["performance"]
    assert performance["marker_cache_used"] is True
    assert performance["extraction_strategy"] == "marker_union"
    assert performance["n_expression_genes"] == 4

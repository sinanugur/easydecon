import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from easydecon.bulk import (
    BulkSignatureResult,
    _score_programs,
    map_bulk_signature,
    plot_bulk_signature,
    prepare_bulk_signature,
)


def _bulk():
    return pd.DataFrame(
        {
            "gene": ["UP1", "UP2", "UP3", "DOWN1", "DOWN2", "DOWN3", "NOISE"],
            "log2FoldChange": [4, 3, 2, -4, -3, -2, 0.1],
            "padj": [0.01] * 7,
            "baseMean": np.arange(7),
        }
    )


def _spatial(*, as_sparse=False):
    genes = ["UP1", "UP2", "UP3", "DOWN1", "DOWN2", "DOWN3", "BG1", "BG2"]
    values = np.array(
        [
            [9, 8, 7, 0, 0, 0, 1, 2],
            [0, 0, 0, 9, 8, 7, 1, 2],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=float,
    )
    table = ad.AnnData(
        X=sparse.csr_matrix(values) if as_sparse else values,
        obs=pd.DataFrame(
            {"celltype": ["A", "B", np.nan], "sample": ["S1", "S1", "S2"]},
            index=["up_spot", "down_spot", "zero_spot"],
        ),
        var=pd.DataFrame(index=genes),
    )
    table.obsm["spatial"] = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
    return table


def test_prepare_deseq_columns_index_input_and_deterministic_top_n():
    bulk = _bulk().set_index("gene")
    prepared = prepare_bulk_signature(
        bulk,
        signature_name="tumor_vs_normal",
        gene_from_index=True,
        top_n_per_direction=2,
    )

    assert prepared.selected_genes.groupby("direction").size().to_dict() == {
        "down": 2,
        "up": 2,
    }
    assert prepared.selected_genes.query("direction == 'up'")["gene"].tolist() == [
        "UP1",
        "UP2",
    ]
    assert "baseMean" in prepared.selected_genes


def test_prepare_excludes_invalid_numeric_and_rejects_duplicates_and_bad_padj():
    bulk = _bulk()
    bulk.loc[0, "log2FoldChange"] = np.nan
    bulk.loc[1, "padj"] = np.inf
    prepared = prepare_bulk_signature(bulk, top_n_per_direction=None)
    assert prepared.diagnostics["n_excluded_invalid_numeric"] == 2

    duplicate = _bulk()
    duplicate.loc[1, "gene"] = "UP1"
    with pytest.raises(ValueError, match="unique"):
        prepare_bulk_signature(duplicate)

    bad_padj = _bulk()
    bad_padj.loc[0, "padj"] = 1.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        prepare_bulk_signature(bad_padj)


def test_map_scores_expected_direction_preserves_negative_balance_and_zero_nan():
    result = map_bulk_signature(
        _spatial(), _bulk(), min_genes=2, rank_cap=8, top_n_per_direction=None
    )

    assert result.scores.loc["up_spot", "up_score"] > result.scores.loc["up_spot", "down_score"]
    assert result.scores.loc["down_spot", "balance_score"] < 0
    assert result.scores.loc["zero_spot", "all_zero"]
    assert pd.isna(result.scores.loc["zero_spot", "up_score"])
    assert result.scores.index.equals(_spatial().obs_names)
    assert result.scores.loc["up_spot", "down_detected_genes"] == 0
    assert result.scores.loc["up_spot", "down_score"] == 0


def test_dense_sparse_equivalence_and_no_input_mutation():
    bulk = _bulk()
    before = bulk.copy(deep=True)
    table = _spatial()
    before_obs = table.obs.copy(deep=True)
    before_x = table.X.copy()
    dense = map_bulk_signature(table, bulk, min_genes=2)
    sparse_result = map_bulk_signature(_spatial(as_sparse=True), bulk, min_genes=2)

    pd.testing.assert_frame_equal(dense.scores, sparse_result.scores)
    pd.testing.assert_frame_equal(bulk, before)
    pd.testing.assert_frame_equal(table.obs, before_obs)
    np.testing.assert_array_equal(table.X, before_x)


def test_full_background_changes_rank_but_unrelated_observation_does_not():
    table = _spatial()
    base = map_bulk_signature(table, _bulk(), min_genes=2, rank_cap=8)
    with_high_background = table.copy()
    with_high_background.X[0, 6:] = 100
    changed = map_bulk_signature(with_high_background, _bulk(), min_genes=2, rank_cap=8)
    assert changed.scores.loc["up_spot", "up_score"] < base.scores.loc["up_spot", "up_score"]

    reordered = table[[1, 0, 2]].copy()
    reordered_result = map_bulk_signature(reordered, _bulk(), min_genes=2, rank_cap=8)
    pd.testing.assert_series_equal(
        base.scores.loc["up_spot"], reordered_result.scores.loc["up_spot"]
    )
    extra = ad.AnnData(
        X=np.ones((1, table.n_vars)),
        obs=pd.DataFrame(index=["unrelated_spot"]),
        var=pd.DataFrame(index=table.var_names),
    )
    extended = ad.concat([table, extra])
    extended_result = map_bulk_signature(extended, _bulk(), min_genes=2, rank_cap=8)
    pd.testing.assert_series_equal(
        base.scores.loc["up_spot"], extended_result.scores.loc["up_spot"]
    )


def test_missing_measured_and_undetected_genes_are_distinguished():
    bulk = pd.DataFrame(
        {
            "gene": ["UP1", "UP2", "MISSING", "DOWN1", "DOWN2"],
            "log2FoldChange": [3, 2, 1, -3, -2],
            "padj": [0.01] * 5,
        }
    )
    result = map_bulk_signature(_spatial(), bulk, min_genes=2, top_n_per_direction=None)
    selected = result.selected_genes.set_index("gene")
    assert not selected.loc["MISSING", "spatially_measured"]
    assert selected.loc["UP2", "available_for_scoring"]
    assert result.scores.loc["down_spot", "up_detected_genes"] == 0
    assert result.scores.loc["down_spot", "up_score"] == 0


def test_one_direction_can_be_unavailable_but_neither_fails():
    up_only = _bulk().query("log2FoldChange > 0")
    result = map_bulk_signature(_spatial(), up_only, min_genes=2)
    assert result.scores["down_score"].isna().all()
    assert result.scores["balance_score"].isna().all()

    with pytest.raises(ValueError, match="Neither signature direction"):
        map_bulk_signature(_spatial(), up_only.head(1), min_genes=2)


def test_duplicate_spatial_gene_identifiers_fail_clearly():
    table = _spatial()
    table.var_names = ["UP1", "UP1", *table.var_names[2:]]
    with pytest.raises(ValueError, match="Spatial var_names must be unique"):
        map_bulk_signature(table, _bulk(), min_genes=2)


def test_prepared_signature_reuse_keeps_original_selection():
    prepared = prepare_bulk_signature(_bulk(), top_n_per_direction=2)
    first = map_bulk_signature(_spatial(), prepared, min_genes=2)
    second_table = _spatial()[:, ["UP1", "UP2", "DOWN1", "DOWN2"]].copy()
    second = map_bulk_signature(second_table, prepared, min_genes=2)

    assert prepared.selected_genes.columns.tolist() == [
        column for column in first.selected_genes if column not in {"spatially_measured", "available_for_scoring"}
    ]
    assert first.parameters["signature_name"] == second.parameters["signature_name"]


def test_annotation_summary_and_storage_are_safe():
    table = _spatial()
    result = map_bulk_signature(
        table,
        _bulk(),
        min_genes=2,
        celltype_column="celltype",
        sample_column="sample",
        store=True,
    )
    assert isinstance(result, BulkSignatureResult)
    assert result.celltype_summary.shape[0] == 2
    assert result.diagnostics["n_missing_annotations"] == 1
    assert "easydecon_bulk_bulk_signature_up_score" in table.obs
    assert "easydecon_bulk_bulk_signature" in table.uns

    before = table.obs.copy()
    with pytest.raises(ValueError, match="overwrite"):
        map_bulk_signature(table, _bulk(), min_genes=2, store=True)
    pd.testing.assert_frame_equal(table.obs, before)
    map_bulk_signature(table, _bulk(), min_genes=2, store=True, overwrite=True)


def test_explicit_background_and_plotting():
    table = _spatial()
    result = map_bulk_signature(
        table,
        _bulk(),
        min_genes=2,
        scoring_gene_universe=table.var_names.tolist(),
        celltype_column="celltype",
    )
    fig, axes = plot_bulk_signature(
        result,
        table,
        include_celltype_summary=True,
        directional_limits=(0, 1),
        balance_limit=1,
    )
    assert len(axes) == 4
    assert fig is not None


@pytest.mark.parametrize("bad_value", [-1, np.nan, np.inf])
def test_negative_or_nonfinite_expression_is_rejected(bad_value):
    table = _spatial()
    table.X[0, 0] = bad_value
    with pytest.raises(ValueError, match="finite, nonnegative"):
        map_bulk_signature(table, _bulk(), min_genes=2)


def test_detection_is_independent_of_rank_credit_dense_and_sparse():
    dense = np.array([[10, 5, 1], [2, 2, 0], [0, 0, 0]], dtype=float)
    marker_codes = np.array([2, 1])
    programs = (np.array([0]), np.array([1]))

    dense_result = _score_programs(dense, marker_codes, programs, 2, 10)
    sparse_result = _score_programs(sparse.csr_matrix(dense), marker_codes, programs, 2, 10)
    np.testing.assert_allclose(dense_result[0], sparse_result[0], equal_nan=True)
    np.testing.assert_array_equal(dense_result[1], sparse_result[1])
    assert dense_result[0][0, 0] == 0
    assert dense_result[1][0, 0] == 1
    assert dense_result[1][1, 1] == 1
    assert dense_result[1][2].tolist() == [0, 0]
    assert np.isnan(dense_result[0][2]).all()


def test_noncanonical_sparse_duplicates_are_consolidated_without_mutation():
    duplicate = sparse.csr_matrix(
        (np.array([3.0, 3.0, 5.0]), np.array([0, 0, 1]), np.array([0, 3])),
        shape=(1, 2),
    )
    before = (duplicate.data.copy(), duplicate.indices.copy(), duplicate.indptr.copy())
    marker_codes = np.array([0])
    programs = (np.array([0]),)

    dense_result = _score_programs(np.array([[6.0, 5.0]]), marker_codes, programs, 2, 10)
    canonical_result = _score_programs(sparse.csr_matrix([[6.0, 5.0]]), marker_codes, programs, 2, 10)
    duplicate_result = _score_programs(duplicate, marker_codes, programs, 2, 10)

    np.testing.assert_allclose(dense_result[0], canonical_result[0])
    np.testing.assert_allclose(dense_result[0], duplicate_result[0])
    np.testing.assert_array_equal(dense_result[1], duplicate_result[1])
    for actual, expected in zip((duplicate.data, duplicate.indices, duplicate.indptr), before):
        np.testing.assert_array_equal(actual, expected)


def test_bulk_plot_uses_shared_limits_and_renders_missing_scores():
    result = map_bulk_signature(_spatial(), _bulk(), min_genes=2)
    figure, axes = plot_bulk_signature(result, _spatial())
    up_colored, down_colored = axes[0].collections[1], axes[1].collections[1]
    assert (up_colored.norm.vmin, up_colored.norm.vmax) == (
        down_colored.norm.vmin,
        down_colored.norm.vmax,
    )
    assert len(axes[0].collections[0].get_offsets()) == 1
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.close(figure)

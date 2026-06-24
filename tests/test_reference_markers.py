import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import easydecon as ed
from easydecon.config import config, set_batch_size, set_n_jobs
from easydecon.extra import EasyDeconResult
from easydecon.markers import (
    PreparedMarkers,
    compute_reference_profile_markers,
    prepare_markers,
    select_prepared_markers,
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


def _reference_adata(counts, groups, genes=None, layer="counts"):
    counts = np.asarray(counts, dtype=float)
    if genes is None:
        genes = [f"G{i + 1}" for i in range(counts.shape[1])]
    adata = ad.AnnData(
        X=counts.copy(),
        obs=pd.DataFrame(
            {"cell_type": groups},
            index=[f"cell_{i}" for i in range(counts.shape[0])],
        ),
        var=pd.DataFrame(index=genes),
    )
    if layer is not None:
        adata.layers[layer] = counts.copy()
    return adata


def _simple_reference():
    return _reference_adata(
        [
            [10, 30, 0],
            [100, 300, 0],
            [30, 10, 0],
            [300, 100, 0],
            [0, 0, 40],
            [0, 0, 400],
        ],
        ["A", "A", "B", "B", "C", "C"],
        ["G1", "G2", "G3"],
    )


def _loose_reference_call(adata, **kwargs):
    params = dict(
        groupby="cell_type",
        layer="counts",
        min_cells_per_group=2,
        min_mean_expression=0.0,
        min_log2fc=0.0,
        min_detection=0.0,
        min_detection_delta=0.0,
    )
    params.update(kwargs)
    return compute_reference_profile_markers(adata, **params)


def test_reference_normalization_is_per_cell():
    markers, _ = _loose_reference_call(_simple_reference())

    row = markers[(markers["group"] == "A") & (markers["names"] == "G2")].iloc[0]

    assert np.isclose(row["mean_target"], 0.75)


def test_reference_group_mean_profiles():
    markers, _ = _loose_reference_call(_simple_reference())

    row = markers[(markers["group"] == "B") & (markers["names"] == "G1")].iloc[0]

    assert np.isclose(row["mean_target"], 0.75)


def test_reference_detection_fractions():
    adata = _reference_adata(
        [[10, 0], [0, 5], [0, 3], [0, 1]],
        ["A", "A", "B", "B"],
        ["G1", "G2"],
    )

    markers, _ = _loose_reference_call(adata)
    row = markers[(markers["group"] == "A") & (markers["names"] == "G1")].iloc[0]

    assert np.isclose(row["detection_target"], 0.5)
    assert np.isclose(row["detection_other_max"], 0.0)


def test_reference_mean_other_contrast():
    markers, _ = _loose_reference_call(_simple_reference(), contrast="mean_other")

    row = markers[(markers["group"] == "A") & (markers["names"] == "G2")].iloc[0]
    expected = np.log2((0.75 + 1e-9) / (0.125 + 1e-9))

    assert np.isclose(row["log2fc_mean"], expected)
    assert np.isclose(row["logfoldchanges"], expected)


def test_reference_max_other_contrast():
    markers, _ = _loose_reference_call(_simple_reference(), contrast="max_other")

    row = markers[(markers["group"] == "A") & (markers["names"] == "G2")].iloc[0]
    expected = np.log2((0.75 + 1e-9) / (0.25 + 1e-9))

    assert np.isclose(row["log2fc_max"], expected)
    assert np.isclose(row["logfoldchanges"], expected)


def test_max_other_rejects_shared_competitor_marker():
    adata = _reference_adata(
        [
            [50, 50, 0],
            [50, 50, 0],
            [45, 0, 55],
            [45, 0, 55],
            [0, 0, 100],
            [0, 0, 100],
        ],
        ["A", "A", "B", "B", "C", "C"],
        ["shared", "specific", "competitor"],
    )

    mean_markers, _ = _loose_reference_call(
        adata,
        contrast="mean_other",
        min_log2fc=1.0,
    )
    max_markers, _ = _loose_reference_call(
        adata,
        contrast="max_other",
        min_log2fc=1.0,
    )

    assert "shared" in mean_markers.loc[mean_markers["group"] == "A", "names"].tolist()
    assert "shared" not in max_markers.loc[max_markers["group"] == "A", "names"].tolist()


def test_detection_threshold_removes_unstable_marker():
    adata = _reference_adata(
        [[10, 1], [0, 1], [0, 5], [0, 5]],
        ["A", "A", "B", "B"],
        ["G1", "G2"],
    )

    with pytest.raises(ValueError, match="produced no markers"):
        _loose_reference_call(adata, min_detection=0.75, min_detection_delta=0.1)


def test_detection_delta_removes_nonspecific_marker():
    adata = _reference_adata(
        [[10], [10], [5], [5]],
        ["A", "A", "B", "B"],
        ["G1"],
    )

    with pytest.raises(ValueError, match="produced no markers"):
        _loose_reference_call(adata, min_detection_delta=0.5)


def test_small_groups_are_skipped():
    adata = _reference_adata(
        [[10, 0], [10, 0], [0, 10], [0, 10], [5, 5]],
        ["A", "A", "B", "B", "C"],
        ["G1", "G2"],
    )

    markers, diagnostics = _loose_reference_call(adata)

    assert not markers.empty
    assert diagnostics["groups_skipped"]["C"]["reason"] == "too_few_cells"


def test_fewer_than_two_valid_groups_raises():
    adata = _reference_adata(
        [[10], [10], [0]],
        ["A", "A", "B"],
        ["G1"],
    )

    with pytest.raises(ValueError, match="at least two groups"):
        _loose_reference_call(adata)


def test_empty_cells_are_excluded_safely():
    adata = _reference_adata(
        [[10, 0], [0, 0], [0, 10], [0, 10], [10, 0]],
        ["A", "A", "B", "B", "A"],
        ["G1", "G2"],
    )

    markers, diagnostics = _loose_reference_call(adata)

    assert diagnostics["n_empty_cells_excluded"] == 1
    assert np.isfinite(markers["scores"]).all()


def test_missing_group_labels_are_excluded():
    adata = _reference_adata(
        [[10, 0], [10, 0], [0, 10], [0, 10], [5, 5]],
        ["A", "A", "B", "B", np.nan],
        ["G1", "G2"],
    )

    _, diagnostics = _loose_reference_call(adata)

    assert diagnostics["n_missing_group_cells_excluded"] == 1


def test_negative_values_raise_dense_and_sparse():
    dense = _reference_adata([[1, -1], [1, 0], [0, 1], [0, 1]], ["A", "A", "B", "B"])
    with pytest.raises(ValueError, match="negative"):
        _loose_reference_call(dense)

    sparse_adata = _reference_adata([[1, 0], [1, 0], [0, 1], [0, 1]], ["A", "A", "B", "B"])
    sparse_adata.layers["counts"] = sparse.csr_matrix([[1, -1], [1, 0], [0, 1], [0, 1]])
    with pytest.raises(ValueError, match="negative"):
        _loose_reference_call(sparse_adata)


def test_nonfinite_values_raise():
    adata = _reference_adata(
        [[1, np.nan], [1, 0], [0, 1], [0, 1]],
        ["A", "A", "B", "B"],
    )

    with pytest.raises(ValueError, match="NaN or infinite"):
        _loose_reference_call(adata)


def test_duplicate_var_names_raise():
    adata = _reference_adata(
        [[1, 0], [1, 0], [0, 1], [0, 1]],
        ["A", "A", "B", "B"],
        ["G1", "G1"],
    )

    with pytest.raises(ValueError, match="var_names must be unique"):
        _loose_reference_call(adata)


def test_mitochondrial_filtering():
    adata = _reference_adata(
        [[10, 10], [10, 10], [0, 10], [0, 10]],
        ["A", "A", "B", "B"],
        ["MT-G1", "G2"],
    )

    markers, _ = _loose_reference_call(adata, drop_mitochondrial=True)

    assert "MT-G1" not in markers["names"].tolist()


def test_ribosomal_filtering():
    adata = _reference_adata(
        [[10, 10, 10], [10, 10, 10], [0, 10, 10], [0, 10, 10]],
        ["A", "A", "B", "B"],
        ["RPS1", "RPL2", "G3"],
    )

    markers, _ = _loose_reference_call(adata, drop_ribosomal=True)

    assert "RPS1" not in markers["names"].tolist()
    assert "RPL2" not in markers["names"].tolist()


def test_output_has_canonical_schema():
    markers, _ = _loose_reference_call(_simple_reference())

    assert {"group", "names", "logfoldchanges", "scores", "marker_rank", "marker_source"}.issubset(markers.columns)
    assert "pvals_adj" not in markers.columns


def test_output_is_deterministically_sorted():
    first, _ = _loose_reference_call(_simple_reference())
    second, _ = _loose_reference_call(_simple_reference())

    assert first[["group", "names"]].to_records(index=False).tolist() == second[["group", "names"]].to_records(index=False).tolist()


def test_prepare_markers_reference():
    prepared = prepare_markers(
        _simple_reference(),
        marker_method="reference",
        groupby="cell_type",
        layer="counts",
        reference_min_cells=2,
        reference_min_mean=0,
        reference_min_log2fc=0,
        reference_min_detection=0,
        reference_min_detection_delta=0,
        verbose=False,
    )

    assert isinstance(prepared, PreparedMarkers)
    assert prepared.marker_method == "reference"
    assert prepared.source == "reference_profile"


def test_rctd_like_alias_normalizes_to_reference():
    prepared = prepare_markers(
        _simple_reference(),
        marker_method="rctd_like",
        groupby="cell_type",
        layer="counts",
        reference_min_cells=2,
        reference_min_mean=0,
        reference_min_log2fc=0,
        reference_min_detection=0,
        reference_min_detection_delta=0,
        verbose=False,
    )

    assert prepared.marker_method == "reference"


def test_reference_signature_is_deterministic():
    kwargs = dict(
        marker_method="reference",
        groupby="cell_type",
        layer="counts",
        reference_min_cells=2,
        reference_min_mean=0,
        reference_min_log2fc=0,
        reference_min_detection=0,
        reference_min_detection_delta=0,
        verbose=False,
    )

    assert prepare_markers(_simple_reference(), **kwargs).signature == prepare_markers(_simple_reference(), **kwargs).signature


def test_reference_signature_changes_with_contrast():
    common = dict(
        marker_method="reference",
        groupby="cell_type",
        layer="counts",
        reference_min_cells=2,
        reference_min_mean=0,
        reference_min_log2fc=0,
        reference_min_detection=0,
        reference_min_detection_delta=0,
        verbose=False,
    )

    first = prepare_markers(_simple_reference(), reference_contrast="mean_other", **common)
    second = prepare_markers(_simple_reference(), reference_contrast="max_other", **common)

    assert first.signature != second.signature


def test_reference_signature_changes_with_threshold():
    common = dict(
        marker_method="reference",
        groupby="cell_type",
        layer="counts",
        reference_min_cells=2,
        reference_min_mean=0,
        reference_min_detection=0,
        reference_min_detection_delta=0,
        verbose=False,
    )

    first = prepare_markers(_simple_reference(), reference_min_log2fc=0, **common)
    second = prepare_markers(_simple_reference(), reference_min_log2fc=0.5, **common)

    assert first.signature != second.signature


def test_reference_prepared_markers_are_spatial_unfiltered():
    prepared = prepare_markers(
        _simple_reference(),
        marker_method="reference",
        groupby="cell_type",
        layer="counts",
        reference_min_cells=2,
        reference_min_mean=0,
        reference_min_log2fc=0,
        reference_min_detection=0,
        reference_min_detection_delta=0,
        verbose=False,
    )

    assert {"G1", "G2", "G3"}.intersection(set(prepared.raw_markers_df["names"]))


def test_reference_prepared_markers_reuse_across_spatial_datasets():
    prepared = prepare_markers(
        _simple_reference(),
        marker_method="reference",
        groupby="cell_type",
        layer="counts",
        reference_min_cells=2,
        reference_min_mean=0,
        reference_min_log2fc=0,
        reference_min_detection=0,
        reference_min_detection_delta=0,
        verbose=False,
    )

    first = select_prepared_markers(prepared, ["G1"], log2fc_min=0, pval_cutoff=1.0)
    second = select_prepared_markers(prepared, ["G2", "G3"], log2fc_min=0, pval_cutoff=1.0)

    assert set(first["names"]) <= {"G1"}
    assert set(second["names"]) <= {"G2", "G3"}
    assert not prepared.raw_markers_df.empty


def test_read_markers_dataframe_reference():
    spatial = ad.AnnData(
        X=np.ones((2, 3)),
        obs=pd.DataFrame(index=["spot1", "spot2"]),
        var=pd.DataFrame(index=["G1", "G2", "G3"]),
    )

    markers, diagnostics = ed.read_markers_dataframe(
        spatial,
        adata=_simple_reference(),
        marker_method="reference",
        groupby="cell_type",
        layer="counts",
        reference_min_cells=2,
        reference_min_mean=0,
        reference_min_log2fc=0,
        reference_min_detection=0,
        reference_min_detection_delta=0,
        log2fc_min=0,
        pval_cutoff=1.0,
        return_diagnostics=True,
        verbose=False,
    )

    assert not markers.empty
    assert diagnostics["generated_reference_profile"] is True
    assert diagnostics["source"] == "reference_profile"


def test_run_easydecon_reference():
    spatial = ad.AnnData(
        X=np.array([[10, 1, 0], [1, 10, 0], [0, 1, 10]], dtype=float),
        obs=pd.DataFrame(index=["spot1", "spot2", "spot3"]),
        var=pd.DataFrame(index=["G1", "G2", "G3"]),
    )

    result = ed.run_easydecon(
        spatial,
        adata=_simple_reference(),
        marker_method="reference",
        groupby="cell_type",
        layer="counts",
        reference_min_cells=2,
        reference_min_mean=0,
        reference_min_log2fc=0,
        reference_min_detection=0,
        reference_min_detection_delta=0,
        filtering_algorithm="quantile",
        method="jaccard",
        min_markers=1,
        log2fc_min=0,
        pval_cutoff=1.0,
        return_result_object=True,
        verbose=False,
    )

    assert isinstance(result, EasyDeconResult)
    assert not result.markers_df.empty
    assert result.posterior_df is not None


def test_run_easydecon_reference_respects_spatial_gene_universe():
    spatial = ad.AnnData(
        X=np.ones((2, 1)),
        obs=pd.DataFrame(index=["spot1", "spot2"]),
        var=pd.DataFrame(index=["G1"]),
    )

    result = ed.run_easydecon(
        spatial,
        adata=_simple_reference(),
        marker_method="reference",
        groupby="cell_type",
        layer="counts",
        reference_min_cells=2,
        reference_min_mean=0,
        reference_min_log2fc=0,
        reference_min_detection=0,
        reference_min_detection_delta=0,
        filtering_algorithm="quantile",
        method="jaccard",
        min_markers=1,
        log2fc_min=0,
        pval_cutoff=1.0,
        return_result_object=True,
        verbose=False,
    )

    assert set(result.markers_df["names"]) <= {"G1"}


def _parent_result(spatial):
    priors = pd.DataFrame({"Parent": [1.0] * spatial.n_obs}, index=spatial.obs.index)
    return EasyDeconResult(
        markers_df=pd.DataFrame(),
        phase1_result=priors.copy(),
        phase2_result=priors.copy(),
        assigned_labels=pd.DataFrame({"easydecon": ["Parent"] * spatial.n_obs}, index=spatial.obs.index),
        priors_df=priors,
        likelihoods_df=priors.copy(),
        posterior_df=priors.copy(),
        assignment_df=priors.copy(),
        diagnostics={"results_column": "easydecon"},
    )


def test_refine_group_phase2_reference():
    spatial = ad.AnnData(
        X=np.array([[10, 1, 0], [1, 10, 0]], dtype=float),
        obs=pd.DataFrame(index=["spot1", "spot2"]),
        var=pd.DataFrame(index=["G1", "G2", "G3"]),
    )

    refined = ed.refine_group(
        spatial,
        parent_result=_parent_result(spatial),
        parent_group="Parent",
        adata=_simple_reference(),
        marker_method="reference",
        groupby="cell_type",
        layer="counts",
        reference_min_cells=2,
        reference_min_mean=0,
        reference_min_log2fc=0,
        reference_min_detection=0,
        reference_min_detection_delta=0,
        mode="phase2",
        method="jaccard",
        min_markers=1,
        log2fc_min=0,
        pval_cutoff=1.0,
        verbose=False,
    )

    assert not refined.conditional_df.empty


def test_refine_group_full_reference():
    spatial = ad.AnnData(
        X=np.array([[10, 1, 0], [1, 10, 0], [0, 1, 10]], dtype=float),
        obs=pd.DataFrame(index=["spot1", "spot2", "spot3"]),
        var=pd.DataFrame(index=["G1", "G2", "G3"]),
    )

    refined = ed.refine_group(
        spatial,
        parent_result=_parent_result(spatial),
        parent_group="Parent",
        adata=_simple_reference(),
        marker_method="reference",
        groupby="cell_type",
        layer="counts",
        reference_min_cells=2,
        reference_min_mean=0,
        reference_min_log2fc=0,
        reference_min_detection=0,
        reference_min_detection_delta=0,
        mode="full",
        filtering_algorithm="quantile",
        method="jaccard",
        min_markers=1,
        log2fc_min=0,
        pval_cutoff=1.0,
        verbose=False,
    )

    assert isinstance(refined.child_result, EasyDeconResult)

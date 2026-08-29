import anndata as ad
import numpy as np
import pandas as pd
import pytest

import easydecon as ed
from easydecon._schema import standardize_marker_dataframe
from easydecon.config import config, set_batch_size, set_n_jobs
from easydecon.markers import (
    compute_reference_profile_markers,
    prepare_markers,
    resolve_phase_marker_tables,
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


def _reference_adata():
    counts = np.array(
        [
            [100, 80, 0, 10],
            [100, 80, 0, 10],
            [0, 20, 100, 80],
            [0, 20, 100, 80],
        ],
        dtype=float,
    )
    adata = ad.AnnData(
        X=counts.copy(),
        obs=pd.DataFrame({"cell_type": ["A", "A", "B", "B"]}),
        var=pd.DataFrame(index=["A_only", "shared_A", "B_only", "B_hi"]),
    )
    adata.layers["counts"] = counts.copy()
    return adata


def _spatial_table():
    return ad.AnnData(
        X=np.array([[10, 8, 0, 1], [0, 1, 10, 8], [5, 5, 5, 5]], dtype=float),
        obs=pd.DataFrame(index=["spot_a", "spot_b", "spot_mix"]),
        var=pd.DataFrame(index=["A_only", "shared_A", "B_only", "B_hi"]),
    )


def _reference_params(**kwargs):
    params = dict(
        groupby="cell_type",
        layer="counts",
        min_cells_per_group=2,
        min_mean_expression=0,
        min_log2fc=0.1,
        min_detection=0,
        min_detection_delta=0,
        reference_presence_min_log2fc=0.1,
        reference_presence_min_detection_delta=0,
        reference_negative_min_log2fc=0.1,
        reference_negative_min_detection=0,
        reference_negative_min_detection_delta=0,
    )
    params.update(kwargs)
    return params


def _workflow_reference_params(**kwargs):
    params = dict(
        groupby="cell_type",
        layer="counts",
        reference_min_cells=2,
        reference_min_mean=0,
        reference_min_log2fc=0.1,
        reference_min_detection=0,
        reference_min_detection_delta=0,
        reference_presence_min_log2fc=0.1,
        reference_presence_min_detection_delta=0,
        reference_negative_min_log2fc=0.1,
        reference_negative_min_detection=0,
        reference_negative_min_detection_delta=0,
    )
    params.update(kwargs)
    return params


def test_shared_reference_output_is_unchanged():
    adata = _reference_adata()

    implicit, _ = compute_reference_profile_markers(adata, **_reference_params())
    explicit, _ = compute_reference_profile_markers(
        adata, **_reference_params(marker_roles="shared")
    )

    pd.testing.assert_frame_equal(implicit, explicit)
    assert "marker_role" not in explicit.columns


def test_phase_specific_reference_generates_roles():
    markers, diagnostics = compute_reference_profile_markers(
        _reference_adata(), **_reference_params(marker_roles="phase_specific")
    )

    assert {"presence", "identity", "negative"}.issubset(
        set(markers["marker_role"])
    )
    assert diagnostics["marker_roles_mode"] == "phase_specific"
    assert diagnostics["presence_markers_per_group"]["A"] > 0
    assert diagnostics["identity_markers_per_group"]["A"] > 0
    assert diagnostics["negative_markers_per_group"]["A"] > 0


def test_presence_and_identity_can_overlap():
    markers, _ = compute_reference_profile_markers(
        _reference_adata(), **_reference_params(marker_roles="phase_specific")
    )

    rows = markers[(markers["group"] == "A") & (markers["names"] == "A_only")]

    assert {"presence", "identity"}.issubset(set(rows["marker_role"]))


def test_phase_specific_marker_rank_is_per_role():
    markers, _ = compute_reference_profile_markers(
        _reference_adata(), **_reference_params(marker_roles="phase_specific")
    )

    ranks = markers.groupby(["group", "marker_role"], sort=False)["marker_rank"].min()

    assert (ranks == 1).all()


def test_negative_markers_have_positive_penalty_strength():
    markers, _ = compute_reference_profile_markers(
        _reference_adata(), **_reference_params(marker_roles="phase_specific")
    )
    negative = markers[markers["marker_role"] == "negative"]

    assert (negative["negative_log2fc"] >= 0).all()
    assert (negative["logfoldchanges"] >= 0).all()
    assert (negative["scores"] >= 0).all()


def test_reference_role_generation_is_deterministic():
    first, _ = compute_reference_profile_markers(
        _reference_adata(), **_reference_params(marker_roles="phase_specific")
    )
    second, _ = compute_reference_profile_markers(
        _reference_adata(), **_reference_params(marker_roles="phase_specific")
    )

    pd.testing.assert_frame_equal(first, second)


def test_reference_shared_and_phase_specific_signatures_differ():
    shared = prepare_markers(
        _reference_adata(),
        marker_method="reference",
        marker_roles="shared",
        verbose=False,
        **_workflow_reference_params(),
    )
    phase_specific = prepare_markers(
        _reference_adata(),
        marker_method="reference",
        marker_roles="phase_specific",
        verbose=False,
        **_workflow_reference_params(),
    )

    assert shared.signature != phase_specific.signature
    assert "marker_role" in phase_specific.raw_markers_df.columns


def test_role_threshold_change_changes_signature():
    first = prepare_markers(
        _reference_adata(),
        marker_method="reference",
        marker_roles="phase_specific",
        verbose=False,
        **_workflow_reference_params(reference_presence_min_log2fc=0.1),
    )
    second = prepare_markers(
        _reference_adata(),
        marker_method="reference",
        marker_roles="phase_specific",
        verbose=False,
        **_workflow_reference_params(reference_presence_min_log2fc=0.2),
    )

    assert first.signature != second.signature


def test_standardization_preserves_presence_identity_duplicates():
    df = pd.DataFrame(
        {
            "group": ["A", "A"],
            "names": ["G1", "G1"],
            "marker_role": ["presence", "identity"],
            "logfoldchanges": [1, 2],
        }
    )

    result = standardize_marker_dataframe(df, log2fc_min=0)

    assert result.shape[0] == 2
    assert set(result["marker_role"]) == {"presence", "identity"}


def test_standardization_deduplicates_same_role():
    df = pd.DataFrame(
        {
            "group": ["A", "A"],
            "names": ["G1", "G1"],
            "marker_role": ["identity", "identity"],
            "scores": [2, 1],
        }
    )

    result = standardize_marker_dataframe(df, top_n_genes=None)

    assert result.shape[0] == 1
    assert result.iloc[0]["scores"] == 2


def test_marker_rank_groups_by_role():
    df = pd.DataFrame(
        {
            "group": ["A", "A", "A", "A"],
            "names": ["P1", "P2", "I1", "I2"],
            "marker_role": ["presence", "presence", "identity", "identity"],
            "scores": [4, 3, 2, 1],
        }
    )

    result = standardize_marker_dataframe(df, top_n_genes=None)

    assert result.groupby(result["marker_role"])["marker_rank"].apply(list).to_dict() == {
        "presence": [1, 2],
        "identity": [1, 2],
    }


def test_negative_signed_lfc_uses_absolute_threshold():
    df = pd.DataFrame(
        {
            "group": ["A", "A"],
            "names": ["neg", "weak"],
            "marker_role": ["negative", "identity"],
            "logfoldchanges": [-2.0, 0.5],
        }
    )

    result = standardize_marker_dataframe(df, log2fc_min=1.0)

    assert result["names"].tolist() == ["neg"]


def test_unknown_role_raises():
    df = pd.DataFrame({"group": ["A"], "names": ["G1"], "marker_role": ["anti"]})

    with pytest.raises(ValueError, match="Allowed values"):
        standardize_marker_dataframe(df)


def _manual_role_markers():
    return standardize_marker_dataframe(
        pd.DataFrame(
            {
                "group": ["A", "A", "A", "A", "B", "B"],
                "names": ["A_only", "shared_A", "B_only", "B_hi", "B_only", "B_hi"],
                "marker_role": [
                    "presence",
                    "identity",
                    "negative",
                    "positive",
                    "presence",
                    "identity",
                ],
                "scores": [6, 5, 4, 3, 2, 1],
                "logfoldchanges": [6, 5, -4, 3, 2, 1],
            }
        ),
        log2fc_min=0,
        top_n_genes=None,
    )


def test_shared_without_roles_uses_same_table_for_both_phases():
    markers = standardize_marker_dataframe(
        pd.DataFrame({"group": ["A"], "names": ["G1"]}), top_n_genes=None
    )

    phase1, phase2, diagnostics = resolve_phase_marker_tables(markers)

    pd.testing.assert_frame_equal(phase1, phase2)
    assert diagnostics["phase1_n_markers"] == 1


def test_shared_with_roles_excludes_negative_from_phase1():
    phase1, _, _ = resolve_phase_marker_tables(_manual_role_markers())

    assert "negative" not in set(phase1["marker_role"])


def test_shared_ucell_includes_negative_in_phase2():
    _, phase2, _ = resolve_phase_marker_tables(_manual_role_markers(), method="ucell")

    assert "negative" in set(phase2["marker_role"])
    assert "presence" not in set(phase2["marker_role"])


def test_shared_non_ucell_excludes_negative_from_phase2():
    _, phase2, _ = resolve_phase_marker_tables(_manual_role_markers(), method="wjaccard")

    assert "negative" not in set(phase2["marker_role"])
    assert "presence" in set(phase2["marker_role"])


def test_phase_specific_phase1_uses_presence_only():
    phase1, _, _ = resolve_phase_marker_tables(
        _manual_role_markers(), marker_roles="phase_specific", method="ucell"
    )

    assert set(phase1["marker_role"]) == {"presence"}


def test_phase_specific_ucell_uses_identity_positive_and_negative():
    _, phase2, _ = resolve_phase_marker_tables(
        _manual_role_markers(), marker_roles="phase_specific", method="ucell"
    )

    assert set(phase2["marker_role"]) == {"identity", "positive", "negative"}


def test_phase_specific_non_ucell_uses_identity_and_positive_only():
    _, phase2, _ = resolve_phase_marker_tables(
        _manual_role_markers(), marker_roles="phase_specific", method="wjaccard"
    )

    assert set(phase2["marker_role"]) == {"identity", "positive"}


def test_phase_specific_missing_presence_raises():
    markers = _manual_role_markers()
    markers = markers[markers["marker_role"] != "presence"]

    with pytest.raises(ValueError, match="No Phase 1 presence markers"):
        resolve_phase_marker_tables(markers, marker_roles="phase_specific")


def test_phase_specific_missing_identity_raises():
    markers = _manual_role_markers()
    markers = markers[markers["marker_role"].isin(["presence", "negative"])]

    with pytest.raises(ValueError, match="No Phase 2 identity markers"):
        resolve_phase_marker_tables(markers, marker_roles="phase_specific", method="ucell")


def test_marker_genes_overrides_phase1_presence_requirement():
    markers = _manual_role_markers()
    markers = markers[markers["marker_role"] != "presence"]

    phase1, phase2, _ = resolve_phase_marker_tables(
        markers,
        marker_roles="phase_specific",
        method="ucell",
        require_phase1=False,
    )

    assert phase1.empty
    assert not phase2.empty


def test_top_n_is_applied_per_group_and_role():
    markers = _manual_role_markers()
    markers = pd.concat([markers, markers.assign(names=markers["names"] + "_x")])

    phase1, phase2, _ = resolve_phase_marker_tables(
        markers, marker_roles="phase_specific", method="ucell", top_n_genes=1
    )

    assert phase1.groupby([phase1["group"], phase1["marker_role"]]).size().max() == 1
    assert phase2.groupby([phase2["group"], phase2["marker_role"]]).size().max() == 1


def test_run_easydecon_phase_specific_reference_ucell():
    result = ed.run_easydecon(
        _spatial_table(),
        adata=_reference_adata(),
        marker_method="reference",
        marker_roles="phase_specific",
        method="ucell",
        filtering_algorithm="quantile",
        min_markers=1,
        log2fc_min=0,
        pval_cutoff=1.0,
        return_result_object=True,
        verbose=False,
        **_workflow_reference_params(),
    )

    assert result.posterior_df is not None
    assert set(result.diagnostics["marker_roles"]["phase1_roles"]) == {"presence"}
    assert {"identity", "negative"}.issubset(
        set(result.diagnostics["marker_roles"]["phase2_roles"])
    )


def test_run_easydecon_phase_specific_reference_non_ucell():
    result = ed.run_easydecon(
        _spatial_table(),
        adata=_reference_adata(),
        marker_method="reference",
        marker_roles="phase_specific",
        method="wjaccard",
        filtering_algorithm="quantile",
        min_markers=1,
        log2fc_min=0,
        pval_cutoff=1.0,
        return_result_object=True,
        verbose=False,
        **_workflow_reference_params(),
    )

    assert "negative" not in result.diagnostics["marker_roles"]["phase2_roles"]


def test_run_easydecon_phase_specific_prepared_markers():
    prepared = prepare_markers(
        _reference_adata(),
        marker_method="reference",
        marker_roles="phase_specific",
        verbose=False,
        **_workflow_reference_params(),
    )

    result = ed.run_easydecon(
        _spatial_table(),
        prepared_markers=prepared,
        marker_roles="phase_specific",
        method="ucell",
        filtering_algorithm="quantile",
        min_markers=1,
        log2fc_min=0,
        pval_cutoff=1.0,
        return_result_object=True,
        verbose=False,
    )

    assert result.diagnostics["markers"]["prepared_markers_used"] is True
    assert result.diagnostics["markers"]["marker_signature"] == prepared.signature


def test_run_easydecon_phase_specific_manual_marker_table():
    result = ed.run_easydecon(
        _spatial_table(),
        markers_df=_manual_role_markers(),
        marker_roles="phase_specific",
        method="ucell",
        filtering_algorithm="quantile",
        min_markers=1,
        log2fc_min=0,
        pval_cutoff=1.0,
        return_result_object=True,
        verbose=False,
    )

    assert result.diagnostics["marker_roles"]["mode"] == "phase_specific"


def test_scanpy_phase_specific_without_roles_raises():
    with pytest.raises(ValueError, match="marker_method='reference'"):
        prepare_markers(
            _reference_adata(),
            marker_method="scanpy",
            marker_roles="phase_specific",
            groupby="cell_type",
            verbose=False,
        )


def test_pydeseq2_phase_specific_without_roles_raises(monkeypatch):
    with pytest.raises(ValueError, match="marker_method='reference'"):
        prepare_markers(
            _reference_adata(),
            marker_method="pydeseq2",
            marker_roles="phase_specific",
            groupby="cell_type",
            sample_col="sample",
            verbose=False,
        )


def test_default_shared_workflow_remains_unchanged():
    kwargs = dict(
        markers_df=standardize_marker_dataframe(
            pd.DataFrame(
                {
                    "group": ["A", "B"],
                    "names": ["A_only", "B_only"],
                    "scores": [1, 1],
                    "logfoldchanges": [1, 1],
                }
            ),
            log2fc_min=0,
        ),
        filtering_algorithm="quantile",
        method="jaccard",
        min_markers=1,
        log2fc_min=0,
        pval_cutoff=1.0,
        return_result_object=True,
        verbose=False,
    )

    default = ed.run_easydecon(sdata=_spatial_table(), **kwargs)
    shared = ed.run_easydecon(sdata=_spatial_table(), **{**kwargs, "marker_roles": "shared"})

    pd.testing.assert_frame_equal(default.phase2_result, shared.phase2_result)


def test_refine_group_phase2_phase_specific_ucell():
    parent = ed.EasyDeconResult(
        markers_df=pd.DataFrame(),
        phase1_result=pd.DataFrame({"Parent": [1, 1, 1]}, index=_spatial_table().obs.index),
        phase2_result=pd.DataFrame({"Parent": [1, 1, 1]}, index=_spatial_table().obs.index),
        assigned_labels=pd.DataFrame({"easydecon": ["Parent", "Parent", "Parent"]}, index=_spatial_table().obs.index),
        priors_df=pd.DataFrame({"Parent": [1, 1, 1]}, index=_spatial_table().obs.index),
        likelihoods_df=pd.DataFrame({"Parent": [1, 1, 1]}, index=_spatial_table().obs.index),
        posterior_df=pd.DataFrame({"Parent": [1, 1, 1]}, index=_spatial_table().obs.index),
        assignment_df=pd.DataFrame({"Parent": [1, 1, 1]}, index=_spatial_table().obs.index),
        diagnostics={"results_column": "easydecon"},
    )

    refined = ed.refine_group(
        _spatial_table(),
        parent_result=parent,
        parent_group="Parent",
        markers_df=_manual_role_markers(),
        marker_roles="phase_specific",
        mode="phase2",
        method="ucell",
        min_markers=1,
        log2fc_min=0,
        pval_cutoff=1.0,
        verbose=False,
    )

    assert refined.child_result is None
    assert "presence" not in refined.diagnostics["phase2_roles"]
    assert "negative" in refined.diagnostics["phase2_roles"]

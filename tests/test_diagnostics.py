from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

from easydecon.diagnostics import (
    summarize_easydecon_result,
    summarize_marker_table,
)


def _result_like(posterior=True):
    index = ["s1", "s2", "s3"]
    markers = pd.DataFrame(
        {
            "group": ["A", "A", "A", "B"],
            "names": ["G1", "G2", "G1", "G3"],
            "marker_source": ["scanpy", "scanpy", "curated", "scanpy"],
        }
    )
    phase1 = pd.DataFrame(
        {"A": [2.0, 0.0, 1.0], "B": [0.0, 0.0, 3.0]}, index=index
    )
    phase2 = pd.DataFrame(
        {"A": [0.8, 0.1, 0.2], "B": [0.2, 0.9, 0.8]}, index=index
    )
    priors = phase1.div(phase1.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    likelihoods = phase2.copy()
    posterior_df = phase2.copy() if posterior else None
    assignment = posterior_df if posterior else phase2
    assigned = pd.DataFrame(
        {"easydecon": ["A", "B", np.nan]},
        index=index,
    )
    diagnostics = {
        "markers": {"marker_method": "scanpy", "source": "dataframe"},
        "posterior_available": posterior,
        "assignment_matrix": "posterior_df" if posterior else "phase2_result",
        "results_column": "easydecon",
        "mask_col": "easydecon_mask",
        "n_phase1_spots": 3,
        "n_phase1_celltypes": 2,
        "n_phase2_spots": 3,
        "n_phase2_celltypes": 2,
    }
    return SimpleNamespace(
        markers_df=markers,
        phase1_result=phase1,
        phase2_result=phase2,
        assigned_labels=assigned,
        priors_df=priors,
        likelihoods_df=likelihoods,
        posterior_df=posterior_df,
        assignment_df=assignment,
        diagnostics=diagnostics,
    )


def test_summarize_marker_table_basic():
    summary = summarize_marker_table(_result_like().markers_df, top_genes=2)

    assert summary["group"].tolist() == ["A", "B"]
    group_a = summary.set_index("group").loc["A"]
    assert group_a["n_markers"] == 3
    assert group_a["n_unique_genes"] == 2
    assert group_a["top_genes"] == "G1, G2"
    assert group_a["marker_sources"] == "scanpy, curated"


def test_summarize_easydecon_result_returns_dataframe():
    summary = summarize_easydecon_result(_result_like())

    assert summary.columns.tolist() == ["section", "metric", "value"]
    assert set(summary["section"]) == {
        "markers",
        "workflow",
        "matrices",
        "posterior",
        "assignments",
        "spatial_alignment",
    }


def test_summarize_easydecon_result_returns_dict():
    summary = summarize_easydecon_result(_result_like(), as_dataframe=False)

    assert {
        "markers",
        "workflow",
        "matrices",
        "posterior",
        "assignments",
    }.issubset(summary)


def test_summarize_easydecon_result_handles_missing_posterior():
    summary = summarize_easydecon_result(
        _result_like(posterior=False), as_dataframe=False
    )

    assert summary["posterior"]["available"] is False
    assert summary["posterior"]["reason_if_missing"] is not None


def test_summarize_easydecon_result_assignment_counts():
    summary = summarize_easydecon_result(_result_like(), as_dataframe=False)
    assignments = summary["assignments"]

    assert assignments["assignment_column"] == "easydecon"
    assert assignments["n_assigned"] == 2
    assert assignments["n_unassigned"] == 1
    assert np.isclose(assignments["assigned_fraction"], 2 / 3)
    assert assignments["label_counts"] == {"A": 1, "B": 1}


def test_summarize_easydecon_result_spatial_alignment():
    spatial = ad.AnnData(
        X=np.ones((3, 1)),
        obs=pd.DataFrame(index=["s1", "s2", "s4"]),
        var=pd.DataFrame(index=["G1"]),
    )

    summary = summarize_easydecon_result(
        _result_like(), sdata=spatial, as_dataframe=False
    )
    alignment = summary["spatial_alignment"]

    assert alignment["n_spatial_spots"] == 3
    assert alignment["n_markers_spots_overlap_phase1"] == 2
    assert alignment["n_markers_spots_overlap_phase2"] == 2
    assert alignment["n_markers_spots_overlap_assignment"] == 2
    assert alignment["n_markers_spots_overlap_posterior"] == 2


def test_package_exports_diagnostics_helpers():
    import easydecon as ed

    assert hasattr(ed, "summarize_easydecon_result")
    assert hasattr(ed, "summarize_marker_table")
    assert "summarize_easydecon_result" in ed.__all__
    assert "summarize_marker_table" in ed.__all__

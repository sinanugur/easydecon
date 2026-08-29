import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from easydecon.easydecon import (
    _build_one_vs_rest_pseudobulk,
    _get_adata_count_matrix,
    compute_pseudobulk_deseq_markers,
    read_markers_dataframe,
)


def _spatial_table():
    return ad.AnnData(
        X=np.ones((2, 4)),
        obs=pd.DataFrame(index=["spot1", "spot2"]),
        var=pd.DataFrame(index=["G1", "G2", "G3", "G4"]),
    )


def _markers():
    return pd.DataFrame(
        {
            "group": ["A", "A", "B"],
            "names": ["G1", "G2", "G3"],
            "scores": [5.0, 3.0, 4.0],
            "logfoldchanges": [1.0, 0.8, 1.2],
            "pvals_adj": [0.01, 0.02, 0.01],
        }
    )


def _reference_without_markers():
    groups = pd.Categorical(["A"] * 6 + ["B"] * 6)
    expression = np.log1p(
        np.array(
            [[20.0, 1.0, 2.0, 2.0]] * 6
            + [[1.0, 20.0, 2.0, 2.0]] * 6,
            dtype=float,
        )
    )
    return ad.AnnData(
        X=expression,
        obs=pd.DataFrame({"cell_type": groups}),
        var=pd.DataFrame(index=["G1", "G2", "G3", "G4"]),
    )


def _pseudobulk_reference():
    rng = np.random.default_rng(42)
    rows = []
    cell_types = []
    sample_ids = []
    for sample_index, sample in enumerate(["S1", "S2", "S3"]):
        for cell_type in ["A", "B"]:
            for _ in range(5):
                if cell_type == "A":
                    means = [35 + sample_index, 2, 6, 8]
                else:
                    means = [2, 35 + sample_index, 6, 8]
                rows.append(rng.poisson(means))
                cell_types.append(cell_type)
                sample_ids.append(sample)
    counts = np.asarray(rows, dtype=int)
    reference = ad.AnnData(
        X=np.log1p(counts.astype(float)),
        obs=pd.DataFrame(
            {"cell_type": cell_types, "sample_id": sample_ids},
            index=[f"cell_{index}" for index in range(len(rows))],
        ),
        var=pd.DataFrame(index=["G1", "G2", "G3", "G4"]),
    )
    reference.layers["counts"] = counts
    return reference


def test_read_markers_dataframe_old_style_file_input(tmp_path):
    filename = tmp_path / "markers.csv"
    _markers().to_csv(filename, index=False)

    result = read_markers_dataframe(_spatial_table(), filename=filename, verbose=False)

    assert result["names"].tolist() == ["G1", "G3", "G2"]
    assert result["marker_source"].eq("file").all()


def test_read_markers_dataframe_accepts_dataframe_without_mutating_it():
    markers = _markers()
    original = markers.copy(deep=True)

    result = read_markers_dataframe(
        _spatial_table(), markers_df=markers, verbose=False
    )

    pd.testing.assert_frame_equal(markers, original)
    assert result["marker_source"].eq("dataframe").all()


def test_read_markers_dataframe_standardizes_custom_columns():
    markers = pd.DataFrame(
        {
            "cell_type": ["A", "B"],
            "gene": ["G1", "G2"],
            "log2FoldChange": [1.2, 0.8],
            "padj": [0.01, 0.02],
        }
    )

    result = read_markers_dataframe(
        _spatial_table(),
        markers_df=markers,
        celltype="cell_type",
        gene_id_column="gene",
        verbose=False,
    )

    assert {
        "group",
        "names",
        "logfoldchanges",
        "pvals_adj",
    }.issubset(result.columns)


def test_read_markers_dataframe_filters_to_spatial_genes():
    markers = _markers()
    markers.loc[len(markers)] = ["A", "NOT_SPATIAL", 10.0, 2.0, 0.001]

    result = read_markers_dataframe(
        _spatial_table(), markers_df=markers, verbose=False
    )

    assert "NOT_SPATIAL" not in result["names"].tolist()


def test_read_markers_dataframe_returns_diagnostics():
    result, diagnostics = read_markers_dataframe(
        _spatial_table(),
        markers_df=_markers(),
        return_diagnostics=True,
        verbose=False,
    )

    assert isinstance(result, pd.DataFrame)
    assert set(diagnostics) == {
        "source",
        "n_markers",
        "n_celltypes",
        "celltypes",
        "marker_counts_per_celltype",
        "n_spatial_genes",
        "marker_method",
        "groupby",
        "generated_rank_genes_groups",
        "rank_genes_groups_key",
        "scanpy_method",
        "generated_pseudobulk_deseq",
        "pseudobulk_deseq",
        "generated_reference_profile",
        "reference_profile",
        "reference_contrast",
        "prepared_markers_used",
        "marker_signature",
        "marker_generation_reused",
        "marker_role_inference",
        "top_n_applied_by",
        "input_kind",
        "preparation",
        "selection",
    }
    assert diagnostics["source"] == "dataframe"
    assert diagnostics["n_markers"] == 3
    assert diagnostics["n_celltypes"] == 2
    assert diagnostics["marker_counts_per_celltype"] == {"A": 2, "B": 1}
    assert diagnostics["n_spatial_genes"] == 4
    assert diagnostics["marker_role_inference"] == {
        "requested_mode": "none",
        "mode": "none",
        "normalized_mode": "none",
        "requested": False,
        "applied": False,
        "existing_roles_preserved": False,
        "input_source": None,
    }
    assert diagnostics["top_n_applied_by"] == "read_markers_dataframe"
    assert diagnostics["input_kind"] == "dataframe"
    assert diagnostics["preparation"]["input_kind"] == "dataframe"
    assert diagnostics["selection"]["n_selected_markers"] == 3


def test_read_markers_dataframe_verbose_false_suppresses_output(capsys):
    read_markers_dataframe(
        _spatial_table(), markers_df=_markers(), verbose=False
    )

    assert "Unique cell types detected" not in capsys.readouterr().out


def test_read_markers_dataframe_reads_existing_rank_genes_groups():
    groups = pd.Categorical(["A"] * 6 + ["B"] * 6)
    expression = np.array(
        [[8.0, 1.0, 2.0, 2.0]] * 6
        + [[1.0, 8.0, 2.0, 2.0]] * 6,
        dtype=float,
    )
    reference = ad.AnnData(
        X=expression,
        obs=pd.DataFrame({"celltype": groups}),
        var=pd.DataFrame(index=["G1", "G2", "G3", "G4"]),
    )
    sc.tl.rank_genes_groups(reference, groupby="celltype", method="t-test")

    result = read_markers_dataframe(
        _spatial_table(), adata=reference, verbose=False
    )

    assert not result.empty
    assert {"group", "names"}.issubset(result.columns)
    assert result["marker_source"].eq("adata.uns['rank_genes_groups']").all()


def test_read_markers_dataframe_generates_scanpy_markers_when_key_missing():
    reference = _reference_without_markers()
    assert "rank_genes_groups" not in reference.uns

    result = read_markers_dataframe(
        _spatial_table(),
        adata=reference,
        groupby="cell_type",
        marker_method="scanpy",
        verbose=False,
    )

    assert {"group", "names"}.issubset(result.columns)
    assert set(result["group"]) == {"A", "B"}


def test_read_markers_dataframe_scanpy_copy_does_not_mutate_input():
    reference = _reference_without_markers()

    read_markers_dataframe(
        _spatial_table(),
        adata=reference,
        groupby="cell_type",
        marker_method="scanpy",
        copy_adata=True,
        verbose=False,
    )

    assert "rank_genes_groups" not in reference.uns


def test_read_markers_dataframe_scanpy_copy_false_mutates_input():
    reference = _reference_without_markers()

    read_markers_dataframe(
        _spatial_table(),
        adata=reference,
        groupby="cell_type",
        marker_method="scanpy",
        copy_adata=False,
        verbose=False,
    )

    assert "rank_genes_groups" in reference.uns


def test_read_markers_dataframe_existing_method_requires_existing_key():
    reference = _reference_without_markers()

    with np.testing.assert_raises_regex(
        ValueError, "marker_method='scanpy'.*groupby"
    ):
        read_markers_dataframe(
            _spatial_table(),
            adata=reference,
            marker_method="existing",
            verbose=False,
        )


def test_read_markers_dataframe_scanpy_requires_groupby():
    reference = _reference_without_markers()

    with np.testing.assert_raises_regex(ValueError, "groupby is required"):
        read_markers_dataframe(
            _spatial_table(),
            adata=reference,
            marker_method="scanpy",
            groupby=None,
            verbose=False,
        )


def test_read_markers_dataframe_scanpy_requires_valid_groupby_column():
    reference = _reference_without_markers()

    with np.testing.assert_raises_regex(
        ValueError, "groupby='missing'.*adata.obs.columns"
    ):
        read_markers_dataframe(
            _spatial_table(),
            adata=reference,
            marker_method="scanpy",
            groupby="missing",
            verbose=False,
        )


def test_read_markers_dataframe_diagnostics_include_scanpy_generation():
    reference = _reference_without_markers()

    _, diagnostics = read_markers_dataframe(
        _spatial_table(),
        adata=reference,
        groupby="cell_type",
        marker_method="scanpy",
        return_diagnostics=True,
        verbose=False,
    )

    assert diagnostics["generated_rank_genes_groups"] is True
    assert diagnostics["marker_method"] == "scanpy"
    assert diagnostics["groupby"] == "cell_type"
    assert diagnostics["rank_genes_groups_key"] == "rank_genes_groups"
    assert diagnostics["scanpy_method"] == "wilcoxon"


def test_get_adata_count_matrix_requires_counts_layer():
    reference = _reference_without_markers()

    with pytest.raises(ValueError, match="raw non-negative integer counts"):
        _get_adata_count_matrix(reference, layer="counts")


def test_get_adata_count_matrix_rejects_non_integer_counts():
    reference = _pseudobulk_reference()
    reference.layers["counts"] = reference.layers["counts"].astype(float) + 0.25

    with pytest.raises(ValueError, match="raw non-negative integer counts"):
        _get_adata_count_matrix(reference, layer="counts")


def test_pseudobulk_requires_groupby_and_sample_col():
    reference = _pseudobulk_reference()

    with pytest.raises(ValueError, match="groupby is required"):
        compute_pseudobulk_deseq_markers(
            reference, groupby=None, sample_col="sample_id"
        )
    with pytest.raises(ValueError, match="sample_col is required"):
        compute_pseudobulk_deseq_markers(
            reference, groupby="cell_type", sample_col=None
        )


def test_build_one_vs_rest_pseudobulk_shapes():
    reference = _pseudobulk_reference()
    counts_df = _get_adata_count_matrix(reference, layer="counts")

    counts_pb, metadata_pb, stats = _build_one_vs_rest_pseudobulk(
        reference,
        target_group="A",
        groupby="cell_type",
        sample_col="sample_id",
        counts_df=counts_df,
        min_cells_per_group=5,
        min_replicates_per_condition=2,
    )

    assert counts_pb.shape == (6, 4)
    assert counts_pb.index.equals(metadata_pb.index)
    assert set(metadata_pb["condition"]) == {"target", "rest"}
    assert any(name.endswith("__A__target") for name in counts_pb.index)
    assert any(name.endswith("__A__rest") for name in counts_pb.index)
    assert stats["n_target_replicates"] == 3
    assert stats["n_rest_replicates"] == 3
    assert stats["skipped"] is False


def test_compute_pseudobulk_deseq_markers_runs_if_pydeseq2_available():
    pytest.importorskip("pydeseq2")
    markers, diagnostics = compute_pseudobulk_deseq_markers(
        _pseudobulk_reference(),
        groupby="cell_type",
        sample_col="sample_id",
        layer="counts",
        min_cells_per_group=5,
        min_replicates_per_condition=2,
        n_cpus=1,
    )

    assert {
        "group",
        "names",
        "logfoldchanges",
        "pvals_adj",
        "scores",
    }.issubset(markers.columns)
    assert diagnostics["groups_completed"]


@pytest.fixture(scope="module")
def _pydeseq_read_result():
    pytest.importorskip("pydeseq2")
    return read_markers_dataframe(
        _spatial_table(),
        adata=_pseudobulk_reference(),
        marker_method="pydeseq2",
        groupby="cell_type",
        sample_col="sample_id",
        layer="counts",
        min_cells_per_group=5,
        min_replicates_per_condition=2,
        deseq_n_cpus=1,
        return_diagnostics=True,
        verbose=False,
    )


def test_read_markers_dataframe_pydeseq2_returns_canonical_markers(
    _pydeseq_read_result,
):
    markers, _ = _pydeseq_read_result

    assert {
        "group",
        "names",
        "logfoldchanges",
        "pvals_adj",
        "scores",
        "marker_rank",
        "marker_source",
    }.issubset(markers.columns)
    assert markers["marker_source"].eq("pydeseq2_pseudobulk").all()
    assert markers.index.name == "group"


def test_read_markers_dataframe_pydeseq2_diagnostics(_pydeseq_read_result):
    _, diagnostics = _pydeseq_read_result

    assert diagnostics["generated_pseudobulk_deseq"] is True
    assert diagnostics["pseudobulk_deseq"] is not None
    assert diagnostics["pseudobulk_deseq"]["groups_completed"]

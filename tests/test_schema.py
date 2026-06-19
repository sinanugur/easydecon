from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

from easydecon._schema import (
    get_table,
    resolve_marker_columns,
    standardize_marker_dataframe,
)


def test_resolve_marker_columns_detects_aliases_case_insensitively():
    df = pd.DataFrame(columns=["CELL_TYPE", "Gene", "LOG2FOLDCHANGE", "Fdr"])

    assert resolve_marker_columns(df) == {
        "group": "CELL_TYPE",
        "names": "Gene",
        "logfoldchanges": "LOG2FOLDCHANGE",
        "pvals_adj": "Fdr",
    }


def test_standardize_marker_dataframe_renames_aliases():
    df = pd.DataFrame(
        {
            "cell_type": ["A"],
            "gene": ["G1"],
            "log2FoldChange": [1.0],
            "padj": [0.01],
        }
    )

    result = standardize_marker_dataframe(df)

    assert {"group", "names", "logfoldchanges", "pvals_adj"}.issubset(result.columns)
    assert result.iloc[0]["group"] == "A"
    assert result.iloc[0]["names"] == "G1"


def test_standardize_marker_dataframe_filters_lfc_and_padj():
    df = pd.DataFrame(
        {
            "group": ["A", "A", "A"],
            "names": ["keep", "low_lfc", "high_padj"],
            "logfoldchanges": [1.0, 0.1, 1.0],
            "pvals_adj": [0.01, 0.01, 0.2],
        }
    )

    result = standardize_marker_dataframe(df, log2fc_min=0.25, pval_cutoff=0.05)

    assert result["names"].tolist() == ["keep"]


def test_standardize_marker_dataframe_drops_ribosomal_and_mitochondrial():
    df = pd.DataFrame(
        {
            "group": ["A"] * 4,
            "names": ["RPS3", "rpl4", "mt-Co1", "GAPDH"],
        }
    )

    result = standardize_marker_dataframe(
        df,
        drop_ribosomal=True,
        drop_mitochondrial=True,
    )

    assert result["names"].tolist() == ["GAPDH"]


def test_standardize_marker_dataframe_filters_gene_universe():
    df = pd.DataFrame({"group": ["A", "A"], "names": ["G1", "G2"]})

    result = standardize_marker_dataframe(df, gene_universe=["G2", "G3"])

    assert result["names"].tolist() == ["G2"]


def test_standardize_marker_dataframe_excludes_celltypes():
    df = pd.DataFrame({"group": ["A", "B"], "names": ["G1", "G2"]})

    result = standardize_marker_dataframe(df, exclude_celltype=["B"])

    assert result["group"].tolist() == ["A"]


def test_standardize_marker_dataframe_keeps_top_n_and_adds_rank():
    df = pd.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B"],
            "names": ["A1", "A2", "A3", "B1", "B2"],
            "scores": [2, 5, 3, 1, 4],
        }
    )

    result = standardize_marker_dataframe(df, top_n_genes=2)

    assert result.groupby(result["group"]).size().to_dict() == {"A": 2, "B": 2}
    assert result.loc["A", "names"].tolist() == ["A2", "A3"]
    assert result.groupby(result["group"])["marker_rank"].apply(list).to_dict() == {
        "A": [1, 2],
        "B": [1, 2],
    }


def test_get_table_returns_anndata_directly():
    table = ad.AnnData(
        X=np.ones((2, 2)),
        obs=pd.DataFrame(index=["spot1", "spot2"]),
        var=pd.DataFrame(index=["gene1", "gene2"]),
    )

    assert get_table(table) is table


def test_get_table_returns_cell_segmentations_table():
    table = ad.AnnData(
        X=np.ones((1, 1)),
        obs=pd.DataFrame(index=["spot1"]),
        var=pd.DataFrame(index=["gene1"]),
    )
    sdata = SimpleNamespace(tables={"cell_segmentations": table})

    assert get_table(sdata) is table

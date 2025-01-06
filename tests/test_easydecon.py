import pytest
import spatialdata as sd
from easydecon.easydecon import *
import pandas as pd
import numpy as np

@pytest.fixture
def sdata(scope="session"):
    # Create a mock spatialdata object for testing
    # Replace this with your own implementation or use a library like unittest.mock
    sdata_small=sd.read_zarr("tests/data/sdata_test.zarr")
    return sdata_small

@pytest.fixture
def macrophage_markers():
    return ["CD14", "CD68", "CD163", "HLA-DRB1", "HLA-DQA1", "HLA-DPA1", "S100A8", "S100A9", "S100A12", "MCEMP1", "AQP9", "IL1B", "CXCL8", "CCL3", "CCL4", "HILPDA", "HIF1A-AS3", "CXCL10", "GBP1", "CXCL9", "IDO1", "SPP1", "APOC1", "ACP5", "APOE", "CCL19", "CD207", "CD1A", "CD1E", "DNASE1L3", "C1QA", "C1QC", "MRC1", "SELENOP", "FOLR2", "F13A1", "LYVE1", "FCER1A", "CCL17", "CCL22", "CD1C", "TOP2A", "MKI67"]

@pytest.fixture
def common_group_name():
    return "Macrophage"

@pytest.fixture
def bin_size():
    return 8

@pytest.fixture
def quantile():
    return 0.20

def test_common_markers_gene_expression_and_filter(sdata,macrophage_markers,common_group_name,bin_size,quantile):
    gene_expression = common_markers_gene_expression_and_filter(sdata, macrophage_markers, common_group_name,bin_size, quantile)
    df=pd.read_csv("tests/data/test_macrophage_gene_expression.csv",index_col=0)
    # Check if the returned DataFrame has the correct column
    assert common_group_name in gene_expression.columns
    # Check if filtering by quantile works
    assert gene_expression[common_group_name].min() >= gene_expression[common_group_name].quantile(quantile) or gene_expression[common_group_name].min() == 0
    assert np.isclose(df,gene_expression,atol=1e-5).all()
    assert (df.index == gene_expression.index).all()


def test_read_markers_dataframe(sdata):
    markers_df=read_markers_dataframe(sdata,"tests/data/test_macro_deg.xlsx",exclude_celltype=["11","12"],top_n_genes=60)
    df=pd.read_csv("tests/data/test_macrophage_markers.csv",index_col=0)
    assert np.isclose(df["logfoldchanges"],markers_df["logfoldchanges"],atol=1e-5).all()

def test_identify_clusters_by_expression(sdata,macrophage_markers,common_group_name,bin_size,quantile):
    gene_expression = common_markers_gene_expression_and_filter(sdata, macrophage_markers, common_group_name, bin_size, quantile)
    markers_df=read_markers_dataframe(sdata,"tests/data/test_macro_deg.xlsx",exclude_celltype=["11","12"],top_n_genes=60)
    df_identify=identify_clusters_by_expression(sdata=sdata,markers_df=markers_df,common_group_name="Macrophage",results_column="Macrophage_clusters")
    df_clusters=pd.read_csv("tests/data/test_macrophage_clusters.csv",index_col=0)
    assert common_group_name in gene_expression.columns
    assert (df_identify.index.sort_values() == df_clusters.index.sort_values()).all()
    assert (df_identify["Macrophage_clusters"].value_counts().values == df_clusters["Macrophage_clusters"].value_counts().values).all()

def test_get_clusters_expression_on_tissue(sdata,common_group_name,quantile):
    #common_markers_gene_expression_and_filter(sdata, macrophage_markers, common_group_name, bin_size, quantile)
    markers_df=read_markers_dataframe(sdata,"tests/data/test_macro_deg.xlsx",exclude_celltype=["11","12"],top_n_genes=60)
    df=pd.read_csv("tests/data/test_macrophage_clusters_expression.csv",index_col=0)
    cluster_expression=get_clusters_expression_on_tissue(sdata,markers_df=markers_df,common_group_name=common_group_name)
    assert (df.index.sort_values() == cluster_expression.index.sort_values()).all()
    assert np.isclose(df.sort_index()["0"],cluster_expression.sort_index()["0"],atol=1e-5).all()

def test_get_clusters_by_similarity_on_tissue_jaccard(sdata,method="jaccard"):
    markers_df=read_markers_dataframe(sdata,"tests/data/test_macro_deg.xlsx",exclude_celltype=["11","12"],top_n_genes=60)
    df=pd.read_csv("tests/data/test_jaccard_similarity.csv",index_col=0)
    cluster_expression=get_clusters_by_similarity_on_tissue(sdata,markers_df=markers_df,method=method,threshold=2)
    assert (df.index.sort_values() == cluster_expression.index.sort_values()).all()
    assert np.isclose(df.sort_index()["0"],cluster_expression.sort_index()["0"],atol=1e-5).all()



def test_get_clusters_by_similarity_on_tissue_wjaccard(sdata,method="wjaccard"):
    markers_df=read_markers_dataframe(sdata,"tests/data/test_macro_deg.xlsx",exclude_celltype=["11","12"],top_n_genes=60)
    df=pd.read_csv("tests/data/test_cosine_similarity.csv",index_col=0)
    cluster_expression=get_clusters_by_similarity_on_tissue(sdata,markers_df=markers_df,method=method)
    assert (df.index.sort_values() == cluster_expression.index.sort_values()).all()
    assert np.isclose(df.sort_index()["0"],cluster_expression.sort_index()["0"],atol=1e-5).all()
"""
def test_identify_clusters_by_similarity(sdata,macrophage_markers,common_group_name, bin_size, quantile):
    gene_expression = common_markers_gene_expression_and_filter(sdata, macrophage_markers, common_group_name, bin_size, quantile)
    markers_df=read_markers_dataframe(sdata,"tests/data/test_macro_deg.xlsx",exclude_celltype=["11","12"],top_n_genes=60)

    df_corr=identify_clusters_by_similarity(sdata=sdata,markers_df=markers_df,common_group_name=common_group_name,method="correlation",results_column="Macrophage_clusters_correlation")
    df_cos=identify_clusters_by_similarity(sdata=sdata,markers_df=markers_df,common_group_name=common_group_name,method="cosine",results_column="Macrophage_clusters_cosine")
    #df_corr_testing=pd.read_csv("tests/data/test_macrophage_clusters_correlation.csv",index_col=0)
    df_cos_testing=pd.read_csv("tests/data/test_macrophage_clusters_cosine.csv",index_col=0,dtype={'Macrophage_clusters_cosine':'str'})

    #df_cos_testing['Macrophage_clusters_cosine'] = pd.Categorical(df_cos_testing['Macrophage_clusters_cosine'], categories=df_cos["Macrophage_clusters_cosine"].cat.categories)

    #assert (df_corr["Macrophage_clusters_correlation"].value_counts().values == df_corr_testing["Macrophage_clusters_correlation"].value_counts().values).all()
    #assert (df_cos["Macrophage_clusters_cosine"].value_counts().values == df_cos_testing["Macrophage_clusters_cosine"].value_counts().values).all()
"""

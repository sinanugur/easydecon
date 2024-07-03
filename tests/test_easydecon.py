import pytest
import spatialdata as sd
from easydecon.easydecon import *
import pandas as pd
import numpy as np

@pytest.fixture
def sdata():
    # Create a mock spatialdata object for testing
    # Replace this with your own implementation or use a library like unittest.mock
    bin_size=8
    sample_id="sampleP2"
    sdata_small=sd.read_zarr("tests/data/sdata_test.zarr")
    return sdata_small


def test_common_markers_gene_expression_and_filter(sdata):
    macrophage_markers = ["CD14","CD68","CD163","HLA-DRB1","HLA-DQA1","HLA-DPA1","S100A8","S100A9","S100A12","MCEMP1","AQP9","IL1B","CXCL8","CCL3","CCL4","HILPDA","HIF1A-AS3","CXCL10","GBP1","CXCL9","IDO1","SPP1","APOC1","ACP5","APOE","CCL19","CD207","CD1A","CD1E","DNASE1L3","C1QA","C1QC","MRC1","SELENOP","FOLR2","F13A1","LYVE1","FCER1A","CCL17","CCL22","CD1C","TOP2A","MKI67"]
    common_group_name= "Macrophage"
    bin_size=8
    quantile=0.20
    gene_expression = common_markers_gene_expression_and_filter(sdata, macrophage_markers, common_group_name, bin_size, quantile)
    df=pd.read_csv("tests/data/test_macrophage_gene_expression.csv",index_col=0)
    df.to_csv("blah.csv")
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

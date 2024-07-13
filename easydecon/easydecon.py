import scanpy as sc
import numpy as np
import pandas as pd
import spatialdata as sp
import spatialdata_io
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from numba import jit
from tqdm import tqdm


def common_markers_gene_expression_and_filter(sdata, marker_genes,common_group_name, bin_size=8,quantile=0.70):
    table_key = f"square_00{bin_size}um"
    table = sdata.tables[table_key]
    filtered_genes = list(set(marker_genes).intersection(table.var_names))
    gene_expression = table[:, filtered_genes].to_df().sum(axis=1).to_frame(common_group_name)
    if common_group_name in table.obs.columns:
        table.obs.drop(columns=[common_group_name], inplace=True)
    
    threshold=gene_expression[gene_expression[common_group_name] !=0].quantile(quantile)
    gene_expression[common_group_name] = np.where(gene_expression[common_group_name].values >= threshold.values, gene_expression[common_group_name], 0)

    table.obs=pd.merge(table.obs, gene_expression, left_index=True, right_index=True)

    gene_expression.index.name = "Index"
    return gene_expression

def read_markers_dataframe(sdata,filename=None,adata=None,exclude_celltype=[],bin_size=8,top_n_genes=60,sort_by_column="logfoldchanges",gene_id_column="names",celltype="group",key="rank_genes_groups"): #100
    table = sdata.tables[f"square_00{bin_size}um"]

    if adata is None:
        if filename is None:
            raise ValueError("Please provide a filename or an adata object")
        else:
            try:
                df=pd.read_csv(filename,dtype={gene_id_column:str,celltype:str})
            except:
                df=pd.read_excel(filename,dtype={gene_id_column:str,celltype:str})
    else:
        try:
            df=sc.get.rank_genes_groups_df(adata,group=None, key=key, pval_cutoff=0.05, log2fc_min=0.25)
        except:
            raise ValueError("Please provide a valid adata object with rank_genes_groups key")
            
        
    df=df[df[gene_id_column].isin(table.var_names)]
    df=df[~df[celltype].isin(exclude_celltype)]
    df = df.sort_values(by=sort_by_column, ascending=False)
    df = df.groupby(celltype).head(top_n_genes)
    df.set_index(celltype,inplace=True)
    return df

def identify_clusters_by_expression(sdata,markers_df,common_group_name=None,bin_size=8,gene_id_column="names",results_column="easydecon",method="mean"):
    table = sdata.tables[f"square_00{bin_size}um"]
    associated_cluster=dict()
    if common_group_name in table.obs.columns:
        spots_with_expression = table.obs[table.obs[common_group_name] != 0].index
    else:
        print("common_group_name column not found in the table, processing all spots.")
        spots_with_expression = table.obs.index

    if method=="mean":
        compute = lambda x: np.mean(x, axis=1).values
    elif method=="median":
        compute = lambda x: np.median(x, axis=1).values
    elif method=="sum":
        compute = lambda x: np.sum(x, axis=1).values

    for spot in spots_with_expression:
        a=dict()

        for cluster in markers_df.index.unique():
            #genes=cluster_membership_df.loc[cluster]["names"].values
            genes=markers_df.loc[cluster][gene_id_column]
            if isinstance(genes, str):
                genes = [genes]
            else:
                genes = genes.values
            #group_expression = sdata.tables[f"square_00{bin_size}um"][spot, genes].to_df().sum(axis=1).values
            #group_expression = sdata.tables[f"square_00{bin_size}um"][spot, genes].to_df().apply(gmean,axis=1).values
            #group_expression = table[spot, genes].to_df().mean(axis=1).values
            group_expression = compute(table[spot, genes].to_df())
            a[cluster]=group_expression
        #max_cluster=str(max(a, key=a.get))

        associated_cluster[spot]=str(max(a, key=a.get))
    
    for spot in set(table.obs.index) - set(spots_with_expression):
        associated_cluster[spot] = None
        
    df=pd.DataFrame(list(associated_cluster.items()), columns=['Index', 'assigned_cluster'])
    df.set_index('Index', inplace=True)
    df[f'{results_column}'] = pd.Categorical(df['assigned_cluster'],categories=markers_df.index.unique())
    df.drop(columns=['assigned_cluster'],inplace=True)

    table.obs.drop(columns=[f'{results_column}'],inplace=True,errors='ignore')
    table.obs=pd.merge(table.obs, df, left_index=True, right_index=True)
    return df


def get_clusters_expression_on_tissue(sdata,markers_df,common_group_name=None,bin_size=8,gene_id_column="names",method="mean"):
    table = sdata.tables[f"square_00{bin_size}um"]
    if common_group_name in table.obs.columns:
        spots_with_expression = table.obs[table.obs[common_group_name] != 0].index
    else:
        print("common_group_name column not found in the table, processing all spots.")
        spots_with_expression = table.obs.index

    if method=="mean":
        compute = lambda x: np.mean(x, axis=1).values
    elif method=="median":
        compute = lambda x: np.median(x, axis=1).values
    elif method=="sum":
        compute = lambda x: np.sum(x, axis=1).values

    # Preallocate DataFrame with zeros
    all_spots = table.obs.index
    all_clusters = markers_df.index.unique()
    df = pd.DataFrame(0, index=all_spots, columns=all_clusters)
    
    # Process only spots with expression
    for spot in spots_with_expression:
        a = {}
        for cluster in all_clusters:
            genes = markers_df.loc[cluster][gene_id_column]
            genes = [genes] if isinstance(genes, str) else genes.values
            group_expression = compute(table[spot, genes].to_df())
            a[cluster] = group_expression
        
        # Directly assign to preallocated DataFrame
        df.loc[spot] = pd.DataFrame.from_dict(a, orient='index').transpose().values
    
    table.obs.drop(columns=all_clusters,inplace=True,errors='ignore')
    table.obs=pd.merge(table.obs, df, left_index=True, right_index=True)
    return df

    

def identify_clusters_by_similarity(sdata,markers_df,common_group_name=None,bin_size=8,gene_id_column="names",similarity_by_column="logfoldchanges",results_column="easydecon_similarity",method="correlation"):
    table = sdata.tables[f"square_00{bin_size}um"]
    tqdm.pandas()

    if common_group_name in table.obs.columns:
        spots_with_expression = table.obs[table.obs[common_group_name] != 0].index
    else:
        print("common_group_name column not found in the table, processing all spots.")
        spots_with_expression = table.obs.index

    if method=="correlation":
        print("Method: Correlation")
        result_df = table[spots_with_expression,].to_df().progress_apply(lambda row: pd.Series({'Index': row.name, 'assigned_cluster': function_row_corr(row, markers_df,gene_id_column=gene_id_column,similarity_by_column=similarity_by_column)}), axis=1)
    elif method=="cosine":
        print("Method: Cosine")
        result_df = table[spots_with_expression,].to_df().progress_apply(lambda row: pd.Series({'Index': row.name, 'assigned_cluster': function_row_cosine(row, markers_df,gene_id_column=gene_id_column,similarity_by_column=similarity_by_column)}), axis=1)
    else:    
        raise ValueError("Please provide a valid method: correlation or cosine")
    others_df= pd.DataFrame({'Index': list(set(table.obs.index) - set(spots_with_expression)), 'assigned_cluster': [None]*len(set(table.obs.index) - set(spots_with_expression))})
    df=pd.concat([result_df,others_df])
    df.set_index('Index', inplace=True)
    df[f'{results_column}'] = pd.Categorical(df['assigned_cluster'],categories=markers_df.index.unique())
    df.drop(columns=['assigned_cluster'],inplace=True)
    table.obs.drop(columns=[f'{results_column}'],inplace=True,errors='ignore')
    table.obs=pd.merge(table.obs, df, left_index=True, right_index=True)
    return df


#@jit(nopython=True)
"""
def function_row_corr(row,markers_df,gene_id_column="names",similarity_by_column="logfoldchanges"):
    a={}
    u=markers_df.index.unique()
    for c in u:
        vector_series=pd.Series(markers_df[[gene_id_column,similarity_by_column]].loc[c][similarity_by_column].values, index=markers_df[[gene_id_column,similarity_by_column]].loc[c][gene_id_column].values)
        vector_series=vector_series.reindex(row.index,fill_value=np.nan)

        a[c]=spearmanr(row, vector_series,nan_policy="omit",axis=1)[0]

    
    return str(max(a, key=a.get))
"""
    
def function_row_corr(row, markers_df, gene_id_column="names", similarity_by_column="logfoldchanges"):
    a = {}
    markers_df_grouped = markers_df.groupby(markers_df.index)
    
    for c, group in markers_df_grouped:
        vector_series = group.set_index(gene_id_column)[similarity_by_column].reindex(row.index, fill_value=np.nan)
        a[c] = spearmanr(row, vector_series, nan_policy="omit")[0]
    
    return str(max(a, key=a.get))

#@jit(nopython=True)
def function_row_cosine(row, markers_df,gene_id_column="names",similarity_by_column="logfoldchanges"):
    a = {}
    for c in markers_df.index.unique():
        vector_series = pd.Series(markers_df[[gene_id_column,similarity_by_column]].loc[c][similarity_by_column].values, index=markers_df[[gene_id_column, similarity_by_column]].loc[c][gene_id_column].values)
        vector_series = vector_series.reindex(row.index, fill_value=np.nan)

        # Calculate cosine distance and handle cases where vectors might be all NaNs after reindexing
        if not vector_series.isnull().all() and not row.isnull().all():
            a[c] = cosine(row.fillna(0), vector_series.fillna(0))
        else:
            a[c] = np.nan  # Assign NaN if either vector is all NaNs

    # Return the key with the minimum cosine distance (maximum similarity)
    # Filter out NaN values from the dictionary
    filtered_a = {k: v for k, v in a.items() if not np.isnan(v)}
    return str(min(filtered_a, key=filtered_a.get)) if filtered_a else None



def test_function():
    print("Hello World Now")



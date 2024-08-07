import scanpy as sc
import numpy as np
import pandas as pd
import spatialdata as sp
import spatialdata_io
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from numba import jit
from tqdm import tqdm

from joblib import Parallel, delayed


# Ensure that the progress_apply method is available
tqdm.pandas()



#when NAN values are present in the data, spatialdata may not produce output, so we need to replace NAN values with 0

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



#this function is used to read the markers from a file or from an single-cell anndata object and return a dataframe
def read_markers_dataframe(sdata,filename=None,adata=None,exclude_celltype=[],bin_size=8,top_n_genes=60,sort_by_column="logfoldchanges",ascending=False,gene_id_column="names",celltype="group",key="rank_genes_groups"): #100
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
            
        
    df = df[df[gene_id_column].isin(table.var_names)]
    df = df[~df[celltype].isin(exclude_celltype)]
    df = df.sort_values(by=sort_by_column, ascending=ascending)
    df = df.groupby(celltype).head(top_n_genes)
    print("Unique cell types detected in the dataframe:")
    print(df[celltype].unique())
    df.set_index(celltype,inplace=True)
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
    for spot in tqdm(spots_with_expression, desc='Processing spots'):
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


def assign_clusters_from_df(sdata,df,bin_size=8,results_column="easydecon"):
    table = sdata.tables[f"square_00{bin_size}um"]
    table.obs.drop(columns=[results_column],inplace=True,errors='ignore')
    table.obs=pd.merge(table.obs, df.idxmax(axis=1).to_frame(results_column).astype('category'), left_index=True, right_index=True)
    return


def process_row(row,func, markers_df, gene_id_column, similarity_by_column,threshold):
    return pd.Series({
        'Index': row.name,
        'assigned_cluster': func(row, markers_df, gene_id_column=gene_id_column, similarity_by_column=similarity_by_column,threshold=threshold)
    })

def get_clusters_by_similarity_on_tissue(sdata,markers_df,common_group_name=None,bin_size=8,gene_id_column="names",similarity_by_column="logfoldchanges",method="correlation",threshold=1):
    table = sdata.tables[f"square_00{bin_size}um"]
    tqdm.pandas()

    if common_group_name in table.obs.columns:
        spots_with_expression = table.obs[table.obs[common_group_name] != 0].index
    else:
        print("common_group_name column not found in the table, processing all spots.")
        spots_with_expression = table.obs.index

    if method=="correlation":
        print("Method: Correlation")
        #result_df = table[spots_with_expression,].to_df().progress_apply(lambda row: pd.Series({'Index': row.name, 'assigned_cluster': function_row_spearman(row, markers_df,gene_id_column=gene_id_column,similarity_by_column=similarity_by_column)}), axis=1)
        func=function_row_spearman
    elif method=="cosine":
        print("Method: Cosine")
        #result_df = table[spots_with_expression,].to_df().progress_apply(lambda row: pd.Series({'Index': row.name, 'assigned_cluster': function_row_cosine(row, markers_df,gene_id_column=gene_id_column,similarity_by_column=similarity_by_column)}), axis=1)
        func=function_row_cosine
    elif method=="jaccard":
        print("Method: Jaccard")
        #result_df = table[spots_with_expression,].to_df().progress_apply(lambda row: pd.Series({'Index': row.name, 'assigned_cluster': function_row_jaccard(row, markers_df,gene_id_column=gene_id_column,threshold=threshold)}), axis=1)
        func=function_row_jaccard
    elif method=="wjaccard":
        print("Method: Weighted Jaccard")
        #result_df = table[spots_with_expression,].to_df().progress_apply(lambda row: pd.Series({'Index': row.name, 'assigned_cluster': function_row_weighted_jaccard(row, markers_df,gene_id_column=gene_id_column,threshold=threshold)}), axis=1)
        func=function_row_weighted_jaccard
    else:    
        raise ValueError("Please provide a valid method: correlation or cosine")
    results = Parallel(n_jobs=6,batch_size=1000)(
    delayed(process_row)(row,func, markers_df, gene_id_column, similarity_by_column,threshold)
    for index, row in tqdm(table[spots_with_expression,].to_df().iterrows(), total=len(table[spots_with_expression,].to_df())))
    result_df = pd.DataFrame(results)
    result_df.set_index("Index",inplace=True)
    result_df=result_df["assigned_cluster"].apply(pd.Series)

    #others_df= pd.DataFrame({'Index': list(set(table.obs.index) - set(spots_with_expression)), 'assigned_cluster': [None]*len(set(table.obs.index) - set(spots_with_expression))})
    others_df = pd.DataFrame(0, index=list(set(table.obs.index) - set(spots_with_expression)), columns=result_df.columns)
    df=pd.concat([result_df,others_df])
    #df.set_index('Index', inplace=True)
    #df[f'{results_column}'] = pd.Categorical(df['assigned_cluster'],categories=markers_df.index.unique())
    #df.drop(columns=['assigned_cluster'],inplace=True)
    #table.obs.drop(columns=[f'{results_column}'],inplace=True,errors='ignore')
    table.obs.drop(columns=df.columns,inplace=True,errors='ignore')
    table.obs=pd.merge(table.obs, df, left_index=True, right_index=True)
    return df


def function_row_corr(row, markers_df, gene_id_column="names", similarity_by_column="logfoldchanges",threshold=1):
    a = {}
    markers_df_grouped = markers_df.groupby(markers_df.index)
    row=min_max_scale(row)

    for c, group in markers_df_grouped:
        vector_series = group.set_index(gene_id_column)[similarity_by_column].reindex(row.index, fill_value=np.nan)
        vector_series = min_max_scale(vector_series)
        sp = spearmanr(row, vector_series, nan_policy="omit")[0]
        a[c] = sp if sp > 0 else 0.0  # Assign 0 if correlation is negative
    
    #return str(max(a, key=a.get))
    return a

def function_row_spearman(row, markers_df,gene_id_column="names",similarity_by_column="logfoldchanges",threshold=1):
    a = {}
    for c in markers_df.index.unique():
        vector_series = pd.Series(markers_df[[gene_id_column,similarity_by_column]].loc[c][similarity_by_column].values, index=markers_df[[gene_id_column, similarity_by_column]].loc[c][gene_id_column].values)
        l = len(vector_series)
        vector_series = vector_series.reindex(row.index, fill_value=np.nan)
        valid_mask = ~vector_series.isna() & ~row.isna()
        t = (row[valid_mask] != 0).sum()
        if t == 0:  # No valid pairs
            a[c] = 0.0
        else:
            sp = (spearmanr(row[valid_mask], vector_series[valid_mask], nan_policy="omit")[0])*(t/l)
            a[c] = sp if sp > 0 else 0.0  # Assign 0 if correlation is negative
    return a



#@jit(nopython=True)
def function_row_cosine(row, markers_df,gene_id_column="names",similarity_by_column="logfoldchanges",threshold=1):
    a = {}
    row=min_max_scale(row)
    for c in markers_df.index.unique():
        vector_series = pd.Series(markers_df[[gene_id_column,similarity_by_column]].loc[c][similarity_by_column].values, index=markers_df[[gene_id_column, similarity_by_column]].loc[c][gene_id_column].values)
        l = len(vector_series)
        vector_series = vector_series.reindex(row.index, fill_value=np.nan)
        vector_series = min_max_scale(vector_series)
        valid_mask = ~vector_series.isna() & ~row.isna()
        t = (row[valid_mask] != 0).sum()
        if t == 0:  # No valid pairs
            a[c] = 0.0
        else:
            a[c] = (1 - cosine(row[valid_mask], vector_series[valid_mask]))*(t/l) #penalize the cosine similarity by the fraction of valid pairs
    return a


def min_max_scale(series):
    min_val = series.min()
    max_val = series.max()
    if min_val == max_val:
        return series.apply(lambda x: 0.0)  # what if all values are the same? for now, return 0
    return (series - min_val) / (max_val - min_val)

def function_row_jaccard(row, markers_df, gene_id_column="names",similarity_by_column="logfoldchanges", threshold=1):
    a = {}
    
    for c in markers_df.index.unique():
        row_set = set(row[row > threshold].sort_values(ascending=False).index)
        vector_set = set(markers_df.loc[c][gene_id_column].values)
        
        # Calculate intersection and union
        intersection = len(row_set.intersection(vector_set))
        union = len(row_set.union(vector_set))
        
        if union == 0:
            jaccard_sim = 0.0  # If both sets are empty, define similarity as 0
        else:
            jaccard_sim = intersection / union
        
        a[c] = jaccard_sim
    
    return a

def function_row_weighted_jaccard(row, markers_df, gene_id_column="names",similarity_by_column="logfoldchanges" ,threshold=1):
    a = {}
    
    row=row[row > threshold]
    for c in markers_df.index.unique():
        row_set = set(row[row > threshold].sort_values(ascending=False).index)
        vector_set = set(markers_df.loc[c][gene_id_column].values)
        
        # Calculate intersection and union
        #intersection = len(row_set.intersection(vector_set))
        #union = len(row_set.union(vector_set))
        intersection=0
        union=0
        penalty=row.values.mean()
        for gene in row_set.union(vector_set):
            weight = row[gene] if gene in row.index else penalty
            if gene in row_set and gene in vector_set:
                intersection += weight
            union += weight


        if union == 0:
            jaccard_sim = 0.0  # If both sets are empty, define similarity as 0
        else:
            jaccard_sim = intersection / union
        
        a[c] = jaccard_sim
    
    #return str(max(a, key=a.get))
    return a


def add_df_to_spatialdata(sdata,df,bin_size=8):
    table = sdata.tables[f"square_00{bin_size}um"]
    table.obs.drop(columns=df.columns,inplace=True,errors='ignore')
    table.obs=pd.merge(table.obs, df, left_index=True, right_index=True)
    print("DataFrame added to SpatialData object")
    print(table.obs.head())
    return

def test_function():
    print("Easydecon loaded!")


"""
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
        #result_df = table[spots_with_expression,].to_df().progress_apply(lambda row: pd.Series({'Index': row.name, 'assigned_cluster': function_row_corr(row, markers_df,gene_id_column=gene_id_column,similarity_by_column=similarity_by_column)}), axis=1)
        result_df = table[spots_with_expression,].to_df().progress_apply(lambda row: pd.Series({'Index': row.name, 'assigned_cluster': function_row_corr(row, markers_df,gene_id_column=gene_id_column,similarity_by_column=similarity_by_column)}), axis=1)
    elif method=="cosine":
        print("Method: Cosine")
        #result_df = table[spots_with_expression,].to_df().progress_apply(lambda row: pd.Series({'Index': row.name, 'assigned_cluster': function_row_cosine(row, markers_df,gene_id_column=gene_id_column,similarity_by_column=similarity_by_column)}), axis=1)
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

"""
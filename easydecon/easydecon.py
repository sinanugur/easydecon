import scanpy as sc
import numpy as np
import pandas as pd
import spatialdata as sp
import spatialdata_io
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from tqdm import tqdm
from .config import config

from joblib import Parallel, delayed
from scipy.stats import zscore

from spatialdata import polygon_query
import warnings


# Ensure that the progress_apply method is available
tqdm.pandas()



#when NAN values are present in the data, spatialdata may not produce output, so we need to replace NAN values with 0

def common_markers_gene_expression_and_filter(sdata, marker_genes,common_group_name, bin_size=8,quantile=0.70):
    try:
        table_key = f"square_00{bin_size}um"
        table = sdata.tables[table_key]
    except:
        table=sdata
    filtered_genes = list(set(marker_genes).intersection(table.var_names))
    gene_expression = table[:, filtered_genes].to_df().sum(axis=1).to_frame(common_group_name)
    if common_group_name in table.obs.columns:
        table.obs.drop(columns=[common_group_name], inplace=True)
    
    threshold=gene_expression[gene_expression[common_group_name] !=0].quantile(quantile)
    gene_expression[common_group_name] = np.where(gene_expression[common_group_name].values > threshold.values, gene_expression[common_group_name], 0)

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
            
        
    df = df[df[gene_id_column].isin(table.var_names)] #check if the var_names are present in the spatial data
    df = df[~df[celltype].isin(exclude_celltype)]
    df = df.sort_values(by=sort_by_column, ascending=ascending)
    df = df.groupby(celltype).head(top_n_genes)
    print("Unique cell types detected in the dataframe:")
    print(df[celltype].unique())
    df.set_index(celltype,inplace=True)
    return df

def get_clusters_expression_on_tissue(sdata,markers_df,common_group_name=None,bin_size=8,gene_id_column="names",method="mean"):

    try:
        table = sdata.tables[f"square_00{bin_size}um"]
    except:
        table=sdata

    markers_df_tmp=markers_df[markers_df[gene_id_column].isin(table.var_names)] #just to be sure the genes are present in the spatial data

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
    all_clusters = markers_df_tmp.index.unique()
    df = pd.DataFrame(0, index=all_spots, columns=all_clusters)
    
    # Process only spots with expression
    for spot in tqdm(spots_with_expression, desc='Processing spots'):
        a = {}
        for cluster in all_clusters:
            genes = markers_df_tmp.loc[cluster][gene_id_column]
            genes = [genes] if isinstance(genes, str) else genes.values
            group_expression = compute(table[spot, genes].to_df())
            a[cluster] = group_expression
        
        # Directly assign to preallocated DataFrame
        df.loc[spot] = pd.DataFrame.from_dict(a, orient='index').transpose().values
    
    table.obs.drop(columns=all_clusters,inplace=True,errors='ignore')
    table.obs=pd.merge(table.obs, df, left_index=True, right_index=True)
    return df




def assign_clusters_from_df(sdata,df,bin_size=8,results_column="easydecon",method="max",diagnostic=None):
    
    try:
        table = sdata.tables[f"square_00{bin_size}um"]
    except:
        table=sdata
    table.obs.drop(columns=[results_column],inplace=True,errors='ignore')
    if method=="max":
        df_reindexed=df[~(df == 0).all(axis=1)].idxmax(axis=1).to_frame(results_column).astype('category').reindex(table.obs.index, fill_value=np.nan)
    elif method=="zmax":
        tmp=df.replace(0,np.nan).apply(lambda x: zscore(x, nan_policy='omit'),axis=0).idxmax(axis=1)
        df_reindexed=tmp.to_frame(results_column).astype('category').reindex(table.obs.index, fill_value=np.nan)
    else:
        raise ValueError("Please provide a valid method: max or zmax")
    table.obs=pd.merge(table.obs, df_reindexed, left_index=True, right_index=True)
    if diagnostic is not None:
        for r in table.obs[["easydecon"]].itertuples(index=True, name='Pandas'):
            if not pd.isna(r.easydecon):
                table.obs.at[r.Index, "diagnostic"] = list(diagnostic.loc[r.Index][r.easydecon])
            else:
                pass

        
    return

def visualize_only_selected_clusters(sdata,clusters,bin_size=8,results_column="easydecon",temp_column="tmp"):
    table = sdata.tables[f"square_00{bin_size}um"]
    table.obs.drop(columns=[temp_column],inplace=True,errors='ignore')
    #table.obs=pd.merge(table.obs, df.idxmax(axis=1).to_frame(results_column).astype('category'), left_index=True, right_index=True)
    table.obs[temp_column]=table.obs[results_column].apply(lambda x: x if x in clusters else np.nan)
    return

def plot_assigned_clusters_from_dataframe(sdata,dataframe,sample_id,bin_size=8,title="Assigned Clusters",cmap="tab20",legend_fontsize=8,figsize=(5,5),dpi=200,method="matplotlib",scale=1):
    assign_clusters_from_df(sdata,df=dataframe,bin_size=8,results_column="plotted_clusters")
    
    sdata.pl.render_images("queried_cytassist").pl.render_shapes(
        f"{sample_id}_square_00{bin_size}um", color="plotted_clusters",cmap=cmap,method=method,scale=scale
    ).pl.show(coordinate_systems="global", title=title, legend_fontsize=legend_fontsize,figsize=figsize,dpi=dpi)

    return


def napari_region_assignment(sdata,key="Shapes",bin_size=8,column="napari",target_coordinate_system="global"):
    
    try:
        sdata[key]
    except:
        raise ValueError("Please provide a valid key for the shapes in the spatial data object that assigned via Napari")
        return
    
    sdata.tables[f"square_00{bin_size}um"].obs.drop(columns=column,inplace=True,errors='ignore')
    indices_list = []
    for g in sdata[key].geometry:
        indices=polygon_query(sdata,polygon=g,target_coordinate_system=target_coordinate_system).tables[f"square_00{bin_size}um"].obs.index
        indices_list.extend(indices)
    
    df=pd.DataFrame("No", index=sdata.tables[f"square_00{bin_size}um"].obs.index, columns=[column])
    df[column]=np.where(df.index.isin(set(indices_list)), 'Yes', 'No')
    sdata.tables[f"square_00{bin_size}um"].obs=pd.merge(sdata.tables[f"square_00{bin_size}um"].obs, df, left_index=True, right_index=True)
    return



def process_row(row,func, **kwargs):
    return pd.Series({
        'Index': row.name,
        #'assigned_cluster': func(row, markers_df, gene_id_column=gene_id_column, similarity_by_column=similarity_by_column,threshold=threshold)
        'assigned_cluster': func(row, **kwargs)
    })

def get_clusters_by_similarity_on_tissue(sdata,markers_df,common_group_name=None,bin_size=8,gene_id_column="names",similarity_by_column="logfoldchanges",method="correlation",threshold=0.1,lambda_param=0.5):
    try:
        table = sdata.tables[f"square_00{bin_size}um"]
    except:
        table=sdata
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
    elif method=="overlap":
        print("Method: Overlap")
        #result_df = table[spots_with_expression,].to_df().progress_apply(lambda row: pd.Series({'Index': row.name, 'assigned_cluster': function_row_jaccard(row, markers_df,gene_id_column=gene_id_column,threshold=threshold)}), axis=1)
        func=function_row_overlap
    elif method=="wjaccard":
        print("Method: Weighted Jaccard")
        #result_df = table[spots_with_expression,].to_df().progress_apply(lambda row: pd.Series({'Index': row.name, 'assigned_cluster': function_row_weighted_jaccard(row, markers_df,gene_id_column=gene_id_column,threshold=threshold)}), axis=1)
        func=function_row_weighted_jaccard
    elif method=="diagnostic":
        print("Method: Get genes similarity diagnostics")
        func=function_row_diagnostic
    else:    
        raise ValueError("Please provide a valid method: correlation, jaccard, wjaccard or cosine")
    results = Parallel(n_jobs=config.n_jobs,batch_size=config.batch_size)(
    delayed(process_row)(row,func, markers_df=markers_df, gene_id_column=gene_id_column, similarity_by_column=similarity_by_column,threshold=threshold,lambda_param=lambda_param)
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
    if method != "diagnostic":
        table.obs.drop(columns=df.columns,inplace=True,errors='ignore')
        table.obs=pd.merge(table.obs, df, left_index=True, right_index=True)
    return df






def function_row_spearman(row, markers_df,**kwargs):
    gene_id_column=kwargs.get("gene_id_column")
    similarity_by_column=kwargs.get("similarity_by_column")

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
def function_row_cosine(row, markers_df,**kwargs):
    gene_id_column=kwargs.get("gene_id_column")
    similarity_by_column=kwargs.get("similarity_by_column")
    a = {}
    #row=min_max_scale(row[row > 0])
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
    series = series.fillna(0)# fill nan values with 0
    min_val = series.min()
    max_val = series.max()
    if min_val == max_val:
        #return series.apply(lambda x: 0.0)  # what if all values are the same? for now, return 0
        #warnings.warn("All values in the series are identical; returning zeros.")
        return pd.Series(0.0, index=series.index)
    return (series - min_val) / (max_val - min_val)

def function_row_jaccard(row, markers_df, **kwargs):
    a = {}
    gene_id_column=kwargs.get("gene_id_column")
    threshold=kwargs.get("threshold")
    for c in markers_df.index.unique():
        row_set = set(row[row > threshold].sort_values(ascending=False).index)
        vector_set = set(markers_df.loc[c][gene_id_column].values)
        
        # Calculate intersection and union
        i = row_set.intersection(vector_set)
        union = len(row_set.union(vector_set))
        
        if union == 0:
            jaccard_sim = 0.0  # If both sets are empty, define similarity as 0
        else:
            jaccard_sim = len(i) / union
        
        a[c] = jaccard_sim
    
    return a

def function_row_overlap(row, markers_df, **kwargs):
    a = {}
    gene_id_column=kwargs.get("gene_id_column")
    threshold=kwargs.get("threshold")
    for c in markers_df.index.unique():
        row_set = set(row[row > threshold].sort_values(ascending=False).index)
        vector_set = set(markers_df.loc[c][gene_id_column].values)
        
        # Calculate intersection and union
        i = row_set.intersection(vector_set)
        union = min(len(row_set),len(vector_set))
        
        if union == 0:
            overlap_sim = 0.0  # If both sets are empty, define similarity as 0
        else:
            overlap_sim = len(i) / union #what if min len differs, investigate!
        
        a[c] = overlap_sim #return the intersection as well so we can use it later for common genes
    
    return a

def function_row_diagnostic(row, markers_df, **kwargs):
    a = {}
    gene_id_column=kwargs.get("gene_id_column")
    threshold=kwargs.get("threshold")
    for c in markers_df.index.unique():
        row_set = set(row[row > threshold].sort_values(ascending=False).index)
        vector_set = set(markers_df.loc[c][gene_id_column].values)
        
        # Calculate intersection and union
        a[c] = row_set.intersection(vector_set)
    return a

"""
#weighted jaccard similarity function
def function_row_weighted_jaccard(row, markers_df, **kwargs):
    gene_id_column=kwargs.get("gene_id_column")
    threshold=kwargs.get("threshold")
    a = {}
    
    row=row[row > threshold]
    for c in markers_df.index.unique():
        row_set = set(row.sort_values(ascending=False).index)
        vector_set = set(markers_df.loc[c][gene_id_column].values)
        
        intersection=0
        union=0
        penalty=row.values.mean()
        i = row_set.intersection(vector_set)
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
"""

def function_row_weighted_jaccard(row, markers_df, **kwargs):

    gene_id_column = kwargs.get("gene_id_column")
    lambda_param = kwargs.get("lambda_param", 0.5)  # Default lambda to 1.0 if not provided
    a = {}
    
    # Ensure 'row' is a pandas Series with gene names as the index
    if not isinstance(row, pd.Series):
        row = pd.Series(row)
    
    # Get the genes and their expression levels from 'row' (target set)
    # Optionally, normalize the expression levels to ensure comparability
    target_genes = row[row > 0]  # Select genes with expression > 0
    # Normalize the expression levels to range between 0 and 1
    max_expr = target_genes.max()
    if max_expr > 0:
        target_weights = target_genes / max_expr
    else:
        target_weights = target_genes  # Will be an empty Series

    for c in markers_df.index.unique():
        # Get the genes and their rankings for the current cluster 'c'
        cluster_df = markers_df.loc[c]
        if isinstance(cluster_df, pd.DataFrame):
            cluster_genes = cluster_df[gene_id_column].reset_index(drop=True)
        else:
            cluster_genes = pd.Series([cluster_df[gene_id_column]])

        # Assign weights to 'cluster_genes' based on exponential rank weighting
        N = len(cluster_genes)
        if N > 0:
            # Exponential decay weighting: weights decrease exponentially with rank
            ranks = np.arange(N)  # Rank positions starting from 0
            weights = np.exp(-lambda_param * ranks)
        else:
            weights = np.array([])
        # Create a pandas Series with weights assigned to genes
        cluster_weights = pd.Series(weights, index=cluster_genes)
        
        # Union of genes in 'cluster_weights' and 'target_weights'
        all_genes = set(cluster_weights.index).union(target_weights.index)
        
        # Initialize numerator and denominator for Weighted Jaccard Index
        numerator = 0.0
        denominator = 0.0
        
        for gene in all_genes:
            # Weight in 'cluster_weights' (pseudo weight based on exponential rank), 0 if gene not present
            a_i = cluster_weights.get(gene, 0.0)
            # Weight in 'target_weights' (normalized expression level), 0 if gene not present
            b_i = target_weights.get(gene, 0.0)
            numerator += min(a_i, b_i)
            denominator += max(a_i, b_i)
        
        # Compute the Weighted Jaccard Index
        if denominator == 0:
            jaccard_sim = 0.0
        else:
            jaccard_sim = numerator / denominator
        
        a[c] = jaccard_sim
    
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
    print("Test function executed!")




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

"""
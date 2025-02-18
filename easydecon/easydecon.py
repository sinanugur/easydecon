import scanpy as sc
import numpy as np
try:
    import fireducks.pandas as pd
    #print("Using fireducks.pandas for enhanced functionality.")
except ImportError:
    import pandas as pd
    #print("fireducks.pandas not found. Falling back to standard pandas.")

import spatialdata as sp
import spatialdata_io
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean

from tqdm.auto import tqdm
from skimage.filters import threshold_otsu
from skimage.filters import threshold_yen
from skimage.filters import threshold_li
from sklearn.mixture import GaussianMixture
import logging
import warnings

def _suppress_warnings_in_worker():
    """ Suppress warnings and logging inside joblib workers. """
    logging.getLogger().setLevel(logging.CRITICAL)
    warnings.simplefilter("ignore")

def process_row_with_suppression(row, func, **kwargs):
    """ Wrapper around `process_row` to suppress warnings in each worker. """
    _suppress_warnings_in_worker()  # Suppress inside the worker
    return process_row(row, func, **kwargs)

from .config import config

from joblib import Parallel, delayed
from scipy.stats import zscore

from spatialdata import polygon_query
import warnings


# Ensure that the progress_apply method is available
tqdm.pandas()
logger = logging.getLogger(__name__)



def common_markers_gene_expression_and_filter(
    sdata: object,
    marker_genes: list[str],
    common_group_name: str,
    exclude_group_names: list[str] = [],
    bin_size: int = 8,
    filtering_algorithm: str = "quantile",
    aggregation_method: str = "sum",
    add_to_obs: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Aggregate and filter marker gene expression in spatial transcriptomics data.

    This function:
      1. Retrieves a specific table from `sdata.tables` based on the `bin_size`.
      2. Excludes certain spots based on provided `exclude_group_names`.
      3. Aggregates selected marker genes (`marker_genes`) using a chosen method.
      4. Applies a specified thresholding algorithm to filter noise.
      5. Optionally merges results back into `table.obs`.

    Parameters
    ----------
    sdata : object
        The spatial transcriptomics data container, expected to have a `.tables` attribute
        which can be indexed with a key like "square_00{bin_size}um".
    marker_genes : List[str]
        A list of marker gene names to be aggregated.
    common_group_name : str
        Column name under which the aggregated/filtered expression will be stored.
    exclude_group_names : List[str], optional
        Spot group names to exclude from analysis. If a name is found in `table.obs.columns`,
        spots where this column is non-zero are excluded. Defaults to an empty list.
    bin_size : int, optional
        The bin size used to construct the key for retrieving the table from `sdata.tables`.
        For example, if `bin_size=8`, the function attempts to retrieve the table at key
        "square_008um". Defaults to 8.
    filtering_algorithm : str, optional
        Thresholding algorithm used to filter gene expression. Valid options:
        - "otsu"
        - "yen"
        - "li"
        - "quantile"
        Defaults to "quantile".
    aggregation_method : str, optional
        Aggregation method for combining the specified marker genes. Valid options:
        - "sum"
        - "mean"
        - "median"
        Defaults to "sum".
    add_to_obs : bool, optional
        Whether to add the aggregated/filtered values to `table.obs`. Defaults to True.
    **kwargs
        Additional keyword arguments. If `filtering_algorithm="quantile"`, this should
        include `quantile` (float between 0 and 1). For example, `quantile=0.7`.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the aggregated and thresholded gene expression values for each spot.
        The DataFrame contains a single column named `common_group_name`.

    Notes
    -----
    - If no valid marker genes are found (i.e., none intersect with `table.var_names`), 
      the function prints a warning and returns an empty DataFrame.
    - If `add_to_obs=True`, an existing column in `table.obs` with the same name as 
      `common_group_name` will be dropped (if present) before merging in the new data.
    """
    try:
        table_key = f"square_00{bin_size}um"
        table = sdata.tables[table_key]
    except:
        table=sdata
    spots_to_be_used = table.obs.index
    if exclude_group_names:
        print("Excluding spots with group names:")
        spots_g = []
        for g in exclude_group_names:
            print(g)
            if g in table.obs.columns:
                spots_g.extend(table.obs[table.obs[g] != 0].index.values.tolist())
            else:
                print(f"Group name {g} not found in the table.")

        spots_to_be_used=spots_to_be_used.difference(spots_g)
            

    filtered_genes = list(set(marker_genes).intersection(table.var_names))
    if not filtered_genes:
        # Log a warning, return empty, or handle gracefully
        print("Warning: None of the specified marker_genes are present in table.var_names.")
        return pd.DataFrame()
    
    aggregation_funcs = {
        "sum": np.sum,
        "mean": np.mean,
        "median": np.median
    }

    if aggregation_method not in aggregation_funcs:
        raise ValueError("Please provide a valid aggregation method: sum, mean, or median")

    aggregator = aggregation_funcs[aggregation_method]
    gene_expression = table[spots_to_be_used, filtered_genes].to_df().agg(aggregator, axis=1)
    gene_expression = gene_expression.to_frame(common_group_name)


    if filtering_algorithm=="otsu":
        threshold=threshold_otsu(gene_expression[gene_expression[common_group_name] !=0].values)
    elif filtering_algorithm=="yen":
        threshold=threshold_yen(gene_expression[gene_expression[common_group_name] !=0].values)
    elif filtering_algorithm=="li":
        threshold=threshold_li(gene_expression[gene_expression[common_group_name] !=0].values)
    elif filtering_algorithm=="quantile":
        threshold=gene_expression[gene_expression[common_group_name] !=0].quantile(kwargs.get("quantile",0.7)).values
    else:
        raise ValueError("Please provide a valid filtering algorithm: otsu, yen, li or use quantile")
    gene_expression[common_group_name] = np.where(gene_expression[common_group_name].values >= threshold, gene_expression[common_group_name], 0)


    gene_expression[common_group_name]=gene_expression[common_group_name].fillna(0)
    if add_to_obs:
        if common_group_name in table.obs.columns:
            table.obs.drop(columns=[common_group_name], inplace=True,errors='ignore')
        table.obs=pd.merge(table.obs, gene_expression, left_index=True, right_index=True,how='left')
        #table.obs = table.obs.join(gene_expression, how='left')  
        table.obs[common_group_name]=table.obs[common_group_name].fillna(0)
    
    return gene_expression


#this function is used to read the markers from a file or from an single-cell anndata object and return a dataframe
def read_markers_dataframe(sdata,filename=None,adata=None,exclude_celltype=[],
                           bin_size=8,top_n_genes=60,sort_by_column="logfoldchanges",
                           ascending=False,gene_id_column="names",celltype="group",key="rank_genes_groups"): #100
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

def get_clusters_expression_on_tissue(sdata,markers_df,common_group_name=None,
                                      bin_size=8,gene_id_column="names",aggregation_method="mean",add_to_obs=True):

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

    if aggregation_method=="mean":
        compute = lambda x: np.mean(x, axis=1).values
    elif aggregation_method=="median":
        compute = lambda x: np.median(x, axis=1).values
    elif aggregation_method=="sum":
        compute = lambda x: np.sum(x, axis=1).values

    # Preallocate DataFrame with zeros
    all_spots = table.obs.index
    all_clusters = markers_df_tmp.index.unique()
    df = pd.DataFrame(0, index=all_spots, columns=all_clusters)
    #tqdm._instances.clear()

    # Process only spots with expression
    for spot in tqdm(spots_with_expression, desc='Processing spots'):
        a = {}
        for cluster in all_clusters:
            genes = markers_df_tmp.loc[[cluster]][gene_id_column]
            genes = [genes] if isinstance(genes, str) else genes.values
            group_expression = compute(table[spot, genes].to_df())
            a[cluster] = group_expression
        
        # Directly assign to preallocated DataFrame
        df.loc[spot] = pd.DataFrame.from_dict(a, orient='index').transpose().values
    
    if add_to_obs:
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



def get_clusters_by_similarity_on_tissue(
    sdata,
    markers_df,
    common_group_name=None,
    bin_size=8,
    gene_id_column="names",
    #similarity_by_column="logfoldchanges",
    method="wjaccard",
    #weight_column=None,
    add_to_obs=True,
    **kwargs,
):
    """
    Compute cluster assignments based on a chosen similarity method.

    Parameters
    ----------
    sdata : AnnData-like object
        Spatial (or single-cell) data containing expression matrices.
        It is expected to have 'tables' attribute with keys like "square_00Xum",
        or simply be treated as a table if the key doesn't exist.
    markers_df : pd.DataFrame
        DataFrame containing marker genes for each cluster.
        Rows typically represent clusters, columns represent information 
        about each gene (e.g., logfoldchanges, names, etc.).
    common_group_name : str, optional
        Name of a column in `table.obs` specifying spots to process. 
        If found, only spots where `common_group_name != 0` are processed.
        Otherwise, all spots are processed. Default is None.
    bin_size : int, optional
        Determines the bin size (like "square_008um") for looking up the table 
        in `sdata.tables`. Default is 8.
    gene_id_column : str, optional
        Name of the column in `markers_df` that contains gene IDs. 
        Default is "names".
    similarity_by_column : str, optional
        Column in `markers_df` used to measure similarity. 
        Default is "logfoldchanges".
    method : str, optional
        Method to use for computing similarity. Supported methods include:
        "correlation", "cosine", "jaccard", "overlap", "wjaccard",
        "diagnostic", "sum", "mean", "median".
        Default is "wjaccard".
    weight_column : str, optional
        Name of an (optional) column for gene weights.
        Only used in certain similarity methods. Default is None.
    add_to_obs : bool, optional
        If True, adds the resulting assignment columns to `table.obs`. 
        Default is True.
    **method_kwargs : 
        Additional, method-specific parameters. For example:
        - For method="wjaccard": supply ``lambda_param``, ``penalty_param``, etc.
        - For method="cosine": supply ``penalty_param``, etc.
        - For method="jaccard": supply ``threshold``, etc.
        - For method="correlation": supply ``penalty_param``, etc.
        


    Returns
    -------
    pd.DataFrame
        A DataFrame whose index matches `table.obs.index` with cluster 
        assignment columns (or other metrics) computed by the specified method.
    """
    # Try to get the appropriate table from sdata; if not present, treat sdata as the table
    try:
        table = sdata.tables[f"square_00{bin_size}um"]
    except (AttributeError, KeyError):
        table = sdata

    
    # Enable tqdm progress bar in pandas
    from tqdm.auto import tqdm
    tqdm.pandas()

    # Determine which spots to process
    if common_group_name in table.obs.columns:
        spots_with_expression = table.obs[table.obs[common_group_name] != 0].index
    else:
        print("common_group_name column not found in the table, processing all spots.")
        spots_with_expression = table.obs.index

    # Select similarity function based on method
    similarity_methods = {
        "correlation": function_row_spearman,
        "cosine": function_row_cosine,
        "jaccard": function_row_jaccard,
        "overlap": function_row_overlap,
        "wjaccard": function_row_weighted_jaccard,
        "diagnostic": function_row_diagnostic,
        "sum": function_row_sum,
        "mean": function_row_mean,
        "median": function_row_median,
        "euclidean": function_row_euclidean,
        "wjaccardperm": permutation_test,
    }
    if method not in similarity_methods:
        raise ValueError(
            "Invalid method. Choose from: correlation, cosine, jaccard, overlap, "
            "wjaccard, diagnostic, sum, mean, median, wjaccardperm"
        )

    func = similarity_methods[method]
    # Show parallelization info
    from .config import config  # Import inside function to prevent issues with joblib reloading
    print("Number of threads used:", config.n_jobs)
    print("Batch size:", config.batch_size)

    # Run computations in parallel
    results = Parallel(
        n_jobs=config.n_jobs,
        backend="loky",  # Ensure workers do not inherit unnecessary imports
    )(
        delayed(process_row_with_suppression)(row, func, markers_df=markers_df, gene_id_column=gene_id_column, **kwargs)
        for _, row in tqdm(table[spots_with_expression,].to_df().iterrows(), total=len(spots_with_expression))
    )


    # Convert results to a DataFrame
    result_df = pd.DataFrame(results)
    result_df.set_index("Index", inplace=True)
    result_df = result_df["assigned_cluster"].apply(pd.Series)

    # For spots not processed (e.g., excluded by common_group_name != 0)
    # fill with zeros or NaNs, depending on your needs
    others_df = pd.DataFrame(
        0, 
        index=list(set(table.obs.index) - set(spots_with_expression)), 
        columns=result_df.columns
    )
    df = pd.concat([result_df, others_df])

    # Optionally merge back into table.obs
    if method != "diagnostic" or add_to_obs:
        # Avoid collisions with existing columns
        table.obs.drop(columns=df.columns, inplace=True, errors='ignore')
        table.obs = pd.merge(table.obs, df, left_index=True, right_index=True)

    return df


def permutation_test(row, markers_df, num_permutations=100, **kwargs):
    observed_scores = function_row_weighted_jaccard(row, markers_df, **kwargs)
    null_distributions = {cluster: [] for cluster in observed_scores.keys()}

    for _ in range(num_permutations):
        # Permute gene labels in 'row'
        permuted_row = row.copy()
        permuted_row.index = np.random.permutation(permuted_row.index)

        # Compute similarity scores for permuted data
        permuted_scores = function_row_weighted_jaccard(permuted_row, markers_df, **kwargs)

        # Collect scores for each cluster
        for cluster, score in permuted_scores.items():
            null_distributions[cluster].append(score)

    # Calculate p-values
    p_values = {}
    for cluster in observed_scores.keys():
        observed_score = observed_scores[cluster]
        null_scores = null_distributions[cluster]
        p = (np.sum(np.array(null_scores) >= observed_score) + 1) / (num_permutations + 1)
        p_values[cluster] = -1*np.log10(p)

    #return observed_scores, p_values
    return p_values



def function_row_spearman(row, markers_df,**kwargs):
    gene_id_column=kwargs.get("gene_id_column","names")
    similarity_by_column=kwargs.get("similarity_by_column","logfoldchanges")
    penalty_param=kwargs.get("penalty_param",0.5)

    a = {}
    for c in markers_df.index.unique():
        vector_series = pd.Series(markers_df[[gene_id_column,similarity_by_column]].loc[[c]][similarity_by_column].values, index=markers_df[[gene_id_column, similarity_by_column]].loc[[c]][gene_id_column].values)
        l = len(vector_series)
        vector_series = vector_series.reindex(row.index, fill_value=np.nan)
        valid_mask = ~vector_series.isna() & ~row.isna()
        t = (row[valid_mask] != 0).sum()
        if t == 0:  # No valid pairs
            a[c] = 0.0
        else:
            sp = (spearmanr(row[valid_mask], vector_series[valid_mask], nan_policy="omit")[0])*((t/l)**penalty_param)
            a[c] = sp if sp > 0 else 0.0  # Assign 0 if correlation is negative
    return a





def function_row_cosine(row, markers_df,**kwargs):
    gene_id_column=kwargs.get("gene_id_column","names")
    similarity_by_column=kwargs.get("similarity_by_column","logfoldchanges")
    penalty_param=kwargs.get("penalty_param",0.5)
    


    a = {}
    for c in markers_df.index.unique():
        vector_series = pd.Series(markers_df[[gene_id_column,similarity_by_column]].loc[[c]][similarity_by_column].values, index=markers_df[[gene_id_column, similarity_by_column]].loc[[c]][gene_id_column].values)
        l = len(vector_series)
        vector_series = vector_series.reindex(row.index, fill_value=np.nan)
        vector_series = min_max_scale(vector_series)
        valid_mask = ~vector_series.isna() & ~row.isna()
        t = (row[valid_mask] != 0).sum()
        if t == 0:  # No valid pairs
            a[c] = 0.0
        else:
            a[c] = (1 - cosine(row[valid_mask], vector_series[valid_mask]))*((t/l)**penalty_param) #penalize the cosine similarity by the fraction of valid pairs
        
    return a




def function_row_euclidean(row, markers_df, **kwargs):
    gene_id_column = kwargs.get("gene_id_column", "names")
    similarity_by_column = kwargs.get("similarity_by_column", "logfoldchanges")
    penalty_param = kwargs.get("penalty_param", 0.5)
    
    a = {}
    for c in markers_df.index.unique():
        vector_series = pd.Series(
            markers_df[[gene_id_column, similarity_by_column]].loc[[c]][similarity_by_column].values,
            index=markers_df[[gene_id_column, similarity_by_column]].loc[[c]][gene_id_column].values
        )
        l = len(vector_series)
        vector_series = vector_series.reindex(row.index, fill_value=np.nan)
        vector_series = min_max_scale(vector_series)
        valid_mask = ~vector_series.isna() & ~row.isna()
        
        # Number of non-zero valid entries
        t = (row[valid_mask] != 0).sum()
        
        if t == 0:  # No valid pairs
            a[c] = 0.0
        else:
            distance_val = euclidean(row[valid_mask], vector_series[valid_mask])
            
            # Convert Euclidean distance to similarity in [0,1]: higher distance -> lower similarity
            similarity_val = 1 / (1 + distance_val)
            
            # Apply the penalty factor
            a[c] = similarity_val * ((t / l) ** penalty_param)
    
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
    #threshold=kwargs.get("threshold")
    for c in markers_df.index.unique():
        #row_set = set(row[row > threshold].sort_values(ascending=False).index)
        row_set = set(row[row > 0].sort_values(ascending=False).index) #non-zero values
        vector_set = set(markers_df.loc[[c]][gene_id_column].values)
        
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
    #threshold=kwargs.get("threshold")
    for c in markers_df.index.unique():
        row_set = set(row[row > 0].sort_values(ascending=False).index) #non-zero values
        vector_set = set(markers_df.loc[[c]][gene_id_column].values)
        
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
    
    for c in markers_df.index.unique():
        row_set = set(row[row > 0].sort_values(ascending=False).index) #non-zero values
        vector_set = set(markers_df.loc[[c]][gene_id_column].values)
        
        # Calculate intersection
        a[c] = row_set.intersection(vector_set)
    return a

def function_row_sum(row, markers_df, **kwargs):
    a = {}
    gene_id_column=kwargs.get("gene_id_column")
    
    for c in markers_df.index.unique():
        #row_set = set(row[row > 0].sort_values(ascending=False).index) #non-zero values
        vector_set = markers_df.loc[[c]][gene_id_column].values
        
        
        # Calculate intersection and union
        #a[c] = row_set.intersection(vector_set)
        a[c] = row[vector_set].sum()
    return a

def function_row_mean(row, markers_df, **kwargs):
    a = {}
    gene_id_column=kwargs.get("gene_id_column")
    
    for c in markers_df.index.unique():
        #row_set = set(row[row > 0].sort_values(ascending=False).index) #non-zero values
        vector_set = markers_df.loc[[c]][gene_id_column].values
        
        
        # Calculate intersection and union
        #a[c] = row_set.intersection(vector_set)
        a[c] = row[vector_set].mean()
    return a

def function_row_median(row, markers_df, **kwargs):
    a = {}
    gene_id_column=kwargs.get("gene_id_column")
    for c in markers_df.index.unique():
        #row_set = set(row[row > 0].sort_values(ascending=False).index) #non-zero values
        vector_set = markers_df.loc[[c]][gene_id_column].values
        
        
        # Calculate intersection and union
        #a[c] = row_set.intersection(vector_set)
        a[c] = row[vector_set].median()
    return a


def function_row_weighted_jaccard(row, markers_df, **kwargs):
    gene_id_column = kwargs.get("gene_id_column","names")
    weight_column = kwargs.get("weight_column", None)  # Name of the weight column in markers_df
    lambda_param = kwargs.get("lambda_param", 1.0)  # Default lambda for exponential decay
    a = {}
    # Get the genes and their expression levels from 'row' (target set)
    # Normalize the expression levels to range between 0 and 1
    target_genes = row[row > 0]  # Select genes with expression > 0
    max_expr = target_genes.max()
    if max_expr > 0:
        target_weights = target_genes / max_expr
    else:
        target_weights = target_genes  # Will be an empty Series
    
    # Determine if pre-calculated weights are to be used
    use_precalculated_weights = weight_column is not None and weight_column in markers_df.columns
    
    # Iterate over each cluster
    for c in markers_df.index.unique():
        # Extract genes and weights for the current cluster
        cluster_df = markers_df.loc[[c]]
        cluster_genes = cluster_df[gene_id_column].reset_index(drop=True)
        if use_precalculated_weights:
            # Use pre-calculated weights
            cluster_weight_values = cluster_df[weight_column].reset_index(drop=True)
            # Normalize cluster weights to range between 0 and 1
            max_weight = cluster_weight_values.max()
            if max_weight > 0:
                cluster_weights = cluster_weight_values / max_weight
            else:
                cluster_weights = cluster_weight_values
            # Create a pandas Series with genes as index and weights as values
            cluster_weights = pd.Series(cluster_weights.values, index=cluster_genes)
        else:
            # Assign exponential rank-based weights
            N = len(cluster_genes)
            if N > 0:
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
            # Weight in 'cluster_weights', 0 if gene not present
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

def apply_filtering_algorithm(data, filtering_algorithm="otsu",quantile=0.7,gaussian_components=2):
    """
    Applies a filtering algorithm to the input data. If the data is a DataFrame,
    the algorithm is applied to each column separately. If the data is array-like,
    the algorithm is applied to the entire array.

    Parameters:
        data (pd.DataFrame or array-like): The input data.
        filtering_algorithm (str): The filtering algorithm to use ('yen', 'otsu', 'li').

    Returns:
        The filtered data with values below the threshold set to zero.
    """
    # Define a helper function to compute the threshold
    def compute_threshold(values, algorithm):
        non_zero_values = values[values != 0]
        if len(non_zero_values) == 0:
            return 0  # All values are zero
        if algorithm == "otsu":
            threshold = threshold_otsu(non_zero_values)
        elif algorithm == "yen":
            threshold = threshold_yen(non_zero_values)
        elif algorithm == "li":
            threshold = threshold_li(non_zero_values)
        elif algorithm == "quantile":
            threshold=np.quantile(non_zero_values,quantile)
        elif algorithm == "gaussian":
            gmm = GaussianMixture(n_components=gaussian_components)
            gmm.fit(non_zero_values.reshape(-1, 1))
            means = gmm.means_.flatten()
            threshold = np.min(means)
        else:
            raise ValueError("Please provide a valid filtering algorithm: yen, otsu or li")
        return threshold

    # Check if the input is a DataFrame
    if isinstance(data, pd.DataFrame):
        filtered_data = data.copy()
        for column in filtered_data.columns:
            values = filtered_data[column].values
            threshold = compute_threshold(values, filtering_algorithm)
            # Set values below the threshold to zero
            filtered_data[column] = np.where(values >= threshold, values, 0)
        return filtered_data
    else:
        # Assume the input is array-like
        values = np.asarray(data)
        threshold = compute_threshold(values, filtering_algorithm)
        # Set values below the threshold to zero
        filtered_values = np.where(values >= threshold, values, 0)
        return filtered_values


def test_function():
    print("Easydecon loaded!")
    print("Test function executed!")


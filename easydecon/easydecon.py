import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
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
from scipy.stats import gamma
from scipy.stats import expon
from scipy.optimize import nnls
from scipy.stats import zscore
from tqdm.auto import tqdm
from skimage.filters import threshold_otsu
from skimage.filters import threshold_yen
from skimage.filters import threshold_li
from sklearn.mixture import GaussianMixture
import logging

from sklearn.linear_model import Ridge, Lasso, ElasticNet

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

from spatialdata import polygon_query
import warnings


# Ensure that the progress_apply method is available
#tqdm.pandas()
logger = logging.getLogger(__name__)

def common_markers_gene_expression_and_filter(
    sdata: object,
    marker_genes,  # can be list, dict, or DataFrame
    common_group_name: str = "MarkerGroup",  # used if marker_genes is a list
    group_col: str = "group",               # DF column holding group IDs
    gene_col: str = "names",                # DF column holding marker gene names
    exclude_group_names: list[str] = [],
    bin_size: int = 8,
    aggregation_method: str = "sum",
    add_to_obs: bool = True,
    filtering_algorithm: str = "permutation",  # or "quantile"
    num_permutations: int = 5000,
    alpha: float = 0.05,
    subsample_size: int = 20000,
    quantile: float = 0.7, #if quantile selected
    parametric: bool = True, #if parametric, gamma or exponential distribution is used
    min_counts_quantile: float = 0,
    n_subs: int = 5,                 # number of subsamples
    **kwargs
) -> pd.DataFrame:
    """
    Extended version allowing marker_genes as a list, dict, or DataFrame, with
    customizable column names for the DataFrame.

    If marker_genes is:
      1) list[str]: Single group of markers -> create one column named `common_group_name`.
      2) dict[str, list[str]]: Multiple groups -> each dict key becomes a column in table.obs.
      3) pd.DataFrame: Must contain columns for groups and gene names (by default 'group' and 'names'),
         but these can be overridden by `group_col` and `gene_col`.

    Steps (for each group):
      1) Compute aggregator (sum, mean, median, cs) for all bins over that group's marker genes.
      2) If filtering_algorithm="permutation", subsample bins (subsample_size) and build a null distribution
         by randomly picking genes of size=len(marker_genes).
      3) If filtering_algorithm="quantile", compute threshold from (1 - quantile).
      4) Apply cutoff to all bins (values below threshold become 0).
      5) Merge results back into `table.obs` if `add_to_obs=True`.

    Parameters
    ----------
    sdata : object
        Spatial data container, with sdata.tables[table_key].
    marker_genes : list, dict, or pd.DataFrame
        - list[str] for a single group of marker genes.
        - dict[group_name -> list_of_markers] for multiple groups.
        - pd.DataFrame with columns [group_col, gene_col].
    common_group_name : str, optional
        If marker_genes is a list, the new column name. Default "MarkerGroup".
    group_col : str, optional
        Column name in marker_genes DataFrame for the group identifier. Default "group".
    gene_col : str, optional
        Column name in marker_genes DataFrame for the gene names. Default "names".
    exclude_group_names : list[str], optional
        Exclude spots where table.obs[g] != 0 for g in exclude_group_names.
    bin_size : int, optional
        Key for retrieving table from sdata (e.g. "square_008um").
    filtering_algorithm : str, optional
        "quantile" or "permutation".
    aggregation_method : str, optional
        "sum", "mean", "median", or "cs" (composite_score).
    add_to_obs : bool, optional
        Whether to merge results back into table.obs.
    num_permutations : int, optional
        Number of permutations (only relevant if filtering_algorithm="permutation").
    alpha : float, optional
        Significance level for the permutation-based cutoff. Default 0.05.
    subsample_size : int, optional
        How many bins to sample for the permutation-based null. Default 50000.
    quantile : float, optional
        Used if filtering_algorithm="quantile". Default 0.7.
    min_counts_quantile : float, optional
        Exclude bins below this quantile of table.obs["total_counts"] from permutation sampling.
        Default 0 (exclude bottom).
    n_subs : int, optional
        Number of smaller subsamples. We derive each subset size from subsample_size // n_subs.
        Default 5.
    **kwargs
        Additional arguments if needed.

    Returns
    -------
    pd.DataFrame
        The final DataFrame with aggregated + thresholded expression for each group.
        Columns = one per group, indexed by bin.
    """

    # -----------------------------------------------------------
    # 0) Convert marker_genes input to a dictionary: group -> list of genes
    # -----------------------------------------------------------
    if isinstance(marker_genes, list):
        # Single group: user gave a plain list of gene names
        group_dict = {common_group_name: marker_genes}

    elif isinstance(marker_genes, dict):
        # Already in dict form: group_name -> [list_of_genes]
        group_dict = marker_genes

    elif isinstance(marker_genes, pd.DataFrame):
        # Expect columns group_col and gene_col
        required_cols = {group_col, gene_col}
        if not required_cols.issubset(marker_genes.columns):
            raise ValueError(
                f"DataFrame for marker_genes must have columns: {group_col}, {gene_col}."
            )
        # Build a dict: group_name -> list_of_genes
        group_dict = {}
        marker_genes_tmp = marker_genes.copy()
        try:
            marker_genes_tmp.index.names=[""]
        except:
            pass

        for gname, sub_df in marker_genes_tmp.groupby(group_col):
            # Extract unique gene names in this group
            genes_for_gname = sub_df[gene_col].unique().tolist()
            group_dict[gname] = genes_for_gname
    else:
        raise TypeError(
            "marker_genes must be a list, dict, or DataFrame with the appropriate columns."
        )

    # 1) Retrieve the table
    try:
        table = sdata.tables[f"square_00{bin_size}um"]
    except (AttributeError, KeyError):
        table = sdata

    # 2) Exclude spots
    spots_to_be_used = table.obs.index
    if exclude_group_names:
        for g in exclude_group_names:
            if g in table.obs.columns:
                spots_excluded = table.obs[table.obs[g] != 0].index
                spots_to_be_used = spots_to_be_used.difference(spots_excluded)

    # Prepare a final DataFrame to collect group results
    result_df = pd.DataFrame(index=spots_to_be_used)

    # Aggregation functions
    aggregation_funcs = {
        "sum": "sum",
        "mean": "mean",
        "median": "median",
        "cs": composite_score,  # or any custom aggregator
    }

    if aggregation_method not in aggregation_funcs:
        raise ValueError("aggregation_method must be one of: sum, mean, median or cs.")
    aggregator = aggregation_funcs[aggregation_method]

    tqdm.pandas()
    # -----------------------------------------------------------
    # Loop over each group in the dictionary
    # -----------------------------------------------------------
    for group_name, gene_list in group_dict.items():
        # Intersect gene_list with table.var_names
        filtered_genes = set(gene_list).intersection(table.var_names)
        filtered_genes = list(filtered_genes)
        if not filtered_genes:
            print(f"Warning: No valid marker genes found for group '{group_name}'.")
            # We'll create a column of all zeros
            result_df[group_name] = 0
            continue

        # Retrieve expression for the selected spots & genes
        expr_matrix = table[spots_to_be_used, filtered_genes].to_df()

        if isinstance(aggregator, str):
            aggregated_vals = expr_matrix.agg(aggregator, axis=1)
        else:
            aggregated_vals = expr_matrix.apply(aggregator, axis=1)

        group_expression = aggregated_vals.to_frame(name=group_name)

        # Apply the chosen filtering algorithm
        if filtering_algorithm == "quantile":
            # Threshold from non-zero aggregator values
            non_zero_vals = group_expression[group_expression[group_name] != 0][group_name]
            threshold = non_zero_vals.quantile(quantile)

        elif filtering_algorithm == "permutation":
            # Subsample spots
            try:
                total_counts_series = table.obs.loc[spots_to_be_used, "total_counts"]
                # e.g., exclude bins below the 0.1 quantile (bottom 10%)
                cutoff_value = total_counts_series.quantile(min_counts_quantile)
                # Keep only bins above that cutoff
                candidate_spots = total_counts_series[total_counts_series >= cutoff_value].index
            except KeyError:
                # No total_counts column
                candidate_spots = spots_to_be_used

            if len(candidate_spots) == 0:
                # Edge case: if everything was below cutoff
                print(f"Warning: no bins passed the total_counts quantile filter for {group_name}.")
                threshold = 0
            else:
                all_null_scores = []
                marker_set_size = len(filtered_genes)

                # Determine each subset size
                subset_size_each = subsample_size // n_subs
                remainder = subsample_size % n_subs  # if not divisible

                # n_subs loops
                for i in range(n_subs):
                    # For remainder distribution, you can let the first few subsets be bigger or smaller
                    current_subset_size = subset_size_each
                    if remainder > 0:
                        current_subset_size += 1
                        remainder -= 1
                    if len(candidate_spots) > current_subset_size:
                        subset_spots = np.random.choice(candidate_spots, size=current_subset_size, replace=False)
                    else:
                        subset_spots = candidate_spots

                    # Build null distribution for this subset
                    all_expr_df = table[subset_spots, :].to_df()

                    for _ in tqdm(
                        range(int(num_permutations/n_subs)),
                        #desc=f"Perm sub {i+1}/{n_subs} of {current_subset_size} for {group_name}",
                        desc=f"Subsample {current_subset_size*(i+1)}/{subsample_size} for {group_name}",
                        leave=True,
                        position=0
                    ):
                        random_genes = np.random.choice(table.var_names, size=marker_set_size, replace=False)
                        if isinstance(aggregator, str):
                            random_vals = all_expr_df[random_genes].agg(aggregator, axis=1)
                        else:
                            random_vals = all_expr_df[random_genes].apply(aggregator, axis=1)
                        all_null_scores.append(random_vals.values)
                # Concatenate results
                null_scores_concat = np.concatenate(all_null_scores)
                nonzero_null_vals = null_scores_concat[null_scores_concat > 0]
                if len(nonzero_null_vals) == 0:
                    print("Warning: no positive values in null distribution, threshold set to 0.")
                    threshold = 0
                else:
                    if not parametric:
                        threshold = np.quantile(nonzero_null_vals, 1 - alpha)
                    else:
                        if aggregation_method != "cs":
                            shape_hat, loc_hat, scale_hat = gamma.fit(nonzero_null_vals,floc=0)
                            threshold = gamma.ppf(1 - alpha, shape_hat, loc=loc_hat, scale=scale_hat)
                            
                        else:
                            loc_hat, scale_hat = expon.fit(nonzero_null_vals,floc=0)
                            threshold = expon.ppf(1 - alpha, loc=loc_hat, scale=scale_hat)
        else:
            raise ValueError("Invalid filtering_algorithm. Use 'quantile' or 'permutation'.")

        # Zero out values below threshold
        group_expression[group_name] = np.where(
            group_expression[group_name] >= threshold,
            group_expression[group_name],
            0
        )
        result_df[group_name] = group_expression[group_name].fillna(0)
    # -----------------------------------------------------------
    # Merge results back into obs if requested
    # -----------------------------------------------------------
    if add_to_obs:
        # Drop existing columns of same names if present
        for col in result_df.columns:
            if col in table.obs.columns:
                table.obs.drop(columns=[col], inplace=True, errors='ignore')
        # Merge
        table.obs = pd.merge(table.obs, result_df, left_index=True, right_index=True, how='left')
        for col in result_df.columns:
            table.obs[col] = table.obs[col].fillna(0)

    return result_df



#this function is used to read the markers from a file or from an single-cell anndata object and return a dataframe
def read_markers_dataframe(sdata,filename=None,adata=None,exclude_celltype=[],
                           bin_size=8,top_n_genes=60,sort_by_column="scores",
                           ascending=False,gene_id_column="names",celltype="group",key="rank_genes_groups",log2fc_min=0.25,pval_cutoff=0.05): #100
    try:
        table = sdata.tables[f"square_00{bin_size}um"]
    except (AttributeError, KeyError):
        table = sdata

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
            df=sc.get.rank_genes_groups_df(adata,group=None, key=key, pval_cutoff=pval_cutoff, log2fc_min=log2fc_min)
        except:
            raise ValueError("Please provide a valid adata object with rank_genes_groups key")
            
    if "logfoldchanges" in df.columns:
        df=df[df["logfoldchanges"] >= log2fc_min]
    if "pvals_adj" in df.columns:
        df=df[df["pvals_adj"] <= pval_cutoff]

    df = df[df[gene_id_column].isin(table.var_names)] #check if the var_names are present in the spatial data
    df = df[~df[celltype].isin(exclude_celltype)]
    df = df.sort_values(by=sort_by_column, ascending=ascending)
    df = df.groupby(celltype).head(top_n_genes)
    print("Unique cell types detected in the dataframe:")
    print(df[celltype].unique())
    df.set_index(celltype,inplace=True,drop=False)
    return df

def get_clusters_expression_on_tissue(sdata,markers_df,common_group_name=None,
                                      bin_size=8,gene_id_column="names",aggregation_method="mean",add_to_obs=True):

    try:
        table = sdata.tables[f"square_00{bin_size}um"]
    except (AttributeError, KeyError):
        table = sdata
        
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
    
    tqdm.pandas()

    # Process only spots with expression
    for spot in tqdm(spots_with_expression, desc='Processing spots',leave=True, position=0):
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


def get_proportions_on_tissue(
    sdata,
    markers_df,
    common_group_name=None,
    bin_size=8,
    gene_id_column="names",
    similarity_by_column="logfoldchanges",
    method="nnls", # Options: 'nnls', 'ridge', 'lasso', 'elastic'
    normalization_method="unit",  # Options: 'unit', 'zscore',"l1"
    add_to_obs=True,
    alpha=0.1,
    l1_ratio=0.7,
    verbose=True,
):
    """
    Compute cell-type proportions per spatial bin using NNLS-based deconvolution.

    Parameters
    ----------
    sdata : AnnData-like object
        Spatial data containing expression matrices.
    markers_df : pd.DataFrame
        DataFrame containing marker genes with fold-change or scores for each cell type.
        The index of markers_df should represent cell-type groups.
    common_group_name : str, optional
        Column in table.obs to specify spots to process. If None, all spots are processed.
    bin_size : int, optional
        Bin size for spatial data lookup, default is 8.
    gene_id_column : str, optional
        Column in markers_df with gene identifiers, default is "names".
    similarity_by_column : str, optional
        Column in markers_df containing fold-change values, default is "logfoldchanges".
    normalization_method : str, optional
        Method for normalizing reference matrix, options are "unit", "l1" or "zscore", default is "unit".
    add_to_obs : bool, optional
        If True, add proportions to table.obs, default is True.
    verbose : bool, optional
        Show progress and info, default is True.

    Returns
    -------
    pd.DataFrame
        Cell-type proportions per spatial bin.
    """
    try:
        table = sdata.tables[f"square_00{bin_size}um"]
    except (AttributeError, KeyError):
        table = sdata

    # Determine spots to process
    if common_group_name in table.obs.columns:
        spots_with_expression = table.obs[table.obs[common_group_name] != 0].index
    else:
        if verbose:
            print("common_group_name not found, processing all spots.")
        spots_with_expression = table.obs.index

    spatial_expr = table[spots_with_expression].to_df().T

    # Prepare combined reference matrix from fold-change values
    marker_groups = markers_df.index.unique()
    all_valid_genes = set()
    group_gene_values = {}

    for group in marker_groups:
        group_markers = markers_df.loc[[group]]
        group_markers = group_markers[group_markers[similarity_by_column] > 0].dropna(subset=[similarity_by_column])
        genes = group_markers[gene_id_column].values
        valid_genes = np.intersect1d(genes, spatial_expr.index)

        if len(valid_genes) == 0:
            if verbose:
                print(f"No valid genes found for group '{group}'. Skipping.")
            continue

        fold_changes = group_markers.set_index(gene_id_column).loc[valid_genes, similarity_by_column]
        group_gene_values[group] = fold_changes
        all_valid_genes.update(valid_genes)

    all_valid_genes = sorted(all_valid_genes)
    ref_matrix_df = pd.DataFrame(0, index=all_valid_genes, columns=marker_groups)

    for group, gene_values in group_gene_values.items():
        ref_matrix_df.loc[gene_values.index, group] = gene_values.values

    # Normalize reference matrix
    ref_matrix_df = ref_matrix_df.replace([np.inf, -np.inf], 0).fillna(0)
    ref_matrix_df = ref_matrix_df.loc[ref_matrix_df.sum(axis=1) > 0]

    if normalization_method == "unit":
        ref_matrix_df = ref_matrix_df.apply(lambda x: x / np.linalg.norm(x) if np.linalg.norm(x) != 0 else x, axis=0)
        spatial_expr=spatial_expr.apply(lambda x: x / np.linalg.norm(x) if np.linalg.norm(x) != 0 else x, axis=0)
    elif normalization_method == "l1":
        ref_matrix_df = ref_matrix_df.apply(lambda x: x / np.linalg.norm(x, ord=1) if np.linalg.norm(x, ord=1) != 0 else x, axis=0)
        spatial_expr=spatial_expr.apply(lambda x: x / np.linalg.norm(x, ord=1) if np.linalg.norm(x, ord=1) != 0 else x, axis=0)
    elif normalization_method == "zscore":
        ref_matrix_df = ref_matrix_df.apply(lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) != 0 else x, axis=0).fillna(0)
        spatial_expr=spatial_expr.apply(lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) != 0 else x, axis=0).fillna(0)
    elif normalization_method is None:
        pass
    else:
        raise ValueError("normalization_method must be 'unit', 'l1' or 'zscore'")

    # NNLS deconvolution in parallel
    if verbose:
        print("Running deconvolution with parallel processing...")
        print(f"Normalization method: {normalization_method}")

    def fit_single_bin(bin_expr):
        _suppress_warnings_in_worker()  # Suppress inside the worker
        if method == "nnls":
            coef, _ = nnls(ref_matrix_df.values, bin_expr)
        elif method == "ridge":
            model = Ridge(alpha=alpha, fit_intercept=False, positive=True)
            model.fit(ref_matrix_df, bin_expr)
            coef = model.coef_
        elif method == "lasso":
            model = Lasso(alpha=alpha, fit_intercept=False, positive=True)
            model.fit(ref_matrix_df, bin_expr)
            coef = model.coef_
        elif method == "elastic":
            model = ElasticNet(alpha=alpha,l1_ratio=l1_ratio, fit_intercept=False, positive=True)
            model.fit(ref_matrix_df, bin_expr)
            coef = model.coef_

        else:
            raise ValueError("method must be 'nnls', 'ridge', 'lasso' or 'elastic'")

        if coef.sum() > 0:
            coef /= coef.sum()
        return coef
    
    if verbose:
        print("Number of threads used:", config.n_jobs)
        print(f"Running deconvolution with method='{method}', alpha={alpha}...")
        
        

    results = Parallel(n_jobs=config.n_jobs)(
        delayed(fit_single_bin)(spatial_expr.loc[ref_matrix_df.index, bin_id].values)
        for bin_id in tqdm(spatial_expr.columns, total=len(spatial_expr.columns), leave=True,position=0)
    )

    proportions_df = pd.DataFrame(
        results, index=spatial_expr.columns, columns=ref_matrix_df.columns.values
    )
    
    others_df = pd.DataFrame(
        0, 
        index=list(set(table.obs.index) - set(spots_with_expression)), 
        columns=ref_matrix_df.columns.values
    )
    df = pd.concat([proportions_df, others_df])

   
    if add_to_obs:
        table.obs.drop(columns=df.columns, inplace=True, errors='ignore')
        table.obs = pd.merge(table.obs, df, left_index=True, right_index=True)

    if verbose:
        print("Deconvolution completed.")

    return df




def assign_clusters_from_df(sdata, df, bin_size=8, results_column="easydecon", method="max", allow_multiple=True, diagnostic=None, fold_change_threshold=2.0):

    tqdm.pandas()

    try:
        table = sdata.tables[f"square_00{bin_size}um"]
    except (AttributeError, KeyError):
        table = sdata

    table.obs.drop(columns=[results_column], inplace=True, errors='ignore')

    df_filtered = df.loc[table.obs.index]

    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    if method == "max":
        df_reindexed = df_filtered[~(df_filtered == 0).all(axis=1)].idxmax(axis=1).to_frame(results_column).astype('category').reindex(table.obs.index, fill_value=np.nan)

    elif method == "zmax":
        tmp = df_filtered.replace(0, np.nan).apply(lambda x: zscore(x, nan_policy='omit'), axis=0).idxmax(axis=1)
        df_reindexed = tmp.to_frame(results_column).astype('category').reindex(table.obs.index, fill_value=np.nan)

    elif method == "hybrid":
        similarity_zscores = df_filtered.apply(zscore, axis=1).fillna(0)
        adaptive_probs = similarity_zscores.apply(softmax, axis=1)

        def adaptive_assign(row):
            sorted_probs = row.sort_values(ascending=False)
            min_probability = 1.0 / len(row)

            if len(sorted_probs) < 2:
                if sorted_probs.iloc[0] >= min_probability:
                    return sorted_probs.index[0]
                else:
                    return np.nan

            top_prob = sorted_probs.iloc[0]
            second_prob = sorted_probs.iloc[1]

            if (top_prob >= min_probability) and (top_prob >= fold_change_threshold * second_prob):
                return sorted_probs.index[0]
            elif allow_multiple:
                eligible = sorted_probs[sorted_probs >= min_probability]
                if not eligible.empty:
                    return '|'.join(eligible.index.tolist())
                else:
                    return np.nan
            else:
                return np.nan

        assigned_clusters = []
        for _, row in tqdm(adaptive_probs.iterrows(), total=adaptive_probs.shape[0], desc="Assigning clusters"):
            assigned_clusters.append(adaptive_assign(row))

        df_reindexed = pd.DataFrame(assigned_clusters, index=adaptive_probs.index, columns=[results_column]).astype('category').reindex(table.obs.index, fill_value=np.nan)

    else:
        raise ValueError("Please provide a valid method: max, zmax, or hybrid")

    table.obs = pd.merge(table.obs, df_reindexed, left_index=True, right_index=True, how="left")

    if diagnostic is not None:
        for r in table.obs[[results_column]].itertuples(index=True, name='Pandas'):
            if not pd.isna(getattr(r, results_column)):
                table.obs.at[r.Index, results_column] = getattr(r, results_column)

    return df_reindexed

def visualize_only_selected_clusters(sdata,clusters,bin_size=8,results_column="easydecon",temp_column="tmp"):
    try:
        table = sdata.tables[f"square_00{bin_size}um"]
    except (AttributeError, KeyError):
        table = sdata

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
        - For method="wjaccard": supply ``lambda_param``, etc.
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
        "euclidean": function_row_euclidean
    }
    if method not in similarity_methods:
        raise ValueError(
            "Invalid method. Choose from: correlation, cosine, jaccard, overlap, "
            "wjaccard, diagnostic, sum, mean, median"
        )

    func = similarity_methods[method]
    # Show parallelization info
    #from .config import config  # Import inside function to prevent issues with joblib reloading
    print("Number of threads used:", config.n_jobs)
    print("Batch size:", config.batch_size)

    # Run computations in parallel
    results = Parallel(
        n_jobs=config.n_jobs,
        backend="loky",  # Ensure workers do not inherit unnecessary imports
    )(
        delayed(process_row_with_suppression)(row, func, markers_df=markers_df, gene_id_column=gene_id_column, **kwargs)
        for _, row in tqdm(table[spots_with_expression,].to_df().iterrows(), total=len(spots_with_expression),leave=True, position=0)
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
    #penalty_param=kwargs.get("penalty_param",0)
    


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
            #a[c] = (1 - cosine(row[valid_mask], vector_series[valid_mask]))*((t/l)**penalty_param) #penalize the cosine similarity by the fraction of valid pairs
            a[c] = (1 - cosine(row[valid_mask], vector_series[valid_mask]))
    return a




def function_row_euclidean(row, markers_df, **kwargs):
    gene_id_column = kwargs.get("gene_id_column", "names")
    similarity_by_column = kwargs.get("similarity_by_column", "logfoldchanges")
    #penalty_param = kwargs.get("penalty_param", 0)
    
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
            #a[c] = similarity_val * ((t / l) ** penalty_param)
            a[c] = similarity_val
    
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

#Szymkiewicz–Simpson 
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
    lambda_param = kwargs.get("lambda_param", 0.25)  # Default lambda for exponential decay
    a = {}
    # Get the genes and their expression levels from 'row' (target set)
    # Normalize the expression levels to range between 0 and 1
    target_genes = row[row > 0]  # Select genes with expression > 0
    max_expr = target_genes.max()
    if max_expr > 0:
        target_weights = target_genes / max_expr
        #target_weights = target_genes / target_genes.sum() #L1 normalization
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
                #cluster_weights = cluster_weight_values / cluster_weight_values.sum() #L1 normalization
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
                #weights /= weights.sum()  # Normalize weights to sum to 1 L1 normalization
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
        if denominator == 0.0:
            jaccard_sim = 0.0
        else:
            jaccard_sim = numerator / denominator
        
        a[c] = jaccard_sim
    
    return a



def add_df_to_spatialdata(sdata,df,bin_size=8):
    try:
        table = sdata.tables[f"square_00{bin_size}um"]
    except (AttributeError, KeyError):
        table = sdata

    table.obs.drop(columns=df.columns,inplace=True,errors='ignore')
    table.obs=pd.merge(table.obs, df, left_index=True, right_index=True)
    print("DataFrame added to SpatialData object")
    print(df.columns)
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



def composite_score(row):
    nonzero = row[row > 0]
    return nonzero.sum() * (len(nonzero) / len(row)) if not nonzero.empty else 0

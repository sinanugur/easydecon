def get_clusters_expression_on_tissue(sdata,markers_df,common_group_name=None,
                                      bin_size=8,gene_id_column="names",aggregation_method="mean",add_to_obs=True):

    try:
        table = sdata.tables[f"square_00{bin_size}um"]
    except (AttributeError, KeyError):
        table = sdata
        
    markers_df_tmp=markers_df[markers_df[gene_id_column].isin(table.var_names)] #just to be sure the genes are present in the spatial data

    if common_group_name in table.obs.columns:
        print(f"Processing spots with {common_group_name} != 0")
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
        print("Adding results to table.obs of sdata object")
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
    alpha=0.01,
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
        table_name = f"square_{bin_size:03}um"
        table = sdata.tables[table_name]
    except (AttributeError, KeyError):
        try:
            table = sdata.tables["table"]
        except (AttributeError, KeyError):
            table = sdata

    # Determine spots to process
    if common_group_name in table.obs.columns:
        print(f"Processing spots with {common_group_name} != 0")
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
    ref_matrix_df["others"] = 1.0  # Add "others" column

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
        
        """
        predicted = ref_matrix_df.values @ coef
        
        # Compute Pearson correlation
        # pearsonr returns (correlation, p-value)
        corr_value, pval = spearmanr(bin_expr, predicted)
        
        return coef, corr_value
        """
    


    if verbose:
        print("Number of threads used:", config.n_jobs)
        print(f"Running deconvolution with method='{method}', alpha={alpha}...")
        
        

    results = Parallel(n_jobs=config.n_jobs)(
        delayed(fit_single_bin)(spatial_expr.loc[ref_matrix_df.index, bin_id].values)
        for bin_id in tqdm(spatial_expr.columns, total=len(spatial_expr.columns), leave=True,position=0)
    )

    #all_coefs = [x[0] for x in results]
    #all_residuals = [x[1] for x in results]

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
        print("Adding results to table.obs of sdata object")
        table.obs.drop(columns=df.columns, inplace=True, errors='ignore')
        table.obs = pd.merge(table.obs, df, left_index=True, right_index=True)

    if verbose:
        print("Deconvolution completed.")
    #residuals_s = pd.Series(all_residuals, index=spatial_expr.columns, name='residual')
    #return (df,residuals_s)
    return df


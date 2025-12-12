from .easydecon import *

def easydecon_workflow(
    sdata,
    markers_df,
    marker_genes=None,                    # This can be a list of genes, You can only give markers_df
    # --- shared / data schema ---
    celltype: str = "group",              # column in markers_df holding cluster IDs
    gene_id_column: str = "names",        # column in markers_df holding gene names
    exclude_group_names: list[str] | None = None,
    bin_size: int = 8,                    # used by both phases and assignment
    # === Phase 1 (priors): common_markers_gene_expression_and_filter ===
    aggregation_method: str = "sum",      # {"sum","mean","median"} supported by your helper funcs
    filtering_algorithm: str = "permutation",  # {"permutation","quantile"}
    num_permutations: int = 5000,         # number of permutations
    parametric: bool = True,              # parametric or empirical quantile
    alpha: float = 0.01,                  # permutation cutoff level
    subsample_size: int = 25000,          # subsample size for permutation
    subsample_signal_quantile: float = 0.1,   #permutation param, between 0 and 1, if 0.1, 10% of the bins with the lowest and highest expression will be discarded
    permutation_gene_pool_fraction: float = 0.3, # top fraction of genes to be used for the null distribution
    n_subs: int = 5,                      # permutation: number of subsamples
    quantile: float = 0.7,                # used only if filtering_algorithm="quantile"
    phase1_output_stat: str = "expression",  # NEW: {"expression","minus_log10_p"}
    # === Phase 2 (evidence): get_clusters_by_similarity_on_tissue ===
    method: str = "wjaccard",             # {"wjaccard","cosine","spearman","euclidean","jaccard","overlap", ...}
    similarity_by_column: str = "logfoldchanges",  # 
    lambda_param: float = 0.25,           # lambda parameter wjaccard
    weight_column: str = "logfoldchanges",  # column in markers_df for weights etc.
    min_markers: int = 3,
    fallback_auc: float = 0.5,
    # === Evidence→likelihood mapping (lightweight, non-DL) ===
    evidence_to_likelihood: str = "softmax",  # {"row_normalize","softmax"}
    softmax_tau: float = 1.0,                 # softmax temperature
    epsilon: float = 1e-12,                   # numerical guard
    # === Bayesian combination weights ===
    prior_weight: float = 1.0,                # weight for phase 1 priors
    likelihood_weight: float = 1.0,           # weight for phase 2 likelihoods
    # === Optional presence gating by priors ===
    apply_prior_presence_mask: bool = False,  # if True, priors gate likelihoods
    prior_presence_threshold: float = 0.0,    # threshold on priors for presence mask
    # === Final assignment: assign_clusters_from_df ===
    results_column: str = "easydecon",
    assign_method: str = "max",           # {"max","hybrid","zmax"} per your implementation
    allow_multiple: bool = False,
    diagnostic=None,
    fold_change_threshold: float = 2.0,

):

    # -----------------------
    # Phase 1: Priors
    # -----------------------
    phase1_result = common_markers_gene_expression_and_filter(
        sdata=sdata,
        marker_genes=markers_df if marker_genes is None else marker_genes,
        celltype=celltype,
        gene_id_column=gene_id_column,
        exclude_group_names=exclude_group_names,
        bin_size=bin_size,
        aggregation_method=aggregation_method,
        add_to_obs=True if marker_genes is not None else False,
        filtering_algorithm=filtering_algorithm,
        num_permutations=num_permutations,
        alpha=alpha,
        subsample_size=subsample_size,
        subsample_signal_quantile=subsample_signal_quantile,
        permutation_gene_pool_fraction=permutation_gene_pool_fraction,
        n_subs=n_subs,
        quantile=quantile,
        parametric=parametric,
        output_stat=phase1_output_stat,
    )

    if not isinstance(phase1_result, pd.DataFrame):
        raise TypeError("Phase 1 result must be a pandas DataFrame (spots x clusters).")

    priors_df = phase1_result.copy()
    priors_df = priors_df.clip(lower=0)
    priors_row_sum = priors_df.sum(axis=1).replace(0, np.nan)
    priors_df = priors_df.div(priors_row_sum, axis=0).fillna(0)


    prior_row_sums = priors_df.sum(axis=1)
    informative_spots = prior_row_sums[prior_row_sums > 0].index
    uninformative_spots = prior_row_sums[prior_row_sums == 0].index

    try:
        table_name = f"square_{bin_size:03}um"
        table = sdata.tables[table_name]
    except (AttributeError, KeyError):
        try:
            table = sdata.tables["table"]
        except (AttributeError, KeyError):
            table = sdata

    mask_col = "easydecon_mask"

    # initialize all spots to 0 (skip)
    table.obs[mask_col] = 0

    # mark informative spots as 1 (process in Phase 2)
    table.obs.loc[
        table.obs.index.intersection(informative_spots),
        mask_col
    ] = 1

    # -----------------------
    # Phase 2: Evidence
    # -----------------------
    phase2_result = get_clusters_by_similarity_on_tissue(
        sdata=sdata,
        markers_df=markers_df,
        bin_size=bin_size,
        gene_id_column=gene_id_column,
        method=method,
        add_to_obs=False,
        #common_group_name="MarkerGroup" if isinstance(marker_genes,list) else None,
        common_group_name=mask_col,
        similarity_by_column=similarity_by_column,
        weight_column=weight_column,
        lambda_param=lambda_param,
        min_markers=min_markers,
        fallback_auc=fallback_auc
    )
    if not isinstance(phase2_result, pd.DataFrame):
        raise TypeError("Phase 2 result must be a pandas DataFrame (spots x clusters).")

    evidence_df = phase2_result.copy()
    if evidence_to_likelihood == "row_normalize":
        min_per_row = evidence_df.min(axis=1)
        needs_shift = (min_per_row < 0)
        if needs_shift.any():
            evidence_df = evidence_df.sub(min_per_row, axis=0)
        evidence_df = evidence_df.clip(lower=0)
        evidence_row_sum = evidence_df.sum(axis=1).replace(0, np.nan)
        likelihoods_df = evidence_df.div(evidence_row_sum, axis=0).fillna(0)

    elif evidence_to_likelihood == "softmax":
        x = evidence_df.to_numpy(dtype=float)
        row_max = np.nanmax(x, axis=1, keepdims=True)
        logits = (x - row_max) / max(softmax_tau, epsilon)
        np.exp(logits, out=logits)
        row_sum = np.sum(logits, axis=1, keepdims=True)
        row_sum[row_sum == 0] = np.nan
        likelihoods_np = logits / row_sum
        likelihoods_np = np.nan_to_num(likelihoods_np, nan=0.0)
        likelihoods_df = pd.DataFrame(likelihoods_np, index=evidence_df.index, columns=evidence_df.columns)

    else:
        raise ValueError("evidence_to_likelihood must be one of {'row_normalize','softmax'}.")

    # -----------------------
    # Posterior combination
    # -----------------------

    if not isinstance(marker_genes,list):
        common_clusters = priors_df.columns.intersection(likelihoods_df.columns)
        if len(common_clusters) == 0:
            raise ValueError("No overlapping cluster columns between Phase 1 and Phase 2 outputs.")
        priors_aligned = priors_df[common_clusters]
        likelihoods_aligned = likelihoods_df[common_clusters]

        common_spots = priors_aligned.index.intersection(likelihoods_aligned.index)
        if len(common_spots) == 0:
            raise ValueError("No overlapping spot/bin indices between Phase 1 and Phase 2 outputs.")
        priors_aligned = priors_aligned.loc[common_spots]
        likelihoods_aligned = likelihoods_aligned.loc[common_spots]

        # Optional: use priors as a presence/absence gate on BOTH priors and likelihoods
        if apply_prior_presence_mask:
            presence_mask = (priors_aligned > prior_presence_threshold).astype(float)
            priors_aligned = priors_aligned * presence_mask
            likelihoods_aligned = likelihoods_aligned * presence_mask

        # Guard against exact zeros before exponentiation,
        # but keep true zeros from masking as zeros:
        priors_safe = priors_aligned.replace(0, np.nan).clip(lower=epsilon).fillna(0)
        likelihoods_safe = likelihoods_aligned.replace(0, np.nan).clip(lower=epsilon).fillna(0)

        posterior_unnorm = (priors_safe ** prior_weight) * (likelihoods_safe ** likelihood_weight)

        row_sum = posterior_unnorm.sum(axis=1)
        zero_rows = (row_sum <= epsilon)
        if zero_rows.any():
            # keep them as zero (no assignment from the posterior)
            posterior_unnorm.loc[zero_rows] = 0.0

        posterior_row_sum = posterior_unnorm.sum(axis=1).replace(0, np.nan)
        posterior_df = posterior_unnorm.div(posterior_row_sum, axis=0).fillna(0)




    else:
        print("Regular workflow, phase 1 used to find most likely postions and phase 2 to assign labels")
        posterior_df = None

    # -----------------------
    # Final assignment
    # -----------------------

    assigned_labels = assign_clusters_from_df(
        sdata,
        df=posterior_df if posterior_df is not None and not isinstance(marker_genes,list) else phase2_result,
        bin_size=bin_size,
        results_column=results_column,
        method=assign_method,
        allow_multiple=allow_multiple,
        diagnostic=diagnostic,
        fold_change_threshold=fold_change_threshold
    )


    print("Finished!")
    print("Posterior df and proportions can be None if the required columns or input parameters missing...")
    return phase1_result, phase2_result, assigned_labels, priors_df, posterior_df if posterior_df is not None and not isinstance(marker_genes,list) else phase2_result



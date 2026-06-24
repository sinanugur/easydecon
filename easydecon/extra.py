from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._schema import get_table
from ._validation import (
    EVIDENCE_TO_LIKELIHOOD_METHODS,
    validate_choice,
    validate_positive,
)
from .easydecon import (
    _validate_finite_nonnegative,
    assign_clusters_from_df,
    common_markers_gene_expression_and_filter,
    get_clusters_by_similarity_on_tissue,
    read_markers_dataframe,
)
from .markers import PreparedMarkers


def _evidence_to_likelihood(
    evidence_df,
    method="softmax",
    softmax_tau=1.0,
) -> pd.DataFrame:
    """Transform Phase 2 evidence into likelihoods.

    This preserves the historical easydecon_workflow behavior for both
    ``row_normalize`` and ``softmax``.
    """
    evidence_df = evidence_df.copy()
    if method == "row_normalize":
        min_per_row = evidence_df.min(axis=1)
        needs_shift = min_per_row < 0
        if needs_shift.any():
            evidence_df = evidence_df.sub(min_per_row, axis=0)
        evidence_df = evidence_df.clip(lower=0)
        evidence_row_sum = evidence_df.sum(axis=1).replace(0, np.nan)
        return evidence_df.div(evidence_row_sum, axis=0).fillna(0)

    if method == "softmax":
        x = evidence_df.to_numpy(dtype=float)
        x = np.where(np.isfinite(x), x, -np.inf)
        row_max = np.max(x, axis=1, keepdims=True)
        valid_rows = np.isfinite(row_max[:, 0])
        logits = np.zeros_like(x, dtype=float)
        valid_logits = (x[valid_rows] - row_max[valid_rows]) / softmax_tau
        logits[valid_rows] = np.exp(valid_logits)
        row_sum = np.sum(logits, axis=1, keepdims=True)
        row_sum[row_sum == 0] = np.nan
        likelihoods_np = logits / row_sum
        likelihoods_np = np.nan_to_num(likelihoods_np, nan=0.0)
        return pd.DataFrame(
            likelihoods_np,
            index=evidence_df.index,
            columns=evidence_df.columns,
        )

    raise ValueError("evidence_to_likelihood must be 'row_normalize' or 'softmax'.")


@dataclass
class EasyDeconResult:
    markers_df: pd.DataFrame
    phase1_result: pd.DataFrame
    phase2_result: pd.DataFrame
    assigned_labels: pd.DataFrame
    priors_df: pd.DataFrame
    likelihoods_df: pd.DataFrame
    posterior_df: pd.DataFrame | None
    assignment_df: pd.DataFrame
    diagnostics: dict
    prepared_markers: PreparedMarkers | None = None


def easydecon_workflow(
    sdata,
    markers_df=None,
    prepared_markers=None,
    marker_genes=None,                    # This can be a list of genes, You can only give markers_df
    filename=None,
    adata=None,
    mask_col = "easydecon_mask",          # If markers_genes given, this column will be used to mask informative spots
    # --- shared / data schema ---
    celltype: str = "group",              # column in markers_df holding cluster IDs
    gene_id_column: str = "names",        # column in markers_df holding gene names
    exclude_group_names: list[str] | None = None,
    bin_size: int = 8,                    # used by both phases and assignment
    # === Marker loading / generation ===
    marker_method: str = "auto",
    groupby: str | None = None,
    sample_col: str | None = None,
    marker_key: str = "rank_genes_groups",
    top_n_genes: int = 60,
    sort_by_column: str = "scores",
    ascending: bool = False,
    log2fc_min: float = 0.25,
    pval_cutoff: float = 0.05,
    drop_ribosomal: bool = True,
    drop_mitochondrial: bool = True,
    table_key=None,
    preferred_table_keys=None,
    marker_source=None,
    scanpy_method: str = "wilcoxon",
    layer=None,
    use_raw=None,
    reference: str = "rest",
    copy_adata: bool = True,
    rank_genes_groups_kwargs=None,
    min_cells_per_group: int = 20,
    min_replicates_per_condition: int = 2,
    deseq_alpha: float = 0.05,
    deseq_n_cpus=None,
    deseq_quiet: bool = True,
    deseq_kwargs=None,
    deseq_stats_kwargs=None,
    verbose: bool = True,
    return_result_object: bool = False,
    return_diagnostics: bool = False,
    # === Phase 1 (priors): common_markers_gene_expression_and_filter ===
    aggregation_method: str = "sum",      # {"sum","mean","median"} supported by your helper funcs
    filtering_algorithm: str = "permutation",  # {"permutation","quantile"}
    num_permutations: int = 5000,         # number of permutations
    parametric: bool = True,              # parametric or empirical quantile
    alpha: float = 0.01,                  # permutation cutoff level
    subsample_size: int = 25000,          # subsample size for permutation
    subsample_signal_quantile: float = 0,   #permutation param, between 0 and 1, if 0.1, 10% of the bins with the lowest and highest expression will be discarded
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
    fallback_auc: float = 0.0,
    expression_threshold: float = 0.1,
    top_n_markers: int | None = None,
    recovery_power: float = 1.0,
    drop_shared_markers: bool = False,
    center_auc: bool = True,
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
    minimum_evidence: float = 0.0,
    tie_tolerance: float = 1e-12,

):
    validate_choice(
        evidence_to_likelihood,
        EVIDENCE_TO_LIKELIHOOD_METHODS,
        "evidence_to_likelihood",
    )
    validate_positive(softmax_tau, "softmax_tau")
    validate_positive(epsilon, "epsilon")
    _validate_finite_nonnegative(fallback_auc, "fallback_auc")
    _validate_finite_nonnegative(expression_threshold, "expression_threshold")
    _validate_finite_nonnegative(recovery_power, "recovery_power")
    _validate_finite_nonnegative(minimum_evidence, "minimum_evidence")
    _validate_finite_nonnegative(tie_tolerance, "tie_tolerance")
    if (
        top_n_markers is not None
        and (
            isinstance(top_n_markers, bool)
            or not isinstance(top_n_markers, int)
            or top_n_markers < 1
        )
    ):
        raise ValueError("top_n_markers must be None or an integer greater than or equal to 1.")
    if not isinstance(drop_shared_markers, bool):
        raise ValueError("drop_shared_markers must be a bool.")
    if not isinstance(center_auc, bool):
        raise ValueError("center_auc must be a bool.")
    if prior_weight < 0 or likelihood_weight < 0:
        raise ValueError("prior_weight and likelihood_weight must be non-negative.")

    original_celltype = celltype
    original_gene_id_column = gene_id_column
    markers_df, marker_diagnostics = read_markers_dataframe(
        sdata=sdata,
        filename=filename,
        adata=adata,
        markers_df=markers_df,
        prepared_markers=prepared_markers,
        exclude_celltype=None,
        bin_size=bin_size,
        top_n_genes=top_n_genes,
        sort_by_column=sort_by_column,
        ascending=ascending,
        gene_id_column=gene_id_column,
        celltype=celltype,
        key=marker_key,
        log2fc_min=log2fc_min,
        pval_cutoff=pval_cutoff,
        drop_ribosomal=drop_ribosomal,
        drop_mitochondrial=drop_mitochondrial,
        table_key=table_key,
        preferred_table_keys=preferred_table_keys,
        source=marker_source,
        return_diagnostics=True,
        verbose=verbose,
        marker_method=marker_method,
        groupby=groupby,
        scanpy_method=scanpy_method,
        layer=layer,
        use_raw=use_raw,
        reference=reference,
        copy_adata=copy_adata,
        rank_genes_groups_kwargs=rank_genes_groups_kwargs,
        sample_col=sample_col,
        min_cells_per_group=min_cells_per_group,
        min_replicates_per_condition=min_replicates_per_condition,
        deseq_alpha=deseq_alpha,
        deseq_n_cpus=deseq_n_cpus,
        deseq_quiet=deseq_quiet,
        deseq_kwargs=deseq_kwargs,
        deseq_stats_kwargs=deseq_stats_kwargs,
    )
    if not isinstance(markers_df, pd.DataFrame):
        raise ValueError("Resolved markers_df must be a pandas DataFrame.")
    missing_marker_columns = {"group", "names"}.difference(markers_df.columns)
    if missing_marker_columns:
        raise ValueError(
            "Resolved markers_df must contain canonical columns 'group' and "
            f"'names'. Missing: {sorted(missing_marker_columns)}."
        )
    if markers_df["group"].nunique() == 0:
        raise ValueError("Resolved markers_df contains no marker groups.")
    if markers_df["names"].nunique() == 0:
        raise ValueError("Resolved markers_df contains no marker genes.")

    celltype = "group"
    gene_id_column = "names"
    marker_genes_is_list = isinstance(marker_genes, list)
    phase1_markers = markers_df if marker_genes is None else marker_genes
    if isinstance(phase1_markers, pd.DataFrame):
        phase1_markers = phase1_markers.copy()
        rename_columns = {}
        if "group" not in phase1_markers and original_celltype in phase1_markers:
            rename_columns[original_celltype] = "group"
        if "names" not in phase1_markers and original_gene_id_column in phase1_markers:
            rename_columns[original_gene_id_column] = "names"
        phase1_markers.rename(columns=rename_columns, inplace=True)

    # -----------------------
    # Phase 1: Priors
    # -----------------------
    phase1_result = common_markers_gene_expression_and_filter(
        sdata=sdata,
        marker_genes=phase1_markers,
        celltype="group",
        gene_id_column="names",
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
        verbose=verbose,
    )

    if not isinstance(phase1_result, pd.DataFrame):
        raise TypeError("Phase 1 result must be a pandas DataFrame (spots x clusters).")

    priors_df = phase1_result.copy()
    priors_df = priors_df.clip(lower=0)
    priors_row_sum = priors_df.sum(axis=1).replace(0, np.nan)
    priors_df = priors_df.div(priors_row_sum, axis=0).fillna(0)


    prior_row_sums = priors_df.sum(axis=1)
    informative_spots = prior_row_sums[prior_row_sums > 0].index
    table = get_table(
        sdata,
        bin_size=bin_size,
        table_key=table_key,
        preferred_table_keys=preferred_table_keys,
    )

    

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
        gene_id_column="names",
        celltype="group",
        method=method,
        add_to_obs=False,
        #common_group_name="MarkerGroup" if isinstance(marker_genes,list) else None,
        common_group_name=mask_col,
        similarity_by_column=similarity_by_column,
        weight_column=weight_column,
        lambda_param=lambda_param,
        min_markers=min_markers,
        fallback_auc=fallback_auc,
        expression_threshold=expression_threshold,
        top_n_markers=top_n_markers,
        recovery_power=recovery_power,
        drop_shared_markers=drop_shared_markers,
        center_auc=center_auc,
        verbose=verbose,
    )
    if not isinstance(phase2_result, pd.DataFrame):
        raise TypeError("Phase 2 result must be a pandas DataFrame (spots x clusters).")

    likelihoods_df = _evidence_to_likelihood(
        phase2_result,
        method=evidence_to_likelihood,
        softmax_tau=softmax_tau,
    )


    # -----------------------
    # Posterior combination
    # -----------------------

    if not marker_genes_is_list:
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
        posterior_df = None

    # -----------------------
    # Final assignment
    # -----------------------

    assignment_df = posterior_df if posterior_df is not None and not marker_genes_is_list else phase2_result
    assigned_labels = assign_clusters_from_df(
        sdata,
        df=assignment_df,
        bin_size=bin_size,
        results_column=results_column,
        method=assign_method,
        allow_multiple=allow_multiple,
        diagnostic=diagnostic,
        fold_change_threshold=fold_change_threshold,
        minimum_evidence=minimum_evidence,
        tie_tolerance=tie_tolerance,
        verbose=verbose,
    )


    diagnostics = {
        "markers": marker_diagnostics,
        "n_phase1_spots": int(phase1_result.shape[0]),
        "n_phase1_celltypes": int(phase1_result.shape[1]),
        "n_phase2_spots": int(phase2_result.shape[0]),
        "n_phase2_celltypes": int(phase2_result.shape[1]),
        "posterior_available": posterior_df is not None,
        "assignment_matrix": (
            "posterior_df" if posterior_df is not None else "phase2_result"
        ),
        "results_column": results_column,
        "mask_col": mask_col,
        "phase2": {
            "method": method,
            "min_markers": min_markers,
            "fallback_auc": fallback_auc,
            "expression_threshold": expression_threshold,
            "top_n_markers": top_n_markers,
            "recovery_power": recovery_power,
            "drop_shared_markers": drop_shared_markers,
            "center_auc": center_auc,
        },
        "assignment": {
            "method": assign_method,
            "minimum_evidence": minimum_evidence,
            "tie_tolerance": tie_tolerance,
            "allow_multiple": allow_multiple,
            "fold_change_threshold": fold_change_threshold,
        },
    }

    if verbose:
        print("Finished easydecon workflow.")
        if posterior_df is None:
            print(
                "Posterior df is None because marker_genes was provided as a "
                "list-style mask workflow."
            )

    if return_result_object:
        return EasyDeconResult(
            markers_df=markers_df,
            phase1_result=phase1_result,
            phase2_result=phase2_result,
            assigned_labels=assigned_labels,
            priors_df=priors_df,
            likelihoods_df=likelihoods_df,
            posterior_df=posterior_df,
            assignment_df=assignment_df,
            diagnostics=diagnostics,
            prepared_markers=prepared_markers,
        )
    result_tuple = (
        phase1_result,
        phase2_result,
        assigned_labels,
        priors_df,
        assignment_df,
    )
    if return_diagnostics:
        return (*result_tuple, diagnostics)
    return result_tuple

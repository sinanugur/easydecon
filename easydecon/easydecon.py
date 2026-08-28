import warnings
from dataclasses import dataclass
from numbers import Real
warnings.simplefilter(action='ignore', category=FutureWarning)
import scanpy as sc
import numpy as np
try:
    import fireducks.pandas as pd
    #print("Using fireducks.pandas for enhanced functionality.")
except ImportError:
    import pandas as pd
    #print("fireducks.pandas not found. Falling back to standard pandas.")

try:
    import spatialdata as sd
    import spatialdata_io
    from spatialdata import get_extent
    from spatialdata import bounding_box_query
    from spatialdata import match_element_to_table
    import spatialdata_plot
except ImportError:
    warnings.warn(
        "SpatialData not found. SpatialData-specific plotting/query helpers may "
        "be unavailable.",
        ImportWarning,
        stacklevel=2,
    )

from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from scipy.stats import gamma
from scipy.stats import zscore
from tqdm.auto import tqdm
from scipy.sparse import issparse

import logging


def _suppress_warnings_in_worker():
    """ Suppress warnings and logging inside joblib workers. """
    logging.getLogger().setLevel(logging.CRITICAL)
    warnings.simplefilter("ignore")

def process_row_with_suppression(row, func, **kwargs):
    """ Wrapper around `process_row` to suppress warnings in each worker. """
    _suppress_warnings_in_worker()  # Suppress inside the worker
    return process_row(row, func, **kwargs)

from .config import config
from ._schema import (
    MarkerSchema,
    get_table,
    normalize_marker_roles,
    resolve_marker_columns,
    standardize_marker_dataframe,
)
from ._validation import (
    AGGREGATION_METHODS,
    ASSIGN_METHODS,
    FILTERING_ALGORITHMS,
    MARKER_METHODS,
    MARKER_ROLE_INFERENCE_MODES,
    MARKER_ROLE_MODES,
    PHASE1_OUTPUT_STATS,
    PYDESEQ2_MARKER_METHODS,
    REFERENCE_MARKER_METHODS,
    SIMILARITY_METHODS,
    UCELL_MARKER_ROLES,
    format_allowed_values,
    validate_choice,
    validate_probability_range,
)

from joblib import Parallel, delayed


# Ensure that the progress_apply method is available
#tqdm.pandas()
logger = logging.getLogger(__name__)

def sparse_var(matrix, axis=0):
    if issparse(matrix):  # Sparse matrix
        mean_sq = matrix.mean(axis=axis).A1 ** 2
        sq_mean = matrix.multiply(matrix).mean(axis=axis).A1
    else:  # Dense numpy array
        mean_sq = np.mean(matrix, axis=axis) ** 2
        sq_mean = np.mean(matrix * matrix, axis=axis)
    return sq_mean - mean_sq


def _aggregate_marker_expression(
    expr_df,
    method,
    coverage_power=0.5,
):
    """Aggregate marker expression by row without Python-level row callbacks."""
    if method == "sum":
        return expr_df.sum(axis=1)
    if method == "mean":
        return expr_df.mean(axis=1)
    if method == "median":
        return expr_df.median(axis=1)
    if method == "coverage":
        n_markers = expr_df.shape[1]
        if n_markers == 0:
            return pd.Series(0.0, index=expr_df.index, dtype=float)

        values = expr_df.to_numpy()
        positive_mask = values > 0
        detected = positive_mask.sum(axis=1)
        positive_sums = np.where(positive_mask, values, 0.0).sum(axis=1)
        scores = np.zeros(values.shape[0], dtype=float)
        valid = detected > 0

        if coverage_power == 0.5:
            scores[valid] = positive_sums[valid] / np.sqrt(
                detected[valid] * n_markers
            )
        else:
            scores[valid] = (
                positive_sums[valid] / detected[valid]
            ) * (detected[valid] / n_markers) ** coverage_power

        return pd.Series(scores, index=expr_df.index)

    raise ValueError(f"Unsupported aggregation method: {method!r}.")


def _validate_permutation_gene_pool_fraction(value):
    """Validate and normalize the Phase 1 permutation-pool setting."""
    if value == "auto":
        return value
    if isinstance(value, str) or isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            "permutation_gene_pool_fraction must be 'auto' or a numeric "
            "value in (0, 1]."
        )
    if not np.isfinite(value) or value <= 0 or value > 1:
        raise ValueError(
            "permutation_gene_pool_fraction must be 'auto' or a numeric "
            "value in (0, 1]."
        )
    return float(value)


def _resolve_permutation_gene_pool_size(
    n_eligible_genes,
    marker_set_size,
    permutation_gene_pool_fraction,
    *,
    pool_multiplier=30,
    min_pool_size=1000,
    min_fraction=0.20,
    max_fraction=0.60,
):
    """Resolve a variance-ranked null-pool size and its effective fraction."""
    setting = _validate_permutation_gene_pool_fraction(
        permutation_gene_pool_fraction
    )
    n_eligible_genes = int(n_eligible_genes)
    marker_set_size = int(marker_set_size)
    if n_eligible_genes < 0:
        raise ValueError("n_eligible_genes must be greater than or equal to 0.")
    if marker_set_size < 0:
        raise ValueError("marker_set_size must be greater than or equal to 0.")

    if n_eligible_genes == 0:
        return {
            "mode": "auto" if setting == "auto" else "numeric",
            "pool_size": 0,
            "effective_fraction": 0.0,
        }

    if setting == "auto":
        min_size_from_fraction = int(np.ceil(min_fraction * n_eligible_genes))
        max_size_from_fraction = int(np.ceil(max_fraction * n_eligible_genes))
        pool_size = max(
            int(min_pool_size),
            int(pool_multiplier * marker_set_size),
            min_size_from_fraction,
        )
        pool_size = min(
            pool_size,
            max_size_from_fraction,
            n_eligible_genes,
        )
        mode = "auto"
    else:
        pool_size = max(1, int(setting * n_eligible_genes))
        pool_size = min(pool_size, n_eligible_genes)
        mode = "numeric"

    # Avoid replacement merely because the requested fraction was smaller than
    # the marker set. Replacement remains the fallback when the whole eligible
    # universe is smaller than the marker set.
    if n_eligible_genes >= marker_set_size:
        pool_size = max(pool_size, marker_set_size)
    pool_size = min(pool_size, n_eligible_genes)

    return {
        "mode": mode,
        "pool_size": int(pool_size),
        "effective_fraction": float(pool_size / n_eligible_genes),
    }


def _get_detected_gene_mask(matrix):
    """Return a sparse-safe mask for genes detected in at least one location."""
    if issparse(matrix):
        detection_counts = np.asarray(matrix.getnnz(axis=0)).ravel()
    else:
        detection_counts = np.count_nonzero(np.asarray(matrix), axis=0)
    return np.asarray(detection_counts).ravel() > 0


def _build_permutation_gene_pool(
    var_names,
    gene_variability,
    detected_gene_mask,
    marker_genes,
    permutation_gene_pool_fraction,
    *,
    group_name=None,
):
    """Build one group's detected, finite-variance, marker-excluded pool."""
    var_names = pd.Index(var_names)
    variability = np.asarray(gene_variability, dtype=float).ravel()
    detected_gene_mask = np.asarray(detected_gene_mask, dtype=bool).ravel()
    if len(var_names) != len(variability) or len(var_names) != len(detected_gene_mask):
        raise ValueError(
            "var_names, gene_variability, and detected_gene_mask must have "
            "the same length."
        )

    finite_variance_mask = np.isfinite(variability)
    target_marker_mask = np.asarray(var_names.isin(marker_genes), dtype=bool)
    eligible_mask = detected_gene_mask & finite_variance_mask & ~target_marker_mask
    eligible_positions = np.flatnonzero(eligible_mask)
    n_eligible_genes = int(len(eligible_positions))
    if n_eligible_genes == 0:
        group_text = f" for group {group_name!r}" if group_name is not None else ""
        raise ValueError(
            "No usable null genes remain"
            f"{group_text} after excluding undetected genes, non-finite "
            "variance genes, and the group's own marker genes."
        )

    pool_info = _resolve_permutation_gene_pool_size(
        n_eligible_genes,
        len(marker_genes),
        permutation_gene_pool_fraction,
    )
    eligible_variability = variability[eligible_positions]
    variance_order = np.argsort(eligible_variability, kind="stable")
    selected_positions = eligible_positions[
        variance_order[-pool_info["pool_size"] :]
    ]
    gene_pool = var_names.to_numpy()[selected_positions]
    pool_info.update(
        {
            "n_detected_genes": int(detected_gene_mask.sum()),
            "n_finite_variance_genes": int(finite_variance_mask.sum()),
            "n_eligible_null_genes": n_eligible_genes,
        }
    )
    return gene_pool, pool_info


def _sample_permutation_genes(rng, gene_pool, marker_set_size):
    """Draw one null marker set while advancing the supplied RNG."""
    return rng.choice(
        gene_pool,
        size=marker_set_size,
        replace=len(gene_pool) < marker_set_size,
    )


def common_markers_gene_expression_and_filter(
    sdata: object,
    marker_genes,  # can be list, dict, or DataFrame
    common_group_name: str = "MarkerGroup",  # will create this column in the spatial data table, used if marker_genes is a list
    celltype: str = "group",               # DF column holding group IDs
    gene_id_column: str = "names",                # DF column holding marker gene names
    exclude_group_names: list[str] | None = None,
    bin_size: int = 8,
    aggregation_method: str = "sum",
    add_to_obs: bool = True,
    filtering_algorithm: str = "permutation",  # or "quantile"
    num_permutations: int = 5000, # permutation param total number of permutations
    alpha: float = 0.01, #significance level for the permutation-based cutoff
    subsample_size: int = 25000, #permutation param
    subsample_signal_quantile: float = 0.1, #permutation param, between 0 and 1, if 0.1, 10% of the bins with the lowest and highest expression will be discarded
    permutation_gene_pool_fraction: float | str = "auto",
    parametric: bool = True, #if parametric, gamma or exponential distribution is used
    n_subs: int = 5,                 # number of subsamples
    quantile: float = 0.7, #if quantile selected,
    output_stat: str = "expression",  # NEW: {"expression", "minus_log10_p"}
    verbose: bool = True,
    random_state: int | None = 10,
    coverage_power: float = 0.5,
    _diagnostics_out=None,
    **kwargs
) -> pd.DataFrame:
    """Compute Phase 1 marker-expression evidence.

    ``marker_genes`` may be a list, a mapping from group to genes, or a marker
    DataFrame. Expression is aggregated for each marker group and filtered with
    the selected algorithm.

    Parameters
    ----------
    sdata
        SpatialData-like container or AnnData-like table.
    marker_genes
        List of genes, mapping from group to genes, or marker DataFrame.
    common_group_name
        Group name used when ``marker_genes`` is a list.
    celltype, gene_id_column
        Marker DataFrame columns for group and gene names.
    exclude_group_names
        Groups whose nonzero rows should be excluded from Phase 1 evidence.
    bin_size
        Bin size used when resolving a SpatialData table.
    aggregation_method
        One of ``"sum"``, ``"mean"``, ``"median"``, or ``"coverage"``.
        Coverage aggregation is the mean expression among detected markers
        multiplied by the detected-marker fraction raised to
        ``coverage_power``.
    coverage_power
        Controls how strongly incomplete marker recovery is penalized for
        coverage aggregation. ``0`` ignores coverage, ``1`` gives the ordinary
        mean for non-negative expression, and the default ``0.5`` applies a
        square-root coverage penalty.
    add_to_obs
        Whether to merge Phase 1 columns into ``table.obs``.
    filtering_algorithm
        One of ``"permutation"``, ``"quantile"``, or ``"nb"``.
    permutation_gene_pool_fraction
        Controls the high-variance background pool used by permutation null
        sampling. A numeric value in ``(0, 1]`` selects that fraction of
        eligible genes. ``"auto"`` chooses a variance-ranked pool size from
        the marker-set size and available spatial gene universe. Undetected
        genes and the current group's own markers are excluded.
    random_state
        Seed for reproducible Phase 1 spot and null-gene sampling. ``None``
        requests non-deterministic sampling.
    output_stat
        ``"expression"`` or ``"minus_log10_p"``. The latter is invalid with
        quantile filtering.
    **kwargs
        Additional method-specific options, including ``nb_global_theta`` for
        NB filtering.

    Returns
    -------
    pd.DataFrame
        Thresholded Phase 1 evidence with spatial locations as rows and marker
        groups as columns.
    """
    validate_choice(
        filtering_algorithm, FILTERING_ALGORITHMS, "filtering_algorithm"
    )
    validate_choice(aggregation_method, AGGREGATION_METHODS, "aggregation_method")
    validate_choice(output_stat, PHASE1_OUTPUT_STATS, "output_stat")
    coverage_power = validate_probability_range(
        coverage_power, "coverage_power"
    )
    if filtering_algorithm == "quantile" and output_stat == "minus_log10_p":
        raise ValueError("output_stat='minus_log10_p' requires filtering_algorithm='permutation' or 'nb'.")
    validate_probability_range(
        alpha, "alpha", inclusive_min=False, inclusive_max=False
    )
    validate_probability_range(quantile, "quantile")
    validate_probability_range(
        subsample_signal_quantile,
        "subsample_signal_quantile",
        inclusive_min=True,
        inclusive_max=False,
        _maximum=0.5,
    )
    permutation_gene_pool_fraction = _validate_permutation_gene_pool_fraction(
        permutation_gene_pool_fraction
    )
    if n_subs < 1:
        raise ValueError("n_subs must be at least 1.")

    exclude_group_names = exclude_group_names or []
    table = get_table(sdata, bin_size=bin_size)

    phase1_diagnostics = {
        "filtering_algorithm": filtering_algorithm,
        "aggregation_method": aggregation_method,
        "coverage_power": float(coverage_power),
        "random_state": random_state,
        "groups": {},
    }
    if _diagnostics_out is not None:
        if not isinstance(_diagnostics_out, dict):
            raise TypeError("_diagnostics_out must be a dict when provided.")
        _diagnostics_out.update(phase1_diagnostics)

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
        try:
            marker_genes_tmp = standardize_marker_dataframe(
                marker_genes,
                schema=MarkerSchema(
                    group_col=celltype,
                    gene_col=gene_id_column,
                ),
                gene_universe=table.var_names,
                top_n_genes=None,
                sort_by_column=None,
                log2fc_min=-np.inf,
                pval_cutoff=1.0,
                source=None,
            )
        except ValueError as exc:
            raise ValueError(
                "Could not resolve group and gene columns from marker_genes "
                "DataFrame."
            ) from exc
        group_dict = {
            group_name: sub_df["names"].drop_duplicates().tolist()
            for group_name, sub_df in marker_genes_tmp.groupby(
                marker_genes_tmp["group"], sort=False
            )
        }
    else:
        raise TypeError(
            "marker_genes must be a list, dict, or DataFrame with the appropriate columns."
        )

    gene_variability = None
    detected_gene_mask = None
    rng = None
    if filtering_algorithm == "permutation":
        gene_variability = sparse_var(table.X, axis=0)
        detected_gene_mask = _get_detected_gene_mask(table.X)
        rng = np.random.default_rng(random_state)

    # 2) Exclude spots
    spots_to_be_used = table.obs.index
    if exclude_group_names:
        for g in exclude_group_names:
            if g in table.obs.columns:
                spots_excluded = table.obs[table.obs[g] != 0].index
                spots_to_be_used = spots_to_be_used.difference(spots_excluded)

    #patch for Xenium, may be deleted
    if isinstance(spots_to_be_used, (list, tuple, set)):
        spots_to_be_used = pd.Index(spots_to_be_used)
    elif isinstance(spots_to_be_used, np.ndarray):
        spots_to_be_used = pd.Index(spots_to_be_used)
    # If it's a boolean array/Series, use it directly as a mask
    if getattr(spots_to_be_used, "dtype", None) is not None and spots_to_be_used.dtype == bool:
        obs_mask = np.asarray(spots_to_be_used, dtype=bool)
    else:
        # treat as labels; if you actually meant positions, change to `.iloc` below
        obs_mask = table.obs_names.isin(spots_to_be_used)

    

    # Prepare a final DataFrame to collect group results
    result_df = pd.DataFrame(index=spots_to_be_used)

    tqdm.pandas()
    # -----------------------------------------------------------
    # Loop over each group in the dictionary
    # -----------------------------------------------------------
    for group_name, gene_list in group_dict.items():
        # Intersect gene_list with table.var_names
        requested_genes = set(gene_list)
        filtered_genes = [
            gene for gene in table.var_names if gene in requested_genes
        ]
        group_diagnostics = {
            "marker_set_size": int(len(filtered_genes)),
            "n_total_genes": int(len(table.var_names)),
        }
        phase1_diagnostics["groups"][group_name] = group_diagnostics
        if not filtered_genes:
            if verbose:
                print(f"Warning: No valid marker genes found for group '{group_name}'.")
            # We'll create a column of all zeros
            result_df[group_name] = 0
            group_diagnostics.update(
                {
                    "status": "no_valid_markers",
                    "positive_spot_count": 0,
                    "positive_spot_fraction": 0.0,
                }
            )
            if detected_gene_mask is not None:
                group_diagnostics["n_detected_genes"] = int(
                    detected_gene_mask.sum()
                )
            continue
        shape_hat = loc_hat = scale_hat = None
        nonzero_null_vals = None
        var_mask = table.var_names.isin(filtered_genes)
        # Retrieve expression for the selected spots & genes
        #expr_matrix = table[spots_to_be_used, filtered_genes].to_df()
        expr_matrix = table[obs_mask, var_mask].to_df()

        aggregated_vals = _aggregate_marker_expression(
            expr_matrix,
            aggregation_method,
            coverage_power=coverage_power,
        )

        group_expression = aggregated_vals.to_frame(name=group_name)

        # Apply the chosen filtering algorithm
        if filtering_algorithm == "quantile":
            # Threshold from non-zero aggregator values
            non_zero_vals = group_expression[group_expression[group_name] != 0][group_name]
            threshold = non_zero_vals.quantile(quantile)

        elif filtering_algorithm == "permutation":
            marker_set_size = len(filtered_genes)
            gene_pool, pool_info = _build_permutation_gene_pool(
                table.var_names,
                gene_variability,
                detected_gene_mask,
                filtered_genes,
                permutation_gene_pool_fraction,
                group_name=group_name,
            )
            group_diagnostics.update(
                {
                    "n_detected_genes": pool_info["n_detected_genes"],
                    "n_eligible_null_genes": pool_info[
                        "n_eligible_null_genes"
                    ],
                    "gene_pool_mode": pool_info["mode"],
                    "requested_gene_pool_fraction": permutation_gene_pool_fraction,
                    "effective_gene_pool_fraction": pool_info[
                        "effective_fraction"
                    ],
                    "gene_pool_size": pool_info["pool_size"],
                    "num_permutations": int(num_permutations),
                    "n_subs": int(n_subs),
                    "sampling_with_replacement": bool(
                        len(gene_pool) < marker_set_size
                    ),
                    "parametric": bool(parametric),
                }
            )

            marker_expr = _aggregate_marker_expression(
                table[:, filtered_genes].to_df(),
                aggregation_method,
                coverage_power=coverage_power,
            )
            signal_low, signal_high = marker_expr.quantile([subsample_signal_quantile, 1-subsample_signal_quantile])
            candidate_spots = marker_expr[(marker_expr >= signal_low) & (marker_expr <= signal_high)].index
            group_diagnostics["candidate_spot_count"] = int(len(candidate_spots))

            if len(candidate_spots) == 0:
                # Edge case: if everything was below cutoff
                if verbose:
                    print(f"Warning: no bins passed the total_counts quantile filter for {group_name}.")
                threshold = 0
                group_diagnostics.update(
                    {
                        "status": "no_candidate_spots",
                        "null_score_count": 0,
                        "nonzero_null_score_count": 0,
                    }
                )
            else:
                all_null_scores = []

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
                        subset_spots = rng.choice(
                            candidate_spots,
                            size=current_subset_size,
                            replace=False,
                        )
                    else:
                        subset_spots = candidate_spots

                    # Build null distribution for this subset
                    all_expr_df = table[subset_spots, :].to_df()
                    for _ in tqdm(
                        range(int(num_permutations/n_subs)),
                        #desc=f"Perm sub {i+1}/{n_subs} of {current_subset_size} for {group_name}",
                        desc=f"Subsample {current_subset_size*(i+1)}/{subsample_size} for {group_name}",
                        leave=True,
                        position=0,
                        disable=not verbose,
                    ):
                        random_genes = _sample_permutation_genes(
                            rng,
                            gene_pool,
                            marker_set_size,
                        )

                        random_vals = _aggregate_marker_expression(
                            all_expr_df[random_genes],
                            aggregation_method,
                            coverage_power=coverage_power,
                        )
                        all_null_scores.append(random_vals.values)
                # Concatenate results
                null_scores_concat = (
                    np.concatenate(all_null_scores)
                    if all_null_scores
                    else np.asarray([], dtype=float)
                )
                finite_null_vals = null_scores_concat[
                    np.isfinite(null_scores_concat)
                ]
                nonzero_null_vals = finite_null_vals[finite_null_vals > 0]
                group_diagnostics.update(
                    {
                        "null_score_count": int(len(null_scores_concat)),
                        "nonzero_null_score_count": int(len(nonzero_null_vals)),
                        "null_median": (
                            float(np.median(finite_null_vals))
                            if len(finite_null_vals)
                            else None
                        ),
                        "null_q95": (
                            float(np.quantile(finite_null_vals, 0.95))
                            if len(finite_null_vals)
                            else None
                        ),
                        "null_q99": (
                            float(np.quantile(finite_null_vals, 0.99))
                            if len(finite_null_vals)
                            else None
                        ),
                    }
                )
                if len(nonzero_null_vals) == 0:
                    if verbose:
                        print("Warning: no positive values in null distribution, threshold set to 0.")
                    threshold = 0
                    group_diagnostics["status"] = "no_positive_null_scores"
                else:
                    if not parametric:
                        threshold = np.quantile(nonzero_null_vals, 1 - alpha)
                    else:
                        shape_hat, loc_hat, scale_hat = gamma.fit(nonzero_null_vals,floc=0)
                        threshold = gamma.ppf(1 - alpha, shape_hat, loc=loc_hat, scale=scale_hat)
                        group_diagnostics.update(
                            {
                                "gamma_shape": float(shape_hat),
                                "gamma_scale": float(scale_hat),
                            }
                        )
                    group_diagnostics["status"] = "ok"
            group_diagnostics["threshold"] = float(threshold)

        elif filtering_algorithm == "nb":
            if "counts" not in table.layers:
                raise ValueError("NB filtering requires raw counts in table.layers['counts'].")

            if aggregation_method != "sum":
                raise ValueError("NB filtering currently supports aggregation_method='sum' only.")

            # Observed marker-sum per bin (raw counts)
            Xg = table[obs_mask, var_mask].layers["counts"]     # sparse is fine
            S = np.asarray(Xg.sum(axis=1)).ravel().astype(float)

            # Library size per bin (raw counts)
            Xall = table[obs_mask, :].layers["counts"]
            total = np.asarray(Xall.sum(axis=1)).ravel().astype(float)
            median_total = np.median(total) + 1e-8
            sf = total / median_total

            # Baseline composition for these marker genes (q_g)
            # Use all bins in obs_mask for stability
            gene_sum = np.asarray(Xg.sum(axis=0)).ravel().astype(float)
            total_sum = total.sum() + 1e-8
            q = gene_sum / total_sum  # length = n_marker_genes

            # Expected mu_bg and variance under global NB dispersion
            theta = kwargs.get("nb_global_theta", 50.0)  # minimal: read from kwargs if provided
            mu = (sf[:, None] * q[None, :] * median_total)
            mu = np.maximum(mu, 1e-8)
            var = mu + (mu ** 2) / float(theta)

            ES = mu.sum(axis=1)
            VS = var.sum(axis=1) + 1e-8

            from scipy.stats import norm
            Z = (S - ES) / np.sqrt(VS)
            pvals = norm.sf(Z)  # right-tail

            # Fill group_expression in the same shape/index you already use
            group_expression = pd.DataFrame({group_name: S}, index=table.obs_names[obs_mask])

            # Then reuse your existing output_stat block, but with:
            # - threshold as bin-specific (expression mode)
            # - pvals from NB (minus_log10_p mode)

            if output_stat == "expression":
                zcrit = norm.isf(alpha)
                thresh = ES + zcrit * np.sqrt(VS)
                kept = np.where(S >= thresh, S, 0.0)
                result_df[group_name] = pd.Series(kept, index=group_expression.index).fillna(0.0)

            elif output_stat == "minus_log10_p":
                pclip = np.clip(pvals, 1e-300, 1.0)
                score = -np.log10(pclip)
                score[pvals > alpha] = 0.0
                result_df[group_name] = pd.Series(score, index=group_expression.index).fillna(0.0)

            else:
                raise ValueError(f"Unsupported output_stat: {output_stat}")

        
        else:
            raise RuntimeError("Unexpected filtering_algorithm after validation.")


        if filtering_algorithm != "nb":
            # --- NEW: choose what to output based on output_stat ---
            if output_stat == "expression":
                # Original behavior: threshold on expression, keep values above threshold
                group_expression[group_name] = np.where(
                    group_expression[group_name] >= threshold,
                    group_expression[group_name],
                    0,
                )
                result_df[group_name] = group_expression[group_name].fillna(0)

            elif output_stat == "minus_log10_p":
                # Compute p-values relative to the null distribution and output -log10(p)
                vals = group_expression[group_name].values

                # p-values: P(null >= observed)
                if (
                    filtering_algorithm == "permutation"
                    and (nonzero_null_vals is None or len(nonzero_null_vals) == 0)
                ):
                    result_df[group_name] = 0.0
                    pvals = None
                elif parametric:
                    pvals = gamma.sf(vals, shape_hat, loc=loc_hat, scale=scale_hat)
                else:
                    # empirical right-tail p-value
                    sorted_null = np.sort(nonzero_null_vals)
                    M = len(sorted_null)
                    # for each v: position of v in null; P(null >= v) = (M - idx) / M
                    idx = np.searchsorted(sorted_null, vals, side="left")
                    pvals = (M - idx) / float(M)

                if pvals is not None:
                    # Clip to avoid log(0)
                    pvals_clipped = np.clip(pvals, 1e-300, 1.0)
                    minus_log10_p = -np.log10(pvals_clipped)

                    # Zero out non-significant entries (p > alpha)
                    minus_log10_p[pvals > alpha] = 0.0

                    result_df[group_name] = pd.Series(
                        minus_log10_p,
                        index=group_expression.index,
                    ).fillna(0.0)

            else:
                raise ValueError(f"Unsupported output_stat: {output_stat}")

        if filtering_algorithm == "quantile":
            group_diagnostics.update(
                {
                    "status": "ok",
                    "threshold": float(threshold),
                }
            )
        positive_spot_count = int((result_df[group_name] > 0).sum())
        group_diagnostics.update(
            {
                "positive_spot_count": positive_spot_count,
                "positive_spot_fraction": (
                    float(positive_spot_count / len(result_df))
                    if len(result_df)
                    else 0.0
                ),
            }
        )

    # -----------------------------------------------------------
    # Merge results back into obs if requested
    # -----------------------------------------------------------
    if add_to_obs:
        if verbose:
            print("Adding results to table.obs of sdata object")
        # Drop existing columns of same names if present
        for col in result_df.columns:
            if col in table.obs.columns:
                table.obs.drop(columns=[col], inplace=True, errors='ignore')
        # Merge
        table.obs = pd.merge(table.obs, result_df, left_index=True, right_index=True, how='left')
        for col in result_df.columns:
            table.obs[col] = table.obs[col].fillna(0)

    return result_df

def get_clusters_by_similarity_on_tissue(
    sdata,
    markers_df,
    common_group_name=None,
    bin_size=8,
    gene_id_column="names",
    celltype="group",
    #similarity_by_column="logfoldchanges",
    method="wjaccard",
    add_to_obs=False,
    verbose=True,
    _diagnostics_out=None,
    _candidate_mask=None,
    **kwargs,
):
    """Compute Phase 2 marker-profile evidence with a chosen method.

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
    celltype : str, optional
        Column in `markers_df` containing cluster/cell type labels when the
        DataFrame index is not already grouped by cell type.
    similarity_by_column : str, optional
        Column in `markers_df` used to measure similarity or weight. 
        Default is "logfoldchanges".
    method : str, optional
        One of ``"correlation"``, ``"cosine"``, ``"jaccard"``,
        ``"overlap"``, ``"wjaccard"``, ``"diagnostic"``, ``"sum"``,
        ``"mean"``, ``"median"``, ``"euclidean"``, ``"auc"``, or
        ``"ucell"``. Default is ``"wjaccard"``.
    add_to_obs : bool, optional
        If True, adds the resulting assignment columns to `table.obs`. 
        Default is False.
    **method_kwargs : 
        Additional, method-specific parameters. For example:
        - For method="wjaccard": supply ``lambda_param``, etc.


    Returns
    -------
    pd.DataFrame
        A DataFrame whose index matches `table.obs.index` with cluster 
        assignment columns (or other metrics) computed by the specified method.
    """
    validate_choice(method, SIMILARITY_METHODS, "method")
    if not isinstance(markers_df, pd.DataFrame):
        raise TypeError("markers_df must be a pandas DataFrame.")
    table = get_table(sdata, bin_size=bin_size)
    markers_df = standardize_marker_dataframe(
        markers_df,
        schema=MarkerSchema(group_col=celltype, gene_col=gene_id_column),
        gene_universe=None,
        top_n_genes=None,
        sort_by_column=None,
        log2fc_min=-np.inf,
        pval_cutoff=1.0,
        source=None,
    )
    celltype = "group"
    gene_id_column = "names"
    marker_groups = markers_df["group"].drop_duplicates().tolist()
    cache_kwargs = dict(kwargs)
    cache_similarity_by_column = cache_kwargs.pop(
        "similarity_by_column", "logfoldchanges"
    )
    cache_weight_column = cache_kwargs.pop("weight_column", "logfoldchanges")
    phase2_cache = _build_phase2_cache(
        markers_df,
        spatial_gene_names=table.var_names,
        method=method,
        gene_id_column=gene_id_column,
        similarity_by_column=cache_similarity_by_column,
        weight_column=cache_weight_column,
        **cache_kwargs,
    )
    phase2_cache.diagnostics["sparse_input"] = bool(issparse(table.X))
    aligned_candidate_mask = None
    if _candidate_mask is not None:
        if not isinstance(_candidate_mask, pd.DataFrame):
            raise TypeError("_candidate_mask must be a pandas DataFrame.")
        aligned_candidate_mask = (
            _candidate_mask.reindex(
                index=table.obs.index,
                columns=phase2_cache.groups,
                fill_value=False,
            )
            .fillna(False)
            .astype(bool)
        )
        candidate_counts = aligned_candidate_mask.sum(axis=1).astype(int)
        active_counts = candidate_counts[candidate_counts > 0]
        total_pairs = int(aligned_candidate_mask.shape[0] * aligned_candidate_mask.shape[1])
        n_candidate_pairs = int(candidate_counts.sum())
        phase2_cache.diagnostics.update(
            {
                "candidate_pruning_enabled": True,
                "n_total_location_group_pairs": total_pairs,
                "n_candidate_pairs": n_candidate_pairs,
                "candidate_fraction": (
                    float(n_candidate_pairs / total_pairs) if total_pairs else 0.0
                ),
                "n_rows_with_candidates": int((candidate_counts > 0).sum()),
                "n_rows_without_candidates": int((candidate_counts == 0).sum()),
                "min_candidates_per_active_row": (
                    int(active_counts.min()) if len(active_counts) else 0
                ),
                "median_candidates_per_active_row": (
                    float(active_counts.median()) if len(active_counts) else 0.0
                ),
                "max_candidates_per_active_row": (
                    int(active_counts.max()) if len(active_counts) else 0
                ),
            }
        )
    else:
        phase2_cache.diagnostics["candidate_pruning_enabled"] = False
    if _diagnostics_out is not None:
        _diagnostics_out.update(phase2_cache.diagnostics)

    
    # Enable tqdm progress bar in pandas

    tqdm.pandas()

    # Determine which spots to process
    if common_group_name in table.obs.columns:
        if verbose:
            print(f"Processing spots with {common_group_name} != 0")
        spots_with_expression = table.obs[table.obs[common_group_name] != 0].index
    else:
        if verbose:
            print("common_group_name column not found in the table, processing all spots.")
        spots_with_expression = table.obs.index
    if aligned_candidate_mask is not None:
        has_candidates = aligned_candidate_mask.any(axis=1)
        spots_with_expression = pd.Index(spots_with_expression).intersection(
            has_candidates[has_candidates].index
        )

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
        "auc": function_row_auc_specific_v2,
        "ucell": function_row_ucell,
    }
    func = similarity_methods[method]
    if len(spots_with_expression) == 0:
        columns = phase2_cache.groups
        df = pd.DataFrame(0.0, index=table.obs.index, columns=columns)
        if method != "diagnostic" and add_to_obs:
            if verbose:
                print("Adding results to table.obs of sdata object")
            table.obs.drop(columns=df.columns, inplace=True, errors="ignore")
            table.obs = pd.merge(
                table.obs, df, left_index=True, right_index=True, how="left"
            )
        return df
    if method in MARKER_UNION_SAFE_METHODS and len(phase2_cache.marker_union) == 0:
        df = pd.DataFrame(
            0.0, index=table.obs.index, columns=phase2_cache.groups
        )
        if add_to_obs:
            if verbose:
                print("Adding results to table.obs of sdata object")
            table.obs.drop(columns=df.columns, inplace=True, errors="ignore")
            table.obs = pd.merge(
                table.obs, df, left_index=True, right_index=True, how="left"
            )
        return df

    # Show parallelization info
    #from .config import config  # Import inside function to prevent issues with joblib reloading
    if verbose:
        print("Number of threads used:", config.n_jobs)
        print("Batch size:", config.batch_size)

    # Run computations in parallel
    row_iterator = _iter_phase2_rows(
        table,
        spots_with_expression,
        phase2_cache.expression_genes,
    )
    total_rows = len(spots_with_expression)
    row_kwargs = dict(kwargs, phase2_cache=phase2_cache)
    group_position_lookup = {
        group: idx for idx, group in enumerate(phase2_cache.groups)
    }

    results = Parallel(
        n_jobs=config.n_jobs,
        batch_size=config.batch_size,
        backend="loky",  # Ensure workers do not inherit unnecessary imports
    )(
        delayed(process_row_with_suppression)(
            row,
            func,
            markers_df=None,
            gene_id_column=gene_id_column,
            candidate_group_positions=(
                tuple(
                    group_position_lookup[group]
                    for group in aligned_candidate_mask.columns[
                        aligned_candidate_mask.loc[obs_name].to_numpy(dtype=bool)
                    ]
                )
                if aligned_candidate_mask is not None
                else None
            ),
            **row_kwargs,
        )
        for obs_name, row in tqdm(
            row_iterator,
            total=total_rows,
            leave=True,
            position=0,
            disable=not verbose,
        )
    )


    # Convert results to a DataFrame
    if results:
        result_df = pd.DataFrame(results)
        result_df.set_index("Index", inplace=True)
        result_df = result_df["assigned_cluster"].apply(pd.Series)
        result_df = (
            result_df.reindex(columns=phase2_cache.groups, fill_value=0.0)
            .fillna(0.0)
        )
    else:
        result_df = pd.DataFrame(
            0.0,
            index=pd.Index([], name=table.obs.index.name),
            columns=phase2_cache.groups,
        )

    # For spots not processed (e.g., excluded by common_group_name != 0)
    # fill with zeros or NaNs, depending on your needs
    other_spots = table.obs.index[~table.obs.index.isin(spots_with_expression)]
    others_df = pd.DataFrame(0, index=other_spots, columns=result_df.columns)
    df = pd.concat([result_df, others_df]).reindex(table.obs.index)


    # Optionally merge back into table.obs
    if method != "diagnostic" and add_to_obs:
        if verbose:
            print("Adding results to table.obs of sdata object")
        table.obs.drop(columns=df.columns, inplace=True, errors='ignore')
        table.obs = pd.merge(table.obs, df, left_index=True, right_index=True)

    return df




#this function is used to read the markers from a file or from an single-cell anndata object and return a dataframe
def _adata_has_rank_genes_groups(adata, key):
    return hasattr(adata, "uns") and key in adata.uns


def _generate_scanpy_rank_genes_groups(
    adata,
    groupby,
    key,
    scanpy_method="wilcoxon",
    layer=None,
    use_raw=None,
    reference="rest",
    copy_adata=True,
    rank_genes_groups_kwargs=None,
):
    if groupby is None:
        raise ValueError(
            "groupby is required to generate markers from AnnData. Provide the "
            "obs column containing cell-type or cluster labels."
        )
    if not hasattr(adata, "obs") or groupby not in adata.obs.columns:
        raise ValueError(f"groupby={groupby!r} was not found in adata.obs.columns.")

    kwargs = dict(rank_genes_groups_kwargs or {})
    try:
        work_adata = adata.copy() if copy_adata else adata
        if str(work_adata.obs[groupby].dtype) != "category":
            work_adata.obs[groupby] = (
                work_adata.obs[groupby].astype(str).astype("category")
            )
        sc.tl.rank_genes_groups(
            work_adata,
            groupby=groupby,
            method=scanpy_method,
            key_added=key,
            layer=layer,
            use_raw=use_raw,
            reference=reference,
            **kwargs,
        )
    except Exception as exc:
        raise ValueError(
            "Could not generate markers with sc.tl.rank_genes_groups. Ensure "
            "adata contains normalized/log-transformed expression or provide a "
            "precomputed markers_df/filename."
        ) from exc
    return work_adata


def _pydeseq2_counts_error():
    return ValueError(
        "PyDESeq2 marker generation requires raw non-negative integer counts. "
        "Provide layer='counts' or another raw-count layer."
    )


def _get_adata_count_matrix(adata, layer="counts"):
    """Return a cells-by-genes raw-count DataFrame for pseudobulk analysis."""
    try:
        if layer is not None:
            if not hasattr(adata, "layers") or layer not in adata.layers:
                raise _pydeseq2_counts_error()
            matrix = adata.layers[layer]
        else:
            matrix = adata.X
    except ValueError:
        raise
    except Exception as exc:
        raise _pydeseq2_counts_error() from exc

    if matrix is None:
        raise _pydeseq2_counts_error()

    if issparse(matrix):
        values_to_check = np.asarray(matrix.data)
    else:
        values_to_check = np.asarray(matrix)

    try:
        finite = np.isfinite(values_to_check)
        if not finite.all():
            raise _pydeseq2_counts_error()
        if np.any(values_to_check < 0):
            raise _pydeseq2_counts_error()
        nonzero_values = values_to_check[values_to_check != 0]
        if nonzero_values.size and not np.allclose(
            nonzero_values, np.round(nonzero_values)
        ):
            raise _pydeseq2_counts_error()
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(
            "PyDESeq2 marker generation requires"
        ):
            raise
        raise _pydeseq2_counts_error() from exc

    dense_counts = matrix.toarray() if issparse(matrix) else np.asarray(matrix)
    return pd.DataFrame(
        dense_counts,
        index=adata.obs_names,
        columns=adata.var_names,
    )


def _build_one_vs_rest_pseudobulk(
    adata,
    target_group,
    groupby,
    sample_col,
    counts_df,
    min_cells_per_group=20,
    min_replicates_per_condition=2,
):
    """Aggregate raw counts by biological sample for one-vs-rest testing."""
    group_values = adata.obs[groupby].astype(str)
    sample_values = adata.obs[sample_col].astype(str)
    target_group = str(target_group)
    pseudobulk_rows = []
    metadata_rows = []

    for sample in sorted(sample_values.drop_duplicates().tolist()):
        sample_mask = sample_values == sample
        condition_masks = {
            "target": sample_mask & (group_values == target_group),
            "rest": sample_mask & (group_values != target_group),
        }
        for condition, mask in condition_masks.items():
            cell_ids = adata.obs_names[np.asarray(mask)]
            if len(cell_ids) < min_cells_per_group:
                continue
            row_name = f"{sample}__{target_group}__{condition}"
            summed = counts_df.loc[cell_ids].sum(axis=0)
            summed.name = row_name
            pseudobulk_rows.append(summed)
            metadata_rows.append((row_name, condition))

    if pseudobulk_rows:
        counts_pb = pd.DataFrame(pseudobulk_rows, columns=counts_df.columns)
        metadata_pb = pd.DataFrame(
            metadata_rows, columns=["pseudobulk_sample", "condition"]
        ).set_index("pseudobulk_sample")
        metadata_pb.index.name = counts_pb.index.name
    else:
        counts_pb = pd.DataFrame(columns=counts_df.columns)
        metadata_pb = pd.DataFrame(columns=["condition"])

    condition_counts = (
        metadata_pb["condition"].value_counts() if not metadata_pb.empty else {}
    )
    n_target = int(condition_counts.get("target", 0))
    n_rest = int(condition_counts.get("rest", 0))
    stats = {
        "n_target_replicates": n_target,
        "n_rest_replicates": n_rest,
        "skipped": False,
    }
    if (
        n_target < min_replicates_per_condition
        or n_rest < min_replicates_per_condition
    ):
        stats["skipped"] = True
        stats["reason"] = (
            "Insufficient pseudobulk replicates after cell-count filtering: "
            f"target={n_target}, rest={n_rest}, required="
            f"{min_replicates_per_condition} per condition."
        )
        return (
            pd.DataFrame(columns=counts_df.columns),
            pd.DataFrame(columns=["condition"]),
            stats,
        )

    return counts_pb, metadata_pb, stats


def _instantiate_pydeseq2_with_fallback(
    constructor,
    *args,
    quiet=True,
    n_cpus=None,
    **kwargs,
):
    attempts = [
        {"quiet": quiet, "n_cpus": n_cpus},
        {"n_cpus": n_cpus},
        {"quiet": quiet},
        {},
    ]
    last_error = None
    for optional_kwargs in attempts:
        call_kwargs = dict(kwargs)
        for name, value in optional_kwargs.items():
            if name not in call_kwargs:
                call_kwargs[name] = value
        try:
            return constructor(*args, **call_kwargs)
        except TypeError as exc:
            last_error = exc
    raise last_error


def _run_pydeseq2_one_vs_rest(
    counts_pb,
    metadata_pb,
    condition_col="condition",
    tested_level="target",
    reference_level="rest",
    alpha=0.05,
    n_cpus=None,
    quiet=True,
    deseq_kwargs=None,
    deseq_stats_kwargs=None,
):
    """Fit one PyDESeq2 target-vs-rest pseudobulk contrast."""
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError as exc:
        raise ImportError(
            "marker_method='pydeseq2' requires pydeseq2. Install it with "
            "`pip install pydeseq2`."
        ) from exc

    dds = _instantiate_pydeseq2_with_fallback(
        DeseqDataSet,
        counts=counts_pb,
        metadata=metadata_pb,
        design_factors=condition_col,
        quiet=quiet,
        n_cpus=n_cpus,
        **dict(deseq_kwargs or {}),
    )
    dds.deseq2()
    stat_res = _instantiate_pydeseq2_with_fallback(
        DeseqStats,
        dds,
        contrast=[condition_col, tested_level, reference_level],
        alpha=alpha,
        quiet=quiet,
        n_cpus=n_cpus,
        **dict(deseq_stats_kwargs or {}),
    )
    stat_res.summary()
    results_df = getattr(stat_res, "results_df", None)
    if results_df is None:
        raise ValueError("PyDESeq2 did not provide a results_df table.")
    return results_df


def compute_pseudobulk_deseq_markers(
    adata,
    groupby,
    sample_col,
    layer="counts",
    min_cells_per_group=20,
    min_replicates_per_condition=2,
    alpha=0.05,
    n_cpus=None,
    quiet=True,
    deseq_kwargs=None,
    deseq_stats_kwargs=None,
):
    """Generate one-vs-rest pseudobulk marker tables with PyDESeq2."""
    if groupby is None:
        raise ValueError("groupby is required for pseudobulk PyDESeq2 markers.")
    if not hasattr(adata, "obs") or groupby not in adata.obs.columns:
        raise ValueError(f"groupby={groupby!r} was not found in adata.obs.columns.")
    if sample_col is None:
        raise ValueError("sample_col is required for pseudobulk PyDESeq2 markers.")
    if sample_col not in adata.obs.columns:
        raise ValueError(f"sample_col={sample_col!r} was not found in adata.obs.columns.")

    counts_df = _get_adata_count_matrix(adata, layer=layer)
    groups = sorted(adata.obs[groupby].astype(str).unique().tolist())
    diagnostics = {
        "method": "pydeseq2_pseudobulk",
        "groupby": groupby,
        "sample_col": sample_col,
        "layer": layer,
        "groups_attempted": groups,
        "groups_completed": [],
        "groups_skipped": {},
        "min_cells_per_group": min_cells_per_group,
        "min_replicates_per_condition": min_replicates_per_condition,
    }
    marker_tables = []

    for target_group in groups:
        counts_pb, metadata_pb, group_stats = _build_one_vs_rest_pseudobulk(
            adata,
            target_group=target_group,
            groupby=groupby,
            sample_col=sample_col,
            counts_df=counts_df,
            min_cells_per_group=min_cells_per_group,
            min_replicates_per_condition=min_replicates_per_condition,
        )
        if group_stats["skipped"]:
            diagnostics["groups_skipped"][target_group] = group_stats
            continue

        results = _run_pydeseq2_one_vs_rest(
            counts_pb,
            metadata_pb,
            alpha=alpha,
            n_cpus=n_cpus,
            quiet=quiet,
            deseq_kwargs=deseq_kwargs,
            deseq_stats_kwargs=deseq_stats_kwargs,
        ).copy()
        required_results = {"log2FoldChange", "padj"}
        if not required_results.issubset(results.columns):
            missing = sorted(required_results.difference(results.columns))
            raise ValueError(
                f"PyDESeq2 results are missing required columns: {missing}."
            )
        results["group"] = target_group
        results["names"] = results.index.astype(str)
        results["logfoldchanges"] = results["log2FoldChange"]
        results["pvals_adj"] = results["padj"]
        if "stat" in results.columns:
            results["scores"] = pd.to_numeric(
                results["stat"], errors="coerce"
            ).abs()
        else:
            adjusted = pd.to_numeric(results["padj"], errors="coerce")
            results["scores"] = -np.log10(
                adjusted.clip(lower=np.finfo(float).tiny)
            )
        marker_tables.append(results.reset_index(drop=True))
        diagnostics["groups_completed"].append(target_group)

    if not marker_tables:
        raise ValueError(
            "No groups produced pseudobulk PyDESeq2 markers. Skipped group "
            f"diagnostics: {diagnostics['groups_skipped']}"
        )
    return pd.concat(marker_tables, ignore_index=True), diagnostics


def read_markers_dataframe(sdata,
                           filename=None,
                           adata=None,
                           exclude_celltype=None,
                           bin_size=8, #if no segmentation found
                           top_n_genes=60,
                           sort_by_column="scores",
                           ascending=False,
                           gene_id_column="names",
                           celltype="group",
                           key="rank_genes_groups",
                           log2fc_min=0.25,
                           pval_cutoff=0.05,
                           drop_ribosomal=False,
                           drop_mitochondrial=False,
                           markers_df=None,
                           prepared_markers=None,
                           table_key=None,
                           preferred_table_keys=None,
                           source=None,
                           return_diagnostics=False,
                           verbose=True,
                           marker_method="auto",
                           groupby=None,
                           scanpy_method="wilcoxon",
                           layer=None,
                           use_raw=None,
                           reference="rest",
                           copy_adata=True,
                           rank_genes_groups_kwargs=None,
                           sample_col=None,
                           min_cells_per_group=20,
                           min_replicates_per_condition=2,
                           deseq_alpha=0.05,
                           deseq_n_cpus=None,
                           deseq_quiet=True,
                           deseq_kwargs=None,
                           deseq_stats_kwargs=None,
                           reference_min_cells=25,
                           reference_min_mean=2e-4,
                           reference_min_log2fc=1.0,
                           reference_min_detection=0.10,
                           reference_min_detection_delta=0.05,
                           reference_pseudocount=1e-9,
                           reference_contrast="max_other",
                           marker_roles: str = "shared",
                           reference_presence_min_log2fc: float = 0.5,
                           reference_presence_min_detection_delta: float = 0.0,
                           reference_negative_min_log2fc: float = 1.0,
                           reference_negative_min_detection: float = 0.10,
                           reference_negative_min_detection_delta: float = 0.05,
                           marker_role_inference: str = "none"):
    """Resolve, standardize, and filter a marker table.

    Input priority is ``prepared_markers``, then ``markers_df``, then
    ``filename``, then ``adata``. The returned DataFrame uses canonical marker
    columns such as ``group`` and ``names`` and is filtered to the spatial gene
    universe.

    Parameters
    ----------
    sdata
        SpatialData-like container or AnnData-like spatial table.
    filename
        CSV or Excel marker file.
    adata
        AnnData reference used for existing Scanpy markers, generated Scanpy
        markers, pseudobulk PyDESeq2 markers, or reference-profile markers.
    markers_df
        Marker DataFrame to process directly.
    prepared_markers
        Reusable marker preparation. Takes priority over all other marker
        sources.
    marker_method
        One of ``"auto"``, ``"existing"``, ``"scanpy"``, ``"pydeseq2"``,
        ``"deseq2"``, ``"pseudobulk_deseq2"``, ``"reference"``, or
        ``"rctd_like"``.
    marker_roles
        ``"shared"`` or ``"phase_specific"``.
    marker_role_inference
        ``"none"`` or ``"scanpy_signed"``.
    return_diagnostics
        If True, return ``(markers_df, diagnostics)``.

    Returns
    -------
    pandas.DataFrame
        Standardized marker table, or ``(markers_df, diagnostics)`` when
        ``return_diagnostics=True``.

    Raises
    ------
    ValueError
        If no marker input is provided, if a source cannot be read, or if role
        settings are incompatible with the selected marker method.
    """
    validate_choice(marker_method, MARKER_METHODS, "marker_method")
    validate_choice(marker_roles, MARKER_ROLE_MODES, "marker_roles")
    validate_choice(
        marker_role_inference,
        MARKER_ROLE_INFERENCE_MODES,
        "marker_role_inference",
    )

    generated_rank_genes_groups = False
    generated_pseudobulk_deseq = False
    generated_reference_profile = False
    used_existing_rank_genes_groups = False
    prepared_markers_used = False
    marker_signature = None
    deseq_diagnostics = None
    reference_diagnostics = None
    role_inference_diagnostics = {
        "mode": marker_role_inference,
        "requested": marker_role_inference != "none",
        "applied": False,
        "existing_roles_preserved": False,
        "input_source": None,
    }
    table = get_table(
        sdata,
        bin_size=bin_size,
        table_key=table_key,
        preferred_table_keys=preferred_table_keys,
    )

    if prepared_markers is not None:
        from .markers import PreparedMarkers, select_prepared_markers

        if not isinstance(prepared_markers, PreparedMarkers):
            raise TypeError("prepared_markers must be a PreparedMarkers object.")
        prepared_has_roles = "marker_role" in prepared_markers.raw_markers_df.columns
        if marker_role_inference == "scanpy_signed" and not prepared_has_roles:
            raise ValueError(
                "PreparedMarkers does not contain inferred marker roles. "
                "Recreate it with marker_role_inference='scanpy_signed'."
            )
        resolved_source = source if source is not None else prepared_markers.source
        df = select_prepared_markers(
            prepared_markers,
            gene_universe=table.var_names,
            exclude_celltype=exclude_celltype,
            top_n_genes=top_n_genes,
            sort_by_column=sort_by_column,
            ascending=ascending,
            log2fc_min=log2fc_min,
            pval_cutoff=pval_cutoff,
            drop_ribosomal=drop_ribosomal,
            drop_mitochondrial=drop_mitochondrial,
            source=resolved_source,
        )
        prepared_markers_used = True
        marker_signature = prepared_markers.signature
        if marker_role_inference == "scanpy_signed" and prepared_has_roles:
            role_inference_diagnostics = {
                "mode": "scanpy_signed",
                "requested": True,
                "applied": False,
                "existing_roles_preserved": True,
                "input_source": resolved_source,
            }
    elif markers_df is not None:
        raw_df = markers_df
        resolved_source = source if source is not None else "dataframe"
    elif filename is not None:
        try:
            raw_df = pd.read_csv(filename)
        except Exception:
            try:
                raw_df = pd.read_excel(filename)
            except Exception as exc:
                raise ValueError(
                    f"Could not read marker file {filename!r} as CSV or Excel."
                ) from exc
        resolved_source = source if source is not None else "file"
    elif adata is not None:
        if marker_method in REFERENCE_MARKER_METHODS:
            if marker_role_inference == "scanpy_signed":
                raise ValueError(
                    "marker_role_inference='scanpy_signed' is intended for Scanpy-style "
                    "signed marker results. It is not applied to reference-profile or "
                    "PyDESeq2 marker generation."
                )
            from .markers import compute_reference_profile_markers

            raw_df, reference_diagnostics = compute_reference_profile_markers(
                adata,
                groupby=groupby,
                layer=layer,
                min_cells_per_group=reference_min_cells,
                min_mean_expression=reference_min_mean,
                min_log2fc=reference_min_log2fc,
                min_detection=reference_min_detection,
                min_detection_delta=reference_min_detection_delta,
                contrast=reference_contrast,
                top_n_genes=None,
                pseudocount=reference_pseudocount,
                drop_ribosomal=False,
                drop_mitochondrial=False,
                marker_roles=marker_roles,
                reference_presence_min_log2fc=reference_presence_min_log2fc,
                reference_presence_min_detection_delta=reference_presence_min_detection_delta,
                reference_negative_min_log2fc=reference_negative_min_log2fc,
                reference_negative_min_detection=reference_negative_min_detection,
                reference_negative_min_detection_delta=reference_negative_min_detection_delta,
            )
            generated_reference_profile = True
            resolved_source = source if source is not None else "reference_profile"
        elif marker_method in PYDESEQ2_MARKER_METHODS:
            if marker_role_inference == "scanpy_signed":
                raise ValueError(
                    "marker_role_inference='scanpy_signed' is intended for Scanpy-style "
                    "signed marker results. It is not applied to reference-profile or "
                    "PyDESeq2 marker generation."
                )
            if marker_roles == "phase_specific":
                raise ValueError(
                    "Automatic phase-specific role generation is currently supported only for "
                    "marker_method='reference'. Provide a marker table with marker_role for "
                    "Scanpy or DESeq-derived markers."
                )
            raw_df, deseq_diagnostics = compute_pseudobulk_deseq_markers(
                adata,
                groupby=groupby,
                sample_col=sample_col,
                layer=layer,
                min_cells_per_group=min_cells_per_group,
                min_replicates_per_condition=min_replicates_per_condition,
                alpha=deseq_alpha,
                n_cpus=deseq_n_cpus,
                quiet=deseq_quiet,
                deseq_kwargs=deseq_kwargs,
                deseq_stats_kwargs=deseq_stats_kwargs,
            )
            generated_pseudobulk_deseq = True
            resolved_source = (
                source if source is not None else "pydeseq2_pseudobulk"
            )
        else:
            if marker_roles == "phase_specific" and marker_role_inference != "scanpy_signed":
                raise ValueError(
                    "Automatic phase-specific role generation is currently supported only for "
                    "marker_method='reference'. Provide a marker table with marker_role for "
                    "Scanpy or DESeq-derived markers."
                )
            if _adata_has_rank_genes_groups(adata, key):
                marker_adata = adata
                source_detail = f"adata.uns[{key!r}]"
                used_existing_rank_genes_groups = True
            elif marker_method == "existing":
                raise ValueError(
                    f"Could not read markers from adata.uns[{key!r}]. Run "
                    "sc.tl.rank_genes_groups first, set marker_method='scanpy' "
                    "with groupby=..., or provide markers_df/filename."
                )
            else:
                marker_adata = _generate_scanpy_rank_genes_groups(
                    adata,
                    groupby=groupby,
                    key=key,
                    scanpy_method=scanpy_method,
                    layer=layer,
                    use_raw=use_raw,
                    reference=reference,
                    copy_adata=copy_adata,
                    rank_genes_groups_kwargs=rank_genes_groups_kwargs,
                )
                generated_rank_genes_groups = True
                source_detail = f"scanpy_generated[{key!r}]"
            try:
                raw_df = sc.get.rank_genes_groups_df(
                    marker_adata,
                    group=None,
                    key=key,
                    pval_cutoff=None,
                    log2fc_min=None,
                )
            except Exception as exc:
                raise ValueError(
                    f"Could not read markers from adata.uns[{key!r}]. "
                    "Run sc.tl.rank_genes_groups first or provide "
                    "markers_df/filename."
                ) from exc
            resolved_source = source if source is not None else source_detail
    else:
        raise ValueError(
            "Please provide prepared_markers, markers_df, filename, or an "
            "adata object with an existing rank_genes_groups result."
        )

    if not prepared_markers_used:
        schema = MarkerSchema(
            group_col=celltype,
            gene_col=gene_id_column,
            lfc_col="logfoldchanges",
            padj_col="pvals_adj",
            score_col="scores",
        )

        if marker_role_inference == "scanpy_signed":
            from .markers import infer_scanpy_signed_marker_roles

            raw_df, role_inference_diagnostics = infer_scanpy_signed_marker_roles(
                raw_df,
                schema=schema,
                log2fc_min=log2fc_min,
            )
            role_inference_diagnostics = {
                **role_inference_diagnostics,
                "requested": True,
                "applied": role_inference_diagnostics.get(
                    "inference_applied", False
                ),
                "input_source": resolved_source,
            }
            if (
                role_inference_diagnostics.get("inference_applied")
                and marker_roles == "phase_specific"
            ):
                raise ValueError(
                    "Signed Scanpy role inference creates positive and negative roles only. "
                    "Use marker_roles='shared', provide a manually annotated marker table "
                    "with presence/identity roles, or use marker_method='reference'."
                )

        # ``scores`` is the historical default, but not every valid marker format
        # provides a score. In that case use the standard metric preference.
        effective_sort_column = sort_by_column
        resolved_columns = resolve_marker_columns(raw_df, schema=schema)
        if sort_by_column == "scores" and "scores" not in resolved_columns:
            effective_sort_column = None

        df = standardize_marker_dataframe(
            raw_df,
            schema=schema,
            gene_universe=table.var_names,
            exclude_celltype=exclude_celltype,
            top_n_genes=top_n_genes,
            sort_by_column=effective_sort_column,
            ascending=ascending,
            log2fc_min=log2fc_min,
            pval_cutoff=pval_cutoff,
            drop_ribosomal=drop_ribosomal,
            drop_mitochondrial=drop_mitochondrial,
            source=resolved_source,
        )

    used_adata = (
        not prepared_markers_used
        and markers_df is None
        and filename is None
        and adata is not None
    )
    generated_rank_genes_groups = generated_rank_genes_groups if used_adata else False
    generated_pseudobulk_deseq = (
        generated_pseudobulk_deseq if used_adata else False
    )
    generated_reference_profile = generated_reference_profile if used_adata else False

    if verbose and generated_rank_genes_groups:
        print(
            "Generated marker genes using sc.tl.rank_genes_groups with "
            f"groupby={groupby!r}, method={scanpy_method!r}."
        )
    if verbose and generated_pseudobulk_deseq:
        print(
            "Generated marker genes using pseudobulk PyDESeq2 with "
            f"groupby={groupby!r}, sample_col={sample_col!r}."
        )
    if verbose:
        print("Unique cell types detected in the dataframe:")
        print(df["group"].unique())

    if return_diagnostics:
        diagnostics = {
            "source": resolved_source,
            "n_markers": int(df.shape[0]),
            "n_celltypes": int(df["group"].nunique()),
            "celltypes": df["group"].drop_duplicates().tolist(),
            "marker_counts_per_celltype": (
                df.groupby(df["group"]).size().to_dict()
            ),
            "n_spatial_genes": int(len(table.var_names)),
            "marker_method": marker_method,
            "groupby": groupby,
            "generated_rank_genes_groups": generated_rank_genes_groups,
            "rank_genes_groups_key": key,
            "scanpy_method": scanpy_method if used_adata else None,
            "generated_pseudobulk_deseq": generated_pseudobulk_deseq,
            "pseudobulk_deseq": deseq_diagnostics,
            "generated_reference_profile": generated_reference_profile,
            "reference_profile": reference_diagnostics,
            "reference_contrast": reference_contrast if generated_reference_profile else None,
            "prepared_markers_used": prepared_markers_used,
            "marker_signature": marker_signature,
            "marker_generation_reused": (
                True if prepared_markers_used else used_existing_rank_genes_groups
            ),
            "marker_role_inference": role_inference_diagnostics,
            "top_n_applied_by": (
                "read_markers_dataframe" if top_n_genes is not None else None
            ),
        }
        if "marker_role" in df.columns:
            diagnostics["marker_roles"] = marker_roles
            diagnostics["marker_role_counts"] = (
                df["marker_role"].value_counts().astype(int).to_dict()
            )
        return df, diagnostics

    return df


from .markers import (
    PreparedMarkers as _PreparedMarkers,
    _adata_has_rank_genes_groups as _adata_has_rank_genes_groups,
    _build_one_vs_rest_pseudobulk as _build_one_vs_rest_pseudobulk,
    _generate_scanpy_rank_genes_groups as _generate_scanpy_rank_genes_groups,
    _get_adata_count_matrix as _get_adata_count_matrix,
    compute_pseudobulk_deseq_markers as compute_pseudobulk_deseq_markers,
    prepare_markers as _prepare_markers,
    select_prepared_markers as _select_prepared_markers,
)


def _build_marker_compat_diagnostics(
    prepared,
    selected_df,
    table,
    *,
    marker_method,
    groupby,
    key,
    scanpy_method,
    marker_roles,
    marker_role_inference,
    prepared_markers_used,
    selection_diagnostics,
    top_n_applied_by,
):
    prep_diagnostics = dict(getattr(prepared, "diagnostics", {}) or {})
    input_kind = prep_diagnostics.get("input_kind")
    role_inference = prep_diagnostics.get("marker_role_inference")
    if prepared_markers_used and marker_role_inference in {"signed", "scanpy_signed"}:
        role_inference = {
            "requested_mode": marker_role_inference,
            "mode": "signed",
            "normalized_mode": "signed",
            "requested": True,
            "applied": False,
            "existing_roles_preserved": "marker_role" in prepared.raw_markers_df,
            "input_source": selection_diagnostics.get("source", prepared.source),
        }
    if not isinstance(role_inference, dict):
        role_inference = {
            "requested_mode": marker_role_inference,
            "mode": "signed" if marker_role_inference == "scanpy_signed" else marker_role_inference,
            "normalized_mode": "signed" if marker_role_inference == "scanpy_signed" else marker_role_inference,
            "requested": marker_role_inference != "none",
            "applied": False,
            "existing_roles_preserved": False,
            "input_source": None,
        }

    diagnostics = {
        "source": selection_diagnostics.get("source", prepared.source),
        "n_markers": int(selected_df.shape[0]),
        "n_celltypes": int(selected_df["group"].nunique()),
        "celltypes": selected_df["group"].drop_duplicates().tolist(),
        "marker_counts_per_celltype": (
            selected_df.groupby(selected_df["group"]).size().to_dict()
        ),
        "n_spatial_genes": int(len(table.var_names)),
        "marker_method": prep_diagnostics.get(
            "marker_method", getattr(prepared, "marker_method", marker_method)
        ),
        "groupby": prep_diagnostics.get("groupby", groupby),
        "generated_rank_genes_groups": bool(
            prep_diagnostics.get("generated_rank_genes_groups", False)
        ),
        "rank_genes_groups_key": key,
        "scanpy_method": (
            scanpy_method
            if str(input_kind or "").startswith("anndata")
            else None
        ),
        "generated_pseudobulk_deseq": bool(
            prep_diagnostics.get("generated_pseudobulk_deseq", False)
        ),
        "pseudobulk_deseq": prep_diagnostics.get("pseudobulk_deseq"),
        "generated_reference_profile": bool(
            prep_diagnostics.get("generated_reference_profile", False)
        ),
        "reference_profile": prep_diagnostics.get("reference_profile"),
        "reference_contrast": prep_diagnostics.get("reference_contrast"),
        "prepared_markers_used": bool(prepared_markers_used),
        "marker_signature": getattr(prepared, "signature", None),
        "marker_generation_reused": bool(
            prepared_markers_used or input_kind == "anndata_existing_scanpy"
        ),
        "marker_role_inference": role_inference,
        "top_n_applied_by": top_n_applied_by,
        "input_kind": input_kind,
        "preparation": prep_diagnostics,
        "selection": selection_diagnostics,
    }
    if "marker_role" in selected_df.columns:
        diagnostics["marker_roles"] = marker_roles
        diagnostics["marker_role_counts"] = (
            selected_df["marker_role"].value_counts().astype(int).to_dict()
        )
    return diagnostics


def read_markers_dataframe(sdata,
                           filename=None,
                           adata=None,
                           exclude_celltype=None,
                           bin_size=8,
                           top_n_genes=60,
                           sort_by_column="scores",
                           ascending=False,
                           gene_id_column="names",
                           celltype="group",
                           key="rank_genes_groups",
                           log2fc_min=0.25,
                           pval_cutoff=0.05,
                           drop_ribosomal=False,
                           drop_mitochondrial=False,
                           markers_df=None,
                           prepared_markers=None,
                           table_key=None,
                           preferred_table_keys=None,
                           source=None,
                           return_diagnostics=False,
                           verbose=True,
                           marker_method="auto",
                           groupby=None,
                           scanpy_method="wilcoxon",
                           layer=None,
                           use_raw=None,
                           reference="rest",
                           copy_adata=True,
                           rank_genes_groups_kwargs=None,
                           sample_col=None,
                           min_cells_per_group=20,
                           min_replicates_per_condition=2,
                           deseq_alpha=0.05,
                           deseq_n_cpus=None,
                           deseq_quiet=True,
                           deseq_kwargs=None,
                           deseq_stats_kwargs=None,
                           reference_min_cells=25,
                           reference_min_mean=2e-4,
                           reference_min_log2fc=1.0,
                           reference_min_detection=0.10,
                           reference_min_detection_delta=0.05,
                           reference_pseudocount=1e-9,
                           reference_contrast="max_other",
                           marker_roles: str = "shared",
                           reference_presence_min_log2fc: float = 0.5,
                           reference_presence_min_detection_delta: float = 0.0,
                           reference_negative_min_log2fc: float = 1.0,
                           reference_negative_min_detection: float = 0.10,
                           reference_negative_min_detection_delta: float = 0.05,
                           marker_role_inference: str = "none"):
    """Compatibility wrapper returning a spatial-selected marker DataFrame.

    ``marker_role_inference`` accepts ``"none"``, preferred ``"signed"`` for
    Scanpy/DE-style signed log-fold-change tables, and ``"scanpy_signed"`` as
    a backward-compatible alias.
    """
    if prepared_markers is not None and not isinstance(prepared_markers, _PreparedMarkers):
        raise TypeError("prepared_markers must be a PreparedMarkers object.")
    if (
        prepared_markers is not None
        and marker_role_inference in {"signed", "scanpy_signed"}
        and "marker_role" not in prepared_markers.raw_markers_df.columns
    ):
        raise ValueError(
            "PreparedMarkers does not contain inferred marker roles. "
            "Recreate it with marker_role_inference='signed'."
        )
    table = get_table(
        sdata,
        bin_size=bin_size,
        table_key=table_key,
        preferred_table_keys=preferred_table_keys,
    )
    prepared = _prepare_markers(
        adata=adata,
        prepared_markers=prepared_markers,
        markers_df=markers_df,
        filename=filename,
        source=source,
        marker_method=marker_method,
        groupby=groupby,
        marker_key=key,
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
        reference_min_cells=reference_min_cells,
        reference_min_mean=reference_min_mean,
        reference_min_log2fc=reference_min_log2fc,
        reference_min_detection=reference_min_detection,
        reference_min_detection_delta=reference_min_detection_delta,
        reference_pseudocount=reference_pseudocount,
        reference_contrast=reference_contrast,
        marker_roles=marker_roles,
        reference_presence_min_log2fc=reference_presence_min_log2fc,
        reference_presence_min_detection_delta=reference_presence_min_detection_delta,
        reference_negative_min_log2fc=reference_negative_min_log2fc,
        reference_negative_min_detection=reference_negative_min_detection,
        reference_negative_min_detection_delta=reference_negative_min_detection_delta,
        marker_role_inference=marker_role_inference,
        marker_role_inference_log2fc_min=log2fc_min,
        celltype=celltype,
        gene_id_column=gene_id_column,
        verbose=False,
    )
    df, selection_diagnostics = _select_prepared_markers(
        prepared,
        gene_universe=table.var_names,
        exclude_celltype=exclude_celltype,
        top_n_genes=top_n_genes,
        sort_by_column=sort_by_column,
        ascending=ascending,
        log2fc_min=log2fc_min,
        pval_cutoff=pval_cutoff,
        drop_ribosomal=drop_ribosomal,
        drop_mitochondrial=drop_mitochondrial,
        source=source,
        return_diagnostics=True,
    )

    prep_diagnostics = getattr(prepared, "diagnostics", {}) or {}
    if verbose and prep_diagnostics.get("generated_rank_genes_groups"):
        print(
            "Generated marker genes using sc.tl.rank_genes_groups with "
            f"groupby={groupby!r}, method={scanpy_method!r}."
        )
    if verbose and prep_diagnostics.get("generated_pseudobulk_deseq"):
        print(
            "Generated marker genes using pseudobulk PyDESeq2 with "
            f"groupby={groupby!r}, sample_col={sample_col!r}."
        )
    if verbose:
        print("Unique cell types detected in the dataframe:")
        print(df["group"].unique())

    if return_diagnostics:
        diagnostics = _build_marker_compat_diagnostics(
            prepared,
            df,
            table,
            marker_method=marker_method,
            groupby=groupby,
            key=key,
            scanpy_method=scanpy_method,
            marker_roles=marker_roles,
            marker_role_inference=marker_role_inference,
            prepared_markers_used=prepared_markers is not None,
            selection_diagnostics=selection_diagnostics,
            top_n_applied_by=(
                "read_markers_dataframe" if top_n_genes is not None else None
            ),
        )
        return df, diagnostics
    return df


def assign_clusters_from_df(
    sdata,
    df,
    bin_size=8,
    results_column="easydecon",
    method="max",
    allow_multiple=False,
    diagnostic=None,
    fold_change_threshold=2.0,
    minimum_evidence=0.0,
    tie_tolerance=1e-12,
    add_to_obs=True,
    verbose=True,
):
    """Convert a score matrix into hard assignments.

    Parameters
    ----------
    sdata
        SpatialData-like container or AnnData-like table.
    df
        Score matrix with spatial locations as rows and marker groups as
        columns.
    bin_size
        Bin size used when resolving a SpatialData table.
    results_column
        Column name for the primary assignment.
    method
        One of ``"max"``, ``"zmax"``, or ``"hybrid"``.
    allow_multiple
        Multiple assignments are supported only with ``method="hybrid"``.
    fold_change_threshold
        Hybrid assignment threshold comparing top and second adaptive
        probabilities.
    minimum_evidence
        Minimum score required before a row can receive a hard assignment.
    tie_tolerance
        Tolerance used to leave near-tied winners unassigned.
    add_to_obs
        If True, merge assignment columns into ``table.obs``.

    Returns
    -------
    pandas.DataFrame
        Assignment labels aligned to ``table.obs.index``.
    """

    validate_choice(method, ASSIGN_METHODS, "method")
    _validate_finite_nonnegative(minimum_evidence, "minimum_evidence")
    _validate_finite_nonnegative(tie_tolerance, "tie_tolerance")
    table = get_table(sdata, bin_size=bin_size)
    df_filtered = df.reindex(table.obs.index).fillna(0)
    if df_filtered.shape[1] == 0:
        raise ValueError("df must contain at least one score/proportion column.")
    table.obs.drop(columns=[results_column], inplace=True, errors='ignore')

    def softmax(row, **kwargs):
        v = row.to_numpy(dtype=float)
        # treat non-finite as very negative so they don't contribute
        v = np.where(np.isfinite(v), v, -np.inf)
        # if row has <2 finite or variance ~0 → uninformative
        finite = np.isfinite(v)
        if finite.sum() < 2 or (np.nanmax(v[finite]) - np.nanmin(v[finite]) < 1e-12):
            return pd.Series(np.zeros_like(v, dtype=float), index=row.index)
        m = np.nanmax(v)
        ex = np.exp(v - m)
        ex[~np.isfinite(ex)] = 0.0
        s = ex.sum()
        probs = ex / s if s > 0 else np.zeros_like(ex)
        return pd.Series(probs, index=row.index)


    if method == "max" and not allow_multiple:
        assignments = df_filtered.apply(
            _select_unique_winner,
            axis=1,
            minimum_evidence=minimum_evidence,
            tie_tolerance=tie_tolerance,
        )
        df_reindexed = assignments.to_frame(results_column).astype(
            "category"
        ).reindex(table.obs.index, fill_value=np.nan)

    elif method == "zmax" and not allow_multiple:
        zscores = df_filtered.replace(0, np.nan).apply(
            lambda x: zscore(x, nan_policy="omit"),
            axis=0,
        )
        zscores = zscores.replace([np.inf, -np.inf], np.nan)
        assignments = zscores.apply(
            _select_unique_winner,
            axis=1,
            minimum_evidence=minimum_evidence,
            tie_tolerance=tie_tolerance,
        )
        df_reindexed = assignments.to_frame(results_column).astype(
            "category"
        ).reindex(table.obs.index, fill_value=np.nan)

    elif method == "hybrid":
        # row-wise zscore
        if allow_multiple and verbose:
            print("Multiple assignments per spot allowed, so hybrid assignment method is selected...")
        original_numeric = df_filtered.apply(pd.to_numeric, errors="coerce")
        original_numeric = original_numeric.replace([np.inf, -np.inf], np.nan)
        original_max = original_numeric.max(axis=1, skipna=True)
        row_is_all_zero_or_nan = (
            ~np.isfinite(original_numeric).any(axis=1)
            | (original_max <= minimum_evidence)
            | original_max.isna()
        )
        informative_mask = ~row_is_all_zero_or_nan
        if not informative_mask.any():
            df_reindexed = pd.DataFrame(
                {results_column: pd.Series(np.nan, index=table.obs.index)}
            ).astype("category")
        else:
            similarity_zscores = df_filtered[informative_mask].apply(
                lambda r: zscore(r.to_numpy(dtype=float), nan_policy='omit'),
                axis=1
            )
            similarity_zscores = pd.DataFrame(
                np.vstack(similarity_zscores.values),
                index=df_filtered[informative_mask].index,
                columns=df_filtered[informative_mask].columns
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
            adaptive_probs = similarity_zscores.apply(softmax, axis=1)

            def adaptive_assign(row):
                if not allow_multiple:
                    original_winner = _select_unique_winner(
                        original_numeric.loc[row.name],
                        minimum_evidence=minimum_evidence,
                        tie_tolerance=tie_tolerance,
                    )
                    if pd.isna(original_winner):
                        return np.nan
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
                        return ';'.join(eligible.index.tolist())
                    else:
                        return np.nan
                else:
                    return np.nan

            assigned_clusters = []
            for _, row in tqdm(
                adaptive_probs.iterrows(),
                total=adaptive_probs.shape[0],
                desc="Assigning clusters",
                leave=True,
                position=0,
                disable=not verbose,
            ):
                assigned_clusters.append(adaptive_assign(row))

            df_reindexed = pd.DataFrame(assigned_clusters, index=adaptive_probs.index, columns=[results_column]).astype('category').reindex(table.obs.index, fill_value=np.nan)

    else:
        raise ValueError("allow_multiple=True requires method='hybrid'.")


    if allow_multiple and method == "hybrid":
        tmp = pd.DataFrame()
        tmp[[f"{results_column}", f"{results_column}_second", f"{results_column}_third"]] = df_reindexed[results_column].str.split(";",n=2,expand=True)

        df_reindexed = tmp.astype('category').reindex(table.obs.index)

    if add_to_obs:
        if verbose:
            print("Adding results to table.obs of sdata object")
        table.obs.drop(
            columns=df_reindexed.columns,
            inplace=True, errors='ignore'
        )

        table.obs = pd.merge(table.obs, df_reindexed, left_index=True, right_index=True, how="left")

    if diagnostic is not None:
        for r in table.obs[[results_column]].itertuples(index=True, name='Pandas'):
            if not pd.isna(getattr(r, results_column)):
                table.obs.at[r.Index, results_column] = getattr(r, results_column)

    return df_reindexed


def visualize_only_selected_clusters(sdata,clusters,bin_size=8,results_column="easydecon",temp_column="tmp"):
    table = get_table(sdata, bin_size=bin_size)

    table.obs.drop(columns=[temp_column],inplace=True,errors='ignore')
    #table.obs=pd.merge(table.obs, df.idxmax(axis=1).to_frame(results_column).astype('category'), left_index=True, right_index=True)
    table.obs[temp_column]=table.obs[results_column].apply(lambda x: x if x in clusters else np.nan)
    return

def plot_assigned_clusters_from_dataframe(sdata,dataframe,sample_id,bin_size=8,title="Assigned Clusters",cmap="tab20",legend_fontsize=8,figsize=(5,5),dpi=200,method="matplotlib",scale=1,verbose=True):
    assign_clusters_from_df(
        sdata,
        df=dataframe,
        bin_size=8,
        results_column="plotted_clusters",
        verbose=verbose,
    )
    
    sdata.pl.render_images("queried_cytassist").pl.render_shapes(
        f"{sample_id}_square_{bin_size:03}um", color="plotted_clusters",cmap=cmap,method=method,scale=scale
    ).pl.show(coordinate_systems="global", title=title, legend_fontsize=legend_fontsize,figsize=figsize,dpi=dpi)

    return


def napari_region_assignment(sdata,key="Shapes",bin_size=8,column="napari",target_coordinate_system="global"):
    try:
        from spatialdata import polygon_query
    except ImportError as exc:
        raise ImportError("napari_region_assignment requires spatialdata to be installed.") from exc

    try:
        sdata[key]
    except:
        raise ValueError("Please provide a valid key for the shapes in the spatial data object that assigned via Napari")
        
    
    sdata.tables[f"square_{bin_size:03}um"].obs.drop(columns=column,inplace=True,errors='ignore')
    indices_list = []
    for g in sdata[key].geometry:
        indices=polygon_query(sdata,polygon=g,target_coordinate_system=target_coordinate_system).tables[f"square_{bin_size:03}um"].obs.index
        indices_list.extend(indices)
    
    df=pd.DataFrame("No", index=sdata.tables[f"square_{bin_size:03}um"].obs.index, columns=[column])
    df[column]=np.where(df.index.isin(set(indices_list)), 'Yes', 'No')
    sdata.tables[f"square_{bin_size:03}um"].obs=pd.merge(sdata.tables[f"square_{bin_size:03}um"].obs, df, left_index=True, right_index=True)
    return



def process_row(row,func, **kwargs):
    return pd.Series({
        'Index': row.name,
        #'assigned_cluster': func(row, markers_df, gene_id_column=gene_id_column, similarity_by_column=similarity_by_column,threshold=threshold)
        'assigned_cluster': func(row, **kwargs)
    })


def _candidate_groups_for_cache(phase2_cache, kwargs):
    positions = kwargs.get("candidate_group_positions")
    if positions is None:
        return phase2_cache.groups
    return tuple(
        phase2_cache.groups[int(pos)]
        for pos in positions
        if 0 <= int(pos) < len(phase2_cache.groups)
    )


MARKER_UNION_SAFE_METHODS = frozenset(
    {
        "correlation",
        "cosine",
        "euclidean",
        "sum",
        "mean",
        "median",
        "diagnostic",
        "auc",
        "ucell",
    }
)

FULL_GENE_ROW_METHODS = frozenset({"jaccard", "overlap", "wjaccard"})


@dataclass(frozen=True)
class _Phase2Cache:
    method: str
    groups: tuple
    expression_genes: tuple
    marker_union: tuple
    group_genes: dict
    group_gene_positions: dict
    group_reference_values: dict
    group_marker_sets: dict
    group_weights: dict
    auc_signatures: object | None
    ucell_signatures: object | None
    uses_full_gene_row: bool
    diagnostics: dict


def _assert_phase2_method_categories():
    categorized = MARKER_UNION_SAFE_METHODS | FULL_GENE_ROW_METHODS
    missing = set(SIMILARITY_METHODS) - categorized
    overlap = MARKER_UNION_SAFE_METHODS & FULL_GENE_ROW_METHODS
    if missing or overlap:
        raise AssertionError(
            "Every similarity method must belong to exactly one Phase 2 "
            f"extraction category. Missing={sorted(missing)}, overlap={sorted(overlap)}."
        )


def _unique_in_order(values):
    seen = set()
    ordered = []
    for value in values:
        value = str(value)
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _build_auc_signatures(
    markers_df,
    spatial_gene_names,
    gene_id_column="names",
    weight_column="logfoldchanges",
    top_n_markers=None,
    drop_shared_markers=False,
):
    top_n_markers = _validate_optional_positive_integer(
        top_n_markers, "top_n_markers"
    )
    marker_lists = {}
    spatial_set = {str(gene) for gene in spatial_gene_names}
    for group in markers_df["group"].drop_duplicates().astype(str).tolist():
        group_df = markers_df.loc[markers_df["group"].astype(str) == group].copy()
        group_df = group_df.loc[group_df[gene_id_column].astype(str).isin(spatial_set)]
        if weight_column in group_df.columns:
            group_df = group_df.sort_values(weight_column, ascending=False, kind="stable")
        genes = group_df[gene_id_column].astype(str).dropna().tolist()
        if top_n_markers is not None:
            genes = genes[:top_n_markers]
        marker_lists[group] = genes

    if drop_shared_markers:
        all_markers = pd.Series(
            [gene for genes in marker_lists.values() for gene in genes], dtype="object"
        )
        gene_counts = all_markers.value_counts() if not all_markers.empty else {}
        marker_lists = {
            group: [gene for gene in genes if gene_counts.get(gene, 0) == 1]
            for group, genes in marker_lists.items()
        }

    spatial_index = pd.Index([str(gene) for gene in spatial_gene_names])
    marker_union = pd.Index(
        sorted(set(gene for genes in marker_lists.values() for gene in genes))
    ).intersection(spatial_index)
    return {
        "groups": tuple(marker_lists),
        "marker_lists": marker_lists,
        "marker_union": marker_union,
    }


def _build_wjaccard_weights(
    markers_df,
    gene_id_column="names",
    weight_column="logfoldchanges",
    lambda_param=0.25,
    spatial_gene_names=None,
):
    use_precalculated_weights = weight_column is not None and weight_column in markers_df.columns
    spatial_set = None if spatial_gene_names is None else {str(gene) for gene in spatial_gene_names}
    group_weights = {}
    for group in markers_df["group"].drop_duplicates().astype(str).tolist():
        cluster_df = markers_df.loc[markers_df["group"].astype(str) == group]
        if spatial_set is not None:
            cluster_df = cluster_df.loc[cluster_df[gene_id_column].astype(str).isin(spatial_set)]
        cluster_genes = cluster_df[gene_id_column].reset_index(drop=True).astype(str)
        if use_precalculated_weights:
            cluster_weight_values = pd.to_numeric(
                cluster_df[weight_column].reset_index(drop=True), errors="coerce"
            )
            max_weight = cluster_weight_values.max()
            if max_weight > 0:
                weights = cluster_weight_values / max_weight
            else:
                weights = cluster_weight_values
            cluster_weights = pd.Series(weights.values, index=cluster_genes)
        else:
            n_genes = len(cluster_genes)
            weights = (
                np.exp(-lambda_param * np.arange(n_genes))
                if n_genes > 0
                else np.array([])
            )
            cluster_weights = pd.Series(weights, index=cluster_genes)
        if cluster_weights.index.has_duplicates:
            cluster_weights = cluster_weights.groupby(level=0).max()
        group_weights[group] = cluster_weights
    return group_weights


def _build_phase2_cache(
    markers_df,
    spatial_gene_names,
    method,
    gene_id_column="names",
    similarity_by_column="logfoldchanges",
    weight_column="logfoldchanges",
    **kwargs,
) -> _Phase2Cache:
    _assert_phase2_method_categories()
    validate_choice(method, SIMILARITY_METHODS, "method")
    spatial_genes = tuple(str(gene) for gene in spatial_gene_names)
    spatial_gene_set = set(spatial_genes)
    spatial_positions = {gene: idx for idx, gene in enumerate(spatial_genes)}
    groups = tuple(markers_df["group"].drop_duplicates().astype(str).tolist())

    raw_group_genes = {}
    for group in groups:
        group_df = markers_df.loc[markers_df["group"].astype(str) == group]
        raw_group_genes[group] = _unique_in_order(
            group_df[gene_id_column].dropna().astype(str).tolist()
        )

    marker_union = tuple(
        gene
        for gene in spatial_genes
        if any(gene in set(genes) for genes in raw_group_genes.values())
    )
    uses_full_gene_row = method in FULL_GENE_ROW_METHODS

    ucell_signatures = None
    auc_signatures = None
    if method == "ucell":
        ucell_signatures = _build_ucell_signatures(
            markers_df,
            gene_universe=spatial_gene_names,
            gene_id_column=gene_id_column,
            celltype="group",
            marker_role_column=kwargs.get("ucell_marker_role_column", "marker_role"),
            weight_column=weight_column,
            top_n_markers=kwargs.get("top_n_markers", None),
            drop_shared_markers=kwargs.get("drop_shared_markers", False),
        )
        marker_union = tuple(str(gene) for gene in ucell_signatures["marker_union"])
    elif method == "auc":
        auc_signatures = _build_auc_signatures(
            markers_df,
            spatial_gene_names=spatial_gene_names,
            gene_id_column=gene_id_column,
            weight_column=weight_column,
            top_n_markers=kwargs.get("top_n_markers", None),
            drop_shared_markers=kwargs.get("drop_shared_markers", False),
        )
        marker_union = tuple(str(gene) for gene in auc_signatures["marker_union"])

    expression_genes = spatial_genes if uses_full_gene_row else marker_union
    expression_positions = {gene: idx for idx, gene in enumerate(expression_genes)}

    group_genes = {}
    group_gene_positions = {}
    group_reference_values = {}
    group_marker_sets = {}
    for group in groups:
        genes = tuple(gene for gene in raw_group_genes[group] if gene in spatial_gene_set)
        group_genes[group] = genes
        group_gene_positions[group] = tuple(
            expression_positions[gene] for gene in genes if gene in expression_positions
        )
        group_marker_sets[group] = set(genes)
        if method in {"correlation", "cosine", "euclidean"} and similarity_by_column in markers_df.columns:
            group_df = markers_df.loc[markers_df["group"].astype(str) == group]
            series = pd.Series(
                pd.to_numeric(group_df[similarity_by_column], errors="coerce").values,
                index=group_df[gene_id_column].astype(str).values,
            )
            group_reference_values[group] = series.reindex(expression_genes)
        else:
            group_reference_values[group] = None

    group_weights = (
        _build_wjaccard_weights(
            markers_df,
            gene_id_column=gene_id_column,
            weight_column=weight_column,
            lambda_param=kwargs.get("lambda_param", 0.25),
            spatial_gene_names=spatial_gene_names,
        )
        if method == "wjaccard"
        else {}
    )
    diagnostics = {
        "method": method,
        "extraction_strategy": "full_gene_universe" if uses_full_gene_row else "marker_union",
        "n_total_spatial_genes": int(len(spatial_genes)),
        "n_expression_genes": int(len(expression_genes)),
        "n_marker_union_genes": int(len(marker_union)),
        "n_groups": int(len(groups)),
        "marker_cache_used": True,
    }
    return _Phase2Cache(
        method=method,
        groups=groups,
        expression_genes=expression_genes,
        marker_union=marker_union,
        group_genes=group_genes,
        group_gene_positions=group_gene_positions,
        group_reference_values=group_reference_values,
        group_marker_sets=group_marker_sets,
        group_weights=group_weights,
        auc_signatures=auc_signatures,
        ucell_signatures=ucell_signatures,
        uses_full_gene_row=uses_full_gene_row,
        diagnostics=diagnostics,
    )


def _matrix_to_csr_or_dense(matrix):
    if issparse(matrix):
        return matrix.tocsr(copy=False), True
    return np.asarray(matrix), False


def _iter_phase2_rows(table, obs_names, gene_names):
    obs_names = pd.Index(obs_names)
    gene_names = pd.Index(gene_names)
    if len(obs_names) == 0:
        return
    if len(gene_names) == 0:
        for obs_name in obs_names:
            yield obs_name, pd.Series(dtype=float, index=gene_names, name=obs_name)
        return

    try:
        subset = table[obs_names, gene_names]
        matrix, is_sparse = _matrix_to_csr_or_dense(subset.X)
    except Exception as exc:
        raise TypeError(
            "Could not extract Phase 2 expression rows from the AnnData-like table."
        ) from exc

    if is_sparse:
        for row_idx, obs_name in enumerate(obs_names):
            values = matrix.getrow(row_idx).toarray().ravel()
            yield obs_name, pd.Series(values, index=gene_names, name=obs_name)
    else:
        for row_idx, obs_name in enumerate(obs_names):
            values = np.asarray(matrix[row_idx, :]).reshape(-1)
            yield obs_name, pd.Series(values, index=gene_names, name=obs_name)


def _validate_finite_nonnegative(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite number greater than or equal to 0.")
    return value


def _select_unique_winner(
    row,
    minimum_evidence=0.0,
    tie_tolerance=1e-12,
):
    """Return the unique top label, or NaN for weak/tied/uninformative rows."""
    minimum_evidence = _validate_finite_nonnegative(
        minimum_evidence, "minimum_evidence"
    )
    tie_tolerance = _validate_finite_nonnegative(tie_tolerance, "tie_tolerance")
    scores = pd.to_numeric(row, errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite_scores = scores.dropna()
    if finite_scores.empty:
        return np.nan

    sorted_scores = finite_scores.sort_values(ascending=False, kind="mergesort")
    top_score = float(sorted_scores.iloc[0])
    if len(sorted_scores) == 1:
        return sorted_scores.index[0] if top_score > minimum_evidence else np.nan

    second_score = float(sorted_scores.iloc[1])
    if top_score <= minimum_evidence:
        return np.nan
    if top_score - second_score <= tie_tolerance:
        return np.nan
    return sorted_scores.index[0]


def _validate_optional_positive_integer(value, name):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError(f"{name} must be None or an integer greater than or equal to 1.")
    return int(value)


def _validate_nonempty_string(value, name):
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _ordered_unique(values):
    seen = set()
    ordered = []
    for value in values:
        value = str(value)
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _sort_ucell_role_markers(group_role_df, gene_id_column, weight_column):
    work = group_role_df.copy()
    if weight_column in work.columns:
        work["_ucell_sort_weight"] = pd.to_numeric(work[weight_column], errors="coerce")
        if "_ucell_role" in work.columns and (work["_ucell_role"] == "negative").all():
            work["_ucell_sort_weight"] = work["_ucell_sort_weight"].abs()
        work = work.sort_values(
            "_ucell_sort_weight",
            ascending=False,
            kind="stable",
            na_position="last",
        )
    elif "marker_rank" in work.columns:
        work["_ucell_sort_rank"] = pd.to_numeric(work["marker_rank"], errors="coerce")
        work = work.sort_values(
            "_ucell_sort_rank",
            ascending=True,
            kind="stable",
            na_position="last",
        )
    return _ordered_unique(work[gene_id_column].dropna().astype(str).tolist())


def _build_ucell_signatures(
    markers_df,
    gene_universe,
    gene_id_column="names",
    celltype="group",
    marker_role_column="marker_role",
    weight_column="logfoldchanges",
    top_n_markers=None,
    drop_shared_markers=False,
) -> dict:
    """Build deterministic UCell-like marker signatures once per Phase 2 run."""
    top_n_markers = _validate_optional_positive_integer(
        top_n_markers, "top_n_markers"
    )
    if not isinstance(drop_shared_markers, bool):
        raise ValueError("drop_shared_markers must be a bool.")
    _validate_nonempty_string(marker_role_column, "ucell_marker_role_column")
    if not isinstance(markers_df, pd.DataFrame):
        raise TypeError("markers_df must be a pandas DataFrame.")

    raw_columns = resolve_marker_columns(
        markers_df,
        MarkerSchema(group_col=celltype, gene_col=gene_id_column),
    )
    if (
        "group" in raw_columns
        and "names" in raw_columns
        and marker_role_column in markers_df.columns
    ):
        raw_roles = markers_df[marker_role_column]
        raw_roles = (
            raw_roles.where(~raw_roles.isna(), "positive")
            .astype(str)
            .str.strip()
            .str.casefold()
            .replace("", "positive")
        )
        raw_role_df = pd.DataFrame(
            {
                "group": markers_df[raw_columns["group"]].astype(str),
                "names": markers_df[raw_columns["names"]].astype(str),
                "_ucell_role": raw_roles,
            }
        ).reset_index(drop=True)
        raw_role_df = raw_role_df[
            raw_role_df["_ucell_role"].isin(["positive", "identity", "negative"])
        ]
        for (group, gene), roles in raw_role_df.groupby(
            [raw_role_df["group"], raw_role_df["names"]]
        )["_ucell_role"]:
            role_set = set(roles)
            if role_set.intersection({"positive", "identity"}) and "negative" in role_set:
                raise ValueError(
                    f"Markers for group {group!r} contain genes marked as both "
                    f"positive and negative: {[gene]}."
                )

    work = standardize_marker_dataframe(
        markers_df,
        schema=MarkerSchema(group_col=celltype, gene_col=gene_id_column),
        gene_universe=None,
        top_n_genes=None,
        sort_by_column=None,
        log2fc_min=-np.inf,
        pval_cutoff=1.0,
        source=None,
    )
    work = work.reset_index(drop=True)
    gene_id_column = "names"
    celltype = "group"
    groups = work[celltype].drop_duplicates().astype(str).tolist()
    gene_universe_set = {str(gene) for gene in gene_universe}
    work = work.loc[work[gene_id_column].astype(str).isin(gene_universe_set)].copy()

    roles, role_column = normalize_marker_roles(
        work,
        marker_role_column=marker_role_column,
        fill_missing_column=True,
    )
    work["_ucell_role"] = roles

    positive = {}
    negative = {}
    counts = {}
    for group in groups:
        group_df = work[work[celltype].astype(str) == str(group)]
        pos_df = group_df[group_df["_ucell_role"].isin(["positive", "identity"])]
        neg_df = group_df[group_df["_ucell_role"] == "negative"]
        pos_genes = _sort_ucell_role_markers(pos_df, gene_id_column, weight_column)
        neg_genes = _sort_ucell_role_markers(neg_df, gene_id_column, weight_column)
        if top_n_markers is not None:
            pos_genes = pos_genes[:top_n_markers]
            neg_genes = neg_genes[:top_n_markers]
        conflicts = sorted(set(pos_genes).intersection(neg_genes))
        if conflicts:
            raise ValueError(
                f"Markers for group {group!r} contain genes marked as both "
                f"positive and negative: {conflicts}."
            )
        positive[group] = list(pos_genes)
        negative[group] = list(neg_genes)

    if drop_shared_markers:
        all_positive = pd.Series(
            [gene for genes in positive.values() for gene in genes], dtype="object"
        )
        shared_counts = all_positive.value_counts() if not all_positive.empty else {}
        positive = {
            group: [gene for gene in genes if shared_counts.get(gene, 0) == 1]
            for group, genes in positive.items()
        }

    gene_universe = pd.Index([str(gene) for gene in gene_universe])
    gene_positions = {gene: idx for idx, gene in enumerate(gene_universe)}
    positive_indexed = {}
    negative_indexed = {}
    marker_genes = set()
    for group in groups:
        pos = [gene for gene in positive[group] if gene in gene_positions]
        neg = [gene for gene in negative[group] if gene in gene_positions]
        positive_indexed[group] = pd.Index(pos)
        negative_indexed[group] = pd.Index(neg)
        marker_genes.update(pos)
        marker_genes.update(neg)
        counts[group] = {
            "n_positive": int(len(pos)),
            "n_negative": int(len(neg)),
        }
    marker_union = pd.Index([gene for gene in gene_universe if gene in marker_genes])
    return {
        "groups": groups,
        "positive": positive_indexed,
        "negative": negative_indexed,
        "marker_union": marker_union,
        "counts": counts,
    }


def _rank_u_signature_score(
    ranks,
    signature_genes,
    max_rank,
) -> float:
    n = len(signature_genes)
    if n == 0:
        return 0.0
    if max_rank < n:
        return 0.0
    max_u = n * (max_rank + 1) - n * (n + 1) / 2
    if max_u <= 0:
        return 0.0
    u_value = ranks.loc[signature_genes].sum() - n * (n + 1) / 2
    score = 1 - (u_value / max_u)
    if not np.isfinite(score):
        return 0.0
    return float(np.clip(score, 0.0, 1.0))


def function_row_ucell(row, markers_df=None, **kwargs):
    """UCell-like normalized rank-U signature score for one spatial location."""
    phase2_cache = kwargs.get("phase2_cache")
    signatures = (
        phase2_cache.ucell_signatures
        if phase2_cache is not None and phase2_cache.ucell_signatures is not None
        else kwargs.get("ucell_signatures")
    )
    if signatures is None:
        signatures = _build_ucell_signatures(
            markers_df,
            gene_universe=row.index,
            gene_id_column=kwargs.get("gene_id_column", "names"),
            celltype=kwargs.get("celltype", "group"),
            marker_role_column=kwargs.get("ucell_marker_role_column", "marker_role"),
            weight_column=kwargs.get("weight_column", "logfoldchanges"),
            top_n_markers=kwargs.get("top_n_markers", None),
            drop_shared_markers=kwargs.get("drop_shared_markers", False),
        )
    groups = signatures["groups"]
    if phase2_cache is not None:
        groups = _candidate_groups_for_cache(phase2_cache, kwargs)
    marker_union = signatures["marker_union"]
    if len(marker_union) == 0:
        return {group: 0.0 for group in groups}

    min_markers = _validate_optional_positive_integer(
        kwargs.get("min_markers", 3), "min_markers"
    )
    expression_threshold = _validate_finite_nonnegative(
        kwargs.get("expression_threshold", 0.0), "expression_threshold"
    )
    recovery_power = _validate_finite_nonnegative(
        kwargs.get("recovery_power", 1.0), "recovery_power"
    )
    ucell_max_rank = _validate_optional_positive_integer(
        kwargs.get("ucell_max_rank", None), "ucell_max_rank"
    )
    ucell_negative_weight = _validate_finite_nonnegative(
        kwargs.get("ucell_negative_weight", 1.0), "ucell_negative_weight"
    )

    expression = pd.to_numeric(row.reindex(marker_union), errors="coerce")
    expression = expression.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    expression = expression.where(expression > expression_threshold, 0.0)
    if expression.max() <= 0:
        return {group: 0.0 for group in groups}
    if len(expression) > 1 and expression.max() - expression.min() <= 1e-12:
        return {group: 0.0 for group in groups}

    effective_max_rank = (
        len(marker_union)
        if ucell_max_rank is None
        else min(int(ucell_max_rank), len(marker_union))
    )
    ranks = expression.rank(method="average", ascending=False)
    capped_ranks = ranks.clip(upper=effective_max_rank + 1)
    scores = {}
    for group in groups:
        positive_genes = signatures["positive"][group]
        negative_genes = signatures["negative"][group]
        if len(positive_genes) < min_markers:
            scores[group] = 0.0
            continue
        detected_positive = expression.loc[positive_genes] > 0
        n_detected_positive = int(detected_positive.sum())
        if n_detected_positive < min_markers:
            scores[group] = 0.0
            continue
        positive_score = _rank_u_signature_score(
            capped_ranks,
            positive_genes,
            effective_max_rank,
        )
        if len(negative_genes) == 0:
            negative_score = 0.0
        else:
            detected_negative = expression.loc[negative_genes] > 0
            if int(detected_negative.sum()) == 0:
                negative_score = 0.0
            else:
                negative_score = _rank_u_signature_score(
                    capped_ranks,
                    negative_genes,
                    effective_max_rank,
                )
        recovery = n_detected_positive / float(len(positive_genes))
        score = max(0.0, positive_score - ucell_negative_weight * negative_score)
        score *= recovery ** recovery_power
        scores[group] = float(np.clip(score if np.isfinite(score) else 0.0, 0.0, 1.0))
    return scores



def function_row_spearman(row, markers_df,**kwargs):
    phase2_cache = kwargs.get("phase2_cache")
    if phase2_cache is not None:
        a = {}
        for c in _candidate_groups_for_cache(phase2_cache, kwargs):
            vector_series = phase2_cache.group_reference_values.get(c)
            if vector_series is None:
                a[c] = 0.0
                continue
            valid_mask = ~vector_series.isna() & ~row.isna()
            if int(valid_mask.sum()) < 2:
                a[c] = 0.0
                continue
            if (
                row[valid_mask].nunique(dropna=True) < 2
                or vector_series[valid_mask].nunique(dropna=True) < 2
            ):
                a[c] = 0.0
                continue
            t = (row[valid_mask] != 0).sum()
            if t == 0:
                a[c] = 0.0
            else:
                l = len(phase2_cache.group_genes.get(c, ()))
                sp = spearmanr(
                    row[valid_mask], vector_series[valid_mask], nan_policy="omit"
                )[0] * ((t / l) if l else 0.0)
                a[c] = sp if np.isfinite(sp) and sp > 0 else 0.0
        return a
    gene_id_column=kwargs.get("gene_id_column","names")
    similarity_by_column=kwargs.get("similarity_by_column","logfoldchanges")
    #penalty_param=kwargs.get("penalty_param",0.5)

    a = {}
    for c in markers_df.index.unique():
        vector_series = pd.Series(markers_df[[gene_id_column,similarity_by_column]].loc[[c]][similarity_by_column].values, index=markers_df[[gene_id_column, similarity_by_column]].loc[[c]][gene_id_column].values)
        l = len(vector_series)
        vector_series = vector_series.reindex(row.index, fill_value=np.nan)
        valid_mask = ~vector_series.isna() & ~row.isna()
        if int(valid_mask.sum()) < 2:
            a[c] = 0.0
            continue
        if (
            row[valid_mask].nunique(dropna=True) < 2
            or vector_series[valid_mask].nunique(dropna=True) < 2
        ):
            a[c] = 0.0
            continue
        t = (row[valid_mask] != 0).sum()
        if t == 0:  # No valid pairs
            a[c] = 0.0
        else:
            #sp = (spearmanr(row[valid_mask], vector_series[valid_mask], nan_policy="omit")[0])*((t/l)**penalty_param)
            sp = (spearmanr(row[valid_mask], vector_series[valid_mask], nan_policy="omit")[0])*((t/l))
            a[c] = sp if np.isfinite(sp) and sp > 0 else 0.0
    return a





def function_row_cosine(row, markers_df,**kwargs):
    phase2_cache = kwargs.get("phase2_cache")
    if phase2_cache is not None:
        a = {}
        for c in _candidate_groups_for_cache(phase2_cache, kwargs):
            vector_series = phase2_cache.group_reference_values.get(c)
            if vector_series is None:
                a[c] = 0.0
                continue
            valid_mask = ~vector_series.isna() & ~row.isna()
            if int(valid_mask.sum()) < 2:
                a[c] = 0.0
                continue
            vector_values = min_max_scale(vector_series[valid_mask])
            if np.linalg.norm(vector_values.to_numpy(dtype=float)) == 0:
                a[c] = 0.0
                continue
            t = (row[valid_mask] != 0).sum()
            if t == 0:
                a[c] = 0.0
            else:
                score = 1 - cosine(row[valid_mask], vector_values)
                a[c] = float(score) if np.isfinite(score) else 0.0
        return a
    gene_id_column=kwargs.get("gene_id_column","names")
    similarity_by_column=kwargs.get("similarity_by_column","logfoldchanges")
    #penalty_param=kwargs.get("penalty_param",0)
    


    a = {}
    for c in markers_df.index.unique():
        vector_series = pd.Series(markers_df[[gene_id_column,similarity_by_column]].loc[[c]][similarity_by_column].values, index=markers_df[[gene_id_column, similarity_by_column]].loc[[c]][gene_id_column].values)
        l = len(vector_series)
        vector_series = vector_series.reindex(row.index, fill_value=np.nan)
        valid_mask = ~vector_series.isna() & ~row.isna()
        if int(valid_mask.sum()) < 2:
            a[c] = 0.0
            continue
        vector_values = min_max_scale(vector_series[valid_mask])
        if np.linalg.norm(vector_values.to_numpy(dtype=float)) == 0:
            a[c] = 0.0
            continue
        t = (row[valid_mask] != 0).sum()
        if t == 0:  # No valid pairs
            a[c] = 0.0
        else:
            #a[c] = (1 - cosine(row[valid_mask], vector_series[valid_mask]))*((t/l)**penalty_param) #penalize the cosine similarity by the fraction of valid pairs
            score = 1 - cosine(row[valid_mask], vector_values)
            a[c] = float(score) if np.isfinite(score) else 0.0
    return a




def function_row_euclidean(row, markers_df, **kwargs):
    phase2_cache = kwargs.get("phase2_cache")
    if phase2_cache is not None:
        a = {}
        for c in _candidate_groups_for_cache(phase2_cache, kwargs):
            vector_series = phase2_cache.group_reference_values.get(c)
            if vector_series is None:
                a[c] = 0.0
                continue
            valid_mask = ~vector_series.isna() & ~row.isna()
            if int(valid_mask.sum()) < 1:
                a[c] = 0.0
                continue
            vector_values = min_max_scale(vector_series[valid_mask])
            t = (row[valid_mask] != 0).sum()
            if t == 0:
                a[c] = 0.0
            else:
                distance_val = euclidean(row[valid_mask], vector_values)
                similarity_val = 1 / (1 + distance_val)
                a[c] = float(similarity_val) if np.isfinite(similarity_val) else 0.0
        return a
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
        valid_mask = ~vector_series.isna() & ~row.isna()
        if int(valid_mask.sum()) < 1:
            a[c] = 0.0
            continue
        vector_values = min_max_scale(vector_series[valid_mask])
        
        # Number of non-zero valid entries
        t = (row[valid_mask] != 0).sum()
        
        if t == 0:  # No valid pairs
            a[c] = 0.0
        else:
            distance_val = euclidean(row[valid_mask], vector_values)
            
            # Convert Euclidean distance to similarity in [0,1]: higher distance -> lower similarity
            similarity_val = 1 / (1 + distance_val)
            
            # Apply the penalty factor
            #a[c] = similarity_val * ((t / l) ** penalty_param)
            a[c] = float(similarity_val) if np.isfinite(similarity_val) else 0.0
    
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


def function_row_auc(row, markers_df, **kwargs):
    """
    AUROC-style similarity between a spot (row) and each cluster's marker set.

    For each cluster c:
      - positives = marker genes of c present in row.index
      - negatives = all other genes in row.index
      - score = AUROC that positives have higher expression than negatives

    Parameters
    ----------
    row : pandas.Series
        Expression values for one spot. Index must be gene IDs.
    markers_df : pandas.DataFrame
        DataFrame with markers. Index = cluster/cell type, and a column with
        gene IDs (e.g. 'names').
    gene_id_column : str, in kwargs
        Name of the column in markers_df that holds gene IDs.
    min_markers : int, in kwargs, optional
        Minimum number of markers that must be present in the spot to compute
        a score. Otherwise returns fallback (default 0.5).
    fallback_auc : float, in kwargs, optional
        Value to use when AUC is undefined (e.g. too few markers or no negatives).

    Returns
    -------
    dict
        {cluster_label: auc_score}
    """
    gene_id_column = kwargs.get("gene_id_column", "names")
    min_markers = kwargs.get("min_markers", 3)
    fallback_auc = kwargs.get("fallback_auc", 0.5)

    # Rank genes once per row (1 = lowest expression, N = highest)
    ranks = row.rank(method="average", ascending=True)
    N = len(ranks)

    # Pre-compute for safety
    if N < 2:
        # Degenerate case: only one gene
        return {c: fallback_auc for c in markers_df.index.unique()}

    scores = {}

    # Iterate clusters / groups in markers_df
    for c in markers_df.index.unique():
        # Get marker genes for this cluster
        gene_list = markers_df.loc[[c], gene_id_column].astype(str)
        # Intersect with genes present in this spot
        genes_pos = pd.Index(gene_list).intersection(ranks.index)

        n_pos = len(genes_pos)
        n_neg = N - n_pos

        # Not enough markers or no negatives -> undefined AUC
        if (n_pos < min_markers) or (n_neg <= 0):
            scores[c] = fallback_auc
            continue

        # Sum of ranks of positives
        sum_ranks = ranks[genes_pos].sum()

        # Wilcoxon U statistic
        U = sum_ranks - n_pos * (n_pos + 1) / 2.0

        # Normalize to AUC
        auc = U / (n_pos * n_neg)

        # Numerical safety: clamp to [0,1]
        if not np.isfinite(auc):
            auc = fallback_auc
        else:
            auc = max(0.0, min(1.0, auc))

        scores[c] = auc

    return scores

import pandas as pd
import numpy as np

def function_row_auc_specific(row, markers_df, **kwargs):
    """
    High-specificity AUROC scoring. 
    Filters out noise and focuses on top markers to reduce false positives.
    """
    gene_id_column = kwargs.get("gene_id_column", "names")
    min_markers = kwargs.get("min_markers", 3)
    fallback_auc = kwargs.get("fallback_auc", 0.5)
    
    # NEW 1: Expression Threshold
    # Only rank genes that have biologically relevant expression.
    # Everything below this is treated as 'negative' background.
    expr_threshold = kwargs.get("expression_threshold", 0.1)
    
    # NEW 2: Limit to Top N Markers
    # Only look for the top N most specific markers per cluster.
    top_n = kwargs.get("top_n_markers", None) 

    # Filter the row first
    valid_genes = row[row > expr_threshold]
    
    # If the spot is empty after filtering, return fallback
    if len(valid_genes) < 2:
         return {c: fallback_auc for c in markers_df.index.unique()}

    # Rank only the expressed genes
    # Genes not in valid_genes are implicitly rank 0 (or below these ranks)
    ranks = valid_genes.rank(method="average", ascending=True)
    N = len(row) # The universe is still the total gene space of the spot

    scores = {}

    for c in markers_df.index.unique():
        # Get marker genes
        gene_list = markers_df.loc[[c], gene_id_column].astype(str)
        
        # NEW 2 Implementation: Slice the top N markers
        if top_n is not None:
            gene_list = gene_list.iloc[:top_n]

        # Intersect with valid (above threshold) genes
        genes_pos = pd.Index(gene_list).intersection(ranks.index)
        
        n_pos_found = len(genes_pos)
        n_total_markers = len(gene_list)

        # STRICTER CHECK: 
        # You might want to fail if we found fewer than X% of the top markers
        if n_pos_found < min_markers:
            scores[c] = fallback_auc
            continue
        
        # Calculate AUC components
        # Note: We treat genes below threshold as having rank 0. 
        # However, for standard AUC calculation in this specific context (Genes vs Background),
        # we usually only care about the ranks of the DETECTED markers relative to DETECTED background.
        
        # Rank Sum of detected markers
        sum_ranks = ranks[genes_pos].sum()
        
        # The number of negatives is the number of valid genes minus the markers found
        n_neg = len(valid_genes) - n_pos_found

        if n_neg <= 0:
             scores[c] = fallback_auc
             continue

        # Wilcoxon U
        U = sum_ranks - n_pos_found * (n_pos_found + 1) / 2.0
        auc = U / (n_pos_found * n_neg)

        # NEW 3: Recovery Penalty (Optional)
        # If you want to penalize spots that only have 3 out of 50 markers,
        # multiply the AUC by the fraction of markers recovered.
        # Uncomment the line below to enable:
        # auc = auc * (n_pos_found / n_total_markers)

        if not np.isfinite(auc):
            auc = fallback_auc
        else:
            auc = max(0.0, min(1.0, auc))

        scores[c] = auc

    return scores


def function_row_auc_specific_v2(row, markers_df, **kwargs):
    """
    Marker-union AUROC score with:
      - missing-marker penalty
      - centered AUC evidence
      - recovery penalty
      - optional top-N marker restriction
      - optional marker-weight sorting

    Returns scores in [0, 1], where 0 means no evidence.
    """
    import numpy as np
    import pandas as pd

    phase2_cache = kwargs.get("phase2_cache")
    if phase2_cache is not None and phase2_cache.auc_signatures is not None:
        signatures = phase2_cache.auc_signatures
        min_markers = kwargs.get("min_markers", 3)
        fallback_score = kwargs.get("fallback_auc", 0.0)
        expr_threshold = kwargs.get("expression_threshold", 0.0)
        recovery_power = kwargs.get("recovery_power", 1.0)
        center_auc = kwargs.get("center_auc", True)
        marker_union = signatures["marker_union"]
        marker_lists = signatures["marker_lists"]
        groups = _candidate_groups_for_cache(phase2_cache, kwargs)
        if len(marker_union) < 2:
            return {c: fallback_score for c in groups}
        x = row.reindex(marker_union).fillna(0.0).astype(float)
        if expr_threshold is not None and expr_threshold > 0:
            x = x.where(x > expr_threshold, 0.0)
        ranks = x.rank(method="average", ascending=True)
        scores = {}
        for c in groups:
            pos = pd.Index(marker_lists[c]).intersection(marker_union)
            n_pos = len(pos)
            n_neg = len(marker_union) - n_pos
            if n_pos < min_markers or n_neg <= 0:
                scores[c] = fallback_score
                continue
            detected = row.reindex(pos).fillna(0.0) > expr_threshold
            n_detected = int(detected.sum())
            if n_detected < min_markers:
                scores[c] = fallback_score
                continue
            sum_ranks = ranks.loc[pos].sum()
            U = sum_ranks - n_pos * (n_pos + 1) / 2.0
            auc = U / (n_pos * n_neg)
            if not np.isfinite(auc):
                scores[c] = fallback_score
                continue
            auc = float(np.clip(auc, 0.0, 1.0))
            score = max(0.0, 2.0 * (auc - 0.5)) if center_auc else auc
            recovery = n_detected / float(n_pos)
            score *= recovery ** recovery_power
            scores[c] = float(np.clip(score, 0.0, 1.0))
        return scores

    gene_id_column = kwargs.get("gene_id_column", "names")
    weight_column = kwargs.get("weight_column", "logfoldchanges")
    min_markers = kwargs.get("min_markers", 3)

    # Important: for centered AUC, fallback should be 0, not 0.5
    fallback_score = kwargs.get("fallback_auc", 0.0)

    expr_threshold = kwargs.get("expression_threshold", 0.0)
    top_n = kwargs.get("top_n_markers", None)
    recovery_power = kwargs.get("recovery_power", 1.0)
    center_auc = kwargs.get("center_auc", True)
    drop_shared_markers = kwargs.get("drop_shared_markers", False)

    # Build cluster -> marker list
    marker_lists = {}
    for c in markers_df.index.unique():
        cdf = markers_df.loc[[c]].copy()

        # Ensure top_n means top by marker strength, not arbitrary row order
        if weight_column in cdf.columns:
            cdf = cdf.sort_values(weight_column, ascending=False)

        genes = cdf[gene_id_column].astype(str).dropna().tolist()

        if top_n is not None:
            genes = genes[:top_n]

        marker_lists[c] = genes

    # Optionally remove markers shared across multiple cell types
    if drop_shared_markers:
        all_marker_series = pd.Series(
            [g for genes in marker_lists.values() for g in genes]
        )
        gene_counts = all_marker_series.value_counts()
        marker_lists = {
            c: [g for g in genes if gene_counts.get(g, 0) == 1]
            for c, genes in marker_lists.items()
        }

    # Use marker union as the AUC universe
    marker_union = pd.Index(
        sorted(set(g for genes in marker_lists.values() for g in genes))
    ).intersection(row.index)

    if len(marker_union) < 2:
        return {c: fallback_score for c in marker_lists}

    # Do not drop low-expression genes. Keep them so absent markers are penalized.
    x = row.reindex(marker_union).fillna(0.0).astype(float)

    # Optional thresholding: set low values to 0, but keep them in ranking universe
    if expr_threshold is not None and expr_threshold > 0:
        x = x.where(x > expr_threshold, 0.0)

    ranks = x.rank(method="average", ascending=True)

    scores = {}

    for c, genes in marker_lists.items():
        pos = pd.Index(genes).intersection(marker_union)
        n_pos = len(pos)
        n_neg = len(marker_union) - n_pos

        if n_pos < min_markers or n_neg <= 0:
            scores[c] = fallback_score
            continue

        detected = (row.reindex(pos).fillna(0.0) > expr_threshold)
        n_detected = int(detected.sum())

        if n_detected < min_markers:
            scores[c] = fallback_score
            continue

        sum_ranks = ranks.loc[pos].sum()
        U = sum_ranks - n_pos * (n_pos + 1) / 2.0
        auc = U / (n_pos * n_neg)

        if not np.isfinite(auc):
            scores[c] = fallback_score
            continue

        auc = float(np.clip(auc, 0.0, 1.0))

        if center_auc:
            score = max(0.0, 2.0 * (auc - 0.5))
        else:
            score = auc

        recovery = n_detected / float(n_pos)
        score *= recovery ** recovery_power

        scores[c] = float(np.clip(score, 0.0, 1.0))

    return scores

def function_row_jaccard(row, markers_df, **kwargs):
    phase2_cache = kwargs.get("phase2_cache")
    if phase2_cache is not None:
        row_set = set(row[row > 0].sort_values(ascending=False).index)
        return {
            c: (
                0.0
                if len(row_set.union(phase2_cache.group_marker_sets[c])) == 0
                else len(row_set.intersection(phase2_cache.group_marker_sets[c]))
                / len(row_set.union(phase2_cache.group_marker_sets[c]))
            )
            for c in _candidate_groups_for_cache(phase2_cache, kwargs)
        }
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
    phase2_cache = kwargs.get("phase2_cache")
    if phase2_cache is not None:
        row_set = set(row[row > 0].sort_values(ascending=False).index)
        scores = {}
        for c in _candidate_groups_for_cache(phase2_cache, kwargs):
            vector_set = phase2_cache.group_marker_sets[c]
            denominator = min(len(row_set), len(vector_set))
            scores[c] = (
                0.0
                if denominator == 0
                else len(row_set.intersection(vector_set)) / denominator
            )
        return scores
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
    phase2_cache = kwargs.get("phase2_cache")
    if phase2_cache is not None:
        row_set = set(row[row > 0].sort_values(ascending=False).index)
        return {
            c: row_set.intersection(phase2_cache.group_marker_sets[c])
            for c in _candidate_groups_for_cache(phase2_cache, kwargs)
        }
    a = {}
    gene_id_column=kwargs.get("gene_id_column")
    
    for c in markers_df.index.unique():
        row_set = set(row[row > 0].sort_values(ascending=False).index) #non-zero values
        vector_set = set(markers_df.loc[[c]][gene_id_column].values)
        
        # Calculate intersection
        a[c] = row_set.intersection(vector_set)
    return a

def function_row_sum(row, markers_df, **kwargs):
    phase2_cache = kwargs.get("phase2_cache")
    if phase2_cache is not None:
        return {
            c: float(row.iloc[list(phase2_cache.group_gene_positions[c])].fillna(0.0).sum())
            if len(phase2_cache.group_gene_positions[c]) > 0
            else 0.0
            for c in _candidate_groups_for_cache(phase2_cache, kwargs)
        }
    a = {}
    gene_id_column=kwargs.get("gene_id_column")
    
    for c in markers_df.index.unique():
        #row_set = set(row[row > 0].sort_values(ascending=False).index) #non-zero values
        vector_set = markers_df.loc[[c]][gene_id_column].values
        
        
        # Calculate intersection and union
        #a[c] = row_set.intersection(vector_set)
        values = row.reindex(vector_set)
        a[c] = float(values.fillna(0.0).sum())
    return a

def function_row_mean(row, markers_df, **kwargs):
    phase2_cache = kwargs.get("phase2_cache")
    if phase2_cache is not None:
        scores = {}
        for c in _candidate_groups_for_cache(phase2_cache, kwargs):
            positions = phase2_cache.group_gene_positions[c]
            if len(positions) == 0:
                scores[c] = 0.0
                continue
            values = row.iloc[list(positions)]
            scores[c] = 0.0 if values.notna().sum() == 0 else float(values.fillna(0.0).mean())
        return scores
    a = {}
    gene_id_column=kwargs.get("gene_id_column")
    
    for c in markers_df.index.unique():
        #row_set = set(row[row > 0].sort_values(ascending=False).index) #non-zero values
        vector_set = markers_df.loc[[c]][gene_id_column].values
        
        
        # Calculate intersection and union
        #a[c] = row_set.intersection(vector_set)
        values = row.reindex(vector_set)
        if len(vector_set) == 0 or values.notna().sum() == 0:
            a[c] = 0.0
        else:
            a[c] = float(values.fillna(0.0).mean())
    return a

def function_row_median(row, markers_df, **kwargs):
    phase2_cache = kwargs.get("phase2_cache")
    if phase2_cache is not None:
        scores = {}
        for c in _candidate_groups_for_cache(phase2_cache, kwargs):
            positions = phase2_cache.group_gene_positions[c]
            if len(positions) == 0:
                scores[c] = 0.0
                continue
            values = row.iloc[list(positions)]
            scores[c] = 0.0 if values.notna().sum() == 0 else float(values.fillna(0.0).median())
        return scores
    a = {}
    gene_id_column=kwargs.get("gene_id_column")
    for c in markers_df.index.unique():
        #row_set = set(row[row > 0].sort_values(ascending=False).index) #non-zero values
        vector_set = markers_df.loc[[c]][gene_id_column].values
        
        
        # Calculate intersection and union
        #a[c] = row_set.intersection(vector_set)
        values = row.reindex(vector_set)
        if len(vector_set) == 0 or values.notna().sum() == 0:
            a[c] = 0.0
        else:
            a[c] = float(values.fillna(0.0).median())
    return a


def function_row_weighted_jaccard(row, markers_df, **kwargs):
    phase2_cache = kwargs.get("phase2_cache")
    if phase2_cache is not None:
        a = {}
        target_genes = row[row > 0]
        max_expr = target_genes.max()
        if max_expr > 0:
            target_weights = target_genes / max_expr
        else:
            target_weights = target_genes
        if target_weights.index.has_duplicates:
            target_weights = target_weights.groupby(level=0).max()

        for c in _candidate_groups_for_cache(phase2_cache, kwargs):
            cluster_weights = phase2_cache.group_weights[c]
            all_genes = set(cluster_weights.index).union(target_weights.index)
            numerator = 0.0
            denominator = 0.0
            for gene in all_genes:
                a_i = cluster_weights.get(gene, 0.0)
                b_i = target_weights.get(gene, 0.0)
                numerator += min(a_i, b_i)
                denominator += max(a_i, b_i)
            a[c] = 0.0 if denominator == 0.0 else numerator / denominator
        return a

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
    if target_weights.index.has_duplicates:
        target_weights = target_weights.groupby(level=0).max()
    
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

        if cluster_weights.index.has_duplicates:
            cluster_weights = cluster_weights.groupby(level=0).max()

        
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



def add_df_to_spatialdata(sdata,df,bin_size=8,verbose=True):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    table = get_table(sdata, bin_size=bin_size)
    df_to_add = df.reindex(table.obs.index)
    table.obs.drop(columns=df_to_add.columns,inplace=True,errors='ignore')
    table.obs = pd.merge(
        table.obs,
        df_to_add,
        left_index=True,
        right_index=True,
        how="left",
        sort=False,
    )
    if verbose:
        print("DataFrame added to SpatialData object")
        print(df.columns)
    return

def test_function():
    print("Easydecon loaded!")
    print("Test function executed!")

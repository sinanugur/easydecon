import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData


def _sample_empirical_depths(depths, n, min_umi=30, jitter=0.05, rng=None):
    """
    Sample target library sizes from an empirical distribution (e.g., GT total_counts).
    Optional small multiplicative jitter to avoid exact duplicates.
    """
    rng = np.random.default_rng() if rng is None else rng
    depths = np.asarray(depths).astype(float)
    depths = depths[np.isfinite(depths) & (depths > 0)]
    if depths.size == 0:
        raise ValueError("Provided gt_depths is empty or invalid.")
    d = rng.choice(depths, size=n, replace=True)
    if jitter and jitter > 0:
        d = d * rng.lognormal(mean=0.0, sigma=float(jitter), size=n)
    d = np.maximum(d, min_umi).astype(int)
    return d


def _sample_lognormal_depths(n, mean, std, min_umi=30, rng=None):
    """
    Sample target depths from a lognormal distribution specified by linear-scale mean/std.
    """
    rng = np.random.default_rng() if rng is None else rng
    mean = float(mean)
    std = float(std)
    if mean <= 0 or std <= 0:
        raise ValueError("mean and std must be > 0 for lognormal depth sampling.")
    var = std * std
    sigma2 = np.log(1.0 + var / (mean * mean))
    sigma = np.sqrt(sigma2)
    mu = np.log(mean) - 0.5 * sigma2
    d = rng.lognormal(mean=mu, sigma=sigma, size=n)
    return np.maximum(d, min_umi).astype(int)


def _sample_nb_spot_sizes(n, mean_cells, theta=2.0, rng=None):
    """
    Zero-truncated Negative Binomial for spot/bin cell counts (heavier tail than Poisson).
    Parameterization:
      mean = mean_cells
      Var = mean + mean^2/theta
    Smaller theta => heavier tail.
    """
    rng = np.random.default_rng() if rng is None else rng
    mean_cells = float(mean_cells)
    theta = float(theta)
    if mean_cells <= 0:
        raise ValueError("mean_cells must be > 0.")
    if theta <= 0:
        raise ValueError("theta must be > 0.")

    p = theta / (theta + mean_cells)
    r = theta
    x = rng.negative_binomial(n=r, p=p, size=n)
    return np.maximum(x, 1).astype(int)


def simulate_visium_hd(
    sc_ref,
    celltype_column,
    n_spots=30000,
    # --- density / cells per bin ---
    mean_cells_per_spot=1.5,
    spot_size_model="nb",        # "poisson" or "nb"
    nb_theta=1.0,                # smaller => heavier tail (try 0.5–2.0)
    # --- depth / UMIs per bin ---
    depth_model="empirical",     # "empirical" or "lognormal"
    gt_depths=None,              # array-like of GT total_counts for empirical sampling
    target_umi_mean=450,         # used if depth_model="lognormal"
    target_umi_std=400,          # used if depth_model="lognormal"
    min_umi=30,
    depth_jitter=0.05,           # used if depth_model="empirical"
    # --- gene panel ---
    gene_list=None,
    # --- cell-type mixture prior ---
    dirichlet_alpha=0.01,        # float | array-like | "uniform" | "inverse_frequency"
    # --- technical noise model ---
    sampling_model="poisson_thin",  # "poisson_thin" (recommended) or "multinomial"
    # --- misc ---
    random_state=0,
    dtype=np.int32
):
    """
    Simulate Visium HD-like bins from a single-cell reference.

    Key changes vs your original:
      - Depths: empirical sampling from GT (or lognormal) to reproduce heavy tails.
      - Spot sizes: NB (heavier tail than Poisson) to increase multi-cell bins.
      - Sampling: Poisson thinning (more realistic variance) or multinomial.
      - Outputs: total_counts and n_genes_by_counts match Scanpy naming.

    Parameters
    ----------
    sc_ref : AnnData
        Single-cell reference with raw counts in layers['counts'] or raw.X or X.
    celltype_column : str
        Column in sc_ref.obs defining cell types.
    depth_model : {"empirical","lognormal"}
        How to sample target total_counts per bin.
    gt_depths : array-like
        Ground truth total_counts (required if depth_model="empirical").
    spot_size_model : {"poisson","nb"}
        How to sample the number of cells aggregated into each bin.
    sampling_model : {"poisson_thin","multinomial"}
        How to generate final counts from aggregated profile and target depth.

    Returns
    -------
    AnnData
        Simulated bins with X as counts (CSR), obs includes total_counts, n_genes_by_counts.
        obsm['proportions_true'] stores the sampled cell-type proportions per bin.
    """
    rng = np.random.default_rng(random_state)

    # --- 0) Ensure unique genes ---
    if not sc_ref.var_names.is_unique:
        sc_ref = sc_ref.copy()
        sc_ref.var_names_make_unique()

    # --- 1) Gene panel filtering ---
    if gene_list is not None:
        available = set(sc_ref.var_names)
        valid_genes = [g for g in gene_list if g in available]
        if len(valid_genes) == 0:
            raise ValueError("None of the genes in 'gene_list' were found in sc_ref.var_names.")
        sc_ref = sc_ref[:, valid_genes].copy()

    # --- 2) Validate and pull counts ---
    if celltype_column not in sc_ref.obs:
        raise ValueError(f"Column '{celltype_column}' not found in sc_ref.obs.")

    if "counts" in sc_ref.layers:
        X_cells = sc_ref.layers["counts"]
    elif sc_ref.raw is not None:
        X_cells = sc_ref.raw.X
    else:
        X_cells = sc_ref.X  # must be raw counts

    if not sp.issparse(X_cells):
        X_cells = sp.csr_matrix(X_cells)
    else:
        X_cells = X_cells.tocsr()

    # --- 3) Cell-type indexing ---
    ct_labels = sc_ref.obs[celltype_column].to_numpy()
    cell_types = sorted(pd.unique(ct_labels))
    K = len(cell_types)
    ct2idx = {ct: i for i, ct in enumerate(cell_types)}
    cells_by_ct = {ct2idx[ct]: np.where(ct_labels == ct)[0] for ct in cell_types}

    # --- 4) Dirichlet alpha setup ---
    if dirichlet_alpha is None or dirichlet_alpha == "uniform":
        alpha = np.ones(K, dtype=float)
    elif dirichlet_alpha == "inverse_frequency":
        counts = np.array([len(cells_by_ct[i]) for i in range(K)], dtype=float)
        freqs = (counts + 1.0) / (counts.sum() + K)
        alpha = (1.0 / freqs)
        alpha = alpha / alpha.sum() * K
    elif isinstance(dirichlet_alpha, (float, int)):
        alpha = np.full(K, float(dirichlet_alpha), dtype=float)
    else:
        alpha = np.asarray(dirichlet_alpha, dtype=float)
        if alpha.size != K:
            raise ValueError(f"dirichlet_alpha has length {alpha.size}, expected {K}.")

    # --- 5) Sample target depths (total_counts) ---
    if depth_model == "empirical":
        if gt_depths is None:
            raise ValueError("depth_model='empirical' requires gt_depths (e.g., GT adata.obs['total_counts']).")
        target_depths = _sample_empirical_depths(
            depths=gt_depths, n=n_spots, min_umi=min_umi, jitter=depth_jitter, rng=rng
        )
    elif depth_model == "lognormal":
        target_depths = _sample_lognormal_depths(
            n=n_spots, mean=target_umi_mean, std=target_umi_std, min_umi=min_umi, rng=rng
        )
    else:
        raise ValueError("depth_model must be one of: 'empirical', 'lognormal'")

    # --- 6) Sample spot sizes (cells per bin) ---
    if spot_size_model == "poisson":
        spot_sizes = rng.poisson(lam=float(mean_cells_per_spot), size=n_spots)
        spot_sizes = np.maximum(spot_sizes, 1).astype(int)
    elif spot_size_model == "nb":
        spot_sizes = _sample_nb_spot_sizes(
            n=n_spots, mean_cells=float(mean_cells_per_spot), theta=float(nb_theta), rng=rng
        )
    else:
        raise ValueError("spot_size_model must be one of: 'poisson', 'nb'")

    # --- 7) Simulation ---
    n_genes = X_cells.shape[1]
    X_sim = sp.lil_matrix((n_spots, n_genes), dtype=dtype)
    P_sim = np.zeros((n_spots, K), dtype=np.float32)

    for i in range(n_spots):
        # A) sample cell-type mixture
        p = rng.dirichlet(alpha)
        P_sim[i, :] = p

        # B) sample cells according to mixture
        n_cells = int(spot_sizes[i])
        counts_per_ct = rng.multinomial(n_cells, p)

        chosen = []
        for ct_i in range(K):
            c = int(counts_per_ct[ct_i])
            if c > 0:
                pool = cells_by_ct[ct_i]
                if pool.size == 0:
                    continue
                chosen.extend(rng.choice(pool, size=c, replace=True).tolist())

        # C) aggregate
        if len(chosen) == 0:
            continue
        agg = X_cells[chosen, :].sum(axis=0)
        if sp.issparse(agg):
            agg_profile = np.asarray(agg).ravel()
        else:
            agg_profile = np.asarray(agg).ravel()

        current_sum = float(agg_profile.sum())
        if current_sum <= 0:
            continue

        target_umi = int(target_depths[i])

        # D) generate observed bin counts
        if sampling_model == "poisson_thin":
            # expected total approx target_umi
            scale = target_umi / current_sum
            lam = agg_profile * scale
            final = rng.poisson(lam=lam).astype(dtype, copy=False)

        elif sampling_model == "multinomial":
            # exact total = min(target_umi, current_sum) if downsampling,
            # but if target_umi > current_sum we just keep agg (no upsampling).
            if target_umi < current_sum:
                prob = agg_profile / current_sum
                final = rng.multinomial(target_umi, prob).astype(dtype, copy=False)
            else:
                final = agg_profile.astype(dtype, copy=False)
        else:
            raise ValueError("sampling_model must be one of: 'poisson_thin', 'multinomial'")

        X_sim[i, :] = final

    # --- 8) Finalize AnnData ---
    X_sim = X_sim.tocsr()
    adata_sim = AnnData(X=X_sim)
    adata_sim.obs_names = [f"Spot_{i}" for i in range(n_spots)]
    adata_sim.var_names = sc_ref.var_names

    adata_sim.obsm["proportions_true"] = pd.DataFrame(
        P_sim, columns=cell_types, index=adata_sim.obs_names
    )

    # Scanpy-style QC names
    adata_sim.obs["total_counts"] = np.asarray(adata_sim.X.sum(axis=1)).ravel()
    adata_sim.obs["n_genes_by_counts"] = np.asarray((adata_sim.X > 0).sum(axis=1)).ravel()

    # (optional) keep your legacy fields if you want
    adata_sim.obs["n_counts"] = adata_sim.obs["total_counts"].astype(float)

    return adata_sim


import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

def compute_multiclass_metrics(
    df: pd.DataFrame,
    pred_col: str = "easydecon",
    true_col: str = "truth",
):
    """
    Compute multiclass classification metrics from a DataFrame with
    prediction and ground-truth columns.

    - Handles mixed-type labels (e.g. float + str) by casting to str.
    - Drops rows where either prediction or truth is NaN.
    """

    # Select columns
    y_true = df[true_col]
    y_pred = df[pred_col]

    # Drop rows with missing values in either column
    mask = y_true.notna() & y_pred.notna()
    y_true = y_true[mask].astype(str)
    y_pred = y_pred[mask].astype(str)

    # Explicit labels (same dtype, safe to sort)
    labels = sorted(pd.unique(pd.concat([y_true, y_pred], ignore_index=True)))

    # Overall accuracy
    acc = accuracy_score(y_true, y_pred)

    # Per-class metrics
    prec_c, rec_c, f1_c, support_c = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    per_class = pd.DataFrame(
        {
            "precision": prec_c,
            "recall": rec_c,
            "f1": f1_c,
            "support": support_c,
        },
        index=labels,
    )

    # Macro and weighted averages
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    overall = {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "n_samples": int(mask.sum()),
    }

    return {
        "overall": overall,
        "per_class": per_class,
        "confusion_matrix": cm_df,
    }

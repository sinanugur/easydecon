def detect_spatial_niches_from_posteriors(
    sdata,
    posterior_df,
    bin_size: int = 8,
    n_neighbors: int = 6,
    n_niches: int = 5,
    auto_n_niches: bool = False,
    n_niches_min: int = 2,
    n_niches_max: int = 10,
    selection_metric: str = "silhouette",  # {"silhouette", "inertia"}
    smooth: bool = True,
    niches_column: str = "niche",
    add_to_obs: bool = True,
    random_state: int = 0,
    return_diagnostics: bool = False,
):
    """
    Detect spatial niches from an easydecon posterior dataframe.

    This function takes per-spot posterior cell-type probabilities (or proportions)
    and identifies recurrent spatial niches as clusters of local compositions.

    Workflow:
      1) Align posterior_df rows to the spatial table.
      2) Extract spatial coordinates for each spot.
      3) Optionally smooth posteriors by averaging over spatial k-nearest neighbors.
      4) Select n_niches (optionally automatically via silhouette/inertia).
      5) Cluster the (smoothed) compositions into `n_niches` groups (niches).
      6) Optionally write niche labels into `table.obs[niches_column]`.

    Parameters
    ----------
    sdata : SpatialData or AnnData-like
        Object containing the spatial transcriptomics data and tables.
        A table named f"square_{bin_size:03}um" is used if present; otherwise
        "table" or `sdata` itself is used.
    posterior_df : pandas.DataFrame
        DataFrame of posteriors / proportions. Rows = spots, columns = cell types.
        Row index must match the spot IDs in table.obs.index (at least partially).
    bin_size : int, optional (default: 8)
        Bin size used in the spatial table name, f"square_{bin_size:03}um".
    n_neighbors : int, optional (default: 6)
        Number of spatial nearest neighbors used for smoothing.
        If `smooth=False`, this is ignored.
    n_niches : int, optional (default: 5)
        Number of spatial niche clusters to detect, if `auto_n_niches=False`.
    auto_n_niches : bool, optional (default: False)
        If True, ignore `n_niches` and select the optimal number automatically
        between [n_niches_min, n_niches_max] using `selection_metric`.
    n_niches_min : int, optional (default: 2)
        Minimum number of niches to consider in automatic selection.
    n_niches_max : int, optional (default: 10)
        Maximum number of niches to consider in automatic selection.
    selection_metric : {"silhouette", "inertia"}, optional (default: "silhouette")
        Metric for automatic selection:
          - "silhouette": choose k with highest silhouette score.
          - "inertia": choose k with strongest elbow-like drop in inertia
            (here simply the largest relative decrease).
    smooth : bool, optional (default: True)
        If True, compute neighborhood-averaged compositions before clustering.
    niches_column : str, optional (default: "niche")
        Name of the column to store niche labels in table.obs when add_to_obs=True.
    add_to_obs : bool, optional (default: True)
        If True, writes niche labels into table.obs[niches_column].
    random_state : int, optional (default: 0)
        Random seed for clustering.
    return_diagnostics : bool, optional (default: False)
        If True, also return a diagnostics dict with candidate k, inertia,
        silhouette (if available), and chosen_k.

    Returns
    -------
    niches : pandas.DataFrame
        DataFrame with a single categorical column `niches_column`
        (index = spot IDs).
    smoothed_posteriors : pandas.DataFrame
        The (optionally) neighborhood-smoothed posterior matrix used for clustering.
    diagnostics : dict, optional
        Only returned when `return_diagnostics=True`. Contains keys:
          - "candidate_k"
          - "inertia"
          - "silhouette"
          - "chosen_k"
          - "selection_metric"
    """
    import numpy as np
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # -------------------------
    # 1) Get spatial table
    # -------------------------
    try:
        table_name = f"square_{bin_size:03}um"
        table = sdata.tables[table_name]
    except (AttributeError, KeyError):
        try:
            table = sdata.tables["table"]
        except (AttributeError, KeyError):
            table = sdata  # assume sdata itself is an AnnData-like table

    # -------------------------
    # 2) Align indices
    # -------------------------
    if not isinstance(posterior_df, pd.DataFrame):
        raise TypeError("posterior_df must be a pandas DataFrame (spots x cell types).")

    common_index = table.obs.index.intersection(posterior_df.index)
    if len(common_index) == 0:
        raise ValueError(
            "No overlapping spot IDs between table.obs.index and posterior_df.index."
        )

    post = posterior_df.loc[common_index].copy()
    post = post.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = post.to_numpy(dtype=float)
    n_spots = X.shape[0]

    # -------------------------
    # 3) Get spatial coordinates
    # -------------------------
    coords = None
    if hasattr(table, "obsm") and hasattr(table.obsm, "keys") and "spatial" in table.obsm.keys():
        idx_pos = table.obs.index.get_indexer(common_index)
        coords = table.obsm["spatial"][idx_pos, :]
    else:
        if {"x", "y"}.issubset(table.obs.columns):
            coords = table.obs.loc[common_index, ["x", "y"]].to_numpy()

    if coords is None:
        raise ValueError(
            "Could not find spatial coordinates. Expected table.obsm['spatial'] "
            "or table.obs[['x', 'y']]."
        )

    # -------------------------
    # 4) Neighborhood smoothing
    # -------------------------
    if smooth and n_neighbors > 1 and n_spots > 1:
        n_neighbors_eff = min(n_neighbors, n_spots)
        nn = NearestNeighbors(n_neighbors=n_neighbors_eff)
        nn.fit(coords)
        neighbor_idx = nn.kneighbors(coords, return_distance=False)
        X_smooth = X[neighbor_idx].mean(axis=1)
    else:
        X_smooth = X

    smoothed_posteriors = pd.DataFrame(
        X_smooth, index=common_index, columns=post.columns
    )

    diagnostics = {
        "candidate_k": None,
        "inertia": None,
        "silhouette": None,
        "chosen_k": None,
        "selection_metric": selection_metric,
    }

    # -------------------------
    # 5) Choose n_niches (optional auto)
    # -------------------------
    if auto_n_niches:
        if n_spots < 3:
            # too few points to do anything fancy
            chosen_k = max(1, min(n_niches, n_spots))
            kmeans = KMeans(
                n_clusters=chosen_k,
                random_state=random_state,
                n_init="auto",
            )
            labels = kmeans.fit_predict(X_smooth)
        else:
            ks = [
                k
                for k in range(n_niches_min, n_niches_max + 1)
                if 1 < k <= n_spots
            ]
            if len(ks) == 0:
                ks = [min(max(2, n_niches_min), n_spots)]

            inertia_list = []
            sil_list = []
            best_score = -np.inf
            best_k = None
            best_labels = None

            for k in ks:
                kmeans_k = KMeans(
                    n_clusters=k,
                    random_state=random_state,
                    n_init="auto",
                )
                labels_k = kmeans_k.fit_predict(X_smooth)
                inertia_k = float(kmeans_k.inertia_)
                inertia_list.append(inertia_k)

                sil_k = np.nan
                if k > 1 and n_spots > k:
                    try:
                        sil_k = float(silhouette_score(X_smooth, labels_k))
                    except Exception:
                        sil_k = np.nan
                sil_list.append(sil_k)

                if selection_metric == "silhouette":
                    score_k = sil_k
                else:  # "inertia"
                    # Use negative inertia so "larger is better"
                    score_k = -inertia_k

                if np.isfinite(score_k) and (score_k > best_score):
                    best_score = score_k
                    best_k = k
                    best_labels = labels_k

            if best_k is None:
                # Fallback: single run with default n_niches
                chosen_k = max(1, min(n_niches, n_spots))
                kmeans = KMeans(
                    n_clusters=chosen_k,
                    random_state=random_state,
                    n_init="auto",
                )
                labels = kmeans.fit_predict(X_smooth)
            else:
                chosen_k = best_k
                labels = best_labels

            diagnostics["candidate_k"] = ks
            diagnostics["inertia"] = inertia_list
            diagnostics["silhouette"] = sil_list
            diagnostics["chosen_k"] = chosen_k
    else:
        # Fixed n_niches
        chosen_k = max(1, min(n_niches, n_spots))
        kmeans = KMeans(
            n_clusters=chosen_k,
            random_state=random_state,
            n_init="auto",
        )
        labels = kmeans.fit_predict(X_smooth)

        # diagnostics (optional)
        if return_diagnostics and chosen_k > 1 and n_spots > chosen_k:
            try:
                sil = float(silhouette_score(X_smooth, labels))
            except Exception:
                sil = np.nan
        else:
            sil = np.nan

        diagnostics["candidate_k"] = [chosen_k]
        diagnostics["inertia"] = [float(kmeans.inertia_)]
        diagnostics["silhouette"] = [sil]
        diagnostics["chosen_k"] = chosen_k

    # -------------------------
    # 6) Wrap labels in DataFrame (categorical)
    # -------------------------
    niches = pd.DataFrame(
        {niches_column: pd.Categorical(labels)},
        index=common_index,
    )
    niches[niches_column] = niches[niches_column].cat.rename_categories(str)

    # -------------------------
    # 7) Write to obs (optional)
    # -------------------------
    if add_to_obs:
        table.obs.drop(columns=niches.columns, inplace=True, errors="ignore")
        table.obs = pd.merge(table.obs, niches, left_index=True, right_index=True)

    if return_diagnostics:
        return niches, smoothed_posteriors, diagnostics
    else:
        return niches, smoothed_posteriors


def summarize_niche_compositions(
    smoothed_posteriors,
    niches_df,
    niches_column: str = "niche",
    normalize_rows: bool = True,
):
    """
    Compute mean cell-type composition per niche.

    Parameters
    ----------
    smoothed_posteriors : pandas.DataFrame
        (spots x cell types) matrix returned by detect_spatial_niches_from_posteriors.
    niches_df : pandas.DataFrame
        DataFrame with a categorical column `niches_column` indexed by spots.
    niches_column : str, optional
        Name of the niche column in niches_df.
    normalize_rows : bool, optional
        If True, renormalize each niche's mean vector to sum to 1.

    Returns
    -------
    pandas.DataFrame
        (n_niches x cell types) mean compositions per niche.
    """
    import pandas as pd
    import numpy as np

    if not isinstance(niches_df, pd.DataFrame):
        raise TypeError("niches_df must be a pandas DataFrame.")
    if niches_column not in niches_df.columns:
        raise ValueError(f"{niches_column!r} not found in niches_df.columns.")

    common_index = smoothed_posteriors.index.intersection(niches_df.index)
    if len(common_index) == 0:
        raise ValueError("No overlapping indices between smoothed_posteriors and niches_df.")

    X = smoothed_posteriors.loc[common_index]
    niches = niches_df.loc[common_index, niches_column]

    mean_mat = X.groupby(niches).mean()

    if normalize_rows:
        row_sums = mean_mat.sum(axis=1).replace(0, np.nan)
        mean_mat = mean_mat.div(row_sums, axis=0)

    return mean_mat

def plot_niche_compositions(
    smoothed_posteriors,
    niches_df,
    niches_column: str = "niche",
    normalize_rows: bool = True,
    figsize=(6, 4),
    legend_fontsize: int = 8,
    rotation: int = 0,
):
    """
    Plot niche-wise mean cell-type compositions as stacked bars.

    Parameters
    ----------
    smoothed_posteriors : pandas.DataFrame
        (spots x cell types) matrix.
    niches_df : pandas.DataFrame
        DataFrame with a categorical column `niches_column` indexed by spots.
    niches_column : str, optional
        Name of the niche column in niches_df.
    normalize_rows : bool, optional
        If True, each bar sums to 1.
    figsize : tuple, optional
        Matplotlib figure size.
    legend_fontsize : int, optional
        Font size for legend.
    rotation : int, optional
        Rotation angle for x-tick labels.
    """
    import matplotlib.pyplot as plt

    mean_mat = summarize_niche_compositions(
        smoothed_posteriors, niches_df, niches_column=niches_column,
        normalize_rows=normalize_rows,
    )

    fig, ax = plt.subplots(figsize=figsize)

    bottom = None
    x = range(mean_mat.shape[0])
    for col in mean_mat.columns:
        vals = mean_mat[col].values
        if bottom is None:
            ax.bar(x, vals, label=col)
            bottom = vals
        else:
            ax.bar(x, vals, bottom=bottom, label=col)
            bottom = bottom + vals

    ax.set_xticks(list(x))
    ax.set_xticklabels(mean_mat.index.astype(str), rotation=rotation)
    ax.set_ylabel("Proportion" if normalize_rows else "Mean posterior")
    ax.set_xlabel("Niche")
    ax.legend(fontsize=legend_fontsize, bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    return fig, ax



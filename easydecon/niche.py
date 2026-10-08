import pandas as pd

from ._schema import get_table


def _resolve_posterior_dataframe(
    posterior_like,
    use_assignment_if_no_posterior=False,
):
    """Resolve a posterior matrix from a DataFrame or result-like object."""
    if isinstance(posterior_like, pd.DataFrame):
        return posterior_like
    if hasattr(posterior_like, "posterior_df"):
        if posterior_like.posterior_df is not None:
            return posterior_like.posterior_df
        if (
            use_assignment_if_no_posterior
            and hasattr(posterior_like, "assignment_df")
            and posterior_like.assignment_df is not None
        ):
            return posterior_like.assignment_df
        raise ValueError(
            "EasyDeconResult does not contain posterior_df. This usually happens "
            "when marker_genes was provided as a list-style mask workflow. Pass "
            "use_assignment_if_no_posterior=True to use assignment_df/phase2 "
            "scores instead."
        )
    raise TypeError(
        "posterior_df must be a pandas DataFrame or an EasyDeconResult-like "
        "object with posterior_df."
    )


def _smooth_spatial_compositions(
    X,
    coords,
    n_neighbors,
    sample_labels=None,
    chunk_size=20_000,
):
    """Average compositions over spatial neighbors in bounded query chunks."""
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    smoothed = np.empty(X.shape, dtype=np.float32)
    if sample_labels is None:
        groups = [(None, None)]
        effective_neighbors = None
    else:
        sample_values = sample_labels.to_numpy()
        groups = (
            (sample, np.flatnonzero(sample_values == sample))
            for sample in pd.unique(sample_labels)
        )
        effective_neighbors = {}

    for sample, positions in groups:
        sample_coords = coords if positions is None else coords[positions]
        sample_size = len(sample_coords)
        n_neighbors_eff = min(n_neighbors, sample_size)
        if effective_neighbors is not None:
            effective_neighbors[sample] = int(n_neighbors_eff)
        nn = NearestNeighbors(n_neighbors=n_neighbors_eff).fit(sample_coords)

        for start in range(0, sample_size, chunk_size):
            stop = min(start + chunk_size, sample_size)
            neighbor_idx = nn.kneighbors(
                sample_coords[start:stop], return_distance=False
            )
            if positions is not None:
                neighbor_idx = positions[neighbor_idx]

            chunk_mean = np.zeros((stop - start, X.shape[1]), dtype=np.float32)
            for neighbor_number in range(n_neighbors_eff):
                chunk_mean += X[neighbor_idx[:, neighbor_number]]
            chunk_mean /= np.float32(n_neighbors_eff)

            output_positions = (
                slice(start, stop) if positions is None else positions[start:stop]
            )
            smoothed[output_positions] = chunk_mean

    return smoothed, effective_neighbors


def _make_niche_model(method, n_clusters, random_state):
    """Create the requested sklearn clustering model."""
    from sklearn.cluster import KMeans, MiniBatchKMeans

    model_class = KMeans if method == "kmeans" else MiniBatchKMeans
    kwargs = {
        "n_clusters": n_clusters,
        "random_state": random_state,
        "n_init": "auto",
    }
    if method == "minibatch":
        kwargs["batch_size"] = 4096
    return model_class(**kwargs)


def _predict_in_chunks(model, matrix, chunk_size):
    """Predict labels in bounded batches."""
    import numpy as np

    labels = np.empty(len(matrix), dtype=np.int32)
    for start in range(0, len(matrix), chunk_size):
        stop = min(start + chunk_size, len(matrix))
        labels[start:stop] = model.predict(matrix[start:stop])
    return labels


def _peak_memory_mb():
    """Return peak RSS in MiB when the standard-library metric is available."""
    try:
        import resource
        import sys

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024 * 1024 if sys.platform == "darwin" else 1024)
    except (ImportError, OSError):
        return None


def _select_niche_fit_positions(
    sample_labels,
    balance_samples,
    max_fit_per_sample,
    random_state,
):
    """Select deterministic per-sample positions for KMeans fitting."""
    import numpy as np

    counts = sample_labels.value_counts(sort=False)
    sample_counts = {sample: int(count) for sample, count in counts.items()}
    if not balance_samples and max_fit_per_sample is None:
        return None, sample_counts, sample_counts.copy()

    target = None
    if balance_samples:
        target = int(counts.min())
        if max_fit_per_sample is not None:
            target = min(target, max_fit_per_sample)

    rng = np.random.default_rng(random_state)
    sample_values = sample_labels.to_numpy()
    selected = []
    fit_counts = {}
    for sample, count in counts.items():
        positions = np.flatnonzero(sample_values == sample)
        n_select = target if balance_samples else min(int(count), max_fit_per_sample)
        if n_select < len(positions):
            positions = rng.choice(positions, size=n_select, replace=False)
        selected.append(positions)
        fit_counts[sample] = int(n_select)
    return np.concatenate(selected), sample_counts, fit_counts


def _select_inertia_elbow(candidate_k, inertia):
    """Return the k farthest below the endpoint line of the inertia curve."""
    import numpy as np

    if len(candidate_k) < 3:
        return candidate_k[0]

    k_values = np.asarray(candidate_k, dtype=float)
    inertia_values = np.asarray(inertia, dtype=float)
    k_range = np.ptp(k_values)
    inertia_range = np.ptp(inertia_values)
    if k_range == 0 or inertia_range == 0:
        return candidate_k[0]

    x = (k_values - k_values.min()) / k_range
    y = (inertia_values - inertia_values.min()) / inertia_range
    endpoint_line = y[0] + (y[-1] - y[0]) * x
    return candidate_k[int(np.argmax(endpoint_line - y))]


def detect_spatial_niches_from_posteriors(
    sdata,
    posterior_df,
    bin_size: int = 8,
    table_key=None,
    preferred_table_keys=None,
    use_assignment_if_no_posterior: bool = False,
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
    sample_column=None,
    balance_samples: bool = False,
    max_fit_per_sample=None,
    silhouette_sample_size=10_000,
    return_model: bool = False,
    clustering_method: str = "kmeans",
    smoothing_chunk_size: int = 20_000,
    prediction_chunk_size: int = 50_000,
    verbose: bool = False,
):
    """
    Detect spatial niches from an easydecon posterior dataframe.

    This function takes per-spot posterior cell-type probabilities (or proportions)
    and identifies recurrent spatial niches as clusters of local compositions.

    Workflow:
      1) Align posterior_df rows to the spatial table.
      2) Extract spatial coordinates for each spot.
      3) Optionally smooth posteriors over spatial neighbors, separately within
         each sample when `sample_column` is provided.
      4) Optionally select a balanced/capped subset for KMeans fitting.
      5) Select n_niches (optionally automatically via silhouette/inertia).
      6) Predict niches for every aligned spatial location.
      7) Optionally write niche labels into `table.obs[niches_column]`.

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
          - "inertia": choose the elbow by maximum deviation below the line
            joining the first and last candidate inertia values.
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
    sample_column : str, optional
        Column in `table.obs` identifying samples. Spatial smoothing is performed
        independently within each sample.
    balance_samples : bool, optional (default: False)
        If True, use the same number of fitting locations from every sample.
    max_fit_per_sample : int, optional
        Maximum fitting locations contributed by each sample. Smoothing and final
        prediction still use every aligned location.
    silhouette_sample_size : int or None, optional (default: 10000)
        Maximum fitting rows used for silhouette scoring. None uses all fitting
        rows.
    return_model : bool, optional (default: False)
        If True, return the final fitted clustering model after the usual outputs.
    clustering_method : {"kmeans", "minibatch", "auto"}, optional
        Clustering implementation. "auto" uses MiniBatchKMeans for fitting
        matrices with at least 50,000 rows and KMeans otherwise.
    smoothing_chunk_size : int, optional (default: 20000)
        Maximum locations queried at once during spatial smoothing.
    prediction_chunk_size : int, optional (default: 50000)
        Maximum locations predicted at once after model fitting.
    verbose : bool, optional (default: False)
        Print concise progress and memory information.

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
    model : sklearn.cluster.KMeans or sklearn.cluster.MiniBatchKMeans, optional
        The final model used to predict all returned labels. Returned only when
        `return_model=True`.
    """
    import numpy as np
    from numbers import Integral

    from sklearn.metrics import silhouette_score

    def _log(message):
        if verbose:
            memory_mb = _peak_memory_mb()
            suffix = "" if memory_mb is None else f"; peak RSS {memory_mb:.1f} MiB"
            print(f"[easydecon.niche] {message}{suffix}", flush=True)

    if selection_metric not in {"silhouette", "inertia"}:
        raise ValueError(
            "selection_metric must be one of {'silhouette', 'inertia'}."
        )
    if clustering_method not in {"kmeans", "minibatch", "auto"}:
        raise ValueError(
            "clustering_method must be one of {'kmeans', 'minibatch', 'auto'}."
        )
    if not isinstance(balance_samples, bool):
        raise TypeError("balance_samples must be a bool.")
    if (
        max_fit_per_sample is not None
        and (
            isinstance(max_fit_per_sample, bool)
            or not isinstance(max_fit_per_sample, Integral)
            or max_fit_per_sample < 1
        )
    ):
        raise ValueError("max_fit_per_sample must be None or an integer >= 1.")
    if (
        silhouette_sample_size is not None
        and (
            isinstance(silhouette_sample_size, bool)
            or not isinstance(silhouette_sample_size, Integral)
            or silhouette_sample_size < 2
        )
    ):
        raise ValueError("silhouette_sample_size must be None or an integer >= 2.")
    for value, name in (
        (smoothing_chunk_size, "smoothing_chunk_size"),
        (prediction_chunk_size, "prediction_chunk_size"),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1.")
    if sample_column is None and balance_samples:
        raise ValueError("balance_samples=True requires sample_column.")
    if sample_column is None and max_fit_per_sample is not None:
        raise ValueError("max_fit_per_sample requires sample_column.")

    posterior_df = _resolve_posterior_dataframe(
        posterior_df,
        use_assignment_if_no_posterior=use_assignment_if_no_posterior,
    )
    table = get_table(
        sdata,
        bin_size=bin_size,
        table_key=table_key,
        preferred_table_keys=preferred_table_keys,
    )
    if sample_column is not None and sample_column not in table.obs.columns:
        raise ValueError(f"sample_column {sample_column!r} was not found in table.obs.")

    # -------------------------
    # 2) Align indices
    # -------------------------
    if not isinstance(posterior_df, pd.DataFrame):
        raise TypeError("posterior_df must be a pandas DataFrame (spots x cell types).")

    _log("Aligning and converting posterior matrix")
    common_index = table.obs.index[table.obs.index.isin(posterior_df.index)]
    if len(common_index) == 0:
        raise ValueError(
            "No overlapping spot IDs between table.obs.index and posterior_df.index."
        )

    posterior_positions = posterior_df.index.get_indexer(common_index)
    feature_columns = list(posterior_df.columns)
    X = np.empty((len(common_index), len(feature_columns)), dtype=np.float32)
    finite_columns = np.zeros(len(feature_columns), dtype=bool)
    for column_number in range(len(feature_columns)):
        values = pd.to_numeric(
            posterior_df.iloc[posterior_positions, column_number], errors="coerce"
        )
        X[:, column_number] = values.to_numpy(dtype=np.float32, na_value=np.nan)
        finite_columns[column_number] = np.isfinite(X[:, column_number]).any()

    if not finite_columns.any():
        raise ValueError("posterior_df contains no usable numeric cell-type columns.")
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    usable_columns = finite_columns.copy()
    for column_number in np.flatnonzero(finite_columns):
        usable_columns[column_number] = np.any(X[:, column_number] != 0)
    if not usable_columns.any():
        raise ValueError(
            "posterior_df contains only zero values after alignment and numeric "
            "conversion."
        )
    if not usable_columns.all():
        X = X[:, usable_columns]
        feature_columns = [
            column
            for column, usable in zip(feature_columns, usable_columns)
            if usable
        ]
    if np.isclose(X.sum(axis=1), 0.0).all():
        raise ValueError(
            "posterior_df contains only zero values after alignment and numeric "
            "conversion."
        )
    n_spots = X.shape[0]
    _log(
        f"Posterior matrix {X.shape}, dtype={X.dtype}, "
        f"approximately {X.nbytes / (1024 ** 2):.1f} MiB"
    )

    sample_labels = None
    if sample_column is not None:
        sample_labels = table.obs.loc[common_index, sample_column]
        if sample_labels.isna().any():
            raise ValueError(
                f"sample_column {sample_column!r} contains missing sample identifiers."
            )

    # -------------------------
    # 3) Get spatial coordinates
    # -------------------------
    coords = None
    if hasattr(table, "obsm") and hasattr(table.obsm, "keys") and "spatial" in table.obsm.keys():
        if len(common_index) == len(table.obs) and common_index.equals(table.obs.index):
            coords = table.obsm["spatial"]
        else:
            idx_pos = table.obs.index.get_indexer(common_index)
            coords = table.obsm["spatial"][idx_pos, :]
    else:
        if {"x", "y"}.issubset(table.obs.columns):
            coords = table.obs.loc[common_index, ["x", "y"]].to_numpy(copy=False)

    if coords is None:
        raise ValueError(
            "Could not find spatial coordinates. Expected table.obsm['spatial'] "
            "or table.obs[['x', 'y']]."
        )

    # -------------------------
    # 4) Neighborhood smoothing
    # -------------------------
    effective_neighbors_by_sample = None
    if smooth and n_neighbors > 1 and n_spots > 1:
        _log("Spatial smoothing started")
        X_smooth, effective_neighbors_by_sample = _smooth_spatial_compositions(
            X,
            coords,
            n_neighbors,
            sample_labels=sample_labels,
            chunk_size=int(smoothing_chunk_size),
        )
        _log("Spatial smoothing completed")
    else:
        X_smooth = X
        if sample_labels is not None:
            effective_neighbors_by_sample = {
                sample: 1 for sample in pd.unique(sample_labels)
            }
        _log("Spatial smoothing skipped")

    if sample_labels is None:
        fit_positions = None
        sample_counts = None
        fit_counts_by_sample = None
        n_samples = 1
        fit_uses_all_locations = True
    else:
        fit_positions, sample_counts, fit_counts_by_sample = (
            _select_niche_fit_positions(
                sample_labels,
                balance_samples,
                max_fit_per_sample,
                random_state,
            )
        )
        n_samples = len(sample_counts)
        fit_uses_all_locations = not balance_samples and max_fit_per_sample is None

    X_fit = X_smooth if fit_uses_all_locations else X_smooth[fit_positions]
    n_fit = len(X_fit)
    selected_clustering_method = (
        "minibatch"
        if clustering_method == "auto" and n_fit >= 50_000
        else "kmeans"
        if clustering_method == "auto"
        else clustering_method
    )
    _log(f"Fitting subset contains {n_fit:,} of {n_spots:,} locations")
    _log(f"Selected clustering algorithm: {selected_clustering_method}")
    effective_silhouette_sample_size = (
        None
        if silhouette_sample_size is None
        else min(int(silhouette_sample_size), n_fit)
    )

    diagnostics = {
        "candidate_k": None,
        "inertia": None,
        "silhouette": None,
        "chosen_k": None,
        "selection_metric": selection_metric,
        "n_total_locations": int(n_spots),
        "n_fit_locations": int(n_fit),
        "sample_column": sample_column,
        "balance_samples": balance_samples,
        "max_fit_per_sample": max_fit_per_sample,
        "sample_counts": sample_counts,
        "fit_counts_by_sample": fit_counts_by_sample,
        "n_samples": int(n_samples),
        "n_neighbors": n_neighbors,
        "effective_neighbors_by_sample": effective_neighbors_by_sample,
        "silhouette_sample_size": effective_silhouette_sample_size,
        "feature_columns": list(feature_columns),
        "clustering_method": clustering_method,
        "selected_clustering_method": selected_clustering_method,
        "smoothing_chunk_size": int(smoothing_chunk_size),
        "prediction_chunk_size": int(prediction_chunk_size),
        "matrix_dtype": str(X_smooth.dtype),
        "matrix_memory_mb": float(X_smooth.nbytes / (1024 ** 2)),
    }

    # -------------------------
    # 5) Choose n_niches (optional auto)
    # -------------------------
    def _silhouette(labels, k):
        if k <= 1 or n_fit <= k:
            return np.nan
        try:
            return float(
                silhouette_score(
                    X_fit,
                    labels,
                    sample_size=effective_silhouette_sample_size,
                    random_state=random_state,
                )
            )
        except Exception:
            return np.nan

    if auto_n_niches:
        ks = [
            k
            for k in range(n_niches_min, n_niches_max + 1)
            if 1 < k <= n_fit
        ]
        if not ks:
            ks = [max(1, min(n_niches, n_fit))]
        inertia_list = []
        sil_list = []
        _log(f"Automatic niche selection fitting {len(ks)} candidate model(s)")
        for k in ks:
            model_k = _make_niche_model(
                selected_clustering_method, k, random_state
            ).fit(X_fit)
            inertia_list.append(float(model_k.inertia_))
            sil_list.append(
                _silhouette(model_k.labels_, k)
                if selection_metric == "silhouette"
                else np.nan
            )
            del model_k

        if selection_metric == "inertia":
            chosen_k = _select_inertia_elbow(ks, inertia_list)
        else:
            finite_scores = [
                (score, k) for k, score in zip(ks, sil_list) if np.isfinite(score)
            ]
            if finite_scores:
                chosen_k = max(finite_scores, key=lambda item: item[0])[1]
            else:
                chosen_k = max(1, min(n_niches, n_fit))

        diagnostics["candidate_k"] = ks
        diagnostics["inertia"] = inertia_list
        diagnostics["silhouette"] = sil_list
        diagnostics["chosen_k"] = chosen_k
    else:
        chosen_k = max(1, min(n_niches, n_fit))
        diagnostics["candidate_k"] = [chosen_k]
        diagnostics["silhouette"] = [np.nan]
        diagnostics["chosen_k"] = chosen_k

    _log("Final model fitting started")
    kmeans = _make_niche_model(
        selected_clustering_method, chosen_k, random_state
    ).fit(X_fit)
    _log("Final model fitting completed")
    if not auto_n_niches:
        diagnostics["inertia"] = [float(kmeans.inertia_)]

    _log("Full-data prediction started")
    labels = _predict_in_chunks(kmeans, X_smooth, int(prediction_chunk_size))
    kmeans.feature_names_in_ = np.asarray(feature_columns, dtype=object)
    _log("Full-data prediction completed")

    # -------------------------
    # 6) Wrap labels in DataFrame (categorical)
    # -------------------------
    _log("Constructing output DataFrames")
    smoothed_posteriors = pd.DataFrame(
        X_smooth,
        index=common_index,
        columns=feature_columns,
        copy=False,
    )
    niches = pd.DataFrame(
        {niches_column: pd.Categorical(labels)},
        index=common_index,
    )
    niches[niches_column] = niches[niches_column].cat.rename_categories(str)

    # -------------------------
    # 7) Write to obs (optional)
    # -------------------------
    if add_to_obs:
        table.obs[niches_column] = niches[niches_column].reindex(table.obs.index)

    diagnostics["peak_memory_mb"] = _peak_memory_mb()

    if return_diagnostics and return_model:
        return niches, smoothed_posteriors, diagnostics, kmeans
    if return_diagnostics:
        return niches, smoothed_posteriors, diagnostics
    if return_model:
        return niches, smoothed_posteriors, kmeans
    return niches, smoothed_posteriors


def detect_niches_from_easydecon_result(
    sdata,
    result,
    bin_size: int = 8,
    use_assignment_if_no_posterior: bool = False,
    **kwargs,
):
    """Detect niches from an EasyDeconResult-like object."""
    return detect_spatial_niches_from_posteriors(
        sdata=sdata,
        posterior_df=result,
        bin_size=bin_size,
        use_assignment_if_no_posterior=use_assignment_if_no_posterior,
        **kwargs,
    )


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
    import numpy as np

    if not isinstance(niches_df, pd.DataFrame):
        raise TypeError("niches_df must be a pandas DataFrame.")
    if niches_column not in niches_df.columns:
        raise ValueError(f"{niches_column!r} not found in niches_df.columns.")

    common_index = smoothed_posteriors.index.intersection(niches_df.index)
    if len(common_index) == 0:
        raise ValueError("No overlapping indices between smoothed_posteriors and niches_df.")

    X = smoothed_posteriors.loc[common_index]
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    niches = niches_df.loc[common_index, niches_column]

    mean_mat = X.groupby(niches, observed=False).mean()

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

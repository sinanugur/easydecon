import numpy as np
import random
import scanpy as sc
import squidpy as sq
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, regularizers
from sklearn.model_selection import train_test_split

def simulate_pseudo_spots_simple(
    sc_ref,              # AnnData (cells × genes), already normalized + log1p
    celltype_column,     # Column in sc_ref.obs with cell‐type labels
    n_spots=30000,
    spot_size=5,
    random_state=0,
    dirichlet_alpha=None
):
    """
    Simulate `n_spots` pseudo‐bulk mixtures from single‐cell data using only sc_ref and its cell labels.

    Parameters
    ----------
    sc_ref : AnnData
        Single‐cell AnnData containing only the selected genes (normalized + log1p).
    celltype_column : str
        Column in sc_ref.obs specifying each cell’s label.
    n_spots : int
        Number of pseudo‐spots to simulate.
    spot_size : int
        Number of cells (or total UMI‐equivalents) per pseudo‐spot.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    X_pseudo : np.ndarray, shape (n_spots, n_genes)
        Each row is the normalized + log1p expression vector of a pseudo‐spot.
    P_pseudo : np.ndarray, shape (n_spots, K)
        Each row is the Dirichlet‐sampled true proportion vector over K cell types.
    cell_types : list of str
        Ordered list of unique cell‐type labels.
    """
    np.random.seed(random_state)
    random.seed(random_state)
    tf.random.set_seed(random_state)

    if celltype_column not in sc_ref.obs:
        raise ValueError(f"Column '{celltype_column}' not found in sc_ref.obs.")

    # Extract expression matrix and cell‐type labels
    try:
        X_cells = sc_ref.X.toarray()                           # shape = (n_cells, n_genes)
    except:
        X_cells = sc_ref.X
    ct_labels = sc_ref.obs[celltype_column].values
    cell_types = sorted(list(set(ct_labels)))
    K = len(cell_types)
    ct2idx = {ct: i for i, ct in enumerate(cell_types)}
    cell2idx = np.array([ct2idx[ct] for ct in ct_labels])  # shape = (n_cells,)

    # Precompute indices of cells for each cell type
    cells_by_ct = {i: np.where(cell2idx == i)[0] for i in range(K)}
    n_genes = X_cells.shape[1]

    # Determine alpha for Dirichlet distribution
    alpha_values_for_dirichlet = np.ones(K) # Default for None or "uniform"
    
    if dirichlet_alpha is None or dirichlet_alpha == "uniform":
        pass # Keeps default np.ones(K)
    elif isinstance(dirichlet_alpha, (float, int)):
        alpha_values_for_dirichlet = np.full(K, float(dirichlet_alpha))
    elif isinstance(dirichlet_alpha, (list, np.ndarray)):
        if len(dirichlet_alpha) == K:
            alpha_values_for_dirichlet = np.array(dirichlet_alpha, dtype=float)
        else:
            raise ValueError(f"Provided dirichlet_alpha array must have length K={K}, but got {len(dirichlet_alpha)}.")
    elif dirichlet_alpha == "inverse_frequency":
        cell_counts_per_type = np.array([len(cells_by_ct[i]) for i in range(K)], dtype=float)
        
        if np.any(cell_counts_per_type == 0):
             print(f"Warning: The following cell types have zero cells in the reference and will be assigned a minimum alpha for 'inverse_frequency': {[cell_types[i] for i, count in enumerate(cell_counts_per_type) if count == 0]}")
        
        total_cells = np.sum(cell_counts_per_type)
        if total_cells == 0: # Should not happen if K > 0 and sc_ref is not empty
            print("Warning: Total cell count in reference is zero. Using uniform alpha for Dirichlet.")
            alpha_values_for_dirichlet = np.ones(K)
        else:
            # Calculate frequencies, adding a small epsilon for types with zero counts
            # to prevent division by zero and ensure they get some alpha.
            # The epsilon ensures they get a high inverse frequency.
            cell_freqs = (cell_counts_per_type + 1e-10) / (total_cells + K * 1e-10) # Add epsilon to total_cells as well
            
            inv_freqs = 1.0 / cell_freqs
            # Scale so that the sum of alphas is K (similar to np.ones(K) in total magnitude)
            alpha_values_for_dirichlet = (inv_freqs / np.sum(inv_freqs)) * K
    else:
        raise ValueError(f"Unsupported value for dirichlet_alpha: {dirichlet_alpha}. Must be None, 'uniform', 'inverse_frequency', float, list, or np.ndarray.")
    
    if np.any(alpha_values_for_dirichlet <= 0):
        # This can happen if a custom alpha list with non-positive values is given.
        raise ValueError(f"Alpha values for Dirichlet distribution must be positive. Got: {alpha_values_for_dirichlet}")

    X_pseudo = np.zeros((n_spots, n_genes), dtype=float)
    P_pseudo = np.zeros((n_spots, K), dtype=float)

    for i in range(n_spots):
        # (a) Draw a random Dirichlet proportion vector
        p = np.random.dirichlet(alpha=alpha_values_for_dirichlet, size=1).flatten()  # shape = (K,)

        # (b) Decide how many cells of each type (multinomial)
        counts_per_ct = np.random.multinomial(spot_size, p)

        # (c) Sample cells (with replacement) per cell type
        chosen_cells = []
        for ct_idx in range(K):
            c = counts_per_ct[ct_idx]
            if c > 0:
                idxs = np.random.choice(cells_by_ct[ct_idx], size=c, replace=True)
                chosen_cells.append(idxs)
        if len(chosen_cells) > 0:
            chosen_cells = np.concatenate(chosen_cells)
        else:
            chosen_cells = np.array([], dtype=int)

        # (d) Sum their normalized expression (already log1p)
        if chosen_cells.size > 0:
            counts_sum = np.sum(X_cells[chosen_cells, :], axis=0)
        else:
            counts_sum = np.zeros(n_genes, dtype=float)

        # (e) Re‐normalize to total = 10k, then log1p again
        total_counts = counts_sum.sum()
        if total_counts > 0:
            normed = np.log1p((counts_sum / total_counts) * 1e4)
        else:
            normed = counts_sum

        X_pseudo[i, :] = normed
        P_pseudo[i, :] = p

    return X_pseudo, P_pseudo, cell_types


def build_deconv_model(
    num_genes,
    num_celltypes,
    hidden_units=[512, 256],
    dropout_rate=0.2
):
    """
    Build a feed‐forward neural network in Keras that maps an expression vector
    (length = num_genes) → a cell‐type proportion vector (length = num_celltypes).

    Parameters
    ----------
    num_genes : int
        Number of input genes (features).
    num_celltypes : int
        Number of output cell types (dimensions of softmax).
    hidden_units : list of int
        Sizes of hidden Dense layers.
    dropout_rate : float
        Dropout fraction after each hidden layer.

    Returns
    -------
    model : tf.keras.Model
        Keras Model (uncompiled).
    """
    inp = layers.Input(shape=(num_genes,), name="expr_input")
    x = inp
    for i, units in enumerate(hidden_units):
        x = layers.Dense(units,kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-4, l2=0), activation="relu", name=f"dense_{i}")(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"drop_{i}")(x)
    x = layers.Dense(num_celltypes, activation=None, name="dense_out")(x)
    out = layers.Activation("softmax", name="prop")(x)

    model = models.Model(inputs=inp, outputs=out, name="DeconvNN")
    return model


def get_proportions_deeplearning_improved(
    sc_adata,
    sp_adata,
    celltype_column,
    gene_list=None,
    use_hvg=True,
    n_top_hvg=2000,
    use_svg=False,
    n_top_svg=2000,
    n_pseudo_spots=30000,
    pseudo_spot_size=1000,
    hidden_units=[512, 256],
    dropout_rate=0.2,
    epochs=50,
    batch_size=256,
    random_state=0,
    verbose=True,
    dirichlet_alpha=None
):
    """
    Train a Keras‐based deconvolution model using either supplied gene_list or
    the intersection of highly variable genes (HVG) from sc_adata and spatially
    variable genes (SVG) from sp_adata, then predict cell‐type proportions.

    Parameters
    ----------
    sc_adata : AnnData
        Single‐cell reference. Must contain raw or normalized counts and a column
        in .obs indicating cell‐type labels (celltype_column).
    sp_adata : AnnData
        Spatial AnnData. Counts are in .X.
    celltype_column : str
        Column in sc_adata.obs specifying each cell’s type.
    gene_list : list of str or None
        If provided, use this list of genes directly (no HVG/SVG). Otherwise, compute
        HVG/SVG as specified by use_hvg/use_svg.
    use_hvg : bool
        If True and gene_list is None, compute HVGs on sc_adata (n_top_hvg).
    n_top_hvg : int
        Number of top HVGs to select from sc_adata.
    use_svg : bool
        If True and gene_list is None, compute spatially variable genes from sp_adata (n_top_svg).
    n_top_svg : int
        Number of top spatially variable genes to select from sp_adata.
    n_pseudo_spots : int
        Number of pseudo‐spots to simulate.
    pseudo_spot_size : int
        Number of cells (or total UMI equivalents) per pseudo‐spot.
    hidden_units : list of int
        Hidden layer sizes for the NN.
    dropout_rate : float
        Dropout fraction after each hidden layer.
    epochs : int
        Maximum epochs to train the NN.
    batch_size : int
        Batch size for training.
    random_state : int
        Seed for reproducibility.
    verbose : bool
        Print progress if True.

    Returns
    -------
    preds : np.ndarray, shape (n_spots_real, K)
        Predicted cell‐type proportion matrix for sp_adata.
    sp_out : AnnData
        The same sp_adata, but with new columns in .obs named “prop_{cell_type}”.
    """
    # 1) Determine gene set
    if gene_list is not None:
        # Use provided gene_list
        common_genes = list(
            set(sc_adata.var_names)
            .intersection(sp_adata.var_names)
            .intersection(gene_list)
        )
        if len(common_genes) < 10:
            raise ValueError(
                f"Too few genes in the intersection of sc_adata, sp_adata, and gene_list: {len(common_genes)}"
            )
        if verbose:
            print(f"Using provided gene_list: {len(common_genes)} genes.")
    else:
        # Compute HVGs on sc_adata
        if use_hvg:
            sc_tmp = sc_adata.copy()
            if audit_anndata(sc_tmp)["looks_log1p"] is False:
                sc.pp.normalize_total(sc_tmp, target_sum=1e4)
                sc.pp.log1p(sc_tmp)
            sc.pp.highly_variable_genes(sc_tmp, n_top_genes=n_top_hvg, subset=False)
            hvg_sc = set(sc_tmp.var_names[sc_tmp.var["highly_variable"]])
            if verbose:
                print(f"Identified {len(hvg_sc)} HVGs in sc_adata.")
        else:
            hvg_sc = set(sc_adata.var_names)

        # Compute SVGs on sp_adata (using Squidpy’s Moran’s I)
        if use_svg:
            sp_tmp = sp_adata.copy()
            if audit_anndata(sp_tmp)["looks_log1p"] is False:
                sc.pp.normalize_total(sp_tmp, target_sum=1e4)
                sc.pp.log1p(sp_tmp)
            sq.gr.spatial_neighbors(sp_tmp, coord_type='grid')  # or 'generic'
            sq.gr.spatial_autocorr(sp_tmp, mode="moran")  # adds .var["moranI"]
            sp_tmp.var["moranI"] = sp_tmp.uns["moranI"].loc[sp_tmp.var_names, "I"].values
            svg_sp = set(sp_tmp.var_names[np.argsort(-sp_tmp.var["moranI"])[:n_top_svg]])
            if verbose:
                print(f"Identified {len(svg_sp)} spatially variable genes in sp_adata.")
        else:
            # Fallback to HVG on spatial if SVG not requested
            sp_tmp = sp_adata.copy()
            if audit_anndata(sp_tmp)["looks_log1p"] is False:
                sc.pp.normalize_total(sp_tmp, target_sum=1e4)
                sc.pp.log1p(sp_tmp)
            sc.pp.highly_variable_genes(sp_tmp, n_top_genes=n_top_hvg, subset=False)
            svg_sp = set(sp_tmp.var_names[sp_tmp.var["highly_variable"]])
            if verbose:
                print(f"Identified {len(svg_sp)} HVGs in sp_adata (as proxy for SVG).")

        # Intersection: HVG_sc ∩ SVG_sp
        common_genes = list(hvg_sc.intersection(svg_sp))
        if len(common_genes) < 10:
            raise ValueError(f"Too few overlapping HVG/SVG genes: {len(common_genes)}")
        if verbose:
            print(f"Using {len(common_genes)} genes from HVG∩SVG.")

    # 2) Subset and normalize sc_adata to common_genes
    sc_ref = sc_adata[:, common_genes].copy()
    if audit_anndata(sc_ref)["looks_log1p"] is False:
        sc.pp.normalize_total(sc_ref, target_sum=1e4)
        sc.pp.log1p(sc_ref)
    
    # 3) Subset and normalize sp_adata to common_genes
    sp_ref = sp_adata[:, common_genes].copy()
    if audit_anndata(sp_ref)["looks_log1p"] is False:
        sc.pp.normalize_total(sp_ref, target_sum=1e4)
        sc.pp.log1p(sp_ref)
    

    # 4) Simulate pseudo‐spots
    X_pseudo, P_pseudo, cell_types = simulate_pseudo_spots_simple(
        sc_ref,
        celltype_column=celltype_column,
        n_spots=n_pseudo_spots,
        spot_size=pseudo_spot_size,
        random_state=random_state,
        dirichlet_alpha=dirichlet_alpha
    )
    K = len(cell_types)
    if verbose:
        print(f"Simulated {X_pseudo.shape[0]} pseudo‐spots with {len(common_genes)} genes over {K} cell types.")

    # 5) Build and compile NN
    num_genes = len(common_genes)
    model = build_deconv_model(
        num_genes=num_genes,
        num_celltypes=K,
        hidden_units=hidden_units,
        dropout_rate=dropout_rate
    )


    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        #loss=losses.MeanSquaredError(),
        loss = losses.KLDivergence(),
        #metrics=[losses.MeanAbsoluteError()]
        metrics=["mae"]
    )
    if verbose:
        model.summary()

    # 6) Train/validation split
    X_tr, X_val, Y_tr, Y_val = train_test_split(
        X_pseudo, P_pseudo,
        test_size=0.2,
        random_state=random_state,
        shuffle=True
    )
    if verbose:
        print(f"Training on {X_tr.shape[0]} pseudo‐spots; validating on {X_val.shape[0]}.")

    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    history = model.fit(
        X_tr, Y_tr,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[earlystop],
        verbose=1 if verbose else 0
    )

    # 7) Predict on real spatial spots

    try:
        X_sp_real = sp_ref.X.toarray()  # shape = (n_spots_real, num_genes)
    except:
        X_sp_real = sp_ref.X
    preds = model.predict(X_sp_real)  # shape = (n_spots_real, K)


    if verbose:
        print(f"Predicted proportions for {X_sp_real.shape[0]} real spots.")

    # 8) Write predictions back into sp_adata.obs
    sp_out = sp_adata.copy()
    for idx_ct, ct in enumerate(cell_types):
        sp_out.obs[f"prop_{ct}"] = preds[:, idx_ct]

    return preds, sp_out



def audit_anndata(adata, target_sum=1e4, log_max_thresh=20.0):
    """
    Returns a dict with three Boolean fields:
      • 'has_normalize_total_key':  True if adata.uns['normalize_total'] is present
      • 'cells_scaled_to_target':   True if sum(adata.X, axis=1) ≈ target_sum
      • 'has_log1p_key':            True if adata.uns['log1p'] is present
      • 'looks_log1p':              Heuristic: dtype is float, max < log_max_thresh
      • 'has_raw_layer':            True if adata.raw is not None
    """
    out = {}
    out["has_normalize_total_key"] = ("normalize_total" in adata.uns)
    out["has_log1p_key"] = ("log1p" in adata.uns)

    # Check raw:
    out["has_raw_layer"] = (adata.raw is not None)

    # Check per-cell sums (if numeric)
    try:
        cell_sums = np.array(adata.X.sum(axis=1)).flatten()
        # Allow 0 if some cells are empty—so check only nonzero cells
        nonzero = cell_sums > 0
        if nonzero.sum() > 0:
            # Check relative tolerance of 1e-3 or absolute of 1e-6
            out["cells_scaled_to_target"] = np.isclose(
                cell_sums[nonzero], target_sum, rtol=1e-3, atol=1e-6
            ).all()
        else:
            out["cells_scaled_to_target"] = False
    except Exception:
        out["cells_scaled_to_target"] = False

    # Check if adata.X appears to be log1p
    arr = adata.X.A if hasattr(adata.X, "A") else adata.X
    mn, mx = float(np.min(arr)), float(np.max(arr))
    out["looks_log1p"] = (
        np.issubdtype(arr.dtype, np.floating) and (mn >= 0) and (mx < log_max_thresh)
    )

    return out


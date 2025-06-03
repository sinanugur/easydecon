import numpy as np
import random
import scanpy as sc
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from sklearn.model_selection import train_test_split

def simulate_pseudo_spots(
    sc_ref,             # AnnData (cells × genes), already normalized + log1p
    markers_df,         # DataFrame indexed by cell type, with a “gene_id_column” listing marker genes
    gene_id_column="names",
    celltype_column=None,
    n_spots=30000,
    spot_size=1000,
    random_state=0
):
    """
    Simulate `n_spots` pseudo‐bulk mixtures from single‐cell data.

    Parameters
    ----------
    sc_ref : AnnData
        Single‐cell AnnData containing only marker genes (normalized + log1p).
    markers_df : pandas.DataFrame
        Must be indexed by cell type. The index name should match a column
        in sc_ref.obs storing the same cell‐type labels.
    gene_id_column : str
        Column name in markers_df that contains gene IDs. Used only to check subset.
    celltype_column : str or None
        If provided, use this to look up sc_ref.obs[celltype_column] for each cell’s label.
        If None, defaults to markers_df.index.name.
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
    """
    np.random.seed(random_state)
    random.seed(random_state)

    if celltype_column is None:
        celltype_column = markers_df.index.name
    if celltype_column is None or celltype_column not in sc_ref.obs:
        raise ValueError(
            "simulate_pseudo_spots requires a valid `celltype_column` that matches sc_ref.obs."
        )

    # Subset sc_ref to exactly the marker genes
    common_genes = list(
        set(sc_ref.var_names).intersection(markers_df[gene_id_column].values)
    )
    if len(common_genes) == 0:
        raise ValueError("No overlap between sc_ref.var_names and markers_df[gene_id_column].")
    sc_sub = sc_ref[:, common_genes].copy()

    # Extract expression matrix and cell-type labels
    X_cells = sc_sub.X.toarray()                          # shape = (n_cells, n_genes)
    ct_labels = sc_sub.obs[celltype_column].values
    unique_ct = sorted(list(set(ct_labels)))
    K = len(unique_ct)
    ct2idx = {ct: i for i, ct in enumerate(unique_ct)}
    cell2idx = np.array([ct2idx[ct] for ct in ct_labels])  # shape = (n_cells,)

    # Precompute indices of cells for each cell type
    cells_by_ct = {i: np.where(cell2idx == i)[0] for i in range(K)}
    n_genes = X_cells.shape[1]

    X_pseudo = np.zeros((n_spots, n_genes), dtype=float)
    P_pseudo = np.zeros((n_spots, K), dtype=float)

    for i in range(n_spots):
        # (a) Draw a random Dirichlet proportion vector
        p = np.random.dirichlet(alpha=np.ones(K), size=1).flatten()  # shape=(K,)

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

        # (e) Renormalize to total = 10k, then log1p again
        total_counts = counts_sum.sum()
        if total_counts > 0:
            normed = np.log1p((counts_sum / total_counts) * 1e4)
        else:
            normed = counts_sum

        X_pseudo[i, :] = normed
        P_pseudo[i, :] = p

    return X_pseudo, P_pseudo


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
        Compiled (no optimizer/loss yet) Keras Model.
    """
    inp = layers.Input(shape=(num_genes,), name="expr_input")
    x = inp
    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, activation="relu", name=f"dense_{i}")(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"drop_{i}")(x)
    x = layers.Dense(num_celltypes, activation=None, name="dense_out")(x)
    out = layers.Activation("softmax", name="prop")(x)

    model = models.Model(inputs=inp, outputs=out, name="DeconvNN")
    return model


def get_proportions_deeplearning(
    sc_adata,
    sp_adata,
    markers_df,
    gene_id_column="names",
    celltype_column=None,
    n_pseudo_spots=30000,
    pseudo_spot_size=1000,
    hidden_units=[512, 256],
    dropout_rate=0.2,
    epochs=50,
    batch_size=256,
    random_state=0,
    verbose=True
):
    """
    Train a Keras‐based deconvolution model on simulated pseudo‐spots and predict
    cell‐type proportions for each spot in sp_adata.

    Parameters
    ----------
    sc_adata : AnnData
        Single‐cell reference. Must contain raw or normalized counts and a column
        in .obs indicating cell‐type labels (matching markers_df.index).
    sp_adata : AnnData
        Spatial AnnData. Count matrix is in .X. Will be normalized internally.
    markers_df : pandas.DataFrame
        Indexed by cell type. Must have a column “gene_id_column” listing marker genes.
        Its index name (or `celltype_column`) must match sc_adata.obs keys for cell‐type.
    gene_id_column : str
        Column in markers_df that lists marker genes.
    celltype_column : str or None
        If provided, use this to look up sc_adata.obs[celltype_column] for each cell’s label.
        If None, defaults to markers_df.index.name.
    n_pseudo_spots : int
        Number of pseudo‐spots to simulate.
    pseudo_spot_size : int
        Number of cells (or total UMI‐equivalents) per pseudo‐spot.
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
        Print training progress if True.

    Returns
    -------
    preds : np.ndarray, shape (n_spots_real, K)
        Predicted cell‐type proportion matrix for sp_adata.
    sp_adata_out : AnnData
        The same sp_adata, but with new columns in .obs named “prop_{cell_type}”.
    """
    if celltype_column is None:
        celltype_column = markers_df.index.name
    if celltype_column is None or celltype_column not in sc_adata.obs:
        raise ValueError(
            "`celltype_column` must be provided or set as markers_df.index.name "
            "and must exist in sc_adata.obs."
        )

    # 1) Intersect genes among sc_adata, sp_adata, and markers_df
    common_genes = list(
        set(sc_adata.var_names)
        .intersection(sp_adata.var_names)
        .intersection(markers_df[gene_id_column].values)
    )
    if len(common_genes) < 10:
        raise ValueError(
            "Too few overlapping genes among sc_adata, sp_adata, and markers_df. "
            f"Found only {len(common_genes)} genes."
        )
    if verbose:
        print(f"Using {len(common_genes)} common marker genes.")

    # 2) Subset and normalize sc_adata
    sc_ref = sc_adata[:, common_genes].copy()
    #sc.pp.normalize_total(sc_ref, target_sum=1e4)
    #sc.pp.log1p(sc_ref)
    sc.pp.scale(sc_ref,max_value=10)

    # 3) Subset and normalize sp_adata
    sp_ref = sp_adata[:, common_genes].copy()
    sc.pp.normalize_total(sp_ref, target_sum=1e4)
    sc.pp.log1p(sp_ref)
    sc.pp.scale(sp_ref,max_value=10)

    # 4) Simulate pseudo‐spots
    X_pseudo, P_pseudo = simulate_pseudo_spots(
        sc_ref,
        markers_df,
        gene_id_column=gene_id_column,
        celltype_column=celltype_column,
        n_spots=n_pseudo_spots,
        spot_size=pseudo_spot_size,
        random_state=random_state
    )
    if verbose:
        print(f"Simulated {X_pseudo.shape[0]} pseudo‐spots with {X_pseudo.shape[1]} genes.")

    # 5) Build and compile NN
    num_genes = len(common_genes)
    cell_types = list(markers_df.index.unique())
    num_ct = len(cell_types)

    model = build_deconv_model(
        num_genes=num_genes,
        num_celltypes=num_ct,
        hidden_units=hidden_units,
        dropout_rate=dropout_rate
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss=losses.MeanSquaredError(),
        metrics=[losses.MeanAbsoluteError()]
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
    X_sp_real = sp_ref.X.toarray()  # shape = (n_spots_real, num_genes)
    preds = model.predict(X_sp_real)  # shape = (n_spots_real, num_ct)
    if verbose:
        print(f"Predicted proportions for {X_sp_real.shape[0]} real spots.")

    # 8) Write predictions back into sp_adata.obs
    sp_out = sp_adata.copy()
    for idx_ct, ct in enumerate(cell_types):
        sp_out.obs[f"prop_{ct}"] = preds[:, idx_ct]

    return preds, sp_out



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


#!/usr/bin/env python

import numpy as np
import pandas as pd
import scipy.sparse as sp
import spatialdata_io
import argparse
import pickle
import scanpy as sc


def parse_args():
    parser = argparse.ArgumentParser(description="Marker expression cell counts")
    parser.add_argument("--markers", "-m", type=str, required=True, help="Comma-separated list of marker genes")
    parser.add_argument("--layer", "-l", type=str, default=None, help="Layer to use")
    parser.add_argument("--use_raw", "-r", action="store_true", help="Use raw data")
    parser.add_argument("--bin_size", "-b", type=int, default=8, help="Bin size")
    parser.add_argument("--sample_id", "-s", type=str, default="C2248F2", help="Sample ID")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input VisiumHD directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output file")
    return parser.parse_args()

def marker_expression_cell_counts(
    adata,
    markers,
    layer=None,
    use_raw=False,
    restrict_to_var_names=True,
):
    """
    Count, for each marker gene, the number of cells with non-zero expression,
    and also counts for markers expressed "together":
      - any_markers_ncells: cells expressing at least one marker
      - all_markers_ncells: cells expressing all markers (intersection)

    Parameters
    ----------
    adata : anndata.AnnData
    markers : list[str]
    layer : str | None
        If provided, uses adata.layers[layer] instead of adata.X.
    use_raw : bool
        If True, uses adata.raw (and ignores `layer`).
    restrict_to_var_names : bool
        If True, drops markers not found. If False, raises if any missing.

    Returns
    -------
    summary : dict
        Includes counts and marker lists.
    per_gene : pd.DataFrame
        Index=marker gene, columns=['ncells_nonzero', 'frac_cells_nonzero'].
    """
    if use_raw:
        if adata.raw is None:
            raise ValueError("use_raw=True but adata.raw is None.")
        var_names = adata.raw.var_names
        X = adata.raw.X
    else:
        var_names = adata.var_names
        X = adata.layers[layer] if layer is not None else adata.X

    markers = list(markers)
    present = [g for g in markers if g in var_names]
    missing = [g for g in markers if g not in var_names]
    if missing and not restrict_to_var_names:
        raise KeyError(f"Missing markers in var_names: {missing}")

    n_cells = adata.n_obs
    if len(present) == 0:
        per_gene = pd.DataFrame(columns=["ncells_nonzero", "frac_cells_nonzero"])
        summary = {
            "n_cells": n_cells,
            "markers_requested": markers,
            "markers_present": [],
            "markers_missing": missing,
            "any_markers_ncells": 0,
            "all_markers_ncells": 0,
        }
        return summary, per_gene

    # Column indices for present genes
    gene_to_idx = {g: i for i, g in enumerate(var_names)}
    cols = np.array([gene_to_idx[g] for g in present], dtype=int)

    # Subset matrix to markers
    Xg = X[:, cols]

    # Non-zero masks
    if sp.issparse(Xg):
        nz = Xg > 0
        nz = nz.tocsr()  # ensure efficient row operations
        per_gene_counts = np.asarray(nz.sum(axis=0)).ravel()
        any_mask = np.asarray(nz.sum(axis=1)).ravel() > 0
        all_mask = np.asarray(nz.sum(axis=1)).ravel() == len(present)
    else:
        nz = Xg > 0
        per_gene_counts = nz.sum(axis=0).astype(int)
        any_mask = nz.any(axis=1)
        all_mask = nz.all(axis=1)

    per_gene = pd.DataFrame(
        {
            "ncells_nonzero": per_gene_counts.astype(int),
            "frac_cells_nonzero": per_gene_counts.astype(float) / float(n_cells),
        },
        index=pd.Index(present, name="gene"),
    ).sort_values("ncells_nonzero", ascending=False)

    summary = {
        "n_cells": n_cells,
        "markers_requested": markers,
        "markers_present": present,
        "markers_missing": missing,
        "any_markers_ncells": int(any_mask.sum()),
        "all_markers_ncells": int(all_mask.sum()),
        "any_markers_frac": float(any_mask.mean()),
        "all_markers_frac": float(all_mask.mean()),
    }
    return summary, per_gene

def main():
    args=parse_args()
    bin_size=args.bin_size
    sample_id=args.sample_id

    sdata=spatialdata_io.visium_hd(args.input,
                               bin_size=bin_size,dataset_id=sample_id,load_all_images=False)

    for table in sdata.tables.values():
        table.var_names_make_unique()
        sc.pp.calculate_qc_metrics(table, inplace=True)
        sc.pp.filter_genes(table, min_cells=3)
        sc.pp.filter_cells(table, min_counts=3)
        #table.layers["counts"] = table.X.copy()
        sc.pp.normalize_total(table, inplace=True,target_sum=1e4)
        sc.pp.log1p(table)
        #sc.pp.scale(table)
        #sc.pp.pca(table)
        #sc.pp.neighbors(table)
        #sc.tl.umap(table)

    #here we filter the spatial data to only include the cells that are in the table
    sdata.shapes[f"{sample_id}_square_00{bin_size}um"]=sdata.shapes[f"{sample_id}_square_00{bin_size}um"][sdata.shapes[f"{sample_id}_square_00{bin_size}um"].index.isin(sdata.tables[f"square_00{bin_size}um"].obs["location_id"])]

    mt_genes = sdata.tables[f"square_00{bin_size}um"].var_names.str.startswith('MT-')
    mt_genes



    sdata.tables[f"square_00{bin_size}um"] = sdata.tables[f"square_00{bin_size}um"][:,~mt_genes]
    table=sdata.tables[f"square_00{bin_size}um"]

    results=marker_expression_cell_counts(table,args.markers.split(","),layer=args.layer,use_raw=args.use_raw)
    print(results[0])
    with open(args.output, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()


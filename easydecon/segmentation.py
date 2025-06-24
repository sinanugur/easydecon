#!/usr/bin/env python
"""
Easydecon bin2cell segmentation helper script with CLI support
"""
import os
import argparse
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import tensorflow as tf
import bin2cell as b2c
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Helper script for bin2cell segmentation with StarDist via CLI parameters"
    )
    parser.add_argument("--sample-id", required=True,
                        help="Unique identifier for the sample (e.g. SampleD1_C2237F2_hirs)")
    parser.add_argument("--binned-002", required=True,
                        help="Path to binned outputs at 0.02µm resolution")
    parser.add_argument("--full-image", required=True,
                        help="Path to the raw source image (TIFF)")
    parser.add_argument("--spaceranger-image-path", required=True,
                        help="Path to Spaceranger cropped spatial images directory")
    parser.add_argument("--mpp", type=float, default=0.5,
                        help="Microns per pixel for HE scaling (default: 0.5)")
    parser.add_argument("--model","--stardist-model", default="2D_versatile_he",
                        help="StarDist model name (default: 2D_versatile_he)")
    parser.add_argument("--min-cells", default=10, type=int,
                        help="Minimum number of cells to filter genes (default: 10)")
    parser.add_argument("--min-counts", default=5, type=int,
                        help="Minimum number of counts to filter cells (default: 5)")
    parser.add_argument("--prob-thresh", type=float, default=0.20,
                        help="Probability threshold for StarDist (default: 0.20)")
    parser.add_argument("--out-dir", default="stardist",
                        help="Directory to save stardist and outputs (default: ./stardist)")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu",
                        help="Device to use for TensorFlow: 'gpu' or 'cpu' (default: gpu)")
    return parser.parse_args()


def run_bin2cell_segmentation(sample_id,
    binned_002,
    full_image,
    spaceranger_image_path,
    mpp=0.5,
    model="2D_versatile_he",
    prob_thresh=0.20,
    min_cells=10,
    min_counts=5,
    out_dir="stardist",
    device="gpu"):

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Configure TensorFlow
    # Configure TensorFlow device
    if device == 'cpu':
        # Disable all GPUs
        tf.config.set_visible_devices([], 'GPU')
    else:
        # Enable first GPU if available
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.set_visible_devices(physical_devices[0], 'GPU')

    # Read Visium data
    adata = b2c.read_visium(
        binned_002,
        source_image_path=full_image,
        spaceranger_image_path=spaceranger_image_path
    )
    adata.var_names_make_unique()

    # Filter
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, min_counts=min_counts)

    # Scale HE image
    he_path = os.path.join(out_dir, f"{sample_id}.he.tiff")
    b2c.scaled_he_image(adata, mpp=mpp, save_path=he_path)

    # Destripe
    b2c.destripe(adata)

    # StarDist segmentation
    npz_path = os.path.join(out_dir, f"{sample_id}.he.npz")
    b2c.stardist(
        image_path=he_path,
        labels_npz_path=npz_path,
        stardist_model=model,
        prob_thresh=prob_thresh
    )

    # Insert and expand labels
    b2c.insert_labels(
        adata,
        labels_npz_path=npz_path,
        basis="spatial",
        spatial_key="spatial_cropped",
        mpp=mpp,
        labels_key="labels_he"
    )
    b2c.expand_labels(
        adata,
        labels_key='labels_he',
        expanded_labels_key="labels_he_expanded"
    )

    # Filter background and convert to str
    bdata = adata[adata.obs['labels_he_expanded'] > 0].copy()
    bdata.obs['labels_he_expanded'] = bdata.obs['labels_he_expanded'].astype(str)

    # Bin-to-cell
    cdata = b2c.bin_to_cell(
        bdata,
        labels_key="labels_he_expanded",
        spatial_keys=["spatial", "spatial_cropped"]
    )

    # Plot bin-count per segment
    fig, ax = plt.subplots(figsize=(20, 20))
    sc.pl.spatial(
        cdata,
        color=["bin_count"],
        img_key=f"{mpp}_mpp",
        basis="spatial_cropped",
        ax=ax,
        show=False,
        colorbar_loc=None
    )
    ax.set_title('Bin-Counts per Segments', fontsize=16)
    scatter = ax.collections[0]

    # Inset colorbar
    cax = inset_axes(
        ax,
        width=0.1,
        height=1.5,
        loc='upper right',
        bbox_to_anchor=(0.95, 0.95),
        bbox_transform=ax.transAxes,
        borderpad=0
    )
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.ax.set_frame_on(True)
    cbar.ax.set_facecolor("white")
    cbar.outline.set_linewidth(1)
    cbar.outline.set_edgecolor('black')
    cbar.ax.tick_params(labelcolor='black')

    ax.axis('off')
    out_pdf = f"{sample_id}_bincounts.pdf"
    plt.savefig(out_pdf, dpi=300)

    # Write output
    out_h5ad = f"{sample_id}_bin2cell.h5ad"
    cdata.write_h5ad(out_h5ad)
    print(f"Outputs saved:\n  HE TIFF: {he_path}\n  Labels NPZ: {npz_path}\n  PDF plot: {out_pdf}\n  H5AD: {out_h5ad}")
    
    return cdata

def main():
    args = parse_args()
    run_bin2cell_segmentation(
        sample_id=args.sample_id,
        binned_002=args.binned_002,
        full_image=args.full_image,
        spaceranger_image_path=args.spaceranger_image_path,
        mpp=args.mpp,
        model=args.model,
        prob_thresh=args.prob_thresh,
        out_dir=args.out_dir,
        device=args.device
    )

if __name__ == "__main__":
    main()

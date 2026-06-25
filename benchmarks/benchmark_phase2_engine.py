"""Synthetic benchmark for the optimized Phase 2 scoring engine."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from easydecon.easydecon import _build_phase2_cache, get_clusters_by_similarity_on_tissue
from easydecon.config import set_n_jobs


def _synthetic_inputs(
    n_spots: int,
    n_genes: int,
    n_groups: int,
    markers_per_group: int,
    sparse_input: bool,
    seed: int = 123,
):
    rng = np.random.default_rng(seed)
    genes = [f"G{i:05d}" for i in range(n_genes)]
    expression = rng.poisson(0.2, size=(n_spots, n_genes)).astype(float)
    for group_idx in range(n_groups):
        start = (group_idx * markers_per_group) % max(1, n_genes)
        marker_positions = np.arange(start, start + markers_per_group) % n_genes
        spot_positions = np.arange(group_idx, n_spots, max(1, n_groups))
        expression[np.ix_(spot_positions, marker_positions)] += rng.poisson(
            3.0, size=(len(spot_positions), len(marker_positions))
        )
    x = sparse.csr_matrix(expression) if sparse_input else expression
    table = ad.AnnData(
        X=x,
        obs=pd.DataFrame(index=[f"spot_{i:05d}" for i in range(n_spots)]),
        var=pd.DataFrame(index=genes),
    )

    marker_rows = []
    for group_idx in range(n_groups):
        start = (group_idx * markers_per_group) % max(1, n_genes)
        marker_positions = np.arange(start, start + markers_per_group) % n_genes
        for rank, gene_idx in enumerate(marker_positions):
            marker_rows.append(
                {
                    "group": f"group_{group_idx:03d}",
                    "names": genes[gene_idx],
                    "logfoldchanges": float(markers_per_group - rank),
                    "scores": float(markers_per_group - rank),
                }
            )
    markers = pd.DataFrame(marker_rows)
    return table, markers


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-spots", type=int, default=1000)
    parser.add_argument("--n-genes", type=int, default=1000)
    parser.add_argument("--n-groups", type=int, default=8)
    parser.add_argument("--markers-per-group", type=int, default=40)
    parser.add_argument("--method", default="sum")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--sparse", action="store_true", help="Use CSR sparse X.")
    parser.add_argument(
        "--candidate-pruning",
        action="store_true",
        help="Benchmark with a deterministic synthetic Phase 2 candidate mask.",
    )
    parser.add_argument(
        "--candidate-fraction",
        type=float,
        default=0.5,
        help="Approximate fraction of location-group pairs kept when pruning.",
    )
    args = parser.parse_args(argv)
    if not 0 < args.candidate_fraction <= 1:
        raise ValueError("--candidate-fraction must be > 0 and <= 1.")
    set_n_jobs(args.n_jobs)

    table, markers = _synthetic_inputs(
        n_spots=args.n_spots,
        n_genes=args.n_genes,
        n_groups=args.n_groups,
        markers_per_group=args.markers_per_group,
        sparse_input=args.sparse,
    )
    cache = _build_phase2_cache(
        markers,
        spatial_gene_names=table.var_names,
        method=args.method,
        gene_id_column="names",
        similarity_by_column="logfoldchanges",
        weight_column="logfoldchanges",
        min_markers=1,
    )
    candidate_mask = None
    if args.candidate_pruning:
        rng = np.random.default_rng(2026)
        mask_values = rng.random((args.n_spots, args.n_groups)) < args.candidate_fraction
        empty_rows = ~mask_values.any(axis=1)
        if empty_rows.any() and args.n_groups > 0:
            mask_values[empty_rows, rng.integers(0, args.n_groups, size=empty_rows.sum())] = True
        candidate_mask = pd.DataFrame(
            mask_values,
            index=table.obs.index,
            columns=cache.groups,
        )

    elapsed = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        result = get_clusters_by_similarity_on_tissue(
            table,
            markers,
            method=args.method,
            min_markers=1,
            add_to_obs=False,
            verbose=False,
            _candidate_mask=candidate_mask,
        )
        elapsed.append(time.perf_counter() - start)

    best = min(elapsed)
    total_pairs = int(args.n_spots * args.n_groups)
    candidate_pairs = (
        int(candidate_mask.to_numpy(dtype=bool).sum())
        if candidate_mask is not None
        else total_pairs
    )
    print(f"method={args.method}")
    print(f"sparse_input={args.sparse}")
    print(f"spots={args.n_spots}")
    print(f"total_genes={args.n_genes}")
    print(f"groups={args.n_groups}")
    print(f"markers_per_group={args.markers_per_group}")
    print(f"n_jobs={args.n_jobs}")
    print(f"candidate_pruning={args.candidate_pruning}")
    print(f"total_location_group_pairs={total_pairs}")
    print(f"candidate_pairs={candidate_pairs}")
    print(f"candidate_fraction={candidate_pairs / total_pairs if total_pairs else 0:.4f}")
    print(f"elapsed_seconds_best={best:.4f}")
    print(f"spots_per_second_best={args.n_spots / best:.2f}")
    print(f"evaluated_pairs_per_second_best={candidate_pairs / best:.2f}")
    print(f"extracted_expression_genes={len(cache.expression_genes)}")
    print(f"marker_union_genes={len(cache.marker_union)}")
    print(f"result_shape={result.shape}")


if __name__ == "__main__":
    main()

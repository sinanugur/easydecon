"""Benchmark the end-to-end easydecon workflow on deterministic synthetic data."""

import argparse
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd

import easydecon as ed
from easydecon.config import config

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples._synthetic import make_synthetic_spatial_and_markers


def run_benchmark(
    *,
    n_spots=500,
    n_genes=200,
    n_celltypes=3,
    method="wjaccard",
    filtering_algorithm="quantile",
    n_jobs=1,
    repeat=3,
):
    sdata, markers_df = make_synthetic_spatial_and_markers(
        n_spots=n_spots,
        n_genes=n_genes,
        n_celltypes=n_celltypes,
    )
    previous_n_jobs = config.n_jobs
    rows = []
    try:
        ed.set_n_jobs(n_jobs)
        for repeat_index in range(1, repeat + 1):
            run_sdata = sdata.copy()
            start = perf_counter()
            result = ed.run_easydecon(
                sdata=run_sdata,
                markers_df=markers_df,
                filtering_algorithm=filtering_algorithm,
                method=method,
                return_result_object=True,
                verbose=False,
            )
            runtime_seconds = perf_counter() - start
            assignment_column = result.diagnostics.get("results_column")
            if assignment_column not in result.assigned_labels.columns:
                assignment_column = result.assigned_labels.columns[0]
            rows.append(
                {
                    "repeat": repeat_index,
                    "n_spots": n_spots,
                    "n_genes": n_genes,
                    "n_celltypes": n_celltypes,
                    "method": method,
                    "filtering_algorithm": filtering_algorithm,
                    "runtime_seconds": runtime_seconds,
                    "n_assigned": int(
                        result.assigned_labels[assignment_column].notna().sum()
                    ),
                    "posterior_available": result.posterior_df is not None,
                }
            )
    finally:
        ed.set_n_jobs(previous_n_jobs)
    return pd.DataFrame(rows)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-spots", type=int, default=500)
    parser.add_argument("--n-genes", type=int, default=200)
    parser.add_argument("--n-celltypes", type=int, default=3)
    parser.add_argument("--method", default="wjaccard")
    parser.add_argument("--filtering-algorithm", default="quantile")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    results = run_benchmark(
        n_spots=args.n_spots,
        n_genes=args.n_genes,
        n_celltypes=args.n_celltypes,
        method=args.method,
        filtering_algorithm=args.filtering_algorithm,
        n_jobs=args.n_jobs,
        repeat=args.repeat,
    )
    print(results.to_string(index=False))
    return results


if __name__ == "__main__":
    main()

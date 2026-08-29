"""Command-line runner for deterministic easydecon synthetic validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.synthetic_validation import (
    default_validation_configurations,
    plot_accuracy_by_scenario,
    plot_candidate_reduction,
    plot_coverage_vs_accuracy,
    plot_runtime_by_configuration,
    run_validation_suite,
    summarize_validation_results,
    validation_metadata,
)


def _parse_csv(value):
    if value is None or value == "":
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_int_csv(value):
    return [int(item) for item in _parse_csv(value)]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default="clean,dropout,shared_markers")
    parser.add_argument("--configurations", default="")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--dense", action="store_true", help="Use dense matrices instead of CSR.")
    parser.add_argument("--output-dir", default="validation_output")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--n-groups", type=int, default=4)
    parser.add_argument("--n-reference-cells-per-group", type=int, default=40)
    parser.add_argument("--n-spots-per-group", type=int, default=35)
    parser.add_argument("--n-genes", type=int, default=400)
    parser.add_argument("--markers-per-group", type=int, default=20)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    scenarios = _parse_csv(args.scenarios)
    seeds = _parse_int_csv(args.seeds)
    if args.configurations:
        configurations = _parse_csv(args.configurations)
    else:
        configurations = default_validation_configurations()

    dataset_kwargs = {
        "n_groups": args.n_groups,
        "n_reference_cells_per_group": args.n_reference_cells_per_group,
        "n_spots_per_group": args.n_spots_per_group,
        "n_genes": args.n_genes,
        "markers_per_group": args.markers_per_group,
    }
    metrics_df, details = run_validation_suite(
        scenarios=scenarios,
        configurations=configurations,
        random_states=seeds,
        repeat=args.repeat,
        sparse=not args.dense,
        verbose=args.verbose,
        dataset_kwargs=dataset_kwargs,
    )
    summary_df = summarize_validation_results(metrics_df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "validation_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "validation_summary.csv", index=False)
    metadata = validation_metadata(
        vars(args),
        scenarios=scenarios,
        configurations=configurations,
        seeds=seeds,
        repeat=args.repeat,
        sparse_mode=not args.dense,
    )
    with (output_dir / "validation_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    if details.get("confusion"):
        import pandas as pd

        pd.concat(details["confusion"], ignore_index=True).to_csv(
            output_dir / "validation_confusion.csv", index=False
        )
    if details.get("pruning_comparisons"):
        import pandas as pd

        pd.DataFrame(details["pruning_comparisons"]).to_csv(
            output_dir / "validation_pruning_comparisons.csv", index=False
        )

    try:
        plot_specs = [
            ("accuracy_by_scenario.png", plot_accuracy_by_scenario, summary_df),
            ("coverage_vs_accuracy.png", plot_coverage_vs_accuracy, metrics_df),
            ("runtime_by_configuration.png", plot_runtime_by_configuration, summary_df),
            ("candidate_reduction.png", plot_candidate_reduction, metrics_df),
        ]
        for filename, func, frame in plot_specs:
            if frame.empty:
                continue
            fig, _ = func(frame)
            fig.savefig(output_dir / filename, dpi=150)
            import matplotlib.pyplot as plt

            plt.close(fig)
    except Exception as exc:
        if args.verbose:
            print(f"Skipping validation plots: {exc}")

    if args.verbose:
        print(f"Wrote validation outputs to {output_dir}")
    return metrics_df, summary_df, metadata


if __name__ == "__main__":
    main()

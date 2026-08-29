# Synthetic workflow benchmark

Run the lightweight benchmark from the repository root:

```bash
python benchmarks/benchmark_synthetic_workflow.py --n-spots 500 --n-genes 200 --repeat 3
```

Quantile filtering is the recommended default for quick comparisons.
Permutation filtering can be substantially slower, while NB filtering depends
on raw counts in the AnnData `counts` layer. The `--n-jobs` option controls
joblib parallelism during phase 2.

## Phase 2 engine benchmark

Benchmark only the Phase 2 similarity engine with deterministic synthetic data:

```bash
python benchmarks/benchmark_phase2_engine.py --method sum --n-spots 1000 --n-genes 1000 --repeat 3
python benchmarks/benchmark_phase2_engine.py --method wjaccard --sparse --repeat 3
```

The script reports elapsed time, spots per second, total genes, extracted
expression genes, marker-union genes, input density mode, and method. Timing is
informational and is not asserted in pytest. Use `--n-jobs` to control Phase 2
joblib parallelism.

## Synthetic validation suite

Run the deterministic synthetic validation suite:

```bash
python benchmarks/run_synthetic_validation.py
```

Target a smaller method/scenario matrix:

```bash
python benchmarks/run_synthetic_validation.py \
    --scenarios clean,dropout,difficult \
    --configurations \
    known_shared_ucell,reference_max_other_ucell,reference_phase_specific_ucell \
    --seeds 0,1,2 \
    --repeat 3 \
    --output-dir validation_output
```

The suite creates `validation_metrics.csv`, `validation_summary.csv`, and
`validation_metadata.json`, plus optional plots when matplotlib is available.
Synthetic data encode known marker assumptions, so results test implementation
behavior under those assumptions. They are not a substitute for real annotated
spatial datasets. Runtime comparisons depend on hardware and environment.
Positive-threshold candidate pruning can alter results; zero-threshold pruning
is checked for posterior-equivalence diagnostics where compatible.

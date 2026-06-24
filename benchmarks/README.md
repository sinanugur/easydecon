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

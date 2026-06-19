# Synthetic workflow benchmark

Run the lightweight benchmark from the repository root:

```bash
python benchmarks/benchmark_synthetic_workflow.py --n-spots 500 --n-genes 200 --repeat 3
```

Quantile filtering is the recommended default for quick comparisons.
Permutation filtering can be substantially slower, while NB filtering depends
on raw counts in the AnnData `counts` layer. The `--n-jobs` option controls
joblib parallelism during phase 2.

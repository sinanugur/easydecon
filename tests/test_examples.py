from benchmarks.benchmark_synthetic_workflow import run_benchmark
from examples import synthetic_niches, synthetic_quickstart, synthetic_scanpy_markers


def test_synthetic_quickstart_runs():
    result, summary, marker_summary = synthetic_quickstart.main(
        return_outputs=True,
        n_spots=30,
        n_genes=20,
        n_celltypes=3,
    )

    assert not result.assigned_labels.empty
    assert not summary.empty
    assert not marker_summary.empty


def test_synthetic_scanpy_markers_runs():
    result = synthetic_scanpy_markers.main(
        return_outputs=True,
        n_spots=30,
        n_genes=20,
        n_celltypes=3,
        n_cells=45,
    )

    assert not result.markers_df.empty
    assert result.diagnostics["markers"]["generated_rank_genes_groups"] is True


def test_synthetic_niches_runs():
    _, niches, smoothed, composition = synthetic_niches.main(
        return_outputs=True,
        n_spots=30,
        n_genes=20,
        n_celltypes=3,
    )

    assert niches.shape[0] == 30
    assert smoothed.shape[0] == 30
    assert not composition.empty


def test_benchmark_script_smoke():
    results = run_benchmark(
        n_spots=30,
        n_genes=20,
        n_celltypes=3,
        repeat=1,
        n_jobs=1,
    )

    assert results.shape[0] == 1
    assert "runtime_seconds" in results.columns
    assert results.loc[0, "runtime_seconds"] >= 0

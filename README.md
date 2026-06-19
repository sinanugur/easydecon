<img src="easydecon-logo.png" alt="easydecon logo" width="130" />

[![PyPI version](https://badge.fury.io/py/easydecon.svg)](https://badge.fury.io/py/easydecon)

# easydecon

easydecon provides marker-gene-based similarity, deconvolution, assignment,
diagnostics, and spatial niche utilities for single-cell and spatial
transcriptomics. Core workflows accept AnnData directly; SpatialData support is
available as an optional extra.

## Installation

Install from PyPI:

```bash
python -m pip install easydecon
```

For development and tests:

```bash
python -m pip install -e ".[test]"
```

Install optional functionality only when needed:

```bash
python -m pip install -e ".[spatial]"
python -m pip install -e ".[deseq]"
python -m pip install -e ".[spatial,deseq,test]"
```

SpatialData extras are only required for SpatialData containers and related
plotting/query helpers. PyDESeq2 is only required for
`marker_method="pydeseq2"`; core AnnData workflows require neither extra.

### Release/install check

```bash
python -m pip install -e ".[test]"
python -m pytest
```

## Quickstart with an existing marker DataFrame

Marker tables need at least `group` and `names` columns.

```python
import easydecon as ed

result = ed.run_easydecon(
    sdata=sdata,
    markers_df=markers_df,
    filtering_algorithm="quantile",
    method="wjaccard",
    return_result_object=True,
    verbose=False,
)

print(result.assigned_labels.head())
print(result.posterior_df.head())
```

## Generate Scanpy markers

Scanpy marker generation expects normalized and log-transformed expression in
the single-cell AnnData object.

```python
result = ed.run_easydecon(
    sdata=sdata,
    adata=sc_adata,
    groupby="cell_type",
    marker_method="scanpy",
    filtering_algorithm="quantile",
    return_result_object=True,
    verbose=False,
)
```

## Generate pseudobulk PyDESeq2 markers

PyDESeq2 requires raw, non-negative integer counts plus biological sample or
replicate labels.

```python
result = ed.run_easydecon(
    sdata=sdata,
    adata=sc_adata,
    groupby="cell_type",
    sample_col="sample_id",
    marker_method="pydeseq2",
    layer="counts",
    min_cells_per_group=20,
    min_replicates_per_condition=2,
    filtering_algorithm="quantile",
    return_result_object=True,
    verbose=False,
)
```

Do not treat individual cells as independent DESeq2 replicates.

## Niche detection

```python
niches, smoothed = ed.detect_niches_from_easydecon_result(
    sdata,
    result,
    n_neighbors=6,
    n_niches=5,
)
composition = ed.summarize_niche_compositions(smoothed, niches)
```

## QC summaries

```python
summary = ed.summarize_easydecon_result(result, sdata=sdata)
marker_summary = ed.summarize_marker_table(result.markers_df)

print(summary)
print(marker_summary)
```

Use `result.posterior_df` and these diagnostics before relying on hard spatial
assignments. With a list-style `marker_genes` mask workflow,
`result.posterior_df` is `None`; use `result.assignment_df` or
`result.phase2_result` instead.

## API overview

- `ed.run_easydecon`
- `ed.read_markers_dataframe`
- `ed.summarize_easydecon_result`
- `ed.summarize_marker_table`
- `ed.detect_niches_from_easydecon_result`
- `ed.summarize_niche_compositions`
- `ed.set_n_jobs`
- `ed.set_batch_size`

## Available methods

- Marker generation: `auto`, `existing`, `scanpy`, or `pydeseq2` (`deseq2` and `pseudobulk_deseq2` are aliases).
- Phase 1 filtering: `quantile`, `permutation`, or `nb`.
- Phase 2 similarity: `wjaccard`, `auc`, `cosine`, `correlation`, `jaccard`, `overlap`, `sum`, `mean`, `median`, or `euclidean`.
- Assignment: `max`, `zmax`, or `hybrid`.

Use `filtering_algorithm="quantile"` for quick tests. Permutation and NB
filtering can be more computationally demanding. Use `verbose=False` to
suppress progress output in automated runs.

## Examples and benchmarks

```bash
python examples/synthetic_quickstart.py
python examples/synthetic_scanpy_markers.py
python examples/synthetic_niches.py
python benchmarks/benchmark_synthetic_workflow.py --repeat 3
```

These scripts build deterministic synthetic AnnData objects in memory. They are
usage templates and smoke tests, not biological performance benchmarks. See
[`examples/README.md`](examples/README.md) and
[`benchmarks/README.md`](benchmarks/README.md).

## Marker DataFrame schema

Required canonical columns:

- `group`: cell type or cluster label
- `names`: gene symbol or gene identifier

Recommended columns are `logfoldchanges`, `pvals_adj`, and `scores`.
easydecon standardizes common aliases such as `cell_type`, `gene`,
`log2FoldChange`, and `padj`, then adds `marker_rank` and `marker_source`.

## Notes and pitfalls

- Marker-gene results depend strongly on marker quality and specificity.
- Gene identifiers must match between single-cell and spatial data.
- Scanpy marker generation expects normalized/log-transformed expression.
- PyDESeq2 pseudobulk requires raw integer counts and biological sample labels.
- Individual cells are not independent DESeq2 replicates.
- Spatial spots or bins can contain mixtures, so interpret hard labels cautiously.
- Inspect `result.posterior_df` and diagnostic summaries before using assignments downstream.

## License and release information

easydecon is released under the MIT License. See [`LICENSE`](LICENSE),
[`CHANGELOG.md`](CHANGELOG.md), and [`RELEASE.md`](RELEASE.md).

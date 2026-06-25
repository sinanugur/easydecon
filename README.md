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

## How easydecon works

1. Marker preparation reads, standardizes, or generates marker tables.
2. Phase 1 measures marker expression and creates normalized priors for which cell types are plausible at each spatial location.
3. Phase 2 computes marker-profile similarity and transforms it into likelihoods.
4. Priors and likelihoods are combined into `posterior_df` when available.
5. Final assignment turns `assignment_df` into hard labels in `assigned_labels`.

Zero priors generally gate posterior probabilities when `prior_weight > 0`.
With list-style `marker_genes`, Phase 1 is used as a mask and
`posterior_df` is `None`.

## Understanding the result

```python
result.markers_df       # selected spatial-compatible markers
result.phase1_result    # raw/thresholded Phase 1 marker-expression evidence
result.priors_df        # row-normalized Phase 1 priors
result.phase2_result    # raw marker-profile similarity evidence
result.likelihoods_df   # normalized Phase 2 evidence
result.posterior_df     # preferred probabilistic output, or None
result.assignment_df    # exact matrix used for final assignment
result.assigned_labels  # hard labels added for plotting/annotation
result.diagnostics      # QC and reproducibility metadata
```

Use `posterior_df` for downstream probabilistic analyses, `assigned_labels` for
hard maps, `priors_df` for presence gating, and `phase2_result` for raw
similarity inspection. See [docs/results.md](docs/results.md) for details.

## Optional Phase 2 candidate pruning

Phase 2 candidate pruning can skip scoring cell types with zero Phase 1 prior
at a spatial location:

```python
result = ed.run_easydecon(
    sdata,
    markers_df=markers_df,
    phase2_candidate_pruning=True,
    phase2_candidate_threshold=0.0,
    return_result_object=True,
)
```

With `phase2_candidate_threshold=0.0`, candidate groups are derived from
positive Phase 1 priors and normally preserve the posterior when
`prior_weight > 0`, while avoiding unnecessary Phase 2 scores. A positive
threshold is more aggressive and can change likelihoods, posteriors, and hard
assignments. The default is disabled. Full hierarchical refinement can use
child Phase 1 priors for pruning; `refine_group(mode="phase2")` cannot because
it does not calculate child priors.

## Refining a broad cell type into subclusters

Use `refine_group` to split one coarse parent group into subtype labels without
rerunning the full parent analysis.

```python
refined = ed.refine_group(
    sdata,
    parent_result=result,
    parent_group="Myeloid",
    markers_df=myeloid_subcluster_markers,
    mode="phase2",
    parent_source="priors",
    parent_threshold=0.0,
)
```

`mode="phase2"` uses the previous Myeloid score as the parent gate and runs
only subtype similarity. `mode="full"` uses the same parent gate, then
calculates new subtype-specific Phase 1 priors and Phase 2 likelihoods.

```python
refined = ed.refine_group(
    sdata,
    parent_result=result,
    parent_group="Myeloid",
    markers_df=myeloid_subcluster_markers,
    mode="full",
    parent_source="priors",
    parent_threshold=0.0,
)

refined.conditional_df  # subtype probabilities within Myeloid
refined.absolute_df     # subtype values scaled by the parent Myeloid score
refined.assigned_labels # hard subtype labels
```

Locations outside the Myeloid gate remain zero and unassigned. See
[docs/refinement.md](docs/refinement.md).

## Simple visualization

```python
import matplotlib.pyplot as plt
import numpy as np

table = ed.get_table(sdata)
coords = np.asarray(table.obsm["spatial"])
cell_type = result.posterior_df.columns[0]
values = result.posterior_df[cell_type].reindex(table.obs.index).fillna(0)

fig, ax = plt.subplots(figsize=(6, 6))
points = ax.scatter(coords[:, 0], coords[:, 1], c=values, s=8)
fig.colorbar(points, ax=ax, label=f"{cell_type} posterior")
ax.set_title(f"{cell_type} spatial posterior")
ax.set_aspect("equal")
fig.tight_layout()
```

More recipes are in [docs/visualization.md](docs/visualization.md).

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

### Reuse generated markers across spatial datasets

Use `prepare_markers` when differential expression is expensive and you want to
reuse the same marker preparation for more than one spatial gene universe.

```python
prepared = ed.prepare_markers(
    sc_adata,
    marker_method="scanpy",
    groupby="cell_type",
)

result_a = ed.run_easydecon(
    spatial_a,
    prepared_markers=prepared,
    return_result_object=True,
)

result_b = ed.run_easydecon(
    spatial_b,
    prepared_markers=prepared,
    return_result_object=True,
)
```

Differential expression runs once. Marker filtering is repeated cheaply for
each spatial gene universe. `PreparedMarkers` does not mutate the single-cell
AnnData; recreate it after changing expression values, annotations, sample
labels, or DE parameters.

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

## Generate reference-profile markers

Reference-profile markers are selected from cell-type mean profiles and
detection fractions after each cell is normalized by its total abundance.

```python
prepared = ed.prepare_markers(
    sc_adata,
    marker_method="reference",
    groupby="cell_type",
    layer="counts",
    reference_contrast="max_other",
)

result = ed.run_easydecon(
    sdata,
    prepared_markers=prepared,
    filtering_algorithm="quantile",
    return_result_object=True,
)
```

`max_other` favors markers that distinguish a group from its strongest
competitor. `mean_other` is more permissive and compares against the average
competing profile. This method does not calculate statistical p-values; its
threshold defaults are starting points and may need dataset-specific tuning.
It is inspired by reference-profile marker-selection concepts, but it is not an
implementation of the RCTD model. See [docs/reference_markers.md](docs/reference_markers.md).

## UCell-like Phase 2 scoring

Use `method="ucell"` when you want Phase 2 evidence to come from within-location
gene ranks rather than weighted marker overlap.

```python
result = ed.run_easydecon(
    sdata,
    markers_df=markers_df,
    filtering_algorithm="quantile",
    method="ucell",
    min_markers=3,
    top_n_markers=50,
    return_result_object=True,
)
```

If your marker table has a `marker_role` column, `positive` and `identity`
markers contribute positive evidence, `negative` markers subtract evidence,
and `presence` markers are ignored by this Phase 2 method. Missing roles are
treated as positive. Tune this scorer with `ucell_max_rank`,
`ucell_negative_weight`, `expression_threshold`, `recovery_power`,
`drop_shared_markers`, and `top_n_markers`. See [docs/ucell.md](docs/ucell.md).

## Phase-specific markers

By default, `marker_roles="shared"` preserves the historical behavior: the same
selected marker table is used for Phase 1 and Phase 2.

Use `marker_roles="phase_specific"` with reference-profile markers when you
want sensitive presence markers for Phase 1 and more specific identity markers
for Phase 2. With `method="ucell"`, negative markers can additionally subtract
evidence.

```python
prepared = ed.prepare_markers(
    sc_adata,
    marker_method="reference",
    marker_roles="phase_specific",
    groupby="cell_type",
    layer="counts",
)

result = ed.run_easydecon(
    sdata,
    prepared_markers=prepared,
    marker_roles="phase_specific",
    method="ucell",
    return_result_object=True,
)
```

Reference-profile generation can emit `presence`, `identity`, and `negative`
rows. The same gene may intentionally be both `presence` and `identity` for the
same group. Scanpy and DESeq-derived markers are not assigned roles
automatically; provide a manually annotated `marker_role` column if you want to
use phase-specific routing with those marker sources.

Phase-specific routing also works in hierarchical refinement:

```python
refined = ed.refine_group(
    sdata,
    parent_result=result,
    parent_group="Myeloid",
    prepared_markers=myeloid_prepared,
    marker_roles="phase_specific",
    mode="full",
    method="ucell",
)
```

See [docs/marker_roles.md](docs/marker_roles.md).

## Phase 2 performance

Phase 2 precomputes marker structures and avoids converting the complete
spatial matrix to a dense DataFrame. Methods whose formulas depend only on
marker genes use the marker union, while Jaccard, overlap, and weighted
Jaccard retain the full gene row to preserve their original definitions.

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
- `ed.prepare_markers`
- `ed.select_prepared_markers`
- `ed.summarize_easydecon_result`
- `ed.summarize_marker_table`
- `ed.detect_niches_from_easydecon_result`
- `ed.summarize_niche_compositions`
- `ed.set_n_jobs`
- `ed.set_batch_size`

## Available methods

- Marker generation: `auto`, `existing`, `scanpy`, `reference`, or `pydeseq2` (`deseq2`, `pseudobulk_deseq2`, and `rctd_like` are aliases).
- Phase 1 filtering: `quantile`, `permutation`, or `nb`.
- Phase 2 similarity: `wjaccard`, `ucell`, `auc`, `cosine`, `correlation`, `jaccard`, `overlap`, `sum`, `mean`, `median`, or `euclidean`.
- Assignment: `max`, `zmax`, or `hybrid`.

Centered AUC uses zero as the default fallback for signatures with insufficient
evidence. Exact or near-tied assignment rows remain unassigned rather than
selecting the first cell-type column. Tune assignment and rank scoring with
`minimum_evidence`, `tie_tolerance`, `top_n_markers`, `recovery_power`,
`drop_shared_markers`, and `center_auc`.

Use `filtering_algorithm="quantile"` for quick tests. Permutation and NB
filtering can be more computationally demanding. Use `verbose=False` to
suppress progress output in automated runs.

## Examples and benchmarks

```bash
python examples/synthetic_quickstart.py
python examples/synthetic_scanpy_markers.py
python examples/synthetic_niches.py
python examples/visualize_results.py
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

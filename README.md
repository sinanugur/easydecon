<img src="easydecon-logo.png" alt="easydecon logo" width="130" />

[![PyPI version](https://badge.fury.io/py/easydecon.svg)](https://badge.fury.io/py/easydecon)

# easydecon

easydecon provides marker-gene based deconvolution, assignment, diagnostics,
and spatial niche utilities for single-cell references and spatial
transcriptomics. It accepts AnnData tables directly; SpatialData support is
available through an optional extra.

Project status: alpha (`0.1.6a0`). APIs are close to release but should still
be checked against the documentation before production use.

## Installation

```bash
python -m pip install easydecon
```

For development and tests:

```bash
python -m pip install -e ".[test]"
```

Optional extras:

```bash
python -m pip install -e ".[spatial]"  # SpatialData support
python -m pip install -e ".[deseq]"    # pseudobulk PyDESeq2 markers
python -m pip install -e ".[docs]"     # documentation build
```

## Quickstart

Marker tables need at least `group` and `names` columns, and marker gene
identifiers must match the spatial table `var_names`.

```python
import easydecon as ed

result = ed.run_easydecon(
    sdata=sdata,
    markers_df=markers_df,
    return_result_object=True,
    verbose=False,
)

print(result.posterior_df.head())
print(result.assigned_labels.head())
print(result.diagnostics)
```

### Signed differential-expression marker roles

For signed Scanpy, DESeq, or PyDESeq2 marker results, opt into
`marker_role_inference="signed"` with `marker_roles="shared"`. Positive log
fold changes become `positive` markers and negative log fold changes become
`negative` markers. Signed Scanpy scores and DESeq-style Wald statistics are
used only as optional direction-consistency checks; fold-change direction is
authoritative. Signed inference creates only positive and negative roles.

`marker_role_inference="scanpy_signed"` remains accepted as a backward-
compatible alias, but new code should use `"signed"`.

`posterior_df` contains relative support among tested marker groups, not
guaranteed absolute cell fractions. Hard assignments discard uncertainty.

## Documentation

Start with [docs/index.rst](docs/index.rst). The documentation covers:

* [installation and quickstart](docs/usage.rst)
* [workflow concepts](docs/workflow.md)
* [marker inputs](docs/marker_inputs.md)
* [results and interpretation](docs/results.md)
* [Scanpy markers](docs/scanpy_markers.md)
* [reference-profile markers](docs/reference_markers.md)
* [Phase 1](docs/phase1.md) and [Phase 2](docs/phase2.md) methods
* [refinement](docs/refinement.md)
* [visualization](docs/visualization.md)
* [synthetic validation](docs/validation.md)

## Marker methods

`prepare_markers` is the reusable marker-preparation entry point. It accepts an
AnnData reference, a marker DataFrame, a CSV/Excel marker file, or an existing
`PreparedMarkers` object, then returns a canonical marker table that is still
independent of any spatial gene universe. `run_easydecon` calls this controller
for you, and stores the resolved preparation as `result.prepared_markers` when
`return_result_object=True`.

```python
prepared = ed.prepare_markers(
    markers_df=deseq_df,
    source="deseq_table",
)

result = ed.run_easydecon(
    sdata,
    prepared_markers=prepared,
    return_result_object=True,
)
```

The prepared object can be reused on another spatial dataset; only inexpensive
spatial gene filtering and phase routing are repeated. `select_prepared_markers`
performs dataset-specific filtering, while internal phase routing applies the
workflow `top_n_genes`. `read_markers_dataframe` remains supported for backward
compatibility when a selected DataFrame is desired directly, but it is no
longer the recommended modern marker-preparation API.

Use `top_n_genes="auto"` to choose a deterministic, spatially usable marker
count independently per cell type (and per marker role when roles are present).
This opt-in mode removes undetected genes and applies a lightweight adaptive
quality cutoff after the usual marker filters. A usable requested DE score
determines marker order while adaptive quality determines the count; otherwise
adaptive quality supplies both. Integer and `None` behavior is unchanged. See
[marker inputs](docs/marker_inputs.md#automatic-spatial-marker-selection).

easydecon can use existing marker tables, existing Scanpy
`rank_genes_groups`, generated Scanpy markers, pseudobulk PyDESeq2 markers,
reference-profile markers, and reusable `PreparedMarkers`. Use `marker_method`
to select generated Scanpy, PyDESeq2, or reference-profile marker workflows
when preparing from AnnData.

Scanpy marker generation expects suitable normalized/log-transformed input.
PyDESeq2 requires raw integer counts and biological replicate labels; do not
treat individual cells as DESeq2 replicates.

## Validation

Synthetic validation scripts in `benchmarks/` are implementation checks and
smoke tests. They do not establish biological superiority.

```bash
python benchmarks/run_synthetic_validation.py \
    --scenarios clean,dropout,shared_markers \
    --output-dir validation_output
```

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
    filtering_algorithm="quantile",
    method="wjaccard",
    return_result_object=True,
    verbose=False,
)

print(result.posterior_df.head())
print(result.assigned_labels.head())
print(result.diagnostics)
```

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
* [UCell-like scoring](docs/ucell.md)
* [refinement](docs/refinement.md)
* [visualization](docs/visualization.md)
* [synthetic validation](docs/validation.md)

## Marker methods

easydecon can use existing marker tables, existing Scanpy
`rank_genes_groups`, generated Scanpy markers, pseudobulk PyDESeq2 markers,
reference-profile markers, and reusable `PreparedMarkers`.
Use `marker_method` to select generated Scanpy, PyDESeq2, or
reference-profile marker workflows.

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

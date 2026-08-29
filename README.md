<img src="easydecon-logo.png" alt="easydecon logo" width="130" />

[![PyPI version](https://badge.fury.io/py/easydecon.svg)](https://badge.fury.io/py/easydecon)

# easydecon

easydecon uses marker genes to deconvolve and assign cell types in spatial
transcriptomics data.

## Installation

```bash
python -m pip install easydecon
```

For development and testing:

```bash
python -m pip install -e ".[test]"
```

Optional extras:

```bash
python -m pip install -e ".[spatial]"  # SpatialData support
python -m pip install -e ".[deseq]"    # pseudobulk PyDESeq2 markers
python -m pip install -e ".[docs]"     # build the documentation
```

## Quickstart

Start with a spatial AnnData table (or a SpatialData object) and a CSV or
Excel marker file. The file needs `group` and `names` columns: `group` is the
cell type and `names` is the marker gene. Gene names in the file must match
the spatial table's `var_names`.

```python
import easydecon as ed

result = ed.run_easydecon(
    sdata=sdata,
    filename="markers.csv",
    return_result_object=True,
    verbose=False,
)

print(result.posterior_df.head())
print(result.assigned_labels.head())
print(result.diagnostics)
```

`posterior_df` shows the relative support for each tested cell type. It is not
necessarily an absolute cell-fraction estimate. `assigned_labels` gives a
single label per location, so it does not retain that uncertainty. Review the
diagnostics before using the assignments downstream.

## Documentation

Begin with [docs/index.rst](docs/index.rst). Useful guides include:

* [installation and quickstart](docs/usage.rst)
* [workflow concepts](docs/workflow.md)
* [marker inputs](docs/marker_inputs.md)
* [results and interpretation](docs/results.md)
* [Scanpy markers](docs/scanpy_markers.md)
* [reference-profile markers](docs/reference_markers.md)
* [Phase 1](docs/phase1.md) and [Phase 2](docs/phase2.md)
* [refinement](docs/refinement.md)
* [visualization](docs/visualization.md)

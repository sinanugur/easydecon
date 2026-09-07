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

Start with a spatial AnnData table (or a SpatialData object) and a signed DE
marker table. Gene names in the file must match the spatial table's
`var_names`.

```python
import easydecon as ed

prepared_markers = ed.prepare_markers(
    filename="pelka_toplevel_deseq_table.csv",
    source="toplevel_markers",
    marker_role_inference="signed",
    marker_role_inference_log2fc_min=0.25,
    marker_roles="shared",
)

result = ed.run_easydecon(sdata=sdata, prepared_markers=prepared_markers)
```

`"signed"` infers positive and negative roles from signed DE log fold changes
while preserving explicit `marker_role` values. `"scanpy_signed"` remains a
legacy alias. With `marker_roles="shared"`, the prepared DE table is reused by
both phases: Phase 1 uses positive/presence-style evidence, while UCell can
also use negative markers for discrimination.

`run_easydecon()` returns an `EasyDeconResult` by default. Useful fields are
`markers_df`, `phase1_result`, `phase2_result`, `priors_df`, `likelihoods_df`,
`posterior_df`, `assigned_labels`, and `diagnostics`.

The defaults implement the recommended workflow. Override only the controls
you need:

```python
result = ed.run_easydecon(
    sdata=sdata,
    prepared_markers=prepared_markers,

    # Marker selection
    top_n_genes="auto",
    log2fc_min=0.25,
    pval_cutoff=0.05,
    auto_marker_min=30,
    auto_marker_max=120,

    marker_roles="shared",

    # Phase 1
    filtering_algorithm="permutation",
    phase1_output_stat="minus_log10_p",
    aggregation_method="coverage",
    alpha=0.05,
    permutation_gene_pool_fraction="auto",
    n_subs=5,

    # Phase 2 and assignment
    method="ucell",
    assign_method="max",
    results_column="easydecon",
)
```

## Documentation

Refer to https://easydecon.readthedocs.io/en/latest/

### Project status and guides

The current software guides are [the documentation index](docs/index.rst),
[installation and usage](docs/usage.rst), [workflow](docs/workflow.md),
[marker inputs](docs/marker_inputs.md), [reference-profile markers](docs/reference_markers.md),
[Scanpy markers](docs/scanpy_markers.md), [Phase 1](docs/phase1.md),
[Phase 2](docs/phase2.md), [results](docs/results.md), [visualization](docs/visualization.md),
and [refinement](docs/refinement.md). Marker loading supports `marker_method`
routes including PyDESeq2 and reference-profile markers.

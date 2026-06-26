# Optional bin2cell segmentation helper

The package exposes a console script:

```text
run_bin2cell_segmentation = easydecon.segmentation:main
```

This helper is optional and experimental relative to the core marker workflow.
It depends on packages that are not part of the base easydecon install,
including `bin2cell`, TensorFlow, StarDist-compatible dependencies, Scanpy, and
Matplotlib. Install and validate those dependencies separately before using the
CLI.

## Command

```bash
run_bin2cell_segmentation \
    --sample-id Sample1 \
    --binned-002 /path/to/binned_outputs \
    --full-image /path/to/full_image.tif \
    --spaceranger-image-path /path/to/spaceranger/spatial \
    --out-dir stardist \
    --device cpu
```

Required arguments:

`--sample-id`
: Identifier used in output filenames.

`--binned-002`
: Path to binned Visium outputs at 0.02 micrometer resolution.

`--full-image`
: Path to the raw source image.

`--spaceranger-image-path`
: Path to the Spaceranger cropped spatial image directory.

Optional arguments and defaults:

| Argument | Default |
| --- | --- |
| `--mpp` | `0.5` |
| `--model`, `--stardist-model` | `2D_versatile_he` |
| `--min-cells` | `10` |
| `--min-counts` | `5` |
| `--prob-thresh` | `0.20` |
| `--nms-thresh` | `0.3` |
| `--out-dir` | `stardist` |
| `--device` | `gpu` |

`--device cpu` disables TensorFlow GPUs. `--device gpu` uses the first visible
GPU when TensorFlow reports one.

## Outputs

The helper writes the scaled HE image and StarDist label file inside
`out_dir`. It writes the bin-count PDF and segmented H5AD in the current
working directory:

* `<out_dir>/<sample_id>.he.tiff`
* `<out_dir>/<sample_id>.he.npz`
* `<sample_id>_bincounts.pdf`
* `<sample_id>_bin2cell.h5ad`

## Common errors

Missing optional dependencies raise an ImportError from
`run_bin2cell_segmentation`. TensorFlow/StarDist installation issues and image
path mismatches are environment-specific; verify the same paths with a small
bin2cell script before treating them as easydecon core workflow problems.

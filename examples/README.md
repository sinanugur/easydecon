# Synthetic examples

Run these examples from the repository root after installing easydecon:

```bash
python examples/synthetic_quickstart.py
python examples/synthetic_scanpy_markers.py
python examples/synthetic_niches.py
```

They generate small, deterministic AnnData objects in memory and should finish
quickly on a laptop. No large datasets are downloaded. The examples avoid
SpatialData and PyDESeq2; the Scanpy example uses the core Scanpy dependency.

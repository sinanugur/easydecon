# Changelog

## 0.1.6a0 - Unreleased

### Added

- Public `run_easydecon` alias.
- `EasyDeconResult` result object.
- Marker loading from DataFrame, file, Scanpy, and pseudobulk PyDESeq2.
- Shared schema helpers for marker tables and spatial table lookup.
- Spatial niche detection from `EasyDeconResult`-like objects.
- Diagnostics summaries for marker tables and workflow results.
- Synthetic examples and benchmark smoke script.
- Optional dependency groups for spatial, deseq, fast, test, and docs.

### Changed

- Lower-level scoring functions now use shared table lookup.
- Similarity scoring preserves spatial observation order.
- Verbose output can be suppressed with `verbose=False`.

### Fixed

- Missing marker genes in sum/mean/median scoring no longer raise `KeyError`.
- Duplicate marker genes in weighted Jaccard are handled safely.
- Empty masks return zero score matrices instead of crashing.

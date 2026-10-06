# Changelog

## 0.1.8

### Added

- Sample-aware pooled niche detection with within-sample spatial smoothing,
  balanced or capped model fitting, full-resolution prediction, expanded
  diagnostics, and optional fitted-model returns.
- Richer Visium HD simulation truth outputs, including realized cell counts and
  proportions, latent proportions, dominant cell type, purity, and complexity.

### Changed

- Automatic niche selection now evaluates the fitting subset, supports sampled
  silhouette scoring, and uses an inertia elbow criterion.
- Hierarchical refinement now defaults to full posterior-based refinement with
  the recommended automatic-marker, coverage, and UCell settings.
- Simulation gene filtering now follows the selected count source and validates
  Dirichlet parameters.

### Fixed

- Explicit marker-column selections take precedence when canonical and custom
  columns coexist.
- Bin2Cell segmentation now supplies explicit cropped-spatial and image keys.
- Zero-truncated simulated cell counts are resampled instead of clamped to one.

## 0.1.7

### Added

- Automatic marker selection and signed marker-role inference.
- Coverage-based Phase 1 aggregation and UCell Phase 2 scoring.

### Changed

- The recommended workflow now uses automatic marker selection, coverage-based
  Phase 1 evidence, and UCell Phase 2 scoring by default.

### Fixed

- Explicit marker-column arguments now take precedence over competing canonical
  columns during marker preparation.

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

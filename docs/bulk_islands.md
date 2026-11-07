# Bulk-program islands

`detect_bulk_islands()` finds connected regions whose existing mapped bulk
score passes explicit score, detection, and local-neighbor requirements. It
does not rerun expression scoring. Niche labels are added only after detection
and therefore cannot change island membership.

```python
import easydecon as ed

# `bulk_result` is an existing result from ed.map_bulk_signature(...).
islands = ed.detect_bulk_islands(
    bulk_result,
    coordinates=table.obsm["spatial"],       # arrays align positionally
    sample_labels=table.obs["sample_id"],    # pandas inputs align by ID
    niche_labels=table.obs["niche"],
    program="up",
    score_threshold=0.25,
    max_neighbor_distance=16,
)

islands.island_summary
islands.niche_composition
fig, axes = ed.plot_bulk_islands(islands)
```

Coordinates and distance thresholds use the units supplied by the caller;
pixels are not assumed to be micrometers. Neighbor support is evaluated once
against the original candidate mask, without smoothing or region expansion.

Thresholds are engineering choices and need sensitivity checks. A specimen
can legitimately contain zero islands; do not force a fixed top percentage of
locations into an island. Cross-specimen comparisons require consistent bulk
signatures, scoring settings, coordinate units, and island thresholds.

These regions are descriptive bulk-program islands. They are not statistical
discoveries, confirmed tumor regions, or malignant-cell classifications.

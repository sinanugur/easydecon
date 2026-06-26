# Glossary

spatial location
: General easydecon term for one row of the spatial expression table.

spot
: A platform-specific capture location.

bin
: A spatial aggregation unit, often from binned Visium HD data.

segmented cell
: A cell-shaped spatial unit from an upstream segmentation workflow.

marker
: A gene used as positive evidence for a marker group.

anti-marker
: A gene used as negative evidence for a group in UCell-like scoring.

presence marker
: A marker role used for Phase 1 presence evidence.

identity marker
: A marker role used for Phase 2 identity evidence.

Phase 1
: Marker-expression evidence step that produces raw evidence and normalized
  priors.

Phase 2
: Marker-profile similarity or rank-evidence step that produces raw evidence
  and likelihoods.

prior
: Row-normalized Phase 1 support for which groups are plausible at each
  spatial location.

likelihood
: Normalized Phase 2 evidence for each marker group at each spatial location.

posterior
: Relative support from combining priors and likelihoods.

assignment
: Hard label selected from `assignment_df`.

abstention
: Leaving a spatial location unassigned because evidence is zero, tied, or
  below assignment thresholds.

candidate pruning
: Optional Phase 2 optimization that scores only groups with sufficient Phase 1
  prior support.

PreparedMarkers
: Reusable, spatial-unfiltered marker preparation.

conditional subtype score
: Refined subtype support within parent-positive locations.

absolute subtype score
: Conditional subtype support multiplied by the parent score.

spatial niche
: Cluster of local composition profiles derived from posterior-like support
  matrices.

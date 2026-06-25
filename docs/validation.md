# Synthetic validation

The synthetic validation suite compares supported easydecon configurations on
deterministic data where the generating labels and marker assumptions are
known. It is intended for implementation validation, not for claims about
biological performance.

Run:

```bash
python benchmarks/run_synthetic_validation.py \
    --scenarios clean,dropout,shared_markers \
    --output-dir validation_output
```

The runner writes:

- `validation_metrics.csv`
- `validation_summary.csv`
- `validation_metadata.json`
- optional PNG plots when matplotlib is available

## Scenarios

- `clean`: strong unique markers, low dropout, no contamination.
- `dropout`: spatial marker dropout.
- `shared_markers`: more shared marker signal between groups.
- `library_shift`: location-to-location library-size variation.
- `contamination`: competing-group contamination and anti-marker signal.
- `mixed`: dominant plus secondary simulated group; `true_scores` stores the
  generating weights.
- `difficult`: dropout, shared markers, contamination, library shift, and
  weaker marker effects.

## Configurations

Default configurations cover known oracle marker tables, reference-profile
marker generation, shared versus phase-specific marker roles, UCell-like and
weighted Jaccard Phase 2 scoring, and candidate-pruned versus unpruned runs.
Known-marker configurations are labelled `marker_source="known"`; they isolate
Phase 2 behavior and should not be described as inferred markers.

To add a custom configuration, create a `ValidationConfiguration` in
`benchmarks.synthetic_validation` and pass it to `run_validation_suite`.

## Metrics

Assignment metrics report coverage, assigned accuracy, overall accuracy, macro
precision/recall/F1, and confusion counts. Coverage and assigned accuracy should
be interpreted together: high assigned accuracy with low coverage means the
method abstained often.

Ranking metrics use `posterior_df` when available, otherwise a row-normalized
non-negative assignment matrix. They include top-1 score accuracy, reciprocal
rank, true-group score, max score, and entropy. These scores measure relative
support among tested groups, not calibrated probabilities.

When posterior rows sum to one, the suite also reports multiclass Brier score
and negative log likelihood. In the `mixed` scenario, composition MAE/RMSE
compare posterior support with the simulation weights in `true_scores`; this is
not a claim that easydecon posteriors are absolute cell fractions.

## Candidate pruning

Zero-threshold candidate pruning derives Phase 2 candidates from positive Phase
1 priors and is compared with an unpruned run for posterior and assignment
equivalence. Positive thresholds are more aggressive and can legitimately
change results.

## Reproducibility

The generator uses `numpy.random.default_rng(random_state)`, deterministic gene
and group names, and records versions and command arguments in
`validation_metadata.json`. Biological metrics should be reproducible for
identical arguments; runtime metrics depend on hardware and environment.

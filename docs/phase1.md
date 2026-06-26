# Phase 1: marker-expression priors

Phase 1 is implemented by `common_markers_gene_expression_and_filter`. It
aggregates expression for marker genes at each spatial location, thresholds the
aggregate signal, and produces the raw Phase 1 evidence matrix. The workflow
clips this matrix at zero and row-normalizes it into `priors_df`.

## Marker representations

Phase 1 accepts:

`list[str]`
: A single marker set. The output group name is `common_group_name`
  (`"MarkerGroup"` by default). In `run_easydecon`, this list-style
  `marker_genes` path is a row-mask workflow and `posterior_df` is `None`.

`dict[str, list[str]]`
: A mapping from group name to marker genes.

`pandas.DataFrame`
: A marker table with group and gene columns, controlled by `celltype` and
  `gene_id_column` before canonicalization.

## Aggregation methods

Supported values are defined in `AGGREGATION_METHODS`:

| Method | Meaning |
| --- | --- |
| `sum` | Sum marker expression per spatial location. |
| `mean` | Mean marker expression per spatial location. |
| `median` | Median marker expression per spatial location. |
| `cs` | Composite score via the package's `composite_score` helper. |

## Filtering algorithms

Supported values are defined in `FILTERING_ALGORITHMS`:

`quantile`
: Computes a threshold from non-zero aggregate values for each group. This is
  the fastest practical starting option for quick runs.

`permutation`
: Builds a null distribution by repeatedly sampling random genes from a
  variable-gene pool. It can use an empirical threshold or a Gamma fit when
  `parametric=True`. Runtime scales with `num_permutations`, `subsample_size`,
  and `n_subs`.

`nb`
: Uses raw counts from `table.layers["counts"]`, a global negative-binomial
  approximation, and currently requires `aggregation_method="sum"`. It raises a
  clear error if the counts layer is missing.

## Output statistic

`phase1_output_stat` in `run_easydecon` is passed as `output_stat`.

`expression`
: Keep thresholded expression or count values.

`minus_log10_p`
: Return `-log10(p)` values for significant locations and zero otherwise.

`output_stat="minus_log10_p"` is invalid with `filtering_algorithm="quantile"`.
It is implemented for `permutation` and `nb`.

## Priors and posterior gating

The workflow converts Phase 1 evidence into `priors_df` by clipping negative
values to zero and row-normalizing. Rows with no Phase 1 evidence become all
zero. When `prior_weight > 0`, a zero prior usually prevents a marker group
from receiving posterior support.

Phase 1 is marker-based evidence. It is not statistically calibrated proof of
cell-type presence, and it depends strongly on marker quality and gene
identifier overlap.

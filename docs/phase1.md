# Phase 1: marker-expression priors

Permutation filtering is easydecon's standard Phase 1 method and the default
used by `run_easydecon`.

Phase 1 is implemented by `common_markers_gene_expression_and_filter`. It
aggregates marker expression for each group at each spatial location, compares
that evidence with the selected thresholding rule, and returns the raw Phase 1
evidence matrix. The workflow clips this matrix at zero and row-normalizes it
into `priors_df`.

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

## Filtering method decision table

| Method | Recommended use | Main advantage | Main limitation |
| --- | --- | --- | --- |
| `permutation` | Standard analysis | Null-based marker evidence | Computational cost |
| `quantile` | Fast exploration and debugging | Low computational cost | No random-gene null distribution |
| `nb` | Raw-count specialized workflows | Count-aware thresholding | Raw counts and current modeling assumptions required |

## Permutation filtering

Use:

```python
filtering_algorithm="permutation"
```

Permutation filtering aggregates the observed marker set and builds a null
distribution by repeatedly drawing random gene sets of the same size. The
background pool is based on the most variable genes according to the current
implementation. Retained evidence exceeds either the empirical null quantile
or the fitted parametric null cutoff.

Relevant parameters:

`num_permutations`
: Number of null draws across subsamples.

`alpha`
: Right-tail cutoff level for retained marker evidence.

`subsample_size`, `n_subs`
: Number of spatial locations sampled for null construction and how that work
  is split.

`subsample_signal_quantile`
: Drops the lowest and highest observed marker-expression locations before
  subsampling null locations.

`permutation_gene_pool_fraction`
: Controls the high-variance background gene pool used for permutation null
  sampling. A numeric value in `(0, 1]` selects that fraction of eligible
  genes. `"auto"` (the default) chooses a pool size using the marker-set size
  and the available spatial gene universe. The target group's own marker genes
  and genes with no spatial detection are excluded from its pool. Automatic
  selection remains variance-based; it does not expression-match control
  genes.

`random_state`
: Controls reproducible Phase 1 spot subsampling and random-gene draws. The
  default is `10`; use `None` for non-deterministic sampling.

`parametric`
: When `True`, fits the implemented Gamma null distribution. When `False`,
  uses the observed null quantile.

`aggregation_method`
: Aggregates expression for both observed and random marker sets.

`phase1_output_stat`
: Selects thresholded expression evidence or `-log10(p)` output.

Runtime increases with marker groups, permutations, and sampled locations.
Permutation filtering is the normal scientific workflow despite its greater
cost. Do not treat its output as perfectly calibrated proof of cell-type
presence; it is null-based marker evidence under the package's implemented
assumptions.

## Quantile filtering

Use:

```python
filtering_algorithm="quantile"
```

Quantile filtering is a fast exploratory shortcut. The standard Phase 1
workflow uses permutation filtering.

Quantile filtering computes the cutoff among nonzero aggregated values for
each marker group. It is deterministic, useful for smoke tests and low-cost
debugging, and much faster than permutation filtering. It is less
statistically grounded than the random-gene null and is sensitive to the
observed expression distribution.

`output_stat="minus_log10_p"` is invalid with quantile filtering because no
null p-value is computed.

## Negative-binomial filtering

Use:

```python
filtering_algorithm="nb"
```

NB filtering is a specialized raw-count alternative. It requires:

* raw non-negative counts in `table.layers["counts"]`; and
* `aggregation_method="sum"`.

For each marker group, the implementation computes observed marker-count sums,
library-size scaled expected marker sums, and a global negative-binomial
dispersion approximation controlled by `nb_global_theta`. With
`phase1_output_stat="expression"`, locations whose observed sum exceeds the
count-aware threshold retain the observed count sum. With
`phase1_output_stat="minus_log10_p"`, significant locations retain
`-log10(p)` and nonsignificant locations become zero.

NB is not the default. It is useful only when raw counts and the current global
dispersion assumptions are appropriate for the dataset.

## Priors and posterior gating

The workflow converts Phase 1 evidence into `priors_df` by clipping negative
values to zero and row-normalizing. Rows with no Phase 1 evidence become all
zero. When `prior_weight > 0`, a zero prior usually prevents a marker group
from receiving posterior support.

Phase 1 is marker-based evidence. It depends strongly on marker quality, gene
identifier overlap, spatial resolution, and the selected filtering algorithm.

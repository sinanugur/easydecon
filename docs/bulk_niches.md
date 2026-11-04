# Bulk-associated programs localized to spatial niches

This optional analysis connects fixed shared spatial niches to one external
bulk differential-expression ranking. It keeps three quantities separate:

1. one bulk enrichment result per fixed niche signature;
2. per-observation expression of the enrichment's supporting genes;
3. per-specimen spatial distance from a user-selected niche interface.

It does not infer malignancy, tumor probability, spatial differential
expression, cell communication, or cell-type-specific causal origin. Mapping
one enrichment result across several specimens does not create independent
bulk tests.

Install the optional preranked-GSEA backend:

```bash
python -m pip install "easydecon[bulk-niche]"
```

No bulk raw counts are required. The complete DESeq2/PyDESeq2 result table and
spatial expression matrix are required.

## Existing niche outputs and bulk table

```python
import pandas as pd
import easydecon as ed

niches, smoothed = ed.detect_spatial_niches_from_posteriors(
    sdata,
    easydecon_result.posterior_df,
    sample_column="specimen",
    balance_samples=True,
    n_niches=5,
)

bulk_de = pd.read_csv("tumor_vs_normal_deseq2.csv")
bulk_niches = ed.analyze_bulk_niches(
    sdata,
    bulk_de,
    niches,
    contrast_name="tumor_vs_normal",
    positive_condition="tumor",
    negative_condition="normal",
    smoothed_evidence=smoothed,
    markers=easydecon_result.markers_df,
    sample_column="specimen",
    patient_column="patient",
    condition_column="condition",
    shared_niche_labels=True,
    ranking_mode="wald_stat",
    stat_column="stat",
    statistic_type="signed_wald",
)
```

`shared_niche_labels=True` is an explicit assertion that all labels came from
one shared multi-sample niche fit. Numeric labels from independently fitted
sections must not be aligned by name.

The default alternative, `ranking_mode="log2foldchange"`, ranks by signed
`log2FoldChange`; extreme noisy effects can dominate, and interpretation
depends on whether the upstream values were shrunken. `wald_stat` is accepted
only when `statistic_type="signed_wald"` is declared. Unsigned likelihood-ratio
statistics are rejected. Missing adjusted p-values from independent filtering
do not remove a gene with a finite ranking statistic. The full eligible bulk
ranking—not only significant DE genes—is the GSEA background.

## Niche-signature construction

Generated signatures use only positive/presence/identity cell-type markers;
negative/exclusion markers are removed. Marker membership weight is
`(1 / marker_rank)`, normalized within cell type. Cell-type evidence is first
averaged within each specimen/niche and then across specimens where that niche
occurs. For each niche and cell type, specificity is
`max(niche mean - mean of other niches, 0)`. Gene support is the sum of
`cell-type specificity × normalized marker weight`; genes are sorted by support
and gene identifier. This is bulk-independent and sample-balanced. These are
composition-derived marker signatures, not experimentally measured niche DE
genes. With one niche, composition weight is used without relative specificity.

If `marker_role` is absent, a signed marker fold-change column is required and
only finite, strictly positive effects are retained. Unsigned role-free lists
are rejected rather than assumed positive. Valid supplied `marker_rank` values
are preserved; otherwise markers are ordered deterministically by decreasing
fold change, decreasing score, increasing adjusted p-value, and gene identifier.

Explicit positive niche signatures can be supplied instead:

```python
bulk_niches = ed.analyze_bulk_niches(
    sdata,
    bulk_de,
    niches,
    contrast_name="tumor_vs_normal",
    niche_signatures={
        "0": ["EPCAM", "KRT8", "KRT18"],
        "1": ["COL1A1", "COL1A2", "DCN"],
    },
    sample_column="specimen",
    shared_niche_labels=True,
)
```

Similar and identical niche signatures are reported as overlapping rather than
treated as independent discoveries.

## Enrichment and local expression

GSEApy preranked enrichment returns ES, NES, nominal p-value, native
`gseapy_fdr_qvalue`, and leading-edge genes. EasyDecon separately reports
`bh_nominal_qvalue_across_niches`, calculated over all tested niche rows for
the contrast. Identical signatures are computed once by GSEApy and mapped back;
the BH calculation still includes every tested niche row. Zero permutation
p-values remain zero and are flagged as permutation-limited rather than being
replaced with invented precision.

Signatures equal to the entire eligible bulk background are marked
`full_background_untestable` and are not submitted to GSEApy.

Leading-edge genes are scored locally with the same full-background,
mean-capped-rank-credit definition described in the bulk-signature guide.
Positive- and negative-bulk programs remain separate. High spatial expression
of a negative-direction program means those genes are expressed locally; it
does not demonstrate local downregulation.

```python
fig1, axes1 = ed.plot_niche_bulk_associations(bulk_niches, sdata)
fig2, axes2 = ed.plot_niche_supporting_program(
    bulk_niches, sdata, program_niche="2", limits=(0, 1)
)
fig3, axes3 = ed.plot_bulk_niche_evidence(bulk_niches)
```

Association maps assign each location its niche's one fixed NES and label this
as niche-level bulk association; they do not imply spot-level significance.
Supporting-program maps show per-observation expression with fixed programs and
shared scales.

## Optional selected-interface analysis

The user chooses the niche pair, analyzed side, and expression program. The
analysis does not search for the strongest gradient.

```python
interface = ed.analyze_niche_interface(
    bulk_niches,
    sdata,
    niche_a="2",
    niche_b="4",
    side="2",
    program_niche="2",
    radius=24,
    near_distance=24,
    interior_min_distance=96,
    microns_per_unit=1.0,
)
fig, axes = ed.plot_niche_interface(interface)
```

A radius graph is constructed independently within every specimen. Boundary
observations are nodes on the analyzed side adjacent to the other niche;
distance zero therefore means a boundary observation, not an exact histologic
boundary. Multi-source shortest paths use physical edge lengths and remain
inside the analyzed niche. Unreachable components and absent interfaces are
reported. Distances are called micrometers only when `microns_per_unit` is
explicitly supplied.

`near_distance` must be strictly smaller than `interior_min_distance`, keeping
the descriptive groups disjoint. Missing interfaces, missing analyzed-side
observations, unavailable expression scores, and insufficient bands are
reported separately.

The returned effects are descriptive per-specimen near-interface minus
interior means. Spots are not treated as biological replicates and patient IDs
are preserved for future replicate-aware analyses.

## Four-specimen interpretation example

Consider one normal and three CRC sections, with normal-like regions also
present in CRC sections, one external `tumor_vs_normal` bulk table, and shared
niches fitted jointly. Bulk association remains fixed per shared niche
signature. Supporting-program expression can vary within that niche between
specimens. A normal-like CRC region is not necessarily equivalent to healthy
normal tissue. A normal specimen lacking the selected tumor interface receives
unavailable distance results rather than a fabricated zero effect.

This cohort supports exploratory localization, not a strong population-level
condition test. Bulk summary statistics cannot reconstruct patient-label
permutations or fully model gene–gene correlation; multiple sections from one
patient are not independent patients, and spots must not be pooled for ordinary
condition tests. Data-selected leading-edge maps are visualization of the same
association, not independent confirmation.

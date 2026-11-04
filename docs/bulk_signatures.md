# Bulk-to-spatial signature mapping

`map_bulk_signature()` maps one upstream bulk differential-expression
comparison onto an AnnData or SpatialData table. It does not run differential
expression or infer cell types. Positive fold changes describe the numerator
condition used by the upstream comparison.

- `up_score`: rank enrichment of genes increased in the bulk comparison.
- `down_score`: rank enrichment of genes decreased in the comparison.
- `balance_score`: `up_score - down_score`, with negative values preserved.

These scores are not local fold changes, cell fractions, cancer probabilities,
spatial differential-expression estimates, or proof of causal cellular origin.

## Basic mapping

```python
import pandas as pd
import easydecon as ed

bulk_de = pd.read_csv("tumor_vs_normal.csv")
result = ed.map_bulk_signature(
    sdata,
    bulk_de,
    signature_name="tumor_vs_normal",
    gene_column="gene",
    lfc_column="log2FoldChange",
    padj_column="padj",
)
fig, axes = ed.plot_bulk_signature(result, sdata)
```

Use `gene_from_index=True` when gene identifiers are deliberately stored in
the DataFrame index. The mapper never guesses that an integer index contains
genes, changes identifier case, strips identifier versions, or translates
symbols to Ensembl identifiers.

## Scoring definition

The scorer is UCell-like rather than exact UCell. Within each observation it
ranks positive expression against the full measured gene background, using
average ranks for ties. With the default `rank_cap=1500`, a detected gene
ranked first receives credit 1, rank 1500 receives 1/1500, and lower-ranked or
zero-expression genes receive 0. A direction's score is the mean credit across
its measured selected genes. A direction with fewer than `min_genes=5`
measured genes is unavailable (`NaN`); an available but undetected signature
scores zero. All-zero observations receive `NaN`.

Supply the same ordered `scoring_gene_universe` in separate datasets when an
explicit common measured background is required. Every supplied background
gene must exist in each dataset.

## Existing EasyDecon annotations

Cell-type summaries are descriptive associations with annotated or
cell-type-enriched regions, especially for mixed spots or bins. They are not
cell-type-resolved expression estimates and no spot-level significance test is
performed.

```python
decon = ed.run_easydecon(sdata, prepared_markers=prepared_markers)
mapped = ed.map_bulk_signature(
    sdata,
    bulk_de,
    signature_name="tumor_vs_normal",
    celltype_column="easydecon",
    sample_column="sample_id",
)
mapped.celltype_summary
fig, axes = ed.plot_bulk_signature(mapped, sdata, include_celltype_summary=True)
```

## Reuse across longitudinal samples

Select the signature once, before spatial intersection, then reuse it:

```python
prepared = ed.prepare_bulk_signature(
    bulk_de,
    signature_name="after_vs_before_treatment",
    top_n_per_direction=100,
)
common_genes = sample_1.var_names.intersection(sample_2.var_names).tolist()
r1 = ed.map_bulk_signature(
    sample_1, prepared, scoring_gene_universe=common_genes, rank_cap=1500
)
r2 = ed.map_bulk_signature(
    sample_2, prepared, scoring_gene_universe=common_genes, rank_cap=1500
)
limit = max(
    r1.scores[["up_score", "down_score"]].max().max(),
    r2.scores[["up_score", "down_score"]].max().max(),
)
ed.plot_bulk_signature(r1, sample_1, directional_limits=(0, limit))
ed.plot_bulk_signature(r2, sample_2, directional_limits=(0, limit))
```

Reuse the fixed signature, scoring parameters, gene background, and plotting
limits. Report signature coverage and sequencing-depth/platform differences;
common settings do not remove platform effects. Biological comparisons need
sample-level replication rather than treating spots as independent replicates.
Mapping time-derived signatures onto one tissue is not an observed spatial
trajectory.

## Several comparisons on one tissue

```python
for name, path in {
    "treatment_vs_baseline": "treatment_vs_baseline.csv",
    "relapse_vs_primary": "relapse_vs_primary.csv",
}.items():
    comparison = pd.read_csv(path)
    mapped = ed.map_bulk_signature(sdata, comparison, signature_name=name)
    ed.plot_bulk_signature(mapped, sdata, directional_limits=(0, 1))
```

This applies to tumor versus normal, before versus after treatment,
longitudinal progression, responders versus nonresponders, and relapse versus
primary-tumor comparisons.

## Exporting upstream DE results

DESeq2 in R:

```r
res <- as.data.frame(results(dds, contrast=c("condition", "tumor", "normal")))
res$gene <- rownames(res)
write.csv(res, "tumor_vs_normal.csv", row.names=FALSE)
```

PyDESeq2 in Python:

```python
from pydeseq2.ds import DeseqStats

stats = DeseqStats(dds, contrast=["condition", "tumor", "normal"])
stats.summary()
bulk_de = stats.results_df.reset_index(names="gene")
bulk_de.to_csv("tumor_vs_normal.csv", index=False)
```

Upstream DE analysis, replication, model design, and contrast orientation
remain the user's responsibility. EasyDecon only filters the exported table
and maps its fixed signatures.

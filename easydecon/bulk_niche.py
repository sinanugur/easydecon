"""Exploratory bulk association of fixed spatial niche signatures."""

from dataclasses import dataclass
from pathlib import Path
import heapq
import re

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

from ._schema import get_table, normalize_marker_roles, resolve_marker_columns
from .bulk import (
    _finite_number,
    _positive_integer,
    _resolve_matrix,
    _score_programs,
    _scoring_background,
)
from .config import config


@dataclass
class BulkNicheResult:
    """Bulk enrichment, spatial programs, and QC for shared niches."""

    niche_signatures: pd.DataFrame
    enrichment: pd.DataFrame
    supporting_genes: pd.DataFrame
    spatial_scores: pd.DataFrame
    coverage_qc: pd.DataFrame
    sample_niche_summary: pd.DataFrame
    niche_composition: pd.DataFrame
    niche_labels: pd.Series
    parameters: dict
    diagnostics: dict
    interface_outputs: object | None = None


@dataclass
class NicheInterfaceResult:
    """Descriptive expression-distance output for one selected interface."""

    observations: pd.DataFrame
    binned_summary: pd.DataFrame
    specimen_effects: pd.DataFrame
    parameters: dict
    diagnostics: dict


def _read_table(value, name):
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{name} must be a pandas DataFrame or CSV/TSV path.")
    path = Path(value)
    if path.suffix.casefold() == ".csv":
        return pd.read_csv(path)
    if path.suffix.casefold() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"{name} files must use a .csv, .tsv, or .txt extension.")


def _resolve_niche_labels(table, niches, niche_column):
    if not table.obs.index.is_unique:
        raise ValueError("table.obs identifiers must be unique.")
    if isinstance(niches, str):
        if niches not in table.obs:
            raise ValueError(f"Niche column {niches!r} was not found in table.obs.")
        labels = table.obs[niches].copy()
        labels.name = niche_column
    elif isinstance(niches, pd.Series):
        labels = niches.copy()
        labels.name = labels.name or niche_column
    elif isinstance(niches, pd.DataFrame):
        if niche_column not in niches:
            raise ValueError(f"{niche_column!r} was not found in niches columns.")
        labels = niches[niche_column].copy()
    else:
        raise TypeError("niches must be a column name, pandas Series, or DataFrame.")
    if not labels.index.is_unique:
        raise ValueError("Niche-label observation identifiers must be unique.")
    missing_obs = labels.index.difference(table.obs.index)
    if len(missing_obs):
        raise ValueError(
            "Niche labels contain observations absent from the spatial table: "
            f"{missing_obs[:5].tolist()}."
        )
    if labels.isna().any():
        raise ValueError("Niche labels must not contain missing values.")
    return labels.astype(str)


def _metadata(table, index, sample_column, patient_column, condition_column):
    requested = [
        column
        for column in (sample_column, patient_column, condition_column)
        if column is not None
    ]
    missing = [column for column in requested if column not in table.obs]
    if missing:
        raise ValueError(
            f"Spatial metadata column(s) not found: {', '.join(map(repr, missing))}."
        )
    metadata = table.obs.loc[index, requested].copy()
    if sample_column is None:
        metadata["_sample"] = "sample"
        sample_key = "_sample"
    else:
        sample_key = sample_column
    if metadata.isna().any().any():
        bad = metadata.columns[metadata.isna().any()].tolist()
        raise ValueError(f"Spatial metadata contains missing values in: {bad}.")
    if sample_column is not None:
        for column in (patient_column, condition_column):
            if column is not None and metadata.groupby(sample_column)[column].nunique().gt(1).any():
                raise ValueError(
                    f"Each {sample_column!r} must map to exactly one {column!r}."
                )
    return metadata, sample_key


def _marker_membership(markers, evidence_columns):
    if hasattr(markers, "raw_markers_df"):
        markers = markers.raw_markers_df
    if not isinstance(markers, pd.DataFrame):
        raise TypeError("markers must be a DataFrame or PreparedMarkers-like object.")
    frame = markers.copy()
    resolved = resolve_marker_columns(frame)
    if "group" not in resolved or "names" not in resolved:
        raise ValueError("markers must contain resolvable group and gene columns.")
    if frame[[resolved["group"], resolved["names"]]].isna().any().any():
        raise ValueError("Marker group and gene identifiers must not be missing.")
    groups = frame[resolved["group"]].astype(str)
    genes = frame[resolved["names"]].astype(str)
    roles, _ = normalize_marker_roles(frame)
    if roles is None:
        if "logfoldchanges" not in resolved:
            raise ValueError(
                "Role-free marker tables require a signed fold-change column or "
                "an explicit marker_role column declaring positive markers."
            )
        signed_effect = pd.to_numeric(
            frame[resolved["logfoldchanges"]], errors="coerce"
        )
        keep = np.isfinite(signed_effect) & signed_effect.gt(0)
    else:
        keep = roles.isin({"positive", "presence", "identity"})
    keep &= groups.isin(set(map(str, evidence_columns)))
    ranks = pd.to_numeric(frame.get("marker_rank"), errors="coerce") if "marker_rank" in frame else pd.Series(np.nan, index=frame.index)
    supplied_rank = frame["marker_rank"].notna() if "marker_rank" in frame else pd.Series(False, index=frame.index)
    if (supplied_rank & (~np.isfinite(ranks) | ranks.le(0))).any():
        raise ValueError("Supplied marker_rank values must be finite and positive.")
    lfc = (
        pd.to_numeric(frame[resolved["logfoldchanges"]], errors="coerce")
        if "logfoldchanges" in resolved else pd.Series(np.nan, index=frame.index)
    )
    score = (
        pd.to_numeric(frame[resolved["scores"]], errors="coerce")
        if "scores" in resolved else pd.Series(np.nan, index=frame.index)
    )
    padj = (
        pd.to_numeric(frame[resolved["pvals_adj"]], errors="coerce")
        if "pvals_adj" in resolved else pd.Series(np.nan, index=frame.index)
    )
    membership = pd.DataFrame(
        {
            "celltype": groups[keep],
            "gene": genes[keep],
            "marker_rank": ranks[keep],
            "_lfc": lfc[keep],
            "_score": score[keep],
            "_padj": padj[keep],
        }
    ).sort_values(
        ["celltype", "marker_rank", "_lfc", "_score", "_padj", "gene"],
        ascending=[True, True, False, False, True, True],
        na_position="last",
        kind="stable",
    ).drop_duplicates(["celltype", "gene"], keep="first")
    for _, positions in membership.groupby("celltype", sort=False).groups.items():
        positions = list(positions)
        missing_rank = membership.loc[positions, "marker_rank"].isna()
        start = membership.loc[positions, "marker_rank"].max()
        start = int(np.ceil(start)) if np.isfinite(start) else 0
        membership.loc[np.asarray(positions)[missing_rank.to_numpy()], "marker_rank"] = np.arange(
            start + 1, start + 1 + int(missing_rank.sum())
        )
    membership["marker_weight"] = 1.0 / membership["marker_rank"]
    membership["marker_weight"] /= membership.groupby("celltype")[
        "marker_weight"
    ].transform("sum")
    missing_celltypes = sorted(set(map(str, evidence_columns)) - set(membership["celltype"]))
    return membership.drop(columns=["_lfc", "_score", "_padj"]).reset_index(drop=True), missing_celltypes


def _generated_signatures(
    labels,
    evidence,
    sample_labels,
    markers,
    top_n_genes,
):
    if not isinstance(evidence, pd.DataFrame):
        raise TypeError("smoothed_evidence must be a pandas DataFrame.")
    if not evidence.index.is_unique:
        raise ValueError("smoothed_evidence observation identifiers must be unique.")
    if set(evidence.index) != set(labels.index):
        raise ValueError("niches and smoothed_evidence must contain exactly the same observations.")
    evidence = evidence.reindex(labels.index).apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(evidence.to_numpy()).all() or (evidence.to_numpy() < 0).any():
        raise ValueError("smoothed_evidence must contain finite, nonnegative values.")
    membership, missing_celltypes = _marker_membership(markers, evidence.columns)
    annotated = evidence.copy()
    annotated["_sample"] = sample_labels.to_numpy()
    annotated["_niche"] = labels.to_numpy()
    sample_means = annotated.groupby(["_sample", "_niche"], sort=False).mean()
    composition = sample_means.groupby(level="_niche", sort=False).mean()
    composition.index.name = "niche"
    niches = labels.drop_duplicates().tolist()
    rows = []
    statuses = {}
    for niche in niches:
        current = composition.loc[niche]
        if len(composition) == 1:
            specificity = current.clip(lower=0)
            mode = "composition_only_single_niche"
        else:
            others = composition.drop(index=niche).mean(axis=0)
            specificity = (current - others).clip(lower=0)
            mode = "relative_specificity"
        contributions = membership.copy()
        contributions["celltype_specificity"] = contributions["celltype"].map(
            specificity.to_dict()
        ).fillna(0.0)
        contributions["contribution"] = (
            contributions["marker_weight"] * contributions["celltype_specificity"]
        )
        contributions = contributions.loc[contributions["contribution"] > 0]
        if contributions.empty:
            statuses[niche] = "empty_no_positive_specificity"
            continue
        gene_rows = []
        for gene, gene_frame in contributions.groupby("gene", sort=False):
            ordered = gene_frame.sort_values(
                ["contribution", "celltype"], ascending=[False, True], kind="stable"
            )
            gene_rows.append(
                {
                    "niche": niche,
                    "gene": gene,
                    "raw_weight": float(gene_frame["contribution"].sum()),
                    "contributing_celltypes": ";".join(ordered["celltype"]),
                    "construction": mode,
                }
            )
        selected = pd.DataFrame(gene_rows).sort_values(
            ["raw_weight", "gene"], ascending=[False, True], kind="stable"
        ).head(top_n_genes)
        selected["weight"] = selected["raw_weight"] / selected["raw_weight"].sum()
        selected["signature_rank"] = np.arange(1, len(selected) + 1)
        rows.extend(selected.to_dict("records"))
        statuses[niche] = "selected"
    columns = [
        "niche", "gene", "raw_weight", "weight", "signature_rank",
        "contributing_celltypes", "construction",
    ]
    signatures = pd.DataFrame(rows, columns=columns)
    diagnostics = {
        "construction": "composition-derived positive marker specificity",
        "signature_status": statuses,
        "missing_marker_celltypes": missing_celltypes,
        "sample_balanced": True,
    }
    return signatures, composition, diagnostics


def _explicit_signatures(value, observed_niches, top_n_genes):
    if isinstance(value, dict):
        rows = []
        for niche, genes in value.items():
            if isinstance(genes, dict):
                entries = genes.items()
            else:
                entries = [(gene, 1.0) for gene in genes]
            seen = set()
            for gene, weight in entries:
                gene = str(gene)
                if gene in seen:
                    raise ValueError(f"Duplicate gene {gene!r} in niche {niche!r} signature.")
                seen.add(gene)
                rows.append({"niche": str(niche), "gene": gene, "weight": weight})
        frame = pd.DataFrame(rows)
    elif isinstance(value, pd.DataFrame):
        frame = value.copy()
        missing = [column for column in ("niche", "gene") if column not in frame]
        if missing:
            raise ValueError("Explicit niche signature DataFrames require niche and gene columns.")
        if "weight" not in frame:
            frame["weight"] = 1.0
    else:
        raise TypeError("niche_signatures must be a mapping or DataFrame.")
    if frame.empty:
        return pd.DataFrame(columns=["niche", "gene", "weight", "signature_rank", "contributing_celltypes", "construction"])
    if frame[["niche", "gene"]].isna().any().any():
        raise ValueError("Explicit niche and gene identifiers must not be missing.")
    frame["niche"] = frame["niche"].astype(str)
    frame["gene"] = frame["gene"].astype(str)
    if frame[["niche", "gene"]].duplicated().any():
        raise ValueError("Explicit niche signatures must not duplicate niche/gene pairs.")
    unknown = sorted(set(frame["niche"]) - set(observed_niches))
    if unknown:
        raise ValueError(f"Explicit signatures contain unobserved niches: {unknown}.")
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
    if not np.isfinite(frame["weight"]).all() or (frame["weight"] < 0).any():
        raise ValueError("Explicit signature weights must be finite and nonnegative.")
    if frame["gene"].str.strip().eq("").any():
        raise ValueError("Explicit signature genes must not be blank.")
    frame = frame.sort_values(
        ["niche", "weight", "gene"], ascending=[True, False, True], kind="stable"
    ).groupby("niche", sort=False, group_keys=False).head(top_n_genes)
    sums = frame.groupby("niche")["weight"].transform("sum")
    if (sums <= 0).any():
        raise ValueError("Each explicit niche signature must have positive total weight.")
    frame["weight"] /= sums
    frame["signature_rank"] = frame.groupby("niche", sort=False).cumcount() + 1
    frame["contributing_celltypes"] = None
    frame["construction"] = "explicit"
    return frame.reset_index(drop=True)


def _prepare_bulk_ranking(
    bulk_de,
    *,
    gene_column,
    gene_from_index,
    lfc_column,
    ranking_mode,
    stat_column,
    statistic_type,
    pvalue_column,
    padj_column,
    base_mean_column,
    min_base_mean,
    gene_universe,
):
    frame = _read_table(bulk_de, "bulk_de")
    if gene_from_index:
        genes = pd.Series(frame.index, index=frame.index)
    elif gene_column in frame:
        genes = frame[gene_column]
    else:
        raise ValueError(
            f"Gene column {gene_column!r} was not found. Use gene_from_index=True "
            "when identifiers are stored in the index."
        )
    if genes.isna().any() or genes.astype(str).str.strip().eq("").any():
        raise ValueError("Bulk gene identifiers must not be missing or blank.")
    genes = genes.astype(str)
    duplicates = genes[genes.duplicated(keep=False)].drop_duplicates().tolist()
    if duplicates:
        raise ValueError(
            "Bulk gene identifiers must be unique; resolve duplicates explicitly. "
            f"Duplicate examples: {duplicates[:5]}."
        )
    if lfc_column not in frame:
        raise ValueError(f"Fold-change column {lfc_column!r} was not found.")
    lfc = pd.to_numeric(frame[lfc_column], errors="coerce")
    if ranking_mode == "log2foldchange":
        ranking = lfc
        ranking_source = lfc_column
    elif ranking_mode == "wald_stat":
        if statistic_type != "signed_wald":
            raise ValueError(
                "ranking_mode='wald_stat' requires statistic_type='signed_wald'. "
                "Unsigned LRT statistics are unsupported."
            )
        if stat_column is None or stat_column not in frame:
            raise ValueError("A valid stat_column is required for signed Wald ranking.")
        ranking = pd.to_numeric(frame[stat_column], errors="coerce")
        ranking_source = stat_column
    else:
        raise ValueError("ranking_mode must be 'log2foldchange' or 'wald_stat'.")

    valid = np.isfinite(ranking)
    exclusion_counts = {"missing_or_nonfinite_ranking": int((~valid).sum())}
    numeric_optional = {}
    for column, label in ((pvalue_column, "pvalue"), (padj_column, "padj")):
        if column is not None and column in frame:
            values = pd.to_numeric(frame[column], errors="coerce")
            finite = np.isfinite(values)
            if (finite & ((values < 0) | (values > 1))).any():
                raise ValueError(f"Finite {label} values must lie within [0, 1].")
            numeric_optional[label] = values
    base_mean = None
    if min_base_mean is not None:
        min_base_mean = _finite_number(min_base_mean, "min_base_mean", minimum=0)
        if base_mean_column is None or base_mean_column not in frame:
            raise ValueError("min_base_mean requires a valid base_mean_column.")
        base_mean = pd.to_numeric(frame[base_mean_column], errors="coerce")
        passes = np.isfinite(base_mean) & (base_mean >= min_base_mean)
        exclusion_counts["below_or_invalid_base_mean"] = int((valid & ~passes).sum())
        valid &= passes
    if gene_universe is not None:
        universe = set(map(str, gene_universe))
        in_universe = genes.isin(universe)
        exclusion_counts["outside_bulk_gene_universe"] = int((valid & ~in_universe).sum())
        valid &= in_universe

    ranked = frame.loc[valid].copy().reset_index(drop=True)
    ranked["gene"] = genes.loc[valid].to_numpy()
    ranked["log2FoldChange"] = lfc.loc[valid].to_numpy(dtype=float)
    ranked["ranking_score"] = ranking.loc[valid].to_numpy(dtype=float)
    if "pvalue" in numeric_optional:
        ranked["pvalue"] = numeric_optional["pvalue"].loc[valid].to_numpy(dtype=float)
    if "padj" in numeric_optional:
        ranked["padj"] = numeric_optional["padj"].loc[valid].to_numpy(dtype=float)
    if base_mean is not None:
        ranked["baseMean"] = base_mean.loc[valid].to_numpy(dtype=float)
    ranked = ranked.sort_values(
        ["ranking_score", "gene"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    if ranked.empty:
        raise ValueError("No genes remain in the eligible bulk ranking.")
    ranked["bulk_rank"] = np.arange(1, len(ranked) + 1)
    diagnostics = {
        "n_input_rows": int(len(frame)),
        "n_eligible_genes": int(len(ranked)),
        "ranking_source": ranking_source,
        "n_ranking_ties": int(ranked["ranking_score"].duplicated(keep=False).sum()),
        "excluded": exclusion_counts,
        "padj_missing_does_not_exclude_wald_ranking": ranking_mode == "wald_stat",
    }
    return ranked, diagnostics


def _run_gseapy_prerank(ranking, gene_sets, *, min_size, max_size, permutations, seed):
    try:
        import gseapy as gp
    except ImportError as exc:
        raise ImportError(
            "Bulk-to-niche enrichment requires the optional 'gseapy' dependency. "
            "Install easydecon[bulk-niche] or gseapy."
        ) from exc
    result = gp.prerank(
        rnk=ranking[["gene", "ranking_score"]],
        gene_sets=gene_sets,
        outdir=None,
        min_size=min_size,
        max_size=max_size,
        permutation_num=permutations,
        seed=seed,
        threads=1,
        no_plot=True,
        verbose=False,
        ascending=None,
    )
    return result.res2d.copy()


def _bh_adjust(values):
    values = np.asarray(values, dtype=float)
    output = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return output
    order = valid[np.argsort(values[valid], kind="stable")]
    adjusted = values[order] * len(order) / np.arange(1, len(order) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output[order] = np.clip(adjusted, 0, 1)
    return output


def _signature_overlap(signatures):
    sets = {
        niche: set(group["gene"])
        for niche, group in signatures.groupby("niche", sort=False)
    }
    rows = []
    niches = list(sets)
    for i, first in enumerate(niches):
        for second in niches[i + 1:]:
            union = sets[first] | sets[second]
            rows.append(
                {
                    "niche_a": first,
                    "niche_b": second,
                    "n_shared": len(sets[first] & sets[second]),
                    "jaccard": len(sets[first] & sets[second]) / len(union) if union else np.nan,
                    "identical": sets[first] == sets[second],
                }
            )
    return rows


def _enrich_signatures(
    signatures,
    observed_niches,
    ranking,
    *,
    min_size,
    max_size,
    permutations,
    seed,
    positive_condition,
    negative_condition,
):
    ranking_genes = set(ranking["gene"])
    signature_sets = {
        niche: signatures.loc[signatures["niche"] == niche, "gene"].tolist()
        for niche in observed_niches
    }
    canonical_to_key = {}
    key_to_genes = {}
    niche_to_key = {}
    rows = []
    for niche in observed_niches:
        overlap = tuple(sorted(set(signature_sets[niche]) & ranking_genes))
        row = {"niche": niche, "n_signature_genes": len(signature_sets[niche]), "n_bulk_overlap": len(overlap)}
        if overlap and len(overlap) == len(ranking):
            row["test_status"] = "full_background_untestable"
        elif len(overlap) < min_size:
            row["test_status"] = "insufficient_overlap" if overlap else "no_overlap"
        elif len(overlap) > max_size:
            row["test_status"] = "above_max_size"
        else:
            key = canonical_to_key.setdefault(overlap, f"signature_{len(canonical_to_key)}")
            key_to_genes[key] = list(overlap)
            niche_to_key[niche] = key
            row["test_status"] = "pending"
        rows.append(row)
    enrichment = pd.DataFrame(rows)
    for column in (
        "es", "nes", "nominal_pvalue", "gseapy_fdr_qvalue",
        "bh_nominal_qvalue_across_niches",
    ):
        enrichment[column] = np.nan
    enrichment["leading_edge_genes"] = ""
    enrichment["bulk_direction"] = None
    enrichment["direction_label"] = None
    enrichment["backend_gene_set"] = enrichment["niche"].map(niche_to_key)
    if key_to_genes:
        backend = _run_gseapy_prerank(
            ranking, key_to_genes,
            min_size=min_size, max_size=max_size,
            permutations=permutations, seed=seed,
        )
        if "Term" not in backend:
            raise ValueError("GSEApy prerank output is missing the 'Term' column.")
        backend = backend.set_index(backend["Term"].astype(str), drop=False)
        required = ["ES", "NES", "NOM p-val", "FDR q-val", "Lead_genes"]
        missing = [column for column in required if column not in backend]
        if missing:
            raise ValueError(f"GSEApy prerank output is missing columns: {missing}.")
        for index, row in enrichment.loc[enrichment["test_status"] == "pending"].iterrows():
            key = row["backend_gene_set"]
            if key not in backend.index:
                enrichment.loc[index, "test_status"] = "backend_omitted"
                continue
            result = backend.loc[key]
            if isinstance(result, pd.DataFrame):
                result = result.iloc[0]
            enrichment.loc[index, "es"] = float(result["ES"])
            enrichment.loc[index, "nes"] = float(result["NES"])
            enrichment.loc[index, "nominal_pvalue"] = float(result["NOM p-val"])
            enrichment.loc[index, "gseapy_fdr_qvalue"] = float(result["FDR q-val"])
            enrichment.loc[index, "leading_edge_genes"] = str(result["Lead_genes"])
            enrichment.loc[index, "test_status"] = "tested"
    tested = enrichment["test_status"] == "tested"
    enrichment.loc[tested, "bh_nominal_qvalue_across_niches"] = _bh_adjust(
        enrichment.loc[tested, "nominal_pvalue"]
    )
    enrichment["nominal_pvalue_is_permutation_zero"] = tested & enrichment["nominal_pvalue"].eq(0)
    positive = tested & enrichment["nes"].gt(0)
    negative = tested & enrichment["nes"].lt(0)
    enrichment.loc[positive, "bulk_direction"] = "positive"
    enrichment.loc[negative, "bulk_direction"] = "negative"
    enrichment.loc[positive, "direction_label"] = f"toward_{positive_condition or 'bulk_numerator'}"
    enrichment.loc[negative, "direction_label"] = f"toward_{negative_condition or 'bulk_denominator'}"
    return enrichment


def _supporting_gene_table(enrichment, signatures, ranking, spatial_genes):
    ranking_lookup = ranking.set_index("gene")
    spatial_genes = set(spatial_genes)
    rows = []
    for item in enrichment.itertuples(index=False):
        genes = [gene for gene in str(item.leading_edge_genes).split(";") if gene]
        for gene in genes:
            row = {
                "niche": item.niche,
                "gene": gene,
                "bulk_direction": item.bulk_direction,
                "spatially_measured": gene in spatial_genes,
            }
            if gene in ranking_lookup.index:
                bulk = ranking_lookup.loc[gene]
                row.update(
                    bulk_rank=int(bulk["bulk_rank"]),
                    ranking_score=float(bulk["ranking_score"]),
                    log2FoldChange=float(bulk["log2FoldChange"]),
                    padj=float(bulk["padj"]) if "padj" in bulk and np.isfinite(bulk["padj"]) else np.nan,
                )
            rows.append(row)
    return pd.DataFrame(
        rows,
        columns=[
            "niche", "gene", "bulk_direction", "spatially_measured",
            "bulk_rank", "ranking_score", "log2FoldChange", "padj",
        ],
    )


def _spatial_program_scores(
    table,
    enrichment,
    labels,
    metadata,
    sample_key,
    *,
    contrast_name,
    layer,
    scoring_gene_universe,
    min_genes,
    rank_cap,
):
    background, positions, explicit_background = _scoring_background(
        table, scoring_gene_universe
    )
    matrix = _resolve_matrix(table, layer)
    if matrix.shape != (len(table.obs), len(table.var_names)):
        raise ValueError("The selected expression matrix shape does not match the spatial table.")
    if explicit_background:
        matrix = matrix[:, positions]
    background_index = {gene: index for index, gene in enumerate(background)}
    programs = []
    for row in enrichment.itertuples(index=False):
        leading = [gene for gene in str(row.leading_edge_genes).split(";") if gene]
        programs.append([gene for gene in leading if gene in background_index])
    marker_genes = list(dict.fromkeys(gene for genes in programs for gene in genes))
    marker_lookup = {gene: index for index, gene in enumerate(marker_genes)}
    marker_codes = np.array([background_index[gene] for gene in marker_genes], dtype=int)
    program_codes = tuple(
        np.array([marker_lookup[gene] for gene in genes], dtype=int) for genes in programs
    )
    batch_size = 1000 if config.batch_size == "auto" else _positive_integer(
        config.batch_size, "configured batch_size"
    )
    raw, detected, all_zero, effective_cap = _score_programs(
        matrix, marker_codes, program_codes, rank_cap, batch_size
    )
    table_metadata = table.obs.copy()
    if sample_key == "_sample":
        table_metadata["_sample"] = "sample"
    long = []
    for program_index, row in enumerate(enrichment.itertuples(index=False)):
        n_measured = len(programs[program_index])
        available = row.test_status == "tested" and n_measured >= min_genes
        frame = pd.DataFrame(
            {
                "observation_id": table.obs.index.astype(str),
                "contrast": contrast_name,
                "program_niche": row.niche,
                "bulk_direction": row.bulk_direction,
                "score": raw[:, program_index] if available else np.nan,
                "detected_genes": detected[:, program_index],
                "detected_fraction": detected[:, program_index] / n_measured if n_measured else np.nan,
                "n_measured_program_genes": n_measured,
                "program_status": "available" if available else "insufficient_spatial_coverage",
                "all_zero": all_zero,
                "observed_niche": labels.reindex(table.obs.index).to_numpy(),
                "specimen": table_metadata[sample_key].astype(str).to_numpy(),
            },
            index=table.obs.index,
        )
        for role, column in (("patient", metadata.attrs.get("patient_column")), ("condition", metadata.attrs.get("condition_column"))):
            if column is not None:
                frame[role] = table.obs[column].astype(str).to_numpy()
        long.append(frame.reset_index(drop=True))
    scores = pd.concat(long, ignore_index=True) if long else pd.DataFrame()
    group_columns = ["specimen", "program_niche", "bulk_direction", "program_status"]
    coverage_rows = []
    for keys, group in scores.groupby(group_columns, dropna=False, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        coverage_rows.append(
            {
                **dict(zip(group_columns, keys)),
                "n_observations": int(len(group)),
                "n_valid_scores": int(group["score"].notna().sum()),
                "n_all_zero": int(group["all_zero"].sum()),
                "mean_detected_fraction": float(group["detected_fraction"].mean()),
                "n_measured_program_genes": int(group["n_measured_program_genes"].iloc[0]),
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    summary_rows = []
    summary_keys = ["specimen", "observed_niche", "program_niche", "bulk_direction"]
    valid_labels = scores["observed_niche"].notna()
    for keys, group in scores.loc[valid_labels].groupby(summary_keys, dropna=False, sort=False):
        valid_scores = group["score"].dropna()
        summary_rows.append(
            {
                **dict(zip(summary_keys, keys)),
                "n_observations": int(len(group)),
                "n_valid_scores": int(len(valid_scores)),
                "mean_score": float(valid_scores.mean()) if len(valid_scores) else np.nan,
                "median_score": float(valid_scores.median()) if len(valid_scores) else np.nan,
                "mean_detected_fraction": float(group["detected_fraction"].mean()),
            }
        )
    return scores, coverage, pd.DataFrame(summary_rows), effective_cap, background.tolist()


def analyze_bulk_niches(
    sdata,
    bulk_de,
    niches,
    *,
    contrast_name,
    smoothed_evidence=None,
    markers=None,
    niche_signatures=None,
    niche_column="niche",
    sample_column=None,
    patient_column=None,
    condition_column=None,
    shared_niche_labels=None,
    positive_condition=None,
    negative_condition=None,
    gene_column="gene",
    gene_from_index=False,
    lfc_column="log2FoldChange",
    ranking_mode="log2foldchange",
    stat_column=None,
    statistic_type=None,
    pvalue_column="pvalue",
    padj_column="padj",
    base_mean_column="baseMean",
    min_base_mean=None,
    bulk_gene_universe=None,
    top_n_genes=50,
    min_signature_size=5,
    max_signature_size=500,
    permutations=1000,
    seed=0,
    layer=None,
    scoring_gene_universe=None,
    scoring_min_genes=3,
    rank_cap=1500,
    table_key=None,
    bin_size=8,
    store=False,
    overwrite=False,
):
    """Associate fixed shared-niche signatures with one bulk contrast."""
    if not isinstance(contrast_name, str) or not contrast_name.strip():
        raise ValueError("contrast_name must be a non-empty string.")
    top_n_genes = _positive_integer(top_n_genes, "top_n_genes")
    min_signature_size = _positive_integer(min_signature_size, "min_signature_size")
    max_signature_size = _positive_integer(max_signature_size, "max_signature_size")
    if max_signature_size < min_signature_size:
        raise ValueError("max_signature_size must be at least min_signature_size.")
    permutations = _positive_integer(permutations, "permutations")
    scoring_min_genes = _positive_integer(scoring_min_genes, "scoring_min_genes")
    rank_cap = _positive_integer(rank_cap, "rank_cap")
    table = get_table(sdata, bin_size=bin_size, table_key=table_key)
    labels = _resolve_niche_labels(table, niches, niche_column)
    metadata, sample_key = _metadata(
        table, labels.index, sample_column, patient_column, condition_column
    )
    metadata.attrs["patient_column"] = patient_column
    metadata.attrs["condition_column"] = condition_column
    sample_labels = metadata[sample_key]
    n_samples = int(sample_labels.nunique())
    if n_samples > 1 and shared_niche_labels is not True:
        raise ValueError(
            "Multi-sample analysis requires shared_niche_labels=True to confirm "
            "that niche labels came from one shared classification, not separate fits."
        )
    observed_niches = labels.drop_duplicates().tolist()
    if niche_signatures is None:
        if smoothed_evidence is None or markers is None:
            raise ValueError(
                "Generated niche signatures require smoothed_evidence and markers; "
                "otherwise provide niche_signatures explicitly."
            )
        signatures, composition, signature_diagnostics = _generated_signatures(
            labels, smoothed_evidence, sample_labels, markers, top_n_genes
        )
    else:
        signatures = _explicit_signatures(
            niche_signatures, observed_niches, top_n_genes
        )
        composition = pd.DataFrame()
        if smoothed_evidence is not None:
            if not isinstance(smoothed_evidence, pd.DataFrame):
                raise TypeError("smoothed_evidence must be a pandas DataFrame.")
            if not smoothed_evidence.index.is_unique or set(smoothed_evidence.index) != set(labels.index):
                raise ValueError("niches and smoothed_evidence must contain exactly the same observations.")
            evidence = smoothed_evidence.reindex(labels.index).apply(pd.to_numeric, errors="coerce")
            if not np.isfinite(evidence.to_numpy()).all() or (evidence.to_numpy() < 0).any():
                raise ValueError("smoothed_evidence must contain finite, nonnegative values.")
            annotated = evidence.copy()
            annotated["_sample"] = sample_labels.to_numpy()
            annotated["_niche"] = labels.to_numpy()
            composition = annotated.groupby(["_sample", "_niche"]).mean().groupby(level="_niche").mean()
            composition.index.name = "niche"
        signature_diagnostics = {"construction": "explicit", "sample_balanced": True}
    ranking, ranking_diagnostics = _prepare_bulk_ranking(
        bulk_de,
        gene_column=gene_column,
        gene_from_index=gene_from_index,
        lfc_column=lfc_column,
        ranking_mode=ranking_mode,
        stat_column=stat_column,
        statistic_type=statistic_type,
        pvalue_column=pvalue_column,
        padj_column=padj_column,
        base_mean_column=base_mean_column,
        min_base_mean=min_base_mean,
        gene_universe=bulk_gene_universe,
    )
    enrichment = _enrich_signatures(
        signatures,
        observed_niches,
        ranking,
        min_size=min_signature_size,
        max_size=max_signature_size,
        permutations=permutations,
        seed=seed,
        positive_condition=positive_condition,
        negative_condition=negative_condition,
    )
    supporting = _supporting_gene_table(
        enrichment, signatures, ranking, map(str, table.var_names)
    )
    scores, coverage, sample_summary, effective_cap, background = _spatial_program_scores(
        table,
        enrichment,
        labels,
        metadata,
        sample_key,
        contrast_name=contrast_name,
        layer=layer,
        scoring_gene_universe=scoring_gene_universe,
        min_genes=scoring_min_genes,
        rank_cap=rank_cap,
    )
    overlap = _signature_overlap(signatures)
    diagnostics = {
        "signature_construction": signature_diagnostics,
        "bulk_ranking": ranking_diagnostics,
        "signature_overlap": overlap,
        "n_identical_signature_pairs": int(sum(row["identical"] for row in overlap)),
        "warnings": [
            "Similar or identical niche signatures are not independent biological discoveries."
        ] if any(row["jaccard"] > 0 for row in overlap) else [],
    }
    parameters = {
        "contrast_name": contrast_name,
        "positive_condition": positive_condition,
        "negative_condition": negative_condition,
        "ranking_mode": ranking_mode,
        "ranking_column": ranking_diagnostics["ranking_source"],
        "min_signature_size": min_signature_size,
        "max_signature_size": max_signature_size,
        "permutations": permutations,
        "seed": seed,
        "gsea_backend": "gseapy.prerank permutation",
        "multiple_testing": {
            "gseapy_fdr_qvalue": "backend-native FDR across unique submitted gene sets",
            "bh_nominal_qvalue_across_niches": "BH over tested niche rows, including duplicate signatures",
        },
        "layer": layer,
        "rank_cap": rank_cap,
        "effective_rank_cap": effective_cap,
        "scoring_definition": "full-background mean capped rank credit (UCell-like)",
        "scoring_gene_background": background,
        "sample_column": sample_column,
        "patient_column": patient_column,
        "condition_column": condition_column,
        "shared_niche_labels_confirmed": shared_niche_labels is True or n_samples == 1,
    }
    result = BulkNicheResult(
        niche_signatures=signatures,
        enrichment=enrichment,
        supporting_genes=supporting,
        spatial_scores=scores,
        coverage_qc=coverage,
        sample_niche_summary=sample_summary,
        niche_composition=composition,
        niche_labels=labels,
        parameters=parameters,
        diagnostics=diagnostics,
    )
    if store:
        _store_result(table, result, overwrite=overwrite)
    return result


def _slug(value):
    return re.sub(r"[^0-9A-Za-z]+", "_", str(value)).strip("_").casefold() or "value"


def _h5ad_safe_dataframe(frame):
    frame = frame.reset_index(drop=True).copy()
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        frame[column] = pd.Categorical(frame[column])
    return frame


def _h5ad_safe_value(value):
    if isinstance(value, pd.DataFrame):
        return _h5ad_safe_dataframe(value)
    if isinstance(value, dict):
        return {
            str(key): _h5ad_safe_value(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        if value and all(isinstance(item, dict) for item in value):
            return _h5ad_safe_dataframe(pd.DataFrame(value))
        return np.asarray(value, dtype=object if any(isinstance(item, str) for item in value) else None)
    if isinstance(value, tuple):
        return _h5ad_safe_value(list(value))
    if isinstance(value, np.generic):
        return value.item()
    return value


def _store_result(table, result, *, overwrite):
    prefix = f"easydecon_bulk_niche_{_slug(result.parameters['contrast_name'])}"
    score_wide = result.spatial_scores.pivot(
        index="observation_id", columns="program_niche", values="score"
    ).reindex(table.obs.index.astype(str))
    score_wide.columns = [f"{prefix}_{_slug(column)}_program_score" for column in score_wide]
    nes = result.enrichment.set_index("niche")["nes"]
    inherited = result.niche_labels.map(nes).reindex(table.obs.index)
    outputs = list(score_wide.columns) + [f"{prefix}_niche_nes"]
    uns_key = prefix
    collisions = [column for column in outputs if column in table.obs]
    if uns_key in table.uns:
        collisions.append(f"uns[{uns_key!r}]")
    if collisions and not overwrite:
        raise ValueError(
            f"Storage would overwrite {collisions}. Use overwrite=True to replace it."
        )
    new_obs = table.obs.copy()
    for column in score_wide:
        new_obs[column] = score_wide[column].to_numpy()
    new_obs[f"{prefix}_niche_nes"] = inherited.to_numpy()
    new_uns = dict(table.uns)
    new_uns[uns_key] = {
        "parameters": _h5ad_safe_value(result.parameters),
        "diagnostics": _h5ad_safe_value(result.diagnostics),
        "enrichment": _h5ad_safe_dataframe(result.enrichment),
    }
    old_obs, old_uns = table.obs, table.uns
    try:
        table.obs = new_obs
        table.uns = new_uns
    except Exception:
        table.obs = old_obs
        table.uns = old_uns
        raise


def _coordinates(table, coordinate_key="spatial"):
    if hasattr(table, "obsm") and coordinate_key in table.obsm:
        coords = np.asarray(table.obsm[coordinate_key])
    elif {"x", "y"}.issubset(table.obs):
        coords = table.obs[["x", "y"]].to_numpy()
    else:
        raise ValueError(
            f"Coordinates not found; expected table.obsm[{coordinate_key!r}] or obs x/y columns."
        )
    if coords.shape != (len(table.obs), 2) or not np.isfinite(coords).all():
        raise ValueError("Spatial coordinates must be a finite (n_observations, 2) array.")
    return coords


def analyze_niche_interface(
    result,
    sdata,
    *,
    niche_a,
    niche_b,
    side,
    program_niche,
    radius,
    near_distance,
    interior_min_distance,
    distance_bins=6,
    coordinate_key="spatial",
    microns_per_unit=None,
    table_key=None,
    bin_size=8,
):
    """Describe supporting-program expression versus one chosen A/B interface."""
    if not isinstance(result, BulkNicheResult):
        raise TypeError("result must be a BulkNicheResult.")
    niche_a, niche_b, side, program_niche = map(str, (niche_a, niche_b, side, program_niche))
    if niche_a == niche_b:
        raise ValueError("niche_a and niche_b must be different niches.")
    if side not in {niche_a, niche_b}:
        raise ValueError("side must equal niche_a or niche_b.")
    other = niche_b if side == niche_a else niche_a
    radius = _finite_number(radius, "radius", minimum=0)
    near_distance = _finite_number(near_distance, "near_distance", minimum=0)
    interior_min_distance = _finite_number(
        interior_min_distance, "interior_min_distance", minimum=0
    )
    if near_distance >= interior_min_distance:
        raise ValueError("near_distance must be less than interior_min_distance.")
    if radius == 0:
        raise ValueError("radius must be greater than zero.")
    distance_bins = _positive_integer(distance_bins, "distance_bins")
    scale = 1.0 if microns_per_unit is None else _finite_number(
        microns_per_unit, "microns_per_unit", minimum=0
    )
    if scale == 0:
        raise ValueError("microns_per_unit must be greater than zero.")
    table = get_table(sdata, bin_size=bin_size, table_key=table_key)
    coords = _coordinates(table, coordinate_key) * scale
    sample_column = result.parameters["sample_column"]
    patient_column = result.parameters["patient_column"]
    condition_column = result.parameters["condition_column"]
    samples = (
        table.obs[sample_column].astype(str)
        if sample_column is not None
        else pd.Series("sample", index=table.obs.index)
    )
    labels = result.niche_labels.reindex(table.obs.index)
    available_programs = set(result.spatial_scores["program_niche"].astype(str))
    if program_niche not in available_programs:
        raise ValueError(
            f"Unknown program_niche {program_niche!r}. Available programs: "
            f"{sorted(available_programs)}."
        )
    score_rows = result.spatial_scores.loc[
        result.spatial_scores["program_niche"].astype(str) == program_niche
    ].set_index("observation_id")
    score_lookup = score_rows["score"]
    rows = []
    interface_counts = {}
    for specimen in samples.drop_duplicates():
        positions = np.flatnonzero(samples.to_numpy() == specimen)
        local_coords = coords[positions]
        local_labels = labels.iloc[positions].to_numpy()
        pairs = cKDTree(local_coords).query_pairs(radius, output_type="ndarray")
        side_local = np.flatnonzero(local_labels == side)
        side_map = {position: i for i, position in enumerate(side_local)}
        adjacency = [[] for _ in side_local]
        boundary = set()
        for first, second in pairs:
            first_label, second_label = local_labels[first], local_labels[second]
            length = float(np.linalg.norm(local_coords[first] - local_coords[second]))
            if first_label == side and second_label == side:
                a, b = side_map[first], side_map[second]
                adjacency[a].append((b, length))
                adjacency[b].append((a, length))
            elif first_label == side and second_label == other:
                boundary.add(side_map[first])
            elif second_label == side and first_label == other:
                boundary.add(side_map[second])
        interface_counts[str(specimen)] = len(boundary)
        distances = np.full(len(side_local), np.inf)
        queue = []
        for node in boundary:
            distances[node] = 0.0
            heapq.heappush(queue, (0.0, node))
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != distances[node]:
                continue
            for neighbor, edge_length in adjacency[node]:
                candidate = distance + edge_length
                if candidate < distances[neighbor]:
                    distances[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor))
        for local_node, local_position in enumerate(side_local):
            table_position = positions[local_position]
            observation_id = str(table.obs.index[table_position])
            if not boundary:
                status = "no_interface"
                distance = np.nan
            elif not np.isfinite(distances[local_node]):
                status = "unreachable"
                distance = np.nan
            elif distances[local_node] == 0:
                status = "boundary"
                distance = 0.0
            else:
                status = "reachable"
                distance = float(distances[local_node])
            row = {
                "observation_id": observation_id,
                "specimen": str(specimen),
                "niche": side,
                "program_niche": program_niche,
                "distance": distance,
                "score": float(score_lookup.get(observation_id, np.nan)),
                "score_status": "available" if pd.notna(score_lookup.get(observation_id, np.nan)) else "unavailable",
                "status": status,
                "x": float(local_coords[local_position, 0]),
                "y": float(local_coords[local_position, 1]),
            }
            if patient_column is not None:
                row["patient"] = str(table.obs.iloc[table_position][patient_column])
            if condition_column is not None:
                row["condition"] = str(table.obs.iloc[table_position][condition_column])
            rows.append(row)
    observation_columns = [
        "observation_id", "specimen", "niche", "program_niche", "distance",
        "score", "score_status", "status", "x", "y",
    ]
    if patient_column is not None:
        observation_columns.append("patient")
    if condition_column is not None:
        observation_columns.append("condition")
    observations = pd.DataFrame(rows, columns=observation_columns)
    binned_rows = []
    effect_rows = []
    for specimen in samples.drop_duplicates():
        group = observations.loc[observations["specimen"] == str(specimen)]
        usable = group.loc[group["distance"].notna() & group["score"].notna()].copy()
        if len(usable):
            max_distance = float(usable["distance"].max())
            edges = np.linspace(0, max(max_distance, 1e-12), distance_bins + 1)
            usable["distance_bin"] = pd.cut(
                usable["distance"], bins=edges, include_lowest=True, duplicates="drop"
            )
            for distance_bin, values in usable.groupby("distance_bin", observed=True):
                binned_rows.append(
                    {
                        "specimen": specimen,
                        "distance_bin": str(distance_bin),
                        "distance_midpoint": float(values["distance"].mean()),
                        "n_observations": int(len(values)),
                        "mean_score": float(values["score"].mean()),
                        "median_score": float(values["score"].median()),
                    }
                )
        near = usable.loc[usable["distance"] <= near_distance, "score"]
        interior = usable.loc[usable["distance"] >= interior_min_distance, "score"]
        if group.empty:
            effect_status = "no_side_observations"
        elif not group["status"].isin({"boundary", "reachable"}).any():
            effect_status = "no_interface"
        elif not group["score"].notna().any():
            effect_status = "unavailable_scores"
        elif not len(near) or not len(interior):
            effect_status = "insufficient_near_or_interior"
        else:
            effect_status = "available"
        effect = {
            "specimen": str(specimen),
            "status": effect_status,
            "n_near": int(len(near)),
            "n_interior": int(len(interior)),
            "mean_near": float(near.mean()) if len(near) else np.nan,
            "mean_interior": float(interior.mean()) if len(interior) else np.nan,
            "near_minus_interior": float(near.mean() - interior.mean()) if len(near) and len(interior) else np.nan,
        }
        for column in ("patient", "condition"):
            if column in group:
                if len(group):
                    effect[column] = group[column].iloc[0]
                else:
                    table_column = patient_column if column == "patient" else condition_column
                    specimen_positions = np.flatnonzero(samples.to_numpy() == specimen)
                    effect[column] = str(table.obs.iloc[specimen_positions[0]][table_column])
        effect_rows.append(effect)
    parameters = {
        "niche_a": niche_a,
        "niche_b": niche_b,
        "side": side,
        "program_niche": program_niche,
        "radius": radius,
        "near_distance": near_distance,
        "interior_min_distance": interior_min_distance,
        "distance_bins": distance_bins,
        "distance_unit": "micrometer" if microns_per_unit is not None else "coordinate_unit",
        "microns_per_unit": microns_per_unit,
        "boundary_definition": "side nodes with a radius-graph edge to the other niche",
    }
    binned_columns = [
        "specimen", "distance_bin", "distance_midpoint", "n_observations",
        "mean_score", "median_score",
    ]
    effect_columns = [
        "specimen", "status", "n_near", "n_interior", "mean_near",
        "mean_interior", "near_minus_interior",
    ]
    if patient_column is not None:
        effect_columns.append("patient")
    if condition_column is not None:
        effect_columns.append("condition")
    return NicheInterfaceResult(
        observations=observations,
        binned_summary=pd.DataFrame(binned_rows, columns=binned_columns),
        specimen_effects=pd.DataFrame(effect_rows, columns=effect_columns),
        parameters=parameters,
        diagnostics={"boundary_nodes_by_specimen": interface_counts},
    )


def plot_niche_bulk_associations(
    result, sdata, *, alpha=0.05, coordinate_key="spatial", table_key=None, bin_size=8, point_size=8
):
    """Plot inherited niche-level NES separately for each specimen."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    table = get_table(sdata, bin_size=bin_size, table_key=table_key)
    coords = _coordinates(table, coordinate_key)
    sample_column = result.parameters["sample_column"]
    samples = table.obs[sample_column].astype(str) if sample_column else pd.Series("sample", index=table.obs.index)
    labels = result.niche_labels.reindex(table.obs.index)
    enrichment = result.enrichment.set_index("niche")
    nes = labels.map(enrichment["nes"])
    qvalue = labels.map(enrichment["bh_nominal_qvalue_across_niches"])
    status = labels.map(enrichment["test_status"])
    finite = result.enrichment.loc[result.enrichment["test_status"] == "tested", "nes"].dropna()
    limit = float(finite.abs().max()) if len(finite) else 1.0
    limit = limit or 1.0
    unique_samples = samples.drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(unique_samples), figsize=(4 * len(unique_samples), 4), squeeze=False)
    for axis, specimen in zip(axes[0], unique_samples):
        mask = samples.eq(specimen).to_numpy()
        significant = mask & qvalue.le(alpha).fillna(False).to_numpy()
        nonsignificant = mask & status.eq("tested").to_numpy() & ~significant
        unavailable = mask & ~status.eq("tested").to_numpy()
        axis.scatter(coords[nonsignificant, 0], coords[nonsignificant, 1], c="#c7c7c7", s=point_size)
        axis.scatter(coords[unavailable, 0], coords[unavailable, 1], c="#222222", marker="x", s=point_size)
        points = axis.scatter(
            coords[significant, 0], coords[significant, 1], c=nes[significant], s=point_size,
            cmap="coolwarm", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        )
        fig.colorbar(points, ax=axis, label="niche-level bulk association (NES)")
        axis.set_title(str(specimen))
        axis.invert_yaxis()
        axis.set_aspect("equal")
    fig.tight_layout()
    return fig, axes[0]


def _resolve_score_limits(values, limits):
    if limits is not None:
        if len(limits) != 2:
            raise ValueError("limits must contain (vmin, vmax).")
        vmin = _finite_number(limits[0], "limits[0]")
        vmax = _finite_number(limits[1], "limits[1]")
        if vmin >= vmax:
            raise ValueError("limits must satisfy vmin < vmax.")
        return vmin, vmax
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return 0.0, 1.0
    vmin = min(0.0, float(finite.min()))
    vmax = float(finite.max())
    return (vmin, vmax) if vmin < vmax else (vmin, vmin + 1.0)


def plot_niche_supporting_program(
    result, sdata, *, program_niche, coordinate_key="spatial", table_key=None, bin_size=8, point_size=8, limits=None
):
    """Plot one fixed leading-edge expression program across specimens."""
    import matplotlib.pyplot as plt

    table = get_table(sdata, bin_size=bin_size, table_key=table_key)
    coords = _coordinates(table, coordinate_key)
    sample_column = result.parameters["sample_column"]
    samples = table.obs[sample_column].astype(str) if sample_column else pd.Series("sample", index=table.obs.index)
    program = result.spatial_scores.loc[
        result.spatial_scores["program_niche"].astype(str) == str(program_niche)
    ].set_index("observation_id")
    if program.empty:
        available = sorted(set(result.spatial_scores["program_niche"].astype(str)))
        raise ValueError(
            f"Unknown program_niche {program_niche!r}. Available programs: {available}."
        )
    values = program["score"].reindex(table.obs.index.astype(str))
    direction = program["bulk_direction"].dropna().iloc[0] if program["bulk_direction"].notna().any() else "unavailable"
    vmin, vmax = _resolve_score_limits(values, limits)
    unique_samples = samples.drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(unique_samples), figsize=(4 * len(unique_samples), 4), squeeze=False)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d9d9")
    for axis, specimen in zip(axes[0], unique_samples):
        mask = samples.eq(specimen).to_numpy()
        specimen_values = values.to_numpy(dtype=float)[mask]
        finite = np.isfinite(specimen_values)
        axis.scatter(
            coords[mask][~finite, 0], coords[mask][~finite, 1],
            c="#d9d9d9", s=point_size,
        )
        points = axis.scatter(
            coords[mask][finite, 0], coords[mask][finite, 1], c=specimen_values[finite], s=point_size,
            cmap=cmap, vmin=vmin, vmax=vmax,
        )
        outline = mask & result.niche_labels.reindex(table.obs.index).astype("object").eq(str(program_niche)).fillna(False).to_numpy()
        axis.scatter(coords[outline, 0], coords[outline, 1], facecolors="none", edgecolors="black", s=point_size * 1.5)
        fig.colorbar(points, ax=axis, label="supporting-program expression score")
        axis.set_title(f"{specimen}: {program_niche} ({direction})")
        axis.invert_yaxis()
        axis.set_aspect("equal")
    fig.tight_layout()
    return fig, axes[0]


def plot_bulk_niche_evidence(result):
    """Plot sample-balanced niche composition and bulk NES evidence."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    composition = result.niche_composition
    if composition.empty:
        axes[0].text(0.5, 0.5, "Composition unavailable", ha="center", va="center")
        axes[0].set_axis_off()
    else:
        image = axes[0].imshow(composition.to_numpy(), aspect="auto", cmap="viridis")
        axes[0].set_yticks(np.arange(len(composition)), composition.index.astype(str))
        axes[0].set_xticks(np.arange(len(composition.columns)), composition.columns.astype(str), rotation=90)
        axes[0].set_title("Sample-balanced niche composition")
        fig.colorbar(image, ax=axes[0])
    enrichment = result.enrichment
    colors = np.where(enrichment["nes"].fillna(0) >= 0, "#b2182b", "#2166ac")
    axes[1].bar(enrichment["niche"].astype(str), enrichment["nes"], color=colors)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("NES")
    axes[1].set_title("Niche-level bulk association")
    fig.tight_layout()
    return fig, axes


def plot_niche_interface(interface_result, *, limits=None):
    """Plot interface observations, per-specimen curves, and effects."""
    import matplotlib.pyplot as plt

    if not isinstance(interface_result, NicheInterfaceResult):
        raise TypeError("interface_result must be a NicheInterfaceResult.")
    observations = interface_result.observations
    specimens = interface_result.specimen_effects["specimen"].astype(str).tolist()
    vmin, vmax = _resolve_score_limits(observations["score"], limits)
    fig, axes = plt.subplots(1, len(specimens) + 2, figsize=(4 * (len(specimens) + 2), 4), squeeze=False)
    for axis, specimen in zip(axes[0], specimens):
        group = observations.loc[observations["specimen"] == specimen]
        finite = group["score"].notna().to_numpy()
        axis.scatter(
            group.loc[~finite, "x"], group.loc[~finite, "y"],
            c="#d9d9d9", s=10,
        )
        points = axis.scatter(
            group.loc[finite, "x"], group.loc[finite, "y"],
            c=group.loc[finite, "score"], cmap="viridis", s=10,
            vmin=vmin, vmax=vmax,
        )
        boundary = group["status"] == "boundary"
        axis.scatter(group.loc[boundary, "x"], group.loc[boundary, "y"], facecolors="none", edgecolors="red", s=24)
        fig.colorbar(points, ax=axis)
        effect = interface_result.specimen_effects.loc[
            interface_result.specimen_effects["specimen"] == specimen
        ]
        effect_status = effect["status"].iloc[0]
        if group.empty or not finite.any():
            axis.text(0.5, 0.5, effect_status.replace("_", " "), transform=axis.transAxes, ha="center")
        axis.set_title(f"{specimen} interface ({effect_status})")
        axis.invert_yaxis()
        axis.set_aspect("equal")
    curve_axis = axes[0, -2]
    for specimen, group in interface_result.binned_summary.groupby("specimen", sort=False):
        effect = interface_result.specimen_effects.loc[
            interface_result.specimen_effects["specimen"] == specimen
        ]
        condition = effect["condition"].iloc[0] if "condition" in effect and len(effect) else None
        label = str(specimen) if condition is None else f"{specimen} ({condition})"
        curve_axis.plot(group["distance_midpoint"], group["mean_score"], marker="o", label=label)
    curve_axis.set_xlabel(f"Distance ({interface_result.parameters['distance_unit']})")
    curve_axis.set_ylabel("Mean program score")
    if curve_axis.lines:
        curve_axis.legend()
    effects = interface_result.specimen_effects
    effect_labels = effects["specimen"].astype(str)
    if "condition" in effects:
        effect_labels = effect_labels + "\n" + effects["condition"].astype(str)
    axes[0, -1].bar(effect_labels, effects["near_minus_interior"])
    axes[0, -1].axhline(0, color="black", linewidth=0.8)
    axes[0, -1].set_ylabel("Near minus interior")
    axes[0, -1].tick_params(axis="x", rotation=90)
    fig.tight_layout()
    return fig, axes[0]


__all__ = [
    "BulkNicheResult",
    "NicheInterfaceResult",
    "analyze_bulk_niches",
    "analyze_niche_interface",
    "plot_niche_bulk_associations",
    "plot_niche_supporting_program",
    "plot_bulk_niche_evidence",
    "plot_niche_interface",
]

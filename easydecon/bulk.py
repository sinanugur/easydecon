"""Map a signed bulk differential-expression signature onto spatial data."""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata

from ._schema import get_table
from .config import config


@dataclass
class PreparedBulkSignature:
    """A spatial-independent, reusable bulk signature selection."""

    selected_genes: pd.DataFrame
    diagnostics: dict
    parameters: dict


@dataclass
class BulkSignatureResult:
    """Scores and metadata returned by :func:`map_bulk_signature`."""

    scores: pd.DataFrame
    selected_genes: pd.DataFrame
    diagnostics: dict
    parameters: dict
    celltype_summary: pd.DataFrame | None = None


def _finite_number(value, name, *, minimum=None, maximum=None):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}.")
    return result


def _positive_integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if result != value or result < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _read_bulk_table(bulk_de):
    if isinstance(bulk_de, pd.DataFrame):
        return bulk_de.copy()
    if not isinstance(bulk_de, (str, Path)):
        raise TypeError("bulk_de must be a pandas DataFrame, CSV/TSV path, or PreparedBulkSignature.")
    path = Path(bulk_de)
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError("bulk_de files must use a .csv, .tsv, or .txt extension.")


def prepare_bulk_signature(
    bulk_de,
    *,
    signature_name="bulk_signature",
    gene_column="gene",
    lfc_column="log2FoldChange",
    padj_column="padj",
    gene_from_index=False,
    padj_max=0.05,
    abs_lfc_min=0.5,
    top_n_per_direction=100,
):
    """Select deterministic up/down signatures from a bulk DE result.

    Selection is independent of any spatial dataset. Positive fold changes
    always refer to the numerator condition in the upstream comparison.
    """
    if not isinstance(signature_name, str) or not signature_name.strip():
        raise ValueError("signature_name must be a non-empty string.")
    padj_max = _finite_number(padj_max, "padj_max", minimum=0, maximum=1)
    abs_lfc_min = _finite_number(abs_lfc_min, "abs_lfc_min", minimum=0)
    if top_n_per_direction is not None:
        top_n_per_direction = _positive_integer(
            top_n_per_direction, "top_n_per_direction"
        )

    frame = _read_bulk_table(bulk_de)
    if gene_from_index:
        genes = pd.Series(frame.index, index=frame.index)
    else:
        if gene_column not in frame.columns:
            raise ValueError(
                f"Gene column {gene_column!r} was not found. Available columns: "
                f"{', '.join(map(str, frame.columns))}. Use gene_from_index=True "
                "when gene identifiers are stored in the index."
            )
        genes = frame[gene_column]
    missing = [column for column in (lfc_column, padj_column) if column not in frame]
    if missing:
        raise ValueError(
            f"Required bulk DE column(s) not found: {', '.join(map(repr, missing))}. "
            f"Available columns: {', '.join(map(str, frame.columns))}."
        )
    if genes.isna().any() or genes.astype(str).str.strip().eq("").any():
        raise ValueError("Bulk DE gene identifiers must not be missing or blank.")
    genes = genes.astype(str)
    duplicate_genes = genes[genes.duplicated(keep=False)].drop_duplicates().tolist()
    if duplicate_genes:
        raise ValueError(
            "Bulk DE gene identifiers must be unique; resolve duplicates explicitly. "
            f"Duplicate examples: {duplicate_genes[:5]}."
        )

    lfc = pd.to_numeric(frame[lfc_column], errors="coerce")
    padj = pd.to_numeric(frame[padj_column], errors="coerce")
    finite_padj = np.isfinite(padj)
    invalid_probability = finite_padj & ((padj < 0) | (padj > 1))
    if invalid_probability.any():
        raise ValueError("Finite adjusted p-values must lie within [0, 1].")
    valid_numeric = np.isfinite(lfc) & finite_padj
    eligible = (
        valid_numeric
        & (padj <= padj_max)
        & (lfc.abs() >= abs_lfc_min)
        & (lfc != 0)
    )

    prepared = frame.loc[eligible].copy().reset_index(drop=True)
    prepared["gene"] = genes.loc[eligible].to_numpy()
    prepared["log2FoldChange"] = lfc.loc[eligible].to_numpy(dtype=float)
    prepared["padj"] = padj.loc[eligible].to_numpy(dtype=float)
    prepared["direction"] = np.where(prepared["log2FoldChange"] > 0, "up", "down")
    prepared["_abs_lfc"] = prepared["log2FoldChange"].abs()
    prepared = prepared.sort_values(
        ["direction", "_abs_lfc", "padj", "gene"],
        ascending=[True, False, True, True],
        kind="stable",
    )
    if top_n_per_direction is not None:
        prepared = prepared.groupby("direction", sort=False, group_keys=False).head(
            top_n_per_direction
        )
    prepared["signature_rank"] = prepared.groupby(
        "direction", sort=False
    ).cumcount() + 1
    prepared = prepared.drop(columns="_abs_lfc").reset_index(drop=True)
    if prepared.empty:
        raise ValueError("No genes passed the bulk signature filters.")

    diagnostics = {
        "n_input_rows": int(len(frame)),
        "n_invalid_lfc": int((~np.isfinite(lfc)).sum()),
        "n_invalid_padj": int((~finite_padj).sum()),
        "n_excluded_invalid_numeric": int((~valid_numeric).sum()),
        "n_eligible_before_top_n": int(eligible.sum()),
        "n_selected": int(len(prepared)),
        "n_selected_up": int((prepared["direction"] == "up").sum()),
        "n_selected_down": int((prepared["direction"] == "down").sum()),
        "warnings": [],
    }
    parameters = {
        "signature_name": signature_name,
        "gene_column": gene_column,
        "lfc_column": lfc_column,
        "padj_column": padj_column,
        "gene_from_index": bool(gene_from_index),
        "padj_max": padj_max,
        "abs_lfc_min": abs_lfc_min,
        "top_n_per_direction": top_n_per_direction,
    }
    return PreparedBulkSignature(prepared, diagnostics, parameters)


def _scoring_background(table, scoring_gene_universe):
    var_names = pd.Index(map(str, table.var_names))
    duplicates = var_names[var_names.duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(
            "Spatial var_names must be unique; resolve duplicates explicitly. "
            f"Duplicate examples: {duplicates[:5]}."
        )
    if scoring_gene_universe is None:
        return var_names, np.arange(len(var_names), dtype=int), False
    background = pd.Index(map(str, scoring_gene_universe))
    if background.has_duplicates:
        raise ValueError("scoring_gene_universe must contain unique gene identifiers.")
    if len(background) == 0:
        raise ValueError("scoring_gene_universe must not be empty.")
    missing = background.difference(var_names).tolist()
    if missing:
        raise ValueError(
            "Every scoring_gene_universe gene must be measured in the spatial table. "
            f"Missing examples: {missing[:5]}."
        )
    positions = var_names.get_indexer(background)
    return background, positions, True


def _resolve_matrix(table, layer):
    if layer is None:
        return table.X
    layers = getattr(table, "layers", None)
    if layers is None or layer not in layers:
        raise ValueError(f"Layer {layer!r} was not found in the spatial table.")
    return layers[layer]


def _score_programs(matrix, marker_codes, program_codes, rank_cap, batch_size):
    n_obs, n_genes = matrix.shape
    effective_cap = min(rank_cap, n_genes)
    marker_lookup = np.full(n_genes, -1, dtype=int)
    marker_lookup[marker_codes] = np.arange(len(marker_codes))
    marker_credits = np.zeros(len(marker_codes), dtype=float)
    marker_detected = np.zeros(len(marker_codes), dtype=bool)
    scores = np.zeros((n_obs, len(program_codes)), dtype=float)
    detected = np.zeros((n_obs, len(program_codes)), dtype=int)
    all_zero = np.zeros(n_obs, dtype=bool)

    is_sparse = sparse.issparse(matrix)
    if is_sparse:
        values = matrix.data
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Spatial expression must contain only finite, nonnegative values.")
        matrix = matrix.tocsr(copy=not getattr(matrix, "has_canonical_format", True))
        if not matrix.has_canonical_format:
            matrix.sum_duplicates()

    for start in range(0, n_obs, batch_size):
        stop = min(start + batch_size, n_obs)
        chunk = matrix[start:stop]
        if not is_sparse:
            chunk = np.asarray(chunk)
            if not np.issubdtype(chunk.dtype, np.number):
                raise ValueError("Spatial expression must be numeric.")
            if not np.isfinite(chunk).all() or (chunk < 0).any():
                raise ValueError(
                    "Spatial expression must contain only finite, nonnegative values."
                )
        for offset in range(stop - start):
            if is_sparse:
                row = chunk.getrow(offset)
                positive = row.data > 0
                indices = row.indices[positive]
                values = row.data[positive]
            else:
                row = np.asarray(chunk[offset]).ravel()
                indices = np.flatnonzero(row > 0)
                values = row[indices]
            output_row = start + offset
            if len(indices) == 0:
                all_zero[output_row] = True
                scores[output_row] = np.nan
                continue
            ranks = rankdata(-values, method="average")
            credits = np.clip((effective_cap - ranks + 1) / effective_cap, 0, 1)
            codes = marker_lookup[indices]
            present = codes >= 0
            marker_credits.fill(0)
            marker_detected.fill(False)
            marker_credits[codes[present]] = credits[present]
            marker_detected[codes[present]] = True
            for program, selected in enumerate(program_codes):
                if len(selected) == 0:
                    scores[output_row, program] = np.nan
                    continue
                direction_values = marker_credits[selected]
                detected[output_row, program] = int(marker_detected[selected].sum())
                scores[output_row, program] = float(direction_values.mean())
    return scores, detected, all_zero, effective_cap


def _score_rows(matrix, marker_codes, direction_codes, rank_cap, batch_size):
    return _score_programs(
        matrix, marker_codes, direction_codes, rank_cap, batch_size
    )


def _summarize_by_annotation(table, scores, celltype_column, sample_column):
    if celltype_column not in table.obs:
        raise ValueError(f"celltype_column {celltype_column!r} was not found in table.obs.")
    if sample_column is not None and sample_column not in table.obs:
        raise ValueError(f"sample_column {sample_column!r} was not found in table.obs.")
    keys = ([sample_column] if sample_column is not None else []) + [celltype_column]
    annotation = table.obs[keys].copy()
    valid = annotation.notna().all(axis=1)
    joined = annotation.loc[valid].join(scores)
    rows = []
    for values, group in joined.groupby(keys, sort=False, observed=True):
        values = values if isinstance(values, tuple) else (values,)
        row = dict(zip(keys, values))
        row["n_observations"] = int(len(group))
        for direction in ("up", "down", "balance"):
            column = f"{direction}_score"
            valid_scores = group[column].dropna()
            row[f"n_valid_{column}"] = int(len(valid_scores))
            row[f"{direction}_score_coverage"] = float(len(valid_scores) / len(group))
            row[f"mean_{column}"] = float(valid_scores.mean()) if len(valid_scores) else np.nan
            row[f"median_{column}"] = float(valid_scores.median()) if len(valid_scores) else np.nan
        for direction in ("up", "down"):
            column = f"{direction}_detected_fraction"
            row[f"mean_{column}"] = float(group[column].mean())
        rows.append(row)
    return pd.DataFrame(rows), int((~valid).sum())


def map_bulk_signature(
    sdata,
    bulk_de,
    *,
    signature_name="bulk_signature",
    gene_column="gene",
    lfc_column="log2FoldChange",
    padj_column="padj",
    gene_from_index=False,
    padj_max=0.05,
    abs_lfc_min=0.5,
    top_n_per_direction=100,
    min_genes=5,
    rank_cap=1500,
    scoring_gene_universe=None,
    table_key=None,
    bin_size=8,
    layer=None,
    celltype_column=None,
    sample_column=None,
    store=False,
    overwrite=False,
):
    """Score a bulk up/down signature in each spatial observation.

    The UCell-like score is the mean per-gene rank credit across the measured
    signature. A detected gene ranked first receives 1, rank ``rank_cap``
    receives ``1 / rank_cap``, and lower-ranked or zero-expression genes
    receive 0. Ranks use the full measured scoring background and average
    ties. A direction requires at least ``min_genes`` measured genes.
    """
    min_genes = _positive_integer(min_genes, "min_genes")
    rank_cap = _positive_integer(rank_cap, "rank_cap")
    if isinstance(bulk_de, PreparedBulkSignature):
        prepared = bulk_de
    else:
        prepared = prepare_bulk_signature(
            bulk_de,
            signature_name=signature_name,
            gene_column=gene_column,
            lfc_column=lfc_column,
            padj_column=padj_column,
            gene_from_index=gene_from_index,
            padj_max=padj_max,
            abs_lfc_min=abs_lfc_min,
            top_n_per_direction=top_n_per_direction,
        )
    table = get_table(sdata, bin_size=bin_size, table_key=table_key)
    background, background_positions, explicit_background = _scoring_background(
        table, scoring_gene_universe
    )
    matrix = _resolve_matrix(table, layer)
    if matrix.shape != (len(table.obs), len(table.var_names)):
        raise ValueError("The selected expression matrix shape does not match table.obs/table.var_names.")
    if explicit_background:
        matrix = matrix[:, background_positions]

    selected = prepared.selected_genes.copy()
    background_set = set(background)
    spatial_set = set(map(str, table.var_names))
    selected["spatially_measured"] = selected["gene"].isin(spatial_set)
    selected["available_for_scoring"] = selected["gene"].isin(background_set)
    direction_genes = {
        direction: selected.loc[
            (selected["direction"] == direction) & selected["available_for_scoring"],
            "gene",
        ].tolist()
        for direction in ("up", "down")
    }
    available = {key: len(value) >= min_genes for key, value in direction_genes.items()}
    if not any(available.values()):
        raise ValueError(
            f"Neither signature direction has at least min_genes={min_genes} genes "
            "in the spatial scoring background."
        )

    marker_genes = list(dict.fromkeys(direction_genes["up"] + direction_genes["down"]))
    marker_index = {gene: index for index, gene in enumerate(marker_genes)}
    background_index = {gene: index for index, gene in enumerate(background)}
    marker_codes = np.array([background_index[gene] for gene in marker_genes], dtype=int)
    direction_codes = tuple(
        np.array([marker_index[gene] for gene in direction_genes[direction]], dtype=int)
        for direction in ("up", "down")
    )
    configured_batch = config.batch_size
    batch_size = 1000 if configured_batch == "auto" else _positive_integer(
        configured_batch, "configured batch_size"
    )
    raw_scores, detected, all_zero, effective_cap = _score_rows(
        matrix, marker_codes, direction_codes, rank_cap, batch_size
    )
    scores = pd.DataFrame(index=table.obs.index.copy())
    for column, direction in enumerate(("up", "down")):
        n_measured = len(direction_genes[direction])
        scores[f"{direction}_score"] = raw_scores[:, column]
        if not available[direction]:
            scores[f"{direction}_score"] = np.nan
        scores[f"{direction}_detected_genes"] = detected[:, column]
        scores[f"{direction}_detected_fraction"] = (
            detected[:, column] / n_measured if n_measured else np.nan
        )
    scores["balance_score"] = scores["up_score"] - scores["down_score"]
    scores["all_zero"] = all_zero

    warnings = []
    for direction in ("up", "down"):
        if not available[direction]:
            warnings.append(
                f"{direction} direction unavailable: {len(direction_genes[direction])} "
                f"measured genes is below min_genes={min_genes}."
            )
    if all_zero.any():
        warnings.append(f"{int(all_zero.sum())} all-zero observations received NaN scores.")
    diagnostics = {
        "selection": dict(prepared.diagnostics),
        "coverage": {
            direction: {
                "n_selected": int((selected["direction"] == direction).sum()),
                "n_spatially_measured": int(
                    ((selected["direction"] == direction) & selected["spatially_measured"]).sum()
                ),
                "n_available_for_scoring": int(len(direction_genes[direction])),
                "available": bool(available[direction]),
            }
            for direction in ("up", "down")
        },
        "detection": {
            "n_all_zero_observations": int(all_zero.sum()),
            "mean_up_detected_fraction": float(scores["up_detected_fraction"].mean()),
            "mean_down_detected_fraction": float(scores["down_detected_fraction"].mean()),
        },
        "n_missing_annotations": 0,
        "warnings": warnings,
    }
    celltype_summary = None
    if celltype_column is not None:
        celltype_summary, missing_annotations = _summarize_by_annotation(
            table, scores, celltype_column, sample_column
        )
        diagnostics["n_missing_annotations"] = missing_annotations
    elif sample_column is not None:
        raise ValueError("sample_column requires celltype_column.")

    parameters = {
        **prepared.parameters,
        "layer": layer,
        "min_genes": min_genes,
        "rank_cap": rank_cap,
        "effective_rank_cap": effective_cap,
        "scoring_definition": "full-background mean capped rank credit (UCell-like)",
        "scoring_gene_background": background.tolist(),
        "explicit_scoring_gene_universe": explicit_background,
        "table_key": table_key,
        "bin_size": bin_size,
    }
    result = BulkSignatureResult(
        scores=scores,
        selected_genes=selected,
        diagnostics=diagnostics,
        parameters=parameters,
        celltype_summary=celltype_summary,
    )
    if store:
        _store_bulk_result(table, result, overwrite=overwrite)
    return result


def _storage_prefix(signature_name):
    slug = re.sub(r"[^0-9A-Za-z]+", "_", signature_name).strip("_").casefold()
    return f"easydecon_bulk_{slug or 'signature'}"


def _store_bulk_result(table, result, *, overwrite):
    if not hasattr(table, "uns"):
        raise TypeError("store=True requires an AnnData-like table with .uns.")
    prefix = _storage_prefix(result.parameters["signature_name"])
    renamed = result.scores.add_prefix(f"{prefix}_")
    uns_key = prefix
    collisions = [column for column in renamed if column in table.obs]
    if uns_key in table.uns:
        collisions.append(f"uns[{uns_key!r}]")
    if collisions and not overwrite:
        raise ValueError(
            "Bulk signature storage would overwrite existing output(s): "
            f"{', '.join(collisions)}. Use overwrite=True to replace them."
        )
    new_obs = table.obs.copy()
    for column in renamed:
        new_obs[column] = renamed[column].to_numpy()
    new_uns = dict(table.uns)
    new_uns[uns_key] = {
        "parameters": result.parameters,
        "diagnostics": result.diagnostics,
    }
    old_obs, old_uns = table.obs, table.uns
    try:
        table.obs = new_obs
        table.uns = new_uns
    except Exception:
        table.obs = old_obs
        table.uns = old_uns
        raise


def plot_bulk_signature(
    result,
    sdata=None,
    *,
    coordinates=None,
    coordinate_key="spatial",
    table_key=None,
    bin_size=8,
    include_celltype_summary=False,
    directional_limits=None,
    balance_limit=None,
    point_size=8,
    figsize=None,
):
    """Plot up, down, and signed balance scores without displaying or saving."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    if not isinstance(result, BulkSignatureResult):
        raise TypeError("result must be a BulkSignatureResult.")
    if coordinates is None:
        if sdata is None:
            raise ValueError("Provide sdata or explicit coordinates for spatial plotting.")
        table = get_table(sdata, bin_size=bin_size, table_key=table_key)
        if not hasattr(table, "obsm") or coordinate_key not in table.obsm:
            raise ValueError(
                f"Spatial coordinates {coordinate_key!r} were not found in table.obsm; "
                "provide coordinates explicitly."
            )
        if not table.obs.index.equals(result.scores.index):
            raise ValueError("Spatial table observations do not align with result.scores.")
        coordinates = table.obsm[coordinate_key]
    if isinstance(coordinates, pd.DataFrame):
        coordinates = coordinates.reindex(result.scores.index).to_numpy()
    else:
        coordinates = np.asarray(coordinates)
    if coordinates.ndim != 2 or coordinates.shape != (len(result.scores), 2):
        raise ValueError("coordinates must have shape (n_observations, 2).")
    if not np.isfinite(coordinates).all():
        raise ValueError("coordinates must contain only finite values.")

    n_panels = 4 if include_celltype_summary else 3
    if include_celltype_summary and result.celltype_summary is None:
        raise ValueError("No celltype_summary is available; map with celltype_column first.")
    fig, axes = plt.subplots(
        1, n_panels, figsize=figsize or (4 * n_panels, 4), squeeze=False
    )
    axes = axes[0]
    if directional_limits is None:
        direction_values = result.scores[["up_score", "down_score"]].to_numpy(dtype=float)
        finite_direction = direction_values[np.isfinite(direction_values)]
        direction_vmin = 0.0
        direction_vmax = float(finite_direction.max()) if len(finite_direction) else 1.0
        direction_vmax = direction_vmax or 1.0
    else:
        if len(directional_limits) != 2:
            raise ValueError("directional_limits must contain (vmin, vmax).")
        direction_vmin = _finite_number(directional_limits[0], "directional_limits[0]")
        direction_vmax = _finite_number(directional_limits[1], "directional_limits[1]")
        if direction_vmin >= direction_vmax:
            raise ValueError("directional_limits must satisfy vmin < vmax.")
    for axis, direction, cmap_name in zip(
        axes[:2], ("up", "down"), ("Reds", "Blues")
    ):
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#d9d9d9")
        values = result.scores[f"{direction}_score"].to_numpy(dtype=float)
        finite = np.isfinite(values)
        axis.scatter(
            coordinates[~finite, 0], coordinates[~finite, 1],
            c="#d9d9d9", s=point_size,
        )
        points = axis.scatter(
            coordinates[finite, 0], coordinates[finite, 1],
            c=values[finite], s=point_size, cmap=cmap,
            vmin=direction_vmin, vmax=direction_vmax,
        )
        fig.colorbar(points, ax=axis)
        axis.set_title(f"{result.parameters['signature_name']} {direction}")
        axis.invert_yaxis()
        axis.set_aspect("equal")
    balance = result.scores["balance_score"]
    limit = balance_limit
    if limit is None:
        finite = balance[np.isfinite(balance)]
        limit = float(np.abs(finite).max()) if len(finite) else 1.0
    limit = _finite_number(limit, "balance_limit", minimum=0)
    limit = limit or 1.0
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#d9d9d9")
    balance_values = balance.to_numpy(dtype=float)
    finite_balance = np.isfinite(balance_values)
    axes[2].scatter(
        coordinates[~finite_balance, 0], coordinates[~finite_balance, 1],
        c="#d9d9d9", s=point_size,
    )
    points = axes[2].scatter(
        coordinates[finite_balance, 0], coordinates[finite_balance, 1],
        c=balance_values[finite_balance], s=point_size,
        cmap=cmap, norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
    )
    fig.colorbar(points, ax=axes[2])
    axes[2].set_title(f"{result.parameters['signature_name']} balance")
    axes[2].invert_yaxis()
    axes[2].set_aspect("equal")
    if include_celltype_summary:
        summary = result.celltype_summary
        label_columns = [
            column for column in summary.columns
            if column not in {"n_observations"} and not column.startswith(("n_valid_", "mean_", "median_"))
            and not column.endswith("_coverage")
        ]
        labels = summary[label_columns].astype(str).agg(" / ".join, axis=1)
        x = np.arange(len(summary))
        axes[3].bar(x - 0.2, summary["mean_up_score"], width=0.4, label="up")
        axes[3].bar(x + 0.2, summary["mean_down_score"], width=0.4, label="down")
        axes[3].set_xticks(x, labels, rotation=90)
        axes[3].set_title("Mean score by annotation")
        axes[3].legend()
    fig.tight_layout()
    return fig, axes


__all__ = [
    "PreparedBulkSignature",
    "BulkSignatureResult",
    "prepare_bulk_signature",
    "map_bulk_signature",
    "plot_bulk_signature",
]

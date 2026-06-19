"""Public API for easydecon."""

from ._version import __version__
from ._schema import (
    MarkerSchema,
    get_table,
    resolve_marker_columns,
    standardize_marker_dataframe,
)
from .config import set_batch_size, set_n_jobs
from .diagnostics import summarize_easydecon_result, summarize_marker_table
from .easydecon import (
    assign_clusters_from_df,
    common_markers_gene_expression_and_filter,
    compute_pseudobulk_deseq_markers,
    get_clusters_by_similarity_on_tissue,
    read_markers_dataframe,
)
from .extra import EasyDeconResult, easydecon_workflow
from .niche import (
    detect_niches_from_easydecon_result,
    detect_spatial_niches_from_posteriors,
    plot_niche_compositions,
    summarize_niche_compositions,
)

run_easydecon = easydecon_workflow

__all__ = [
    "__version__",
    "read_markers_dataframe",
    "common_markers_gene_expression_and_filter",
    "get_clusters_by_similarity_on_tissue",
    "assign_clusters_from_df",
    "compute_pseudobulk_deseq_markers",
    "set_n_jobs",
    "set_batch_size",
    "easydecon_workflow",
    "run_easydecon",
    "EasyDeconResult",
    "detect_spatial_niches_from_posteriors",
    "detect_niches_from_easydecon_result",
    "summarize_niche_compositions",
    "plot_niche_compositions",
    "MarkerSchema",
    "resolve_marker_columns",
    "standardize_marker_dataframe",
    "get_table",
    "summarize_easydecon_result",
    "summarize_marker_table",
]

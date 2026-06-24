"""Public API for easydecon."""

from ._version import __version__
from ._schema import (
    MarkerSchema,
    get_table,
    resolve_marker_columns,
    standardize_marker_dataframe,
)
from ._validation import MARKER_ROLE_MODES, UCELL_MARKER_ROLES
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
from .markers import (
    PreparedMarkers,
    compute_reference_profile_markers,
    prepare_markers,
    select_prepared_markers,
)
from .niche import (
    detect_niches_from_easydecon_result,
    detect_spatial_niches_from_posteriors,
    plot_niche_compositions,
    summarize_niche_compositions,
)
from .refinement import RefinedGroupResult, refine_group

run_easydecon = easydecon_workflow

__all__ = [
    "__version__",
    "read_markers_dataframe",
    "common_markers_gene_expression_and_filter",
    "get_clusters_by_similarity_on_tissue",
    "assign_clusters_from_df",
    "compute_pseudobulk_deseq_markers",
    "PreparedMarkers",
    "compute_reference_profile_markers",
    "prepare_markers",
    "select_prepared_markers",
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
    "RefinedGroupResult",
    "refine_group",
    "UCELL_MARKER_ROLES",
    "MARKER_ROLE_MODES",
]

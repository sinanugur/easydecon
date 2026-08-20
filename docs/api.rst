API reference
=============

This page documents the public objects exported by ``easydecon.__all__``.
Private helpers and implementation modules are intentionally omitted.

.. currentmodule:: easydecon

Version
-------

.. autodata:: __version__

Core workflow
-------------

.. autofunction:: run_easydecon

.. autofunction:: easydecon_workflow

.. autoclass:: EasyDeconResult
   :members:

Marker loading and preparation
------------------------------

``prepare_markers`` owns source loading, marker generation, alias resolution,
canonical preparation, and optional source-level role inference. It returns a
spatial-unfiltered ``PreparedMarkers`` object. ``select_prepared_markers`` owns
spatial-specific marker selection. ``resolve_phase_marker_tables`` is an
internal workflow helper for Phase 1/Phase 2 routing and workflow top-N
selection. ``read_markers_dataframe`` remains a supported backward-compatible
convenience wrapper that returns a selected DataFrame.

.. autofunction:: read_markers_dataframe

.. autofunction:: prepare_markers

.. autofunction:: select_prepared_markers

.. autofunction:: compute_reference_profile_markers

.. autofunction:: compute_pseudobulk_deseq_markers

.. autoclass:: PreparedMarkers
   :members:

Phase functions and assignment
------------------------------

.. autofunction:: common_markers_gene_expression_and_filter

.. autofunction:: get_clusters_by_similarity_on_tissue

.. autofunction:: assign_clusters_from_df

.. autofunction:: add_df_to_spatialdata

Refinement
----------

.. autofunction:: refine_group

.. autoclass:: RefinedGroupResult
   :members:

Diagnostics
-----------

.. autofunction:: summarize_easydecon_result

.. autofunction:: summarize_marker_table

Niches
------

.. autofunction:: detect_spatial_niches_from_posteriors

.. autofunction:: detect_niches_from_easydecon_result

.. autofunction:: summarize_niche_compositions

.. autofunction:: plot_niche_compositions

Schema helpers
--------------

.. autoclass:: MarkerSchema
   :members:

.. autofunction:: get_table

.. autofunction:: resolve_marker_columns

.. autofunction:: standardize_marker_dataframe

Runtime configuration
---------------------

.. autofunction:: set_n_jobs

.. autofunction:: set_batch_size

Public constants
----------------

.. autodata:: UCELL_MARKER_ROLES

.. autodata:: MARKER_ROLE_MODES

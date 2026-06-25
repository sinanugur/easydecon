API Reference
=============

The following sections expose the public Python API. Objects are grouped by
module for clarity. The documentation uses ``autodoc`` to pull docstrings
directly from the source code.

.. contents::
   :local:
   :depth: 2

Package API
-----------

The top-level ``easydecon`` package exports the main workflow and helper
objects for normal use:

``run_easydecon``
   Alias for ``easydecon_workflow``.

``EasyDeconResult``
   Result object returned by ``run_easydecon(..., return_result_object=True)``.

``PreparedMarkers``, ``prepare_markers``, ``select_prepared_markers``
   Reusable marker-preparation utilities.

``RefinedGroupResult``, ``refine_group``
   Hierarchical refinement helpers.

``get_table``, ``resolve_marker_columns``, ``standardize_marker_dataframe``
   Schema and table-resolution helpers.

``summarize_easydecon_result``, ``summarize_marker_table``
   Diagnostics summaries.

``detect_niches_from_easydecon_result``, ``summarize_niche_compositions``
   Spatial niche utilities.

Workflow
--------

.. automodule:: easydecon.easydecon
   :members:
   :undoc-members:
   :show-inheritance:

Workflow Orchestration
----------------------

.. automodule:: easydecon.extra
   :members:
   :undoc-members:
   :show-inheritance:

Marker Preparation
------------------

.. automodule:: easydecon.markers
   :members:
   :undoc-members:
   :show-inheritance:

Schema Helpers
--------------

.. automodule:: easydecon._schema
   :members:
   :undoc-members:
   :show-inheritance:

Diagnostics
-----------

.. automodule:: easydecon.diagnostics
   :members:
   :undoc-members:
   :show-inheritance:

Refinement
----------

.. automodule:: easydecon.refinement
   :members:
   :undoc-members:
   :show-inheritance:

Spatial Niches
--------------

.. automodule:: easydecon.niche
   :members:
   :undoc-members:
   :show-inheritance:

Segmentation Utilities
----------------------

.. automodule:: easydecon.segmentation
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. automodule:: easydecon.config
   :members:
   :undoc-members:
   :show-inheritance:

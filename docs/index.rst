EasyDecon Documentation
=======================

``easydecon`` provides marker-gene based deconvolution utilities for
single-cell references and spatial transcriptomics. It reads or generates
marker tables, scores marker evidence in spatial locations, combines Phase 1
priors with Phase 2 likelihoods, and returns both probabilistic support
matrices and hard assignments.

The package is designed around explicit, inspectable matrices rather than a
black-box cell-fraction model. Marker quality, gene identifier overlap, spatial
resolution, and the chosen scoring method all affect interpretation, so the
guides below separate the basic workflow from advanced marker roles,
refinement, candidate pruning, and niche detection.

Start here
----------

* :doc:`usage` for installation and a minimal workflow.
* :doc:`workflow` for the conceptual flow from markers to assignments.
* :doc:`marker_inputs` for supported marker sources and table schema.
* :doc:`results` for result matrices and interpretation.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   usage
   workflow
   marker_inputs

.. toctree::
   :maxdepth: 2
   :caption: Core workflow

   phase1
   phase2
   ucell
   results

.. toctree::
   :maxdepth: 2
   :caption: Marker preparation

   prepared_markers
   scanpy_markers
   reference_markers
   marker_roles

.. toctree::
   :maxdepth: 2
   :caption: Advanced workflows

   refinement
   candidate_pruning
   niches
   visualization

.. toctree::
   :maxdepth: 2
   :caption: Validation and reference

   validation
   segmentation
   troubleshooting
   glossary
   api

Installation and quickstart
===========================

Supported Python
----------------

``easydecon`` requires Python 3.10 or newer. The package metadata currently
advertises support for Python 3.10 through 3.13.

Installation
------------

Install the release package from PyPI::

    python -m pip install easydecon

For development from a checkout::

    python -m pip install -e ".[test]"

Install optional extras only when needed::

    python -m pip install -e ".[spatial]"
    python -m pip install -e ".[deseq]"
    python -m pip install -e ".[docs]"
    python -m pip install -e ".[test]"

The ``spatial`` extra is needed for SpatialData containers and related
plotting/query helpers. The ``deseq`` extra is needed for
``marker_method="pydeseq2"``. Core AnnData workflows use the required package
dependencies.

Minimal input expectations
--------------------------

You need:

* an AnnData table, or a SpatialData object containing an AnnData-like table;
* an expression matrix whose ``var_names`` are gene identifiers;
* a marker table with at least ``group`` and ``names`` columns; and
* marker gene identifiers that overlap the spatial expression ``var_names``.

The generic term in the docs is "spatial location". A spatial location may be a
spot, bin, or segmented cell depending on the upstream data.

Minimal workflow
----------------

.. code-block:: python

    import easydecon as ed

    result = ed.run_easydecon(
        sdata=sdata,
        markers_df=markers_df,
        return_result_object=True,
        verbose=False,
    )

    result.posterior_df
    result.assigned_labels
    result.diagnostics

``posterior_df`` contains relative posterior support among tested marker
groups, not guaranteed absolute biological cell fractions. ``assigned_labels``
contains hard assignments and therefore discards uncertainty. Inspect
``diagnostics`` before relying on assignments downstream.

``run_easydecon`` defaults to the standard Phase 1 permutation workflow and
the default Phase 2 weighted Jaccard method.

Fast exploratory run
--------------------

Quantile filtering is a fast exploratory shortcut. The standard Phase 1
workflow uses permutation filtering, so final analyses should normally return
to ``filtering_algorithm="permutation"``.

.. code-block:: python

    result = ed.run_easydecon(
        sdata=sdata,
        markers_df=markers_df,
        filtering_algorithm="quantile",
        method="wjaccard",
        return_result_object=True,
        verbose=False,
    )

Next steps
----------

* :doc:`marker_inputs` explains marker tables, generated markers, and
  ``PreparedMarkers``.
* :doc:`workflow` explains the full Phase 1 -> Phase 2 -> posterior flow.
* :doc:`results` explains which matrix to use for each downstream task.
* :doc:`visualization` gives Matplotlib recipes for maps and summaries.

Compatibility tuple return
--------------------------

Without ``return_result_object=True``, ``run_easydecon`` returns the historical
five-value tuple::

    phase1_result, phase2_result, assigned_labels, priors_df, assignment_df = ed.run_easydecon(
        sdata=sdata,
        markers_df=markers_df,
    )

Set ``return_diagnostics=True`` to append the diagnostics dictionary to that
tuple. New code should prefer ``return_result_object=True`` because it exposes
``likelihoods_df``, ``posterior_df``, ``assignment_df``, and marker diagnostics
with stable attribute names.

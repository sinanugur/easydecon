Getting Started
===============

Installation
------------

It is recommended to install the package in a virtual environment or a Conda
environment. To create a Conda environment, run the following command::

    conda create -n easydecon python=3.10.14
    conda activate easydecon

You can install from PyPI::

    pip install easydecon

Optional extras keep heavier dependencies out of the base install::

    pip install "easydecon[spatial]"
    pip install "easydecon[deseq]"

To install directly from GitHub using pip into the active environment, run the
following command::

    pip install git+https://github.com/sinanugur/easydecon.git


Absolute Minimal Usage
----------------------

.. code-block:: python

    import easydecon as ed

    # sdata can be a SpatialData object or an AnnData table with spatial data.
    # Marker tables need at least group and names columns.
    markers_df = ed.read_markers_dataframe(
        sdata,
        filename="scanpy_deseq_table.csv",
    )

    result = ed.run_easydecon(
        sdata,
        markers_df=markers_df,
        filtering_algorithm="quantile",
        method="wjaccard",
        return_result_object=True,
        verbose=False,
    )

    result.phase1_result
    result.priors_df
    result.phase2_result
    result.likelihoods_df
    result.posterior_df
    result.assignment_df
    result.assigned_labels
    result.diagnostics

The result object is the preferred interface because it exposes both raw
evidence and normalized matrices. ``posterior_df`` is the preferred
probabilistic output when available. If ``marker_genes`` is provided as a plain
list, Phase 1 is used as a location mask and ``posterior_df`` is ``None``; use
``assignment_df`` or ``phase2_result`` for that workflow.

Legacy Tuple Return
-------------------

Without ``return_result_object=True``, ``run_easydecon`` keeps the historical
five-value tuple return::

    phase1_result, phase2_result, assigned_labels, priors_df, assignment_df = ed.run_easydecon(
        sdata,
        markers_df=markers_df,
    )

Set ``return_diagnostics=True`` to append the diagnostics dictionary to that
tuple.

Marker Sources
--------------

``run_easydecon`` can use an existing marker table, generate Scanpy markers,
generate pseudobulk PyDESeq2 markers, or reuse prepared markers.

.. code-block:: python

    result = ed.run_easydecon(
        sdata,
        adata=sc_adata,
        groupby="cell_type",
        marker_method="scanpy",
        filtering_algorithm="quantile",
        return_result_object=True,
        verbose=False,
    )

    prepared = ed.prepare_markers(
        sc_adata,
        marker_method="scanpy",
        groupby="cell_type",
    )

    result = ed.run_easydecon(
        sdata,
        prepared_markers=prepared,
        return_result_object=True,
    )

Use ``marker_method="pydeseq2"`` with ``sample_col`` and raw count data for
pseudobulk marker generation. Use ``marker_method="reference"`` for lightweight
reference-profile markers.

Runtime Configuration
---------------------

.. code-block:: python

    ed.set_n_jobs(1)        # serial execution
    ed.set_n_jobs(-1)       # all available CPUs
    ed.set_batch_size(256)  # joblib batch size
    ed.set_batch_size("auto")

``set_n_jobs(0)`` is rejected. ``set_batch_size`` accepts a positive integer or
``"auto"``.

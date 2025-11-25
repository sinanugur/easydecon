Getting Started
===============

Installation
------------

It is recommended to install the package in a virtual environment or a Conda environment. To create a Conda environment, run the following command::

    conda create -n easydecon python=3.10.14
    conda activate easydecon

You can install from PyPi::

    pip install easydecon

To install directly from GitHub using pip into the active environment, run the
following command::

    pip install git+https://github.com/sinanugur/easydecon.git


Absolute Minimal Usage
----------------------

.. code-block:: python

    from easydecon.easydecon import *
    from easydecon.config import *
    from easydecon.extra import *

    # read your DESeq table into a markers_df
    # sdata is your VisiumHD file in SpatialData format or segmented AnnData object,
    # assumed you QC and etc.
    markers_df = read_markers_dataframe(sdata, filename="scanpy_deseq_table.csv")

    # run easydecon
    ph1, ph2, assigned_labels, posterior_df, proportions_df = easydecon_workflow(
        sdata, markers_df=markers_df
    )

    # or setting prior genes
    ph1, ph2, assigned_labels, posterior_df, proportions_df = easydecon_workflow(
        sdata,
        markers_df=markers_df,
        marker_genes=["gene1", "gene2", "gene3"],
    )

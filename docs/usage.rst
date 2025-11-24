Getting Started
===============

Installation
------------

Install ``easydecon`` from source to access the latest features::

   git clone https://github.com/your-org/easydecon.git
   cd easydecon
   pip install -e .[dev]

If you plan to build the documentation locally, install the optional docs
requirements::

   pip install -r docs/requirements.txt

Quick Usage
-----------

Once installed, import the package to access the high-level utilities::

   import easydecon
   from easydecon.easydecon import common_markers_gene_expression_and_filter

Refer to the :doc:`api` section for full details on the available modules and
functions.

Read the Docs Configuration
---------------------------

This repository includes a minimal ``.readthedocs.yaml`` configuration so the
site can be built on `Read the Docs <https://readthedocs.org/>`_. The build uses
Sphinx with the Read the Docs theme and mocks heavy scientific dependencies to
keep API generation lightweight.

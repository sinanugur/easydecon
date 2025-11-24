import os
import sys

# Add project root to sys.path for autodoc
sys.path.insert(0, os.path.abspath('..'))

project = 'easydecon'
author = 'Project contributors'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': False,
}

# Mock optional dependencies to keep autodoc light-weight
autodoc_mock_imports = [
    'scanpy',
    'fireducks',
    'pandas',
    'spatialdata',
    'spatialdata_io',
    'spatialdata_plot',
    'numpy',
    'scipy',
    'tqdm',
    'sklearn',
]

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

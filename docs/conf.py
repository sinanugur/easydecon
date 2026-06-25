import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/easydecon-matplotlib")

# Add project root to sys.path for autodoc
sys.path.insert(0, os.path.abspath('..'))

project = 'easydecon'
author = 'Project contributors'
release = (
    (Path(__file__).resolve().parents[1] / "easydecon" / "_version.py")
    .read_text(encoding="utf-8")
    .split('__version__ = "')[1]
    .split('"')[0]
)

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': False,
}

# Mock optional/problematic dependencies to keep autodoc light-weight.
autodoc_mock_imports = [
    'scanpy',
    'fireducks',
    'spatialdata',
    'spatialdata_io',
    'spatialdata_plot',
    'squidpy',
    'tensorflow',
    'bin2cell'
]


html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static'] if (Path(__file__).parent / '_static').exists() else []

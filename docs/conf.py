"""Sphinx configuration for easydecon documentation."""

from __future__ import annotations

import os
import re
import sys
from importlib.metadata import version as package_version
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parent
ROOT = DOCS_DIR.parent

os.environ.setdefault("MPLCONFIGDIR", "/tmp/easydecon-matplotlib")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

sys.path.insert(0, str(ROOT))

project = "easydecon"
author = "Project contributors"


def _fallback_version() -> str:
    version_file = ROOT / "easydecon" / "_version.py"
    match = re.search(
        r"^__version__\s*=\s*['\"]([^'\"]+)['\"]",
        version_file.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if match is None:
        return "0+unknown"
    return match.group(1)


try:
    release = package_version("easydecon")
except Exception:
    release = _fallback_version()

version = release
language = "en"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autosectionlabel_prefix_document = True

myst_heading_anchors = 3
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

autodoc_mock_imports = [
    "spatialdata",
    "spatialdata_io",
    "spatialdata_plot",
    "pydeseq2",
    "tensorflow",
    "bin2cell",
    "fireducks",
]

intersphinx_mapping = {}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"] if (DOCS_DIR / "_static").exists() else []

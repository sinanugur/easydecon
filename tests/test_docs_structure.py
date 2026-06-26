from pathlib import Path

import easydecon as ed
from easydecon import _validation


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


REQUIRED_GUIDES = {
    "usage.rst",
    "workflow.md",
    "marker_inputs.md",
    "phase1.md",
    "phase2.md",
    "results.md",
    "prepared_markers.md",
    "scanpy_markers.md",
    "reference_markers.md",
    "marker_roles.md",
    "ucell.md",
    "refinement.md",
    "candidate_pruning.md",
    "niches.md",
    "visualization.md",
    "validation.md",
    "segmentation.md",
    "troubleshooting.md",
    "glossary.md",
    "api.rst",
}


TOCTREE_TARGETS = {
    "usage",
    "workflow",
    "marker_inputs",
    "phase1",
    "phase2",
    "results",
    "prepared_markers",
    "scanpy_markers",
    "reference_markers",
    "marker_roles",
    "ucell",
    "refinement",
    "candidate_pruning",
    "niches",
    "visualization",
    "validation",
    "segmentation",
    "troubleshooting",
    "glossary",
    "api",
}


def test_docs_index_exists():
    assert (DOCS / "index.rst").is_file()


def test_all_toctree_targets_exist():
    for target in TOCTREE_TARGETS:
        assert (DOCS / f"{target}.rst").is_file() or (DOCS / f"{target}.md").is_file()


def test_public_api_names_are_documented():
    api = (DOCS / "api.rst").read_text(encoding="utf-8")
    for name in ed.__all__:
        assert name in api


def test_no_private_automodule_dump():
    api = (DOCS / "api.rst").read_text(encoding="utf-8")
    assert ":undoc-members:" not in api
    assert ".. automodule:: easydecon.easydecon" not in api
    assert ".. automodule:: easydecon.extra" not in api
    assert ".. automodule:: easydecon.markers" not in api


def test_required_guides_exist():
    for filename in REQUIRED_GUIDES:
        assert (DOCS / filename).is_file()


def test_docs_do_not_reference_removed_methods():
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in DOCS.iterdir()
        if path.suffix in {".md", ".rst"} and path.name != "README.md"
    )
    for method in _validation.SIMILARITY_METHODS:
        assert method in text
    assert 'method="spearman"' not in (DOCS / "phase2.md").read_text(encoding="utf-8")


def test_root_readme_links_to_documentation():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/index.rst" in readme
    assert "docs/usage.rst" in readme


def test_docs_readme_not_in_main_toctree():
    index = (DOCS / "index.rst").read_text(encoding="utf-8")
    assert "README" not in index

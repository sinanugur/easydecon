import re
from pathlib import Path

from easydecon import _validation


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def _first_python_block(text: str) -> str:
    match = re.search(r"```python\n(.*?)\n```", text, flags=re.DOTALL)
    assert match is not None
    return match.group(1)


def test_root_readme_primary_quickstart_uses_defaults():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    quickstart = _first_python_block(readme)
    assert 'filtering_algorithm="quantile"' not in quickstart
    assert 'method="ucell"' not in quickstart


def test_usage_positions_permutation_as_standard_phase1():
    usage = (DOCS / "usage.rst").read_text(encoding="utf-8").casefold()
    assert "standard phase 1 permutation workflow" in usage
    assert "fast exploratory shortcut" in usage


def test_phase1_orders_and_covers_filtering_methods():
    phase1 = (DOCS / "phase1.md").read_text(encoding="utf-8")
    assert phase1.index("## Permutation filtering") < phase1.index("## Quantile filtering")
    assert phase1.index("## Quantile filtering") < phase1.index("## Negative-binomial filtering")
    for method in _validation.FILTERING_ALGORITHMS:
        assert f'filtering_algorithm="{method}"' in phase1


def test_phase2_covers_and_positions_methods():
    phase2 = (DOCS / "phase2.md").read_text(encoding="utf-8")
    phase2_words = " ".join(phase2.split())
    for method in _validation.SIMILARITY_METHODS:
        assert f"`{method}`" in phase2
    assert "Weighted Jaccard is the default Phase 2 method" in phase2_words
    assert (
        "UCell-like scoring is useful when rank robustness or anti-marker evidence is desired"
        in phase2_words
    )


def test_primary_refinement_and_pruning_examples_do_not_require_ucell():
    candidate = (DOCS / "candidate_pruning.md").read_text(encoding="utf-8")
    refinement = (DOCS / "refinement.md").read_text(encoding="utf-8")
    assert candidate.index('method="wjaccard"') < candidate.find('method="ucell"') or 'method="ucell"' not in candidate
    assert refinement.index('method="wjaccard"') < refinement.index('method="ucell"')


def test_ucell_not_under_advanced_workflows_toctree():
    index = (DOCS / "index.rst").read_text(encoding="utf-8")
    advanced = index.split(":caption: Advanced workflows", maxsplit=1)[1]
    advanced = advanced.split(".. toctree::", maxsplit=1)[0]
    assert "ucell" not in advanced

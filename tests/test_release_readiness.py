import importlib
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def _pyproject():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_version_is_public():
    import easydecon as ed

    assert isinstance(ed.__version__, str)
    assert "__version__" in ed.__all__


def test_pyproject_exists():
    assert (ROOT / "pyproject.toml").exists()


def test_license_file_matches_pyproject_expectation():
    license_file = _pyproject()["project"]["license"].get("file")

    if license_file is not None:
        assert (ROOT / license_file).is_file()


def test_console_scripts_are_importable():
    scripts = _pyproject()["project"].get("scripts", {})

    for target in scripts.values():
        module_name, function_name = target.split(":", maxsplit=1)
        module = importlib.import_module(module_name)
        assert hasattr(module, function_name)
        assert callable(getattr(module, function_name))


def test_changelog_exists():
    assert (ROOT / "CHANGELOG.md").is_file()


def test_readme_has_a_minimal_recommended_workflow():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for expected in (
        "run_easydecon",
        "prepare_markers",
        'marker_role_inference="signed"',
        "EasyDeconResult",
        "https://easydecon.readthedocs.io/en/latest/",
    ):
        assert expected in readme

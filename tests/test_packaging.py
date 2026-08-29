def test_version_import_if_defined():
    import easydecon

    if hasattr(easydecon, "__version__"):
        assert isinstance(easydecon.__version__, str)


def test_core_public_imports_without_optional_runtime():
    import easydecon as ed

    assert callable(ed.run_easydecon)
    assert callable(ed.summarize_easydecon_result)
    assert callable(ed.detect_niches_from_easydecon_result)


def test_optional_spatial_import_is_not_required():
    import easydecon
    import easydecon.easydecon

    assert easydecon is not None

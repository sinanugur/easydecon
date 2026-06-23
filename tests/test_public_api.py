def test_package_imports_public_api():
    import easydecon as ed

    assert hasattr(ed, "run_easydecon")
    assert hasattr(ed, "easydecon_workflow")
    assert hasattr(ed, "read_markers_dataframe")
    assert hasattr(ed, "EasyDeconResult")
    assert hasattr(ed, "PreparedMarkers")
    assert hasattr(ed, "prepare_markers")
    assert hasattr(ed, "select_prepared_markers")
    assert hasattr(ed, "detect_niches_from_easydecon_result")
    assert hasattr(ed, "standardize_marker_dataframe")


def test_run_easydecon_alias():
    import easydecon as ed

    assert ed.run_easydecon is ed.easydecon_workflow


def test_all_contains_public_names():
    import easydecon as ed

    for name in [
        "run_easydecon",
        "read_markers_dataframe",
        "EasyDeconResult",
        "PreparedMarkers",
        "prepare_markers",
        "select_prepared_markers",
    ]:
        assert name in ed.__all__

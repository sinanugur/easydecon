def test_package_imports_public_api():
    import easydecon as ed

    assert hasattr(ed, "run_easydecon")
    assert hasattr(ed, "easydecon_workflow")
    assert hasattr(ed, "read_markers_dataframe")
    assert hasattr(ed, "EasyDeconResult")
    assert hasattr(ed, "PreparedMarkers")
    assert hasattr(ed, "compute_reference_profile_markers")
    assert hasattr(ed, "prepare_markers")
    assert hasattr(ed, "select_prepared_markers")
    assert hasattr(ed, "RefinedGroupResult")
    assert hasattr(ed, "refine_group")
    assert hasattr(ed, "detect_niches_from_easydecon_result")
    assert hasattr(ed, "standardize_marker_dataframe")
    assert hasattr(ed, "UCELL_MARKER_ROLES")
    assert "positive" in ed.UCELL_MARKER_ROLES
    assert hasattr(ed, "MARKER_ROLE_MODES")
    assert "phase_specific" in ed.MARKER_ROLE_MODES


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
        "compute_reference_profile_markers",
        "prepare_markers",
        "select_prepared_markers",
        "RefinedGroupResult",
        "refine_group",
        "UCELL_MARKER_ROLES",
        "MARKER_ROLE_MODES",
    ]:
        assert name in ed.__all__

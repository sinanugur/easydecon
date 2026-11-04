import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import easydecon.bulk_niche as bn


def _table(*, sparse_x=False):
    genes = ["A1", "A2", "A3", "B1", "B2", "B3", "BG"]
    x = np.array(
        [
            [8, 7, 6, 0, 0, 0, 1],
            [7, 6, 5, 0, 0, 0, 1],
            [0, 0, 0, 8, 7, 6, 1],
            [0, 0, 0, 7, 6, 5, 1],
            [8, 7, 6, 0, 0, 0, 1],
            [7, 6, 5, 0, 0, 0, 1],
            [0, 0, 0, 8, 7, 6, 1],
            [0, 0, 0, 7, 6, 5, 1],
        ],
        dtype=float,
    )
    obs = pd.DataFrame(
        {
            "sample": ["S1"] * 4 + ["S2"] * 4,
            "patient": ["P1"] * 4 + ["P2"] * 4,
            "condition": ["normal"] * 4 + ["crc"] * 4,
        },
        index=[f"spot{i}" for i in range(8)],
    )
    table = ad.AnnData(
        X=sparse.csr_matrix(x) if sparse_x else x,
        obs=obs,
        var=pd.DataFrame(index=genes),
    )
    table.obsm["spatial"] = np.vstack(
        [np.column_stack([np.arange(4), np.zeros(4)])] * 2
    ).astype(float)
    return table


def _niches():
    return pd.DataFrame(
        {"niche": ["A", "A", "B", "B", "A", "A", "B", "B"]},
        index=[f"spot{i}" for i in range(8)],
    )


def _evidence():
    return pd.DataFrame(
        {
            "T": [0.9, 0.8, 0.1, 0.2, 0.85, 0.75, 0.15, 0.25],
            "U": [0.1, 0.2, 0.9, 0.8, 0.15, 0.25, 0.85, 0.75],
        },
        index=[f"spot{i}" for i in range(8)],
    )


def _markers():
    return pd.DataFrame(
        {
            "group": ["T", "T", "T", "T", "U", "U", "U"],
            "names": ["A1", "A2", "A3", "B1", "B1", "B2", "B3"],
            "marker_rank": [1, 2, 3, 4, 1, 2, 3],
            "marker_role": ["positive", "identity", "presence", "negative", "positive", "positive", "positive"],
        }
    )


def _bulk():
    return pd.DataFrame(
        {
            "gene": ["A1", "A2", "A3", "BG", "B3", "B2", "B1"],
            "log2FoldChange": [3, 2, 1, 0, -1, -2, -3],
            "stat": [6, 5, 4, 0, -4, -5, -6],
            "pvalue": [0.001, 0.002, 0.003, 1, 0.003, 0.002, 0.001],
            "padj": [0.01, 0.02, 0.03, np.nan, 0.03, 0.02, 0.01],
            "baseMean": [100] * 7,
        }
    )


@pytest.fixture(autouse=True)
def mock_gseapy(monkeypatch):
    calls = []

    def fake(ranking, gene_sets, **kwargs):
        calls.append(gene_sets)
        rows = []
        for term, genes in gene_sets.items():
            is_a = any(gene.startswith("A") for gene in genes)
            rows.append(
                {
                    "Term": term,
                    "ES": 0.8 if is_a else -0.8,
                    "NES": 2.0 if is_a else -2.0,
                    "NOM p-val": 0.01 if is_a else 0.04,
                    "FDR q-val": 0.02 if is_a else 0.08,
                    "Lead_genes": ";".join(genes),
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr(bn, "_run_gseapy_prerank", fake)
    return calls


def _analyze(table=None, **kwargs):
    return bn.analyze_bulk_niches(
        table if table is not None else _table(),
        _bulk(),
        _niches(),
        contrast_name="tumor_vs_normal",
        smoothed_evidence=_evidence(),
        markers=_markers(),
        sample_column="sample",
        patient_column="patient",
        condition_column="condition",
        shared_niche_labels=True,
        positive_condition="tumor",
        negative_condition="normal",
        ranking_mode="wald_stat",
        stat_column="stat",
        statistic_type="signed_wald",
        min_signature_size=2,
        scoring_min_genes=2,
        **kwargs,
    )


def test_generated_signatures_are_deterministic_bulk_independent_and_sample_balanced():
    first = _analyze()
    reversed_bulk = _bulk().copy()
    reversed_bulk["stat"] *= -1
    second = bn.analyze_bulk_niches(
        _table(), reversed_bulk, _niches(), contrast_name="reverse",
        smoothed_evidence=_evidence(), markers=_markers(), sample_column="sample",
        shared_niche_labels=True, ranking_mode="wald_stat", stat_column="stat",
        statistic_type="signed_wald", min_signature_size=2, scoring_min_genes=2,
    )
    pd.testing.assert_frame_equal(first.niche_signatures, second.niche_signatures)
    assert first.niche_composition.loc["A", "T"] == pytest.approx(0.825)
    assert "B1" not in first.niche_signatures.query("niche == 'A'")["gene"].tolist()


def test_marker_membership_requires_and_deterministically_ranks_positive_markers():
    signed = pd.DataFrame(
        {
            "group": ["T", "T", "T"],
            "names": ["negative_gene", "positive_low", "positive_high"],
            "logfoldchanges": [-4.0, 2.0, 4.0],
        }
    )
    first, _ = bn._marker_membership(signed, ["T"])
    second, _ = bn._marker_membership(signed.iloc[::-1], ["T"])
    assert first["gene"].tolist() == ["positive_high", "positive_low"]
    pd.testing.assert_frame_equal(first, second)

    explicit = signed.assign(marker_role=["negative", "positive", "identity"])
    membership, _ = bn._marker_membership(explicit, ["T"])
    assert set(membership["gene"]) == {"positive_low", "positive_high"}

    with pytest.raises(ValueError, match="signed fold-change.*marker_role"):
        bn._marker_membership(
            pd.DataFrame({"group": ["T"], "names": ["ambiguous"]}), ["T"]
        )


def test_single_niche_uses_labeled_composition_only_fallback():
    niches = _niches()
    niches["niche"] = "A"
    result = bn.analyze_bulk_niches(
        _table(), _bulk(), niches, contrast_name="single",
        smoothed_evidence=_evidence(), markers=_markers(), sample_column="sample",
        shared_niche_labels=True, min_signature_size=2, scoring_min_genes=2,
    )
    assert set(result.niche_signatures["construction"]) == {"composition_only_single_niche"}
    interface = bn.analyze_niche_interface(
        result, _table(), niche_a="A", niche_b="B", side="A", program_niche="A",
        radius=1.1, near_distance=0, interior_min_distance=1,
    )
    assert set(interface.specimen_effects["status"]) == {"no_interface"}
    assert interface.binned_summary.empty
    figure, _ = bn.plot_niche_interface(interface)
    plt.close(figure)


def test_enrichment_mapping_direction_and_multiple_testing_semantics(mock_gseapy):
    result = _analyze()
    enrichment = result.enrichment.set_index("niche")
    assert enrichment.loc["A", "bulk_direction"] == "positive"
    assert enrichment.loc["B", "bulk_direction"] == "negative"
    assert enrichment.loc["A", "direction_label"] == "toward_tumor"
    assert enrichment.loc["B", "direction_label"] == "toward_normal"
    assert enrichment.loc["A", "bh_nominal_qvalue_across_niches"] == pytest.approx(0.02)
    assert enrichment.loc["B", "bh_nominal_qvalue_across_niches"] == pytest.approx(0.04)
    assert result.parameters["multiple_testing"]["gseapy_fdr_qvalue"].startswith("backend-native")
    assert len(mock_gseapy) == 1


def test_explicit_identical_overlapping_and_empty_signatures_compute_once(mock_gseapy):
    signatures = {"A": ["A1", "A2"], "B": ["A1", "A2"]}
    result = bn.analyze_bulk_niches(
        _table(), _bulk(), _niches(), contrast_name="contrast",
        niche_signatures=signatures, sample_column="sample", shared_niche_labels=True,
        min_signature_size=2, scoring_min_genes=2,
    )
    assert len(mock_gseapy[0]) == 1
    assert result.diagnostics["n_identical_signature_pairs"] == 1
    assert result.enrichment["bh_nominal_qvalue_across_niches"].tolist() == pytest.approx([0.01, 0.01])

    empty = bn.analyze_bulk_niches(
        _table(), _bulk(), _niches(), contrast_name="contrast",
        niche_signatures={"A": ["NOT_MEASURED"], "B": ["B1", "B2"]},
        sample_column="sample", shared_niche_labels=True,
        min_signature_size=2, scoring_min_genes=2,
    )
    assert empty.enrichment.set_index("niche").loc["A", "test_status"] == "no_overlap"
    undersized = bn.analyze_bulk_niches(
        _table(), _bulk(), _niches(), contrast_name="contrast",
        niche_signatures={"A": ["A1"], "B": ["B1", "B2"]},
        sample_column="sample", shared_niche_labels=True,
        min_signature_size=2, scoring_min_genes=2,
    )
    assert undersized.enrichment.set_index("niche").loc["A", "test_status"] == "insufficient_overlap"


def test_full_background_signatures_are_not_submitted(mock_gseapy):
    all_genes = _bulk()["gene"].tolist()
    only_full = bn.analyze_bulk_niches(
        _table(), _bulk(), _niches(), contrast_name="full",
        niche_signatures={"A": all_genes, "B": all_genes},
        sample_column="sample", shared_niche_labels=True,
        min_signature_size=2, max_signature_size=20,
    )
    assert not mock_gseapy
    assert set(only_full.enrichment["test_status"]) == {"full_background_untestable"}
    assert only_full.enrichment["nes"].isna().all()

    mixed = bn.analyze_bulk_niches(
        _table(), _bulk(), _niches(), contrast_name="mixed",
        niche_signatures={"A": all_genes, "B": ["B1", "B2"]},
        sample_column="sample", shared_niche_labels=True,
        min_signature_size=2, max_signature_size=20,
    )
    assert len(mock_gseapy) == 1
    statuses = mixed.enrichment.set_index("niche")["test_status"]
    assert statuses.loc["A"] == "full_background_untestable"
    assert statuses.loc["B"] == "tested"


def test_bulk_validation_alignment_and_shared_label_confirmation():
    with pytest.raises(ValueError, match="signed_wald"):
        bn.analyze_bulk_niches(
            _table(), _bulk(), _niches(), contrast_name="x",
            niche_signatures={"A": ["A1", "A2"], "B": ["B1", "B2"]},
            sample_column="sample", shared_niche_labels=True,
            ranking_mode="wald_stat", stat_column="stat", statistic_type="lrt",
            min_signature_size=2,
        )
    with pytest.raises(ValueError, match="shared_niche_labels=True"):
        bn.analyze_bulk_niches(
            _table(), _bulk(), _niches(), contrast_name="x",
            niche_signatures={"A": ["A1", "A2"], "B": ["B1", "B2"]},
            sample_column="sample", min_signature_size=2,
        )
    bad_evidence = _evidence().drop(index="spot0")
    with pytest.raises(ValueError, match="exactly the same observations"):
        bn.analyze_bulk_niches(
            _table(), _bulk(), _niches(), contrast_name="x",
            smoothed_evidence=bad_evidence, markers=_markers(), sample_column="sample",
            shared_niche_labels=True, min_signature_size=2,
        )
    duplicated = _bulk()
    duplicated.loc[1, "gene"] = "A1"
    with pytest.raises(ValueError, match="unique"):
        bn.analyze_bulk_niches(
            _table(), duplicated, _niches(), contrast_name="x",
            niche_signatures={"A": ["A1", "A2"], "B": ["B1", "B2"]},
            sample_column="sample", shared_niche_labels=True, min_signature_size=2,
        )


def test_sparse_dense_scores_zero_handling_fixed_settings_and_no_mutation():
    dense_table = _table()
    before_obs = dense_table.obs.copy()
    before_x = dense_table.X.copy()
    dense = _analyze(dense_table)
    sparse_result = _analyze(_table(sparse_x=True))
    pd.testing.assert_frame_equal(dense.spatial_scores, sparse_result.spatial_scores)
    pd.testing.assert_frame_equal(dense_table.obs, before_obs)
    np.testing.assert_array_equal(dense_table.X, before_x)
    a_scores = dense.spatial_scores.query("program_niche == 'A'").set_index("observation_id")
    assert a_scores.loc["spot0", "score"] == a_scores.loc["spot4", "score"]

    zero = _table()
    zero.X[0] = 0
    zero_result = _analyze(zero)
    zero_scores = zero_result.spatial_scores.query("observation_id == 'spot0'")
    assert zero_scores["score"].isna().all()
    assert zero_scores["all_zero"].all()


def test_one_enrichment_result_is_reused_across_specimens():
    result = _analyze()
    assert "specimen" not in result.enrichment
    assert set(result.spatial_scores["specimen"]) == {"S1", "S2"}
    assert set(result.coverage_qc["specimen"]) == {"S1", "S2"}


def test_known_interface_distances_no_cross_sample_edges_and_patient_retention():
    result = _analyze()
    interface = bn.analyze_niche_interface(
        result,
        _table(),
        niche_a="A", niche_b="B", side="A", program_niche="A",
        radius=1.1, near_distance=0, interior_min_distance=1,
    )
    distances = interface.observations.set_index("observation_id")["distance"]
    assert distances.loc["spot1"] == 0
    assert distances.loc["spot0"] == 1
    assert distances.loc["spot5"] == 0
    assert distances.loc["spot4"] == 1
    assert interface.diagnostics["boundary_nodes_by_specimen"] == {"S1": 1, "S2": 1}
    assert set(interface.specimen_effects["patient"]) == {"P1", "P2"}
    effects = interface.specimen_effects.set_index("specimen")
    scores = result.spatial_scores.query("program_niche == 'A'").set_index("observation_id")["score"]
    assert effects.loc["S1", "near_minus_interior"] == pytest.approx(scores["spot1"] - scores["spot0"])
    assert effects.loc["S1", "status"] == "available"


@pytest.mark.parametrize("near, interior", [(1, 1), (2, 1)])
def test_interface_requires_disjoint_distance_bands(near, interior):
    with pytest.raises(ValueError, match="near_distance must be less"):
        bn.analyze_niche_interface(
            _analyze(), _table(), niche_a="A", niche_b="B", side="A",
            program_niche="A", radius=1.1,
            near_distance=near, interior_min_distance=interior,
        )


def test_interface_rejects_identical_niches_and_unknown_program():
    result = _analyze()
    with pytest.raises(ValueError, match="must be different"):
        bn.analyze_niche_interface(
            result, _table(), niche_a="A", niche_b="A", side="A",
            program_niche="A", radius=1.1, near_distance=0, interior_min_distance=1,
        )
    with pytest.raises(ValueError, match="Unknown program_niche"):
        bn.analyze_niche_interface(
            result, _table(), niche_a="A", niche_b="B", side="A",
            program_niche="missing", radius=1.1, near_distance=0, interior_min_distance=1,
        )


def test_missing_interface_and_disconnected_nodes_are_unavailable():
    table = _table()
    niches = _niches()
    niches.loc[niches.index[:4], "niche"] = "A"
    result = bn.analyze_bulk_niches(
        table, _bulk(), niches, contrast_name="x",
        niche_signatures={"A": ["A1", "A2"], "B": ["B1", "B2"]},
        sample_column="sample", patient_column="patient", shared_niche_labels=True,
        min_signature_size=2, scoring_min_genes=2,
    )
    interface = bn.analyze_niche_interface(
        result, table, niche_a="A", niche_b="B", side="A", program_niche="A",
        radius=1.1, near_distance=0, interior_min_distance=1,
    )
    s1 = interface.observations.query("specimen == 'S1'")
    assert set(s1["status"]) == {"no_interface"}
    assert interface.specimen_effects.set_index("specimen").loc["S1", "status"] == "no_interface"
    assert list(interface.binned_summary.columns) == [
        "specimen", "distance_bin", "distance_midpoint", "n_observations",
        "mean_score", "median_score",
    ]

    disconnected_table = _table()
    disconnected_table.obsm["spatial"][0] = [100, 100]
    disconnected = _analyze(disconnected_table)
    disconnected_interface = bn.analyze_niche_interface(
        disconnected, disconnected_table, niche_a="A", niche_b="B", side="A",
        program_niche="A", radius=1.1, near_distance=0, interior_min_distance=1,
    )
    assert disconnected_interface.observations.set_index("observation_id").loc["spot0", "status"] == "unreachable"


def test_interface_statuses_for_absent_side_and_unavailable_scores():
    table = _table()
    niches = _niches()
    niches.loc[niches.index[:4], "niche"] = "B"
    result = bn.analyze_bulk_niches(
        table, _bulk(), niches, contrast_name="x",
        niche_signatures={"A": ["A1", "A2"], "B": ["B1", "B2"]},
        sample_column="sample", shared_niche_labels=True,
        min_signature_size=2, scoring_min_genes=2,
    )
    absent = bn.analyze_niche_interface(
        result, table, niche_a="A", niche_b="B", side="A", program_niche="A",
        radius=1.1, near_distance=0, interior_min_distance=1,
    )
    assert absent.specimen_effects.set_index("specimen").loc["S1", "status"] == "no_side_observations"

    available_result = _analyze()
    available_result.spatial_scores.loc[
        available_result.spatial_scores["program_niche"] == "A", "score"
    ] = np.nan
    unavailable = bn.analyze_niche_interface(
        available_result, _table(), niche_a="A", niche_b="B", side="A",
        program_niche="A", radius=1.1, near_distance=0, interior_min_distance=1,
    )
    assert set(unavailable.specimen_effects["status"]) == {"unavailable_scores"}
    assert unavailable.binned_summary.empty
    figure, _ = bn.plot_niche_interface(unavailable)
    plt.close(figure)

    mixed_result = _analyze()
    mixed_result.spatial_scores.loc[
        (mixed_result.spatial_scores["program_niche"] == "A")
        & (mixed_result.spatial_scores["specimen"] == "S1"),
        "score",
    ] = np.nan
    mixed = bn.analyze_niche_interface(
        mixed_result, _table(), niche_a="A", niche_b="B", side="A",
        program_niche="A", radius=1.1, near_distance=0, interior_min_distance=1,
    )
    assert mixed.specimen_effects.set_index("specimen")["status"].to_dict() == {
        "S1": "unavailable_scores", "S2": "available",
    }


def test_storage_collision_and_plot_smoke_tests():
    table = _table()
    result = _analyze(table, store=True)
    assert "easydecon_bulk_niche_tumor_vs_normal" in table.uns
    before = table.obs.copy()
    with pytest.raises(ValueError, match="overwrite"):
        _analyze(table, store=True)
    pd.testing.assert_frame_equal(table.obs, before)

    interface = bn.analyze_niche_interface(
        result, table, niche_a="A", niche_b="B", side="A", program_niche="A",
        radius=1.1, near_distance=0, interior_min_distance=1,
    )
    figures = [
        bn.plot_niche_bulk_associations(result, table)[0],
        bn.plot_niche_supporting_program(result, table, program_niche="A")[0],
        bn.plot_bulk_niche_evidence(result)[0],
        bn.plot_niche_interface(interface)[0],
    ]
    assert all(figure is not None for figure in figures)
    for figure in figures:
        plt.close(figure)


def test_h5ad_storage_roundtrip_with_overlap_and_unavailable_result(tmp_path):
    table = _table()
    all_genes = _bulk()["gene"].tolist()
    result = bn.analyze_bulk_niches(
        table, _bulk(), _niches(), contrast_name="roundtrip",
        niche_signatures={"A": all_genes, "B": ["B1", "B2"]},
        sample_column="sample", patient_column="patient", shared_niche_labels=True,
        min_signature_size=2, max_signature_size=20, scoring_min_genes=2,
        store=True,
    )
    path = tmp_path / "bulk_niche.h5ad"
    table.write_h5ad(path)
    restored = ad.read_h5ad(path)
    stored = restored.uns["easydecon_bulk_niche_roundtrip"]
    assert isinstance(stored["enrichment"], pd.DataFrame)
    assert set(stored["enrichment"]["test_status"].astype(str)) == {
        "full_background_untestable", "tested",
    }
    assert isinstance(stored["diagnostics"]["signature_overlap"], pd.DataFrame)
    assert stored["parameters"]["contrast_name"] == "roundtrip"
    assert "easydecon_bulk_niche_roundtrip_b_program_score" in restored.obs
    with pytest.raises(ValueError, match="overwrite"):
        bn._store_result(table, result, overwrite=False)


def test_interface_plot_shared_and_explicit_limits_with_missing_scores():
    result = _analyze()
    result.spatial_scores.loc[
        (result.spatial_scores["program_niche"] == "A")
        & (result.spatial_scores["observation_id"] == "spot1"),
        "score",
    ] = np.nan
    interface = bn.analyze_niche_interface(
        result, _table(), niche_a="A", niche_b="B", side="A", program_niche="A",
        radius=1.1, near_distance=0, interior_min_distance=1,
    )
    figure, axes = bn.plot_niche_interface(interface)
    colored = [axis.collections[1] for axis in axes[:2]]
    assert {(collection.norm.vmin, collection.norm.vmax) for collection in colored} == {
        (colored[0].norm.vmin, colored[0].norm.vmax)
    }
    assert len(axes[0].collections[0].get_offsets()) == 1
    plt.close(figure)

    figure, axes = bn.plot_niche_interface(interface, limits=(0.2, 0.8))
    for axis in axes[:2]:
        assert (axis.collections[1].norm.vmin, axis.collections[1].norm.vmax) == (0.2, 0.8)
    plt.close(figure)

    interface.observations["score"] = 0.5
    figure, _ = bn.plot_niche_interface(interface)
    plt.close(figure)
    interface.observations["score"] = 0.0
    figure, _ = bn.plot_niche_interface(interface)
    plt.close(figure)


def test_actual_gseapy_prerank_direction_if_installed(monkeypatch):
    pytest.importorskip("gseapy")
    monkeypatch.undo()
    ranking, _ = bn._prepare_bulk_ranking(
        _bulk(), gene_column="gene", gene_from_index=False,
        lfc_column="log2FoldChange", ranking_mode="wald_stat",
        stat_column="stat", statistic_type="signed_wald",
        pvalue_column="pvalue", padj_column="padj",
        base_mean_column="baseMean", min_base_mean=None, gene_universe=None,
    )
    output = bn._run_gseapy_prerank(
        ranking, {"positive": ["A1", "A2", "A3"], "negative": ["B1", "B2", "B3"]},
        min_size=2, max_size=5, permutations=50, seed=0,
    ).set_index("Term")
    assert float(output.loc["positive", "NES"]) > 0
    assert float(output.loc["negative", "NES"]) < 0

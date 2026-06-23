from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from examples import visualize_results
from easydecon.extra import EasyDeconResult


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def visualization_outputs():
    outputs = visualize_results.main(
        show=False,
        return_outputs=True,
        n_spots=30,
        n_genes=20,
        n_celltypes=3,
    )
    try:
        yield outputs
    finally:
        for figure in outputs[1:]:
            plt.close(figure)


def test_visualize_results_runs(visualization_outputs):
    result, assignment_figure, posterior_figure, counts_figure = visualization_outputs

    assert isinstance(result, EasyDeconResult)
    assert isinstance(assignment_figure, Figure)
    assert isinstance(posterior_figure, Figure)
    assert isinstance(counts_figure, Figure)


def test_plot_assignments_accepts_unassigned_locations(visualization_outputs):
    result = visualization_outputs[0]
    sdata, _ = visualize_results.make_synthetic_spatial_and_markers(
        n_spots=30,
        n_genes=20,
        n_celltypes=3,
    )
    assignment_column = result.diagnostics.get(
        "results_column",
        result.assigned_labels.columns[0],
    )
    result.assigned_labels.loc[result.assigned_labels.index[0], assignment_column] = np.nan

    figure, axis = visualize_results.plot_assignments(sdata, result)

    assert isinstance(figure, Figure)
    assert axis.get_title() == "easydecon assignments"
    plt.close(figure)


def test_plot_celltype_posterior_requires_existing_celltype(visualization_outputs):
    result = visualization_outputs[0]
    sdata, _ = visualize_results.make_synthetic_spatial_and_markers(
        n_spots=30,
        n_genes=20,
        n_celltypes=3,
    )

    with pytest.raises(ValueError, match="Available cell types"):
        visualize_results.plot_celltype_posterior(
            sdata,
            result,
            "not_a_cell_type",
        )


def test_visualization_docs_exist():
    assert (ROOT / "docs" / "results.md").is_file()
    assert (ROOT / "docs" / "visualization.md").is_file()


def test_visualization_docs_mention_core_outputs():
    text = (ROOT / "docs" / "visualization.md").read_text(encoding="utf-8")

    for expected in (
        "posterior_df",
        "priors_df",
        "assigned_labels",
        "plot_niche_compositions",
    ):
        assert expected in text

import pytest

from easydecon._validation import (
    format_allowed_values,
    validate_choice,
    validate_positive,
    validate_probability_range,
)
from easydecon.easydecon import (
    common_markers_gene_expression_and_filter,
    get_clusters_by_similarity_on_tissue,
)
from easydecon.extra import easydecon_workflow


def test_format_allowed_values_sorted():
    assert format_allowed_values({"b", "a"}) == "'a', 'b'"


def test_validate_choice_accepts_valid_value():
    assert validate_choice("x", {"x"}, "thing") == "x"


def test_validate_choice_rejects_invalid_value():
    with pytest.raises(ValueError) as exc_info:
        validate_choice("bad", {"good"}, "thing")

    message = str(exc_info.value)
    assert "thing must be one of" in message
    assert "Got 'bad'" in message


@pytest.mark.parametrize("value", [0, 1])
def test_validate_probability_range_inclusive_accepts_endpoints(value):
    assert validate_probability_range(value, "probability") == value


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_validate_probability_range_inclusive_rejects_outside(value):
    with pytest.raises(ValueError, match="probability must be between 0 and 1"):
        validate_probability_range(value, "probability")


@pytest.mark.parametrize(
    ("value", "kwargs"),
    [
        (0, {"inclusive_min": False}),
        (1, {"inclusive_max": False}),
    ],
)
def test_validate_probability_range_exclusive(value, kwargs):
    with pytest.raises(ValueError):
        validate_probability_range(value, "probability", **kwargs)


def test_validate_positive():
    assert validate_positive(0.1, "value") == 0.1
    with pytest.raises(ValueError):
        validate_positive(0, "value")
    assert validate_positive(0, "value", allow_zero=True) == 0
    with pytest.raises(ValueError):
        validate_positive(-1, "value", allow_zero=True)


def test_common_markers_invalid_filtering_algorithm_message():
    with pytest.raises(ValueError) as exc_info:
        common_markers_gene_expression_and_filter(
            None,
            [],
            filtering_algorithm="invalid",
        )

    message = str(exc_info.value)
    assert "filtering_algorithm must be one of" in message
    assert "'nb'" in message
    assert "'permutation'" in message
    assert "'quantile'" in message


def test_get_clusters_invalid_similarity_method_message():
    with pytest.raises(ValueError) as exc_info:
        get_clusters_by_similarity_on_tissue(
            None,
            None,
            method="invalid",
        )

    message = str(exc_info.value)
    assert "method must be one of" in message
    assert "'auc'" in message
    assert "'wjaccard'" in message


def test_workflow_invalid_evidence_to_likelihood_message():
    with pytest.raises(ValueError) as exc_info:
        easydecon_workflow(None, evidence_to_likelihood="invalid")

    message = str(exc_info.value)
    assert "evidence_to_likelihood must be one of" in message
    assert "'row_normalize'" in message
    assert "'softmax'" in message

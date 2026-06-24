"""Shared allowed values and lightweight validation helpers."""

from numbers import Real


MARKER_METHODS = frozenset(
    {
        "auto",
        "existing",
        "scanpy",
        "pydeseq2",
        "deseq2",
        "pseudobulk_deseq2",
        "reference",
        "rctd_like",
    }
)
PYDESEQ2_MARKER_METHODS = frozenset(
    {"pydeseq2", "deseq2", "pseudobulk_deseq2"}
)
REFERENCE_MARKER_METHODS = frozenset({"reference", "rctd_like"})
REFERENCE_CONTRASTS = frozenset({"mean_other", "max_other"})
FILTERING_ALGORITHMS = frozenset({"permutation", "quantile", "nb"})
PHASE1_OUTPUT_STATS = frozenset({"expression", "minus_log10_p"})
AGGREGATION_METHODS = frozenset({"sum", "mean", "median", "cs"})
SIMILARITY_METHODS = frozenset(
    {
        "correlation",
        "cosine",
        "jaccard",
        "overlap",
        "wjaccard",
        "diagnostic",
        "sum",
        "mean",
        "median",
        "euclidean",
        "auc",
        "ucell",
    }
)
EVIDENCE_TO_LIKELIHOOD_METHODS = frozenset({"row_normalize", "softmax"})
ASSIGN_METHODS = frozenset({"max", "zmax", "hybrid"})
UCELL_MARKER_ROLES = frozenset({"positive", "negative", "presence", "identity"})
MARKER_ROLE_MODES = frozenset({"shared", "phase_specific"})


def format_allowed_values(values) -> str:
    """Format allowed values deterministically for user-facing messages."""
    return ", ".join(repr(value) for value in sorted(values))


def validate_choice(value, allowed, name: str):
    """Return *value* when it belongs to *allowed*, otherwise raise."""
    if value not in allowed:
        raise ValueError(
            f"{name} must be one of: {format_allowed_values(allowed)}. "
            f"Got {value!r}."
        )
    return value


def _format_interval(minimum, maximum, inclusive_min, inclusive_max):
    left = "[" if inclusive_min else "("
    right = "]" if inclusive_max else ")"
    return f"{left}{minimum:g}, {maximum:g}{right}"


def validate_probability_range(
    value,
    name: str,
    *,
    inclusive_min=True,
    inclusive_max=True,
    _minimum=0.0,
    _maximum=1.0,
):
    """Validate a numeric probability-like value against bounded endpoints."""
    message = (
        f"{name} must be between 0 and 1."
        if _minimum == 0 and _maximum == 1 and inclusive_min and inclusive_max
        else f"{name} must be in "
        f"{_format_interval(_minimum, _maximum, inclusive_min, inclusive_max)}."
    )
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(message)
    minimum_ok = value >= _minimum if inclusive_min else value > _minimum
    maximum_ok = value <= _maximum if inclusive_max else value < _maximum
    if not minimum_ok or not maximum_ok:
        raise ValueError(message)
    return value


def validate_positive(value, name: str, *, allow_zero=False):
    """Validate that a numeric value is positive (or non-negative)."""
    message = (
        f"{name} must be greater than or equal to 0."
        if allow_zero
        else f"{name} must be greater than 0."
    )
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(message)
    if (allow_zero and value < 0) or (not allow_zero and value <= 0):
        raise ValueError(message)
    return value


__all__ = [
    "MARKER_METHODS",
    "PYDESEQ2_MARKER_METHODS",
    "REFERENCE_MARKER_METHODS",
    "REFERENCE_CONTRASTS",
    "FILTERING_ALGORITHMS",
    "PHASE1_OUTPUT_STATS",
    "AGGREGATION_METHODS",
    "SIMILARITY_METHODS",
    "EVIDENCE_TO_LIKELIHOOD_METHODS",
    "ASSIGN_METHODS",
    "UCELL_MARKER_ROLES",
    "MARKER_ROLE_MODES",
    "format_allowed_values",
    "validate_choice",
    "validate_probability_range",
    "validate_positive",
]

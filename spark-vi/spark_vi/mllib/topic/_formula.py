"""Formula-handling plumbing for the StreamingSTM MLlib shim (Path B).

Uses `formulaic` to parse R-style formula strings and produce a ModelSpec
that maps doc covariates to design matrix rows. Rejects "stateful transforms"
(splines, standardization) that v1 explicitly does not support
(see ADR 0022, ADR 0024).

Categorical level discovery uses the schema-frame trick (ADR 0024):
Spark-native distinct + cardinality bound, build a tiny pandas
schema-frame, hand to formulaic for level capture. The fitted ModelSpec
is data-independent at application time.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Imported lazily so the spark-vi base install doesn't require formulaic.
def _formulaic():
    try:
        import formulaic
        return formulaic
    except ImportError as e:
        raise ImportError(
            "Formula path requires the optional 'formula' extra: "
            "pip install spark-vi[formula]"
        ) from e


# v1 rejection list — these are formulaic's built-in stateful transforms.
# Adding more here is forward-compatible: anything that learns state from
# the schema-frame's bogus placeholder values must be reject-listed.
_STATEFUL_REJECT_LIST = {
    "bs", "ns", "cr", "scale", "center", "standardize",
}


def _iter_terms(formula: Any):
    """Yield all Term objects from a formulaic Formula, regardless of structure.

    Handles both SimpleFormula (one-sided, iterable directly) and
    StructuredFormula (two-sided, has .lhs / .rhs).
    """
    if hasattr(formula, "lhs") and hasattr(formula, "rhs"):
        # Two-sided StructuredFormula
        for term in formula.lhs:
            yield term
        for term in formula.rhs:
            yield term
    else:
        # One-sided SimpleFormula — iterable list of terms
        for term in formula:
            yield term


def validate_formula(formula_str: str) -> None:
    """Parse the formula and reject any stateful transform we don't support in v1.

    Raises ValueError with a clear remediation message if a rejected
    transform appears. See ADR 0022 v1 scope decision.
    """
    formulaic = _formulaic()
    formula = formulaic.Formula(formula_str)

    # Walk the parsed term tree and collect callable names that appear
    # in factor expressions.
    found = set()
    for term in _iter_terms(formula):
        for factor in term.factors:
            expr = str(factor.expr)
            for name in _STATEFUL_REJECT_LIST:
                if f"{name}(" in expr:
                    found.add(name)

    if found:
        rejected = ", ".join(sorted(found))
        raise ValueError(
            f"STM v1 does not support stateful formula transforms: {rejected}. "
            f"Workarounds: bin continuous covariates categorically (e.g. "
            f"`age_decile`), or pre-compute the basis columns yourself and "
            f"pass them as raw continuous covariates. See ADR 0022."
        )

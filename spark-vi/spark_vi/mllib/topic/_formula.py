"""Formula-handling plumbing for the STM MLlib shim's formula path.

Uses `formulaic` to parse R-style formula strings and produce a ModelSpec
that maps doc covariates to design matrix rows. Rejects "stateful transforms"
(splines, standardization) that v1 explicitly does not support
(see ADR 0022, ADR 0024).

Categorical level discovery uses the schema-frame trick (ADR 0024):
Spark-native distinct + cardinality bound, build a tiny pandas
schema-frame, hand to formulaic for level capture. The fitted ModelSpec
is data-independent at application time.

Persistence (ADR 0056): a ModelSpec is never pickled into a saved model. The spec
is fitted on the schema-frame alone (every categorical level, placeholders for the
continuous columns), so the formula plus that frame IS its whole learned state:
``covariate_spec_state`` writes both as JSON and ``model_spec_from_state`` refits
the formula on the stored frame, which reproduces the spec, and its encoding of any
row, exactly.
"""
from __future__ import annotations

import re
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


def fit_model_spec(
    formula: str,
    covariate_pdf: pd.DataFrame,
) -> tuple[Any, list[str]]:
    """Build a formulaic ModelSpec from a covariate DataFrame.

    Used directly when the caller has a small pandas DataFrame in hand
    (a schema-frame, or a pre-collected covariate frame for building a covariate
    sidecar). For large Spark DataFrames, use fit_model_spec_from_spark
    which builds the schema-frame from per-column distinct() queries.
    """
    formulaic = _formulaic()
    validate_formula(formula)

    materializer = formulaic.Formula(formula).get_model_matrix(covariate_pdf)
    spec = materializer.model_spec
    names = list(materializer.columns)
    return spec, names


def apply_model_spec(spec: Any, covariate_pdf: pd.DataFrame) -> np.ndarray:
    """Apply a fitted ModelSpec to a new DataFrame; return the design matrix.

    Raises ValueError if the DataFrame contains categorical levels not seen
    during fit — formulaic silently maps unknown levels to the reference
    category, which would produce silent mis-specification downstream.
    """
    # Pre-check: for each categorical factor in encoder_state, ensure all
    # values in covariate_pdf are within the fitted level set.
    encoder_state = getattr(spec, "encoder_state", {}) or {}
    for factor_key, state_tuple in encoder_state.items():
        if not (isinstance(state_tuple, tuple) and len(state_tuple) == 2):
            continue
        kind, state_dict = state_tuple
        if not (hasattr(kind, "value") and kind.value == "categorical"):
            continue
        categories = state_dict.get("categories") if isinstance(state_dict, dict) else None
        if categories is None:
            continue
        # Derive column name: extract first argument from C(...) wrapper
        col = factor_key
        m = re.match(r"^C\(\s*([^,)\s]+)", col)
        if m:
            col = m.group(1)
        if col not in covariate_pdf.columns:
            continue
        known = set(categories)
        seen = set(covariate_pdf[col].dropna().unique())
        unseen = seen - known
        if unseen:
            raise ValueError(
                f"Column '{col}' contains levels not seen during ModelSpec fit: "
                f"{sorted(unseen)}. Re-fit the ModelSpec on data that covers all "
                f"levels, or drop/remap the unseen categories."
            )

    materialized = spec.get_model_matrix(covariate_pdf)
    return np.asarray(materialized, dtype=np.float64)


def discover_categorical_levels_spark(
    spark_df: Any,
    categorical_cols: list[str],
    max_levels: int,
) -> dict[str, list]:
    """Spark-native level discovery for each categorical column.

    Bounds cardinality via approxCountDistinct first; raises if over max_levels.
    Returns levels sorted lexicographically for determinism.
    """
    from pyspark.sql import functions as F

    levels: dict[str, list] = {}
    for col in categorical_cols:
        n_distinct = spark_df.select(F.approxCountDistinct(col)).first()[0]
        if n_distinct > max_levels:
            raise ValueError(
                f"Categorical '{col}' has approximately {n_distinct} distinct "
                f"levels, above max_levels={max_levels}. Consider manual binning "
                f"or coarser representation."
            )
        rows = spark_df.select(col).distinct().collect()
        vals = sorted(r[col] for r in rows if r[col] is not None)
        levels[col] = vals
    return levels


def fit_model_spec_from_spark(
    formula: str,
    spark_df: Any,
    categorical_cols: list[str],
    continuous_cols: list[str],
    max_levels: int = 10_000,
) -> tuple[Any, list[str]]:
    """Schema-frame discovery: build ModelSpec without materializing the full data.

    Process (ADR 0024):
      1. Discover categorical levels via Spark distinct() with cardinality bound.
      2. Build a tiny pandas schema-frame containing each level at least once
         (plus zero-valued placeholders for continuous columns).
      3. Fit ModelSpec against the schema-frame via formulaic — captures
         the level set in transform_state.
      4. Validate transform_state contains *only* the categorical level
         mappings (no spline knots, no scale/center stats).
    """
    schema_pdf = build_schema_frame(spark_df, categorical_cols, continuous_cols,
                                    max_levels)
    return fit_model_spec_on_schema_frame(formula, schema_pdf, categorical_cols)


def build_schema_frame(
    spark_df: Any,
    categorical_cols: list[str],
    continuous_cols: list[str],
    max_levels: int = 10_000,
) -> pd.DataFrame:
    """The schema-frame (ADR 0024): a tiny pandas frame holding every level of each
    categorical column at least once (levels discovered by Spark distinct() under the
    cardinality bound) and a 0.0 placeholder for each continuous column. Fitting a
    formula on it captures the level sets without materializing the data."""
    levels = discover_categorical_levels_spark(spark_df, categorical_cols, max_levels)
    max_n_levels = max((len(v) for v in levels.values()), default=1)
    rows = []
    for i in range(max(max_n_levels, 1)):
        row = {}
        for col in categorical_cols:
            col_levels = levels[col]
            row[col] = col_levels[i % len(col_levels)]
        for col in continuous_cols:
            row[col] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def fit_model_spec_on_schema_frame(
    formula: str, schema_pdf: pd.DataFrame, categorical_cols: list[str],
) -> tuple[Any, list[str]]:
    """Fit the formula's ModelSpec on a schema-frame and check that it learned
    nothing but the categorical level mappings (no spline knots, no scale / centre
    statistics, which would have been learned from the placeholder values)."""
    spec, names = fit_model_spec(formula, schema_pdf)
    extras = _unexpected_transform_state(spec, categorical_cols)
    if extras:
        raise ValueError(
            f"Formula introduced unexpected stateful transforms: {extras}. "
            f"This should have been caught by validate_formula; please file a bug."
        )
    return spec, names


# --------------------------------------------------------------------------- #
# Which formula variables are categorical, and the spec as JSON (ADR 0056).   #
# --------------------------------------------------------------------------- #
# The first argument of a C(...) call: `C(sex)`, `C(site, contr.treatment(...))`
# and `C(age > 50)` each name a variable whose raw values the schema-frame must hold.
_C_FIRST_ARG = re.compile(r"\bC\(\s*([A-Za-z_]\w*)")


def formula_variables(formula: str) -> list[str]:
    """The data columns the formula reads, sorted."""
    formulaic = _formulaic()
    return sorted(str(v) for v in formulaic.Formula(formula).required_variables)


def split_formula_variables(
    formula: str, spark_types: dict[str, str],
) -> tuple[list[str], list[str]]:
    """``(categorical, continuous)`` formula variables (ADR 0024 step 1): a variable
    is categorical when it is the first argument of a ``C(...)`` in the formula or
    its Spark type is string or boolean; every other variable is continuous.
    ``spark_types`` maps a column name to its Spark ``simpleString`` type."""
    variables = formula_variables(formula)
    missing = [v for v in variables if v not in spark_types]
    if missing:
        raise ValueError(f"The covariate formula reads column(s) {missing} that the "
                         f"DataFrame does not have.")
    in_c = set(_C_FIRST_ARG.findall(formula))
    categorical = [v for v in variables
                   if v in in_c or spark_types[v] in ("string", "boolean")]
    continuous = [v for v in variables if v not in categorical]
    return categorical, continuous


_JSON_DTYPES = {"object", "bool", "int8", "int16", "int32", "int64",
                "float32", "float64"}


def covariate_spec_state(
    formula: str, schema_pdf: pd.DataFrame, covariate_names: list[str],
    categorical_cols: list[str],
) -> dict:
    """The JSON-able learned state of a schema-frame-fitted ModelSpec: the formula,
    the schema-frame (column order, dtypes, values) and the design-column names it
    produced. ``model_spec_from_state`` rebuilds the spec from it.

    Why JSON and not the ModelSpec: a pickle is not a portable file format and must
    never be loaded from an untrusted source, so a published package does not
    unpickle saved models. formulaic offers no other serialization of a spec (its
    encoder state holds contrast objects), and none is needed: the spec was fitted
    on the schema-frame alone, so refitting the formula on the same frame
    reproduces it exactly."""
    columns = []
    for col in schema_pdf.columns:
        dtype = str(schema_pdf[col].dtype)
        values = schema_pdf[col].tolist()
        if dtype not in _JSON_DTYPES or (dtype == "object" and not all(
                isinstance(v, str) for v in values)):
            raise ValueError(
                f"Covariate column '{col}' has dtype {dtype} (value types "
                f"{sorted({type(v).__name__ for v in values})}); a saved covariate "
                f"spec supports string, boolean, integer and floating columns.")
        columns.append({"name": str(col), "dtype": dtype, "values": values})
    return {"formula": formula, "categorical_cols": list(categorical_cols),
            "schema_frame": columns, "covariate_names": list(covariate_names)}


def model_spec_from_state(state: dict) -> Any:
    """Rebuild the ModelSpec that ``covariate_spec_state`` describes, and check it
    produces the design columns it did at fit."""
    schema_pdf = pd.DataFrame({
        c["name"]: pd.Series(c["values"], dtype=c["dtype"])
        for c in state["schema_frame"]})
    spec, names = fit_model_spec_on_schema_frame(
        state["formula"], schema_pdf, list(state.get("categorical_cols", [])))
    if list(names) != list(state["covariate_names"]):
        raise ValueError(
            f"The covariate formula rebuilt with columns {names}, not the "
            f"{state['covariate_names']} it was fitted with (another formulaic "
            f"version?).")
    return spec


def _unexpected_transform_state(spec: Any, categorical_cols: list[str]) -> list[str]:
    """Identify transform_state entries that aren't covariate-level captures."""
    # formulaic stores transform-specific state per term/factor; the exact
    # API shape may vary across versions. The safest check is to enumerate
    # the transform-state keys and confirm none reference rejected callables.
    ts = getattr(spec, "transform_state", None) or {}
    extras = []
    for key, value in ts.items():
        key_str = str(key).lower()
        for rejected in _STATEFUL_REJECT_LIST:
            if rejected in key_str:
                extras.append(key)
                break
    return extras

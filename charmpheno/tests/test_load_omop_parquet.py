"""Local parquet loader for OMOP-shaped data."""
from pathlib import Path

import pytest

FIXTURE = Path(__file__).resolve().parent / "data" / "tiny_omop.parquet"


def test_load_omop_parquet_returns_canonical_shape(spark):
    from charmpheno.omop import validate
    from charmpheno.omop.local import load_omop_parquet

    df = load_omop_parquet(str(FIXTURE), spark=spark)
    validate(df)  # must not raise
    assert df.count() == 5
    assert {c.name for c in df.schema.fields} >= {
        "person_id", "visit_occurrence_id", "concept_id", "concept_name"
    }


def test_load_omop_parquet_raises_on_missing_file(spark):
    from charmpheno.omop.local import load_omop_parquet

    with pytest.raises(Exception):
        load_omop_parquet("/nonexistent/path.parquet", spark=spark)

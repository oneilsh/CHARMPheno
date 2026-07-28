"""The exclude_vocabularies filter predicate: drops named vocabularies, keeps
NULL (unmapped concepts). The live BigQuery read is cluster-covered; this tests
the pure predicate against a local-Spark frame.

Uses the shared session-scoped `spark` fixture from tests/conftest.py rather
than a private local one: a module-local fixture that calls SparkSession
.stop() on teardown kills the single JVM SparkContext shared with every other
test module's session-scoped `spark` fixture, leaving later test modules
holding a dead SparkContext (AttributeError: 'NoneType' object has no
attribute 'sc') -- reproduced when running the full `tests/` suite."""


def test_exclude_vocabularies_predicate_drops_named_keeps_null(spark):
    from charmpheno.omop.bigquery import _exclude_vocab_filter
    df = spark.createDataFrame(
        [(1, "PPI"), (2, "SNOMED"), (3, None), (4, "PPI")],
        "concept_id int, vocabulary_id string")
    kept = {r["concept_id"] for r in _exclude_vocab_filter(df, ("PPI",)).collect()}
    assert kept == {2, 3}          # PPI dropped; NULL (unmapped) KEPT


def test_exclude_vocabularies_empty_is_identity(spark):
    from charmpheno.omop.bigquery import _exclude_vocab_filter
    df = spark.createDataFrame(
        [(1, "PPI"), (2, None)], "concept_id int, vocabulary_id string")
    assert _exclude_vocab_filter(df, ()).count() == 2      # no filter applied

"""Smoke tests for charmpheno.evaluate.lda_compare.

Doesn't assert correctness (that's covered by the spark_vi integration test
and downstream visual inspection of biplots). Pins shape and that the API
runs end-to-end on a tiny fixture.
"""
import numpy as np
from pyspark.sql.types import StructType, StructField, IntegerType, StringType


def _tiny_omop_df_with_topics(spark):
    schema = StructType([
        StructField("person_id", IntegerType(), False),
        StructField("visit_occurrence_id", IntegerType(), False),
        StructField("concept_id", IntegerType(), False),
        StructField("concept_name", StringType(), True),
        StructField("true_topic_id", IntegerType(), False),
    ])
    rows = []
    for p in range(1, 4):
        for v in range(2):
            for cid in [10, 20, 30]:
                rows.append((p, v, cid, str(cid), 0))
            for cid in [40, 50, 60]:
                rows.append((p, v, cid, str(cid), 1))
    return spark.createDataFrame(rows, schema=schema)


def test_run_ours_produces_artifacts_of_expected_shape(spark):
    from charmpheno.evaluate.lda_compare import run_ours
    from charmpheno.omop import to_bow_dataframe
    from spark_vi.core import BOWDocument, VIConfig

    df_raw = _tiny_omop_df_with_topics(spark)
    bow_df, vocab_map = to_bow_dataframe(df_raw)
    rdd = bow_df.rdd.map(BOWDocument.from_spark_row)

    np.random.seed(0)
    art = run_ours(
        rdd=rdd, vocab_size=len(vocab_map), K=2,
        config=VIConfig(max_iterations=5, mini_batch_fraction=0.5,
                         random_seed=0, convergence_tol=1e-9),
    )
    assert art.topics_matrix.shape == (2, len(vocab_map))
    assert art.topic_prevalence.shape == (2,)
    assert art.elbo_trace is not None
    assert len(art.elbo_trace) <= 5
    assert art.wall_time_seconds > 0
    assert art.final_log_likelihood is None


def test_run_mllib_produces_artifacts_of_expected_shape(spark):
    from charmpheno.evaluate.lda_compare import run_mllib
    from charmpheno.omop import to_bow_dataframe

    df_raw = _tiny_omop_df_with_topics(spark)
    bow_df, vocab_map = to_bow_dataframe(df_raw)

    art = run_mllib(df=bow_df, vocab_size=len(vocab_map), K=2,
                    max_iter=5, seed=0)
    assert art.topics_matrix.shape == (2, len(vocab_map))
    assert art.topic_prevalence.shape == (2,)
    assert art.elbo_trace is None
    assert art.final_log_likelihood is not None

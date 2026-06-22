"""STM end-to-end smoke: synthetic data through covariates build, fit,
and dashboard bundle.

Marked slow; runs locally without BigQuery (uses the synthetic corpus
helper from spark-vi tests as a starting point).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")
formulaic = pytest.importorskip("formulaic")


@pytest.mark.slow
def test_end_to_end_synthetic_stm(spark, tmp_path: Path):
    """Build covariates -> fit STM -> assert bundle structure.

    Uses synthetic corpus + person tables generated locally. Does not
    exercise BigQuery; the covariates build path is exercised against
    in-memory Spark DataFrames.
    """
    from charmpheno.omop.covariates import build_patient_covariate_df
    from spark_vi.mllib.topic.stm import StreamingSTM

    # Synthetic person table.
    person_pdf = pd.DataFrame({
        "person_id": list(range(1, 51)),
        "sex":       ["M", "F"] * 25,
        "age":       np.linspace(20, 70, 50).tolist(),
    })
    person_df = spark.createDataFrame(person_pdf)

    # Synthetic BOW corpus.
    K, V, doc_len = 4, 30, 60
    rng = np.random.default_rng(42)
    beta = rng.dirichlet(np.full(V, 0.1), size=K)
    bow_rows = []
    for pid in range(1, 51):
        x = np.array([1.0, 1.0 if person_pdf.sex.iloc[pid - 1] == "M" else 0.0,
                      person_pdf.age.iloc[pid - 1] / 50.0])
        eta = rng.normal(scale=0.3, size=K)
        theta = np.exp(eta - eta.max()); theta /= theta.sum()
        z = rng.choice(K, size=doc_len, p=theta)
        w = np.array([rng.choice(V, p=beta[zi]) for zi in z])
        unique, counts = np.unique(w, return_counts=True)
        from pyspark.ml.linalg import SparseVector
        bow_rows.append({
            "person_id": pid, "doc_id": pid,
            "features": SparseVector(V, unique.tolist(), counts.astype(float).tolist()),
        })
    bow_df = spark.createDataFrame(pd.DataFrame(bow_rows))

    # Build covariates from formula.
    cov_df, model_spec, names = build_patient_covariate_df(
        person_df, covariate_formula="~ C(sex) + age",
        categorical_cols=["sex"], continuous_cols=["age"],
    )

    # Join + fit.
    joined = bow_df.join(cov_df, on="person_id")
    est = StreamingSTM(
        K=K, features_col="features",
        covariates_col="covariates", covariate_names=names,
        random_seed=42,
    )
    model = est.fit(joined, max_iter=10, subsampling_rate=1.0,
                    tau0=64.0, kappa=0.7, save_interval=5)

    # Inspect bundle via adapt_stm.
    from charmpheno.export.dashboard import adapt_stm
    out = tmp_path / "bundle"
    out.mkdir()
    Gamma = model.global_params["Gamma"]
    adapt_stm(out_dir=out, Gamma=Gamma, covariate_names=names,
              K=Gamma.shape[1], P=Gamma.shape[0])
    cov_json = json.loads((out / "covariate_effects.json").read_text())
    assert len(cov_json) == len(names)
    assert all("per_topic" in row for row in cov_json)

    # Faithful corpus-mean corpus_prevalence path (the dashboard's
    # "default topic proportion"). Exercises the real formula-expanded cov_df
    # and the real fitted Gamma through the distributed reduction + adapter.
    from charmpheno.omop.covariates import (
        corpus_mean_proportions_from_covariate_df,
    )
    from charmpheno.export.model_adapter import adapt

    faithful = corpus_mean_proportions_from_covariate_df(
        cov_df, np.asarray(Gamma, dtype=np.float64)
    )
    assert faithful.shape == (K,)
    np.testing.assert_allclose(faithful.sum(), 1.0)

    # Wire it through the generic adapter exactly as build_dashboard_cloud does
    # (driver augments these metadata fields post-fit).
    model.metadata["model_class"] = "stm"
    model.metadata["covariate_manifest"] = {"covariate_names": names}
    export = adapt(model, stm_corpus_prevalence=faithful)
    np.testing.assert_allclose(export.corpus_prevalence, faithful)

    # With non-degenerate covariates (sex, age vary across the cohort), the
    # faithful corpus mean must differ from the intercept-only stand-in —
    # otherwise the whole feature would be a no-op.
    stand_in = adapt(model).corpus_prevalence
    assert not np.allclose(export.corpus_prevalence, stand_in)


@pytest.mark.slow
def test_combined_cohort_comorbid_two_documents(spark, tmp_path):
    """A comorbid patient yields two source-labeled documents that fit and
    join on the composite key; covariates are per-(person, cohort)."""
    import numpy as np
    import pandas as pd
    from charmpheno.omop.doc_spec import PatientCohortDocSpec
    from charmpheno.omop.topic_prep import to_bow_dataframe
    from charmpheno.omop.covariates import build_patient_covariate_df
    from pyspark.sql import functions as F

    # Events already tagged with source_cohort (as _combine_cohorts would).
    # person 1 is comorbid (cancer + dementia), person 2 cancer, person 3 dementia.
    rng = np.random.default_rng(0)
    rows = []
    def emit(pid, cohort, codes):
        for c in codes:
            rows.append((pid, cohort, str(c)))
    emit(1, "cancer", rng.integers(0, 10, 40))
    emit(1, "dementia", rng.integers(10, 20, 40))
    emit(2, "cancer", rng.integers(0, 10, 40))
    emit(3, "dementia", rng.integers(10, 20, 40))
    events = spark.createDataFrame(rows, ["person_id", "source_cohort", "concept_id"])

    bow_df, vocab_map = to_bow_dataframe(
        events, doc_spec=PatientCohortDocSpec(min_doc_length=0),
        token_col="concept_id",
    )
    bow_df = bow_df.withColumn(
        "source_cohort", F.split(F.col("doc_id"), ":").getItem(0))
    # Comorbid person 1 -> two docs.
    docs = {(r["person_id"], r["source_cohort"]) for r in bow_df.collect()}
    assert (1, "cancer") in docs and (1, "dementia") in docs

    person_pdf = pd.DataFrame({
        "person_id":     [1, 1, 2, 3],
        "source_cohort": ["cancer", "dementia", "cancer", "dementia"],
        "sex":           ["M", "M", "F", "F"],
        "age":           [60.0, 60.0, 70.0, 80.0],
    })
    cov_df, _, names = build_patient_covariate_df(
        spark.createDataFrame(person_pdf),
        covariate_formula="~ C(source_cohort) + C(sex) + age",
        categorical_cols=["source_cohort", "sex"], continuous_cols=["age"],
        key_cols=["person_id", "source_cohort"],
    )
    joined = bow_df.join(cov_df, on=["person_id", "source_cohort"], how="inner")
    # Each document joins to exactly one covariate row.
    assert joined.count() == bow_df.count()

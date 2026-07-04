"""Glue test for the fit-driver concentration-readout chain.

Exercises exactly what analysis/local/fit_stm_local.py and
analysis/cloud/stm_bigquery_cloud.py do: build a `joined`-shaped Spark
DataFrame (features/covariates/group columns), map it via
_vector_to_stm_document, and feed the resulting doc RDD into
corpus_concentration_stm_rdd -- without running a full fit. See
tests/test_concentration_stm.py for the underlying numpy<->RDD parity
coverage of corpus_concentration_stm_rdd itself; this test only proves the
DataFrame -> STMDocument -> readout wiring the drivers rely on.
"""
from __future__ import annotations

import numpy as np
from pyspark.ml.linalg import DenseVector, SparseVector
from pyspark.sql import Row

from spark_vi.mllib.topic._common import _vector_to_stm_document
from spark_vi.mllib.topic.stm import corpus_concentration_stm_rdd
from tests._stm_synth import fit_stm, synthetic_gated_corpus


def _global_params_from_fit(gp):
    return {"lambda": gp["lambda"], "Gamma": gp["Gamma"], "Sigma": gp["Sigma"]}


def test_stm_readout_from_dataframe(spark):
    docs, planted, part = synthetic_gated_corpus(
        groups=("A", "B"), fg_per_group=1, bg_k=2, V=40, D=40, doc_len=25,
        bg_frac=0.5, seed=5,
    )
    K = part.K
    gp = fit_stm(docs, K=K, V=40, sigma_init=1.0, n_iter=20, partition=part, seed=5)
    global_params = _global_params_from_fit(gp)

    # Build the exact `joined`-shaped DataFrame the drivers construct:
    # one row per doc, features as a SparseVector (V,), covariates as a
    # DenseVector (P,), source_cohort as the doc's (single) gating group.
    V = 40
    rows = []
    for doc in docs:
        features = SparseVector(V, sorted(doc.indices.tolist()),
                                 [float(doc.counts[list(doc.indices).index(i)])
                                  for i in sorted(doc.indices.tolist())])
        covariates = DenseVector(doc.x.tolist())
        (group,) = doc.groups
        rows.append(Row(features=features, covariates=covariates,
                         source_cohort=group))
    df = spark.createDataFrame(rows)

    doc_rdd = df.rdd.map(lambda row: _vector_to_stm_document(
        row, features_col="features", covariates_col="covariates",
        group_col="source_cohort"))

    summary = corpus_concentration_stm_rdd(
        doc_rdd, global_params, part, reference=0)

    assert set(summary.keys()) == {"n_docs", "top_mass", "eff_topics"}
    assert summary["n_docs"] == len(docs)
    assert 1.0 <= summary["eff_topics"]["mean"] <= K

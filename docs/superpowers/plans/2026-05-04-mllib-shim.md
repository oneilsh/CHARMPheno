# MLlib Estimator/Transformer shim for VanillaLDA — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap `spark_vi.models.lda.VanillaLDA` in `pyspark.ml.base.Estimator` / `Model` subclasses so it plugs into MLlib pipelines and presents a familiar API.

**Architecture:** Two thin classes in a new `spark_vi.mllib` subpackage. `VanillaLDAEstimator._fit` translates MLlib-shaped Params into `(VanillaLDA, VIConfig)` and runs the existing `VIRunner`. `VanillaLDAModel._transform` applies a Python UDF that calls `_cavi_doc_inference` per row. No new SVI math; the shim is a translation layer.

**Tech Stack:** Python, PySpark (`pyspark.ml.base`, `pyspark.ml.linalg`, `pyspark.ml.param`), NumPy, scipy.special, pytest.

**Spec:** [docs/superpowers/specs/2026-05-04-mllib-shim-design.md](../specs/2026-05-04-mllib-shim-design.md)

---

## File Structure

**Create:**
- `spark-vi/spark_vi/mllib/__init__.py` — re-exports the two shim classes.
- `spark-vi/spark_vi/mllib/lda.py` — `VanillaLDAEstimator`, `VanillaLDAModel`, plus private helpers `_vector_to_bow_document`, `_build_model_and_config`, `_validate_unsupported_params`.
- `spark-vi/tests/test_mllib_lda.py` — fast unit tests for the shim.
- `docs/decisions/0009-mllib-shim.md` — ADR documenting the design choices.

**Modify:**
- `charmpheno/charmpheno/evaluate/lda_compare.py` — rewrite `run_ours` to use the shim. Signature unchanged.

**Don't touch:**
- `spark-vi/spark_vi/__init__.py` — no top-level export of `mllib` (opt-in import path).
- `spark-vi/spark_vi/models/lda.py` — the shim wraps `VanillaLDA`, doesn't modify it.
- `charmpheno/tests/test_lda_compare.py` — slow parity test stays as the source-of-truth gate.

---

## Task 1: Scaffold the package

**Files:**
- Create: `spark-vi/spark_vi/mllib/__init__.py`
- Create: `spark-vi/spark_vi/mllib/lda.py`

- [ ] **Step 1: Create the empty package**

```python
# spark-vi/spark_vi/mllib/__init__.py
"""MLlib Estimator/Transformer shims for spark_vi models.

Opt-in import path — `import spark_vi` does not transitively load this
subpackage, so users who don't need MLlib integration don't pay the
pyspark.ml import cost. Per ADR 0009.
"""
from spark_vi.mllib.lda import VanillaLDAEstimator, VanillaLDAModel

__all__ = ["VanillaLDAEstimator", "VanillaLDAModel"]
```

- [ ] **Step 2: Create the empty module with class skeletons**

```python
# spark-vi/spark_vi/mllib/lda.py
"""MLlib Estimator/Transformer shim for VanillaLDA.

Wraps spark_vi.models.lda.VanillaLDA + spark_vi.core.runner.VIRunner so the
model behaves like a pyspark.ml.clustering.LDA-shaped Estimator/Model pair.
The shim is a translation layer; all SVI logic lives in VanillaLDA. See
docs/superpowers/specs/2026-05-04-mllib-shim-design.md and ADR 0009.
"""
from __future__ import annotations

from pyspark.ml.base import Estimator, Model


class VanillaLDAEstimator(Estimator):
    """Stub — params and _fit added in subsequent tasks."""


class VanillaLDAModel(Model):
    """Stub — state and methods added in subsequent tasks."""
```

- [ ] **Step 3: Verify the package imports**

Run: `python -c "from spark_vi.mllib import VanillaLDAEstimator, VanillaLDAModel; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add spark-vi/spark_vi/mllib/__init__.py spark-vi/spark_vi/mllib/lda.py
git commit -m "mllib shim: scaffold spark_vi.mllib package"
```

---

## Task 2: Param surface with MLlib-matching defaults

Declare every Param on the Estimator using the standard MLlib `Param(Params._dummy(), name, doc, typeConverter=...)` pattern. Defaults match `pyspark.ml.clustering.LDA` for the shared subset and ADR 0008 for our extras (`gammaShape`, `caviMaxIter`, `caviTol`).

**Files:**
- Modify: `spark-vi/spark_vi/mllib/lda.py`
- Test: `spark-vi/tests/test_mllib_lda.py`

- [ ] **Step 1: Write the failing test**

```python
# spark-vi/tests/test_mllib_lda.py
"""Tests for spark_vi.mllib.lda — fast unit tests for the MLlib shim."""
from __future__ import annotations

import pytest


def test_default_params_match_mllib_lda():
    """Each shared Param defaults to the same value pyspark.ml.clustering.LDA uses."""
    from pyspark.ml.clustering import LDA as MLlibLDA
    from spark_vi.mllib.lda import VanillaLDAEstimator

    ours = VanillaLDAEstimator()
    theirs = MLlibLDA()

    for name in [
        "k", "maxIter", "featuresCol", "topicDistributionCol",
        "optimizer", "learningOffset", "learningDecay",
        "subsamplingRate", "optimizeDocConcentration",
    ]:
        assert ours.getOrDefault(name) == theirs.getOrDefault(name), (
            f"Param {name!r} default mismatch: ours={ours.getOrDefault(name)!r} "
            f"theirs={theirs.getOrDefault(name)!r}"
        )


def test_our_extras_have_adr_0008_defaults():
    from spark_vi.mllib.lda import VanillaLDAEstimator

    e = VanillaLDAEstimator()
    assert e.getOrDefault("gammaShape") == 100.0
    assert e.getOrDefault("caviMaxIter") == 100
    assert e.getOrDefault("caviTol") == 1e-3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest spark-vi/tests/test_mllib_lda.py -v`
Expected: FAIL with `AttributeError`/`Param … not declared`.

- [ ] **Step 3: Declare the Params on the Estimator**

Replace the `VanillaLDAEstimator` stub with:

```python
# Add at the top of spark-vi/spark_vi/mllib/lda.py with the existing imports
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import (
    HasFeaturesCol, HasMaxIter, HasSeed,
)
from pyspark import keyword_only


class VanillaLDAEstimator(Estimator, HasFeaturesCol, HasMaxIter, HasSeed):
    """MLlib-shaped Estimator wrapping spark_vi.models.lda.VanillaLDA.

    Param defaults mirror pyspark.ml.clustering.LDA for the shared subset
    and ADR 0008 for our extras (gammaShape, caviMaxIter, caviTol).
    """

    k = Param(
        Params._dummy(), "k",
        "number of topics (clusters) to infer; must be >= 1",
        typeConverter=TypeConverters.toInt,
    )
    topicDistributionCol = Param(
        Params._dummy(), "topicDistributionCol",
        "output column with estimates of topic mixture for each document",
        typeConverter=TypeConverters.toString,
    )
    optimizer = Param(
        Params._dummy(), "optimizer",
        "optimizer; only 'online' is supported by this shim",
        typeConverter=TypeConverters.toString,
    )
    learningOffset = Param(
        Params._dummy(), "learningOffset",
        "tau0 in the Robbins-Monro step rho_t = (tau0 + t)^-kappa",
        typeConverter=TypeConverters.toFloat,
    )
    learningDecay = Param(
        Params._dummy(), "learningDecay",
        "kappa in the Robbins-Monro step rho_t = (tau0 + t)^-kappa",
        typeConverter=TypeConverters.toFloat,
    )
    subsamplingRate = Param(
        Params._dummy(), "subsamplingRate",
        "fraction of corpus sampled per mini-batch",
        typeConverter=TypeConverters.toFloat,
    )
    docConcentration = Param(
        Params._dummy(), "docConcentration",
        "Dirichlet concentration alpha on theta; scalar (symmetric) only — vector raises",
        typeConverter=TypeConverters.toListFloat,
    )
    topicConcentration = Param(
        Params._dummy(), "topicConcentration",
        "Dirichlet concentration eta on beta; scalar (symmetric) only",
        typeConverter=TypeConverters.toFloat,
    )
    optimizeDocConcentration = Param(
        Params._dummy(), "optimizeDocConcentration",
        "whether to optimize alpha; True is rejected (see ADR 0008)",
        typeConverter=TypeConverters.toBoolean,
    )
    gammaShape = Param(
        Params._dummy(), "gammaShape",
        "shape parameter for Gamma init of variational gamma; ADR 0008 default 100.0",
        typeConverter=TypeConverters.toFloat,
    )
    caviMaxIter = Param(
        Params._dummy(), "caviMaxIter",
        "max iterations for the inner CAVI loop per document",
        typeConverter=TypeConverters.toInt,
    )
    caviTol = Param(
        Params._dummy(), "caviTol",
        "relative tolerance on gamma for CAVI early stop",
        typeConverter=TypeConverters.toFloat,
    )

    @keyword_only
    def __init__(
        self,
        *,
        k: int = 10,
        maxIter: int = 20,
        seed: int | None = None,
        featuresCol: str = "features",
        topicDistributionCol: str = "topicDistribution",
        optimizer: str = "online",
        learningOffset: float = 1024.0,
        learningDecay: float = 0.51,
        subsamplingRate: float = 0.05,
        docConcentration: list[float] | None = None,
        topicConcentration: float | None = None,
        optimizeDocConcentration: bool = False,
        gammaShape: float = 100.0,
        caviMaxIter: int = 100,
        caviTol: float = 1e-3,
    ) -> None:
        super().__init__()
        self._setDefault(
            k=10, maxIter=20,
            featuresCol="features", topicDistributionCol="topicDistribution",
            optimizer="online",
            learningOffset=1024.0, learningDecay=0.51, subsamplingRate=0.05,
            optimizeDocConcentration=False,
            gammaShape=100.0, caviMaxIter=100, caviTol=1e-3,
        )
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, **kwargs) -> "VanillaLDAEstimator":
        """Standard MLlib pattern: set any subset of params after construction."""
        return self._set(**kwargs)

    def _fit(self, dataset):
        raise NotImplementedError("Implemented in a later task.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest spark-vi/tests/test_mllib_lda.py -v`
Expected: PASS for both tests.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/mllib/lda.py spark-vi/tests/test_mllib_lda.py
git commit -m "mllib shim: declare Param surface with MLlib-matching defaults"
```

---

## Task 3: Vector ↔ BOWDocument helper

Convert a Spark `Vector` (Sparse or Dense) to a `BOWDocument`. Used by both `_fit` and `_transform`.

**Files:**
- Modify: `spark-vi/spark_vi/mllib/lda.py`
- Test: `spark-vi/tests/test_mllib_lda.py`

- [ ] **Step 1: Write the failing tests**

```python
# Append to spark-vi/tests/test_mllib_lda.py
import numpy as np


def test_vector_to_bow_document_handles_sparse_vector():
    from pyspark.ml.linalg import Vectors
    from spark_vi.mllib.lda import _vector_to_bow_document

    sv = Vectors.sparse(5, [0, 2, 4], [1.0, 3.0, 2.0])
    doc = _vector_to_bow_document(sv)

    np.testing.assert_array_equal(doc.indices, [0, 2, 4])
    np.testing.assert_array_equal(doc.counts, [1.0, 3.0, 2.0])
    assert doc.length == 6


def test_vector_to_bow_document_handles_dense_vector_with_zeros():
    """DenseVectors with embedded zeros should round-trip to a sparse BOWDocument."""
    from pyspark.ml.linalg import Vectors
    from spark_vi.mllib.lda import _vector_to_bow_document

    dv = Vectors.dense([0.0, 2.0, 0.0, 5.0, 0.0])
    doc = _vector_to_bow_document(dv)

    np.testing.assert_array_equal(doc.indices, [1, 3])
    np.testing.assert_array_equal(doc.counts, [2.0, 5.0])
    assert doc.length == 7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest spark-vi/tests/test_mllib_lda.py::test_vector_to_bow_document_handles_sparse_vector spark-vi/tests/test_mllib_lda.py::test_vector_to_bow_document_handles_dense_vector_with_zeros -v`
Expected: FAIL with `ImportError: cannot import name '_vector_to_bow_document'`.

- [ ] **Step 3: Implement the helper**

Add to `spark-vi/spark_vi/mllib/lda.py` (top-level, after imports):

```python
import numpy as np
from pyspark.ml.linalg import DenseVector, SparseVector, Vector

from spark_vi.core.types import BOWDocument


def _vector_to_bow_document(v: Vector) -> BOWDocument:
    """Convert a pyspark.ml.linalg Vector to a BOWDocument.

    SparseVector indices/values pass through. DenseVectors are sparsified
    (nonzero entries only) so the downstream CAVI loop sees the same shape
    of input regardless of the producer (CountVectorizer emits Sparse,
    user-constructed inputs may be Dense).
    """
    if isinstance(v, SparseVector):
        indices = np.asarray(v.indices, dtype=np.int32)
        counts = np.asarray(v.values, dtype=np.float64)
    elif isinstance(v, DenseVector):
        values = np.asarray(v.values, dtype=np.float64)
        nz = np.nonzero(values)[0].astype(np.int32)
        indices = nz
        counts = values[nz]
    else:
        raise TypeError(
            f"_vector_to_bow_document expected Sparse/DenseVector, got {type(v).__name__}"
        )
    return BOWDocument(indices=indices, counts=counts, length=int(counts.sum()))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest spark-vi/tests/test_mllib_lda.py -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/mllib/lda.py spark-vi/tests/test_mllib_lda.py
git commit -m "mllib shim: Vector -> BOWDocument helper (Sparse + Dense)"
```

---

## Task 4: Param translation to (VanillaLDA, VIConfig)

Translate the Estimator's Param values into the constructor args for `VanillaLDA` and `VIConfig`. Pure function of Params; testable without Spark.

**Files:**
- Modify: `spark-vi/spark_vi/mllib/lda.py`
- Test: `spark-vi/tests/test_mllib_lda.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to spark-vi/tests/test_mllib_lda.py
def test_param_translation_to_model_and_config():
    from spark_vi.core.config import VIConfig
    from spark_vi.models.lda import VanillaLDA
    from spark_vi.mllib.lda import VanillaLDAEstimator, _build_model_and_config

    e = VanillaLDAEstimator(
        k=7, maxIter=42, seed=2026,
        learningOffset=512.0, learningDecay=0.6,
        subsamplingRate=0.1,
        docConcentration=[0.05], topicConcentration=0.02,
        gammaShape=50.0, caviMaxIter=200, caviTol=1e-4,
    )
    model, config = _build_model_and_config(e, vocab_size=100)

    assert isinstance(model, VanillaLDA)
    assert model.K == 7
    assert model.V == 100
    assert model.alpha == pytest.approx(0.05)
    assert model.eta == pytest.approx(0.02)
    assert model.gamma_shape == pytest.approx(50.0)
    assert model.cavi_max_iter == 200
    assert model.cavi_tol == pytest.approx(1e-4)

    assert isinstance(config, VIConfig)
    assert config.max_iterations == 42
    assert config.learning_rate_tau0 == pytest.approx(512.0)
    assert config.learning_rate_kappa == pytest.approx(0.6)
    assert config.mini_batch_fraction == pytest.approx(0.1)
    assert config.random_seed == 2026


def test_param_translation_resolves_none_concentrations_to_one_over_k():
    """Per ADR 0008: alpha = eta = 1/K when caller passes None (the default)."""
    from spark_vi.mllib.lda import VanillaLDAEstimator, _build_model_and_config

    e = VanillaLDAEstimator(k=4)
    model, _ = _build_model_and_config(e, vocab_size=10)

    assert model.alpha == pytest.approx(0.25)
    assert model.eta == pytest.approx(0.25)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest spark-vi/tests/test_mllib_lda.py::test_param_translation_to_model_and_config spark-vi/tests/test_mllib_lda.py::test_param_translation_resolves_none_concentrations_to_one_over_k -v`
Expected: FAIL with `ImportError: cannot import name '_build_model_and_config'`.

- [ ] **Step 3: Implement the helper**

Add to `spark-vi/spark_vi/mllib/lda.py`:

```python
from spark_vi.core.config import VIConfig
from spark_vi.models.lda import VanillaLDA


def _build_model_and_config(
    estimator: "VanillaLDAEstimator",
    vocab_size: int,
) -> tuple[VanillaLDA, VIConfig]:
    """Translate Estimator Params into (VanillaLDA, VIConfig).

    Symmetric-alpha-only: docConcentration may be None or a length-1 list
    (the latter is what TypeConverters.toListFloat produces for a scalar
    input). Vector docConcentration is rejected by _validate_unsupported_params,
    not here.
    """
    k = estimator.getOrDefault("k")

    doc_conc = estimator.getOrDefault("docConcentration") if estimator.isSet("docConcentration") else None
    if doc_conc is None:
        alpha = 1.0 / k
    else:
        alpha = float(doc_conc[0])

    topic_conc = estimator.getOrDefault("topicConcentration") if estimator.isSet("topicConcentration") else None
    eta = 1.0 / k if topic_conc is None else float(topic_conc)

    model = VanillaLDA(
        K=k,
        vocab_size=vocab_size,
        alpha=alpha,
        eta=eta,
        gamma_shape=estimator.getOrDefault("gammaShape"),
        cavi_max_iter=estimator.getOrDefault("caviMaxIter"),
        cavi_tol=estimator.getOrDefault("caviTol"),
    )

    seed = estimator.getOrDefault("seed") if estimator.isSet("seed") else None
    config = VIConfig(
        max_iterations=estimator.getOrDefault("maxIter"),
        learning_rate_tau0=estimator.getOrDefault("learningOffset"),
        learning_rate_kappa=estimator.getOrDefault("learningDecay"),
        mini_batch_fraction=estimator.getOrDefault("subsamplingRate"),
        random_seed=seed,
    )
    return model, config
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest spark-vi/tests/test_mllib_lda.py -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/mllib/lda.py spark-vi/tests/test_mllib_lda.py
git commit -m "mllib shim: param translation to (VanillaLDA, VIConfig)"
```

---

## Task 5: Reject unsupported configurations

Validation that runs at fit time: `optimizer != "online"`, `optimizeDocConcentration=True`, vector `docConcentration` (length > 1).

**Files:**
- Modify: `spark-vi/spark_vi/mllib/lda.py`
- Test: `spark-vi/tests/test_mllib_lda.py`

- [ ] **Step 1: Write the failing tests**

```python
# Append to spark-vi/tests/test_mllib_lda.py
def test_unsupported_optimizer_em_raises():
    from spark_vi.mllib.lda import VanillaLDAEstimator, _validate_unsupported_params

    e = VanillaLDAEstimator(optimizer="em")
    with pytest.raises(ValueError, match="optimizer"):
        _validate_unsupported_params(e)


def test_optimize_doc_concentration_true_raises():
    from spark_vi.mllib.lda import VanillaLDAEstimator, _validate_unsupported_params

    e = VanillaLDAEstimator(optimizeDocConcentration=True)
    with pytest.raises(ValueError, match="optimizeDocConcentration"):
        _validate_unsupported_params(e)


def test_vector_doc_concentration_raises():
    from spark_vi.mllib.lda import VanillaLDAEstimator, _validate_unsupported_params

    e = VanillaLDAEstimator(k=3, docConcentration=[0.1, 0.1, 0.1])
    with pytest.raises(ValueError, match="docConcentration"):
        _validate_unsupported_params(e)


def test_scalar_doc_concentration_is_accepted():
    """A length-1 list (what toListFloat does to a scalar) must not raise."""
    from spark_vi.mllib.lda import VanillaLDAEstimator, _validate_unsupported_params

    e = VanillaLDAEstimator(docConcentration=[0.1])
    _validate_unsupported_params(e)  # should not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest spark-vi/tests/test_mllib_lda.py -v -k "unsupported or optimize or doc_concentration"`
Expected: FAIL with `ImportError: cannot import name '_validate_unsupported_params'`.

- [ ] **Step 3: Implement the validator**

Add to `spark-vi/spark_vi/mllib/lda.py`:

```python
def _validate_unsupported_params(estimator: "VanillaLDAEstimator") -> None:
    """Raise ValueError for any configuration the shim cannot honor.

    Three cases (per ADR 0008 / ADR 0009):
      * optimizer != "online" — we are SVI-only.
      * optimizeDocConcentration=True — symmetric-alpha-only.
      * vector docConcentration (length > 1) — symmetric-alpha-only.

    Silent fallback would mislead users about what they are getting.
    """
    optimizer = estimator.getOrDefault("optimizer")
    if optimizer != "online":
        raise ValueError(
            f"VanillaLDAEstimator only supports optimizer='online', got {optimizer!r}. "
            f"The 'em' optimizer is not implemented in this shim."
        )

    if estimator.getOrDefault("optimizeDocConcentration"):
        raise ValueError(
            "VanillaLDAEstimator does not support optimizeDocConcentration=True. "
            "Empirical-Bayes alpha optimization is deferred per ADR 0008 'Future work'. "
            "Set optimizeDocConcentration=False (the default) and pass a fixed alpha "
            "via docConcentration."
        )

    if estimator.isSet("docConcentration"):
        doc_conc = estimator.getOrDefault("docConcentration")
        if doc_conc is not None and len(doc_conc) > 1:
            raise ValueError(
                f"VanillaLDAEstimator only supports symmetric (scalar) docConcentration, "
                f"got vector of length {len(doc_conc)}. Asymmetric alpha is deferred per "
                f"ADR 0008 'Future work'."
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest spark-vi/tests/test_mllib_lda.py -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/mllib/lda.py spark-vi/tests/test_mllib_lda.py
git commit -m "mllib shim: reject em / optimizeDocConcentration / vector alpha"
```

---

## Task 6: Implement `_fit` end-to-end

Build the model and config, materialize the BOW RDD from the DataFrame, run `VIRunner.fit`, return a `VanillaLDAModel` carrying the trained `VIResult` and a copy of every Param.

**Files:**
- Modify: `spark-vi/spark_vi/mllib/lda.py`
- Test: `spark-vi/tests/test_mllib_lda.py`

- [ ] **Step 1: Write the failing test (and a fixture)**

```python
# Append to spark-vi/tests/test_mllib_lda.py
@pytest.fixture(scope="module")
def tiny_corpus_df(spark):
    """3-topic well-separated corpus, ~30 docs, vocab size 9.

    Topic 0 favors words 0,1,2; topic 1 favors 3,4,5; topic 2 favors 6,7,8.
    Each doc is a near-mixture of one topic.
    """
    from pyspark.ml.linalg import Vectors

    rng = np.random.default_rng(0)
    rows = []
    favored = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}
    for doc_id in range(30):
        topic = doc_id % 3
        counts = np.zeros(9, dtype=np.float64)
        for w in rng.choice(favored[topic], size=20, replace=True):
            counts[w] += 1.0
        # add a little noise to other vocab so vectors aren't identical
        for w in rng.choice(9, size=2, replace=True):
            counts[w] += 1.0
        rows.append((Vectors.dense(counts.tolist()),))
    return spark.createDataFrame(rows, schema=["features"])


def test_fit_returns_model_with_correct_shape(tiny_corpus_df):
    from spark_vi.mllib.lda import VanillaLDAEstimator, VanillaLDAModel

    estimator = VanillaLDAEstimator(k=3, maxIter=5, seed=0, subsamplingRate=1.0)
    model = estimator.fit(tiny_corpus_df)

    assert isinstance(model, VanillaLDAModel)
    assert model.vocabSize() == 9
    # Param round-trip: model exposes the same configuration the Estimator had.
    assert model.getOrDefault("k") == 3
    assert model.getOrDefault("maxIter") == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest spark-vi/tests/test_mllib_lda.py::test_fit_returns_model_with_correct_shape -v`
Expected: FAIL with `NotImplementedError` (from the `_fit` stub) or `AttributeError: 'VanillaLDAModel' has no attribute 'vocabSize'`.

- [ ] **Step 3: Implement `_fit` and the Model's Param surface + `vocabSize`**

Add the same Param declarations to `VanillaLDAModel` (it must be a `Params` instance with the same surface so getters work post-fit), implement `_fit`, and add `vocabSize`. Replace the `VanillaLDAModel` stub:

```python
# Replace the VanillaLDAModel stub in spark-vi/spark_vi/mllib/lda.py
class VanillaLDAModel(Model, HasFeaturesCol, HasMaxIter, HasSeed):
    """MLlib-shaped Model wrapping a trained spark_vi VIResult.

    Carries the trained global parameters plus a copy of every Param from
    the Estimator that produced it, so post-fit getters (model.getK(), ...)
    return the configuration that was actually used.
    """

    # Same Params as the Estimator, declared again so they live on this class.
    k = VanillaLDAEstimator.k
    topicDistributionCol = VanillaLDAEstimator.topicDistributionCol
    optimizer = VanillaLDAEstimator.optimizer
    learningOffset = VanillaLDAEstimator.learningOffset
    learningDecay = VanillaLDAEstimator.learningDecay
    subsamplingRate = VanillaLDAEstimator.subsamplingRate
    docConcentration = VanillaLDAEstimator.docConcentration
    topicConcentration = VanillaLDAEstimator.topicConcentration
    optimizeDocConcentration = VanillaLDAEstimator.optimizeDocConcentration
    gammaShape = VanillaLDAEstimator.gammaShape
    caviMaxIter = VanillaLDAEstimator.caviMaxIter
    caviTol = VanillaLDAEstimator.caviTol

    def __init__(self, result) -> None:  # result: VIResult
        super().__init__()
        self._result = result

    @property
    def result(self):
        """The trained VIResult (global_params, elbo_trace, n_iterations, ...)."""
        return self._result

    def vocabSize(self) -> int:
        """V dimension of the trained lambda."""
        return int(self._result.global_params["lambda"].shape[1])

    def _transform(self, dataset):
        raise NotImplementedError("Implemented in a later task.")
```

Then implement `_fit` on `VanillaLDAEstimator`. Replace its `_fit` stub with:

```python
    def _fit(self, dataset) -> "VanillaLDAModel":
        from spark_vi.core.runner import VIRunner

        _validate_unsupported_params(self)

        first_features = dataset.select(self.getOrDefault("featuresCol")).head(1)
        if not first_features:
            raise ValueError("Cannot fit on an empty DataFrame.")
        vocab_size = first_features[0][0].size

        model_obj, config = _build_model_and_config(self, vocab_size=vocab_size)

        features_col = self.getOrDefault("featuresCol")
        bow_rdd = (
            dataset.select(features_col).rdd
            .map(lambda row: _vector_to_bow_document(row[0]))
        )

        runner = VIRunner(model_obj, config=config)
        result = runner.fit(bow_rdd)

        out_model = VanillaLDAModel(result)
        # Copy every Param value the Estimator has set or has a default for, so
        # the Model's getters reflect the configuration that produced it.
        for param in self.params:
            if self.isSet(param):
                out_model._set(**{param.name: self.getOrDefault(param)})
            elif self.hasDefault(param):
                out_model._setDefault(**{param.name: self.getOrDefault(param)})
        return out_model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest spark-vi/tests/test_mllib_lda.py::test_fit_returns_model_with_correct_shape -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/mllib/lda.py spark-vi/tests/test_mllib_lda.py
git commit -m "mllib shim: implement _fit returning VanillaLDAModel"
```

---

## Task 7: `topicsMatrix()`

Return the (V × K) row-normalized topic-word matrix as an MLlib `DenseMatrix`. Internally we store λ as (K × V); transpose and normalize.

**Files:**
- Modify: `spark-vi/spark_vi/mllib/lda.py`
- Test: `spark-vi/tests/test_mllib_lda.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to spark-vi/tests/test_mllib_lda.py
def test_topics_matrix_shape_and_normalization(tiny_corpus_df):
    from pyspark.ml.linalg import DenseMatrix
    from spark_vi.mllib.lda import VanillaLDAEstimator

    estimator = VanillaLDAEstimator(k=3, maxIter=5, seed=0, subsamplingRate=1.0)
    model = estimator.fit(tiny_corpus_df)

    tm = model.topicsMatrix()
    assert isinstance(tm, DenseMatrix)
    assert tm.numRows == 9   # vocab size V
    assert tm.numCols == 3   # K
    # Each column (a topic) sums to 1 (row-stochastic over vocab in MLlib's orientation).
    arr = tm.toArray()
    np.testing.assert_allclose(arr.sum(axis=0), 1.0, atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest spark-vi/tests/test_mllib_lda.py::test_topics_matrix_shape_and_normalization -v`
Expected: FAIL with `AttributeError: 'VanillaLDAModel' object has no attribute 'topicsMatrix'`.

- [ ] **Step 3: Implement `topicsMatrix`**

Add to `VanillaLDAModel` in `spark-vi/spark_vi/mllib/lda.py`:

```python
    def topicsMatrix(self):
        """Topic-word distribution as an MLlib DenseMatrix of shape (V, K).

        Internally we keep lambda as (K, V); the transpose-and-normalize
        here matches MLlib's convention where `topicsMatrix` is indexed
        by (vocab term, topic).
        """
        from pyspark.ml.linalg import DenseMatrix

        lam = self._result.global_params["lambda"]
        beta = lam / lam.sum(axis=1, keepdims=True)  # (K, V), row-stochastic
        K, V = beta.shape
        # DenseMatrix expects column-major flattened values.
        return DenseMatrix(numRows=V, numCols=K, values=beta.T.flatten("F").tolist())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest spark-vi/tests/test_mllib_lda.py::test_topics_matrix_shape_and_normalization -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/mllib/lda.py spark-vi/tests/test_mllib_lda.py
git commit -m "mllib shim: topicsMatrix returns (V x K) DenseMatrix"
```

---

## Task 8: `describeTopics()`

Return a DataFrame with the top-`maxTermsPerTopic` terms per topic.

**Files:**
- Modify: `spark-vi/spark_vi/mllib/lda.py`
- Test: `spark-vi/tests/test_mllib_lda.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to spark-vi/tests/test_mllib_lda.py
def test_describe_topics_returns_top_k_per_topic(tiny_corpus_df):
    from spark_vi.mllib.lda import VanillaLDAEstimator

    estimator = VanillaLDAEstimator(k=3, maxIter=5, seed=0, subsamplingRate=1.0)
    model = estimator.fit(tiny_corpus_df)

    df = model.describeTopics(maxTermsPerTopic=4)
    rows = df.orderBy("topic").collect()

    assert [r["topic"] for r in rows] == [0, 1, 2]
    for r in rows:
        assert len(r["termIndices"]) == 4
        assert len(r["termWeights"]) == 4
        # Weights must be descending.
        weights = list(r["termWeights"])
        assert weights == sorted(weights, reverse=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest spark-vi/tests/test_mllib_lda.py::test_describe_topics_returns_top_k_per_topic -v`
Expected: FAIL with `AttributeError: 'VanillaLDAModel' object has no attribute 'describeTopics'`.

- [ ] **Step 3: Implement `describeTopics`**

Add to `VanillaLDAModel`:

```python
    def describeTopics(self, maxTermsPerTopic: int = 10):
        """DataFrame of (topic, termIndices, termWeights) — top terms per topic.

        Schema and orientation match pyspark.ml.clustering.LDAModel.describeTopics.
        """
        from pyspark.sql import SparkSession
        from pyspark.sql.types import (
            ArrayType, DoubleType, IntegerType, StructField, StructType,
        )

        if maxTermsPerTopic < 1:
            raise ValueError(f"maxTermsPerTopic must be >= 1, got {maxTermsPerTopic}")

        lam = self._result.global_params["lambda"]
        beta = lam / lam.sum(axis=1, keepdims=True)  # (K, V), row-stochastic
        K, V = beta.shape
        m = min(maxTermsPerTopic, V)

        rows = []
        for k in range(K):
            order = np.argsort(beta[k])[::-1][:m]
            rows.append((
                int(k),
                [int(i) for i in order],
                [float(beta[k, i]) for i in order],
            ))

        schema = StructType([
            StructField("topic", IntegerType(), False),
            StructField("termIndices", ArrayType(IntegerType(), False), False),
            StructField("termWeights", ArrayType(DoubleType(), False), False),
        ])
        return SparkSession.builder.getOrCreate().createDataFrame(rows, schema=schema)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest spark-vi/tests/test_mllib_lda.py::test_describe_topics_returns_top_k_per_topic -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/mllib/lda.py spark-vi/tests/test_mllib_lda.py
git commit -m "mllib shim: describeTopics with MLlib-shaped schema"
```

---

## Task 9: `_transform` via UDF

Add a `topicDistribution` Vector column to the input DataFrame using a UDF that runs CAVI per row.

**Files:**
- Modify: `spark-vi/spark_vi/mllib/lda.py`
- Test: `spark-vi/tests/test_mllib_lda.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to spark-vi/tests/test_mllib_lda.py
def test_transform_adds_topic_distribution_column(tiny_corpus_df):
    from pyspark.ml.linalg import Vector
    from spark_vi.mllib.lda import VanillaLDAEstimator

    estimator = VanillaLDAEstimator(k=3, maxIter=5, seed=0, subsamplingRate=1.0)
    model = estimator.fit(tiny_corpus_df)

    out = model.transform(tiny_corpus_df)
    assert "topicDistribution" in out.columns

    rows = out.select("topicDistribution").collect()
    for r in rows:
        td = r["topicDistribution"]
        assert isinstance(td, Vector)
        arr = np.asarray(td.toArray())
        assert arr.shape == (3,)
        np.testing.assert_allclose(arr.sum(), 1.0, atol=1e-6)


def test_transform_respects_custom_topic_distribution_col(tiny_corpus_df):
    from spark_vi.mllib.lda import VanillaLDAEstimator

    estimator = VanillaLDAEstimator(
        k=3, maxIter=5, seed=0, subsamplingRate=1.0,
        topicDistributionCol="theta",
    )
    model = estimator.fit(tiny_corpus_df)
    out = model.transform(tiny_corpus_df)
    assert "theta" in out.columns
    assert "topicDistribution" not in out.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest spark-vi/tests/test_mllib_lda.py -v -k "transform_adds or transform_respects"`
Expected: FAIL with `NotImplementedError` from the `_transform` stub.

- [ ] **Step 3: Implement `_transform`**

Replace the `_transform` stub in `VanillaLDAModel` with:

```python
    def _transform(self, dataset):
        from pyspark.ml.linalg import DenseVector, VectorUDT
        from pyspark.sql import functions as F
        from pyspark.sql.types import StructType
        from scipy.special import digamma

        from spark_vi.models.lda import _cavi_doc_inference

        lam = self._result.global_params["lambda"]
        expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))

        alpha = float(self.getOrDefault("docConcentration")[0]) \
            if self.isSet("docConcentration") and self.getOrDefault("docConcentration") is not None \
            else 1.0 / self.getOrDefault("k")
        gamma_shape = float(self.getOrDefault("gammaShape"))
        cavi_max_iter = int(self.getOrDefault("caviMaxIter"))
        cavi_tol = float(self.getOrDefault("caviTol"))
        K = expElogbeta.shape[0]

        sc = dataset.sparkSession.sparkContext
        bcast = sc.broadcast({
            "expElogbeta": expElogbeta,
            "alpha": alpha,
            "gamma_shape": gamma_shape,
            "cavi_max_iter": cavi_max_iter,
            "cavi_tol": cavi_tol,
            "K": K,
        })

        def _infer(features):
            params = bcast.value
            doc = _vector_to_bow_document(features)
            rng = np.random.default_rng()
            gamma_init = rng.gamma(
                shape=params["gamma_shape"],
                scale=1.0 / params["gamma_shape"],
                size=params["K"],
            )
            gamma, _, _, _ = _cavi_doc_inference(
                indices=doc.indices,
                counts=doc.counts,
                expElogbeta=params["expElogbeta"],
                alpha=params["alpha"],
                gamma_init=gamma_init,
                max_iter=params["cavi_max_iter"],
                tol=params["cavi_tol"],
            )
            return DenseVector(gamma / gamma.sum())

        infer_udf = F.udf(_infer, returnType=VectorUDT())

        try:
            out_col = self.getOrDefault("topicDistributionCol")
            features_col = self.getOrDefault("featuresCol")
            return dataset.withColumn(out_col, infer_udf(F.col(features_col)))
        finally:
            bcast.unpersist(blocking=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest spark-vi/tests/test_mllib_lda.py -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/mllib/lda.py spark-vi/tests/test_mllib_lda.py
git commit -m "mllib shim: _transform via UDF emitting topicDistribution column"
```

---

## Task 10: `logLikelihood` / `logPerplexity` stubs

Stub with `NotImplementedError` and a clear pointer to the training-time ELBO trace.

**Files:**
- Modify: `spark-vi/spark_vi/mllib/lda.py`
- Test: `spark-vi/tests/test_mllib_lda.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to spark-vi/tests/test_mllib_lda.py
def test_log_likelihood_and_log_perplexity_raise_not_implemented(tiny_corpus_df):
    from spark_vi.mllib.lda import VanillaLDAEstimator

    estimator = VanillaLDAEstimator(k=3, maxIter=5, seed=0, subsamplingRate=1.0)
    model = estimator.fit(tiny_corpus_df)

    with pytest.raises(NotImplementedError, match="ELBO"):
        model.logLikelihood(tiny_corpus_df)
    with pytest.raises(NotImplementedError, match="ELBO"):
        model.logPerplexity(tiny_corpus_df)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest spark-vi/tests/test_mllib_lda.py::test_log_likelihood_and_log_perplexity_raise_not_implemented -v`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement the stubs**

Add to `VanillaLDAModel`:

```python
    def logLikelihood(self, dataset):
        raise NotImplementedError(
            "logLikelihood is not implemented in this v1 shim. The training-time "
            "ELBO trace is available on the underlying VIResult via "
            "VanillaLDAModel.result.elbo_trace."
        )

    def logPerplexity(self, dataset):
        raise NotImplementedError(
            "logPerplexity is not implemented in this v1 shim. The training-time "
            "ELBO trace is available on the underlying VIResult via "
            "VanillaLDAModel.result.elbo_trace."
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest spark-vi/tests/test_mllib_lda.py -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/mllib/lda.py spark-vi/tests/test_mllib_lda.py
git commit -m "mllib shim: stub logLikelihood / logPerplexity"
```

---

## Task 11: Rewrite `run_ours` to use the shim

The comparison driver becomes "fit two MLlib-shaped Estimators with matched params, compare outputs."

**Files:**
- Modify: `charmpheno/charmpheno/evaluate/lda_compare.py`

- [ ] **Step 1: Inspect the current `run_ours` and the slow parity test**

Run: `git show HEAD:charmpheno/charmpheno/evaluate/lda_compare.py | head -80`
Run: `grep -n "run_ours" charmpheno/tests/test_lda_compare.py charmpheno/charmpheno/evaluate/lda_compare.py analysis/local/compare_lda_local.py`
Expected: `run_ours` is called with positional args `(rdd, vocab_size, K, config: VIConfig)` from those callers.

- [ ] **Step 2: Rewrite `run_ours`**

Replace the body of `run_ours` in `charmpheno/charmpheno/evaluate/lda_compare.py` with:

```python
def run_ours(
    rdd: RDD,
    vocab_size: int,
    K: int,
    config: VIConfig,
) -> LDARunArtifacts:
    """Fit VanillaLDA via the MLlib-shaped shim; collect artifacts.

    Wraps the input RDD[BOWDocument] back into a DataFrame[Vector] so the
    shim's DataFrame-shaped API accepts it. Net effect: this function is
    now structurally symmetric with run_mllib (both fit MLlib-shaped
    Estimators on a DataFrame), tightening the head-to-head comparison.
    """
    from pyspark.ml.linalg import Vectors, VectorUDT
    from pyspark.sql.types import StructField, StructType
    from pyspark.sql import SparkSession

    from spark_vi.mllib.lda import VanillaLDAEstimator

    spark = SparkSession.builder.getOrCreate()

    def _bow_to_row(doc):
        # Sparse vector with the doc's indices and counts.
        return (Vectors.sparse(
            vocab_size,
            [int(i) for i in doc.indices],
            [float(c) for c in doc.counts],
        ),)

    schema = StructType([StructField("features", VectorUDT(), False)])
    df = spark.createDataFrame(rdd.map(_bow_to_row), schema=schema)

    estimator = VanillaLDAEstimator(
        k=K,
        maxIter=config.max_iterations,
        seed=config.random_seed,
        learningOffset=config.learning_rate_tau0,
        learningDecay=config.learning_rate_kappa,
        subsamplingRate=config.mini_batch_fraction or 1.0,
    )

    t0 = time.perf_counter()
    model = estimator.fit(df)
    t1 = time.perf_counter()
    wall = t1 - t0

    n_iter = max(1, model.result.n_iterations)
    per_iter = [wall / n_iter] * n_iter

    tm = model.topicsMatrix().toArray().T  # (K, V), row-stochastic
    tm = tm / tm.sum(axis=1, keepdims=True)

    transformed = model.transform(df).select("topicDistribution").collect()
    prev = np.zeros(K)
    for r in transformed:
        prev += np.asarray(r["topicDistribution"].toArray())

    return LDARunArtifacts(
        topics_matrix=tm,
        topic_prevalence=prev,
        elbo_trace=list(model.result.elbo_trace),
        per_iter_seconds=per_iter,
        wall_time_seconds=wall,
        final_log_likelihood=None,
    )
```

Remove the now-unused imports (`VIRunner`, `VanillaLDA` directly, `BOWDocument`) if no other function in the file references them — keep them if they're still used by other functions.

- [ ] **Step 3: Run the existing fast tests**

Run: `pytest charmpheno/tests/test_lda_compare.py -v -m "not slow"`
Expected: PASS for the smoke tests.

- [ ] **Step 4: Run the slow parity test**

Run: `pytest charmpheno/tests/test_lda_compare.py -v -m slow`
Expected: PASS — `best_diag < 0.20 nats`, with typical observed ~0.01 nats per ADR 0008.

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/evaluate/lda_compare.py
git commit -m "lda_compare: rewrite run_ours on top of the mllib shim"
```

---

## Task 12: ADR 0009

Document the design choices.

**Files:**
- Create: `docs/decisions/0009-mllib-shim.md`

- [ ] **Step 1: Write the ADR**

```markdown
# ADR 0009 — MLlib Estimator/Transformer shim for VanillaLDA

**Status:** Accepted
**Date:** 2026-05-04
**Context:** [docs/superpowers/specs/2026-05-04-mllib-shim-design.md](../superpowers/specs/2026-05-04-mllib-shim-design.md)

## Context

`VanillaLDA` (ADR 0008) is fittable via `VIRunner` on an `RDD[BOWDocument]`.
That API is correct but unfamiliar to anyone coming from MLlib, and the
head-to-head comparison driver
(`charmpheno/charmpheno/evaluate/lda_compare.py`) had to maintain two
non-symmetric run wrappers (`run_ours` for our framework, `run_mllib` for
`pyspark.ml.clustering.LDA`). Both pain points point to the same fix: an
Estimator/Model pair that wraps `VanillaLDA` in the MLlib API shape.

## Decisions

### Subclass `pyspark.ml.base.{Estimator, Model}`

The shim subclasses MLlib's base Estimator and Model (not the concrete
`pyspark.ml.clustering.LDA`). This gives `Pipeline` integration and Param
introspection without committing to mirroring every MLlib LDA Param. We
mirror the surface that maps cleanly to our implementation; the rest is
rejected at fit time with a clear error.

### LDA-specific shim now; defer generic `VIModel` adapter

We write `VanillaLDAEstimator` / `VanillaLDAModel` concretely. When
`OnlineHDP` lands, its second data point will inform whether the abstraction
boundary is real. Designing a generic adapter from one model is likely to
produce an LDA-shaped abstraction with HDP-shaped duct tape later.

### MLlib param names on the shim; our extras camelCased

Shared params use MLlib's names exactly (`k`, `maxIter`, `learningOffset`,
`learningDecay`, `subsamplingRate`, `docConcentration`, `topicConcentration`,
`featuresCol`, `topicDistributionCol`, `optimizer`, `optimizeDocConcentration`).
Our framework-specific extras adopt MLlib's camelCase convention for
consistency on the shim's surface (`gammaShape`, `caviMaxIter`, `caviTol`).
Internal `VanillaLDA` and `VIConfig` keep their snake_case names.

### Vector-column DataFrames in/out

The shim accepts a single Vector column (default `featuresCol="features"`) —
the standard MLlib `CountVectorizer` output shape — and emits a Vector
column (`topicDistributionCol`). It does not duplicate `CountVectorizer`'s
work. Anyone who wants to feed `RDD[BOWDocument]` directly continues to use
`VIRunner` straight.

### Reject unsupported configurations explicitly

Three rejections at fit time, each with a clear message pointing to ADR 0008
where applicable:

- `optimizer != "online"` (we are SVI-only).
- `optimizeDocConcentration=True` (deferred per ADR 0008 future work).
- Vector `docConcentration` (symmetric-α-only per ADR 0008).

Silent fallback would mislead about what users are getting.

### Persistence (`MLReadable` / `MLWritable`) deferred

The driving v1 use case is comparison and Pipeline ergonomics, not
`Pipeline.save()`. When a concrete user appears, the question of which
Param values and which parts of `VIResult` to round-trip becomes specific
instead of speculative. Until then, users persist via `VIResult.export_zip`
and reconstruct.

### `_transform` via Python UDF

The Model's `transform` applies a UDF over the Vector column rather than
routing through `VIRunner.transform`. Reattaching `VIRunner.transform`'s
`RDD[dict]` output to the DataFrame's other columns is awkward; a UDF
preserves all columns trivially and matches MLlib's own LDA transform
implementation pattern.

### `logLikelihood` / `logPerplexity` stubbed

Held-out perplexity for variational LDA requires deriving a held-out ELBO
bound; non-trivial and there is no concrete user. Stubs raise
`NotImplementedError` and point to `VIResult.elbo_trace` for the closest
existing analog.

## Relation to prior ADRs

- [ADR 0007](0007-vimodel-inference-capability.md) —
  `VanillaLDA.infer_local` (the optional capability) is what makes
  `_transform` possible. The shim's UDF is a thin wrapper around the same
  CAVI inner loop.
- [ADR 0008](0008-vanilla-lda-design.md) — the symmetric-α-only constraint
  and the deferral of `optimizeDocConcentration` carry forward unchanged;
  the shim makes those constraints visible at the API boundary.

## Future work

- `MLReadable` / `MLWritable` when a concrete user wants `Pipeline.save()`.
- Generic `VIModel` → MLlib adapter, when OnlineHDP gives us the second
  data point.
- Held-out `logLikelihood` / `logPerplexity`.
- Empirical-Bayes α (the missing path that would let us flip
  `optimizeDocConcentration=True` from rejection to support).
```

- [ ] **Step 2: Verify ADR 0009 references resolve**

Run: `ls docs/superpowers/specs/2026-05-04-mllib-shim-design.md docs/decisions/0007-vimodel-inference-capability.md docs/decisions/0008-vanilla-lda-design.md`
Expected: all three paths print, no errors.

- [ ] **Step 3: Commit**

```bash
git add docs/decisions/0009-mllib-shim.md
git commit -m "ADR 0009: MLlib Estimator/Transformer shim"
```

---

## Task 13: Final verification — full test pass

After all the per-task tests pass individually, verify the full suite is green and no other tests regressed.

**Files:** none modified.

- [ ] **Step 1: Run the full spark-vi test suite**

Run: `pytest spark-vi/tests/ -v`
Expected: all green, including `test_mllib_lda.py`.

- [ ] **Step 2: Run the full charmpheno test suite (without slow tests)**

Run: `pytest charmpheno/tests/ -v -m "not slow"`
Expected: all green.

- [ ] **Step 3: Run the slow parity test as the final source-of-truth gate**

Run: `pytest charmpheno/tests/test_lda_compare.py::test_vanilla_lda_matches_mllib_on_well_separated_corpus -v -m slow`
Expected: PASS — `best_diag < 0.20 nats`.

- [ ] **Step 4: If anything in step 1–3 fails, fix and re-run**

Fix the regression in the relevant module, re-run the failing test, then re-run the full suite.

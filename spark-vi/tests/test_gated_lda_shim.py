import numpy as np
import pytest

pyspark = pytest.importorskip("pyspark")
from pyspark.ml.linalg import Vectors

# `spark` is the session-scoped fixture from tests/conftest.py (Java
# security-manager option + PYSPARK_PYTHON pinning already applied there);
# a bare `local[1]` SparkSession built without those options fails to start
# on newer JDKs, so this file reuses the shared fixture rather than
# redefining its own.


def test_gated_shim_fit_transform_smoke(spark):
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    parent = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    V = 30
    # tiny planted rows: features SparseVector + frontier (array of node ids)
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(40):
        leaf = int(rng.choice([3, 4, 5, 6]))
        idx = sorted(rng.choice(V, size=6, replace=False).tolist())
        rows.append((Vectors.sparse(V, idx, [1.0] * len(idx)), [leaf]))
    df = spark.createDataFrame(rows, ["features", "frontier"])

    est = GatedLDAEstimator(featuresCol="features", labelCol="frontier",
                            parent=parent, nBg=2, tpn=1, maxIter=3, seed=0)
    model = est.fit(df)
    out = model.transform(df)
    assert "nodeAffinity" in out.columns
    aff = out.select("nodeAffinity").head()[0]
    n_nodes = len({1, 2, 3, 4, 5, 6})
    assert len(aff) == n_nodes            # one affinity per DAG node

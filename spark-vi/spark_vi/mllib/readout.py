"""DagHeadsEstimator / DagHeadsModel: the readout's per-node heads as an MLlib stage
(ADR 0051 §4).

The heads read any vector column (θ from a topic model, raw-code presence, any
representation) and a multi-label target with its observation mask, and fit one L2
logistic head per label-DAG node with the batched solver (`spark_vi.readout.fit_heads`).
`transform` appends each row's per-node probabilities and, given the DAG, the closure
product (`readout.stacked`), so

    Pipeline(stages=[DagGatedLDAEstimator(...), DagHeadsEstimator(...)])

is the end-to-end case-finding model. A thin shim: fitting, scoring and the heads
file are the readout's own; this module only maps Params onto them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pyspark.ml import Estimator, Model
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasFeaturesCol, HasMaxIter

from spark_vi.readout.heads import READOUT_GTOL, READOUT_MAX_ITER

_DEFAULTS = dict(featuresCol="topicDistribution", labelCol="label",
                 labelMaskCol="labelMask", probabilityCol="nodeProbability",
                 stackedProbabilityCol="", regParam=1.0, maxIter=READOUT_MAX_ITER,
                 tol=READOUT_GTOL, thetaTopM=0, engine="spark", dagParent="")


class _DagHeadsParams(HasFeaturesCol, HasMaxIter):
    labelCol = Param(Params._dummy(), "labelCol",
                     "Array column of per-node 0/1 labels (length C).",
                     typeConverter=TypeConverters.toString)
    labelMaskCol = Param(Params._dummy(), "labelMaskCol",
                         "Array column of per-node 0/1 observation flags: node c's head "
                         "is fit on the rows where it is 1.",
                         typeConverter=TypeConverters.toString)
    probabilityCol = Param(Params._dummy(), "probabilityCol",
                           "Output vector column: P(node | row) from each node's head.",
                           typeConverter=TypeConverters.toString)
    stackedProbabilityCol = Param(
        Params._dummy(), "stackedProbabilityCol",
        "Output vector column of the closure product (empty = not computed; "
        "needs dagParent).", typeConverter=TypeConverters.toString)
    regParam = Param(Params._dummy(), "regParam", "L2 penalty of every head.",
                     typeConverter=TypeConverters.toFloat)
    tol = Param(Params._dummy(), "tol", "Gradient tolerance of the batched L-BFGS.",
                typeConverter=TypeConverters.toFloat)
    thetaTopM = Param(Params._dummy(), "thetaTopM",
                      "Keep each row's top-m features (0 = all), in fit and transform.",
                      typeConverter=TypeConverters.toInt)
    engine = Param(Params._dummy(), "engine",
                   "'spark' (a treeAggregate per pass) or 'local' (the split held "
                   "in driver memory).", typeConverter=TypeConverters.toString)
    dagParent = Param(Params._dummy(), "dagParent",
                      "JSON {child: [parents]} over node ids 0..C-1 (for the "
                      "closure product).", typeConverter=TypeConverters.toString)

    def setDagParent(self, parent):
        """Set the label DAG from a `{child: parent or [parents]}` dict."""
        norm = {str(int(c)): ([int(p) for p in ps] if isinstance(ps, (list, tuple))
                              else [int(ps)]) for c, ps in parent.items()}
        return self._set(dagParent=json.dumps(norm, sort_keys=True))

    def getDagParent(self):
        s = self.getOrDefault("dagParent")
        return {int(c): ps for c, ps in json.loads(s).items()} if s else None


class DagHeadsEstimator(_DagHeadsParams, Estimator):
    """Fit the per-node heads; returns a `DagHeadsModel`."""

    def __init__(self, *, featuresCol="topicDistribution", labelCol="label",
                 labelMaskCol="labelMask", probabilityCol="nodeProbability",
                 stackedProbabilityCol="", regParam=1.0, maxIter=READOUT_MAX_ITER,
                 tol=READOUT_GTOL, thetaTopM=0, engine="spark", dagParent=""):
        super().__init__()
        self._setDefault(**_DEFAULTS)
        self._set(featuresCol=featuresCol, labelCol=labelCol, labelMaskCol=labelMaskCol,
                  probabilityCol=probabilityCol,
                  stackedProbabilityCol=stackedProbabilityCol, regParam=regParam,
                  maxIter=maxIter, tol=tol, thetaTopM=thetaTopM, engine=engine,
                  dagParent=dagParent)

    def _fit(self, dataset):
        from pyspark.sql import functions as F

        from spark_vi.readout import LocalReadoutData, SparkReadoutData, fit_heads

        g = self.getOrDefault
        fc, lc, mc = g("featuresCol"), g("labelCol"), g("labelMaskCol")
        first = dataset.select(fc, lc).head()
        if first is None:
            raise ValueError("DagHeadsEstimator: empty dataset")
        K, C = int(first[0].size), len(first[1])
        cols = dict(topic_col=fc, label_col=lc, mask_col=mc)
        topm = int(g("thetaTopM"))
        if g("engine") == "local":
            df = dataset.withColumn("_row_id", F.monotonically_increasing_id())
            data = LocalReadoutData.from_spark(df, C, K, topm=topm, id_col="_row_id",
                                               **cols)
        elif g("engine") == "spark":
            data = SparkReadoutData(dataset, C, K, topm=topm, **cols)
        else:
            raise ValueError(f"engine must be 'spark' or 'local', got {g('engine')!r}")
        V, b_raw, const, degenerate, info = fit_heads(
            data, l2=float(g("regParam")), gtol=float(g("tol")),
            max_iter=int(g("maxIter")), label="DagHeads")
        model = DagHeadsModel(V, b_raw, const, degenerate, theta_topm=topm, info=info)
        return self._copyValues(model)


class DagHeadsModel(_DagHeadsParams, Model):
    """The fitted heads: raw-feature coefficients `V (C, K)`, `b_raw (C,)` and the
    constant fallback of nodes whose observed training rows were single-class."""

    def __init__(self, V, b_raw, const, degenerate, *, theta_topm=0, info=None):
        super().__init__()
        self._setDefault(**_DEFAULTS)
        self.V = np.asarray(V, dtype=np.float64)
        self.b_raw = np.asarray(b_raw, dtype=np.float64)
        self.const = np.asarray(const, dtype=np.float64)
        self.degenerate = np.asarray(degenerate, dtype=bool)
        self.theta_topm = int(theta_topm)
        self.info = info or {}

    @property
    def numNodes(self) -> int:
        return int(self.V.shape[0])

    def _transform(self, dataset):
        from pyspark.ml.linalg import DenseVector, VectorUDT
        from pyspark.sql import functions as F

        from spark_vi.readout.kernels import _SCORE_CLIP, to_array, topm_sparse

        g = self.getOrDefault
        sc = dataset.sparkSession.sparkContext
        parent = self.getDagParent()
        stacked_col = g("stackedProbabilityCol")
        if stacked_col and parent is None:
            raise ValueError("stackedProbabilityCol needs dagParent")
        M = None
        if stacked_col:
            from spark_vi.dag.ops import closure_matrix
            M = closure_matrix(parent, self.numNodes).astype(np.float32)
        # Nothing array-shaped rides a closure (ADR 0047): one broadcast, whose
        # lifetime is the returned DataFrame's (the ContextCleaner reclaims it).
        bc = sc.broadcast({"V": self.V, "b": self.b_raw, "deg": self.degenerate,
                           "const": self.const.astype(np.float32),
                           "topm": self.theta_topm, "M": M})

        def _score(v):
            p_ = bc.value
            x = to_array(v)
            if p_["topm"] > 0:
                idx, val = topm_sparse(x, p_["topm"])
                z = p_["V"][:, idx] @ val + p_["b"]
            else:
                z = p_["V"] @ x + p_["b"]
            p = (1.0 / (1.0 + np.exp(-np.clip(z, -_SCORE_CLIP, _SCORE_CLIP)))
                 ).astype(np.float32)
            p[p_["deg"]] = p_["const"][p_["deg"]]
            return p

        out = dataset.withColumn(
            g("probabilityCol"),
            F.udf(lambda v: DenseVector(_score(v).astype(np.float64)), VectorUDT())(
                F.col(g("featuresCol"))))
        if stacked_col:
            from spark_vi.readout.stacked import stacked_proba

            def _stack(v):
                return DenseVector(stacked_proba(
                    to_array(v)[None, :], bc.value["M"])[0].astype(np.float64))
            out = out.withColumn(stacked_col, F.udf(_stack, VectorUDT())(
                F.col(g("probabilityCol"))))
        return out

    def save(self, path):
        """Write the heads (`readout.heads.write_heads`) and the Params to `path`."""
        from spark_vi.readout.heads import write_heads

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        C, K = self.V.shape
        if not write_heads(path, "model", self.V, self.b_raw, self.const,
                           self.degenerate, C, K, self.theta_topm):
            raise OSError(f"could not write the heads to {path}")
        params = {p.name: self.getOrDefault(p) for p in self.params
                  if self.hasDefault(p) or self.isSet(p)}
        (path / "params.json").write_text(json.dumps(
            {"model_class": "DagHeadsModel", "params": params}, sort_keys=True))

    @classmethod
    def load(cls, path):
        from spark_vi.readout.heads import read_heads

        path = Path(path)
        meta = json.loads((path / "params.json").read_text())
        if meta.get("model_class") != "DagHeadsModel":
            raise ValueError(f"{path} holds a {meta.get('model_class')!r}, not a "
                             "DagHeadsModel")
        got = read_heads(path, "model")
        if got is None:
            raise ValueError(f"no readable heads in {path}")
        V, b_raw, const, degenerate, _C, _K, topm = got
        model = cls(V, b_raw, const, degenerate, theta_topm=topm)
        model._set(**meta["params"])
        return model

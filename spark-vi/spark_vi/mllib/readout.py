"""DagHeadsEstimator / DagHeadsModel: the readout's per-node heads as an MLlib stage
(ADR 0051 §4; user API, spec 2026-10-30 R6, R8, R13).

The heads read any vector column (theta from a topic model, raw-code presence, any
representation) and fit one L2 logistic head per node of a label DAG with the batched
solver (`spark_vi.readout.fit_heads`). The inputs are the DAG-gated estimator's, in
the user's own node ids:

- ``labelsCol``: an ``array<string>`` / ``array<int>`` column of the nodes each
  document attests (ancestors and duplicates allowed; empty or null = background);
- EITHER a DAG (``setDag``, parsed and validated as the topic estimator does; the
  node set is learned from the fit DataFrame with the same ``minDocsPerNode`` /
  ``unknownLabels`` rules: support, prune, land, reduce, splice) OR a fitted
  ``DagGatedLDAModel`` (``setTopicModel``), whose node set (kept DAG, id map, label
  landing) is reused so head i and the model's engine node i are the same node.

The per-node targets are built here (`spark_vi.readout.labels.NodeLabeler`): a
document is positive for every kept node in the closure of its landed labels, and
``labelMaskMode`` picks which nodes it is evidence for ('full' or 'closure'). Gate
labels and readout labels may differ (R8): each estimator has its own ``labelsCol``.

Output index space: index 0 is the synthetic root (the "any label" head), indices
1..N the kept nodes in ``nodes()`` order; ``nodeIndex(node)`` maps a node to its
index. ``transform`` appends ``probabilityCol`` (a length N+1 vector) and, when
``stackedProbabilityCol`` is set, the closure product over the kept DAG
(`readout.stacked`). Nodes whose observed training rows are empty or single-class
are predicted as a constant and reported by ``degenerateNodes()``.

Persistence (R13): ``saveDir`` / ``saveInterval`` / ``resumeFrom`` as in the topic
shims, on any Hadoop-filesystem URI (`spark_vi.io.fs`). Both stages are MLlib
``MLWritable`` / ``MLReadable`` (``save`` / ``write().overwrite().save`` / ``load``,
and inside a ``Pipeline`` / ``PipelineModel``): Params metadata plus data files, the
metadata written last. The model keeps the heads and the node set; the estimator
keeps its Params and, when it holds one (``setTopicModel``), the topic model. Where a
save may go: single stages save / load on any path, a driver-local one included (their
metadata is written and read on the driver); ``Pipeline.save`` /
``PipelineModel.load`` use MLlib's own executor-side metadata I/O, so on a multi-node
cluster a Pipeline needs a shared filesystem (HDFS, GCS), not a driver-local path.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
from pyspark import keyword_only
from pyspark.ml import Estimator, Model
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasFeaturesCol, HasMaxIter

from spark_vi.dag.nodeset import (ROOT, UNKNOWN_LABEL_POLICIES, LabelIndex, NodeSet,
                                  SupportCounts, _finish, _partition_support,
                                  count_support_spark, learn_node_set)
from spark_vi.mllib._common import (_PersistenceParams, _StagePersistence,
                                    save_stage_in_place)
from spark_vi.mllib.topic.dag_gated_lda import (_decode_dag, _encode_dag,
                                                check_labels_column)
from spark_vi.readout.heads import READOUT_GTOL, READOUT_MAX_ITER
from spark_vi.readout.labels import LABEL_MASK_MODES, NodeLabeler

_LOG = "[dag-heads]"
_CKPT_NAME = "heads_ckpt.npz"          # the mid-solve checkpoint inside saveDir
_HEADS_LABEL = "model"                 # write_heads' arm label for the saved model
_TARGET_COLS = ("__dag_heads_label", "__dag_heads_mask")

_DEFAULTS = dict(featuresCol="topicDistribution", labelsCol="labels", dagEdges="",
                 minDocsPerNode=1, unknownLabels="error", labelMaskMode="full",
                 probabilityCol="nodeProbability", stackedProbabilityCol="",
                 regParam=1.0, maxIter=READOUT_MAX_ITER, tol=READOUT_GTOL,
                 thetaTopM=0, engine="spark", saveInterval=-1, saveDir="",
                 resumeFrom="")


class _DagHeadsParams(HasFeaturesCol, HasMaxIter, _PersistenceParams):
    labelsCol = Param(
        Params._dummy(), "labelsCol",
        "array<string> or array<int> column of the DAG nodes each document attests "
        "(any nodes: ancestors and duplicates allowed; empty or null = background). "
        "The element type must match the DAG's ids. Default 'labels'.",
        typeConverter=TypeConverters.toString)
    dagEdges = Param(
        Params._dummy(), "dagEdges",
        "the label DAG, set with setDag (a dict {child: parent | [parents]}, or a "
        "pandas / Spark DataFrame (child, parent)); stored validated as a JSON edge "
        "list in the user's ids. Set this OR a fitted topic model (setTopicModel).",
        typeConverter=TypeConverters.toString)
    minDocsPerNode = Param(
        Params._dummy(), "minDocsPerNode",
        "with a DAG: a node gets a head only if at least this many training "
        "documents' label closures reach it; a pruned node's labels land on its "
        "nearest kept ancestors. Default 1. Not settable with a topic model (its "
        "node set is used as is).",
        typeConverter=TypeConverters.toInt)
    unknownLabels = Param(
        Params._dummy(), "unknownLabels",
        "a label that is not a DAG node: 'error' (default; refused by name) or "
        "'ignore' (dropped; a document left with no label is background).",
        typeConverter=TypeConverters.toString)
    labelMaskMode = Param(
        Params._dummy(), "labelMaskMode",
        "which nodes a document is evidence for: 'full' (default; every node, a "
        "background document is a negative everywhere) or 'closure' (its label "
        "closure and the DAG siblings of each positive; a background document "
        "observes nothing).",
        typeConverter=TypeConverters.toString)
    probabilityCol = Param(
        Params._dummy(), "probabilityCol",
        "output vector column: P(node | row) from each head, index 0 the root "
        "('any label'), index nodeIndex(node) each kept node.",
        typeConverter=TypeConverters.toString)
    stackedProbabilityCol = Param(
        Params._dummy(), "stackedProbabilityCol",
        "output vector column of the closure product over the kept DAG (empty = not "
        "computed).", typeConverter=TypeConverters.toString)
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
    # The persistence Params, with the heads' meaning (same names and rules as the
    # topic shims': saveInterval > 0 needs saveDir; 0 is refused).
    saveInterval = Param(
        Params._dummy(), "saveInterval",
        "write a resumable solver checkpoint into saveDir every N L-BFGS iterations. "
        "-1 (default) = no mid-fit checkpoint; the checkpoint is removed once the "
        "solve finishes.", typeConverter=TypeConverters.toInt)
    saveDir = Param(
        Params._dummy(), "saveDir",
        "directory or Hadoop-filesystem URI the fitted model is saved to at the end "
        "of fit (DagHeadsModel.load reads it; written in place: other files there "
        "are kept, an earlier model replaced); also where saveInterval checkpoints "
        "go. Empty (default) = nothing written.",
        typeConverter=TypeConverters.toString)
    resumeFrom = Param(
        Params._dummy(), "resumeFrom",
        "a saveDir of an earlier fit: resume from its solver checkpoint when one is "
        "there (a crashed fit), else warm-start from the model saved there. The "
        "earlier fit must have the same nodes. Empty (default) = a fresh solve.",
        typeConverter=TypeConverters.toString)

    def setDag(self, dag):
        """Set the label DAG: a dict ``{child: parent | [parents] | None}``, a pandas
        or Spark DataFrame ``(child, parent)``, or the JSON this Param stores;
        validated now (acyclic, ids ``str`` / ``int``)."""
        if isinstance(dag, str):
            return self._set(dagEdges=_encode_dag(_decode_dag(dag)))
        return self._set(dagEdges=_encode_dag(dag))

    def getDag(self) -> dict:
        """The DAG as given (validated): ``{node: [parents]}``; ``{}`` when unset."""
        raw = str(self.getOrDefault(self.dagEdges))
        if not raw:
            return {}
        parsed = _decode_dag(raw)
        return {n: [p for p in parsed.parents[n] if p is not ROOT] for n in parsed.nodes}


def _scan_labels_spark(dataset, labels_col, landing, policy):
    """The topic-model path's one pass over the labels: the document count, and the
    unknown labels refused by name on the driver (as the DAG path's support pass
    does), rather than inside an executor task."""
    index = LabelIndex(nodes=tuple(landing), closure={n: () for n in landing})
    bc = dataset.sparkSession.sparkContext.broadcast(index)
    try:
        rdd = dataset.select(labels_col).rdd
        if rdd.getNumPartitions() == 0:
            sc = SupportCounts()
        else:
            sc = (rdd.mapPartitions(lambda it, _bc=bc: _partition_support(it, _bc, 0))
                  .treeReduce(lambda a, b: a.merge(b)))
    finally:
        bc.unpersist()
    _support, n_docs = _finish(sc, index, policy)
    return n_docs


def _with_targets(dataset, labeler, labels_col, label_col, mask_col):
    """Append the per-node target vectors built from `labels_col`: `label_col` (a
    SparseVector of the closure) and `mask_col` (all ones under 'full', the observed
    indices under 'closure'). The labeler rides a broadcast whose lifetime is the
    returned DataFrame's (ADR 0047)."""
    from pyspark.ml.linalg import DenseVector, SparseVector, VectorUDT
    from pyspark.sql import functions as F
    from pyspark.sql.types import StructField, StructType

    bc = dataset.sparkSession.sparkContext.broadcast(labeler)

    def _targets(labels, _bc=bc):
        lab = _bc.value
        act, obs = lab(labels)
        C = lab.C
        y = SparseVector(C, list(act), [1.0] * len(act))
        m = (DenseVector(np.ones(C)) if obs is None
             else SparseVector(C, list(obs), [1.0] * len(obs)))
        return (y, m)

    schema = StructType([StructField("y", VectorUDT(), False),
                         StructField("m", VectorUDT(), False)])
    tmp = "__dag_heads_targets"
    return (dataset.withColumn(tmp, F.udf(_targets, schema)(F.col(labels_col)))
            .withColumn(label_col, F.col(f"{tmp}.y"))
            .withColumn(mask_col, F.col(f"{tmp}.m"))
            .drop(tmp))


class DagHeadsEstimator(_DagHeadsParams, _StagePersistence, Estimator):
    """Fit the per-node heads; returns a `DagHeadsModel`.

    Needs ``labelsCol`` and ``featuresCol`` on the fit DataFrame, and either a DAG
    (``setDag``) or a fitted ``DagGatedLDAModel`` (``setTopicModel``), not both. In a
    ``Pipeline`` after a ``DagGatedLDAEstimator`` the stage cannot see the fitted
    topic model, so it is given the same DAG (and ``minDocsPerNode``): on the same
    labels that learns the same node set.

    Persistence layout (``save`` / ``load``)::

        <path>/topicModel/   the held DagGatedLDAModel (its own layout), only when
                             setTopicModel was called
        <path>/metadata/     Params; extra field hasTopicModel; written last

    The topic model is not a Param (it is a fitted stage), so it is saved beside the
    Params rather than dropped: a loaded estimator fits exactly as the saved one."""

    @keyword_only
    def __init__(self, *, featuresCol="topicDistribution", labelsCol="labels",
                 minDocsPerNode=1, unknownLabels="error", labelMaskMode="full",
                 probabilityCol="nodeProbability", stackedProbabilityCol="",
                 regParam=1.0, maxIter=READOUT_MAX_ITER, tol=READOUT_GTOL,
                 thetaTopM=0, engine="spark", saveInterval=-1, saveDir="",
                 resumeFrom="", dag=None, topicModel=None):
        super().__init__()
        self._setDefault(**_DEFAULTS)
        self._topic_model = None
        kwargs = dict(self._input_kwargs)
        dag = kwargs.pop("dag", None)
        tm = kwargs.pop("topicModel", None)
        self._set(**kwargs)
        if dag is not None:
            self.setDag(dag)
        if tm is not None:
            self.setTopicModel(tm)

    def setTopicModel(self, model):
        """Reuse a fitted ``DagGatedLDAModel``'s node set (kept DAG, id map, label
        landing): head i is the model's engine node i, so the heads align with its
        topic blocks."""
        from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAModel
        if not isinstance(model, DagGatedLDAModel):
            raise TypeError(f"setTopicModel takes a fitted DagGatedLDAModel; got "
                            f"{type(model).__name__}")
        model._ns()                                  # refuses a model with no node set
        self._topic_model = model
        return self

    def getTopicModel(self):
        return self._topic_model

    # ---- persistence (_StagePersistence hooks) ----------------------------------

    def _save_data(self, path: str) -> None:
        from spark_vi.io import fs
        if self._topic_model is not None:
            self._topic_model.write().save(fs.join(path, "topicModel"))

    def _extra_metadata(self):
        return {"hasTopicModel": self._topic_model is not None}

    @classmethod
    def _load_data(cls, path: str, metadata: dict) -> "DagHeadsEstimator":
        from spark_vi.io import fs
        from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAModel
        est = cls()
        if metadata.get("hasTopicModel"):
            est._topic_model = DagGatedLDAModel.load(fs.join(path, "topicModel"))
        return est

    def _node_set(self, dataset, labels_col, policy):
        """`(NodeSet, n_docs, source line)` from the topic model or the DAG."""
        has_dag = bool(self.getOrDefault("dagEdges"))
        tm = self._topic_model
        if tm is not None and has_dag:
            raise ValueError("DagHeadsEstimator: set a DAG (setDag) or a topic model "
                             "(setTopicModel), not both")
        if tm is not None:
            if self.isSet("minDocsPerNode"):
                raise ValueError("minDocsPerNode applies to a DAG; with a topic model "
                                 "the heads use the model's node set as is")
            ns = tm._ns()
            ns = dataclasses.replace(ns, frontier=dataclasses.replace(
                ns.frontier, unknown_labels=policy))
            check_labels_column(dataset, labels_col, tuple(ns.frontier.landing))
            n_docs = _scan_labels_spark(dataset, labels_col, ns.frontier.landing,
                                        policy)
            return ns, n_docs, f"the topic model's node set ({len(ns.nodes)} nodes)"
        if not has_dag:
            raise ValueError("DagHeadsEstimator needs a label DAG (setDag) or a fitted "
                             "DagGatedLDAModel (setTopicModel)")
        parsed = _decode_dag(str(self.getOrDefault("dagEdges")))
        check_labels_column(dataset, labels_col, parsed.nodes)
        support, n_docs = count_support_spark(dataset, labels_col, parsed,
                                              unknown_labels=policy)
        ns = learn_node_set(parsed, support, n_docs=n_docs,
                            min_docs_per_node=int(self.getOrDefault("minDocsPerNode")),
                            unknown_labels=policy)
        return ns, n_docs, (f"node set: nodes given {len(parsed)} / kept "
                            f"{len(ns.nodes)} / pruned {len(ns.pruned)} / spliced "
                            f"{len(ns.spliced)} (minDocsPerNode="
                            f"{ns.min_docs_per_node})")

    def _checkpointing(self, ns, K):
        """The persistence Params -> `fit_heads` checkpoint arguments.

        Returns `(checkpoint_path, checkpoint_every, resume_path, warm_start)`."""
        from spark_vi.io import fs

        interval = int(self.getOrDefault("saveInterval"))
        save_dir = str(self.getOrDefault("saveDir"))
        resume_from = str(self.getOrDefault("resumeFrom"))
        if interval == 0:
            raise ValueError("saveInterval=0 is not meaningful; use -1 to disable "
                             "checkpoints")
        if interval > 0 and not save_dir:
            raise ValueError("saveInterval > 0 requires saveDir to be set")
        ckpt = fs.join(save_dir, _CKPT_NAME) if save_dir and interval > 0 else None
        every = interval if interval > 0 else 10
        if not resume_from:
            return ckpt, every, None, None
        rc = fs.join(resume_from, _CKPT_NAME)
        if fs.exists(rc):
            return ckpt, every, rc, None
        if fs.exists(fs.join(resume_from, "metadata")):
            prev = DagHeadsModel.load(resume_from)
            if prev.nodes() != list(ns.nodes):
                raise ValueError(f"resumeFrom={resume_from!r} holds heads for other "
                                 f"nodes ({len(prev.nodes())} nodes) than this fit's "
                                 f"({len(ns.nodes)})")
            if prev.V.shape[1] != K:
                raise ValueError(f"resumeFrom={resume_from!r} holds heads over "
                                 f"{prev.V.shape[1]} features; this fit has {K}")
            return ckpt, every, None, (prev.V, prev.b_raw)
        raise FileNotFoundError(f"resumeFrom={resume_from!r} holds neither a heads "
                                f"checkpoint ({_CKPT_NAME}) nor a saved DagHeadsModel")

    def _fit(self, dataset):
        from pyspark.sql import functions as F

        from spark_vi.readout import LocalReadoutData, SparkReadoutData, fit_heads

        g = self.getOrDefault
        labels_col, fc = str(g("labelsCol")), str(g("featuresCol"))
        policy, mode = str(g("unknownLabels")), str(g("labelMaskMode"))
        if policy not in UNKNOWN_LABEL_POLICIES:
            raise ValueError(f"unknownLabels must be one of {UNKNOWN_LABEL_POLICIES}; "
                             f"got {policy!r}")
        if mode not in LABEL_MASK_MODES:
            raise ValueError(f"labelMaskMode must be one of {LABEL_MASK_MODES}; got "
                             f"{mode!r}")
        engine = str(g("engine"))
        if engine not in ("spark", "local"):
            raise ValueError(f"engine must be 'spark' or 'local', got {engine!r}")
        ns, n_docs, source = self._node_set(dataset, labels_col, policy)
        if n_docs == 0:
            raise ValueError("DagHeadsEstimator: empty dataset")
        labeler = NodeLabeler.from_node_set(ns, mask_mode=mode, unknown_labels=policy)
        C = labeler.C
        K = int(dataset.select(fc).head()[0].size)
        ckpt, every, resume_path, warm = self._checkpointing(ns, K)

        lc, mc = _TARGET_COLS
        df = _with_targets(dataset.select(fc, labels_col), labeler, labels_col, lc, mc)
        cols = dict(topic_col=fc, label_col=lc, mask_col=mc)
        topm = int(g("thetaTopM"))
        if engine == "local":
            df = df.withColumn("_row_id", F.monotonically_increasing_id())
            data = LocalReadoutData.from_spark(df, C, K, topm=topm, id_col="_row_id",
                                               **cols)
        else:
            data = SparkReadoutData(df, C, K, topm=topm, **cols)
        print(f"{_LOG} {n_docs} docs; {source}; {C} heads (root + {C - 1} nodes), "
              f"labelMaskMode={mode}", flush=True)
        V, b_raw, const, degenerate, info = fit_heads(
            data, l2=float(g("regParam")), gtol=float(g("tol")),
            max_iter=int(g("maxIter")), label="DagHeads", warm_start=warm,
            checkpoint_path=ckpt, checkpoint_every=every,
            resume=resume_path is not None, resume_path=resume_path)
        model = self._copyValues(DagHeadsModel(ns, V, b_raw, const, degenerate,
                                               theta_topm=topm, info=info))
        deg = model.degenerateNodes()
        print(f"{_LOG} {len(deg)} degenerate node(s) predicted as a constant "
              f"(observed training rows empty or single-class)", flush=True)
        if g("saveDir"):
            # Into saveDir in place: it may be a run directory holding other files,
            # and an earlier model there is replaced, nothing else.
            save_stage_in_place(model, str(g("saveDir")))
        return model


class DagHeadsModel(_DagHeadsParams, _StagePersistence, Model):
    """The fitted heads: raw-feature coefficients `V (C, K)`, `b_raw (C,)`, the
    constant fallback of nodes whose observed training rows were single-class, and
    the node set they are indexed by (index 0 the root, 1..N `nodes()`).

    Persistence layout (``save`` / ``load``)::

        <path>/readout_heads_model.npz   V, b_raw, const, degenerate, theta_topm
                                         (+ W_std): `readout.heads.write_heads`, the
                                         file the drivers' scoring passes read
        <path>/node_set.json             the node set (NodeSet.to_dict; user ids
                                         with their str / int types)
        <path>/metadata/                 Params, written last
    """

    def __init__(self, node_set, V, b_raw, const, degenerate, *, theta_topm=0,
                 info=None):
        super().__init__()
        self._setDefault(**_DEFAULTS)
        self._node_set = node_set
        self.V = np.asarray(V, dtype=np.float64)
        self.b_raw = np.asarray(b_raw, dtype=np.float64)
        self.const = np.asarray(const, dtype=np.float64)
        self.degenerate = np.asarray(degenerate, dtype=bool)
        self.theta_topm = int(theta_topm)
        self.info = info or {}
        if self.V.shape[0] != len(node_set.user_ids):
            raise ValueError(f"{self.V.shape[0]} heads for a node set of "
                             f"{len(node_set.user_ids) - 1} nodes + the root")

    # ---- node-keyed outputs, user ids throughout -----------------------------

    def nodes(self) -> list:
        """The kept nodes, in output order: node ``nodes()[i]`` is index ``i + 1`` of
        ``probabilityCol`` (index 0 is the root, the 'any label' head)."""
        return list(self._node_set.nodes)

    def nodeIndex(self, node) -> int:
        """The output index of a kept node (``None`` = the root, index 0). KeyError
        names a node that has no head (pruned, spliced or not a DAG node)."""
        if node is None:
            return 0
        ns = self._node_set
        e = ns.engine_id.get(node)
        if e is None:
            why = ("was pruned" if node in ns.pruned else
                   "was spliced" if node in ns.spliced else
                   "was dropped as unsupported" if node in ns.unsupported else
                   "is not a DAG node")
            raise KeyError(f"node {node!r} has no head: it {why}")
        return int(e)

    def degenerateNodes(self) -> dict:
        """``{node: constant}`` for the heads whose observed training rows were empty
        (constant 0.0), all negative (0.0) or all positive (1.0): no finite optimum,
        so the head predicts that constant. The root's key is ``None``."""
        ids = self._node_set.user_ids
        return {(None if i == 0 else ids[i]): float(self.const[i])
                for i in np.flatnonzero(self.degenerate)}

    def dag(self) -> dict:
        """The kept DAG the heads are indexed by: ``{node: [parents]}``."""
        return {n: [p for p in ps if p is not ROOT]
                for n, ps in self._node_set.parents.items()}

    def prunedNodes(self) -> dict:
        """``{dropped node: [landing nodes]}`` (see DagGatedLDAModel.prunedNodes)."""
        return {n: list(ls) for n, ls in self._node_set.pruned.items()}

    def splicedNodes(self) -> list:
        return list(self._node_set.spliced)

    def unsupportedNodes(self) -> dict:
        """``{dropped node: [landing nodes]}`` for the nodes the topic model dropped
        as unsupported (see DagGatedLDAModel.dropUnsupportedNodes): no head."""
        return {n: list(ls) for n, ls in self._node_set.unsupported.items()}

    def _labeler(self) -> NodeLabeler:
        return NodeLabeler.from_node_set(
            self._node_set, mask_mode=str(self.getOrDefault("labelMaskMode")),
            unknown_labels=str(self.getOrDefault("unknownLabels")))

    def attachLabels(self, dataset, labelCol="label", labelMaskCol="labelMask"):
        """Append the per-node target vectors the heads were fit against, built from
        ``labelsCol`` with this model's node set and ``labelMaskMode`` (for scoring a
        held-out split with `spark_vi.readout.evaluate`)."""
        return _with_targets(dataset, self._labeler(),
                             str(self.getOrDefault("labelsCol")), labelCol,
                             labelMaskCol)

    def _transform(self, dataset):
        from pyspark.ml.linalg import DenseVector, VectorUDT
        from pyspark.sql import functions as F

        from spark_vi.readout.kernels import _SCORE_CLIP, to_array, topm_sparse

        g = self.getOrDefault
        sc = dataset.sparkSession.sparkContext
        stacked_col = g("stackedProbabilityCol")
        M = None
        if stacked_col:
            from spark_vi.dag.ops import closure_matrix
            from spark_vi.readout.stacked import closure_operator
            M = closure_operator(closure_matrix(self._node_set.engine_parents,
                                                self.V.shape[0]))
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

    # ---- persistence (_StagePersistence hooks) ----------------------------------

    def _save_data(self, path: str) -> None:
        from spark_vi.io import fs
        from spark_vi.readout.heads import write_heads

        C, K = self.V.shape
        if not write_heads(path, _HEADS_LABEL, self.V, self.b_raw, self.const,
                           self.degenerate, C, K, self.theta_topm,
                           W_std=self.info.get("W_std")):
            raise OSError(f"could not write the heads to {path}")
        fs.write_text(fs.join(path, "node_set.json"),
                      json.dumps(self._node_set.to_dict(), sort_keys=True))

    @classmethod
    def _load_data(cls, path: str, metadata: dict) -> "DagHeadsModel":
        from spark_vi.io import fs
        from spark_vi.readout.heads import read_heads

        got = read_heads(path, _HEADS_LABEL)
        if got is None:
            raise ValueError(f"no readable heads in {path}")
        V, b_raw, const, degenerate, _C, _K, topm = got
        ns = NodeSet.from_dict(json.loads(fs.read_text(fs.join(path, "node_set.json"))))
        return cls(ns, V, b_raw, const, degenerate, theta_topm=topm)

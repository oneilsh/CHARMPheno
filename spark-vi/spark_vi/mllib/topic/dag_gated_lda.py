"""MLlib Estimator/Model shim for the DAG-gated LDA engine (ADR 0050; user API, spec
2026-10-30 R1-R5, R7, R9, R11).

Mirrors mllib/topic/lda.py (ADR 0009): a translation layer over
``spark_vi.models.topic.dag_gated_lda.DagGatedOnlineLDA`` + ``VIRunner``. All SVI
logic lives in the engine; this shim learns the node set, marshals DataFrame columns
into ``GatedBOWDocument`` rows, builds the gate seeds, and wraps the trained
``VIResult`` as an MLlib-shaped Model.

What the user supplies, in their OWN node ids (``str`` or ``int``, kept as given: ``1``
and ``"1"`` are different nodes):

- the DAG (``setDag``): a dict ``{child: parent | [parents]}``, a pandas or a Spark
  DataFrame ``(child, parent)``; parentless nodes hang from a synthetic root;
- ``labelsCol``: an ``array<string>`` / ``array<int>`` column of the nodes each
  document attests (any nodes; ancestors and duplicates allowed; empty or null =
  a background document);
- the features: ``featuresCol`` (one vector column) or ``featuresCols`` (one vector
  column per domain), each domain's width being its vectors' fixed size; optionally
  ``vocabularies``, the term ids of each feature column (needed only by the seed
  extras ``nodeTerms`` / ``gateSeedProfile``, which name terms).

``fit`` learns the node set from the DataFrame it is given (``spark_vi.dag.nodeset``):
support counted over each document's label closure, nodes with support <
``minDocsPerNode`` pruned (their labels land on their nearest kept ancestors), the
kept DAG transitively reduced and single-child chains spliced. The kept nodes get
dense ENGINE ids (internal; the model holds the map), each document's labels are
reduced to its engine frontier executor-side through a broadcast ``FrontierFn``, and
the E-step restricts the document to the gate ``allowed_set(frontier)``. K is derived
(``gateNBg + kept nodes * gateTpn``). ``transform`` folds documents in UNGATED over
all K topics (labels are unknown at scoring time) and appends
``topicDistributionCol``.

The fitted model answers in user ids (R5): ``dag()``, ``prunedNodes()``,
``splicedNodes()``, ``nodeSupport()``, ``nodeTopics(node)``, ``backgroundTopics()``,
``describeNodes()`` (with the term ids when ``vocabularies`` are set), and
``nodeStatistics()``, how well the data sustains each node's block (R14.1). Dropping
the flat blocks is explicit: ``dropUnsupportedNodes()`` returns a new model without
them (R14.2), reported in ``unsupportedNodes()``. The node set
rides the VIResult's metadata (``dag_node_set``), so ``save`` / ``load`` (MLlib
persistence: Params metadata + the VIResult, any Hadoop URI, ``Pipeline`` too) keep it, and
``resumeFrom`` refuses a checkpoint whose node set differs from the one learned from
the data it is given.

Defaults are MLlib LDA's where a Param has an analogue; ``RECOMMENDED_PARAMS`` /
``DagGatedLDAEstimator.recommended(...)`` is the documented working preset (see the
estimator's docstring).
"""
from __future__ import annotations

import json
from typing import Callable

import numpy as np
from pyspark import StorageLevel, keyword_only
from pyspark.ml.base import Estimator, Model
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasFeaturesCol, HasMaxIter, HasSeed

from spark_vi.core.config import VIConfig
from spark_vi.dag.layout import DagLayout
from spark_vi.dag.nodeset import (
    ROOT,
    UNKNOWN_LABEL_POLICIES,
    NodeSet,
    _coerce_id,
    count_support_spark,
    drop_nodes,
    learn_node_set,
    node_sort_key,
    parse_dag,
)

from spark_vi.mllib._common import (
    DriverParamsWritable,
    _PersistenceParams,
    _StagePersistence,
    apply_persistence_params,
)
from spark_vi.mllib.topic._common import _concat_domain_features, _vector_to_bow_document
from spark_vi.models.topic.dag_gated_lda import (
    DagGatedOnlineLDA,
    expElogbeta_from_lambda,
)
from spark_vi.models.topic.types import GatedBOWDocument

_LOG = "[dag-gated-lda]"
# Where the fit's E-step runs (the `engine` Param).
FIT_ENGINES = ("spark", "local")
_NODE_SET_KEY = "dag_node_set"


def _row_to_gated_document(row, features_cols, domain_sizes, frontier_fn,
                           labels_col) -> GatedBOWDocument:
    """A Spark row -> GatedBOWDocument: the per-domain vectors concatenated into the
    engine's one token-id space (each validated against its fixed width), and the
    row's labels (user ids) reduced to the engine-id frontier by `frontier_fn`."""
    indices, counts = _concat_domain_features([row[c] for c in features_cols],
                                              domain_sizes)
    return GatedBOWDocument(indices=indices, counts=counts, length=int(counts.sum()),
                            frontier=frontier_fn(row[labels_col]))


# --------------------------------------------------------------------------------
# JSON codecs for the structured Params. JSON keeps a number and a string apart, so
# user ids round-trip with their type (1 stays 1, "1" stays "1"); every structure is
# a list of pairs because JSON object keys are always strings.
# --------------------------------------------------------------------------------

def _encode_dag(dag) -> str:
    """Any accepted DAG input (R1) -> validated -> a JSON edge list
    `[[child, parent], ...]`, `[child, null]` for a parentless node."""
    parsed = parse_dag(dag)
    edges = []
    for c in parsed.nodes:
        for p in parsed.parents[c]:
            edges.append([c, None if p is ROOT else p])
    return json.dumps(edges)


def _decode_dag(raw: str):
    out: dict = {}
    for c, p in json.loads(raw):
        ps = out.setdefault(_coerce_id(c, what="child"), [])
        if p is not None:
            ps.append(_coerce_id(p, what="parent"))
    return parse_dag(out)


def _term_ref(t):
    """A term reference: a bare term id (`str` / `int`) or a qualified
    `(featureColumn, term)` pair (list or tuple), returned as a tuple."""
    if isinstance(t, (list, tuple)):
        if len(t) != 2 or not isinstance(t[0], str):
            raise TypeError(f"a qualified term must be a (featureColumn, term) pair; "
                            f"got {t!r}")
        return (t[0], _coerce_id(t[1], what="term"))
    return _coerce_id(t, what="term")


def _ref_json(r):
    return list(r) if isinstance(r, tuple) else r


def _sort_refs(refs):
    return sorted(refs, key=lambda r: ((1,) + (r[0],) + node_sort_key(r[1])
                                       if isinstance(r, tuple)
                                       else (0, "") + node_sort_key(r)))


def check_labels_column(dataset, labels_col, node_ids):
    """Up front, before any Spark job: `labelsCol` exists, is an array of
    strings or integers, and its element type is one the DAG's ids (`node_ids`) have
    (ids are matched exactly, so an array<string> column against an all-int DAG
    would match nothing). Shared with the readout heads (`DagHeadsEstimator`)."""
    from pyspark.sql.types import (ArrayType, ByteType, IntegerType, LongType,
                                   ShortType, StringType)
    if labels_col not in dataset.columns:
        raise ValueError(f"labelsCol={labels_col!r} is not a column of the input "
                         f"(columns: {dataset.columns})")
    dt = dataset.schema[labels_col].dataType
    if not isinstance(dt, ArrayType):
        raise ValueError(f"labelsCol={labels_col!r} must be an array<string> or "
                         f"array<int> column of node ids; it is {dt.simpleString()}")
    et = dt.elementType
    has_int = any(isinstance(n, int) for n in node_ids)
    has_str = any(isinstance(n, str) for n in node_ids)
    if isinstance(et, StringType):
        if not has_str:
            raise ValueError(
                f"labelsCol={labels_col!r} is array<string> but every DAG node id "
                f"is an int; ids are matched exactly (1 and '1' are different "
                f"nodes): cast the labels or give the DAG string ids")
    elif isinstance(et, (IntegerType, LongType, ShortType, ByteType)):
        if not has_int:
            raise ValueError(
                f"labelsCol={labels_col!r} is {dt.simpleString()} but every DAG "
                f"node id is a str; ids are matched exactly (1 and '1' are "
                f"different nodes): cast the labels or give the DAG int ids")
    else:
        raise ValueError(f"labelsCol={labels_col!r} must be array<string> or "
                         f"array<int>; it is {dt.simpleString()}")


def _node_column(ids):
    """`(Spark type, cell)` for a column of user node ids (``describeNodes``,
    ``nodeStatistics``): LONG when every id is an int, STRING when every one is a str,
    and STRING with the ints written in decimal when they are mixed (there ``1`` and
    ``"1"`` render alike; ``nodeTopics`` tells them apart)."""
    from pyspark.sql.types import LongType, StringType
    all_int = bool(ids) and all(isinstance(n, int) for n in ids)
    all_str = all(isinstance(n, str) for n in ids)
    if all_int or all_str:
        return (LongType() if all_int else StringType()), (lambda n: n)
    return StringType(), str


class _DagGatedLDAParams(HasFeaturesCol, HasMaxIter, HasSeed, _PersistenceParams):
    """Shared Param surface for DagGatedLDAEstimator and DagGatedLDAModel.

    The SVI schedule, Dirichlet and CAVI knobs mirror ``_OnlineLDAParams``; on top
    sit the DAG and labels (``dagEdges``, set with ``setDag``; ``labelsCol``,
    ``minDocsPerNode``, ``unknownLabels``), the layout (``gateNBg``, ``gateTpn``),
    the leaves-first schedule and its seeds, and the multi-domain columns.
    Persistence Params (saveInterval, saveDir, resumeFrom) come from
    ``_PersistenceParams``.

    The DAG Param is ``dagEdges``, not ``dag``: the fitted model's ``dag()`` method
    (the KEPT DAG, R5) owns that name. The constructor / ``setParams`` keyword is
    ``dag`` (routed through ``setDag``).

    Every plain Param has MLlib's ``setX`` / ``getX`` pair (``setLabelsCol``,
    ``setFeaturesCols``, ``setGateNBg``, ...); the structured ones are set through
    ``setDag``, ``setVocabularies``, ``setNodeTerms`` and ``setGateSeedProfile``.
    Each Param's help text states its default."""

    # The shared Params (featuresCol, maxIter, seed) are re-declared so their help
    # text states this shim's defaults (R11); the names and semantics are MLlib's.
    featuresCol = Param(
        Params._dummy(), "featuresCol",
        "the one vector column of term counts, used when featuresCols is empty. "
        "Default 'features'.",
        typeConverter=TypeConverters.toString,
    )
    maxIter = Param(
        Params._dummy(), "maxIter",
        "number of SVI iterations (>= 0); a resumed fit runs this many more. "
        "Default 20 (MLlib LDA).",
        typeConverter=TypeConverters.toInt,
    )
    seed = Param(
        Params._dummy(), "seed",
        "random seed for the initialization and the mini-batch sampling. Default "
        "unset: nondeterministic (only an explicitly set seed reaches the fit).",
        typeConverter=TypeConverters.toInt,
    )
    topicDistributionCol = Param(
        Params._dummy(), "topicDistributionCol",
        "output column of transform: each document's topic mixture theta (length K). "
        "Default 'topicDistribution'.",
        typeConverter=TypeConverters.toString,
    )
    learningOffset = Param(
        Params._dummy(), "learningOffset",
        "tau0 in the Robbins-Monro step rho_t = (tau0 + t)^-kappa; larger damps the "
        "early iterations. Default 1024 (MLlib LDA); the recommended preset uses 64.",
        typeConverter=TypeConverters.toFloat,
    )
    learningDecay = Param(
        Params._dummy(), "learningDecay",
        "kappa in the Robbins-Monro step rho_t = (tau0 + t)^-kappa, in (0.5, 1]. "
        "Default 0.51 (MLlib LDA).",
        typeConverter=TypeConverters.toFloat,
    )
    subsamplingRate = Param(
        Params._dummy(), "subsamplingRate",
        "fraction of the documents sampled per mini-batch (<= 0 or >= 1 = full "
        "batch). Default 0.05 (MLlib LDA); the recommended preset uses 0.1.",
        typeConverter=TypeConverters.toFloat,
    )
    docConcentration = Param(
        Params._dummy(), "docConcentration",
        "Dirichlet concentration alpha on each document's theta: a scalar "
        "(symmetric) or a length-K vector. Default unset = 1/K (MLlib LDA), which "
        "is very small when K is in the thousands; the recommended preset uses 0.5.",
        typeConverter=TypeConverters.toListFloat,
    )
    topicConcentration = Param(
        Params._dummy(), "topicConcentration",
        "Dirichlet concentration eta on each topic's term distribution (the prior on "
        "lambda; one scalar for every domain), > 0. Default unset = 1/K (MLlib LDA).",
        typeConverter=TypeConverters.toFloat,
    )
    optimizeDocConcentration = Param(
        Params._dummy(), "optimizeDocConcentration",
        "learn alpha by empirical Bayes during the fit: one Newton step per "
        "iteration, alpha tied within each node's block, from the histogram of the "
        "documents' gates. Default False, unlike MLlib LDA (True): on deep DAGs "
        "learned alpha tends to 1/K and can leave deep nodes' blocks without "
        "words, so alpha is held fixed unless asked.",
        typeConverter=TypeConverters.toBoolean,
    )
    gammaShape = Param(
        Params._dummy(), "gammaShape",
        "shape of the Gamma(shape, 1/shape) draw that initializes the per-document "
        "gamma and the topics' lambda. Default 100.",
        typeConverter=TypeConverters.toFloat,
    )
    caviMaxIter = Param(
        Params._dummy(), "caviMaxIter",
        "maximum iterations of the per-document coordinate-ascent (CAVI) loop, at "
        "fit and at transform. Default 100.",
        typeConverter=TypeConverters.toInt,
    )
    caviTol = Param(
        Params._dummy(), "caviTol",
        "relative tolerance on gamma that stops the per-document CAVI loop early. "
        "Default 0.001.",
        typeConverter=TypeConverters.toFloat,
    )
    dagEdges = Param(
        Params._dummy(), "dagEdges",
        "the label DAG, set with setDag (a dict {child: parent | [parents]}, or a "
        "pandas / Spark DataFrame (child, parent)); stored validated as a JSON edge "
        "list [[child, parent | null], ...] in the user's ids. Parentless nodes hang "
        "from a synthetic root. Required: no default.",
        typeConverter=TypeConverters.toString,
    )
    labelsCol = Param(
        Params._dummy(), "labelsCol",
        "array<string> or array<int> column of the DAG nodes each document attests "
        "(any nodes: ancestors and duplicates allowed; empty or null = a background "
        "document). The element type must match the DAG's ids. Default 'labels'.",
        typeConverter=TypeConverters.toString,
    )
    minDocsPerNode = Param(
        Params._dummy(), "minDocsPerNode",
        "a node gets a topic block only if at least this many training documents' "
        "label closures reach it; a pruned node's labels land on its nearest kept "
        "ancestors. Default 1 (only nodes no document reaches are pruned).",
        typeConverter=TypeConverters.toInt,
    )
    unknownLabels = Param(
        Params._dummy(), "unknownLabels",
        "a label that is not a DAG node: 'error' (refused by name) or 'ignore' "
        "(dropped; a document left with no label is background). Default 'error'.",
        typeConverter=TypeConverters.toString,
    )
    engine = Param(
        Params._dummy(), "engine",
        "where the fit's E-step runs: 'spark' (the executors: VIRunner) or 'local' "
        "(the training documents are collected to the driver once and the fit runs "
        "in memory: LocalRunner, over localProcesses forked processes). The node set, "
        "the seed passes and transform stay on Spark either way. 'local' pays off "
        "when the corpus fits the driver's memory and a Spark iteration is mostly "
        "overhead (shipping the topic matrix, scheduling, merging). Default 'spark'.",
        typeConverter=TypeConverters.toString,
    )
    localProcesses = Param(
        Params._dummy(), "localProcesses",
        "with engine 'local': worker processes for the E-step (0 = one per CPU of "
        "the driver). Default 0.",
        typeConverter=TypeConverters.toInt,
    )
    gateNBg = Param(
        Params._dummy(), "gateNBg",
        "number of shared background topics, open to every document; "
        "K = gateNBg + kept nodes * gateTpn. Default 10 (with gateTpn=0 the model is "
        "plain LDA with MLlib's default k=10); the recommended preset uses 200.",
        typeConverter=TypeConverters.toInt,
    )
    gateTpn = Param(
        Params._dummy(), "gateTpn",
        "topics per kept DAG node (0 = a flat layout: K = gateNBg, plain LDA). "
        "Default 1.",
        typeConverter=TypeConverters.toInt,
    )
    gateWarm = Param(
        Params._dummy(), "gateWarm",
        "leaves-first gate schedule: iterations per node-height level before the "
        "next level's blocks open (leaves first, the top-level nodes last); 0 = the "
        "single-round gate (every block open from iteration 0). Default 0; the "
        "recommended preset uses 2.",
        typeConverter=TypeConverters.toInt,
    )
    gateOpenSeed = Param(
        Params._dummy(), "gateOpenSeed",
        "with gateWarm > 0: seed each block, the moment it opens, from the term "
        "counts of the documents its gate admits (one pass before the fit); False "
        "leaves late blocks at their random init, where they tend to stay empty. "
        "Default True.",
        typeConverter=TypeConverters.toBoolean,
    )
    gateSeedMode = Param(
        Params._dummy(), "gateSeedMode",
        "with gateWarm > 0 and gateOpenSeed: the open-time seed. 'balanced': each "
        "child subtree's documents weighted equally, plus the documents labelled at "
        "the node itself, with the descendants' nodeTerms removed; 'subtree': the "
        "term counts of every document the block admits. Default 'balanced'.",
        typeConverter=TypeConverters.toString,
    )
    gateSeedLeaves = Param(
        Params._dummy(), "gateSeedLeaves",
        "with gateWarm > 0 and gateOpenSeed: also seed every LEAF block at iteration "
        "0 from the term counts of its own documents (otherwise leaves start from the "
        "random init). Default False; the recommended preset uses True.",
        typeConverter=TypeConverters.toBoolean,
    )
    featuresCols = Param(
        Params._dummy(), "featuresCols",
        "ordered list of per-domain vector columns. When set, each row's per-domain "
        "vectors are concatenated into the engine's single term-index space and the "
        "engine carries a per-domain lambda {m: (K, V_m)}; a domain's width is its "
        "vectors' size (the column's ML attribute size when present, else the first "
        "row's), and every row is checked against it. Default [] (use the single "
        "featuresCol).",
        typeConverter=TypeConverters.toListString,
    )
    vocabularies = Param(
        Params._dummy(), "vocabularies",
        "set with setVocabularies: one list of term ids (str or int) per feature "
        "column (featuresCols, else [featuresCol]), aligned with the vector indices "
        "(length = that column's vector size). Used by nodeTerms / gateSeedProfile, "
        "which name terms, and by describeNodes' terms column. Terms are matched "
        "exactly as given, like node ids: vectorizer vocabularies hold strings, so "
        "int term ids are passed as their strings there. Stored as JSON. Default "
        "unset.",
        typeConverter=TypeConverters.toString,
    )
    nodeTerms = Param(
        Params._dummy(), "nodeTerms",
        "set with setNodeTerms: {node: [term]}, the terms that attest each node (user "
        "node ids, term ids of the vocabularies). With gateSeedMode='balanced', a "
        "node's terms are removed from its ancestors' open-time seeds. A term in two "
        "vocabularies must be qualified as (featureColumn, term). Entries of nodes "
        "not kept by the fit, and terms in no vocabulary, are dropped (counted in the "
        "log). Stored as JSON. Default unset.",
        typeConverter=TypeConverters.toString,
    )
    gateSeedProfile = Param(
        Params._dummy(), "gateSeedProfile",
        "set with setGateSeedProfile: {node: {term: weight}}, an external per-node "
        "term profile. With gateWarm > 0 it REPLACES a profiled node's seed in the "
        "domains its terms belong to (leaves at iteration 0, later blocks when they "
        "open); other domains and unprofiled nodes keep the count seed. Same id, "
        "qualification and dropping rules as nodeTerms. Stored as JSON. Default "
        "unset.",
        typeConverter=TypeConverters.toString,
    )

    # ---- structured setters / getters (JSON in the Param, Python out) ----------

    def setDag(self, dag):
        """Set the label DAG: a dict ``{child: parent | [parents] | None}``, a pandas
        or Spark DataFrame with columns ``(child, parent)`` (a Spark frame is
        collected once: a label DAG is small), or the JSON this Param stores. It is
        validated now (acyclic, ids ``str`` / ``int``)."""
        if isinstance(dag, str):
            return self._set(dagEdges=_encode_dag(_decode_dag(dag)))
        return self._set(dagEdges=_encode_dag(dag))

    def getDag(self) -> dict:
        """The DAG as given (validated): ``{node: [parents]}``, ``[]`` for a
        top-level node; ``{}`` when unset."""
        raw = str(self.getOrDefault(self.dagEdges))
        if not raw:
            return {}
        parsed = _decode_dag(raw)
        return {n: [p for p in parsed.parents[n] if p is not ROOT] for n in parsed.nodes}

    def _parsed_dag(self):
        raw = str(self.getOrDefault(self.dagEdges))
        if not raw:
            raise ValueError("the DAG-gated LDA needs a label DAG: call setDag(...) (a "
                             "depth-1 DAG {node: None, ...} is a flat labelled-LDA gate)")
        return _decode_dag(raw)

    def setVocabularies(self, vocabularies):
        """One list of term ids per feature column (aligned with the vector indices),
        or the JSON this Param stores; None / [] clears it."""
        if vocabularies is None or vocabularies == "" or vocabularies == []:
            return self._set(vocabularies="")
        if isinstance(vocabularies, str):
            vocabularies = json.loads(vocabularies)
        out = []
        for m, vocab in enumerate(vocabularies):
            ids = [_coerce_id(t, what=f"term of vocabulary {m}") for t in vocab]
            if len(set(ids)) != len(ids):
                dup = sorted({t for t in ids if ids.count(t) > 1}, key=node_sort_key)
                raise ValueError(f"vocabulary {m} repeats term(s) {dup[:10]}")
            out.append(ids)
        return self._set(vocabularies=json.dumps(out))

    def getVocabularies(self) -> list:
        raw = str(self.getOrDefault(self.vocabularies))
        return [list(v) for v in json.loads(raw)] if raw else []

    def setNodeTerms(self, node_terms):
        """``{node: [term]}`` (a term is an id or a ``(featureColumn, term)`` pair), or
        the JSON this Param stores; None / {} clears it."""
        if node_terms is None or node_terms == "" or node_terms == {}:
            return self._set(nodeTerms="")
        if isinstance(node_terms, str):
            node_terms = {_coerce_id(n): ts for n, ts in json.loads(node_terms)}
        enc = []
        for n in sorted((_coerce_id(n) for n in node_terms), key=node_sort_key):
            refs = {_term_ref(t) for t in node_terms[n]}
            if refs:
                enc.append([n, [_ref_json(r) for r in _sort_refs(refs)]])
        return self._set(nodeTerms=json.dumps(enc))

    def getNodeTerms(self) -> dict:
        """``{node: [term]}``; a qualified term comes back as a tuple."""
        raw = str(self.getOrDefault(self.nodeTerms))
        if not raw:
            return {}
        return {_coerce_id(n): [_term_ref(t) for t in ts] for n, ts in json.loads(raw)}

    def setGateSeedProfile(self, profile):
        """``{node: {term: weight}}`` (a term is an id or a ``(featureColumn, term)``
        tuple), or the JSON this Param stores; None / {} clears it. Weights <= 0 are
        dropped."""
        if profile is None or profile == "" or profile == {}:
            return self._set(gateSeedProfile="")
        if isinstance(profile, str):
            profile = {_coerce_id(n): {_term_ref(t): w for t, w in tw}
                       for n, tw in json.loads(profile)}
        enc = []
        for n in sorted((_coerce_id(n) for n in profile), key=node_sort_key):
            tw = {}
            for t, w in profile[n].items():
                if float(w) > 0:
                    tw[_term_ref(t)] = float(w)
            if tw:
                enc.append([n, [[_ref_json(r), tw[r]] for r in _sort_refs(tw)]])
        return self._set(gateSeedProfile=json.dumps(enc))

    def getGateSeedProfile(self) -> dict:
        """``{node: {term: weight}}``; a qualified term comes back as a tuple."""
        raw = str(self.getOrDefault(self.gateSeedProfile))
        if not raw:
            return {}
        return {_coerce_id(n): {_term_ref(t): float(w) for t, w in tw}
                for n, tw in json.loads(raw)}

    def _feature_columns(self) -> tuple[list, bool]:
        """(the feature columns, multi-domain?): featuresCols, else [featuresCol]."""
        fcols = list(self.getOrDefault("featuresCols") or [])
        if fcols:
            return fcols, True
        return [self.getOrDefault("featuresCol")], False


# Plain Params get MLlib's setX / getX pair (generated below; a getter the shared
# mixins already define is kept). The structured Params have hand-written ones
# (setDag / setVocabularies / setNodeTerms / setGateSeedProfile), and there is no
# setDagEdges: the DAG is only ever set through setDag's validation.
_PLAIN_PARAMS = (
    "featuresCol", "maxIter", "seed", "topicDistributionCol", "learningOffset",
    "learningDecay", "subsamplingRate", "docConcentration", "topicConcentration",
    "optimizeDocConcentration", "gammaShape", "caviMaxIter", "caviTol", "labelsCol",
    "minDocsPerNode", "unknownLabels", "gateNBg", "gateTpn", "gateWarm",
    "gateOpenSeed", "gateSeedMode", "gateSeedLeaves", "featuresCols", "engine",
    "localProcesses",
)


def _add_plain_accessors(cls, names):
    for name in names:
        cap = name[0].upper() + name[1:]

        def setter(self, value, _n=name):
            return self._set(**{_n: value})

        def getter(self, _n=name):
            return self.getOrDefault(_n)

        setter.__name__, getter.__name__ = f"set{cap}", f"get{cap}"
        setter.__doc__ = f"Set {name}: see the Param's help text (explainParam)."
        getter.__doc__ = f"The value (or default) of {name}."
        if f"set{cap}" not in vars(cls):
            setattr(cls, f"set{cap}", setter)
        if not hasattr(cls, f"get{cap}"):
            setattr(cls, f"get{cap}", getter)


_add_plain_accessors(_DagGatedLDAParams, _PLAIN_PARAMS)


# MLlib-like defaults (spec decision 1): every Param with an analogue on
# pyspark.ml.clustering.LDA takes LDA's default (maxIter 20, learningOffset 1024,
# learningDecay 0.51, subsamplingRate 0.05, docConcentration / topicConcentration
# unset = 1/K) except optimizeDocConcentration, False here (True in MLlib: learned
# alpha starves deep blocks on deep DAGs); gateNBg = 10 makes the flat layout
# (gateTpn = 0) LDA's k = 10; the other gate Params are the simplest working ones
# (one topic per node, the single-round gate).
_DAG_GATED_LDA_DEFAULTS = dict(
    maxIter=20,
    featuresCol="features", topicDistributionCol="topicDistribution",
    learningOffset=1024.0, learningDecay=0.51, subsamplingRate=0.05,
    optimizeDocConcentration=False,
    gammaShape=100.0, caviMaxIter=100, caviTol=1e-3,
    dagEdges="", labelsCol="labels", minDocsPerNode=1, unknownLabels="error",
    gateNBg=10, gateTpn=1,
    gateWarm=0, gateOpenSeed=True, gateSeedMode="balanced", gateSeedLeaves=False,
    featuresCols=[], vocabularies="", nodeTerms="", gateSeedProfile="",
    engine="spark", localProcesses=0,
    # docConcentration / topicConcentration / seed: no default (isSet; unset = 1/K,
    # 1/K, nondeterministic).
)

# The working preset (spec R10, decision 1): the configuration that trains every
# block of a deep DAG on a large corpus. Departures from the defaults, and why:
# - gateNBg 200: a wide shared background soaks up the words common to all
#   documents, so node blocks learn what distinguishes their documents;
# - learningOffset 64, subsamplingRate 0.1, maxIter 60: the topics move within a
#   few dozen mini-batch iterations rather than hundreds;
# - docConcentration 0.5 held fixed (optimizeDocConcentration False): learned alpha
#   collapses towards 1/K, where a document's words go to its ancestors' and the
#   background's blocks and deep blocks stay empty;
# - gateWarm 2 with the balanced open-time seed and leaf seeds: blocks open leaves
#   first, each starting from the term counts of the documents it admits, so a
#   deep block does not start flat against sharp ancestor topics.
# Not in the preset: minDocsPerNode (scale it to the corpus: a block needs enough
# documents to estimate a topic; ~100 on large corpora), seed, and the column /
# DAG Params.
RECOMMENDED_PARAMS = dict(
    gateNBg=200, gateTpn=1,
    maxIter=60, learningOffset=64.0, learningDecay=0.51, subsamplingRate=0.1,
    docConcentration=[0.5], optimizeDocConcentration=False,
    gammaShape=100.0, caviMaxIter=100, caviTol=1e-3,
    gateWarm=2, gateOpenSeed=True, gateSeedMode="balanced", gateSeedLeaves=True,
)

# Params whose setter encodes structured input; routed through it from kwargs.
_STRUCTURED = {"dag": "setDag", "vocabularies": "setVocabularies",
               "nodeTerms": "setNodeTerms", "gateSeedProfile": "setGateSeedProfile"}


def _doc_alpha(doc_conc, K: int) -> np.ndarray:
    """docConcentration -> a length-K alpha (unset -> 1/K; length 1 -> scalar)."""
    if doc_conc is None:
        return np.full(K, 1.0 / K, dtype=np.float64)
    if len(doc_conc) == 1:
        return np.full(K, float(doc_conc[0]), dtype=np.float64)
    if len(doc_conc) != K:
        raise ValueError(
            f"docConcentration vector must have length K={K} (gate layout), "
            f"got {len(doc_conc)}.")
    return np.asarray(doc_conc, dtype=np.float64)


class _TermIndex:
    """Term id -> index in the CONCATENATED vocabulary, over the feature columns'
    vocabularies.

    The ambiguity rule: a bare term id resolves only if exactly one vocabulary holds
    it. A term held by two or more vocabularies must be written as a
    ``(featureColumn, term)`` pair, else ``resolve`` raises naming the term and the
    columns; a pair resolves in that column's vocabulary only. A term in no
    vocabulary (or a pair whose column lacks it) resolves to None (dropped and
    counted by the caller); a pair naming a column that is not a feature column
    raises.

    Terms are matched EXACTLY as given, as node ids are (`10` and `"10"` differ).
    Vocabularies from a vectorizer (`CountVectorizer`, `DagGatedFeaturizer`) are
    STRINGS, so int term ids are passed as their strings there. To keep that from
    failing silently, a term that misses but whose other-typed twin (`str(t)` for an
    int, `int(t)` for a decimal string) is in a vocabulary raises, naming both."""

    def __init__(self, fcols, vocabs, sizes):
        self.fcols = list(fcols)
        offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(int).tolist()
        self.bounds = offsets
        self.by_col = {}
        self.where: dict = {}
        for m, (c, vocab) in enumerate(zip(fcols, vocabs)):
            self.by_col[c] = {t: offsets[m] + i for i, t in enumerate(vocab)}
            for t in vocab:
                self.where.setdefault(t, []).append(c)

    @staticmethod
    def _twin(t):
        if isinstance(t, int):
            return str(t)
        if isinstance(t, str) and t.lstrip("-").isdigit():
            return int(t)
        return None

    def _type_miss(self, t, present):
        tw = self._twin(t)
        if tw is not None and present(tw):
            raise ValueError(
                f"term {t!r} ({type(t).__name__}) is not in the vocabularies, but "
                f"{tw!r} ({type(tw).__name__}) is: terms are matched exactly; "
                f"vectorizer vocabularies hold strings, so pass {tw!r}")

    def resolve(self, ref):
        if isinstance(ref, tuple):
            col, t = ref
            if col not in self.by_col:
                raise ValueError(f"term {ref!r} names column {col!r}, which is not a "
                                 f"feature column ({self.fcols})")
            i = self.by_col[col].get(t)
            if i is None:
                self._type_miss(t, lambda x: x in self.by_col[col])
            return i
        cols = self.where.get(ref)
        if not cols:
            self._type_miss(ref, lambda x: x in self.where)
            return None
        if len(cols) > 1:
            raise ValueError(
                f"term {ref!r} is in the vocabularies of {cols}; write it as a "
                f"(featureColumn, term) pair, e.g. ({cols[0]!r}, {ref!r})")
        return self.by_col[cols[0]][ref]


class DagGatedLDAEstimator(_DagGatedLDAParams, Estimator, DriverParamsWritable):
    """MLlib-shaped Estimator wrapping ``DagGatedOnlineLDA``.

    Required: a DAG (``setDag``) and, on the fit DataFrame, ``labelsCol`` plus the
    feature column(s). The node set (which nodes get a topic block) is learned from
    that DataFrame; there is no train/test notion here (split before ``fit``).

    Defaults are MLlib's where a Param has an analogue on
    ``pyspark.ml.clustering.LDA`` (maxIter 20, learningOffset 1024, learningDecay
    0.51, subsamplingRate 0.05, docConcentration and topicConcentration 1/K,
    but optimizeDocConcentration False: MLlib's True starves deep blocks on deep
    DAGs); the gate's are simple: 10 background topics (the
    flat layout, ``gateTpn=0``, is then LDA's k=10), one topic per node, every block
    open from the first iteration. Each Param's help text (``explainParams()``)
    states its default.

    The **recommended preset** (``RECOMMENDED_PARAMS``; build with
    ``DagGatedLDAEstimator.recommended(dag=..., **overrides)``) is the working
    configuration for a deep DAG on a large corpus::

        gateNBg=200, gateTpn=1, maxIter=60, learningOffset=64, learningDecay=0.51,
        subsamplingRate=0.1, docConcentration=[0.5], optimizeDocConcentration=False,
        gammaShape=100, caviMaxIter=100, caviTol=1e-3,
        gateWarm=2, gateOpenSeed=True, gateSeedMode='balanced', gateSeedLeaves=True

    i.e. a wide background, a short Robbins-Monro warm-up, a fixed alpha, and the
    leaves-first schedule with every block seeded from its documents' term counts
    when it opens. Set ``minDocsPerNode`` for the corpus alongside it (a block needs
    enough documents to estimate a topic; ~100 on large corpora).

    ``resumeFrom`` continues a checkpoint only when the node set learned from the
    given DataFrame is the checkpoint's (same kept nodes, kept DAG and layout);
    otherwise ``fit`` refuses before training, naming the difference.

    Persistence: ``save`` / ``load`` (MLlib's Params metadata, written and read on
    the driver: ``spark_vi.mllib._common.DriverParamsWritable``) keep every Param, the structured ones (the DAG,
    vocabularies, nodeTerms, gateSeedProfile) as the JSON they are stored as, so
    str / int ids survive; the ``setOnIteration`` callback is not persisted."""

    @classmethod
    def recommended(cls, **overrides) -> "DagGatedLDAEstimator":
        """An estimator with the recommended preset (``RECOMMENDED_PARAMS``) set, then
        ``overrides`` on top (any constructor keyword: ``dag``, ``labelsCol``,
        ``minDocsPerNode``, ``seed``, a preset value to change, ...)."""
        kw = {k: (list(v) if isinstance(v, list) else v)
              for k, v in RECOMMENDED_PARAMS.items()}
        kw.update(overrides)
        return cls(**kw)

    @keyword_only
    def __init__(
        self,
        *,
        maxIter: int = 20,
        seed: int | None = None,
        featuresCol: str = "features",
        topicDistributionCol: str = "topicDistribution",
        learningOffset: float = 1024.0,
        learningDecay: float = 0.51,
        subsamplingRate: float = 0.05,
        docConcentration: list[float] | None = None,
        topicConcentration: float | None = None,
        optimizeDocConcentration: bool = False,
        gammaShape: float = 100.0,
        caviMaxIter: int = 100,
        caviTol: float = 1e-3,
        dag=None,
        labelsCol: str = "labels",
        minDocsPerNode: int = 1,
        unknownLabels: str = "error",
        gateNBg: int = 10,
        gateTpn: int = 1,
        gateWarm: int = 0,
        gateOpenSeed: bool = True,
        gateSeedMode: str = "balanced",
        gateSeedLeaves: bool = False,
        featuresCols: list[str] | None = None,
        vocabularies=None,
        nodeTerms=None,
        gateSeedProfile=None,
        engine: str = "spark",
        localProcesses: int = 0,
        # _PersistenceParams kwargs — see that mixin's docstring; these MUST
        # appear here explicitly (not just on the mixin) for kwarg-style
        # construction.
        saveInterval: int = -1,
        saveDir: str = "",
        resumeFrom: str = "",
    ) -> None:
        super().__init__()
        self._setDefault(**_DAG_GATED_LDA_DEFAULTS)
        self._set_persistence_defaults()
        # Diagnostic-only per-iteration callback (mirrors OnlineLDAEstimator).
        # Stored as an instance attribute — callables aren't MLlib-serializable.
        self._on_iteration = None
        self.setParams(**self._input_kwargs)

    @keyword_only
    def setParams(self, **kwargs) -> "DagGatedLDAEstimator":
        # An explicit None leaves a Param unset (its default / isSet semantics).
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        structured = {k: kwargs.pop(k) for k in list(kwargs) if k in _STRUCTURED}
        self._set(**kwargs)
        for k, v in structured.items():
            getattr(self, _STRUCTURED[k])(v)
        return self

    def setOnIteration(
        self, fn: Callable[[int, dict, list[float]], None] | None,
    ) -> "DagGatedLDAEstimator":
        """Register a per-iteration diagnostic callback for the next fit.

        Signature fn(iter_num, global_params, elbo_trace); runs on the driver in
        the fit hot path. Must not mutate global_params. Not persisted."""
        self._on_iteration = fn
        return self

    # ---- input checks -----------------------------------------------------------

    def _domain_sizes(self, dataset, fcols):
        """Each feature column's width: its ML attribute size when the column
        carries one (vectorizer output does), else the first row's vector size.
        Every row is checked against these widths when it is read."""
        for c in fcols:
            if c not in dataset.columns:
                raise ValueError(f"feature column {c!r} is not a column of the input "
                                 f"(columns: {dataset.columns})")
        sizes = []
        for c in fcols:
            md = dataset.schema[c].metadata or {}
            n = (md.get("ml_attr") or {}).get("num_attrs")
            sizes.append(int(n) if n is not None else None)
        if any(s is None for s in sizes):
            first = dataset.select(*fcols).head(1)
            if not first:
                raise ValueError("Cannot fit on an empty DataFrame.")
            sizes = [s if s is not None else int(first[0][i].size)
                     for i, s in enumerate(sizes)]
        return sizes

    def _checked_vocabularies(self, fcols, sizes):
        vocabs = self.getVocabularies()
        if not vocabs:
            return None
        if len(vocabs) != len(fcols):
            raise ValueError(f"vocabularies has {len(vocabs)} list(s) for "
                             f"{len(fcols)} feature column(s) {fcols}")
        for c, v, n in zip(fcols, vocabs, sizes):
            if len(v) != n:
                raise ValueError(f"the vocabulary of {c!r} has {len(v)} terms but its "
                                 f"vectors have size {n}")
        return vocabs

    # ---- seed extras: user ids -> engine ids and concatenated vocab indices ------

    def _translate_seed_extras(self, ns: NodeSet, fcols, sizes, vocabs):
        """(own, profile, summary): `nodeTerms` -> `{engine id: [concat idx]}` and
        `gateSeedProfile` -> `{engine id: {concat idx: weight}}`, the forms the engine's
        seeding reads.

        A node that is not kept (pruned, spliced, or not a DAG node) has its entries
        DROPPED rather than moved to where its labels land: its documents now count
        as its landing nodes' own documents, so its terms belong in their seeds
        rather than being removed from them (nodeTerms), and a profile describes the
        node it names, not its ancestor (gateSeedProfile). Terms in no vocabulary are
        dropped too; both are counted in the summary."""
        nt, prof = self.getNodeTerms(), self.getGateSeedProfile()
        if not nt and not prof:
            return {}, {}, ""
        if vocabs is None:
            raise ValueError("nodeTerms / gateSeedProfile name terms: set vocabularies "
                             "(one list of term ids per feature column)")
        ti = _TermIndex(fcols, vocabs, sizes)
        parts = []

        def resolve_nodes(extra, name, terms_of):
            out, n_unkept, n_terms_dropped = {}, 0, 0
            for node, entry in extra.items():
                e = ns.engine_id.get(node)
                if e is None:
                    n_unkept += 1
                    continue
                got = {}
                for ref, w in terms_of(entry):
                    i = ti.resolve(ref)
                    if i is None:
                        n_terms_dropped += 1
                    else:
                        got[i] = w
                if got:
                    out[e] = got
            parts.append(f"{name}: {len(out)} node(s) used, {n_unkept} not kept "
                         f"dropped, {n_terms_dropped} term(s) in no vocabulary dropped")
            return out

        own = {e: sorted(d) for e, d in resolve_nodes(
            nt, "nodeTerms", lambda ts: ((t, 1.0) for t in ts)).items()} if nt else {}
        pro = resolve_nodes(prof, "gateSeedProfile",
                            lambda tw: tw.items()) if prof else {}
        return own, pro, "; ".join(parts)

    def _gate_seeds(self, rdd, lay, vocab_size, bounds, own, prof):
        """(open_seed, init_seed, summary) for the leaves-first schedule.

        The open-time seed is one pass over the cached document RDD
        (`gated_init.gate_open_seed_counts`), with the balanced mode reading `own`
        (`{engine id: [concat idx]}`); with gateSeedLeaves the leaf rows move to the
        iteration-0 seed. A profile (`{engine id: {concat idx: weight}}`) replaces a
        profiled node's seed in each domain (`bounds` slice) its terms touch: leaves
        at iteration 0, later blocks when they open; other domains keep the count
        seed / init."""
        seed = None
        mode = str(self.getOrDefault("gateSeedMode"))
        leaves = bool(self.getOrDefault("gateSeedLeaves"))
        if bool(self.getOrDefault("gateOpenSeed")):
            from spark_vi.models.topic.gated_init import gate_open_seed_counts
            seed = gate_open_seed_counts(rdd, lay, vocab_size, mode=mode,
                                         own_idx=(own if mode == "balanced" else None),
                                         leaves=leaves)
        init = None
        if seed is not None and leaves:
            # Leaf rows go in at iteration 0; the open-time seed keeps the rest.
            init = np.zeros_like(seed)
            for u in lay.nodes:
                if lay.height(u) == 0:
                    for k in lay.block[u]:
                        init[k] = seed[k]
                        seed[k] = 0.0
        n_leaf = n_late = 0
        if prof:
            if init is None:
                init = np.zeros((lay.K, vocab_size), dtype=np.float64)
            for u, w in prof.items():
                row = np.zeros(vocab_size)
                for i, x in w.items():
                    row[i] = x
                doms = sorted({int(np.searchsorted(bounds, i, side="right")) - 1
                               for i in w})
                for k in lay.block[u]:
                    if lay.height(u) == 0:
                        tgt = init
                        n_leaf += 1
                    else:
                        if seed is None:
                            seed = np.zeros((lay.K, vocab_size), dtype=np.float64)
                        tgt = seed
                        n_late += 1
                    for m in doms:
                        lo, hi = bounds[m], bounds[m + 1]
                        tgt[k, lo:hi] = row[lo:hi]
            if not n_leaf and not leaves:
                init = None
        summary = (f"open-time seed "
                   f"{'ON (' + mode + ')' if seed is not None else 'off'}"
                   + ("; leaf blocks seeded at iteration 0 from their own documents"
                      if leaves else "")
                   + (f"; profile seed on {n_leaf} leaf + {n_late} later topic(s), "
                      "count seed elsewhere" if prof else ""))
        return seed, init, summary

    # ---- resume -------------------------------------------------------------------

    def _check_resume_node_set(self, ns: NodeSet, n_bg: int, tpn: int) -> None:
        """With ``resumeFrom`` set: refuse, before any training, a checkpoint whose
        node set is not the one just learned from the data.

        The checkpoint's lambda rows are topic blocks of ITS kept nodes, in its
        engine-id order; continuing them on another node set would train each block
        on another node's documents (or fail on a shape inside the engine). So the
        kept nodes, their engine order and kept parents, and the layout (gateNBg,
        gateTpn) must all match. Only the manifest is read (no arrays). A missing
        manifest is left to ``apply_persistence_params``, which names it."""
        resume = str(self.getOrDefault("resumeFrom") or "")
        if not resume:
            return
        from spark_vi.io import fs
        mpath = fs.join(resume, "manifest.json")
        if not fs.exists(mpath):
            return
        md = (json.loads(fs.read_text(mpath)).get("metadata") or {})
        saved = md.get(_NODE_SET_KEY)
        where = f"resumeFrom={resume!r}"
        if not saved:
            raise ValueError(f"{where}: the checkpoint carries no node set "
                             f"(metadata {_NODE_SET_KEY!r}); it was not written by "
                             f"DagGatedLDAEstimator, so it cannot be resumed here")
        old = NodeSet.from_dict(saved)
        old_ids, new_ids = old.user_ids[1:], ns.user_ids[1:]
        fix = ("resume on the data, DAG, minDocsPerNode and unknownLabels the "
               "checkpoint was fitted with, or fit without resumeFrom")
        if set(old_ids) != set(new_ids):
            added = sorted(set(new_ids) - set(old_ids), key=node_sort_key)
            removed = sorted(set(old_ids) - set(new_ids), key=node_sort_key)
            raise ValueError(
                f"{where}: the node set learned from this data is not the "
                f"checkpoint's: {len(added)} kept node(s) added (e.g. {added[:5]}), "
                f"{len(removed)} removed (e.g. {removed[:5]}); the checkpoint keeps "
                f"{len(old_ids)}, this data {len(new_ids)}. Its topic blocks belong to "
                f"its nodes: {fix}")
        if old_ids != new_ids or old.engine_parents != ns.engine_parents:
            moved = [n for n in new_ids
                     if tuple(old.parents.get(n, ())) != tuple(ns.parents[n])]
            raise ValueError(
                f"{where}: the same {len(new_ids)} nodes are kept, but the kept DAG "
                f"differs from the checkpoint's ({len(moved)} node(s) with other kept "
                f"parents, e.g. {moved[:5]}): {fix}")
        lay_old = (int(saved.get("n_bg", -1)), int(saved.get("tpn", -1)))
        if lay_old != (n_bg, tpn):
            raise ValueError(
                f"{where}: the checkpoint's layout is gateNBg={lay_old[0]}, "
                f"gateTpn={lay_old[1]}; this estimator has gateNBg={n_bg}, "
                f"gateTpn={tpn}. Set them to the checkpoint's to resume.")

    # ---- fit --------------------------------------------------------------------

    def _fit_local(self, engine, config, rdd, resume_path):
        """The SVI loop in the driver's memory (engine 'local'): the documents are
        collected once (``toLocalIterator``: a partition at a time, so
        ``spark.driver.maxResultSize`` does not apply) and fitted by ``LocalRunner``
        over ``localProcesses`` forked workers. The same model, schedule, seeds and
        loop as the Spark engine; the mini-batches are drawn by NumPy instead of
        ``RDD.sample``, so the fit agrees in distribution, not bit for bit."""
        import time

        from spark_vi.core.local_runner import LocalRunner
        t0 = time.perf_counter()
        rows = list(rdd.toLocalIterator())
        runner = LocalRunner(engine, config=config,
                             processes=int(self.getOrDefault("localProcesses")))
        print(f"{_LOG} local engine: {len(rows)} documents collected to the driver in "
              f"{time.perf_counter() - t0:.1f}s; E-step over {runner.processes} "
              f"process(es)", flush=True)
        return runner.fit(rows, resume_from=resume_path,
                          on_iteration=self._on_iteration)

    def _fit(self, dataset) -> "DagGatedLDAModel":
        from spark_vi.core.runner import VIRunner

        fit_engine = str(self.getOrDefault("engine"))
        if fit_engine not in FIT_ENGINES:
            raise ValueError(f"engine must be one of {FIT_ENGINES}; got {fit_engine!r}")
        parsed = self._parsed_dag()
        policy = str(self.getOrDefault("unknownLabels"))
        if policy not in UNKNOWN_LABEL_POLICIES:
            raise ValueError(f"unknownLabels must be one of {UNKNOWN_LABEL_POLICIES}; "
                             f"got {policy!r}")
        labels_col = str(self.getOrDefault("labelsCol"))
        check_labels_column(dataset, labels_col, parsed.nodes)
        fcols, multi = self._feature_columns()
        sizes = self._domain_sizes(dataset, fcols)
        vocabs = self._checked_vocabularies(fcols, sizes)
        vocab_size = int(sum(sizes))
        bounds = np.concatenate([[0], np.cumsum(sizes)]).astype(int).tolist()

        # R3: support over the fit DataFrame, then prune / land / reduce / splice.
        support, n_docs = count_support_spark(dataset, labels_col, parsed,
                                              unknown_labels=policy)
        if n_docs == 0:
            raise ValueError("Cannot fit on an empty DataFrame.")
        ns = learn_node_set(parsed, support, n_docs=n_docs,
                            min_docs_per_node=int(self.getOrDefault("minDocsPerNode")),
                            unknown_labels=policy)
        print(f"{_LOG} node set: {n_docs} docs; nodes given {len(parsed)} / kept "
              f"{len(ns.nodes)} / pruned {len(ns.pruned)} / spliced {len(ns.spliced)} "
              f"(minDocsPerNode={ns.min_docs_per_node})", flush=True)

        n_bg, tpn = int(self.getOrDefault("gateNBg")), int(self.getOrDefault("gateTpn"))
        self._check_resume_node_set(ns, n_bg, tpn)
        lay = DagLayout(ns.engine_parents, n_bg=n_bg, tpn=tpn)
        doc_conc = (self.getOrDefault("docConcentration")
                    if self.isSet("docConcentration") else None)
        if self.isSet("topicConcentration"):
            eta = float(self.getOrDefault("topicConcentration"))
            if not eta > 0:
                raise ValueError(f"topicConcentration must be > 0; got {eta}")
        else:
            eta = 1.0 / lay.K
        seed = self.getOrDefault("seed") if self.isSet("seed") else None
        engine = DagGatedOnlineLDA(
            lay, vocab_size,
            alpha=_doc_alpha(doc_conc, lay.K), eta=eta,
            domains=(sizes if multi else None),      # None = single fused vocab
            gamma_shape=self.getOrDefault("gammaShape"),
            cavi_max_iter=self.getOrDefault("caviMaxIter"),
            cavi_tol=self.getOrDefault("caviTol"), random_seed=seed)
        engine.extra_metadata = {_NODE_SET_KEY: {**ns.to_dict(), "n_bg": n_bg,
                                                 "tpn": tpn}}
        mbf = float(self.getOrDefault("subsamplingRate"))
        config = VIConfig(
            max_iterations=self.getOrDefault("maxIter"),
            learning_rate_tau0=self.getOrDefault("learningOffset"),
            learning_rate_kappa=self.getOrDefault("learningDecay"),
            mini_batch_fraction=(mbf if 0.0 < mbf < 1.0 else None),
            random_seed=seed,
        )
        # Validate persistence Params and splice checkpoint_dir/interval into
        # VIConfig; resume_path is a Path or None (continues the RM counter).
        config, resume_path = apply_persistence_params(self, config)

        # The labels -> engine frontier map rides a BROADCAST (ADR 0047: no
        # structure in the task closure; the closure holds only the handle).
        sc = dataset.sparkSession.sparkContext
        frontier_bc = sc.broadcast(ns.frontier)

        def _to_doc(row, _fcs=tuple(fcols), _ds=tuple(sizes), _bc=frontier_bc,
                    _lc=labels_col):
            return _row_to_gated_document(row, _fcs, _ds, _bc.value, _lc)

        rdd = (dataset.select(*fcols, labels_col).rdd.map(_to_doc)
               .persist(StorageLevel.MEMORY_AND_DISK))
        try:
            rdd.count()  # materialize for VIRunner's strict cache precondition
            # Leaves-first gate schedule.
            gw = int(self.getOrDefault("gateWarm"))
            if gw > 0:
                own, prof, extras = self._translate_seed_extras(ns, fcols, sizes, vocabs)
                if extras:
                    print(f"{_LOG} seed extras: {extras}", flush=True)
                open_seed, init_seed, summary = self._gate_seeds(
                    rdd, lay, vocab_size, bounds, own, prof)
                engine.set_gate_schedule(gw, open_seed=open_seed, init_seed=init_seed)
                print(f"{_LOG} leaves-first gate: {gw} iters/level over "
                      f"heights 0..{engine._gate_H} (full gate from iter "
                      f"{gw * engine._gate_H}); {summary}", flush=True)
            # Per-node tied empirical-Bayes alpha: needs the static
            # frontier histogram, which only exists once the document RDD does.
            if bool(self.getOrDefault("optimizeDocConcentration")):
                hist = {fr: int(n) for fr, n in
                        rdd.map(lambda d: d.frontier).countByValue().items()}
                engine.set_alpha_policy(hist, optimize=True)
                print(f"{_LOG} empirical-Bayes alpha on ({len(hist)} "
                      f"frontier groups)", flush=True)
            if fit_engine == "local":
                result = self._fit_local(engine, config, rdd, resume_path)
            else:
                result = VIRunner(engine, config=config).fit(
                    rdd, resume_from=resume_path, on_iteration=self._on_iteration)
        finally:
            rdd.unpersist(blocking=False)
            frontier_bc.unpersist(blocking=False)

        out_model = DagGatedLDAModel(result)
        for param in self.params:
            if self.isSet(param):
                out_model._set(**{param.name: self.getOrDefault(param)})
            elif self.hasDefault(param):
                out_model._setDefault(**{param.name: self.getOrDefault(param)})
        return out_model


class DagGatedLDAModel(_DagGatedLDAParams, _StagePersistence, Model):
    """MLlib-shaped Model wrapping a trained DagGatedOnlineLDA VIResult.

    ``transform`` appends ``topicDistributionCol`` (theta) by folding each
    document in UNGATED over all K topics with the same CAVI the engine uses at
    train time. The node-keyed accessors (``dag``, ``prunedNodes``, ``splicedNodes``,
    ``unsupportedNodes``, ``nodeSupport``, ``nodeTopics``, ``backgroundTopics``,
    ``describeNodes``, ``nodeStatistics``) answer in the user's node ids; engine ids
    stay internal. ``dropUnsupportedNodes`` returns a new model without the nodes
    whose blocks are flat (spec R14).

    Persistence (MLlib's, spec R13): ``save`` / ``write().overwrite().save`` /
    ``load`` and inside a ``PipelineModel``, on any Hadoop-filesystem URI. Layout::

        <path>/metadata/   Params (DefaultParamsWriter), written last
        <path>/result/     the VIResult, in the fit checkpoints' format
                           (spark_vi.io.export.save_result: manifest.json,
                           params/*.npy, traces/*.npy); the node set rides its
                           metadata (dag_node_set)

    ``load`` refuses a result stamped with another model class. A fit checkpoint
    (``saveDir``) holds only the VIResult: ``DagGatedLDAModel(load_result(saveDir))``
    wraps it (Params at their defaults)."""

    _expected_model_class = "DagGatedOnlineLDA"

    def __init__(self, result) -> None:  # result: VIResult
        super().__init__()
        self._result = result
        self._setDefault(**_DAG_GATED_LDA_DEFAULTS)
        self._set_persistence_defaults()
        md = (result.metadata or {}).get(_NODE_SET_KEY)
        self._node_set = NodeSet.from_dict(md) if md else None
        self._n_bg = int(md["n_bg"]) if md else None
        self._tpn = int(md["tpn"]) if md else None

    @property
    def result(self):
        """The trained VIResult (global_params, elbo_trace, n_iterations, ...)."""
        return self._result

    # ---- persistence (_StagePersistence hooks) ----------------------------------

    def _save_data(self, path: str) -> None:
        from spark_vi.io import fs
        from spark_vi.io.export import save_result
        save_result(self._result, fs.join(path, "result"))

    @classmethod
    def _load_data(cls, path: str, metadata: dict) -> "DagGatedLDAModel":
        from spark_vi.io import fs
        from spark_vi.io.export import load_result
        result = load_result(fs.join(path, "result"))
        saved = (result.metadata or {}).get("model_class")
        if saved != cls._expected_model_class:
            raise ValueError(f"{path}: the saved result is a {saved!r}, not a "
                             f"{cls._expected_model_class!r}")
        return cls(result)

    # ---- node-keyed outputs (R5), user ids throughout ---------------------------

    def _ns(self) -> NodeSet:
        if self._node_set is None:
            raise ValueError("this model carries no node set (its VIResult metadata "
                             f"has no {_NODE_SET_KEY!r}); it was not fitted by "
                             "DagGatedLDAEstimator")
        return self._node_set

    def _frontier_fn(self):
        """The fit's labels -> engine-frontier policy (landing of pruned / spliced
        labels, frontier reduction, unknownLabels): internal, for the readout heads."""
        return self._ns().frontier

    def _layout(self) -> DagLayout:
        """The engine layout (engine ids): internal."""
        return DagLayout(self._ns().engine_parents, n_bg=self._n_bg, tpn=self._tpn)

    def dag(self) -> dict:
        """The kept DAG, transitively reduced after pruning and splicing:
        ``{node: [parents]}``, ``[]`` for a top-level node."""
        return {n: [p for p in ps if p is not ROOT]
                for n, ps in self._ns().parents.items()}

    def prunedNodes(self) -> dict:
        """``{dropped node: [landing nodes]}`` for the nodes pruned by
        ``minDocsPerNode``: their labels landed on their nearest kept ancestors. ``[]``
        means no kept ancestor: that label was dropped, and a document whose only
        labels land nowhere is background. (A dropped TOP-level node can still have
        documents through a kept descendant; those documents are not background.)"""
        return {n: list(ls) for n, ls in self._ns().pruned.items()}

    def splicedNodes(self) -> list:
        """The nodes removed as single-child chain links (each admitted exactly its
        one kept child's documents, so its block would have trained on identical
        data); their labels land on their nearest kept ancestors."""
        return list(self._ns().spliced)

    def unsupportedNodes(self) -> dict:
        """``{dropped node: [landing nodes]}`` for the nodes removed by
        ``dropUnsupportedNodes`` (their blocks were flat: the data did not sustain
        them); their labels land on their nearest kept ancestors, ``[]`` = nowhere.
        Empty on a model never cleaned."""
        return {n: list(ls) for n, ls in self._ns().unsupported.items()}

    def nodeSupport(self) -> dict:
        """``{kept node: number of training documents whose label closure reaches
        it}``."""
        return dict(self._ns().support)

    def nodeTopics(self, node) -> list:
        """The topic indices (into the K of ``describeTopics`` / ``transform``) of a
        kept node's block. KeyError names a node that is not kept."""
        ns = self._ns()
        e = ns.engine_id.get(node)
        if e is None:
            why = ("was pruned" if node in ns.pruned else
                   "was spliced" if node in ns.spliced else
                   "was dropped as unsupported" if node in ns.unsupported else
                   "is not a DAG node")
            raise KeyError(f"node {node!r} has no topic block: it {why}")
        lo = self._n_bg + (e - 1) * self._tpn
        return list(range(lo, lo + self._tpn))

    def backgroundTopics(self) -> list:
        """The shared background topics' indices (``0 .. gateNBg-1``)."""
        self._ns()
        return list(range(self._n_bg))

    def describeNodes(self, maxTermsPerTopic: int = 10):
        """DataFrame ``(node, topic, termIndices, termWeights, terms)``: one row per
        topic (background topics have a null node), top terms as in
        ``describeTopics``.

        ``node`` is a LONG column when every kept node id is an int, a STRING column
        when every one is a str, and a STRING column (ints written in decimal) when
        the ids are mixed; there ``1`` and ``"1"`` render alike, so use ``nodeTopics``
        to tell them apart.

        ``terms`` (``array<string>``, aligned with ``termIndices``) names each term
        from ``vocabularies``: the term id as a string with one feature column, and
        ``"<featureColumn>:<term>"`` with several (``termIndices`` index the
        concatenated vocabulary, so the column says which domain a term is from;
        split at the first ``:`` when column names have none). Null on every row
        when ``vocabularies`` is not set."""
        from pyspark.sql import SparkSession
        from pyspark.sql.types import (ArrayType, DoubleType, IntegerType, StringType,
                                       StructField, StructType)
        ns = self._ns()
        node_type, as_cell = _node_column(ns.nodes)
        owner = {}
        for n in ns.nodes:
            for k in self.nodeTopics(n):
                owner[k] = as_cell(n)
        names = self._term_names()
        rows = [(owner.get(k),) + r
                + ((None,) if names is None else ([names[i] for i in r[1]],))
                for k, r in enumerate(self._top_terms(maxTermsPerTopic))]
        schema = StructType([
            StructField("node", node_type, True),
            StructField("topic", IntegerType(), False),
            StructField("termIndices", ArrayType(IntegerType(), False), False),
            StructField("termWeights", ArrayType(DoubleType(), False), False),
            StructField("terms", ArrayType(StringType(), False), True),
        ])
        return SparkSession.builder.getOrCreate().createDataFrame(rows, schema=schema)

    # ---- node statistics and dropping unsupported nodes (R14) -------------------

    def _node_stat_rows(self, flatThreshold) -> list:
        """One tuple per kept node, in engine order: ``(node, depth, support,
        supportFraction, mass, massToParent, flat)`` (see ``nodeStatistics``)."""
        from spark_vi.eval.topic.sharpness import support_fraction
        thr = float(flatThreshold)
        ns, lay = self._ns(), self._layout()
        sf, mass = support_fraction(self._result.global_params["lambda"])
        bmass = {e: float(sum(mass[t] for t in lay.block[e])) for e in lay.nodes}
        rows = []
        for e, u in enumerate(ns.nodes, start=1):
            blk = lay.block[e]
            frac = float(min(sf[t] for t in blk)) if blk else None
            par = [p for p in ns.engine_parents[e] if p != 0]
            heaviest = max((bmass[p] for p in par), default=None)
            m2p = (bmass[e] / heaviest if heaviest else None)
            rows.append((u, int(lay.depth(e)), int(ns.support[u]), frac, bmass[e], m2p,
                         bool(frac is not None and frac > thr)))
        return rows

    def nodeStatistics(self, flatThreshold: float = 0.5):
        """DataFrame, one row per kept node in engine order (spec R14.1): how well the
        data sustains each node's topic block. Reporting only: the model is
        unchanged (``dropUnsupportedNodes`` acts on it).

        - ``node``: the node id, typed as in ``describeNodes`` (long when every id is
          an int, string otherwise);
        - ``depth`` (int): the longest path from the synthetic root (top level = 1);
        - ``support`` (long): training documents whose labels' closure reaches the
          node (``nodeSupport()``);
        - ``supportFraction`` (double): the block's effective support exp(H(β)) over
          its dominant domain's vocabulary (``spark_vi.eval.topic.sharpness``): about
          0 = sharp, 1 = flat at the topicConcentration prior. With ``gateTpn > 1``
          the MINIMUM over the block's topics (a block is flat only when every topic
          in it is). Null when the layout gives nodes no topics (``gateTpn`` 0);
        - ``mass`` (double): the block's λ mass (pseudo-counts plus prior);
        - ``massToParent`` (double): ``mass`` over its heaviest kept parent's; null
          for a top-level node;
        - ``flat`` (boolean): ``supportFraction > flatThreshold``.

        A flat block is one the data did not renew: under SVI an unrenewed block
        decays to the prior, whatever it was seeded with."""
        from pyspark.sql import SparkSession
        from pyspark.sql.types import (BooleanType, DoubleType, IntegerType,
                                       LongType, StructField, StructType)
        ns = self._ns()
        node_type, as_cell = _node_column(ns.nodes)
        rows = [(as_cell(r[0]),) + r[1:] for r in self._node_stat_rows(flatThreshold)]
        schema = StructType([
            StructField("node", node_type, False),
            StructField("depth", IntegerType(), False),
            StructField("support", LongType(), False),
            StructField("supportFraction", DoubleType(), True),
            StructField("mass", DoubleType(), False),
            StructField("massToParent", DoubleType(), True),
            StructField("flat", BooleanType(), False),
        ])
        return SparkSession.builder.getOrCreate().createDataFrame(rows, schema=schema)

    def dropUnsupportedNodes(self, maxSupportFraction: float = 0.5) -> "DagGatedLDAModel":
        """A NEW model without the nodes whose block is flat (``nodeStatistics(
        flatThreshold=maxSupportFraction)``'s ``flat``); this model is unchanged
        (spec R14.2).

        The node set loses them as pruning would have removed them
        (``spark_vi.dag.nodeset.drop_nodes``): their labels land on their nearest kept
        ancestors (nowhere when none), their kept children re-attach to their nearest
        kept ancestors and the kept DAG is re-reduced, pruned / spliced nodes that
        landed on them land further up, and the engine ids are renumbered. Every
        K-sized fitted array keeps only the surviving topics, background first, then
        the surviving blocks in the NEW engine order: λ (per domain), α, the gate's
        ``gate_open_at`` (and the α trace). Scalars (η, ``gate_iter``) are kept. No
        refit and no re-splice: a surviving block is exactly as fitted, so the
        cleaned model scores documents with the surviving topics' λ unchanged.

        Reported by ``unsupportedNodes()`` and persisted with the node set; a dropped
        node gets no topic block, no gate and no readout head (``DagHeadsEstimator.
        setTopicModel`` reads this model's node set). With nothing flat, an
        equivalent copy."""
        flat = [r[0] for r in self._node_stat_rows(maxSupportFraction) if r[-1]]
        return self._without_nodes(flat)

    def _without_nodes(self, nodes) -> "DagGatedLDAModel":
        """The model with the kept nodes `nodes` dropped (see
        ``dropUnsupportedNodes``); any kept nodes, flat or not."""
        from spark_vi.core.result import VIResult
        ns = self._ns()
        n_bg, tpn = self._n_bg, self._tpn
        new_ns = drop_nodes(ns, nodes)
        K_old = n_bg + len(ns.nodes) * tpn
        keep = list(range(n_bg))
        for u in new_ns.nodes:
            lo = n_bg + (ns.engine_id[u] - 1) * tpn
            keep.extend(range(lo, lo + tpn))
        idx = np.asarray(keep, dtype=np.int64)

        def topics(arr, what):
            a = np.asarray(arr)
            if a.ndim == 0:
                return a.copy()
            if a.shape[0] != K_old:
                raise ValueError(f"{what} has leading size {a.shape[0]}, not the "
                                 f"model's K={K_old}")
            return a[idx]

        gp = {}
        for key, v in self._result.global_params.items():
            if key == "lambda":
                gp[key] = ({m: topics(lam, f"lambda[{m}]") for m, lam in v.items()}
                           if isinstance(v, dict) else topics(v, "lambda"))
            elif key in ("alpha", "gate_open_at"):
                gp[key] = topics(v, key)
            elif np.ndim(v) == 0:
                gp[key] = np.array(v)                      # eta, gate_iter
            else:
                raise ValueError(f"global parameter {key!r} has shape {np.shape(v)}; "
                                 "dropping nodes knows only lambda, alpha, "
                                 "gate_open_at and scalars")
        traces = dict(self._result.diagnostic_traces or {})
        if "alpha" in traces:
            traces["alpha"] = [topics(a, "the alpha trace")
                               if np.ndim(a) == 1 and np.shape(a)[0] == K_old else a
                               for a in traces["alpha"]]
        md = dict(self._result.metadata or {})
        md[_NODE_SET_KEY] = {**new_ns.to_dict(), "n_bg": n_bg, "tpn": tpn}
        if "K" in md:
            md["K"] = int(idx.size)
        res = VIResult(global_params=gp, elbo_trace=list(self._result.elbo_trace),
                       n_iterations=self._result.n_iterations,
                       converged=self._result.converged, metadata=md,
                       diagnostic_traces=traces)
        return self._copyValues(DagGatedLDAModel(res))

    def _term_names(self):
        """The concatenated vocabulary as display strings (see ``describeNodes``), or
        None without ``vocabularies``. Checked against the trained widths."""
        vocabs = self.getVocabularies()
        if not vocabs:
            return None
        fcols, _ = self._feature_columns()
        if len(vocabs) != len(fcols):
            raise ValueError(f"vocabularies has {len(vocabs)} list(s) for "
                             f"{len(fcols)} feature column(s) {fcols}")
        if sum(len(v) for v in vocabs) != self.vocabSize():
            raise ValueError(f"the vocabularies hold {sum(len(v) for v in vocabs)} "
                             f"terms but the model's vocabulary size is "
                             f"{self.vocabSize()}")
        multi = len(fcols) > 1
        return [f"{c}:{t}" if multi else str(t)
                for c, v in zip(fcols, vocabs) for t in v]

    # ---- topic-indexed outputs ---------------------------------------------------

    def vocabSize(self) -> int:
        """V of the trained lambda (summed over domains when multi-domain)."""
        lam = self._result.global_params["lambda"]
        if isinstance(lam, dict):
            return int(sum(v.shape[1] for v in lam.values()))
        return int(lam.shape[1])

    def _beta(self) -> np.ndarray:
        """(K, V) row-normalized topic-word mean, per domain when multi-domain
        (each domain block normalized on its own row, then concatenated)."""
        lam = self._result.global_params["lambda"]
        if isinstance(lam, dict):
            return np.concatenate(
                [lam[m] / lam[m].sum(axis=1, keepdims=True) for m in sorted(lam)],
                axis=1)
        return lam / lam.sum(axis=1, keepdims=True)

    def topicsMatrix(self):
        """Topic-word distribution as an MLlib DenseMatrix of shape (V, K)."""
        from pyspark.ml.linalg import DenseMatrix
        beta = self._beta()
        K, V = beta.shape
        return DenseMatrix(numRows=V, numCols=K, values=beta.T.flatten("F").tolist())

    def trainedAlpha(self) -> np.ndarray:
        """Trained α vector (length K). Method, not @property, to avoid colliding
        with the ``docConcentration`` Param descriptor (ADR 0012)."""
        return self._result.global_params["alpha"]

    def _top_terms(self, maxTermsPerTopic):
        if maxTermsPerTopic < 1:
            raise ValueError(f"maxTermsPerTopic must be >= 1, got {maxTermsPerTopic}")
        beta = self._beta()
        K, V = beta.shape
        m = min(maxTermsPerTopic, V)
        rows = []
        for k in range(K):
            order = np.argsort(beta[k])[::-1][:m]
            rows.append((int(k), [int(i) for i in order],
                         [float(beta[k, i]) for i in order]))
        return rows

    def describeTopics(self, maxTermsPerTopic: int = 10):
        """DataFrame of (topic, termIndices, termWeights) — top terms per topic
        (term indices in the concatenated vocabulary when multi-domain)."""
        from pyspark.sql import SparkSession
        from pyspark.sql.types import (
            ArrayType, DoubleType, IntegerType, StructField, StructType,
        )
        schema = StructType([
            StructField("topic", IntegerType(), False),
            StructField("termIndices", ArrayType(IntegerType(), False), False),
            StructField("termWeights", ArrayType(DoubleType(), False), False),
        ])
        return SparkSession.builder.getOrCreate().createDataFrame(
            self._top_terms(maxTermsPerTopic), schema=schema)

    def _transform(self, dataset):

        from pyspark.ml.linalg import DenseVector, VectorUDT
        from pyspark.sql import functions as F

        from spark_vi.models.topic.lda import _cavi_doc_inference, _hashed_gamma_init

        lam = self._result.global_params["lambda"]
        # Multi-domain: λ is a per-domain dict {m:(K,V_m)}; the CAVI reads the
        # concatenated (K, V) expElogbeta (each domain normalized on its own row).
        domain_sizes = ([int(lam[m].shape[1]) for m in sorted(lam)]
                        if isinstance(lam, dict) else None)
        expElogbeta = expElogbeta_from_lambda(lam)
        K = expElogbeta.shape[0]

        sc = dataset.sparkSession.sparkContext
        bcast = sc.broadcast({
            "expElogbeta": expElogbeta, "alpha": self._result.global_params["alpha"],
            "gamma_shape": float(self.getOrDefault("gammaShape")),
            "cavi_max_iter": int(self.getOrDefault("caviMaxIter")),
            "cavi_tol": float(self.getOrDefault("caviTol")), "K": K,
            "domain_sizes": domain_sizes,
        })

        def _infer(*features):
            p = bcast.value
            if p["domain_sizes"] is not None:
                indices, counts = _concat_domain_features(features, p["domain_sizes"])
            else:
                doc = _vector_to_bow_document(features[0])
                indices, counts = doc.indices, doc.counts
            # Content-deterministic gamma_init: identical docs get identical init on
            # every run, so scoring is reproducible and independent of Spark
            # partition/executor order.
            gamma_init = _hashed_gamma_init(indices, counts, p["K"], p["gamma_shape"])
            gamma, _, _, _ = _cavi_doc_inference(
                indices=indices, counts=counts,
                expElogbeta=p["expElogbeta"], alpha=p["alpha"],
                gamma_init=gamma_init,
                max_iter=p["cavi_max_iter"], tol=p["cavi_tol"],
            )
            return DenseVector(gamma / gamma.sum())

        infer_udf = F.udf(_infer, returnType=VectorUDT())
        out_col = self.getOrDefault("topicDistributionCol")
        fcols, _ = self._feature_columns()
        # Broadcast lifetime is the returned DataFrame's (its UDF closure holds
        # bcast); ContextCleaner reclaims it on GC — do NOT eagerly unpersist.
        return dataset.withColumn(out_col, infer_udf(*[F.col(c) for c in fcols]))

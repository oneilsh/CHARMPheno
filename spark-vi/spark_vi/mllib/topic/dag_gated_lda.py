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
``describeNodes()``. The node set rides the VIResult's metadata
(``dag_node_set``), so ``save`` / ``load`` keep it.
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
    learn_node_set,
    node_sort_key,
    parse_dag,
)
from spark_vi.mllib._common import (
    _PersistableModel,
    _PersistenceParams,
    apply_persistence_params,
)
from spark_vi.mllib.topic._common import _concat_domain_features, _vector_to_bow_document
from spark_vi.models.topic.dag_gated_lda import (
    DagGatedOnlineLDA,
    expElogbeta_from_lambda,
)
from spark_vi.models.topic.types import GatedBOWDocument

_LOG = "[dag-gated-lda]"
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
    ``dag`` (routed through ``setDag``)."""

    topicDistributionCol = Param(
        Params._dummy(), "topicDistributionCol",
        "output column with the topic mixture (theta) for each document",
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
        "fraction of corpus sampled per mini-batch (<= 0 or >= 1 = full batch)",
        typeConverter=TypeConverters.toFloat,
    )
    docConcentration = Param(
        Params._dummy(), "docConcentration",
        "Dirichlet concentration alpha on theta; scalar (symmetric) or length-K "
        "vector. Unset = 1/K, which is razor-small when K is in the thousands "
        "(0.5 is a sensible start there).",
        typeConverter=TypeConverters.toListFloat,
    )
    topicConcentration = Param(
        Params._dummy(), "topicConcentration",
        "Dirichlet concentration eta on each topic's term distribution (the prior on "
        "lambda; one scalar for every domain). Unset = 1/K, as MLlib's online LDA.",
        typeConverter=TypeConverters.toFloat,
    )
    optimizeDocConcentration = Param(
        Params._dummy(), "optimizeDocConcentration",
        "learn the per-node tied empirical-Bayes alpha (the gated Newton step) "
        "from the static frontier histogram. Default False.",
        typeConverter=TypeConverters.toBoolean,
    )
    gammaShape = Param(
        Params._dummy(), "gammaShape",
        "shape parameter for the Gamma init of gamma/lambda (ADR 0008 default 100.0)",
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
    dagEdges = Param(
        Params._dummy(), "dagEdges",
        "the label DAG, set with setDag (a dict {child: parent | [parents]}, or a "
        "pandas / Spark DataFrame (child, parent)); stored validated as a JSON edge "
        "list [[child, parent | null], ...] in the user's ids. Parentless nodes hang "
        "from a synthetic root. Required.",
        typeConverter=TypeConverters.toString,
    )
    labelsCol = Param(
        Params._dummy(), "labelsCol",
        "array<string> or array<int> column of the DAG nodes each document attests "
        "(any nodes: ancestors and duplicates allowed; empty or null = background). "
        "The element type must match the DAG's ids. Default 'labels'.",
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
        "a label that is not a DAG node: 'error' (default; refused by name) or "
        "'ignore' (dropped; a document left with no label is background).",
        typeConverter=TypeConverters.toString,
    )
    gateNBg = Param(
        Params._dummy(), "gateNBg",
        "number of shared background topics. Default 2.",
        typeConverter=TypeConverters.toInt,
    )
    gateTpn = Param(
        Params._dummy(), "gateTpn",
        "topics per DAG node (0 = a flat layout: K == gateNBg, plain LDA). Default 1.",
        typeConverter=TypeConverters.toInt,
    )
    gateWarm = Param(
        Params._dummy(), "gateWarm",
        "leaves-first gate schedule: iterations per node-height level before the "
        "next level opens; 0 (default) = the single-round gate. See "
        "DagGatedOnlineLDA.set_gate_schedule.",
        typeConverter=TypeConverters.toInt,
    )
    gateOpenSeed = Param(
        Params._dummy(), "gateOpenSeed",
        "with gateWarm > 0: seed each block the moment it opens from the term counts "
        "of the documents its gate admits (one pass before the fit). Default True; "
        "False leaves late blocks at their init (they die on arrival in simulation).",
        typeConverter=TypeConverters.toBoolean,
    )
    gateSeedMode = Param(
        Params._dummy(), "gateSeedMode",
        "with gateWarm > 0: the open-time seed — 'balanced' (default; each child's "
        "subtree share equally plus the documents labelled at the node itself, with "
        "the descendants' nodeTerms removed) or 'subtree' (the term counts of the "
        "documents a block admits).",
        typeConverter=TypeConverters.toString,
    )
    gateSeedLeaves = Param(
        Params._dummy(), "gateSeedLeaves",
        "with gateWarm > 0 and gateOpenSeed: also seed every LEAF block at iteration "
        "0 from the term counts of its own documents. Default False.",
        typeConverter=TypeConverters.toBoolean,
    )
    featuresCols = Param(
        Params._dummy(), "featuresCols",
        "ordered list of per-domain vector columns. When set, each row's per-domain "
        "vectors are concatenated into the engine's single token-id space and the "
        "engine carries a per-domain lambda {m: (K, V_m)}; a domain's width is its "
        "vectors' size (the column's ML attribute size when present, else the first "
        "row's), and every row is checked against it. Empty (default) = the single "
        "featuresCol.",
        typeConverter=TypeConverters.toListString,
    )
    vocabularies = Param(
        Params._dummy(), "vocabularies",
        "optional, set with setVocabularies: one list of term ids (str or int) per "
        "feature column (featuresCols, else [featuresCol]), aligned with the vector "
        "indices (length = that column's vector size). Needed only by nodeTerms / "
        "gateSeedProfile, which name terms. Terms are matched exactly as given, like "
        "node ids: vectorizer vocabularies hold strings, so int term ids are passed "
        "as their strings there. Stored as JSON.",
        typeConverter=TypeConverters.toString,
    )
    nodeTerms = Param(
        Params._dummy(), "nodeTerms",
        "optional, set with setNodeTerms: {node: [term]}, the terms that attest each "
        "node (user node ids, term ids of the vocabularies). With "
        "gateSeedMode='balanced', a node's terms are removed from its ancestors' "
        "open-time seeds. A term in two vocabularies must be qualified as "
        "(featureColumn, term). Entries of nodes not kept by the fit, and terms in no "
        "vocabulary, are dropped (counted in the log). Stored as JSON.",
        typeConverter=TypeConverters.toString,
    )
    gateSeedProfile = Param(
        Params._dummy(), "gateSeedProfile",
        "optional, set with setGateSeedProfile: {node: {term: weight}}, an external "
        "per-node term profile. With gateWarm > 0 it REPLACES a profiled node's seed "
        "in the domains its terms belong to (leaves at iteration 0, later blocks when "
        "they open); other domains and unprofiled nodes keep the count seed. Same id, "
        "qualification and dropping rules as nodeTerms. Stored as JSON.",
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


_DAG_GATED_LDA_DEFAULTS = dict(
    maxIter=20,
    featuresCol="features", topicDistributionCol="topicDistribution",
    learningOffset=1024.0, learningDecay=0.51, subsamplingRate=0.05,
    optimizeDocConcentration=False,
    gammaShape=100.0, caviMaxIter=100, caviTol=1e-3,
    dagEdges="", labelsCol="labels", minDocsPerNode=1, unknownLabels="error",
    gateNBg=2, gateTpn=1,
    gateWarm=0, gateOpenSeed=True, gateSeedMode="balanced", gateSeedLeaves=False,
    featuresCols=[], vocabularies="", nodeTerms="", gateSeedProfile="",
    # docConcentration / topicConcentration: no default (isSet; unset = 1/K).
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


class DagGatedLDAEstimator(_DagGatedLDAParams, Estimator):
    """MLlib-shaped Estimator wrapping ``DagGatedOnlineLDA``.

    Required: a DAG (``setDag``) and, on the fit DataFrame, ``labelsCol`` plus the
    feature column(s). The node set (which nodes get a topic block) is learned from
    that DataFrame; there is no train/test notion here (split before ``fit``)."""

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
        gateNBg: int = 2,
        gateTpn: int = 1,
        gateWarm: int = 0,
        gateOpenSeed: bool = True,
        gateSeedMode: str = "balanced",
        gateSeedLeaves: bool = False,
        featuresCols: list[str] | None = None,
        vocabularies=None,
        nodeTerms=None,
        gateSeedProfile=None,
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

    # ---- fit --------------------------------------------------------------------

    def _fit(self, dataset) -> "DagGatedLDAModel":
        from spark_vi.core.runner import VIRunner

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


class DagGatedLDAModel(_DagGatedLDAParams, _PersistableModel, Model):
    """MLlib-shaped Model wrapping a trained DagGatedOnlineLDA VIResult.

    ``transform`` appends ``topicDistributionCol`` (theta) by folding each
    document in UNGATED over all K topics with the same CAVI the engine uses at
    train time. The node-keyed accessors (``dag``, ``prunedNodes``, ``splicedNodes``,
    ``nodeSupport``, ``nodeTopics``, ``backgroundTopics``, ``describeNodes``) answer
    in the user's node ids; engine ids stay internal. Persistable via
    ``_PersistableModel`` (the node set rides the VIResult metadata); ``load``
    rejects a checkpoint stamped with any other model class."""

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
                   "was spliced" if node in ns.spliced else "is not a DAG node")
            raise KeyError(f"node {node!r} has no topic block: it {why}")
        lo = self._n_bg + (e - 1) * self._tpn
        return list(range(lo, lo + self._tpn))

    def backgroundTopics(self) -> list:
        """The shared background topics' indices (``0 .. gateNBg-1``)."""
        self._ns()
        return list(range(self._n_bg))

    def describeNodes(self, maxTermsPerTopic: int = 10):
        """DataFrame ``(node, topic, termIndices, termWeights)``: one row per topic
        (background topics have a null node), top terms as in ``describeTopics``.

        ``node`` is a LONG column when every kept node id is an int, a STRING column
        when every one is a str, and a STRING column (ints written in decimal) when
        the ids are mixed; there ``1`` and ``"1"`` render alike, so use ``nodeTopics``
        to tell them apart."""
        from pyspark.sql import SparkSession
        from pyspark.sql.types import (ArrayType, DoubleType, IntegerType, LongType,
                                       StringType, StructField, StructType)
        ns = self._ns()
        ids = ns.nodes
        all_int = bool(ids) and all(isinstance(n, int) for n in ids)
        all_str = all(isinstance(n, str) for n in ids)
        node_type = LongType() if all_int else StringType()
        owner = {}
        for n in ids:
            for k in self.nodeTopics(n):
                owner[k] = n if (all_int or all_str) else str(n)
        rows = [(owner.get(k),) + r for k, r in enumerate(self._top_terms(maxTermsPerTopic))]
        schema = StructType([
            StructField("node", node_type, True),
            StructField("topic", IntegerType(), False),
            StructField("termIndices", ArrayType(IntegerType(), False), False),
            StructField("termWeights", ArrayType(DoubleType(), False), False),
        ])
        return SparkSession.builder.getOrCreate().createDataFrame(rows, schema=schema)

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

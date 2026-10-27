"""MLlib Estimator/Model shim for the DAG-gated LDA engine (ADR 0050).

Mirrors mllib/topic/lda.py (ADR 0009): a translation layer over
``spark_vi.models.topic.dag_gated_lda.DagGatedOnlineLDA`` + ``VIRunner``. All SVI
logic lives in the engine; this shim marshals DataFrame columns into
``GatedBOWDocument`` rows, builds the gate seeds from the cached document RDD, and
wraps the trained ``VIResult`` as an MLlib-shaped Model.

Training is GATED: each row carries features plus a ``frontierCol`` (the doc's
most-specific attested DAG nodes; empty = background), and the E-step restricts the
doc to ``DagLayout.allowed_set(frontier)``. K is derived from the layout
(``gateNBg + n_nodes * gateTpn``). ``transform`` folds documents in UNGATED over all
K topics (the label is unknown at scoring time) and appends ``topicDistributionCol``.

Initialization is the random Gamma draw plus, under the leaves-first schedule
(``gateWarm > 0``), the data-driven gate seeds (``gateOpenSeed``, ``gateSeedMode``,
``gateSeedLeaves``, ``gateSeedProfile``). Multi-domain fits (``featuresCols``) carry
a per-domain lambda ``{m: (K, V_m)}`` over one concatenated token-id space.
"""
from __future__ import annotations

import json
import operator
from typing import Callable

import numpy as np
from pyspark import StorageLevel, keyword_only
from pyspark.ml.base import Estimator, Model
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasFeaturesCol, HasMaxIter, HasSeed

from spark_vi.core.config import VIConfig
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
from spark_vi.dag import DagLayout
from spark_vi.models.topic.types import GatedBOWDocument


def _row_to_gated_document(row, features_col, frontier_col, features_cols=None,
                           domain_sizes=None) -> GatedBOWDocument:
    """Build a GatedBOWDocument from a Spark row.

    ``features_col`` is a Sparse/Dense feature vector; with ``features_cols`` (and
    ``domain_sizes``) the N per-domain vectors are concatenated into the engine's
    single token-id space instead (each row validated against the fixed widths).
    ``frontier_col`` is an array of node ids; empty/None = background doc."""
    if features_cols:
        indices, counts = _concat_domain_features(
            [row[c] for c in features_cols], domain_sizes)
        length = int(counts.sum())
    else:
        bow = _vector_to_bow_document(row[features_col])
        indices, counts, length = bow.indices, bow.counts, bow.length
    frontier = frozenset(int(x) for x in (row[frontier_col] or []))
    return GatedBOWDocument(indices=indices, counts=counts, length=length,
                            frontier=frontier)


class _DagGatedLDAParams(HasFeaturesCol, HasMaxIter, HasSeed, _PersistenceParams):
    """Shared Param surface for DagGatedLDAEstimator and DagGatedLDAModel.

    The SVI schedule, Dirichlet and CAVI knobs mirror ``_OnlineLDAParams``; on top
    sit the gate (``gateParent``, ``gateNBg``, ``gateTpn``, ``frontierCol``), the
    leaves-first schedule and its seeds, and the multi-domain columns. Persistence
    Params (saveInterval, saveDir, resumeFrom) come from ``_PersistenceParams``, so
    checkpoint + resume UX is identical across the topic-model shims."""

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
        "vector. Unset = 1/K, which is razor-small at whole-Mondo K; the central "
        "configuration sets 0.5.",
        typeConverter=TypeConverters.toListFloat,
    )
    optimizeDocConcentration = Param(
        Params._dummy(), "optimizeDocConcentration",
        "learn the per-node tied empirical-Bayes alpha (the gated Newton step, "
        "insight 0059) from the static frontier histogram. Default False "
        "(insight 0091: keep it off).",
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
    gateParent = Param(
        Params._dummy(), "gateParent",
        "JSON-encoded DAG parent map {child_node: parent_node or [parent_nodes]} "
        "(root 0 omitted). Each node's topic block is welded to its DAG subtree's "
        "documents by the gated E-step. Required; K is derived from the layout "
        "(gateNBg + n_nodes * gateTpn). A depth-1 DAG is a flat labelled-LDA gate.",
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
    frontierCol = Param(
        Params._dummy(), "frontierCol",
        "column carrying each doc's DAG frontier (array of most-specific attested "
        "node ids; empty = background). Gates that doc's topic training to "
        "DagLayout.allowed_set(frontier).",
        typeConverter=TypeConverters.toString,
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
        "with gateWarm > 0: seed each block the moment it opens from the word counts "
        "of the documents its gate admits (one pass before the fit). Default True; "
        "False leaves late blocks at their init (they die on arrival in simulation).",
        typeConverter=TypeConverters.toBoolean,
    )
    gateSeedMode = Param(
        Params._dummy(), "gateSeedMode",
        "with gateWarm > 0: the open-time seed — 'balanced' (default; each child's "
        "subtree share equally plus the parent's own coded documents, descendants' "
        "own codes removed via gateSeedOwnCodes; exp 0160) or 'subtree' (the word "
        "counts of the documents a block admits).",
        typeConverter=TypeConverters.toString,
    )
    gateSeedLeaves = Param(
        Params._dummy(), "gateSeedLeaves",
        "with gateWarm > 0 and gateOpenSeed: also seed every LEAF block at iteration "
        "0 from the word counts of its own documents (exp 0162). Default False.",
        typeConverter=TypeConverters.toBoolean,
    )
    gateSeedDomain = Param(
        Params._dummy(), "gateSeedDomain",
        "the PRIMARY domain: the index (into featuresCols; 0 for a single "
        "featuresCol) of the domain whose vocabulary gateSeedOwnCodes and "
        "gateSeedProfile index. Default 0.",
        typeConverter=TypeConverters.toInt,
    )
    gateSeedOwnCodes = Param(
        Params._dummy(), "gateSeedOwnCodes",
        "with gateSeedMode='balanced': JSON {engine node id: [primary-domain vocab "
        "idx]} of each node's own attesting codes; removed from its ancestors' seeds.",
        typeConverter=TypeConverters.toString,
    )
    gateSeedProfile = Param(
        Params._dummy(), "gateSeedProfile",
        "with gateWarm > 0: JSON {engine node id: {primary-domain vocab idx: weight}} "
        "— an external phenotype profile (e.g. HPO) that REPLACES a profiled node's "
        "primary-domain seed: leaves at iteration 0, later blocks when they open. "
        "Unprofiled nodes keep the histogram seed. Empty (default) = histogram only.",
        typeConverter=TypeConverters.toString,
    )
    featuresCols = Param(
        Params._dummy(), "featuresCols",
        "ordered list of per-domain feature columns (features_0 .. features_{N-1}). "
        "When set, each row's per-domain sparse vectors are concatenated into the "
        "engine's single token-id space and the engine carries a per-domain lambda "
        "{m: (K, V_m)}. Empty (default) = the single featuresCol.",
        typeConverter=TypeConverters.toListString,
    )
    domainBounds = Param(
        Params._dummy(), "domainBounds",
        "optional authoritative cumulative per-domain vocab offsets [0, V_0, "
        "V_0+V_1, ...] (len == len(featuresCols)+1). When unset, the per-domain "
        "widths are read from the first row's vector sizes; set it as the escape "
        "hatch for an unrepresentative first row (every row is then validated "
        "against these widths).",
        typeConverter=TypeConverters.toListInt,
    )

    def setGateParent(self, parent):
        """Set the gate from a DAG parent map {child: parent | [parents]}
        (JSON-encoded into the string Param), or a pre-encoded JSON string."""
        if parent is None or parent == "":
            return self._set(gateParent="")
        encoded = parent if isinstance(parent, str) else json.dumps(
            {int(c): ([int(x) for x in p] if isinstance(p, (list, tuple, set))
                      else int(p)) for c, p in parent.items()})
        return self._set(gateParent=encoded)

    def getGateParent(self) -> str:
        return str(self.getOrDefault(self.gateParent))

    def setGateSeedProfile(self, prof):
        """{engine node id: {vocab idx: weight}} (or a pre-encoded JSON string);
        empty/None clears it. operator.index on every id: a float id would
        otherwise truncate silently and seed the wrong token."""
        if prof is None or prof == "" or prof == {}:
            return self._set(gateSeedProfile="")
        encoded = prof if isinstance(prof, str) else json.dumps(
            {int(operator.index(k)): {int(operator.index(i)): float(w)
                                      for i, w in v.items() if float(w) > 0}
             for k, v in prof.items() if v})
        return self._set(gateSeedProfile=encoded)

    def getGateSeedProfile(self) -> dict:
        raw = str(self.getOrDefault(self.gateSeedProfile))
        if not raw:
            return {}
        return {int(k): {int(i): float(w) for i, w in v.items()}
                for k, v in json.loads(raw).items()}

    def setGateSeedOwnCodes(self, own):
        """{engine node id: [vocab idx]} (or pre-encoded JSON); empty clears it."""
        if own is None or own == "" or own == {}:
            return self._set(gateSeedOwnCodes="")
        encoded = own if isinstance(own, str) else json.dumps(
            {int(operator.index(k)): sorted({operator.index(i) for i in v})
             for k, v in own.items() if v})
        return self._set(gateSeedOwnCodes=encoded)

    def getGateSeedOwnCodes(self) -> dict:
        raw = str(self.getOrDefault(self.gateSeedOwnCodes))
        if not raw:
            return {}
        return {int(k): [int(i) for i in v] for k, v in json.loads(raw).items()}

    def _layout(self) -> DagLayout:
        """The DagLayout this estimator/model's gate Params describe."""
        raw = str(self.getOrDefault("gateParent"))
        if not raw:
            raise ValueError(
                "gateParent is required: the DAG-gated LDA needs a label DAG "
                "(use a depth-1 map {node: 0} for a flat labelled-LDA gate).")
        try:
            parent_map = {int(c): p for c, p in json.loads(raw).items()}
        except (json.JSONDecodeError, AttributeError, ValueError) as exc:
            raise ValueError(f"gateParent is not a valid JSON DAG map: {exc}") from exc
        return DagLayout(parent_map, n_bg=int(self.getOrDefault("gateNBg")),
                         tpn=int(self.getOrDefault("gateTpn")))


_DAG_GATED_LDA_DEFAULTS = dict(
    maxIter=20,
    featuresCol="features", topicDistributionCol="topicDistribution",
    learningOffset=1024.0, learningDecay=0.51, subsamplingRate=0.05,
    optimizeDocConcentration=False,
    gammaShape=100.0, caviMaxIter=100, caviTol=1e-3,
    gateParent="", gateNBg=2, gateTpn=1, frontierCol="frontier",
    gateWarm=0, gateOpenSeed=True, gateSeedMode="balanced", gateSeedLeaves=False,
    gateSeedDomain=0, gateSeedOwnCodes="", gateSeedProfile="",
    featuresCols=[],   # domainBounds intentionally omitted: it uses isSet (no default)
)


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


class DagGatedLDAEstimator(_DagGatedLDAParams, Estimator):
    """MLlib-shaped Estimator wrapping ``DagGatedOnlineLDA``.

    ``gateParent`` is required (set it with ``setGateParent``); every training row
    must carry ``frontierCol``."""

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
        optimizeDocConcentration: bool = False,
        gammaShape: float = 100.0,
        caviMaxIter: int = 100,
        caviTol: float = 1e-3,
        gateParent: str = "",
        gateNBg: int = 2,
        gateTpn: int = 1,
        frontierCol: str = "frontier",
        gateWarm: int = 0,
        gateOpenSeed: bool = True,
        gateSeedMode: str = "balanced",
        gateSeedLeaves: bool = False,
        gateSeedDomain: int = 0,
        gateSeedOwnCodes: str = "",
        gateSeedProfile: str = "",
        featuresCols: list[str] | None = None,
        domainBounds: list[int] | None = None,
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
        # featuresCols/domainBounds carry no positive default (featuresCols defaults
        # to [] via _setDefault; domainBounds uses isSet). Drop an explicit None so
        # kwarg-style construction that leaves them unset does not clobber the
        # default / trip the list typeConverter.
        kwargs = {k: v for k, v in self._input_kwargs.items()
                  if not (k in ("featuresCols", "domainBounds", "docConcentration")
                          and v is None)}
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, **kwargs) -> "DagGatedLDAEstimator":
        return self._set(**kwargs)

    def setOnIteration(
        self, fn: Callable[[int, dict, list[float]], None] | None,
    ) -> "DagGatedLDAEstimator":
        """Register a per-iteration diagnostic callback for the next fit.

        Signature fn(iter_num, global_params, elbo_trace); runs on the driver in
        the fit hot path. Must not mutate global_params. Not persisted."""
        self._on_iteration = fn
        return self

    def _domain_sizes(self, dataset):
        """(features_col, domain_sizes, vocab_size) from the first row / domainBounds."""
        fcols = list(self.getOrDefault("featuresCols") or [])
        if fcols:
            first = dataset.select(*fcols).head(1)
            if not first:
                raise ValueError("Cannot fit on an empty DataFrame.")
            if self.isSet("domainBounds"):
                # Explicit bounds are AUTHORITATIVE: the escape hatch for a dataset
                # whose first row is unrepresentative; every row is validated
                # against these widths by _concat_domain_features.
                bounds = [int(b) for b in self.getOrDefault("domainBounds")]
                if len(bounds) != len(fcols) + 1 or bounds[0] != 0 or \
                        any(b <= a for a, b in zip(bounds, bounds[1:])):
                    raise ValueError(
                        f"domainBounds {bounds} must be strictly increasing, start at "
                        f"0, and have len(featuresCols)+1 = {len(fcols) + 1} entries")
                sizes = [b - a for a, b in zip(bounds, bounds[1:])]
            else:
                sizes = [int(first[0][i].size) for i in range(len(fcols))]
            return None, sizes, sum(sizes)
        features_col = self.getOrDefault("featuresCol")
        first = dataset.select(features_col).head(1)
        if not first:
            raise ValueError("Cannot fit on an empty DataFrame.")
        return features_col, None, int(first[0][0].size)

    def _primary_slice(self, vocab_size, domain_sizes):
        """(lo, hi): the primary domain's (gateSeedDomain) span in the
        concatenated vocabulary."""
        d = int(self.getOrDefault("gateSeedDomain"))
        sizes = list(domain_sizes) if domain_sizes else [int(vocab_size)]
        if not 0 <= d < len(sizes):
            raise ValueError(
                f"gateSeedDomain={d} is outside the {len(sizes)} domain(s) of this fit")
        lo = int(sum(sizes[:d]))
        return lo, lo + int(sizes[d])

    def _gate_seeds(self, rdd, lay, vocab_size, domain_sizes):
        """(open_seed, init_seed, summary) for the leaves-first schedule.

        The open-time seed is one pass over the cached document RDD
        (`gated_init.gate_open_seed_counts`); with gateSeedLeaves the leaf rows move
        to the iteration-0 seed. The own codes (gateSeedOwnCodes) and a profile
        (gateSeedProfile) index the PRIMARY domain's vocabulary (gateSeedDomain);
        they are shifted to the concatenated id space here. A profile replaces a
        profiled node's primary-domain slice; the other domains keep their
        histogram / init."""
        seed = None
        mode = str(self.getOrDefault("gateSeedMode"))
        leaves = bool(self.getOrDefault("gateSeedLeaves"))
        lo, hi = self._primary_slice(vocab_size, domain_sizes)
        if bool(self.getOrDefault("gateOpenSeed")):
            from spark_vi.models.topic.gated_init import gate_open_seed_counts
            own = None
            if mode == "balanced":
                own = {u: [lo + int(i) for i in idx if 0 <= int(i) < hi - lo]
                       for u, idx in self.getGateSeedOwnCodes().items()}
            seed = gate_open_seed_counts(rdd, lay, vocab_size, mode=mode,
                                         own_idx=own, leaves=leaves)
        init = None
        if seed is not None and leaves:
            # Leaf rows go in at iteration 0; the open-time seed keeps the rest.
            init = np.zeros_like(seed)
            for u in lay.nodes:
                if lay.height(u) == 0:
                    for k in lay.block[u]:
                        init[k] = seed[k]
                        seed[k] = 0.0
        prof = self.getGateSeedProfile()
        n_leaf = n_late = 0
        if prof:
            Vp = hi - lo
            if init is None:
                init = np.zeros((lay.K, vocab_size), dtype=np.float64)
            for u, w in prof.items():
                if u not in lay.block:
                    continue
                row = np.zeros(Vp)
                for i, x in w.items():
                    if 0 <= i < Vp:
                        row[i] = x
                if not row.any():
                    continue
                for k in lay.block[u]:
                    if lay.height(u) == 0:
                        init[k, lo:hi] = row
                        n_leaf += 1
                    else:
                        if seed is None:
                            seed = np.zeros((lay.K, vocab_size), dtype=np.float64)
                        seed[k, lo:hi] = row
                        n_late += 1
            if not n_leaf and not leaves:
                init = None
        summary = (f"open-time seed "
                   f"{'ON (' + mode + ')' if seed is not None else 'off'}"
                   + ("; leaf blocks seeded at iteration 0 from their own documents"
                      if leaves else "")
                   + (f"; profile seed on {n_leaf} leaf + {n_late} later topic(s), "
                      "histogram elsewhere" if prof else ""))
        return seed, init, summary

    def _fit(self, dataset) -> "DagGatedLDAModel":
        from spark_vi.core.runner import VIRunner

        lay = self._layout()
        features_col, domain_sizes, vocab_size = self._domain_sizes(dataset)
        frontier_col = self.getOrDefault("frontierCol")
        if frontier_col not in dataset.columns:
            raise ValueError(
                f"frontierCol={frontier_col!r} is not a column of the input; the "
                f"gated fit needs a per-doc frontier.")
        doc_conc = (self.getOrDefault("docConcentration")
                    if self.isSet("docConcentration") else None)
        seed = self.getOrDefault("seed") if self.isSet("seed") else None
        engine = DagGatedOnlineLDA(
            lay, vocab_size,
            alpha=_doc_alpha(doc_conc, lay.K), eta=1.0 / lay.K,
            domains=domain_sizes,                    # None = single fused vocab
            gamma_shape=self.getOrDefault("gammaShape"),
            cavi_max_iter=self.getOrDefault("caviMaxIter"),
            cavi_tol=self.getOrDefault("caviTol"), random_seed=seed)
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

        fcols = list(self.getOrDefault("featuresCols") or [])
        select_cols = (list(fcols) if fcols else [features_col]) + [frontier_col]

        def _to_doc(row, _fc=features_col, _frc=frontier_col,
                    _fcs=(fcols or None), _ds=domain_sizes):
            return _row_to_gated_document(row, _fc, _frc, features_cols=_fcs,
                                          domain_sizes=_ds)

        rdd = (dataset.select(*select_cols).rdd.map(_to_doc)
               .persist(StorageLevel.MEMORY_AND_DISK))
        rdd.count()  # materialize for VIRunner's strict cache precondition

        try:
            # Leaves-first gate schedule (exp 0153/0154; report 2026-10-19).
            # Fit-only: nothing about the corpus or the bundle changes.
            gw = int(self.getOrDefault("gateWarm"))
            if gw > 0:
                open_seed, init_seed, summary = self._gate_seeds(
                    rdd, lay, vocab_size, domain_sizes)
                engine.set_gate_schedule(gw, open_seed=open_seed, init_seed=init_seed)
                print(f"[dag-gated-lda] leaves-first gate: {gw} iters/level over "
                      f"heights 0..{engine._gate_H} (full gate from iter "
                      f"{gw * engine._gate_H}); {summary}", flush=True)
            # Per-node tied empirical-Bayes alpha (insight 0059): needs the static
            # frontier histogram, which only exists once the document RDD does.
            if bool(self.getOrDefault("optimizeDocConcentration")):
                hist = {fr: int(n) for fr, n in
                        rdd.map(lambda d: d.frontier).countByValue().items()}
                engine.set_alpha_policy(hist, optimize=True)
                print(f"[dag-gated-lda] empirical-Bayes alpha on ({len(hist)} "
                      f"frontier groups)", flush=True)
            result = VIRunner(engine, config=config).fit(
                rdd, resume_from=resume_path, on_iteration=self._on_iteration)
        finally:
            rdd.unpersist(blocking=False)

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
    train time. Persistable via ``_PersistableModel``; ``load`` rejects a
    checkpoint stamped with any other model class."""

    _expected_model_class = "DagGatedOnlineLDA"

    def __init__(self, result) -> None:  # result: VIResult
        super().__init__()
        self._result = result
        self._setDefault(**_DAG_GATED_LDA_DEFAULTS)
        self._set_persistence_defaults()

    @property
    def result(self):
        """The trained VIResult (global_params, elbo_trace, n_iterations, ...)."""
        return self._result

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

    def describeTopics(self, maxTermsPerTopic: int = 10):
        """DataFrame of (topic, termIndices, termWeights) — top terms per topic
        (term indices in the concatenated vocabulary when multi-domain)."""
        from pyspark.sql import SparkSession
        from pyspark.sql.types import (
            ArrayType, DoubleType, IntegerType, StructField, StructType,
        )
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
        schema = StructType([
            StructField("topic", IntegerType(), False),
            StructField("termIndices", ArrayType(IntegerType(), False), False),
            StructField("termWeights", ArrayType(DoubleType(), False), False),
        ])
        return SparkSession.builder.getOrCreate().createDataFrame(rows, schema=schema)

    def _transform(self, dataset):
        import hashlib

        from pyspark.ml.linalg import DenseVector, VectorUDT
        from pyspark.sql import functions as F

        from spark_vi.models.topic.lda import _cavi_doc_inference

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
            h = hashlib.blake2b(digest_size=8)
            h.update(np.ascontiguousarray(indices, dtype=np.int32).tobytes())
            h.update(np.ascontiguousarray(counts, dtype=np.float64).tobytes())
            rng = np.random.default_rng(int.from_bytes(h.digest(), "little"))
            gamma_init = rng.gamma(p["gamma_shape"], 1.0 / p["gamma_shape"], size=p["K"])
            gamma, _, _, _ = _cavi_doc_inference(
                indices=indices, counts=counts,
                expElogbeta=p["expElogbeta"], alpha=p["alpha"],
                gamma_init=gamma_init,
                max_iter=p["cavi_max_iter"], tol=p["cavi_tol"],
            )
            return DenseVector(gamma / gamma.sum())

        infer_udf = F.udf(_infer, returnType=VectorUDT())
        out_col = self.getOrDefault("topicDistributionCol")
        fcols = list(self.getOrDefault("featuresCols") or [])
        feat_args = ([F.col(c) for c in fcols] if fcols
                     else [F.col(self.getOrDefault("featuresCol"))])
        # Broadcast lifetime is the returned DataFrame's (its UDF closure holds
        # bcast); ContextCleaner reclaims it on GC — do NOT eagerly unpersist.
        return dataset.withColumn(out_col, infer_udf(*feat_args))

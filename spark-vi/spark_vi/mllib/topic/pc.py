"""MLlib Estimator/Transformer shim for OnlinePCLDA (Prediction-Constrained LDA).

Mirrors mllib/topic/lda.py (ADR 0009): a translation layer over
``spark_vi.models.topic.pc.OnlinePCLDA`` + ``VIRunner``. All SVI logic lives in
OnlinePCLDA; this shim only marshals DataFrame columns into ``PCDocument``s and
wraps the trained ``VIResult`` as an MLlib-shaped Model.

At ``weightY == 0`` (increment 1, the unsupervised SVI path) the label columns
(``labelCol``, ``labelMaskCol``) are THREADED into every ``PCDocument`` exactly
as the STM shim threads its covariate column, but tolerated-absent — the model
never reads y/label_mask, so a DataFrame with no label columns fits fine
(placeholder zeros are carried), and ``_transform`` appends only the label-free
``topicDistributionCol``.

At ``weightY > 0`` (increment 2, supervised) ``labelCol`` is REQUIRED — the
head trains on it — and ``_transform`` additionally appends a head-derived
``probabilityCol`` = ``sigmoid(w_CK · θ)`` (per-label P(y=1)); the label-free
``topicDistributionCol`` is unchanged (identical CAVI to train time).
"""
from __future__ import annotations

import json
from pathlib import Path
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
from spark_vi.mllib.topic._common import _vector_to_bow_document
from spark_vi.models.topic.pc import DagClosureHead, OnlinePCLDA
from spark_vi.models.topic.types import PCDocument


def _row_to_pc_document(
    row,
    features_col: str | None,
    label_col: str | None,
    label_mask_col: str | None,
    C: int,
    features_cols: list[str] | None = None,
    domain_sizes: list[int] | None = None,
) -> PCDocument:
    """Build a PCDocument from a Spark row, tolerating absent label columns.

    ``features_col`` is a Sparse/Dense feature vector (sparsified to nonzero
    entries by ``_vector_to_bow_document``). ``label_col`` — when set — is the
    doc's (C,) outcome vector (a scalar for the single-label C == 1 case is
    promoted to a length-1 array); ``label_mask_col`` — when set — its (C,)
    observed mask. When a column is absent the corresponding field is a
    placeholder: ``y`` = zeros(C), and ``label_mask`` = zeros(C) (nothing
    observed) unless an explicit ``label_col`` is present, in which case a
    missing mask defaults to ones(C) (all observed). At weightY == 0 none of
    this is read — the placeholders exist only to keep the row type stable for
    increment 2.

    MULTI-DOMAIN: when ``features_cols`` (and ``domain_sizes``) are given, the N
    per-domain sparse vectors are concatenated into the engine's single token-id
    space via ``_concat_domain_features`` (each row validated against the fixed
    per-domain widths), superseding ``features_col``. The label/mask handling is
    unchanged (the head is domain-agnostic).
    """
    if features_cols:
        from spark_vi.mllib.topic.gated_lda import _concat_domain_features
        indices, counts = _concat_domain_features(
            [row[c] for c in features_cols], domain_sizes)
        length = int(counts.sum())
    else:
        bow = _vector_to_bow_document(row[features_col])
        indices, counts, length = bow.indices, bow.counts, bow.length

    if label_col is None:
        y = np.zeros(C, dtype=np.float64)
    else:
        raw = row[label_col]
        y = np.asarray(
            raw if isinstance(raw, (list, tuple, np.ndarray)) else [raw],
            dtype=np.float64,
        )

    if label_mask_col is not None:
        raw_m = row[label_mask_col]
        mask = np.asarray(
            raw_m if isinstance(raw_m, (list, tuple, np.ndarray)) else [raw_m],
            dtype=np.float64,
        )
    elif label_col is not None:
        mask = np.ones(C, dtype=np.float64)   # labelled doc, all cells observed
    else:
        mask = np.zeros(C, dtype=np.float64)  # no labels at all

    return PCDocument(
        indices=indices,
        counts=counts,
        length=length,
        y=y,
        label_mask=mask,
    )


def _row_to_gated_pc_document(
    row,
    features_col: str | None,
    label_col: str | None,
    label_mask_col: str | None,
    frontier_col: str,
    C: int,
    features_cols: list[str] | None = None,
    domain_sizes: list[int] | None = None,
):
    """Build a GatedPCDocument (Gated-PC): a PCDocument plus a DAG frontier.

    Reuses :func:`_row_to_pc_document` for the features/label/mask fields (including
    the multi-domain concatenation when ``features_cols`` is set), then reads
    ``frontier_col`` — an array of node ids (empty/None = ungated background) — and
    attaches it so the gated E-step can restrict this doc's training to its subtree.
    """
    from spark_vi.models.topic.types import GatedPCDocument
    pc = _row_to_pc_document(row, features_col, label_col, label_mask_col, C,
                             features_cols=features_cols, domain_sizes=domain_sizes)
    raw_f = row[frontier_col]
    frontier = frozenset(int(x) for x in (raw_f or []))
    return GatedPCDocument(
        indices=pc.indices, counts=pc.counts, length=pc.length,
        y=pc.y, label_mask=pc.label_mask, frontier=frontier,
    )


class _OnlinePCLDAParams(HasFeaturesCol, HasMaxIter, HasSeed, _PersistenceParams):
    """Shared Param surface for OnlinePCLDAEstimator and OnlinePCLDAModel.

    The LDA param subset (k, topicDistributionCol, the SVI schedule knobs, the
    Dirichlet concentrations, the CAVI knobs) mirrors ``_OnlineLDAParams``; on
    top sit the Prediction-Constrained params: ``numLabels`` (C), ``labelCol``,
    ``labelMaskCol``, and ``weightY`` (default 0.0 for increment 1). Persistence
    Params (saveInterval, saveDir, resumeFrom) come from ``_PersistenceParams``
    (the same mixin the LDA/HDP shims use), so checkpoint + resume UX is
    identical across the topic-model shims.
    """

    k = Param(
        Params._dummy(), "k",
        "number of topics (clusters) to infer; must be >= 1",
        typeConverter=TypeConverters.toInt,
    )
    topicDistributionCol = Param(
        Params._dummy(), "topicDistributionCol",
        "output column with the label-free topic mixture (theta) for each document",
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
        "fraction of corpus sampled per mini-batch (None/1.0 = full batch)",
        typeConverter=TypeConverters.toFloat,
    )
    docConcentration = Param(
        Params._dummy(), "docConcentration",
        "Dirichlet concentration alpha on theta; scalar (symmetric) or length-k vector",
        typeConverter=TypeConverters.toListFloat,
    )
    topicConcentration = Param(
        Params._dummy(), "topicConcentration",
        "Dirichlet concentration eta on beta; scalar (symmetric)",
        typeConverter=TypeConverters.toFloat,
    )
    optimizeDocConcentration = Param(
        Params._dummy(), "optimizeDocConcentration",
        "whether to optimize alpha via Newton-Raphson (Blei 2003 App. A.4.2)",
        typeConverter=TypeConverters.toBoolean,
    )
    optimizeTopicConcentration = Param(
        Params._dummy(), "optimizeTopicConcentration",
        "whether to optimize eta (symmetric scalar) via Newton-Raphson",
        typeConverter=TypeConverters.toBoolean,
    )
    gammaShape = Param(
        Params._dummy(), "gammaShape",
        "shape parameter for Gamma init of variational gamma/lambda (ADR 0008 default 100.0)",
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
    # -- Prediction-Constrained params -------------------------------------
    numLabels = Param(
        Params._dummy(), "numLabels",
        "C = number of binary outcome heads (rows of the logistic head w_CK); >= 1",
        typeConverter=TypeConverters.toInt,
    )
    labelCol = Param(
        Params._dummy(), "labelCol",
        "column carrying the (C,) binary outcome vector; unused at weightY == 0",
        typeConverter=TypeConverters.toString,
    )
    labelMaskCol = Param(
        Params._dummy(), "labelMaskCol",
        "column carrying the (C,) per-cell observed mask; unused at weightY == 0",
        typeConverter=TypeConverters.toString,
    )
    weightY = Param(
        Params._dummy(), "weightY",
        "prediction-loss weight (the PC dial). 0.0 (default) = unsupervised "
        "LDA-MAP; > 0 turns on the supervised head + topic correction",
        typeConverter=TypeConverters.toFloat,
    )
    probabilityCol = Param(
        Params._dummy(), "probabilityCol",
        "output column with the head-derived per-label P(y=1) = sigmoid(w_CK . theta); "
        "appended by transform only when weightY > 0",
        typeConverter=TypeConverters.toString,
    )
    lambdaW = Param(
        Params._dummy(), "lambdaW",
        "L2 ridge on the head weights w_CK (scaled by weightY); authors' default 0.001",
        typeConverter=TypeConverters.toFloat,
    )
    gradCaviIters = Param(
        Params._dummy(), "gradCaviIters",
        "fixed CAVI unroll depth for the differentiated label-free pi (bounds the "
        "autograd tape); default 20",
        typeConverter=TypeConverters.toInt,
    )
    headLrScale = Param(
        Params._dummy(), "headLrScale",
        "extra multiplier on the head SGD step (RM <-> weightY decoupling knob); default 1.0",
        typeConverter=TypeConverters.toFloat,
    )
    topicTrust = Param(
        Params._dummy(), "topicTrust",
        "trust-region fraction capping the supervised topic correction on lambda to "
        "topicTrust * ||unsup lambda step|| (scale-invariant divergence guard); default 0.1",
        typeConverter=TypeConverters.toFloat,
    )
    weightYWarmupIters = Param(
        Params._dummy(), "weightYWarmupIters",
        "linearly ramp the effective weightY from 0 over this many global steps "
        "(0 = no warmup)",
        typeConverter=TypeConverters.toInt,
    )
    headOptimizer = Param(
        Params._dummy(), "headOptimizer",
        "head optimizer: 'sgd' (default; the RM-damped step rho*headLrScale*weightY*g, "
        "one first-order step per SVI iteration) or 'newton' (a per-iteration ridge-Newton "
        "/ IRLS step that CONVERGES the logistic head on the current theta — the settled "
        "head fix; ADR 0039). sgd does not converge the coupled head against a moving theta "
        "(insight 0065); newton is scale-invariant and aggregatable",
        typeConverter=TypeConverters.toString,
    )
    headLr = Param(
        Params._dummy(), "headLr",
        "the step-damping fraction for headOptimizer='newton' (~0.5-1.0); ignored for "
        "'sgd'. Default 0.05",
        typeConverter=TypeConverters.toFloat,
    )
    headNewtonRidge = Param(
        Params._dummy(), "headNewtonRidge",
        "relative ridge (fraction of mean(diag(H))) conditioning the per-label IRLS "
        "solve for headOptimizer='newton'; only stabilizes the solve, does not bias the "
        "head direction (AUC is scale-invariant to head magnitude). Default 0.01",
        typeConverter=TypeConverters.toFloat,
    )
    headL2 = Param(
        Params._dummy(), "headL2",
        "ABSOLUTE L2 ridge on the head weights w_CK for headOptimizer='newton' "
        "(= Hughes lambda_w, the ridge on the corpus-summed head gradient, so "
        "|w| ~ |g|/headL2). Default 1e-3; the good basin is wide (~1e-4..1e-2). "
        "0.0 disables it and BLOWS UP on the separable topics PC creates (ADR 0041). "
        "Distinct from lambdaW (the 'sgd' head's ridge).",
        typeConverter=TypeConverters.toFloat,
    )
    headIntercept = Param(
        Params._dummy(), "headIntercept",
        "newton head: fit a per-node UNPENALIZED intercept (base rate). Default False.",
        typeConverter=TypeConverters.toBoolean,
    )
    headStandardize = Param(
        Params._dummy(), "headStandardize",
        "newton head: z-score the θ features per topic before the logistic (the big "
        "conditioning lever — raw θ spans Σλ 1e2..1e6). Requires headIntercept. False.",
        typeConverter=TypeConverters.toBoolean,
    )
    # -- Topic-side DAG gate (Gated-PC composition; ADR 0042) ---------------
    gateParent = Param(
        Params._dummy(), "gateParent",
        "JSON-encoded topic-side DAG parent map {child_node: parent_node or "
        "[parent_nodes]} (root omitted) selecting the GATED topic engine "
        "(GatedOnlineLDA): each node's topic block is welded to its DAG-subtree's "
        "documents via the gated E-step. Empty (default) = ungated OnlineLDA. When "
        "set, K is DERIVED from the layout (gateNBg + n_nodes*gateTpn), overriding k, "
        "and every training row must carry frontierCol. Independent of closureParents "
        "(topic-side gate vs label-side head — they may use the same or different DAGs).",
        typeConverter=TypeConverters.toString,
    )
    gateNBg = Param(
        Params._dummy(), "gateNBg",
        "number of shared background topics for the topic-side gate (gateParent); "
        "only used when gateParent is set. Default 2.",
        typeConverter=TypeConverters.toInt,
    )
    gateTpn = Param(
        Params._dummy(), "gateTpn",
        "topics per DAG node for the topic-side gate (gateParent); only used when "
        "gateParent is set. Default 1.",
        typeConverter=TypeConverters.toInt,
    )
    localizeHead = Param(
        Params._dummy(), "localizeHead",
        "LOCALIZED head (requires gateParent): each label c's logistic reads ONLY its "
        "topic support DagLayout.allowed(c) (background + its gated block + ancestors), "
        "not all K, so the per-node Newton is O(|support|^3) not O(K^3) — the whole-Mondo "
        "scale fix (insight 0071; ADR 0042 done right, hierarchy in the head SUPPORT not a "
        "closure product). Only affects headOptimizer='newton'. Default False (dense).",
        typeConverter=TypeConverters.toBoolean,
    )
    frontierCol = Param(
        Params._dummy(), "frontierCol",
        "column carrying each doc's DAG frontier (array of most-specific attested "
        "node ids; empty = background). Required when gateParent is set; gates that "
        "doc's topic training to DagLayout.allowed_set(frontier).",
        typeConverter=TypeConverters.toString,
    )
    # -- Multi-domain features (MixEHR-style per-domain vocabularies) --------
    featuresCols = Param(
        Params._dummy(), "featuresCols",
        "ordered list of per-domain feature columns (features_0 .. features_{N-1}; "
        "domain 0 = conditions by convention). When set, each row's per-domain "
        "sparse vectors are concatenated into the engine's single token-id space and "
        "the injected GatedOnlineLDA carries a per-domain lambda {m:(K,V_m)}; the "
        "supervised topic correction is scattered back per domain. Empty (default) = "
        "single fused featuresCol. Requires gateParent (the gate is the shared "
        "per-node structure across domains).",
        typeConverter=TypeConverters.toListString,
    )
    domainBounds = Param(
        Params._dummy(), "domainBounds",
        "optional authoritative cumulative per-domain vocab offsets [0, V_0, V_0+V_1, "
        "...] (len == len(featuresCols)+1). When unset, the per-domain widths are read "
        "from the first row's vector sizes; set it as the escape hatch for an "
        "unrepresentative first row (every row is then validated against these widths).",
        typeConverter=TypeConverters.toListInt,
    )
    closureParents = Param(
        Params._dummy(), "closureParents",
        "JSON-encoded length-C list of parent-index lists selecting the DAG-CLOSURE "
        "head (Mondo label-side hierarchy) instead of the flat C-way logistic head. "
        "closureParents[l] lists the DIRECT parent LABEL indices of node l (a root "
        "has []), in the same [0, C) space as the label vector; the head models "
        "log P(node_l) = sum over the is-a closure of log sigmoid(w_a . theta), so "
        "P(child) <= P(parent). Empty (default) = flat head. len must equal numLabels. "
        "For this head prefer headOptimizer='newton' (it supplies a quasi-Newton "
        "Fisher; SGD does not converge the head — ADR 0039).",
        typeConverter=TypeConverters.toString,
    )
    warmStartFrom = Param(
        Params._dummy(), "warmStartFrom",
        "path to a previously-written save dir whose global params (topics/lambda) "
        "SEED this fit as its initial global params, with a FRESH Robbins-Monro "
        "iteration counter (rho restarts near rho_0). Empty (default) = cold start. "
        "DISTINCT from resumeFrom, which CONTINUES the counter (decayed rho) — a "
        "warm start needs the undecayed schedule so the head can move. The "
        "unsupervised-warm-start protocol (Hughes et al.): fit phase 1 at "
        "weightY == 0 (learns topics, leaves the head at its zero init), then "
        "warm-start a supervised phase 2 (weightY > 0) from it. Mutually exclusive "
        "with resumeFrom.",
        typeConverter=TypeConverters.toString,
    )

    def setWarmStartFrom(self, value: str):
        """Set warmStartFrom (path to a phase-1 checkpoint to warm-init from; empty = cold start)."""
        return self._set(warmStartFrom=value)

    def getWarmStartFrom(self) -> str:
        return str(self.getOrDefault(self.warmStartFrom))

    def setClosureParents(self, parents) -> "OnlinePCLDAEstimator":
        """Select the DAG-closure head from a length-C sequence of parent-index lists
        (JSON-encoded into the string Param). Empty/None restores the flat head."""
        if parents is None or parents == "":
            return self._set(closureParents="")
        encoded = parents if isinstance(parents, str) else json.dumps(
            [[int(p) for p in ps] for ps in parents])
        return self._set(closureParents=encoded)

    def getClosureParents(self) -> str:
        return str(self.getOrDefault(self.closureParents))

    def setGateParent(self, parent) -> "OnlinePCLDAEstimator":
        """Select the GATED topic engine from a DAG parent map {child: parent |
        [parents]} (JSON-encoded into the string Param). Empty/None = ungated."""
        if parent is None or parent == "":
            return self._set(gateParent="")
        encoded = parent if isinstance(parent, str) else json.dumps(
            {int(c): ([int(x) for x in p] if isinstance(p, (list, tuple, set))
                      else int(p)) for c, p in parent.items()})
        return self._set(gateParent=encoded)

    def getGateParent(self) -> str:
        return str(self.getOrDefault(self.gateParent))


def _build_model_and_config(
    estimator: "OnlinePCLDAEstimator", vocab_size: int,
    domains: list[int] | None = None,
) -> tuple[OnlinePCLDA, VIConfig]:
    """Translate Estimator Params into (OnlinePCLDA, VIConfig).

    docConcentration follows the LDA-shim convention (unset -> 1/k symmetric;
    length-1 -> scalar; length-k -> asymmetric vector).

    ``domains`` (per-domain vocab widths summing to ``vocab_size``) selects the
    MULTI-DOMAIN gated engine (per-domain lambda {m:(K,V_m)}); requires the
    topic-side gate (gateParent). None = single fused vocabulary.
    """
    k = estimator.getOrDefault("k")

    doc_conc = (
        estimator.getOrDefault("docConcentration")
        if estimator.isSet("docConcentration") else None
    )
    if doc_conc is None:
        alpha = 1.0 / k
    elif len(doc_conc) == 1:
        alpha = float(doc_conc[0])
    else:
        if len(doc_conc) != k:
            raise ValueError(
                f"docConcentration vector must have length k={k}, got {len(doc_conc)}."
            )
        alpha = np.asarray(doc_conc, dtype=np.float64)

    topic_conc = (
        estimator.getOrDefault("topicConcentration")
        if estimator.isSet("topicConcentration") else None
    )
    eta = 1.0 / k if topic_conc is None else float(topic_conc)
    seed = estimator.getOrDefault("seed") if estimator.isSet("seed") else None

    # DAG-closure head (Mondo label-side hierarchy) iff closureParents is supplied;
    # else the default flat C-way logistic head. Parsed here so a malformed structure
    # or C-mismatch fails at fit time with a clear message.
    C = int(estimator.getOrDefault("numLabels"))
    closure_raw = str(estimator.getOrDefault("closureParents"))
    head = None
    if closure_raw:
        try:
            parents = json.loads(closure_raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"closureParents is not valid JSON: {exc}") from exc
        if len(parents) != C:
            raise ValueError(
                f"closureParents has {len(parents)} nodes but numLabels={C}; the "
                "DAG must have one node per label head")
        head = DagClosureHead(parents)

    # Topic-side DAG gate (Gated-PC, ADR 0042): when gateParent is supplied, inject a
    # GatedOnlineLDA topic engine whose gated E-step welds each node's topic block to
    # its DAG subtree's documents. K is DERIVED from the layout (overriding k); the head
    # (flat or DAG-closure above) rides on the ungated label-free theta unchanged.
    topic_engine = None
    topic_support = None
    gate_raw = str(estimator.getOrDefault("gateParent"))
    if gate_raw:
        from spark_vi.models.topic.dag_placement import DagLayout
        from spark_vi.models.topic.gated_lda import GatedOnlineLDA
        try:
            parent_map = {int(c): p for c, p in json.loads(gate_raw).items()}
        except (json.JSONDecodeError, AttributeError, ValueError) as exc:
            raise ValueError(f"gateParent is not a valid JSON DAG map: {exc}") from exc
        lay = DagLayout(parent_map, n_bg=int(estimator.getOrDefault("gateNBg")),
                        tpn=int(estimator.getOrDefault("gateTpn")))
        k = lay.K                                    # layout owns K, overrides the k param
        # Doc-concentration alpha for the gated engine. Recomputed against lay.K (the
        # earlier doc_conc branch used the pre-gate k). The DEFAULT 1/K is razor-small at
        # whole-Mondo K (0.0022 at K=444), which collapses the doc-topic posterior theta
        # to a near-degenerate point where the differentiable-CAVI Jacobian d(theta)/d(eb)
        # UNDERFLOWS (~2.7e-90) — the supervised shaping gradient then cannot flow back to
        # the topics AT ALL (insight: alpha-collapse kills PC shaping upstream of the head;
        # the shaping Jacobian is only alive for alpha >~ 0.5). A scalar docConcentration
        # (e.g. 0.5) lifts alpha out of the collapse regime.
        if doc_conc is None:
            alpha = np.full(lay.K, 1.0 / lay.K, dtype=np.float64)
        elif len(doc_conc) == 1:
            alpha = np.full(lay.K, float(doc_conc[0]), dtype=np.float64)
        else:
            if len(doc_conc) != lay.K:
                raise ValueError(
                    f"docConcentration vector must have length K={lay.K} (gate layout), "
                    f"got {len(doc_conc)}.")
            alpha = np.asarray(doc_conc, dtype=np.float64)
        topic_engine = GatedOnlineLDA(
            lay, vocab_size, alpha=alpha, eta=1.0 / lay.K,
            domains=domains,                         # None = single fused vocab
            gamma_shape=estimator.getOrDefault("gammaShape"),
            cavi_max_iter=estimator.getOrDefault("caviMaxIter"),
            cavi_tol=estimator.getOrDefault("caviTol"), random_seed=seed)
        # LOCALIZED head: node c's logistic support = background + its gated block +
        # ancestors + SIBLINGS (DagLayout.allowed_with_siblings). Siblings are the
        # closure objective's contrast set — without them the wide top level collapses
        # (exp 0089 run 1: root's 29-way cond AUC fell to 0.57). Still O(depth + fan-out)
        # ≪ K (insight 0071). C = numLabels (incl root 0).
        if bool(estimator.getOrDefault("localizeHead")):
            C = int(estimator.getOrDefault("numLabels"))
            topic_support = [lay.allowed_with_siblings(c) for c in range(C)]
    elif domains is not None:
        raise ValueError(
            "featuresCols/domainBounds (multi-domain) require gateParent to be set: "
            "the per-node gate is the shared structure the per-domain topic blocks "
            "specialize under. Set gateParent, or use a single fused featuresCol.")

    model = OnlinePCLDA(
        K=k,
        vocab_size=vocab_size,
        C=estimator.getOrDefault("numLabels"),
        weight_y=float(estimator.getOrDefault("weightY")),
        lambda_w=float(estimator.getOrDefault("lambdaW")),
        grad_cavi_iters=int(estimator.getOrDefault("gradCaviIters")),
        head_lr_scale=float(estimator.getOrDefault("headLrScale")),
        topic_trust=float(estimator.getOrDefault("topicTrust")),
        weight_y_warmup_iters=int(estimator.getOrDefault("weightYWarmupIters")),
        head_optimizer=str(estimator.getOrDefault("headOptimizer")),
        head_lr=float(estimator.getOrDefault("headLr")),
        head_newton_ridge=float(estimator.getOrDefault("headNewtonRidge")),
        head_l2=float(estimator.getOrDefault("headL2")),
        head_intercept=bool(estimator.getOrDefault("headIntercept")),
        head_standardize=bool(estimator.getOrDefault("headStandardize")),
        alpha=alpha,
        eta=eta,
        optimize_alpha=estimator.getOrDefault("optimizeDocConcentration"),
        optimize_eta=estimator.getOrDefault("optimizeTopicConcentration"),
        gamma_shape=estimator.getOrDefault("gammaShape"),
        cavi_max_iter=estimator.getOrDefault("caviMaxIter"),
        cavi_tol=estimator.getOrDefault("caviTol"),
        random_seed=seed,
        head=head,
        topic_engine=topic_engine,
        topic_support=topic_support,
    )

    mbf = float(estimator.getOrDefault("subsamplingRate"))
    config = VIConfig(
        max_iterations=estimator.getOrDefault("maxIter"),
        learning_rate_tau0=estimator.getOrDefault("learningOffset"),
        learning_rate_kappa=estimator.getOrDefault("learningDecay"),
        mini_batch_fraction=(mbf if 0.0 < mbf < 1.0 else None),
        random_seed=seed,
    )
    return model, config


_ONLINE_PCLDA_DEFAULTS = dict(
    k=10, maxIter=20,
    featuresCol="features", topicDistributionCol="topicDistribution",
    learningOffset=1024.0, learningDecay=0.51, subsamplingRate=0.05,
    optimizeDocConcentration=True, optimizeTopicConcentration=False,
    gammaShape=100.0, caviMaxIter=100, caviTol=1e-3,
    numLabels=1, weightY=0.0, probabilityCol="probability",
    lambdaW=0.001, gradCaviIters=20, headLrScale=1.0, topicTrust=0.1,
    weightYWarmupIters=0, headOptimizer="sgd", headLr=0.05, headNewtonRidge=0.01,
    headL2=1e-3, headIntercept=False, headStandardize=False,
    closureParents="", warmStartFrom="",
    gateParent="", gateNBg=2, gateTpn=1, localizeHead=False, frontierCol="frontier",
    featuresCols=[],   # domainBounds intentionally omitted: it uses isSet (no default)
)


class OnlinePCLDAEstimator(_OnlinePCLDAParams, Estimator):
    """MLlib-shaped Estimator wrapping ``spark_vi.models.topic.pc.OnlinePCLDA``.

    Param defaults mirror the LDA shim for the shared subset. ``weightY``
    defaults to 0.0 (increment 1 = unsupervised); ``labelCol``/``labelMaskCol``
    default unset and are tolerated-absent on that path.
    """

    @keyword_only
    def __init__(
        self,
        *,
        k: int = 10,
        maxIter: int = 20,
        seed: int | None = None,
        featuresCol: str = "features",
        topicDistributionCol: str = "topicDistribution",
        learningOffset: float = 1024.0,
        learningDecay: float = 0.51,
        subsamplingRate: float = 0.05,
        docConcentration: list[float] | None = None,
        topicConcentration: float | None = None,
        optimizeDocConcentration: bool = True,
        optimizeTopicConcentration: bool = False,
        gammaShape: float = 100.0,
        caviMaxIter: int = 100,
        caviTol: float = 1e-3,
        numLabels: int = 1,
        labelCol: str | None = None,
        labelMaskCol: str | None = None,
        weightY: float = 0.0,
        probabilityCol: str = "probability",
        lambdaW: float = 0.001,
        gradCaviIters: int = 20,
        headLrScale: float = 1.0,
        topicTrust: float = 0.1,
        weightYWarmupIters: int = 0,
        headOptimizer: str = "sgd",
        headLr: float = 0.05,
        headNewtonRidge: float = 0.01,
        headL2: float = 1e-3,
        headIntercept: bool = False,
        headStandardize: bool = False,
        closureParents: str = "",
        gateParent: str = "",
        gateNBg: int = 2,
        gateTpn: int = 1,
        localizeHead: bool = False,
        frontierCol: str = "frontier",
        featuresCols: list[str] | None = None,
        domainBounds: list[int] | None = None,
        warmStartFrom: str = "",
        # _PersistenceParams kwargs — see that mixin's docstring; these MUST
        # appear here explicitly (not just on the mixin) for kwarg-style
        # construction (the cloud driver builds via kwargs).
        # test_constructor_accepts_persistence_kwargs pins this.
        saveInterval: int = -1,
        saveDir: str = "",
        resumeFrom: str = "",
    ) -> None:
        super().__init__()
        self._setDefault(**_ONLINE_PCLDA_DEFAULTS)
        self._set_persistence_defaults()
        # Diagnostic-only per-iteration callback (mirrors OnlineLDAEstimator).
        # Stored as an instance attribute — callables aren't MLlib-serializable
        # and persistence is deferred (ADR 0009).
        self._on_iteration = None
        # featuresCols/domainBounds carry no positive default (featuresCols defaults
        # to [] via _setDefault; domainBounds uses isSet). Drop an explicit None so
        # kwarg-style construction that leaves them unset does not clobber the
        # default / trip the list typeConverter — mirrors the gated_lda shim.
        kwargs = {k: v for k, v in self._input_kwargs.items()
                  if not (k in ("featuresCols", "domainBounds") and v is None)}
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, **kwargs) -> "OnlinePCLDAEstimator":
        return self._set(**kwargs)

    def setOnIteration(
        self, fn: Callable[[int, dict, list[float]], None] | None,
    ) -> "OnlinePCLDAEstimator":
        """Register a per-iteration diagnostic callback for the next fit.

        Signature fn(iter_num, global_params, elbo_trace); runs on the driver in
        the fit hot path. Must not mutate global_params. Not persisted.
        """
        self._on_iteration = fn
        return self

    def _fit(self, dataset) -> "OnlinePCLDAModel":
        from spark_vi.core.runner import VIRunner

        weight_y = float(self.getOrDefault("weightY"))
        # Multi-domain (featuresCols) vs single fused (featuresCol) vocabulary. When
        # featuresCols is set, per-domain widths come from the first row's vector
        # sizes (or explicit domainBounds), the total vocab is their sum, and the
        # per-domain gated engine is selected downstream (_build_model_and_config).
        fcols = list(self.getOrDefault("featuresCols") or [])
        if fcols:
            first = dataset.select(*fcols).head(1)
            if not first:
                raise ValueError("Cannot fit on an empty DataFrame.")
            if self.isSet("domainBounds"):
                bounds = [int(b) for b in self.getOrDefault("domainBounds")]
                if len(bounds) != len(fcols) + 1 or bounds[0] != 0 or \
                        any(b <= a for a, b in zip(bounds, bounds[1:])):
                    raise ValueError(
                        f"domainBounds {bounds} must be strictly increasing, start at "
                        f"0, and have len(featuresCols)+1 = {len(fcols) + 1} entries")
                domain_sizes = [b - a for a, b in zip(bounds, bounds[1:])]
            else:
                domain_sizes = [int(first[0][i].size) for i in range(len(fcols))]
            vocab_size = sum(domain_sizes)
            features_col = None
        else:
            features_col = self.getOrDefault("featuresCol")
            first = dataset.select(features_col).head(1)
            if not first:
                raise ValueError("Cannot fit on an empty DataFrame.")
            vocab_size = first[0][0].size
            domain_sizes = None

        model_obj, config = _build_model_and_config(
            self, vocab_size=vocab_size, domains=domain_sizes)

        # Validate persistence Params and splice checkpoint_dir/interval into
        # VIConfig. Returns (config, resume_path) where resume_path is a Path
        # or None. Mirrors OnlineLDAEstimator._fit exactly — both weight_y == 0
        # and > 0 fits go through this one VIRunner.fit code path.
        config, resume_path = apply_persistence_params(self, config)

        # warmStartFrom is a PC-specific, fresh-counter warm-INIT (distinct from
        # resumeFrom's continue-counter). Resolve + validate it here rather than
        # in the shared apply_persistence_params so the LDA/HDP resume surface is
        # untouched. Empty (default) => cold start (warm_start_path stays None).
        warm_start = self.getOrDefault("warmStartFrom")
        warm_start_path = None
        if warm_start:
            if resume_path is not None:
                raise ValueError(
                    "resumeFrom and warmStartFrom are mutually exclusive: "
                    "resumeFrom continues the Robbins-Monro counter, "
                    "warmStartFrom resets it. Set at most one."
                )
            if not (Path(warm_start) / "manifest.json").exists():
                raise FileNotFoundError(
                    f"No manifest.json at warmStartFrom path: {warm_start}"
                )
            warm_start_path = Path(warm_start)

        label_col = self.getOrDefault("labelCol") if self.isSet("labelCol") else None
        label_mask_col = (
            self.getOrDefault("labelMaskCol") if self.isSet("labelMaskCol") else None
        )
        C = self.getOrDefault("numLabels")

        # Supervised training needs labels: without labelCol every doc carries an
        # all-zero observed mask, so the head/topic correction would see no
        # signal and the "supervised" fit would silently reduce to unsupervised.
        # Fail fast instead.
        if weight_y != 0.0 and label_col is None:
            raise ValueError(
                f"weightY={weight_y} > 0 requires labelCol to be set (the head "
                "trains on it); no labelCol was provided."
            )

        # Column set to pull: features always; label columns only when present
        # (tolerated-absent at weightY == 0). Threaded like the STM shim threads
        # its covariate column. When the topic-side gate is on (gateParent set),
        # also pull frontierCol and marshal GatedPCDocument (the gated E-step reads
        # .frontier; the head reads .y/.label_mask — same row type serves both).
        gated = bool(str(self.getOrDefault("gateParent")))
        frontier_col = self.getOrDefault("frontierCol") if gated else None
        if gated and frontier_col not in dataset.columns:
            raise ValueError(
                f"gateParent is set but frontierCol={frontier_col!r} is not a column "
                f"of the input; the gated fit needs a per-doc frontier.")
        select_cols = list(fcols) if fcols else [features_col]
        if label_col is not None:
            select_cols.append(label_col)
        if label_mask_col is not None:
            select_cols.append(label_mask_col)
        if frontier_col is not None:
            select_cols.append(frontier_col)

        def _to_pc(row, _fc=features_col, _lc=label_col, _mc=label_mask_col,
                   _frc=frontier_col, _C=C, _fcs=(fcols or None), _ds=domain_sizes):
            if _frc is not None:
                return _row_to_gated_pc_document(row, _fc, _lc, _mc, _frc, _C,
                                                 features_cols=_fcs, domain_sizes=_ds)
            return _row_to_pc_document(row, _fc, _lc, _mc, _C,
                                       features_cols=_fcs, domain_sizes=_ds)

        pc_rdd = (
            dataset.select(*select_cols).rdd
            .map(_to_pc)
            .persist(StorageLevel.MEMORY_AND_DISK)
        )
        pc_rdd.count()  # materialize for VIRunner's strict cache precondition

        try:
            result = VIRunner(model_obj, config=config).fit(
                pc_rdd,
                resume_from=resume_path,
                warm_start_from=warm_start_path,
                on_iteration=self._on_iteration,
            )
        finally:
            pc_rdd.unpersist(blocking=False)

        out_model = OnlinePCLDAModel(result)
        for param in self.params:
            if self.isSet(param):
                out_model._set(**{param.name: self.getOrDefault(param)})
            elif self.hasDefault(param):
                out_model._setDefault(**{param.name: self.getOrDefault(param)})
        return out_model


class OnlinePCLDAModel(_OnlinePCLDAParams, _PersistableModel, Model):
    """MLlib-shaped Model wrapping a trained OnlinePCLDA VIResult.

    ``transform`` appends the label-free ``topicDistributionCol`` (theta) via
    the SAME CAVI used at train time — the faithfulness invariant. When the fit
    was supervised (``weightY > 0``) it ALSO appends a head-derived
    ``probabilityCol`` = ``sigmoid(w_CK . theta)`` (per-label P(y=1)); see
    ``predictProbability``.

    Persistable via ``_PersistableModel`` (save/load the wrapped VIResult),
    exactly as ``OnlineLDAModel`` is. ``_expected_model_class`` is the PC model
    tag the runner stamps (``OnlinePCLDA``), so ``load`` REJECTS a checkpoint
    from any other class — a PC checkpoint cannot load as LDA and vice-versa.
    """

    # Stamped into result.metadata by VIRunner as ``model_class`` (the runner
    # uses type(model).__name__ on the underlying VIModel). Used by
    # _PersistableModel.load to reject checkpoints from other model classes.
    _expected_model_class = "OnlinePCLDA"

    def __init__(self, result) -> None:  # result: VIResult
        super().__init__()
        self._result = result
        self._setDefault(**_ONLINE_PCLDA_DEFAULTS)
        self._set_persistence_defaults()

    @property
    def result(self):
        """The trained VIResult (global_params, elbo_trace, n_iterations, ...)."""
        return self._result

    def vocabSize(self) -> int:
        """V dimension of the trained lambda."""
        return int(self._result.global_params["lambda"].shape[1])

    def topicsMatrix(self):
        """Topic-word distribution as an MLlib DenseMatrix of shape (V, K)."""
        from pyspark.ml.linalg import DenseMatrix

        lam = self._result.global_params["lambda"]
        beta = lam / lam.sum(axis=1, keepdims=True)
        K, V = beta.shape
        return DenseMatrix(numRows=V, numCols=K, values=beta.T.flatten("F").tolist())

    def trainedAlpha(self) -> np.ndarray:
        """Trained α vector (length K) — the doc-topic Dirichlet concentration.

        Empirical-Bayes optimum when ``optimizeDocConcentration=True``, else the
        constructor input (broadcast to length K). Parity with ``OnlineLDAModel``;
        the α lives on the same LDA delegate the PC model wraps. Method (not
        @property) to avoid colliding with the ``docConcentration`` Param descriptor
        — see ADR 0012 §"Trained-scalar accessors".
        """
        return self._result.global_params["alpha"]

    def trainedTopicConcentration(self) -> float:
        """Trained η scalar — the topic-word Dirichlet concentration.

        Empirical-Bayes optimum when ``optimizeTopicConcentration=True``, else the
        initial η. Parity with ``OnlineLDAModel``; method (not @property) to avoid
        colliding with the same-named Param descriptor (ADR 0012)."""
        return float(self._result.global_params["eta"])

    def describeTopics(self, maxTermsPerTopic: int = 10):
        """DataFrame of (topic, termIndices, termWeights) — top terms per topic.

        Identical schema/orientation to ``OnlineLDAModel.describeTopics`` (and
        ``pyspark.ml.clustering.LDAModel.describeTopics``): the PC topics are a
        row-normalized λ exactly as in unsupervised LDA — the ``weight_y`` shaping
        changes *which* topics are learned, not how they are read out. The natural
        way to inspect PC's supervised topics (e.g. the disease-carrying topic in a
        rare-disease fit).
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

    def headWeights(self) -> np.ndarray:
        """The logistic head w_CK (C x K). All-zero after an increment-1 fit
        (the head is seeded and left at init on the unsupervised path)."""
        return self._result.global_params["w_CK"]

    def _transform(self, dataset):
        import hashlib

        from pyspark.ml.linalg import DenseVector, VectorUDT
        from pyspark.sql import functions as F
        from scipy.special import digamma

        from spark_vi.models.topic.lda import _cavi_doc_inference

        lam = self._result.global_params["lambda"]
        # Multi-domain: λ is a per-domain dict {m:(K,V_m)}; fuse to the concatenated
        # (K, V) representation the shared CAVI reads (each domain row-normalized on
        # its own vocab, then concatenated in domain order — the same fusing the
        # gated engine's _assemble_expElogbeta does). domain_sizes drives the per-doc
        # feature concatenation below.
        if isinstance(lam, dict):
            ms = sorted(lam)
            domain_sizes = [int(lam[m].shape[1]) for m in ms]
            expElogbeta = np.concatenate(
                [np.exp(digamma(lam[m]) - digamma(lam[m].sum(axis=1, keepdims=True)))
                 for m in ms], axis=1)
        else:
            domain_sizes = None
            expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
        alpha = self._result.global_params["alpha"]
        gamma_shape = float(self.getOrDefault("gammaShape"))
        cavi_max_iter = int(self.getOrDefault("caviMaxIter"))
        cavi_tol = float(self.getOrDefault("caviTol"))
        K = expElogbeta.shape[0]

        sc = dataset.sparkSession.sparkContext
        bcast = sc.broadcast({
            "expElogbeta": expElogbeta, "alpha": alpha, "gamma_shape": gamma_shape,
            "cavi_max_iter": cavi_max_iter, "cavi_tol": cavi_tol, "K": K,
            "domain_sizes": domain_sizes,
        })

        def _infer(*features):
            p = bcast.value
            if p["domain_sizes"] is not None:
                from spark_vi.mllib.topic.gated_lda import _concat_domain_features
                indices, counts = _concat_domain_features(features, p["domain_sizes"])
            else:
                doc = _vector_to_bow_document(features[0])
                indices, counts = doc.indices, doc.counts
            # Content-deterministic gamma_init (mirrors OnlinePCLDA/GatedOnlineLDA):
            # identical docs get identical init on every run, so a scoring path is
            # reproducible and independent of Spark partition/executor order.
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
        out = dataset.withColumn(out_col, infer_udf(*feat_args))

        # Supervised fit -> also append the head-derived per-label probability.
        # The topicDistribution UDF already inferred theta from the SAME CAVI; the
        # head is a cheap sigmoid(w_CK . theta) on top of it (no second inference).
        if float(self.getOrDefault("weightY")) != 0.0:
            from spark_vi.models.topic.pc import _predict_proba_np
            w_CK = self._result.global_params["w_CK"]
            wbcast = sc.broadcast(w_CK)
            # Per-node intercept (head_intercept). Zeros when off, so the prediction is
            # unchanged for legacy heads.
            b_CK = self._result.global_params.get("b_CK")
            bbcast = sc.broadcast(None if b_CK is None else np.asarray(b_CK, np.float64))
            # DAG-closure head -> broadcast its (C,C) closure matrix so the per-label
            # probability is the closure PRODUCT P(node_l), not the flat sigmoid. Flat
            # head -> None. We broadcast arrays only (never the head object, whose
            # autograd closure is unpicklable), reusing the engine's predict fn.
            closure_raw = str(self.getOrDefault("closureParents"))
            closure_matrix = None
            if closure_raw:
                from spark_vi.models.topic.pc import DagClosureHead
                closure_matrix = DagClosureHead(json.loads(closure_raw))._closure_matrix
            mbcast = sc.broadcast(closure_matrix)

            def _proba(theta, _wb=wbcast, _mb=mbcast, _bb=bbcast):
                th = np.asarray(theta.toArray(), dtype=np.float64)
                return DenseVector(_predict_proba_np(th, _wb.value, _mb.value, _bb.value))

            proba_udf = F.udf(_proba, returnType=VectorUDT())
            prob_col = self.getOrDefault("probabilityCol")
            out = out.withColumn(prob_col, proba_udf(F.col(out_col)))
        return out

    def predictProbability(self, dataset):
        """Append the head-derived ``probabilityCol`` = sigmoid(w_CK . theta).

        Per-label P(y=1) for every doc, from the trained logistic head on top of
        the label-free CAVI theta. Requires a supervised fit (``weightY > 0``): on
        the unsupervised path the head is at its zero seed (sigmoid(0) = 0.5 for
        every doc/label), so no meaningful probability exists.

        This is exactly the column ``transform`` already appends when
        ``weightY > 0``; provided as a named entry point for callers that only want
        the probability.
        """
        if float(self.getOrDefault("weightY")) == 0.0:
            raise NotImplementedError(
                "predictProbability requires a supervised fit (weightY > 0); the "
                "unsupervised head is at its zero seed (P == 0.5 everywhere)."
            )
        return self.transform(dataset)

    def logLikelihood(self, dataset):
        """Not implemented in this v1 shim (parity with ``OnlineLDAModel``).

        The training-time ELBO trace is available on the underlying VIResult via
        ``OnlinePCLDAModel.result.elbo_trace`` (the unsupervised LDA bound; the
        supervised NLL is a penalty on the globals, not part of the reported bound).
        """
        raise NotImplementedError(
            "logLikelihood is not implemented in this v1 shim. The training-time "
            "ELBO trace is available on the underlying VIResult via "
            "OnlinePCLDAModel.result.elbo_trace."
        )

    def logPerplexity(self, dataset):
        """Not implemented in this v1 shim (parity with ``OnlineLDAModel``).

        See ``logLikelihood``; use ``result.elbo_trace`` for the training-time bound.
        """
        raise NotImplementedError(
            "logPerplexity is not implemented in this v1 shim. The training-time "
            "ELBO trace is available on the underlying VIResult via "
            "OnlinePCLDAModel.result.elbo_trace."
        )

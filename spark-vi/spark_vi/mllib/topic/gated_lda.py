"""MLlib Estimator/Model shim for GatedOnlineLDA (hierarchical case-finding placement).

Mirrors mllib/topic/lda.py (ADR 0009): a translation layer over GatedOnlineLDA + VIRunner.
fit trains GATED (each row carries features + a frontier = set of DAG node ids); transform
folds held-out docs in UNGATED (full-K) and emits per-node affinity. init="random" is the
validated default; init="spectral" seeds lambda from anchor-word spectral recovery (Arora et al.
2013), routed by spectralMethod between DENSE (collect the corpus to the driver, exact V×V,
mirroring the STM shim's dense spectral path, mllib/topic/stm.py) and SCALABLE (distributed
random-projection sketch over the RDD, ADR 0032 — the gated analogue, for large vocabularies).
"auto" picks dense below spectralMaxVocab and scalable at/above it (resolve_spectral_method).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from pyspark import StorageLevel, keyword_only
from pyspark.ml.base import Estimator, Model
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasFeaturesCol, HasLabelCol, HasMaxIter, HasSeed

from spark_vi.core.config import VIConfig
from spark_vi.models.topic.dag_placement import DagLayout
from spark_vi.models.topic.domains import domains_to_bounds, resolve_per_domain
from spark_vi.models.topic.gated_lda import GatedOnlineLDA
from spark_vi.models.topic.types import GatedBOWDocument
from spark_vi.mllib.topic._common import _vector_to_bow_document


class _GatedLDAParams(HasFeaturesCol, HasLabelCol, HasMaxIter, HasSeed):
    featuresCols = Param(Params._dummy(), "featuresCols",
                         "ordered per-domain feature column names (multi-domain, "
                         "MixEHR-style per-modality vocabularies; Li, Nair, Lu et al. "
                         "2020, Nat. Commun.). Unset = single-domain via featuresCol.",
                         typeConverter=TypeConverters.toListString)
    domainBounds = Param(Params._dummy(), "domainBounds",
                         "optional explicit cumulative per-domain offsets "
                         "[0, V_0, V_0+V_1, ...]; normally DERIVED from the first "
                         "row's per-column vector sizes.",
                         typeConverter=TypeConverters.toListInt)
    omega = Param(Params._dummy(), "omega",
                  "per-domain modality weight on the doc-topic accumulation "
                  "(theta only; lambda sstats and the data loglik use TRUE counts). "
                  "Default all 1.0 = faithful MixEHR, volume speaks (Li, Nair, Lu "
                  "et al. 2020). A tuned-vs-task tempering weight, NOT fitted.",
                  typeConverter=TypeConverters.toListFloat)
    etaPerDomain = Param(Params._dummy(), "etaPerDomain",
                         "per-domain Dirichlet prior on the topic-word blocks; "
                         "unset = the scalar 1/K used by the single-domain path.",
                         typeConverter=TypeConverters.toListFloat)
    parent = Param(Params._dummy(), "parent",
                   "DAG parent map {child_int: parent_int or [parent_ints]}, anchor->0",
                   typeConverter=TypeConverters.identity)
    nBg = Param(Params._dummy(), "nBg", "number of shared background topics",
                typeConverter=TypeConverters.toInt)
    tpn = Param(Params._dummy(), "tpn", "topics per DAG node",
                typeConverter=TypeConverters.toInt)
    nodeAffinityCol = Param(Params._dummy(), "nodeAffinityCol",
                            "output column: per-node affinity Vector",
                            typeConverter=TypeConverters.toString)
    caviMaxIter = Param(Params._dummy(), "caviMaxIter", "inner CAVI max iters",
                        typeConverter=TypeConverters.toInt)
    caviTol = Param(Params._dummy(), "caviTol", "inner CAVI tolerance",
                    typeConverter=TypeConverters.toFloat)
    gammaShape = Param(Params._dummy(), "gammaShape", "Gamma init shape for gamma/lambda",
                       typeConverter=TypeConverters.toFloat)
    nodeAlphaScale = Param(Params._dummy(), "nodeAlphaScale",
                           "multiplier on the per-node-topic Dirichlet alpha "
                           "relative to the background alpha (1/K); 1.0 = symmetric "
                           "(default). <1 down-weights the node blocks so a document "
                           "needs stronger evidence to place mass on a disease node — "
                           "an asymmetric prior (Wallach et al. 2009) that reflects "
                           "the low prevalence of any single node and, at ungated "
                           "transform time, keeps background docs on the background "
                           "block.",
                           typeConverter=TypeConverters.toFloat)
    miniBatchFraction = Param(Params._dummy(), "miniBatchFraction",
                              "SVI mini-batch fraction in (0, 1]; each iteration "
                              "samples this fraction of the corpus. 0.0 = full-batch "
                              "(default). Mini-batching is what makes the decaying "
                              "Robbins-Monro step size appropriate (Hoffman et al. 2013).",
                              typeConverter=TypeConverters.toFloat)
    learningRateTau0 = Param(Params._dummy(), "learningRateTau0",
                             "SVI step-size delay tau0 in rho_t = (tau0 + t + 1)^-kappa "
                             "(Hoffman et al. 2013). Larger tau0 down-weights early "
                             "iterations (a gentler, less aggressive slow start). "
                             "Default 1.0.",
                             typeConverter=TypeConverters.toFloat)
    learningRateKappa = Param(Params._dummy(), "learningRateKappa",
                              "SVI step-size decay exponent kappa in rho_t = "
                              "(tau0 + t + 1)^-kappa. Robbins-Monro convergence holds "
                              "for kappa in (0.5, 1]; larger kappa decays faster. "
                              "Default 0.7.",
                              typeConverter=TypeConverters.toFloat)
    init = Param(Params._dummy(), "init",
                 "lambda init strategy: 'random' (default) or 'spectral' "
                 "(dense block-aligned anchor-word seed, Arora et al. 2013)",
                 typeConverter=TypeConverters.toString)
    spectralMaxVocab = Param(Params._dummy(), "spectralMaxVocab",
                             "max vocab for the DENSE spectral init (V x V driver "
                             "co-occurrence); the spectralMethod='auto' threshold "
                             "below which dense is used (scalable at/above)",
                             typeConverter=TypeConverters.toInt)
    spectralMethod = Param(Params._dummy(), "spectralMethod",
                           "spectral routing: 'auto'|'dense'|'scalable' "
                           "(auto -> dense if V < spectralMaxVocab else scalable)")
    spectralD = Param(Params._dummy(), "spectralD",
                      "random-projection dim for scalable init (0 = auto: "
                      "min(V, max(K, 1000)))")
    spectralMinDocFreq = Param(Params._dummy(), "spectralMinDocFreq",
                               "min within-group document frequency for a scalable "
                               "anchor candidate")
    anchorScope = Param(Params._dummy(), "anchorScope",
                        "which docs feed each spectral anchor set: 'closure' "
                        "(default; node u from every doc with u in its closure, "
                        "background from all docs) or 'frontier' (node u only from "
                        "docs where u is the most-specific attested node, background "
                        "only from empty-frontier docs) — 'frontier' stops "
                        "background/ancestors stealing a descendant's anchor")
    spectralTopoOrder = Param(Params._dummy(), "spectralTopoOrder",
                              "spectral init deflation order: 'forward' (default; nodes "
                              "ancestors-first, each deflated against its ancestors = "
                              "increment-over-ancestors) or 'reverse' (leaves-first, each "
                              "deflated against its descendants = residual-after-descendants)",
                              typeConverter=TypeConverters.toString)
    optimizeDocConcentration = Param(Params._dummy(), "optimizeDocConcentration",
                                     "learn an asymmetric per-node Dirichlet alpha "
                                     "from data (Wallach et al. 2009); nodeAlphaScale "
                                     "sets the initial alpha, optimize refines it. "
                                     "Default False.",
                                     typeConverter=TypeConverters.toBoolean)
    transformAlphaMode = Param(Params._dummy(), "transformAlphaMode",
                               "deployment (transform/fold-in) Dirichlet alpha: "
                               "'fitted' (default; the fitted alpha, asymmetric if "
                               "optimizeDocConcentration) | 'symmetric' (flat per-topic "
                               "alpha = transformAlpha, default 1/K; neutral between "
                               "nodes) | 'block_balanced' (all nodes equal, background "
                               "collective mass = transformBgWeight; neutral between "
                               "nodes AND controls the bg-vs-node baseline). Decouples "
                               "the fitting-aid alpha from the deployment prior so a "
                               "learned alpha need not bias the placement argmax.",
                               typeConverter=TypeConverters.toString)
    transformAlpha = Param(Params._dummy(), "transformAlpha",
                           "per-topic alpha for transformAlphaMode='symmetric' "
                           "(<=0 -> 1/K).", typeConverter=TypeConverters.toFloat)
    transformBgWeight = Param(Params._dummy(), "transformBgWeight",
                              "collective background prior mass in (0,1) for "
                              "transformAlphaMode='block_balanced' (nodes split the "
                              "rest equally; total concentration 1.0). Default 0.5.",
                              typeConverter=TypeConverters.toFloat)


# Per-domain lambda scale for a SPECTRAL seed. spectral_init.split_domains returns
# ROW-STOCHASTIC blocks (it renormalizes each block to sum 1, discarding whatever
# lambda scale the joint carried), so when the shim converts a scalable joint seed
# to a per-domain dict it must scale each block back up to lambda magnitude. This is
# imported (not redefined) from gated_init so the shim's scalable->dict conversion
# and gated_init's dense/multidomain seeds cannot drift to different strengths --
# they are the SAME constant (SP3a whole-branch review Minor 1).
from spark_vi.models.topic.gated_init import (          # noqa: E402
    SPECTRAL_LAMBDA_SCALE as _SPECTRAL_LAMBDA_SCALE,
)


def _concat_domain_features(vectors, sizes):
    """Concatenate per-domain sparse vectors into the engine's single id space.

    The engine stores one topic-word matrix per domain but consumes ONE
    concatenated token-id space, with a token's domain recovered by
    `searchsorted(domain_bounds, w)`. Domain m's local ids therefore shift by
    `sum(sizes[:m])`. Because each domain's ids are already ascending and the
    domains are laid out in order, the concatenated ids are GLOBALLY sorted,
    which is what the E-step's `expElogbeta[:, indices]` gather assumes.

    Raises ValueError naming the domain and both widths if a vector disagrees
    with the established layout: the layout is derived once per fit, and a row
    that silently re-lays-out the vocabulary would corrupt the fit with no
    symptom (SP3a design).

    Equally, `len(vectors)` must equal `len(sizes)`: a bare zip TRUNCATES to the
    shorter, silently dropping a trailing domain — the same invisible re-layout,
    reachable wherever the columns and the widths do not arrive together (e.g. a
    model carrying featuresCols and domainBounds as separate Params).
    """
    import numpy as np
    vectors, sizes = list(vectors), list(sizes)
    if len(vectors) != len(sizes):
        raise ValueError(
            f"_concat_domain_features got {len(vectors)} vector(s) for "
            f"{len(sizes)} domain size(s); the counts must match or a trailing "
            f"domain would be silently dropped")
    idx_parts, cnt_parts, offset = [], [], 0
    for m, (v, width) in enumerate(zip(vectors, sizes)):
        if int(v.size) != int(width):
            raise ValueError(
                f"featuresCols domain {m} vector size {int(v.size)} != expected "
                f"{int(width)} (layout derived from the first row); every row must "
                f"use the same per-domain vocabulary widths")
        bow = _vector_to_bow_document(v)
        idx_parts.append(bow.indices.astype(np.int64) + offset)
        cnt_parts.append(bow.counts)
        offset += int(width)
    indices = np.concatenate(idx_parts).astype(np.int32) if idx_parts else np.empty(0, np.int32)
    counts = np.concatenate(cnt_parts) if cnt_parts else np.empty(0, np.float64)
    return indices, counts


def _layout(est_or_model) -> DagLayout:
    return DagLayout(est_or_model.getOrDefault("parent"),
                     n_bg=est_or_model.getOrDefault("nBg"),
                     tpn=est_or_model.getOrDefault("tpn"))


def _deployment_alpha(fitted_alpha, lay, mode, scalar, bg_weight):
    """Build the length-K Dirichlet alpha used at TRANSFORM (fold-in) time.

    'fitted'         -> the fitted alpha unchanged (asymmetric if the model learned it).
    'symmetric'      -> a flat per-topic alpha (scalar>0 else 1/K); neutral between nodes.
    'block_balanced' -> all node blocks equal AND an explicit background-vs-nodes split:
                        background collective mass = bg_weight, the n_nodes node blocks
                        split (1-bg_weight) equally (each node's tpn topics share its
                        node share); total concentration 1.0. Neutral between nodes,
                        controls the bg-vs-node baseline independent of topic counts.

    Decouples the fitting-aid alpha from the deployment prior: a learned alpha can help
    the fit while the deployment stays inter-node-neutral so it does not bias the
    node-affinity argmax (insight 0061). NOTE: for RANKING/AUC the inter-node-symmetric
    modes are equivalent (an equal per-node baseline cancels in the argmax); they differ
    on the bg-vs-node baseline, which matters for calibration / operating points.
    """
    K = lay.K
    if mode == "fitted":
        return np.asarray(fitted_alpha, dtype=np.float64)
    if mode == "symmetric":
        a = float(scalar) if scalar and scalar > 0 else 1.0 / K
        return np.full(K, a, dtype=np.float64)
    if mode == "block_balanced":
        w = float(bg_weight)
        if not (0.0 < w < 1.0):
            raise ValueError(f"transformBgWeight must be in (0,1), got {bg_weight}")
        out = np.empty(K, dtype=np.float64)
        out[:lay.n_bg] = w / lay.n_bg                        # background collective = w
        per_node_topic = (1.0 - w) / (len(lay.nodes) * lay.tpn)  # nodes equal, share 1-w
        for u in lay.nodes:
            out[lay.block[u]] = per_node_topic
        return out
    raise ValueError(
        f"unknown transformAlphaMode {mode!r}; "
        "expected 'fitted' | 'symmetric' | 'block_balanced'")


class GatedLDAEstimator(_GatedLDAParams, Estimator):
    @keyword_only
    def __init__(self, *, featuresCol="features", featuresCols=None,
                 domainBounds=None, omega=None, etaPerDomain=None,
                 labelCol="frontier", parent=None,
                 nBg=2, tpn=1, maxIter=20, seed=None, caviMaxIter=100, caviTol=1e-3,
                 gammaShape=100.0, init="random", spectralMaxVocab=8000,
                 spectralMethod="auto", spectralD=0, spectralMinDocFreq=5,
                 anchorScope="closure", spectralTopoOrder="forward",
                 nodeAlphaScale=1.0, miniBatchFraction=0.0,
                 learningRateTau0=1.0, learningRateKappa=0.7,
                 optimizeDocConcentration=False, transformAlphaMode="fitted",
                 transformAlpha=0.0, transformBgWeight=0.5):
        super().__init__()
        # featuresCols defaults to [] = unset = single-domain via featuresCol.
        # domainBounds gets NO default on purpose: `isSet("domainBounds")` is what
        # distinguishes "authoritative explicit layout" from "derive from row one".
        # omega and etaPerDomain get no default for the same reason: unset must
        # reach the engine as None / the scalar 1/K, which is the pre-multi-domain
        # behavior, and `omega=[1, 1, ...]` is only legal WITH domains.
        self._setDefault(featuresCol="features", featuresCols=[],
                         labelCol="frontier", nBg=2, tpn=1,
                         maxIter=20, nodeAffinityCol="nodeAffinity",
                         caviMaxIter=100, caviTol=1e-3, gammaShape=100.0,
                         init="random", spectralMaxVocab=8000,
                         spectralMethod="auto", spectralD=0, spectralMinDocFreq=5,
                         anchorScope="closure", spectralTopoOrder="forward",
                         nodeAlphaScale=1.0,
                         miniBatchFraction=0.0, learningRateTau0=1.0,
                         learningRateKappa=0.7, optimizeDocConcentration=False,
                         transformAlphaMode="fitted", transformAlpha=0.0,
                         transformBgWeight=0.5)
        self.setParams(**self._input_kwargs)
        # Diagnostic-only per-iteration callback (mirrors OnlineLDAEstimator).
        # Stored as an instance attribute, not a Param — callables aren't
        # MLlib-serializable and persistence is deferred (ADR 0009).
        self._on_iteration = None

    @keyword_only
    def setParams(self, **kwargs):
        return self._set(**kwargs)

    def setOnIteration(
        self, fn: Callable[[int, dict, list[float]], None] | None,
    ) -> "GatedLDAEstimator":
        """Register a per-iteration diagnostic callback for the next fit.

        Signature: fn(iter_num, global_params, elbo_trace). Runs on the driver in
        the fit's hot path; throttle with a modulo if non-trivial. The callback
        must not mutate global_params (the same dict feeds the next iteration's
        broadcast). Not persisted (callables aren't MLlib-serializable; ADR 0009).
        """
        self._on_iteration = fn
        return self

    def _fit(self, dataset) -> "GatedLDAModel":
        from spark_vi.core.runner import VIRunner
        if self.getOrDefault("parent") is None:
            raise ValueError("GatedLDAEstimator requires a `parent` DAG map.")
        lay = _layout(self)

        fcols = list(self.getOrDefault("featuresCols") or [])
        label_col = self.getOrDefault("labelCol")
        if fcols:
            first = dataset.select(*fcols).head(1)
            if not first:
                raise ValueError("Cannot fit on an empty DataFrame.")
            if self.isSet("domainBounds"):
                # Explicit bounds are AUTHORITATIVE, not merely cross-checked: this
                # is the escape hatch for a dataset whose first row is
                # unrepresentative, so it must not be rejected for disagreeing with
                # that row. Every row -- the first included -- is then validated
                # against these widths by _concat_domain_features.
                bounds = [int(b) for b in self.getOrDefault("domainBounds")]
                if len(bounds) != len(fcols) + 1 or bounds[0] != 0 or \
                        any(b <= a for a, b in zip(bounds, bounds[1:])):
                    raise ValueError(
                        f"domainBounds {bounds} must be strictly increasing, start "
                        f"at 0, and have len(featuresCols)+1 = {len(fcols) + 1} entries")
                sizes = [b - a for a, b in zip(bounds, bounds[1:])]
            else:
                sizes = [int(first[0][i].size) for i in range(len(fcols))]
            V = sum(sizes)
            domains = sizes
        else:
            features_col = self.getOrDefault("featuresCol")
            first = dataset.select(features_col).head(1)
            if not first:
                raise ValueError("Cannot fit on an empty DataFrame.")
            V = first[0][0].size
            domains = None
        seed = self.getOrDefault("seed") if self.isSet("seed") else None
        init = self.getOrDefault("init")

        # Validate init early (fail fast on the driver, not deep in a Spark task).
        from spark_vi.models.topic.gated_init import INIT_STRATEGIES
        if init != "random" and init not in INIT_STRATEGIES:
            raise ValueError(
                f"unknown init strategy {init!r}; "
                f"known: {['random'] + sorted(INIT_STRATEGIES)}"
            )
        # Dirichlet alpha over the K topics. Symmetric 1/K by default; when
        # nodeAlphaScale != 1, the per-node blocks (contiguous topics
        # [n_bg, K), background is [0, n_bg)) are scaled to make them a priori
        # rarer — a block-asymmetric prior (Wallach et al. 2009). OnlineLDA
        # accepts a length-K alpha vector; this is the INITIAL alpha. With
        # optimizeDocConcentration=False (default) it holds through the fit and
        # into transform; with it True, the gated per-node Newton step refines it.
        node_alpha_scale = float(self.getOrDefault("nodeAlphaScale"))
        alpha_vec = np.full(lay.K, 1.0 / lay.K, dtype=np.float64)
        alpha_vec[lay.n_bg:] *= node_alpha_scale

        optimize_alpha = bool(self.getOrDefault("optimizeDocConcentration"))
        frontier_hist = None
        if optimize_alpha:
            # Static allowed-set group structure from the (fixed) training labels.
            # Foreground+background scale; collected once at fit time.
            frontier_hist = {
                frozenset(int(x) for x in (fr or [])): int(n)
                for fr, n in (
                    dataset.select(label_col).rdd
                    .map(lambda r: frozenset(int(x) for x in (r[0] or [])))
                    .countByValue().items())
            }
        # omega / etaPerDomain are multi-domain-only quantities, forwarded when SET
        # and otherwise left at the pre-multi-domain defaults: None (unweighted,
        # MixEHR-faithful raw volume; Li, Nair, Lu et al. 2020) and the scalar 1/K
        # eta the single-domain path has always used.
        #
        # Deliberately NOT gated on `fcols`: a per-domain weight with no domains is
        # a contradiction, and the engine RAISES a named error for it
        # (_resolve_omega / resolve_per_domain). Gating here would instead discard
        # the caller's omega silently -- a fit that runs and serves an unweighted
        # theta the caller never asked for.
        model_obj = GatedOnlineLDA(
            lay, V, init=init, domains=domains,
            optimize_alpha=optimize_alpha, frontier_histogram=frontier_hist,
            alpha=alpha_vec,
            omega=(list(self.getOrDefault("omega")) if self.isSet("omega") else None),
            eta=(list(self.getOrDefault("etaPerDomain"))
                 if self.isSet("etaPerDomain") else 1.0 / lay.K),
            gamma_shape=self.getOrDefault("gammaShape"),
            cavi_max_iter=self.getOrDefault("caviMaxIter"),
            cavi_tol=self.getOrDefault("caviTol"),
            random_seed=seed,
        )
        # SVI schedule: mini_batch_fraction 0.0 (the shim default) -> None ==
        # full-batch (every iteration sees the whole corpus). A value in (0, 1]
        # switches to mini-batch SVI, which is what makes the decaying step size
        # legitimate (a decaying rho over full batches stalls — VIConfig docs).
        mbf = float(self.getOrDefault("miniBatchFraction"))
        config = VIConfig(
            max_iterations=self.getOrDefault("maxIter"), random_seed=seed,
            mini_batch_fraction=(mbf if mbf and mbf > 0.0 else None),
            learning_rate_tau0=float(self.getOrDefault("learningRateTau0")),
            learning_rate_kappa=float(self.getOrDefault("learningRateKappa")),
        )

        if fcols:
            n_dom = len(fcols)

            def _to_gated(row):
                indices, counts = _concat_domain_features(
                    [row[i] for i in range(n_dom)], sizes)
                frontier = frozenset(int(x) for x in (row[n_dom] or []))
                return GatedBOWDocument(indices=indices, counts=counts,
                                        length=int(counts.sum()), frontier=frontier)

            rdd = (dataset.select(*fcols, label_col).rdd.map(_to_gated)
                   .persist(StorageLevel.MEMORY_AND_DISK))
        else:
            def _to_gated(row):
                bow = _vector_to_bow_document(row[0])
                frontier = frozenset(int(x) for x in (row[1] or []))
                return GatedBOWDocument(indices=bow.indices, counts=bow.counts,
                                        length=bow.length, frontier=frontier)

            rdd = (dataset.select(features_col, label_col).rdd.map(_to_gated)
                   .persist(StorageLevel.MEMORY_AND_DISK))
        rdd.count()

        # Non-random init: seed lambda from anchor-word spectral recovery. init
        # "spectral" routes dense (collect the corpus to the driver, exact V×V,
        # validated small-V default) vs scalable (distributed random-projection
        # sketch, ADR 0032 — the gated analogue), by spectralMethod. Passed to
        # initialize_global via data_summary; dense hands {train_docs,train_labels},
        # scalable hands a precomputed {spectral_lambda}.
        data_summary = None
        if init != "random":
            from spark_vi.mllib.topic.stm import resolve_spectral_method
            resolved = resolve_spectral_method(
                self.getOrDefault("spectralMethod"), V,
                threshold=self.getOrDefault("spectralMaxVocab"))
            if resolved == "scalable":
                from spark_vi.models.topic.gated_init import (
                    scalable_block_aligned_lambda,
                )
                sd = int(self.getOrDefault("spectralD"))
                lam0 = scalable_block_aligned_lambda(
                    rdd, lay, V,
                    d=(sd if sd > 0 else None),
                    seed=(seed or 0),
                    min_doc_freq=int(self.getOrDefault("spectralMinDocFreq")),
                    anchor_scope=self.getOrDefault("anchorScope"),
                    topo_order=self.getOrDefault("spectralTopoOrder"),
                )
                if fcols:
                    # scalable_block_aligned_lambda returns the JOINT (K, V) lambda,
                    # but the engine's multi-domain arm consumes a per-domain dict
                    # {m: (K, V_m)} (it calls .items() on it). Convert here, mirroring
                    # gated_init.multidomain_spectral_lambda's final step exactly:
                    # split_domains renormalizes each block ROW-STOCHASTIC, dropping
                    # the joint's lambda scale, so _SPECTRAL_LAMBDA_SCALE is re-applied
                    # — a bare split would hand the engine a seed of row mass ~1
                    # instead of ~200, which looks like a working fit that simply never
                    # concentrates. Reachable on shipped defaults: spectralMethod="auto"
                    # routes scalable once V >= spectralMaxVocab, and a CONCATENATED
                    # multi-domain V crosses that far more easily than a single one.
                    from spark_vi.models.topic.spectral_init import split_domains
                    blocks = split_domains(lam0, domains_to_bounds(sizes).tolist())
                    lam0 = {m: blocks[m] * _SPECTRAL_LAMBDA_SCALE + 1e-9
                            for m in range(len(sizes))}
                data_summary = {"spectral_lambda": lam0}
            else:  # dense — collect-to-driver exact path
                # Multi-domain: build the driver-side docs through the SAME
                # _concat_domain_features helper the fit's row mapper uses, so the
                # spectral seed sees exactly the concatenated ids the fit will see
                # (a divergent layout here would seed the wrong vocabulary silently).
                if fcols:
                    collected = dataset.select(*fcols, label_col).collect()
                    train_docs, train_labels = [], []
                    for r in collected:
                        indices, counts = _concat_domain_features(
                            [r[i] for i in range(len(fcols))], sizes)
                        train_docs.append(np.repeat(indices, counts.astype(int)))
                        train_labels.append(
                            frozenset(int(x) for x in (r[len(fcols)] or [])))
                else:
                    collected = dataset.select(features_col, label_col).collect()
                    train_docs, train_labels = [], []
                    for r in collected:
                        bow = _vector_to_bow_document(r[0])
                        train_docs.append(np.repeat(bow.indices, bow.counts.astype(int)))
                        train_labels.append(frozenset(int(x) for x in (r[1] or [])))
                # anchor_scope/topo_order ride along so initialize_global's dense
                # strategy honors them too (the scalable path already baked them
                # into lam0).
                data_summary = {"train_docs": train_docs, "train_labels": train_labels,
                                "anchor_scope": self.getOrDefault("anchorScope"),
                                "topo_order": self.getOrDefault("spectralTopoOrder")}

        try:
            result = VIRunner(model_obj, config=config).fit(
                rdd, data_summary=data_summary, on_iteration=self._on_iteration)
        finally:
            rdd.unpersist(blocking=False)

        out = GatedLDAModel(result, parent=self.getOrDefault("parent"),
                            nBg=self.getOrDefault("nBg"), tpn=self.getOrDefault("tpn"))
        for p in self.params:
            if self.isSet(p):
                out._set(**{p.name: self.getOrDefault(p)})
            elif self.hasDefault(p):
                out._setDefault(**{p.name: self.getOrDefault(p)})
        return out


class GatedLDAModel(_GatedLDAParams, Model):
    # Documents intent for a future _PersistableModel checkpoint/persist mixin; the model
    # does not inherit _PersistableModel in v1, so this attribute is currently inert
    # (persistence is deferred to v2). Kept because it costs nothing and records the plan.
    _expected_model_class = "GatedOnlineLDA"

    def __init__(self, result, *, parent, nBg, tpn):
        super().__init__()
        self._result = result
        # featuresCols=[] (= unset = single-domain) must be a DEFAULT here, not only
        # on the estimator: a directly-constructed model -- which is how a loader
        # rebuilds one from a saved VIResult -- gets no Params copied from a fit, and
        # `_transform` reads featuresCols unconditionally, so without this default
        # getOrDefault raises KeyError on a path that has nothing to do with
        # multi-domain (a single-domain loader never sets featuresCols).
        self._setDefault(featuresCol="features", featuresCols=[],
                         labelCol="frontier", nBg=nBg, tpn=tpn,
                         parent=parent, nodeAffinityCol="nodeAffinity",
                         caviMaxIter=100, caviTol=1e-3, gammaShape=100.0,
                         transformAlphaMode="fitted", transformAlpha=0.0,
                         transformBgWeight=0.5)

    @property
    def result(self):
        return self._result

    def _transform_omega(self, n_domains: int) -> np.ndarray:
        """Resolve the length-n_domains modality weight the fold-in must apply.

        Precedence: the `omega` Param when SET (an explicit deployment dial), else
        the fitted omega recorded in `result.metadata` by GatedOnlineLDA
        .get_metadata, else all 1.0 (unweighted raw volume = faithful MixEHR; Li,
        Nair, Lu et al. 2020, Nat. Commun.).

        The metadata fallback is what makes a model rebuilt from a LOADED VIResult
        serve the theta it was fitted with: omega never enters global_params (it
        weights theta during inference, not any stored parameter), so without it a
        loaded multi-domain model would silently fold in unweighted -- exactly the
        train/serve skew applying omega here exists to prevent.

        Validation goes through `domains.resolve_per_domain` -- the ONE per-domain
        hyperparameter validator, which exists because three hand-rolled copies had
        diverged -- with allow_zero=True because 0.0 is a legal omega ("drop this
        domain"). A hand-rolled LENGTH check would not be enough: this is now the
        only place a bad omega can enter (the engine validates at fit time, and an
        omega Param can be _set on a model afterwards), and a NEGATIVE omega drives
        gamma negative, whereupon digamma returns NaN and transform emits NaN
        nodeAffinity with no error raised at all. Sharing the validator also means
        the shim and the engine report the same caller error the same way.
        """
        if self.isSet("omega"):
            raw = self.getOrDefault("omega")
        else:
            recorded = (self._result.metadata or {}).get("omega")
            raw = np.ones(n_domains, dtype=np.float64) if recorded is None else recorded
        return resolve_per_domain(raw, n_domains, "omega", allow_zero=True)

    def _transform(self, dataset):
        from pyspark.ml.linalg import DenseVector, VectorUDT
        from pyspark.sql import functions as F
        from scipy.special import digamma
        from spark_vi.models.topic.lda import _cavi_doc_inference

        lay = _layout(self)
        lam = self._result.global_params["lambda"]
        if isinstance(lam, dict):
            # Per-domain dict lambda: each block normalizes over its OWN vocabulary
            # (the MixEHR per-modality model, where a token's domain is exogenous),
            # then the blocks concatenate into the engine's single id space. This is
            # GatedOnlineLDA._assemble_expElogbeta's arithmetic; it is repeated here
            # because _transform holds a VIResult, not a model instance.
            lam_blocks = [lam[m] for m in sorted(lam)]
            expElogbeta = np.concatenate(
                [np.exp(digamma(b) - digamma(b.sum(axis=1, keepdims=True)))
                 for b in lam_blocks], axis=1)
            sizes = [int(b.shape[1]) for b in lam_blocks]
            bounds = domains_to_bounds(sizes)
            omega = self._transform_omega(len(sizes))
        else:
            # Single-domain: unchanged. omega would have nothing to weight here, so
            # it is rejected rather than dropped -- the same call the engine's
            # _resolve_omega makes, at the only other place omega is consumed.
            if self.isSet("omega"):
                raise ValueError(
                    "omega requires multi-domain mode: the fitted lambda is a "
                    "single (K, V) array, so there are no domains to weight")
            expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
            sizes, bounds, omega = None, None, None
        # featuresCols set = the multi-domain INPUT shape: one vector column per
        # domain, concatenated by the same helper the fit's row mapper used (which
        # also rejects a row whose widths disagree with the fitted layout). Unset =
        # one already-concatenated features column, the single-domain path.
        fcols = list(self.getOrDefault("featuresCols") or [])
        if fcols and sizes is None:
            raise ValueError(
                f"featuresCols={fcols} is set but the fitted lambda is a single "
                f"(K, V) array, so the per-domain widths needed to concatenate "
                f"those columns are unknown; fit with featuresCols or transform a "
                f"single concatenated featuresCol")
        if fcols and len(fcols) != len(sizes):
            # Knowable on the DRIVER: _concat_domain_features would catch it too,
            # but only once per row inside a Spark task, as a py4j-wrapped error.
            raise ValueError(
                f"featuresCols has {len(fcols)} column(s) but the fitted lambda "
                f"has {len(sizes)} domain(s) of widths {sizes}; the counts must "
                f"match or a trailing domain would be silently dropped")
        # Deployment alpha (may differ from the fitted alpha — see _deployment_alpha):
        # decouples a learned fitting-aid alpha from the fold-in prior.
        alpha = _deployment_alpha(
            self._result.global_params["alpha"], lay,
            self.getOrDefault("transformAlphaMode"),
            float(self.getOrDefault("transformAlpha")),
            float(self.getOrDefault("transformBgWeight")))
        gamma_shape = float(self.getOrDefault("gammaShape"))
        cavi_max_iter = int(self.getOrDefault("caviMaxIter"))
        cavi_tol = float(self.getOrDefault("caviTol"))
        K = expElogbeta.shape[0]
        nodes = list(lay.nodes)
        blocks = {u: lay.block[u] for u in nodes}

        sc = dataset.sparkSession.sparkContext
        bcast = sc.broadcast({
            "expElogbeta": expElogbeta, "alpha": alpha, "gamma_shape": gamma_shape,
            "cavi_max_iter": cavi_max_iter, "cavi_tol": cavi_tol, "K": K,
            "nodes": nodes, "blocks": blocks,
            "sizes": (sizes if fcols else None), "bounds": bounds, "omega": omega,
        })

        def _affinity(*features):
            import hashlib
            p = bcast.value
            if p["sizes"] is None:
                doc = _vector_to_bow_document(features[0])
                indices, counts = doc.indices, doc.counts
            else:
                indices, counts = _concat_domain_features(features, p["sizes"])
            # Content-deterministic gamma_init (mirrors GatedOnlineLDA.local_update's
            # blake2b-of-content seeding): identical docs get identical init on every
            # run, so held-out node-affinity scores are reproducible and independent
            # of Spark partition/executor order. gammaShape is high (concentrated init
            # near 1.0) so this barely moves the CAVI fixed point — but a scoring path
            # feeding AUC / precision@sens must not be run-to-run random.
            h = hashlib.blake2b(digest_size=8)
            h.update(np.ascontiguousarray(indices, dtype=np.int32).tobytes())
            h.update(np.ascontiguousarray(counts, dtype=np.float64).tobytes())
            rng = np.random.default_rng(int.from_bytes(h.digest(), "little"))
            gamma_init = rng.gamma(p["gamma_shape"], 1.0 / p["gamma_shape"], size=p["K"])
            # omega weights theta, and theta is what this function returns, so the
            # deployed read-out must use the SAME per-token weight the fit used or
            # fitted and served theta diverge silently. None = the identity path,
            # byte-identical to the pre-omega single-domain call.
            w_tok = None
            if p["bounds"] is not None:
                dom = np.searchsorted(p["bounds"], indices, side="right") - 1
                w_tok = p["omega"][dom]
            gamma, _, _, _ = _cavi_doc_inference(
                indices=indices, counts=counts, expElogbeta=p["expElogbeta"],
                alpha=p["alpha"], gamma_init=gamma_init,
                max_iter=p["cavi_max_iter"], tol=p["cavi_tol"],
                gamma_count_weight=w_tok)
            theta = gamma / gamma.sum()
            return DenseVector([float(theta[p["blocks"][u]].sum()) for u in p["nodes"]])

        udf = F.udf(_affinity, returnType=VectorUDT())
        # Broadcast lifetime is the returned DataFrame's: its UDF closure holds
        # bcast, so do NOT eagerly unpersist (a lazy transform has no action to
        # unpersist after — see VIRunner.transform). ContextCleaner reclaims it
        # when the DataFrame is garbage-collected.
        out_col = self.getOrDefault("nodeAffinityCol")
        in_cols = ([F.col(c) for c in fcols] if fcols
                   else [F.col(self.getOrDefault("featuresCol"))])
        return dataset.withColumn(out_col, udf(*in_cols))

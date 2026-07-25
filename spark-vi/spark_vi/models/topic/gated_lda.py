"""GatedOnlineLDA: the SVI (variational) twin of the collapsed-Gibbs placement engine.

Overrides local_update, initialize_global, and update_global (the gated per-node α step);
compute_elbo and infer_local are ALSO overridden, but only branch away from the inherited
OnlineLDA behavior in multi-domain mode (see below) — with domains=None they delegate to
super() and are byte-identical to the base class.
  * local_update — restrict each training doc's CAVI to DagLayout.allowed_set(frontier)
    (the exact variational analogue of the Gibbs gate in dag_placement.fit_gated); sstats
    for disallowed topics stay zero, welding each node's topic to its subtree's documents.
    An empty frontier is a labeled background doc, gated to the background block only (NOT
    full-K) — the same convention as the Gibbs oracle and the gated STM.
  * initialize_global — dispatch on a pluggable init strategy (default "random").

Multi-domain (MixEHR-style; Li, Nair, Lu et al. 2020, Nat. Commun.), keyed on the `domains`
constructor arg (a sequence of per-domain vocab sizes summing to vocab_size): global_params
["lambda"] becomes a literal per-domain dict {m: (K, V_m)} instead of one (K, V) array, each
block independently row-normalized (`_assemble_expElogbeta`/`_split_to_domains`), each with
its own eta_m prior (`_eta_vocab_vector`). An optional per-domain MODALITY WEIGHT omega_m
(`_resolve_omega`, default None = all 1.0 = MixEHR-faithful raw volume) tempers domain m's
contribution to the shared theta and NOTHING else: gamma = alpha + Sigma_m omega_m
Sigma_{tokens in m} count * phi, while phi_norm, the lambda sufficient statistics and the
data log-likelihood all keep TRUE counts. omega is a tuned-vs-task pseudo-likelihood weight,
not a fitted parameter (see `_resolve_omega`), and under omega != 1 `compute_elbo` reports
the omega == 1 bound — a convergence diagnostic, not a bound on the weighted objective.
Each token's domain stays live at the point gamma/phi are formed (`_token_domains`, the v2
seam). Apart from that one per-token weight the gated CAVI never changes — it only ever
sees a concatenated (K, V) expElogbeta and per-doc concatenated indices — so local_update,
update_global, compute_elbo, and infer_local each branch on `self.domains is None` and their
domains=None arm is the original single-array code, unchanged. combine_stats and VIRunner
integration are untouched either way (sufficient stats stay a flat concatenated array).
TRAINING is gated (above); DEPLOYMENT is different — GatedLDAModel._transform folds held-out
docs in UNGATED full-K CAVI (the label is unknown at scoring time, which is the whole point)
-> theta -> node_affinity.

Validated against the collapsed-Gibbs oracle (dag_placement.fit_gated): placement (node-AUC by
depth, MRR, top2) matches at every depth; the DAG gate supplies the identifiability spectral
init provides in ungated LDA, so random init is the default (see the design spec's prototype
findings). References: Hoffman, Blei, Bach (2010) Online LDA; Griffiths & Steyvers (2004) the
oracle; the placement design docs/superpowers/specs/2026-07-15-gated-svi-placement-engine-design.md.
"""
from __future__ import annotations

import hashlib
from typing import Any, Iterable

import numpy as np
from scipy.special import digamma

from spark_vi.inference.concentration_optimization import gated_alpha_newton_step
from spark_vi.models.topic.dag_placement import DagLayout
from spark_vi.models.topic.lda import OnlineLDA, _cavi_doc_inference, _dirichlet_kl
from spark_vi.models.topic.types import GatedBOWDocument


def node_affinity(theta: np.ndarray, lay: DagLayout) -> dict[int, float]:
    """Per-node affinity from a full-K theta: the mass on each node's topic block.

    The SVI analogue of dag_placement.profile's per-node readout. The full dict IS the
    case-finding output; do not collapse to a single node."""
    return {u: float(theta[lay.block[u]].sum()) for u in lay.nodes}


class GatedOnlineLDA(OnlineLDA):
    def __init__(self, lay: DagLayout, vocab_size: int, *, init: str = "random",
                 optimize_alpha: bool = False,
                 frontier_histogram: dict | None = None,
                 domains: list[int] | None = None,
                 eta: float | Iterable[float] | None = None,
                 omega: float | Iterable[float] | None = None, **kw) -> None:
        # optimize_alpha is handled by the gated per-node Newton step (this class),
        # NOT OnlineLDA's full-K alpha_newton_step; pass it to the parent as False
        # so the inherited update_global never runs the vanilla alpha step.
        #
        # eta: OnlineLDA.__init__ requires a single valid positive scalar (it is
        # never optimized here — optimize_eta is disallowed below), so when eta is
        # a per-domain sequence we forward its mean as that placeholder scalar
        # (self.eta / global_params["eta"]; used only for iteration_summary-style
        # display in multi-domain mode). The per-domain prior actually consumed by
        # update_global/compute_elbo is `self._eta_domains` (set below, after
        # super().__init__ so a bare `eta=None` can default off self.eta).
        forward_eta = (
            float(np.mean(list(eta)))
            if (domains is not None and eta is not None and not np.isscalar(eta))
            else eta
        )
        super().__init__(K=lay.K, vocab_size=vocab_size, optimize_alpha=False,
                          eta=forward_eta, **kw)
        self.lay = lay
        self.init = init
        # Multi-domain (MixEHR-style, Li, Nair, Lu et al. 2020, Nat. Commun.) storage:
        # `domains` is a sequence of per-domain vocab sizes [V_0, V_1, ...] that MUST
        # sum to vocab_size. `domains=None` (default) keeps the single-domain (K, V)
        # array path byte-for-byte unchanged everywhere in this class. When set,
        # global_params["lambda"] becomes a literal per-domain dict {m: (K, V_m)};
        # `_domain_bounds` are the cumulative offsets [0, V_0, V_0+V_1, ...] used to
        # assemble/split a concatenated (K, V) view for the shared gated CAVI.
        if domains is not None:
            if sum(domains) != vocab_size:
                raise ValueError(
                    f"domains {list(domains)} sum to {sum(domains)}, "
                    f"expected vocab_size={vocab_size}"
                )
            self.domains = list(domains)
            self._domain_bounds = np.concatenate(
                ([0], np.cumsum(self.domains))).astype(np.int64)
            # Per-domain eta prior (MixEHR-style; Li 2020): a scalar broadcasts to
            # every domain (matches the domains=None default); a length-n_domains
            # sequence gives each domain its own concentration eta_m.
            if eta is None:
                self._eta_domains = [self.eta] * len(self.domains)
            elif np.isscalar(eta):
                self._eta_domains = [float(eta)] * len(self.domains)
            else:
                eta_seq = [float(e) for e in eta]
                if len(eta_seq) != len(self.domains):
                    raise ValueError(
                        f"eta sequence length {len(eta_seq)} != "
                        f"n_domains {len(self.domains)}"
                    )
                if any(e <= 0 for e in eta_seq):
                    raise ValueError(f"all eta components must be > 0, got {eta_seq}")
                self._eta_domains = eta_seq
        else:
            self.domains = None
            self._domain_bounds = None
            self._eta_domains = None
        # Per-domain modality weight omega (see _resolve_omega). None (the default)
        # = the unweighted MixEHR-faithful path, byte-identical to pre-omega code.
        self.omega = self._resolve_omega(omega)
        if getattr(self, "optimize_eta", False):
            # The gated update_global override optimizes only alpha; it never
            # touches eta. Fail fast rather than silently ignoring optimize_eta
            # (out of scope for the gated engine; the shim never enables it).
            raise NotImplementedError(
                "optimize_eta is not supported by GatedOnlineLDA (only the gated "
                "per-node alpha is learned; eta stays fixed)."
            )
        self.optimize_alpha = bool(optimize_alpha)          # gated flag (drives our override)
        if self.optimize_alpha and frontier_histogram is None:
            raise ValueError(
                "optimize_alpha=True requires frontier_histogram "
                "{frozenset(frontier): count} — the static allowed-set group structure."
            )
        self._frontier_histogram = frontier_histogram
        # Tied-alpha layout: index 0 = background, i = lay.nodes[i-1].
        self._block_sizes = np.array(
            [lay.n_bg] + [lay.tpn] * len(lay.nodes), dtype=np.float64)
        self._topic_to_tied = np.zeros(lay.K, dtype=np.int64)   # bg topics -> 0
        for i, u in enumerate(lay.nodes, start=1):
            for k in lay.block[u]:
                self._topic_to_tied[k] = i

    def initialize_global(self, data_summary: Any | None) -> dict[str, np.ndarray]:
        """Random Gamma lambda (default), or a pluggable init strategy's lambda.

        "random": inherited OnlineLDA Gamma init — the validated default (the gate already
        welds topics to nodes, so no symmetry-breaking seed is needed). Other strategies
        resolve from gated_init.INIT_STRATEGIES and need the training corpus in data_summary.
        An unknown name raises ValueError.

        Multi-domain (self.domains set, MixEHR-style, Li, Nair, Lu et al. 2020, Nat.
        Commun.): global_params["lambda"] is a per-domain dict {m: (K, V_m)} instead of
        a single (K, V) array; alpha/eta are unaffected (they live in the shared
        DAG-gated theta, not per-domain). "random" draws one Gamma(gamma_shape,
        1/gamma_shape) block per domain (see `_random_domain_lambda`, the per-domain
        analogue of OnlineLDA.initialize_global's single draw). domains=None keeps the
        single-array path below UNCHANGED (byte-identical)."""
        if self.domains is not None:
            if self.init == "random":
                lam = self._random_domain_lambda()
            elif data_summary is not None and "spectral_lambda" in data_summary:
                # Scalable/shim path (SP3): the per-domain dict lambda is precomputed
                # on the RDD and handed over (mirrors the single-domain spectral_lambda
                # handoff below); use it directly.
                sl = data_summary["spectral_lambda"]
                lam = {int(m): np.asarray(v, dtype=np.float64) for m, v in sl.items()}
            else:
                # Multi-domain spectral seed: block-aligned anchor recipe with the
                # per-domain candidate floor, split into per-domain lambda_m. Fixes the
                # random-init topic-death of insight 0066 (a node anchors on its
                # sparse-domain word, defining the topic across both domains via Q_01).
                from spark_vi.models.topic.gated_init import (
                    INIT_STRATEGIES, multidomain_spectral_lambda)
                if self.init not in INIT_STRATEGIES:
                    raise ValueError(
                        f"unknown init strategy {self.init!r}; "
                        f"known: {['random'] + sorted(INIT_STRATEGIES)}")
                scope = (data_summary or {}).get("anchor_scope", "closure")
                topo = (data_summary or {}).get("topo_order", "forward")
                lam = multidomain_spectral_lambda(
                    data_summary, self.lay, self.domains,
                    anchor_scope=scope, topo_order=topo)
            return {
                "lambda": lam,
                "alpha": self.alpha.copy(),         # defensive copy — runner mutates
                "eta": np.array(self.eta),          # 0-d ndarray for combine_stats type-uniformity
            }
        if self.init == "random":
            return super().initialize_global(data_summary)
        from spark_vi.models.topic.gated_init import INIT_STRATEGIES
        if self.init not in INIT_STRATEGIES:
            raise ValueError(
                f"unknown init strategy {self.init!r}; "
                f"known: {['random'] + sorted(INIT_STRATEGIES)}"
            )
        gp = super().initialize_global(data_summary)
        # Scalable path: the shim precomputed the (K,V) lambda on the RDD and
        # handed it over via data_summary (mirrors the STM shim's spectral_beta);
        # use it directly. Dense path: run the collect-to-driver strategy.
        if data_summary is not None and "spectral_lambda" in data_summary:
            gp["lambda"] = np.asarray(data_summary["spectral_lambda"], dtype=np.float64)
        else:
            scope = (data_summary or {}).get("anchor_scope", "closure")
            topo = (data_summary or {}).get("topo_order", "forward")
            gp["lambda"] = INIT_STRATEGIES[self.init](
                data_summary, self.lay, self.V, anchor_scope=scope, topo_order=topo)
        return gp

    def _random_domain_lambda(self) -> dict[int, np.ndarray]:
        """Per-domain Gamma(gamma_shape, 1/gamma_shape) draw, one (K, V_m) block per
        domain in domain order — the multi-domain analogue of
        OnlineLDA.initialize_global's single (K, V) draw (same distribution, same RNG
        seeding contract). Drawing the blocks sequentially from one RNG stream means a
        single-domain-shaped `domains=[V]` reproduces the domains=None draw exactly."""
        if self.random_seed is None:
            def draw(size):
                return np.random.gamma(
                    shape=self.gamma_shape, scale=1.0 / self.gamma_shape, size=size)
        else:
            rng = np.random.default_rng(self.random_seed)

            def draw(size):
                return rng.gamma(
                    shape=self.gamma_shape, scale=1.0 / self.gamma_shape, size=size)
        return {m: draw((self.lay.K, v)) for m, v in enumerate(self.domains)}

    def _assemble_expElogbeta(self, lam_dict: dict[int, np.ndarray]) -> np.ndarray:
        """Assemble the concatenated (K, V) expElogbeta the shared gated CAVI consumes
        from a per-domain lambda dict {m: (K, V_m)} (MixEHR-style storage; Li, Nair, Lu
        et al. 2020, Nat. Commun.).

        Each domain is normalized ON ITS OWN full row — exp(psi(lam_m) -
        psi(lam_m.sum(axis=1))) — then the blocks are concatenated in domain order.
        This is type-safe: a domain's rows can never be normalized against another
        domain's mass (the failure mode a single pooled array cannot prevent)."""
        blocks = [
            np.exp(digamma(lam_dict[m]) - digamma(lam_dict[m].sum(axis=1, keepdims=True)))
            for m in range(len(self.domains))
        ]
        return np.concatenate(blocks, axis=1)

    def _split_to_domains(self, concat: np.ndarray) -> dict[int, np.ndarray]:
        """Slice a concatenated (K, V) array into a per-domain dict {m: (K, V_m)} at
        self._domain_bounds — the inverse of `_assemble_expElogbeta`'s concatenation."""
        bounds = self._domain_bounds
        return {
            m: concat[:, bounds[m]:bounds[m + 1]]
            for m in range(len(self.domains))
        }

    def _eta_vocab_vector(self) -> np.ndarray:
        """Length-V Dirichlet-prior intercept vector consumed by update_global's
        natural-gradient target (eta_vec) and compute_elbo's global KL.

        domains=None: self.eta broadcast over the full vocabulary (the existing
        scalar-eta behavior — byte-identical to OnlineLDA's `eta + ...`, since a
        scalar and a length-V array filled with that scalar add identically).
        Multi-domain: each domain m's own `self._eta_domains[m]` (MixEHR-style
        per-domain concentration; Li, Nair, Lu et al. 2020, Nat. Commun.)
        broadcast over its own block, concatenated in domain order."""
        if self.domains is None:
            return np.full(self.V, self.eta, dtype=np.float64)
        out = np.empty(self.V, dtype=np.float64)
        bounds = self._domain_bounds
        for m in range(len(self.domains)):
            out[bounds[m]:bounds[m + 1]] = self._eta_domains[m]
        return out

    def _resolve_omega(self, omega) -> np.ndarray | None:
        """Validate the per-domain modality weight into a length-n_domains float
        array, or None for the unweighted default.

        omega_m is the PSEUDO-LIKELIHOOD (tempering) weight domain m's tokens carry
        in the doc-topic (gamma) accumulation: gamma = alpha + Sigma_m omega_m
        Sigma_{tokens in m} count * phi. It is NOT a fitted quantity and NOT part
        of the generative model: MixEHR (Li, Nair, Lu et al. 2020, Nat. Commun.)
        lets raw token volume speak for itself, i.e. omega == 1 for every modality,
        which is why None/1.0 is the default. omega is TUNED FOR A DOWNSTREAM TASK
        (de-biasing a domain whose token volume reflects utilization rather than
        signal), never read off the fit -- observed volume cannot identify it.

        Contract: None (all 1.0), a scalar (the same weight on every domain -- a
        global tempering of the doc-topic term), or a length-n_domains sequence of
        finite nonnegative weights (0.0 = drop that domain from theta entirely,
        while its lambda still trains on its true counts). Requires multi-domain
        mode: with domains=None there is no modality axis to weight, so a passed
        omega is a contradiction and raises rather than being silently ignored.

        The scalar/sequence dispatch is by the resolved array's ndim, NOT
        np.isscalar (which is False for a 0-d ndarray, so np.array(0.5) would take
        the sequence branch and fail opaquely); iterables are materialized first so
        a one-shot iterator is not consumed twice. The VALUE check runs before that
        dispatch, so neither branch can bypass it (a scalar -1.0 slipping through
        gives negative theta mass and a negative node_affinity score -- garbage
        rankings with no error raised).
        """
        if omega is None:
            return None
        if self.domains is None:
            raise ValueError(
                "omega requires multi-domain mode (domains=[V_0, V_1, ...]); a "
                "single-domain model has no modality to weight")
        n = len(self.domains)
        raw = (omega if isinstance(omega, np.ndarray)
               else list(omega) if hasattr(omega, "__iter__") else omega)
        arr = np.asarray(raw, dtype=np.float64)
        # Value check BEFORE the shape dispatch, so it cannot be short-circuited by
        # the scalar/0-d branch: a negative omega yields negative theta mass, which
        # node_affinity would sum into a negative node score and silently corrupt
        # the placement ranking rather than failing.
        if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
            raise ValueError(
                f"omega components must be finite and >= 0, got {arr.tolist()}")
        if arr.ndim == 0:
            return np.full(n, float(arr), dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != n:
            raise ValueError(
                f"omega must be a scalar or a length-{n} (n_domains) sequence, "
                f"got shape {arr.shape}")
        return arr

    def _token_domains(self, indices: np.ndarray) -> np.ndarray:
        """Per-token domain index for a document's concatenated vocabulary ids:
        the m with _domain_bounds[m] <= id < _domain_bounds[m+1], via searchsorted
        on the cumulative per-domain offsets. Multi-domain only (domains=None has
        no domain axis); ids must lie in [0, vocab_size).

        THIS IS THE v2 SEAM. Every per-token quantity that must know which
        modality a token came from resolves it here -- today the omega gamma-weight
        (`_gamma_count_weight`), tomorrow a v2 per-domain proportion pi_m or a
        per-domain theta-contribution instrument. Keeping the lookup a named unit
        (rather than inlining a domain-agnostic gather) is what keeps each token's
        domain live at the point gamma/phi are formed."""
        if self.domains is None:
            raise ValueError(
                "_token_domains requires multi-domain mode (domains=[V_0, ...])")
        return np.searchsorted(
            self._domain_bounds, np.asarray(indices), side="right") - 1

    def _gamma_count_weight(self, indices: np.ndarray) -> np.ndarray | None:
        """Per-token omega weight for the gamma recurrence: omega[_token_domains(w)].

        None when omega is unset -- `_cavi_doc_inference` then runs its original
        unweighted recurrence, so the default path is bit-for-bit unchanged."""
        if self.omega is None:
            return None
        return self.omega[self._token_domains(indices)]

    def local_update(
        self,
        rows: Iterable[GatedBOWDocument],
        global_params: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Gated E-step: per doc, CAVI over expElogbeta[allowed] with alpha[allowed];
        scatter sstats to lambda_stats[allowed, indices]. Disallowed topics get zero
        contribution — the variational twin of the Gibbs gate.

        allowed = lay.allowed_set(frontier) = background ∪ closure blocks of the frontier.
        An EMPTY frontier is a labeled background document (a known negative: this item
        belongs to no node), so it is gated to the BACKGROUND block only — NOT full-K.
        This matches both the collapsed-Gibbs oracle (dag_placement.fit_gated, which gates
        every training doc via allowed_set, so empty -> background-only) and the gated STM
        (partition.TopicBlockPartition.allowed_indices: a group with no foreground block
        contributes to the background only). Letting background docs go full-K instead
        lets the large background population train the node topics, collapsing them into
        generic comorbidity and destroying node specificity. Cost is O(|allowed|) per
        token = the doc's frontier closure (bounded by DAG depth), not K."""
        lam = global_params["lambda"]
        alpha = global_params["alpha"]
        if self.domains is None:
            expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
            lambda_stats = np.zeros_like(lam)
        else:
            # Multi-domain (MixEHR-style; Li, Nair, Lu et al. 2020, Nat. Commun.):
            # assemble the concatenated expElogbeta from the per-domain dict lambda.
            # Everything below is IDENTICAL either way — the gated CAVI only ever
            # sees a concatenated (K, V) expElogbeta and each doc's concatenated
            # token indices; it does not know domains exist. lambda_stats stays a
            # single concatenated (K, V) array (not a dict) so VIRunner's mini-batch
            # scaling / combine_stats / broadcast are untouched (see the plan's
            # "transient sufficient-stats stay concatenated" constraint).
            expElogbeta = self._assemble_expElogbeta(lam)
            lambda_stats = np.zeros((self.K, self.V), dtype=np.float64)

        doc_loglik_sum = 0.0
        doc_theta_kl_sum = 0.0
        n_docs = 0
        node_theta_sum = (
            np.zeros(self._block_sizes.shape[0], dtype=np.float64)
            if self.optimize_alpha else None)

        # gamma_init draws Gamma(gamma_shape, 1/gamma_shape) per doc, sized to the gated
        # allowed set — same content-deterministic seeding contract as OnlineLDA.local_update
        # (lda.py), so a distributed fit is reproducible regardless of Spark partition /
        # executor / iteration order. When random_seed is None, falls back to the global RNG.
        for doc in rows:
            # allowed_set(empty) == background block only, so a labeled background doc
            # (empty frontier) trains the background topics but NOT the node topics —
            # the same gate the Gibbs oracle and the gated STM apply. (Deployment /
            # transform is a separate, deliberately ungated full-K path; see
            # GatedLDAModel._transform in the mllib shim.)
            allowed = self.lay.allowed_set(doc.frontier)
            if self.random_seed is None:
                gamma_init = np.random.gamma(
                    shape=self.gamma_shape,
                    scale=1.0 / self.gamma_shape,
                    size=len(allowed),
                )
            else:
                h = hashlib.blake2b(digest_size=8)
                h.update(str(self.random_seed).encode("utf-8"))
                h.update(np.ascontiguousarray(doc.indices, dtype=np.int32).tobytes())
                h.update(np.ascontiguousarray(doc.counts, dtype=np.float64).tobytes())
                doc_seed = int.from_bytes(h.digest(), "little")
                doc_rng = np.random.default_rng(doc_seed)
                gamma_init = doc_rng.gamma(
                    shape=self.gamma_shape,
                    scale=1.0 / self.gamma_shape,
                    size=len(allowed),
                )
            gamma, expElogthetad, phi_norm, _ = _cavi_doc_inference(
                indices=doc.indices,
                counts=doc.counts,
                expElogbeta=expElogbeta[allowed],
                alpha=alpha[allowed],
                gamma_init=gamma_init,
                max_iter=self.cavi_max_iter,
                tol=self.cavi_tol,
                # Per-domain modality weight, theta-ONLY: scales this doc's gamma
                # accumulation by each token's omega_m and touches nothing else.
                # None (omega unset, incl. every domains=None model) = the
                # unweighted recurrence, bit-for-bit unchanged.
                gamma_count_weight=self._gamma_count_weight(doc.indices),
            )
            sstats_row = np.outer(expElogthetad, doc.counts / phi_norm)
            lambda_stats[np.ix_(allowed, doc.indices)] += sstats_row
            doc_loglik_sum += float(np.sum(doc.counts * np.log(phi_norm)))
            # Gated posterior KL: KL(q(theta_d) || p(theta_d)) restricted to the doc's
            # allowed sub-simplex (gamma and alpha[allowed] are both length-|allowed|),
            # the gated analogue of OnlineLDA.local_update's full-K _dirichlet_kl call
            # (Blei/Hoffman-style variational Dirichlet KL; see lda.py's _dirichlet_kl).
            # This only feeds compute_elbo's convergence bound below — it does NOT
            # touch lambda_stats, n_docs, or doc_loglik_sum, so the fit trajectory
            # (lambda update) and the SVI-vs-Gibbs equivalence gate are unaffected.
            doc_theta_kl_sum += _dirichlet_kl(gamma, alpha[allowed])
            n_docs += 1

            if node_theta_sum is not None:
                # Per-tied-block Σ_{k in block} (ψ(γ_k) − ψ(γ_sum)) over this doc's
                # allowed topics; γ is aligned with `allowed`. Blocks absent from
                # `allowed` contribute nothing (they stay at their prior).
                e_log_theta_d = digamma(gamma) - digamma(gamma.sum())
                np.add.at(node_theta_sum, self._topic_to_tied[allowed], e_log_theta_d)

        result = {
            "lambda_stats": lambda_stats,
            "doc_loglik_sum": np.array(doc_loglik_sum),
            "doc_theta_kl_sum": np.array(doc_theta_kl_sum),
            "n_docs": np.array(float(n_docs)),
        }
        if node_theta_sum is not None:
            result["e_log_theta_node_sum"] = node_theta_sum
        return result

    def update_global(self, global_params, target_stats, learning_rate):
        """SVI natural-gradient λ step (inherited form) + gated per-node α step.

        The λ update is the same natural-gradient step OnlineLDA.update_global
        computes; we recompute it here (a few lines) rather than toggling the
        parent's optimize flags, so the gated α path is explicit. η is never
        optimized in the gated engine.

        Multi-domain (MixEHR-style; Li, Nair, Lu et al. 2020, Nat. Commun.):
        `lam` is a per-domain dict {m: (K, V_m)}. The natural-gradient target is
        computed ONCE over the concatenated vocabulary (assembled expElogbeta *
        the concatenated lambda_stats sufficient statistic, plus the per-domain
        eta_vec intercept), then split back into a per-domain dict and blended
        per domain — same (1-rho)*old + rho*target form as the single-array path,
        just applied block-by-block.
        """
        lam = global_params["lambda"]
        alpha = global_params["alpha"]
        eta = global_params["eta"]
        if self.domains is None:
            expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
            target_lam = eta + expElogbeta * target_stats["lambda_stats"]
            new_lam = (1.0 - learning_rate) * lam + learning_rate * target_lam
        else:
            expElogbeta = self._assemble_expElogbeta(lam)
            eta_vec = self._eta_vocab_vector()          # per-domain eta_m, broadcast per block
            target = eta_vec + expElogbeta * target_stats["lambda_stats"]
            target_dict = self._split_to_domains(target)
            new_lam = {
                m: (1.0 - learning_rate) * lam[m] + learning_rate * target_dict[m]
                for m in range(len(self.domains))
            }
        new_alpha = alpha
        if self.optimize_alpha:
            new_alpha = self._gated_alpha_update(alpha, target_stats, learning_rate)
        return {"lambda": new_lam, "alpha": new_alpha, "eta": eta}

    def _gated_alpha_update(self, alpha_full, target_stats, learning_rate):
        """Contract α to tied space, take one damped gated Newton step, expand back.

        Blei-Ng-Jordan 2003 A.4.2 generalized to per-node tying + gated
        sub-simplices; see gated_alpha_newton_step. Floors α at 1e-3.
        """
        nodes = self.lay.nodes
        B = self._block_sizes.shape[0]
        # contract: one representative topic per tied block (tying keeps them equal)
        a_tied = np.empty(B, dtype=np.float64)
        a_tied[0] = alpha_full[0]                              # a background topic
        for i, u in enumerate(nodes, start=1):
            a_tied[i] = alpha_full[self.lay.block[u][0]]
        # static group structure from the frontier histogram
        groups = list(self._frontier_histogram.items())
        group_counts = np.array([c for _, c in groups], dtype=np.float64)
        memb = np.zeros((len(groups), B), dtype=bool)
        for g, (frontier, _) in enumerate(groups):
            for k in self.lay.allowed_set(frontier):
                memb[g, self._topic_to_tied[k]] = True
        delta = gated_alpha_newton_step(
            a_tied, self._block_sizes,
            target_stats["e_log_theta_node_sum"], group_counts, memb)
        a_tied_new = np.maximum(a_tied + learning_rate * delta, 1e-3)
        # expand back to length-K
        out = alpha_full.copy()
        out[: self.lay.n_bg] = a_tied_new[0]
        for i, u in enumerate(nodes, start=1):
            out[self.lay.block[u]] = a_tied_new[i]
        return out

    def compute_elbo(self, global_params, aggregated_stats):
        """ELBO = doc-data-likelihood − doc-level KL − global KL (same accounting
        as OnlineLDA.compute_elbo; see its docstring for the sign convention).
        domains=None delegates to the inherited single-array global KL, unchanged.

        Multi-domain (MixEHR-style; Li, Nair, Lu et al. 2020, Nat. Commun.): each
        topic's global KL decomposes into a SUM over domains of independent
        per-domain-block Dirichlet KLs — one KL(Dirichlet(lam_m[k]) ||
        Dirichlet(eta_m . 1_{V_m})) per (topic, domain), each block scored against
        its OWN eta_m prior — NOT a single KL over the naively concatenated
        vector. Dirichlet KL is not separable across a concatenation in general
        (gammaln of the FULL row-sum differs from the sum of gammaln of each
        block's own sum); this is exactly why `_assemble_expElogbeta` normalizes
        each domain block on its own row in the first place. Token-loglik and
        theta-KL are unchanged either way: local_update aggregates them as scalars
        over the concatenated vocabulary, independent of how lambda is stored.

        UNDER omega != 1 THIS IS THE omega == 1 BOUND, not an exact bound on the
        omega-weighted objective: the modality weight tempers q(theta_d)'s
        pseudo-likelihood (see `_resolve_omega`), and the terms aggregated here
        (token loglik from TRUE counts, Dirichlet KLs) are the unweighted ELBO's
        evaluated at the weighted fit's q. It remains finite and comparable across
        iterations of ONE fit -- a convergence DIAGNOSTIC -- but it is not
        comparable across different omega, and MONOTONICITY IS NOT GUARANTEED:
        nothing here establishes that the omega == 1 bound increases along a fit
        that optimizes the omega-weighted objective. That matters operationally,
        because VIModel.has_converged (core/model.py) stops on the TWO-SIDED rule
        abs(curr - prev) / max(abs(prev), 1e-12) < tol, which a non-monotone trace
        can satisfy at a turning point. Read it as a trace to inspect, not as a
        certificate. It is deliberately NOT rewritten to chase the weighted
        objective.
        """
        if self.domains is None:
            return super().compute_elbo(global_params, aggregated_stats)
        lam = global_params["lambda"]
        eta_vec = self._eta_vocab_vector()
        bounds = self._domain_bounds
        global_kl = 0.0
        for m in range(len(self.domains)):
            eta_vec_m = eta_vec[bounds[m]:bounds[m + 1]]
            lam_m = lam[m]
            for k in range(self.K):
                global_kl += _dirichlet_kl(lam_m[k], eta_vec_m)
        return float(
            float(aggregated_stats["doc_loglik_sum"])
            - float(aggregated_stats["doc_theta_kl_sum"])
            - global_kl
        )

    def infer_local(self, row, global_params):
        """Single-document E-step under fixed global params (deployment fold-in;
        UNGATED full-K — see the module docstring). domains=None delegates to the
        inherited single-array path, unchanged. Multi-domain (MixEHR-style; Li,
        Nair, Lu et al. 2020, Nat. Commun.) assembles the concatenated expElogbeta
        from the per-domain lambda dict first; the CAVI call is otherwise
        identical to OnlineLDA.infer_local."""
        if self.domains is None:
            return super().infer_local(row, global_params)
        lam = global_params["lambda"]
        alpha = global_params["alpha"]
        expElogbeta = self._assemble_expElogbeta(lam)
        gamma_init = np.random.gamma(
            shape=self.gamma_shape,
            scale=1.0 / self.gamma_shape,
            size=self.K,
        )
        gamma, _, _, _ = _cavi_doc_inference(
            indices=row.indices,
            counts=row.counts,
            expElogbeta=expElogbeta,
            alpha=alpha,
            gamma_init=gamma_init,
            max_iter=self.cavi_max_iter,
            tol=self.cavi_tol,
            # theta is the deployment READ-OUT, and omega is a weight ON theta, so
            # the fold-in applies it exactly as the training E-step does.
            gamma_count_weight=self._gamma_count_weight(row.indices),
        )
        theta = gamma / gamma.sum()
        return {"gamma": gamma, "theta": theta}

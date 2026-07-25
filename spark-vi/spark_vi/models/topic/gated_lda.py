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
seam), feeding both that weight and the per-domain THETA-CONTRIBUTION INSTRUMENT
(local_update's `theta_contribution_by_domain` stat, surfaced by `iteration_summary`): the
omega-weighted evidence mass each domain adds to gamma, which collapses to omega_m times
domain-m token volume EXACTLY (insight 0069) and is therefore an omega-application trace,
NOT the volume-imbalance diagnostic omega could be tuned against. Apart from that one
per-token weight the gated CAVI never changes — it only ever sees a concatenated (K, V)
expElogbeta and per-doc
concatenated indices — so local_update, update_global, compute_elbo, infer_local and
iteration_summary each branch on `self.domains is None` and their domains=None arm is the
original single-array code, unchanged. combine_stats and VIRunner integration are untouched
either way (sufficient stats stay flat arrays: the concatenated lambda_stats plus the
length-n_domains instrument, both correctly summed by the default elementwise combine).
v2 SEAM SCOPE: `_token_domains` keeps each token's modality live where gamma/phi are formed,
but `_cavi_doc_inference` itself still gathers phi domain-agnostically (`eb_d =
expElogbeta[:, indices]`). A v2 generative per-domain proportion pi contributes an
expElogpi_k[m_n] factor of shape (K, n_unique), which does NOT fit through the current
signature: v2 is a ONE-PARAMETER addition to `_cavi_doc_inference` plus passing
`self._expElogpi[:, tok_dom]` at each caller. Cheap, but not zero-plumbing — do not plan on
"no signature change".
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
from spark_vi.models.topic.domains import domains_to_bounds, resolve_per_domain
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
        # Materialize `domains` once, up front: len() below must not consume a
        # one-shot iterable that `self.domains = list(domains)` then re-reads.
        domains = None if domains is None else list(domains)
        # eta: OnlineLDA.__init__ requires a single valid positive scalar (it is
        # never optimized here — optimize_eta is disallowed below), so when eta is
        # a per-domain sequence we forward its mean as that placeholder scalar
        # (self.eta / global_params["eta"]; used only for iteration_summary-style
        # display in multi-domain mode). The per-domain prior actually consumed by
        # update_global/compute_elbo is `self._eta_domains` (resolved here when eta
        # was given, or after super().__init__ off self.eta when it was not).
        #
        # Resolution goes through the shared `domains.resolve_per_domain`, so a 0-d
        # ndarray eta (np.array(0.02)) is treated as the scalar it is instead of
        # raising "iteration over a 0-d array", and a per-domain eta with
        # domains=None raises a NAMED error here rather than an opaque TypeError
        # from OnlineLDA's `eta <= 0` comparison against a list.
        eta_domains = None
        if eta is not None:
            if domains is not None:
                eta_domains = resolve_per_domain(eta, len(domains), "eta")
                forward_eta = float(eta_domains.mean())
            else:
                materialized = (list(eta) if (not isinstance(eta, np.ndarray)
                                              and hasattr(eta, "__iter__")) else eta)
                if np.ndim(materialized) != 0:
                    raise ValueError(
                        "a per-domain eta sequence requires multi-domain mode "
                        "(domains=[V_0, V_1, ...]); with domains=None there is "
                        "one vocabulary and eta must be a positive scalar")
                forward_eta = eta
        else:
            forward_eta = None
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
            self._domain_bounds = domains_to_bounds(self.domains)
            # Per-domain eta prior (MixEHR-style; Li 2020): a scalar broadcasts to
            # every domain (matches the domains=None default); a length-n_domains
            # sequence gives each domain its own concentration eta_m. Resolved
            # above when eta was passed; a bare eta=None defaults off self.eta
            # (OnlineLDA's own validated scalar default), which is why this half
            # runs after super().__init__.
            self._eta_domains = (
                eta_domains if eta_domains is not None
                else resolve_per_domain(self.eta, len(self.domains), "eta"))
        else:
            self.domains = None
            self._domain_bounds = None
            self._eta_domains = None
        # Per-domain modality weight omega (see _resolve_omega). None (the default)
        # = the unweighted MixEHR-faithful path, byte-identical to pre-omega code.
        self.omega = self._resolve_omega(omega)
        # Last aggregated per-domain theta-contribution (the omega-application
        # trace, NOT a volume-imbalance diagnostic; see local_update's docstring
        # for what it is and is not). Stashed driver-side by update_global so
        # iteration_summary -- whose signature only carries global_params -- can
        # surface it. None until the first M-step; multi-domain only.
        self._theta_contribution_by_domain = None
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
        single-array path below UNCHANGED (byte-identical).

        The init NAME is validated FIRST in both modes, before the precomputed-lambda
        shortcut. It used to be checked only on the recipe branch, so multi-domain
        init="banana" with a precomputed spectral_lambda silently succeeded where
        single-domain raised. Dispatch is then through
        gated_init.MULTIDOMAIN_INIT_STRATEGIES, so a strategy with no multi-domain
        implementation raises NotImplementedError rather than quietly getting the
        spectral recipe run under its name."""
        if self.domains is not None:
            if self.init == "random":
                lam = self._random_domain_lambda()
            else:
                from spark_vi.models.topic.gated_init import (
                    INIT_STRATEGIES, MULTIDOMAIN_INIT_STRATEGIES)
                if self.init not in INIT_STRATEGIES:
                    raise ValueError(
                        f"unknown init strategy {self.init!r}; "
                        f"known: {['random'] + sorted(INIT_STRATEGIES)}")
                if data_summary is not None and "spectral_lambda" in data_summary:
                    # Scalable/shim path (SP3): the per-domain dict lambda is
                    # precomputed on the RDD and handed over (mirrors the
                    # single-domain spectral_lambda handoff below); use it directly.
                    sl = data_summary["spectral_lambda"]
                    lam = {int(m): np.asarray(v, dtype=np.float64)
                           for m, v in sl.items()}
                elif self.init not in MULTIDOMAIN_INIT_STRATEGIES:
                    raise NotImplementedError(
                        f"init strategy {self.init!r} has no multi-domain "
                        f"(per-domain dict lambda) implementation; known "
                        f"multi-domain strategies: "
                        f"{['random'] + sorted(MULTIDOMAIN_INIT_STRATEGIES)}")
                else:
                    # Multi-domain spectral seed: block-aligned anchor recipe with the
                    # per-domain candidate floor, split into per-domain lambda_m. Fixes
                    # the random-init topic-death of insight 0066 (a node anchors on its
                    # sparse-domain word, defining the topic across both domains via Q_01).
                    scope = (data_summary or {}).get("anchor_scope", "closure")
                    topo = (data_summary or {}).get("topo_order", "forward")
                    lam = MULTIDOMAIN_INIT_STRATEGIES[self.init](
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
        self._domain_bounds — the inverse of `_assemble_expElogbeta`'s concatenation.

        The blocks are VIEWS into `concat`, not copies: writing to a returned block
        writes through to the source array. Deliberate — both callers (update_global's
        natural-gradient blend, and the round-trip test) only READ the blocks and build
        fresh arrays from them, so copying would add an O(K·V) allocation per M-step for
        no behavioral gain. A future caller that needs to mutate a block must .copy()."""
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

        Resolution is the shared `domains.resolve_per_domain` (the single
        scalar-or-per-domain resolver, also used for eta_m and the Gibbs oracle's
        beta_prior_m), with allow_zero=True because 0.0 is a legal omega. Two
        properties it guarantees and this dial needs: the scalar/sequence dispatch
        is by the resolved array's ndim, NOT np.isscalar (which is False for a 0-d
        ndarray, so np.array(0.5) would take the sequence branch and fail
        opaquely), and the VALUE check runs BEFORE that dispatch so neither branch
        can bypass it (a scalar -1.0 slipping through gives negative theta mass and
        a negative node_affinity score -- garbage rankings with no error raised).
        """
        if omega is None:
            return None
        if self.domains is None:
            raise ValueError(
                "omega requires multi-domain mode (domains=[V_0, V_1, ...]); a "
                "single-domain model has no modality to weight")
        return resolve_per_domain(omega, len(self.domains), "omega",
                                  allow_zero=True)

    def _token_domains(self, indices: np.ndarray) -> np.ndarray:
        """Per-token domain index for a document's concatenated vocabulary ids:
        the m with _domain_bounds[m] <= id < _domain_bounds[m+1], via searchsorted
        on the cumulative per-domain offsets. Multi-domain only (domains=None has
        no domain axis); ids must lie in [0, vocab_size).

        THIS IS THE v2 SEAM. Every per-token quantity that must know which
        modality a token came from resolves it here -- today the omega gamma-weight
        (`_gamma_count_weight`) and the per-domain theta-contribution instrument
        (`local_update`), tomorrow a v2 per-domain proportion pi_m. Keeping the
        lookup a named unit (rather than inlining a domain-agnostic gather) is what
        keeps each token's domain live at the point gamma/phi are formed.

        Out-of-vocabulary ids raise ValueError NAMING the offending id: searchsorted
        SATURATES rather than failing (an id == vocab_size returns n_domains, one
        past the last domain), so without this check a bad id either surfaces
        downstream as an opaque "index out of bounds" from the omega gather, or --
        if the per-domain weight/stat arrays ever grew a sentinel entry -- silently
        resolves to the WRONG domain with no error at all. The check is two
        reductions over the doc's unique ids, negligible beside the per-doc CAVI."""
        if self.domains is None:
            raise ValueError(
                "_token_domains requires multi-domain mode (domains=[V_0, ...])")
        idx = np.asarray(indices)
        if idx.size:
            lo, hi = int(idx.min()), int(idx.max())
            if lo < 0 or hi >= self.V:
                bad = lo if lo < 0 else hi
                raise ValueError(
                    f"token id {bad} is outside the vocabulary range "
                    f"[0, {self.V}) spanned by domains {self.domains}")
        return np.searchsorted(self._domain_bounds, idx, side="right") - 1

    def _gamma_count_weight(self, indices: np.ndarray) -> np.ndarray | None:
        """Per-token omega weight for the gamma recurrence: omega[_token_domains(w)].

        None when omega is unset -- `_cavi_doc_inference` then runs its original
        unweighted recurrence, so the default path is bit-for-bit unchanged.
        Used by `infer_local` (deployment fold-in); `local_update` resolves the seam
        once per document instead, because it needs the token domains for the
        theta-contribution instrument as well and must not pay for two lookups."""
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
        token = the doc's frontier closure (bounded by DAG depth), not K.

        MULTI-DOMAIN ONLY, additionally emits `theta_contribution_by_domain`, a
        length-n_domains array: the total omega-weighted evidence mass each domain
        contributed to the doc-topic accumulation gamma over this batch. Definition,
        exactly. For one document the CAVI gamma recurrence is

            gamma = alpha + expElogtheta_d * (eb_d @ (w * counts / phi_norm))

        (`_cavi_doc_inference`; w = the per-token omega weight, 1 when omega is
        unset), so the evidence mass the tokens add to gamma OVER AND ABOVE the
        prior alpha is

            Sigma_k (gamma_k - alpha_k)
              = Sigma_n w_n * counts_n * (eb_d[:, n] . expElogtheta_d) / phi_norm_n
              = Sigma_n w_n * counts_n * (phi_norm_n - 1e-100) / phi_norm_n

        -- the second line is an identity, not an approximation, because phi_norm_n
        IS eb_d[:, n] . expElogtheta_d + 1e-100 by construction. So each token
        contributes exactly its own omega-weighted count, less a shave from the
        1e-100 underflow guard (below float64 resolution unless the token's topic
        mass has itself underflowed, in which case the token genuinely moves gamma
        by nothing). Grouping that per-token term by `_token_domains` and summing
        over the batch is this stat; summing it over domains recovers the batch's
        whole gamma increment. It is evaluated at the CONVERGED expElogtheta_d /
        phi_norm returned by the CAVI loop -- i.e. it is the increment of the next
        (fixed-point) sweep, the same convention the lambda sufficient statistics
        `outer(expElogthetad, counts / phi_norm)` already use.

        WHAT IT IS, AND IS NOT (insight 0069): this quantity collapses to
        omega_m * (domain-m token volume) EXACTLY -- for every omega, not just
        omega = 1 -- because the guard factor (phi_norm - 1e-100) / phi_norm is
        bit-exactly 1.0 in float64 away from the underflow floor. A partition of
        the gamma INCREMENT is a partition of the evidence, and the CAVI
        gamma-update conserves evidence mass across topics, so it cannot depend on
        how the mass was distributed. It is therefore NOT a posterior diagnostic
        and CANNOT show whether one modality dominates the shared theta, so it
        cannot inform omega tuning (a per-domain token count is its free
        equivalent). What it IS: an exact trace that omega was applied, to which
        domains, in what proportion -- the cheapest regression guard on "omega
        weights theta" (exactly 0.25x when omega_m is, exactly 0 at omega_m = 0).
        Measuring domination needs a quantity sensitive to WHERE the mass landed
        (each domain's marginal contribution to fitted theta, or a
        leave-one-domain-out refit). Nonnegative by construction. The stat is
        additive across partitions, so the default `combine_stats` elementwise sum
        aggregates it with no override. domains=None emits no such key (the
        single-domain stats dict is unchanged)."""
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
        # Per-domain theta-contribution instrument (see the docstring). Emitted for
        # every multi-domain batch, empty or not, so the stats dict's key set does
        # not depend on partition contents.
        theta_contribution = (
            None if self.domains is None
            else np.zeros(len(self.domains), dtype=np.float64))

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
            # v2 SEAM, resolved ONCE per doc and shared by both per-token consumers:
            # the omega gamma-weight and the per-domain theta-contribution below.
            # domains=None has no domain axis, and omega then cannot be set either
            # (`_resolve_omega` rejects it), so both stay None and the recurrence is
            # the original unweighted one, bit-for-bit.
            if self.domains is None:
                tok_dom = None
                w_tok = None
            else:
                tok_dom = self._token_domains(doc.indices)
                w_tok = None if self.omega is None else self.omega[tok_dom]
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
                gamma_count_weight=w_tok,
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

            if theta_contribution is not None:
                # Per-token share of Sigma_k (gamma_k - alpha_k) at the converged
                # (expElogthetad, phi_norm): w_n * counts_n * (phi_norm_n - 1e-100)
                # / phi_norm_n, grouped by the token's domain. The guard factor is
                # the exact algebraic residue of _cavi_doc_inference's
                # `phi_norm = eb_d.T @ expElogthetad + 1e-100` -- see the docstring.
                # Nonnegative termwise (counts, omega and phi_norm - 1e-100 all are),
                # so the accumulated stat is nonnegative too.
                share = doc.counts * ((phi_norm - 1e-100) / phi_norm)
                if w_tok is not None:
                    share = share * w_tok
                theta_contribution += np.bincount(
                    tok_dom, weights=share, minlength=len(self.domains))

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
        if theta_contribution is not None:
            result["theta_contribution_by_domain"] = theta_contribution
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
        just applied block-by-block. The aggregated per-domain theta-contribution
        instrument (see local_update) is stashed here for `iteration_summary`: this
        is the driver-side hook that sees the combined stats, and
        iteration_summary's signature carries only global_params. It is read-only
        bookkeeping -- nothing in the fit consumes it.
        """
        lam = global_params["lambda"]
        alpha = global_params["alpha"]
        eta = global_params["eta"]
        if self.domains is not None and "theta_contribution_by_domain" in target_stats:
            # Post-scaling (VIRunner hands update_global the corpus-equivalent
            # target_stats), so in mini-batch mode this reads as a whole-corpus
            # contribution rather than one batch's.
            self._theta_contribution_by_domain = np.asarray(
                target_stats["theta_contribution_by_domain"], dtype=np.float64)
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

    def iteration_summary(self, global_params: dict[str, np.ndarray]) -> str:
        """Per-iter α / η / Σλ view, dict-λ aware, plus the θ-contribution read.

        domains=None delegates to OnlineLDA.iteration_summary and is byte-identical
        to it. Multi-domain REQUIRES an override: the inherited implementation does
        `float(global_params["eta"])` and `lam.sum(axis=1)`, and neither works on a
        per-domain η sequence or a per-domain dict λ -- so VIRunner (core/runner.py,
        which calls this every iteration) raised outright on any multi-domain fit
        driven through it. Every multi-domain test on this branch drove its fits
        locally, which is why it stayed hidden; the multi-domain path is now tested.

        What the multi-domain line adds over the single-domain one:
          * η_m -- each domain's own Dirichlet concentration (MixEHR-style
            per-domain prior; Li, Nair, Lu et al. 2020, Nat. Commun.), since there
            is no longer a single scalar η to print.
          * Σλ_k PER DOMAIN -- λ row mass spread within each block. Pooling the
            blocks would hide exactly the failure this diagnostic is for: one
            domain's topics diverging in mass while the other's stay flat.
          * θ_contrib_m -- the aggregated per-domain θ-contribution from the last
            M-step (see local_update for the exact definition) with each domain's
            share of the total. Read it as an ω-APPLICATION TRACE, not as a
            volume-imbalance diagnostic: it equals ω_m × (domain-m token volume)
            exactly, for every ω (insight 0069), so it confirms the dial reached the
            γ accumulation but says nothing about whether a modality dominates the
            fitted θ. Omitted before the first M-step, when no batch has been
            aggregated yet.
        """
        if self.domains is None:
            return super().iteration_summary(global_params)
        alpha = np.asarray(global_params["alpha"])
        lam = global_params["lambda"]
        eta_str = ", ".join(f"{e:.4g}" for e in self._eta_domains)
        lam_parts = []
        for m in range(len(self.domains)):
            rs = lam[m].sum(axis=1)
            lam_parts.append(f"Σλ_k[m{m}: min={rs.min():.3g} max={rs.max():.3g}]")
        out = (
            f"α[min={alpha.min():.4g} max={alpha.max():.4g} mean={alpha.mean():.4g}], "
            f"η_m=[{eta_str}], "
            + " ".join(lam_parts)
        )
        contrib = self._theta_contribution_by_domain
        if contrib is not None:
            total = float(contrib.sum())
            frac = contrib / total if total > 0.0 else np.zeros_like(contrib)
            out += (
                f", θ_contrib_m=[{', '.join(f'{c:.4g}' for c in contrib)}]"
                f" frac=[{', '.join(f'{f:.3g}' for f in frac)}]"
            )
        return out

    def get_metadata(self) -> dict[str, Any]:
        """Shape constants plus, in multi-domain mode, the constants needed to
        RECONSTRUCT a fitted model from a saved VIResult.

        `domains` fixes the per-domain vocabulary widths that slice the
        concatenated id space; `eta_m` and `omega` are not recoverable from
        global_params -- in multi-domain mode global_params["eta"] is only a
        scalar-mean placeholder and omega never enters global_params at all
        (it weights theta during inference, not any stored parameter). Without
        these three a saved multi-domain result cannot be interpreted, let
        alone served. All values are plain Python types: `metadata` is written
        verbatim into manifest.json.

        eta provenance: multi-domain update_global/compute_elbo read eta from
        `self._eta_domains`, not from global_params, which is sound only because
        `optimize_eta` is rejected in __init__ so eta cannot change during a
        fit. `test_optimize_eta_rejected_pins_the_eta_provenance_invariant`
        pins that.
        """
        md = super().get_metadata()
        if self.domains is not None:
            md["domains"] = [int(v) for v in self.domains]
            md["eta_m"] = [float(x) for x in self._eta_domains]
            md["omega"] = [float(x) for x in self.omega]
        return md

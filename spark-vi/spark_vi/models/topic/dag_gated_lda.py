"""The DAG-gated LDA engine: LDA whose topics are laid out in blocks over a label DAG.

THE MODEL. `DagLayout` gives every non-root DAG node a block of `tpn` topics, plus a
shared background block. A TRAINING document may use only the background and the
blocks of the nodes in the closure of its frontier (its most-specific labels and all
their ancestors): its CAVI runs over that allowed set, and every other topic gets zero
sufficient statistics from it. So no label enters the likelihood — the supervision is
which topics a document may use, and it welds each node's block to the documents in
its subtree. A document with an empty frontier is a labeled background document and
may use the background block only (not all K). DEPLOYMENT is different: the MLlib
model's transform folds a held-out document in UNGATED over all K topics (its label is
unknown at scoring time, which is the point) -> theta.

WHAT THIS CLASS CHANGES in OnlineLDA:
  * local_update — the gated CAVI above (and the leaves-first schedule's height cap,
    `set_gate_schedule`).
  * initialize_global — the random Gamma draw (per domain when multi-domain), plus the
    leaves-first schedule's seeds when set.
  * update_global — optional per-node tied empirical-Bayes alpha (`set_alpha_policy`,
    `gated_alpha_newton_step`), and the schedule's frozen / newly opened blocks.
  * compute_elbo, infer_local, iteration_summary — differ only in multi-domain mode.

Multi-domain (MixEHR-style; Li, Nair, Lu et al. 2020, Nat. Commun.), keyed on the `domains`
constructor arg (a sequence of per-domain vocab sizes summing to vocab_size): global_params
["lambda"] becomes a per-domain dict {m: (K, V_m)} instead of one (K, V) array, each block
row-normalized on its own under the one scalar eta prior. The gated CAVI never changes —
it sees one concatenated (K, V) expElogbeta and per-doc concatenated indices — so the
sufficient statistics stay one concatenated array (combine_stats and the runners are
untouched).

VALIDATED against a collapsed-Gibbs sampler of the same gated model (the test-suite
oracle `_gibbs_oracle.fit_gated`): placement (node AUC by depth, MRR, top-2) matches at
every depth. The gate supplies the identifiability a spectral init provides in ungated
LDA, so lambda starts from the random draw (plus the optional gate seeds). References:
Hoffman, Blei, Bach (2010) Online LDA; Griffiths & Steyvers (2004);
docs/superpowers/specs/2026-07-15-gated-svi-placement-engine-design.md.
"""
from __future__ import annotations

import hashlib
from typing import Any, Iterable

import numpy as np
from scipy.special import digamma

from spark_vi.inference.concentration_optimization import gated_alpha_newton_step
from spark_vi.dag import DagLayout
from spark_vi.models.topic.domains import domains_to_bounds
from spark_vi.models.topic.lda import OnlineLDA, _cavi_doc_inference, _dirichlet_kl
from spark_vi.models.topic.types import GatedBOWDocument


def expElogbeta_from_lambda(lam) -> np.ndarray:
    """exp(E[log beta]) as ONE concatenated (K, V) array, from a (K, V) lambda or a
    per-domain dict {m: (K, V_m)}.

    Each domain is normalized ON ITS OWN full row — exp(psi(lam_m) -
    psi(lam_m.sum(axis=1))) — then the blocks are concatenated in domain order, so a
    domain's rows can never be normalized against another domain's mass (the failure
    mode a single pooled array cannot prevent; MixEHR-style storage, Li, Nair, Lu et
    al. 2020, Nat. Commun.). The one helper the engine, the MLlib transform and the
    distributed readout all use to read a fitted lambda."""
    if isinstance(lam, dict):
        return np.concatenate(
            [np.exp(digamma(lam[m]) - digamma(lam[m].sum(axis=1, keepdims=True)))
             for m in sorted(lam)], axis=1)
    return np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))


class DagGatedOnlineLDA(OnlineLDA):
    def __init__(self, lay: DagLayout, vocab_size: int, *,
                 optimize_alpha: bool = False,
                 frontier_histogram: dict | None = None,
                 domains: list[int] | None = None, **kw) -> None:
        # optimize_alpha is handled by the gated per-node Newton step (this class),
        # NOT OnlineLDA's full-K alpha_newton_step; pass it to the parent as False
        # so the inherited update_global never runs the vanilla alpha step.
        #
        # Materialize `domains` once, up front: len() below must not consume a
        # one-shot iterable that `self.domains = list(domains)` then re-reads.
        domains = None if domains is None else list(domains)
        super().__init__(K=lay.K, vocab_size=vocab_size, optimize_alpha=False, **kw)
        self.lay = lay
        # Multi-domain (MixEHR-style, Li, Nair, Lu et al. 2020, Nat. Commun.) storage:
        # `domains` is a sequence of per-domain vocab sizes [V_0, V_1, ...] that MUST
        # sum to vocab_size. `domains=None` (default) keeps the single-domain (K, V)
        # array path. When set, global_params["lambda"] becomes a literal per-domain
        # dict {m: (K, V_m)}; `_domain_bounds` are the cumulative offsets
        # [0, V_0, V_0+V_1, ...] used to assemble/split a concatenated (K, V) view for
        # the shared gated CAVI. One scalar eta is the prior on every domain's block.
        if domains is not None:
            if sum(domains) != vocab_size:
                raise ValueError(
                    f"domains {list(domains)} sum to {sum(domains)}, "
                    f"expected vocab_size={vocab_size}"
                )
            self.domains = list(domains)
            self._domain_bounds = domains_to_bounds(self.domains)
        else:
            self.domains = None
            self._domain_bounds = None
        if getattr(self, "optimize_eta", False):
            # The gated update_global override optimizes only alpha; it never
            # touches eta. Fail fast rather than silently ignoring optimize_eta.
            raise NotImplementedError(
                "optimize_eta is not supported by the gated engine (only the gated "
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
        # Leaves-first gate schedule (off by default; see set_gate_schedule).
        self.gate_warm = 0
        self._gate_H = 0
        self._topic_height = None
        self._open_seed = None
        self._init_seed = None

    def set_gate_schedule(self, warm: int, open_seed: np.ndarray | None = None,
                          init_seed: np.ndarray | None = None) -> None:
        """Fit in ROUNDS by node height: leaves first, then each level up.

        Exp 0152 read 3b: in a single-round gated fit a parent block can hold its
        dominant child's vocabulary while a minor child fills with comorbidity. The
        schedule lets every child claim its own words before its parent's block
        exists: for the first `warm` iterations a document's gate admits only
        closure nodes of height 0 (leaves), the next `warm` height <= 1, and so on;
        after `warm * max_height` iterations the gate is the production gate. A
        node's height is strictly above its children's, so on a multi-parent DAG
        every child opens before any parent.

        A CLOSED block's lambda is frozen (its zero sufficient statistics would pull
        it to the flat prior every step). A block opened late onto already-sharp
        competitors starts flat, wins no responsibility and decays — dead on arrival
        (report 2026-10-19-leaves-first-gate-sim) — so `open_seed`, a (K, V) array
        over the concatenated vocabulary, gives each block a start the moment it
        opens: its row is the word counts of the documents whose gate admits the
        block (`gated_init.gate_open_seed_counts`), normalized and scaled to a typical
        open node topic's mass. The seed is the cluster MEAN, not an extreme word:
        competition with the sharp children and background strips what they
        already explain, leaving the parent its shared stratum.

        `init_seed` (K, V), optional: rows applied at iteration 0 to the blocks that
        are open from the start (the leaves), e.g. an external phenotype profile
        (HPO) where one exists. In either seed a row with no mass in a domain leaves
        that domain's lambda as it was, so a profile that covers only the primary
        domain replaces only that domain's start.

        The state rides global_params (`gate_iter`, `gate_open_at`), so executors see
        the round every iteration; the seeds are driver-only (`__getstate__` drops
        them). `warm=0` (the default) is the production gate, byte-identical."""
        self.gate_warm = int(warm)
        self._init_seed = None
        if self.gate_warm <= 0:
            self.gate_warm, self._gate_H, self._topic_height, self._open_seed = 0, 0, None, None
            return
        self._topic_height = np.full(self.K, -1, dtype=np.int64)   # bg: always open
        for u in self.lay.nodes:
            self._topic_height[self.lay.block[u]] = self.lay.height(u)
        self._gate_H = int(self._topic_height.max()) if self.lay.nodes else 0
        if open_seed is not None:
            open_seed = np.asarray(open_seed, dtype=np.float64)
            if open_seed.shape != (self.K, self.V):
                raise ValueError(f"open_seed must be (K, V) = ({self.K}, {self.V}), "
                                 f"got {open_seed.shape}")
        self._open_seed = open_seed
        if init_seed is not None:
            init_seed = np.asarray(init_seed, dtype=np.float64)
            if init_seed.shape != (self.K, self.V):
                raise ValueError(f"init_seed must be (K, V) = ({self.K}, {self.V}), "
                                 f"got {init_seed.shape}")
        self._init_seed = init_seed

    def _gate_level(self, t: int):
        """Max open height at iteration t, or None (everything open)."""
        if self.gate_warm <= 0 or t >= self.gate_warm * self._gate_H:
            return None
        return t // self.gate_warm

    def _gate_open_mask(self, level) -> np.ndarray:
        if level is None:
            return np.ones(self.K, dtype=bool)
        return self._topic_height <= level

    def initialize_global(self, data_summary: Any | None) -> dict[str, np.ndarray]:
        gp = self._initialize_global_base(data_summary)
        if self.gate_warm > 0:
            gp["gate_iter"] = np.array(0, dtype=np.int64)
            open0 = self._gate_open_mask(self._gate_level(0))
            gp["gate_open_at"] = np.where(open0, 0, -1).astype(np.int64)
            if self._init_seed is not None:
                rows = open0.copy()
                rows[:self.lay.n_bg] = False
                gp["lambda"] = self._seed_rows(gp["lambda"], rows, self._init_seed,
                                               ref=rows, t=-1)
        return gp

    def _seed_rows(self, lam, rows, seed, *, ref, t):
        """Replace `rows` of lambda (array or per-domain dict) with `seed`, per domain,
        at the mean row mass of the `ref` rows. A seed row with no mass in a domain
        leaves that domain's row unchanged. A deterministic multiplicative jitter
        breaks ties between the topics of one block (tpn > 1)."""
        def one(lam_m, seed_m, eta_m):
            lam_m = np.array(lam_m, dtype=np.float64, copy=True)
            idx = np.flatnonzero(rows)
            if idx.size == 0 or seed_m is None:
                return lam_m
            sd = seed_m[idx]
            tot = sd.sum(axis=1, keepdims=True)
            has = tot[:, 0] > 0
            if not has.any():
                return lam_m
            r = ref if ref.any() else np.ones(len(ref), bool)
            mass = float(lam_m[r].sum(axis=1).mean())
            rng = np.random.default_rng(
                (int(self.random_seed or 0) * 1_000_003 + t + 7) % (2 ** 32))
            jit = rng.gamma(100.0, 1.0 / 100.0, size=sd.shape)
            new = eta_m + mass * (sd / np.where(tot > 0, tot, 1.0)) * jit
            lam_m[idx[has]] = new[has]
            return lam_m

        eta = float(self.eta)
        if self.domains is None:
            return one(lam, seed, eta)
        b = self._domain_bounds
        return {m: one(lam[m], None if seed is None else seed[:, b[m]:b[m + 1]], eta)
                for m in range(len(self.domains))}

    def _initialize_global_base(self, data_summary: Any | None) -> dict[str, np.ndarray]:
        """Random Gamma lambda: the inherited OnlineLDA draw, or one block per domain.

        The gate welds topics to nodes, so no symmetry-breaking seed is needed; the
        data-driven gate seeds (`set_gate_schedule`) are applied on top. Multi-domain
        (self.domains set, MixEHR-style, Li, Nair, Lu et al. 2020, Nat. Commun.):
        global_params["lambda"] is a per-domain dict {m: (K, V_m)} drawn by
        `_random_domain_lambda`; alpha/eta are unaffected (they live in the shared
        DAG-gated theta, not per-domain). domains=None is OnlineLDA's own draw."""
        if self.domains is None:
            return super().initialize_global(data_summary)
        return {
            "lambda": self._random_domain_lambda(),
            "alpha": self.alpha.copy(),         # defensive copy — runner mutates
            "eta": np.array(self.eta),          # 0-d ndarray for combine_stats type-uniformity
        }

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
        """The concatenated (K, V) expElogbeta the shared gated CAVI consumes, from a
        per-domain lambda dict {m: (K, V_m)} (see `expElogbeta_from_lambda`)."""
        return expElogbeta_from_lambda({m: lam_dict[m] for m in range(len(self.domains))})

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

    def __getstate__(self):
        """Pickle WITHOUT the driver-only gate seeds (ADR 0047's closure clause).

        The model object rides every task closure (VIRunner's ``_model=model``
        default-arg capture), and ADR 0047 forbids array-shaped payloads in task
        closures. The (K, V) open-time and init seeds are read only by the driver's
        own instance (`initialize_global`, `_gate_step`), so every pickled copy
        carries them as None."""
        state = dict(self.__dict__)
        state["_open_seed"] = None
        state["_init_seed"] = None
        return state

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
        This matches both the collapsed-Gibbs oracle (the test-suite oracle _gibbs_oracle.fit_gated, which gates
        every training doc via allowed_set, so empty -> background-only) and the gated STM
        (partition.TopicBlockPartition.allowed_indices: a group with no foreground block
        contributes to the background only). Letting background docs go full-K instead
        lets the large background population train the node topics, collapsing them into
        generic comorbidity and destroying node specificity. Cost is O(|allowed|) per
        token = the doc's frontier closure (bounded by DAG depth), not K."""
        lam = global_params["lambda"]
        alpha = global_params["alpha"]
        # Leaves-first schedule round (None = the production gate).
        gate_level = (self._gate_level(int(global_params["gate_iter"]))
                      if "gate_iter" in global_params else None)
        # Multi-domain: the gated CAVI only ever sees a concatenated (K, V)
        # expElogbeta and each doc's concatenated token indices; lambda_stats stays a
        # single concatenated (K, V) array so VIRunner's mini-batch scaling /
        # combine_stats / broadcast are untouched.
        expElogbeta = expElogbeta_from_lambda(lam)
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
            # (empty frontier) trains the background topics but NOT the node topics.
            # (Deployment / transform is a separate, deliberately ungated full-K path.)
            allowed = self.lay.allowed_set(doc.frontier, max_height=gate_level)
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
            )
            sstats_row = np.outer(expElogthetad, doc.counts / phi_norm)
            lambda_stats[np.ix_(allowed, doc.indices)] += sstats_row
            doc_loglik_sum += float(np.sum(doc.counts * np.log(phi_norm)))
            # Gated posterior KL: KL(q(theta_d) || p(theta_d)) restricted to the doc's
            # allowed sub-simplex (gamma and alpha[allowed] are both length-|allowed|),
            # the gated analogue of OnlineLDA.local_update's full-K _dirichlet_kl call.
            # It only feeds compute_elbo's convergence bound — it does not touch the
            # lambda statistics, so the fit trajectory is unaffected.
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
        the concatenated lambda_stats sufficient statistic, plus the eta
        intercept), then split back into a per-domain dict and blended per domain
        — same (1-rho)*old + rho*target form as the single-array path.
        """
        lam = global_params["lambda"]
        alpha = global_params["alpha"]
        eta = global_params["eta"]
        if self.domains is None:
            expElogbeta = expElogbeta_from_lambda(lam)
            target_lam = eta + expElogbeta * target_stats["lambda_stats"]
            new_lam = (1.0 - learning_rate) * lam + learning_rate * target_lam
        else:
            expElogbeta = self._assemble_expElogbeta(lam)
            target = float(self.eta) + expElogbeta * target_stats["lambda_stats"]
            target_dict = self._split_to_domains(target)
            new_lam = {
                m: (1.0 - learning_rate) * lam[m] + learning_rate * target_dict[m]
                for m in range(len(self.domains))
            }
        new_alpha = alpha
        if self.optimize_alpha:
            new_alpha = self._gated_alpha_update(alpha, target_stats, learning_rate)
        out = {"lambda": new_lam, "alpha": new_alpha, "eta": eta}
        if "gate_iter" in global_params:
            out.update(self._gate_step(global_params, new_lam))
        return out

    def _gate_step(self, global_params, new_lam):
        """Freeze closed blocks, advance the round, seed the blocks that open next
        (see set_gate_schedule). Returns the updated lambda + gate keys."""
        t = int(global_params["gate_iter"])
        open_at = np.asarray(global_params["gate_open_at"], dtype=np.int64).copy()
        closed = open_at < 0
        old = global_params["lambda"]
        nxt = self._gate_open_mask(self._gate_level(t + 1))
        newly = nxt & closed
        open_at[newly] = t + 1
        node_open = (~closed).copy()
        node_open[:self.lay.n_bg] = False
        # Freeze closed blocks, then seed the ones that open next iteration.
        if self.domains is None:
            lam = np.array(new_lam, dtype=np.float64, copy=True)
            lam[closed] = old[closed]
        else:
            lam = {}
            for m in range(len(self.domains)):
                lam[m] = np.array(new_lam[m], dtype=np.float64, copy=True)
                lam[m][closed] = old[m][closed]
        if self._open_seed is not None and newly.any():
            lam = self._seed_rows(lam, newly, self._open_seed,
                                  ref=(node_open if node_open.any() else ~closed), t=t)
        return {"lambda": lam, "gate_iter": np.array(t + 1, dtype=np.int64),
                "gate_open_at": open_at}

    def set_alpha_policy(self, frontier_histogram, *, optimize):
        """Attach the static frontier histogram and turn the per-node tied
        empirical-Bayes alpha (the gated Newton step, insight 0059) on or off after
        construction — the seam the MLlib estimator uses, because the histogram
        comes from the persisted document RDD, which exists only after the engine
        has been built. Call before the fit starts. Idempotent."""
        hist = {frozenset(int(x) for x in fr): int(n)
                for fr, n in dict(frontier_histogram).items()}
        self._frontier_histogram = hist
        self.optimize_alpha = bool(optimize)
        return self

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
        Dirichlet(eta . 1_{V_m})) per (topic, domain) — NOT a single KL over the
        naively concatenated vector. Dirichlet KL is not separable across a
        concatenation in general (gammaln of the FULL row-sum differs from the sum
        of gammaln of each block's own sum); this is exactly why
        `_assemble_expElogbeta` normalizes each domain block on its own row in the
        first place. Token-loglik and theta-KL are unchanged either way.
        """
        if self.domains is None:
            return super().compute_elbo(global_params, aggregated_stats)
        lam = global_params["lambda"]
        global_kl = 0.0
        for m, v_m in enumerate(self.domains):
            prior = np.full(int(v_m), float(self.eta), dtype=np.float64)
            for k in range(self.K):
                global_kl += _dirichlet_kl(lam[m][k], prior)
        return float(
            float(aggregated_stats["doc_loglik_sum"])
            - float(aggregated_stats["doc_theta_kl_sum"])
            - global_kl
        )

    def infer_local(self, row, global_params):
        """Single-document E-step under fixed global params (deployment fold-in;
        UNGATED full-K — see the module docstring). domains=None delegates to the
        inherited single-array path, unchanged. Multi-domain assembles the
        concatenated expElogbeta from the per-domain lambda dict first; the CAVI
        call is otherwise identical to OnlineLDA.infer_local."""
        if self.domains is None:
            return super().infer_local(row, global_params)
        expElogbeta = self._assemble_expElogbeta(global_params["lambda"])
        gamma_init = np.random.gamma(
            shape=self.gamma_shape,
            scale=1.0 / self.gamma_shape,
            size=self.K,
        )
        gamma, _, _, _ = _cavi_doc_inference(
            indices=row.indices,
            counts=row.counts,
            expElogbeta=expElogbeta,
            alpha=global_params["alpha"],
            gamma_init=gamma_init,
            max_iter=self.cavi_max_iter,
            tol=self.cavi_tol,
        )
        theta = gamma / gamma.sum()
        return {"gamma": gamma, "theta": theta}

    def iteration_summary(self, global_params: dict[str, np.ndarray]) -> str:
        """Per-iter α / η / Σλ view, dict-λ aware.

        domains=None delegates to OnlineLDA.iteration_summary. Multi-domain needs an
        override: the inherited implementation does `lam.sum(axis=1)`, which does not
        work on a per-domain dict λ, and VIRunner calls this every iteration. It
        reports Σλ_k PER DOMAIN: pooling the blocks would hide one domain's topics
        diverging in mass while the other's stay flat.
        """
        if self.domains is None:
            return super().iteration_summary(global_params)
        alpha = np.asarray(global_params["alpha"])
        lam = global_params["lambda"]
        lam_parts = []
        for m in range(len(self.domains)):
            rs = lam[m].sum(axis=1)
            lam_parts.append(f"Σλ_k[m{m}: min={rs.min():.3g} max={rs.max():.3g}]")
        return (
            f"α[min={alpha.min():.4g} max={alpha.max():.4g} mean={alpha.mean():.4g}], "
            f"η={float(self.eta):.4g}, "
            + " ".join(lam_parts)
        )

    def get_metadata(self) -> dict[str, Any]:
        """Shape constants plus, in multi-domain mode, `domains`: the per-domain
        vocabulary widths that slice the concatenated id space, without which a
        saved multi-domain result cannot be interpreted. Plain Python types only
        (`metadata` is written verbatim into manifest.json)."""
        md = super().get_metadata()
        if self.domains is not None:
            md["domains"] = [int(v) for v in self.domains]
        return md

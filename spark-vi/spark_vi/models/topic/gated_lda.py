"""GatedOnlineLDA: the SVI (variational) twin of the collapsed-Gibbs placement engine.

Overrides exactly two OnlineLDA methods:
  * local_update — restrict each training doc's CAVI to DagLayout.allowed_set(frontier)
    (the exact variational analogue of the Gibbs gate in dag_placement.fit_gated); sstats
    for disallowed topics stay zero, welding each node's topic to its subtree's documents.
    An empty frontier is a labeled background doc, gated to the background block only (NOT
    full-K) — the same convention as the Gibbs oracle and the gated STM.
  * initialize_global — dispatch on a pluggable init strategy (default "random").

Everything else (update_global SVI natural-gradient beta step, compute_elbo, combine_stats,
VIRunner integration) is inherited from OnlineLDA. TRAINING is gated (above); DEPLOYMENT is
different — GatedLDAModel._transform folds held-out docs in UNGATED full-K CAVI (the label is
unknown at scoring time, which is the whole point) -> theta -> node_affinity.

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
                 frontier_histogram: dict | None = None, **kw) -> None:
        # optimize_alpha is handled by the gated per-node Newton step (this class),
        # NOT OnlineLDA's full-K alpha_newton_step; pass it to the parent as False
        # so the inherited update_global never runs the vanilla alpha step.
        super().__init__(K=lay.K, vocab_size=vocab_size, optimize_alpha=False, **kw)
        self.lay = lay
        self.init = init
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
        An unknown name raises ValueError."""
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
        expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))

        lambda_stats = np.zeros_like(lam)
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
        """
        lam = global_params["lambda"]
        alpha = global_params["alpha"]
        eta = global_params["eta"]
        expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
        target_lam = eta + expElogbeta * target_stats["lambda_stats"]
        new_lam = (1.0 - learning_rate) * lam + learning_rate * target_lam
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

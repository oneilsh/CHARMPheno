"""Co-sampled Gibbs read-out engine over the DAG-offset PG-STM (step 2). Emits
offset-INCREMENT posterior draws on the compiler's identified quotient DAG.

Inference substrate: stick-breaking + Polya-Gamma augmentation (Polson, Scott &
Windle 2013; Linderman, Johnson & Adams 2015). The offset block is drawn from its
matrix-normal conditional each sweep (a proper joint chain, not a ridge point).
"""
import numpy as np

from spark_vi.models.topic.pg_stm import (
    stick_layout, gated_theta, gated_counts, omega_sample, psi_posterior,
    _draw_block_sigma,
)
from spark_vi.models.topic.pg_stm_dag import (
    dag_offset_ridge, offset_penalty, _softgate_estep_doc,
)
from spark_vi.models.topic._linalg import safe_inverse


def dag_offset_ridge_draw(WtW, WtM, Sigma, *, penalty, rng):
    """One matrix-normal draw of the offset coefficient block:
    C ~ MN(mean = (WtW + diag(penalty))^{-1} WtM, row_cov = (WtW + diag(penalty))^{-1},
    col_cov = Sigma). Its expectation is exactly dag_offset_ridge(WtW, WtM, penalty).
    Depth-scaled `penalty` is the diagonal Gaussian prior precision on the coefficient
    rows; Sigma is the stick-space residual covariance shared across the K-1 columns."""
    WtW = np.asarray(WtW, dtype=np.float64)
    WtM = np.asarray(WtM, dtype=np.float64)
    A = WtW + np.diag(np.asarray(penalty, dtype=np.float64))
    mean = dag_offset_ridge(WtW, WtM, penalty=penalty)
    Ainv = np.linalg.inv(A)
    L_row = np.linalg.cholesky((Ainv + Ainv.T) / 2.0)
    L_col = np.linalg.cholesky((Sigma + Sigma.T) / 2.0)
    Z = rng.standard_normal(mean.shape)
    return mean + L_row @ Z @ L_col.T


class PGSTMDagGibbs:
    """Warm-started, co-sampled Gibbs sweep over the DAG-offset gated PG-STM.

    Adapts the validated design-wall probe (`scratchpad/design_wall_gibbs_probe.py`,
    `offset_gibbs`, insight-0056) into a reusable engine kernel: an exact blocked
    Gibbs sampler under Polya-Gamma augmentation (Polson, Scott & Windle 2013;
    Linderman, Johnson & Adams 2015) over a stick-breaking gated topic model whose
    per-document mean is Gamma^T x_d + the additive offset over the document's DAG
    ancestral closure. Each sweep, per document: theta is composed EXACTLY from the
    current stick-space psi (no delta method), token topics z are sampled
    categorically, the gated Polya-Gamma sufficient stats drive an exact omega and
    psi draw. Globals redrawn every sweep: beta (Dirichlet, WARM-STARTED from
    ``beta_init`` so the chain stays in one label-switching basin — the co-sampling
    fix), the offset coefficient block C (matrix-normal draw via
    `dag_offset_ridge_draw`, fed back so the offset block is a proper joint chain
    rather than a ridge point), and Sigma (block inverse-Wishart, `_draw_block_sigma`).

    A document with more than one candidate closure in ``doc_candidates`` (a
    partial-label / hidden-leaf document) is a fractional-z mixture over its
    candidates (PLDA-style partial-label semantics, Ramage et al. 2011): each sweep,
    `_softgate_estep_doc` scores the candidates' marginal evidence under the current
    globals, and ONE candidate is resampled (a collapsed indicator-augmentation
    Gibbs step) to fix that sweep's covariate row and closure before the same exact
    per-document draw runs. A single hard candidate skips the resampling and always
    uses its own (fixed) closure, matching the probe.

    ``dag`` is the (already-quotient) DagGate the orchestrator compiled; the ridge
    penalty is `offset_penalty(...)` unless ``penalty_override`` is supplied (the
    merged-node fix: a quotient node created by collapsing a depth-span-s chain gets
    the summed penalty of the chain it replaces, computed by the orchestrator).
    """

    def __init__(self, K, V, partition, dag, *, P, n_iter=200, burn=100,
                lam_base=0.25, gamma_depth=1.0, gamma_ridge=1e-6, beta_eta=0.1,
                seed=0):
        self.K, self.V, self.partition, self.dag = int(K), int(V), partition, dag
        self.P = int(P)
        self.n_iter, self.burn = int(n_iter), int(burn)
        self.lam_base, self.gamma_depth, self.gamma_ridge = lam_base, gamma_depth, gamma_ridge
        self.beta_eta, self.seed = beta_eta, seed
        self.layout = stick_layout(partition)

    def run(self, docs, doc_candidates, *, beta_init=None, penalty_override=None,
            sigma_fixed=None):
        """Run the sweep. ``doc_candidates[d]`` is a list of (prior, nodes) pairs (the
        document's group is ``next(iter(docs[d].groups))``, shared across all of a
        document's candidates). Returns
        {"increment_draws": (n_kept, U, K-1) array (U = dag.n_offset_nodes; row u-1 is
        node u, root dropped), "beta": (K, V) final draw, "Sigma": (K-1, K-1) final
        draw, "membership": {doc_index: final-sweep candidate weights} for docs with
        more than one candidate (empty dict if none)}."""
        rng = np.random.default_rng(self.seed)
        K, V, P = self.K, self.V, self.P
        Ksm1 = K - 1
        dag = self.dag
        layout = self.layout
        B = layout["B"]
        U = dag.n_offset_nodes
        Pw = P + U

        penalty = (np.asarray(penalty_override, dtype=np.float64)
                  if penalty_override is not None
                  else offset_penalty(P, dag, gamma_ridge=self.gamma_ridge,
                                      lam_base=self.lam_base, gamma_depth=self.gamma_depth))

        D = len(docs)
        doc_group = [next(iter(doc.groups)) for doc in docs]
        group_docs = {g: [] for g in self.partition.groups}
        for d, g in enumerate(doc_group):
            group_docs[g].append(d)
        group_counts = {g: len(v) for g, v in group_docs.items()}

        doc_active = [layout["groups"][doc_group[d]]["active"] for d in range(D)]
        doc_allowed = [layout["groups"][doc_group[d]]["allowed"] for d in range(D)]
        doc_words = [np.repeat(np.asarray(doc.indices, np.int64),
                               np.asarray(doc.counts, np.int64)) for doc in docs]
        psi_docs = [np.zeros(len(doc_active[d])) for d in range(D)]

        # candidates as (prior, nodes, group) triples, per doc (group is shared
        # across a document's candidates -- _softgate_estep_doc's expected shape)
        doc_cands = [[(p, nodes, doc_group[d]) for (p, nodes) in doc_candidates[d]]
                    for d in range(D)]

        if beta_init is not None:
            beta = np.asarray(beta_init, dtype=np.float64).copy()
        else:
            beta = rng.random((K, V)) + self.beta_eta
            beta /= beta.sum(axis=1, keepdims=True)

        Cf = np.zeros((Pw, Ksm1))
        # sigma_fixed holds Sigma constant (skip the block-IW draw) -- a calibration
        # diagnostic: pinning Sigma at a known truth isolates Sigma-estimation error from
        # increment-prior misspecification in the coverage decomposition (insight 0057).
        Sigma = (np.eye(Ksm1) if sigma_fixed is None
                 else np.asarray(sigma_fixed, dtype=np.float64).copy())

        # augmented covariate w = [x ; offset_indicator(nodes)]; the offset half is
        # re-sampled each sweep for soft (multi-candidate) docs, fixed for hard docs
        W = np.zeros((D, Pw))
        for d in range(D):
            W[d, :P] = np.asarray(docs[d].x, dtype=np.float64)
            W[d, P:] = dag.offset_indicator(doc_cands[d][0][1])

        nu0 = float(Ksm1 + 2)
        kept = []
        membership = {}
        for it in range(self.n_iter):
            sig_inv = {g: safe_inverse(Sigma[np.ix_(layout["groups"][g]["active"],
                                                     layout["groups"][g]["active"])])
                      for g in self.partition.groups}
            log_beta = np.log(beta)
            word_topic_counts = np.zeros((K, V))
            S = np.zeros((Ksm1, Ksm1))
            M = np.zeros((D, Ksm1))
            for d, doc in enumerate(docs):
                g = doc_group[d]
                glay = layout["groups"][g]
                active = doc_active[d]; allowed = doc_allowed[d]; m_g = glay["m_g"]
                cands = doc_cands[d]

                if len(cands) > 1:
                    weights, _z_bar, _esteps = _softgate_estep_doc(
                        doc, cands, layout["groups"], log_beta, Cf, Sigma, dag,
                        K=K, B=B, inner_rounds=8, inner_tol=1e-3)
                    c = int(rng.choice(len(cands), p=weights))
                    W[d, P:] = dag.offset_indicator(cands[c][1])
                    if it == self.n_iter - 1:
                        membership[d] = weights

                psi_active = psi_docs[d]
                mu_active = (Cf.T @ W[d])[active]
                psi_bg, psi_gate, psi_fg = psi_active[:B - 1], psi_active[B - 1], psi_active[B:]
                theta = gated_theta(psi_bg, psi_gate, psi_fg)
                words = doc_words[d]
                if words.shape[0] > 0:
                    Pword = theta[None, :] * beta[np.ix_(allowed, words)].T
                    Pword_sum = Pword.sum(axis=1, keepdims=True)
                    Pword = np.where(Pword_sum > 0, Pword / np.where(Pword_sum > 0, Pword_sum, 1.0),
                                     1.0 / len(allowed))
                    cdf = np.cumsum(Pword, axis=1); cdf /= cdf[:, -1:]
                    u = rng.random(words.shape[0])
                    z_local = (u[:, None] < cdf).argmax(axis=1)
                    n_allowed = np.bincount(z_local, minlength=len(allowed)).astype(float)
                    np.add.at(word_topic_counts, (allowed[z_local], words), 1.0)
                else:
                    n_allowed = np.zeros(len(allowed))
                n_bg, n_fg = n_allowed[:B], n_allowed[B:]
                gate_a, gate_b, b_bg, b_fg = gated_counts(n_bg, n_fg)
                a_active = np.concatenate([n_bg[:B - 1], np.array([gate_a]), n_fg[:m_g - 1]])
                b_active = np.concatenate([b_bg, np.array([gate_b]), b_fg])
                omega = omega_sample(b_active, psi_active, rng)
                m, Vd = psi_posterior(a_active, b_active, mu_active, sig_inv[g], omega)
                psi_active = rng.multivariate_normal(m, Vd)
                psi_docs[d] = psi_active
                M[d, active] = psi_active
                e_active = psi_active - mu_active
                S[np.ix_(active, active)] += np.outer(e_active, e_active)

            # beta: warm-started but SAMPLED every sweep (co-sampling)
            for k in range(K):
                beta[k] = rng.dirichlet(word_topic_counts[k] + self.beta_eta)
            # offset block: matrix-normal draw, fed back (Task 1 primitive)
            Cf = dag_offset_ridge_draw(W.T @ W, W.T @ M, Sigma, penalty=penalty, rng=rng)
            # Sigma: block inverse-Wishart draw (unless held fixed for the diagnostic)
            if sigma_fixed is None:
                Sigma = _draw_block_sigma(S, layout, self.partition, group_counts, D,
                                          Psi0_scale=1.0, nu0=nu0, Ksm1=Ksm1, rng=rng)

            if it >= self.burn:
                kept.append(Cf[P:].copy())

        increment_draws = np.array(kept) if kept else np.empty((0, U, Ksm1))
        return {"increment_draws": increment_draws, "beta": beta, "Sigma": Sigma,
                "membership": membership}

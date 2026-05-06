"""Diagnose the collapse-on-iter-1 mode observed in alpha_drift_probe.py.

Reproduces the failing seed (23) without Spark — we drive `local_update`
directly so we can dump γ, expElogthetad, and the per-doc statistic.

Hypothesis to test: the `RuntimeWarning: divide by zero in log` at
spark_vi/models/lda.py:342 indicates `expElogthetad` underflowed to 0
for some doc. Two candidate root causes:

  (a) γ converged near zero in some component (would mean CAVI is
      genuinely producing pathological posteriors at this init).
  (b) γ is fine, but `digamma(γ_dk) − digamma(Σγ)` is so negative that
      `exp(...)` underflows to 0 even though γ_dk itself is reasonable
      (would mean the optimization at lda.py:342 needs guarding).

For each, we want to see γ_d's component magnitudes for any doc whose
log(expElogthetad) hit -inf.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from scipy.special import digamma

from spark_vi.core import BOWDocument
from spark_vi.models.lda import VanillaLDA, _cavi_doc_inference

# Match probe's setup
K, V, D = 3, 100, 10_000
docs_avg_len = 100
true_alpha = np.array([0.1, 0.5, 0.9])
SEED = 23


def gen_corpus():
    rng_b = np.random.default_rng(11)
    true_beta = rng_b.dirichlet(np.full(V, 0.05), size=K)
    rng = np.random.default_rng(11)
    docs = []
    for d in range(D):
        theta_d = rng.dirichlet(true_alpha)
        N_d = max(1, rng.poisson(docs_avg_len))
        zs = rng.choice(K, size=N_d, p=theta_d)
        ws = np.array([rng.choice(V, p=true_beta[z]) for z in zs])
        unique, counts = np.unique(ws, return_counts=True)
        docs.append(BOWDocument(
            indices=unique.astype(np.int32),
            counts=counts.astype(np.float64),
            length=int(counts.sum()),
        ))
    return true_beta, docs


def main():
    true_beta, docs = gen_corpus()
    print(f"Generated {len(docs)} docs.")
    print(f"true_alpha = {true_alpha}")
    print()

    # Replicate the probe's seeding before model init (this is when
    # initialize_global runs `np.random.gamma` for λ).
    np.random.seed(SEED)
    model = VanillaLDA(K=K, vocab_size=V, optimize_alpha=True)
    global_params = model.initialize_global(None)

    lam = global_params["lambda"]
    print(f"λ shape={lam.shape}, sum={lam.sum():.2f}, "
          f"min={lam.min():.4g}, max={lam.max():.4g}")
    print(f"  Σλ_k = {lam.sum(axis=1)}")
    print()

    # Walk a small batch through CAVI and inspect.
    expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
    print(f"expElogbeta: min={expElogbeta.min():.4g}, "
          f"max={expElogbeta.max():.4g}")
    print(f"  Σ_v expElogbeta[k] = {expElogbeta.sum(axis=1)}")
    print()

    # Run CAVI on a few docs and find any with expElogthetad ≈ 0
    bad_docs = []
    sample_docs = docs[:500]
    print(f"Running CAVI on {len(sample_docs)} sample docs ...")
    for d_idx, doc in enumerate(sample_docs):
        gamma_init = np.random.gamma(
            shape=model.gamma_shape, scale=1.0 / model.gamma_shape, size=K
        )
        gamma, expElth, phi_norm, n = _cavi_doc_inference(
            indices=doc.indices, counts=doc.counts,
            expElogbeta=expElogbeta, alpha=global_params["alpha"],
            gamma_init=gamma_init, max_iter=model.cavi_max_iter,
            tol=model.cavi_tol,
        )
        # Diagnostic: any zero-or-near in expElogthetad?
        if (expElth <= 1e-300).any() or not np.isfinite(np.log(expElth + 1e-300)).all():
            bad_docs.append((d_idx, doc, gamma, expElth, phi_norm, n))

    print(f"  found {len(bad_docs)} doc(s) with expElogthetad <= 1e-300 "
          f"(out of {len(sample_docs)})")
    print()

    if bad_docs:
        for d_idx, doc, gamma, expElth, phi_norm, n in bad_docs[:5]:
            print(f"--- doc index {d_idx} ---")
            print(f"  doc length: {doc.length}, n_unique: {len(doc.indices)}")
            print(f"  γ (final)       = {gamma}")
            print(f"  γ.sum()         = {gamma.sum():.4g}")
            print(f"  digamma(γ)      = {digamma(gamma)}")
            print(f"  digamma(Σγ)     = {digamma(gamma.sum()):.4g}")
            print(f"  digamma(γ) − digamma(Σγ) = {digamma(gamma) - digamma(gamma.sum())}")
            print(f"  expElogthetad   = {expElth}")
            print(f"  phi_norm tail   = min={phi_norm.min():.4g}, "
                  f"max={phi_norm.max():.4g}")
            print(f"  CAVI iters used = {n}")
            print(f"  log(expElth)    = {np.log(expElth + 1e-300)}")
            print()
    else:
        print("[probe] No degenerate docs found in the first 500 — sweeping all.")
        # Try a broader sweep without storing
        any_bad = False
        for d_idx, doc in enumerate(docs):
            gamma_init = np.random.gamma(
                shape=model.gamma_shape, scale=1.0 / model.gamma_shape, size=K
            )
            gamma, expElth, _, _ = _cavi_doc_inference(
                indices=doc.indices, counts=doc.counts,
                expElogbeta=expElogbeta, alpha=global_params["alpha"],
                gamma_init=gamma_init, max_iter=model.cavi_max_iter,
                tol=model.cavi_tol,
            )
            if (expElth <= 1e-300).any():
                print(f"--- doc index {d_idx} (full-corpus sweep) ---")
                print(f"  γ = {gamma}")
                print(f"  γ.sum() = {gamma.sum():.4g}")
                print(f"  digamma(γ) − digamma(Σγ) = {digamma(gamma) - digamma(gamma.sum())}")
                print(f"  expElogthetad = {expElth}")
                any_bad = True
                if d_idx > 1500:
                    break
        if not any_bad:
            print("[probe] No degenerate docs in the full corpus either.")
            print()
            print("Hypothesis A (γ near zero) and B (digamma diff < -700) both")
            print("rejected for the first iter at this seed. Collapse must come")
            print("from a different mechanism — possibly the Newton step itself.")
            # Compute the actual e_log_theta_sum and Newton step manually
            print()
            print("Computing the actual α Newton step at iter 1 ...")
            np.random.seed(SEED)  # reset for reproducible λ init
            model2 = VanillaLDA(K=K, vocab_size=V, optimize_alpha=True)
            gp = model2.initialize_global(None)
            stats = model2.local_update(docs[:500], gp)
            print(f"  e_log_theta_sum (batch sum, 500 docs) = {stats['e_log_theta_sum']}")
            print(f"  n_docs = {stats['n_docs']}")
            # corpus-equivalent scale factor
            scale = D / 500
            scaled = stats['e_log_theta_sum'] * scale
            print(f"  scaled to D={D}: {scaled}")
            # Newton step
            from spark_vi.models.lda import _alpha_newton_step
            delta = _alpha_newton_step(gp["alpha"], scaled, D=float(D))
            print(f"  Δα (raw Newton step) = {delta}")
            # rho_t for iter 1
            rho_1 = (1024 + 1 + 1) ** -0.7
            print(f"  ρ_1 = {rho_1:.6f}")
            new_alpha = gp["alpha"] + rho_1 * delta
            print(f"  new α (before floor) = {new_alpha}")
            print(f"  new α (after floor) = {np.maximum(new_alpha, 1e-3)}")


if __name__ == "__main__":
    main()

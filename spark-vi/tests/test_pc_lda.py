"""Increment-1 recovery-parity gate for OnlinePCLDA (weight_y == 0).

The faithful VI port of ``analysis/pc``, increment 1, is the UNSUPERVISED SVI
path: at ``weight_y == 0`` the Prediction-Constrained objective is unsupervised
LDA-MAP, so ``OnlinePCLDA`` must (a) recover planted topics on the ``analysis/pc``
synthetic known-signal generator, (b) do so comparably to the exact in-memory
oracle ``PCTopicModel(weight_y=0)``, and (c) be equivalent to plain
``OnlineLDA`` (same code path — the ``weight_y == 0`` faithfulness gate).

This is RECOVERY PARITY, not numeric identity vs the oracle: the reference's
label-free local step is a NEF-MAP point estimate (100 unrolled exp-grad
steps), ``OnlineLDA``'s is mean-field CAVI Dirichlet γ. Same role, different
estimator — they agree on strong planted signal, not bit-for-bit (design
§"estimator gap"). So the oracle comparison is on matched-topic cosine, and the
OnlineLDA comparison IS an equivalence (identical estimator).

The planted corpus is the ``analysis/pc`` generator's ``_make_corpus`` (K_DOM
dominant disjoint-block topics + one predictive block); we reuse it verbatim and
fit with K = K_true so the unsupervised model can isolate every disjoint block.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

# The analysis/pc synthetic known-signal generator (reused verbatim) + its regime
# constants. analysis.pc.tests is a package; importing the module for its helper
# does not trigger collection here.
from analysis.pc.tests.test_synthetic_signal import (
    _make_corpus, SEED, D, V, K_DOM, SIG_BLOCK,
)

K_TRUE = K_DOM + 1  # generator plants K_DOM dominant blocks + 1 predictive block


def _planted_topics() -> np.ndarray:
    """Reconstruct the generator's (K_true, V) topic-word matrix.

    Mirrors ``_make_corpus``'s fixed, seed-independent block layout: K_DOM
    dominant topics on disjoint word blocks of the first V-SIG_BLOCK columns,
    plus one predictive topic owning the last SIG_BLOCK columns.
    """
    topics = np.full((K_TRUE, V), 0.01)
    dom_region = V - SIG_BLOCK
    dom_bl = dom_region // K_DOM
    for k in range(K_DOM):
        topics[k, k * dom_bl:(k + 1) * dom_bl] += 1.0
    topics[K_DOM, V - SIG_BLOCK:] += 1.0
    topics /= topics.sum(axis=1, keepdims=True)
    return topics


def _matched_cosine(beta_fit: np.ndarray, beta_true: np.ndarray) -> np.ndarray:
    """Hungarian-matched per-topic cosine between two (K, V) row-stochastic
    topic-word matrices. Returns the K matched cosines."""
    nf = beta_fit / np.linalg.norm(beta_fit, axis=1, keepdims=True)
    nt = beta_true / np.linalg.norm(beta_true, axis=1, keepdims=True)
    cos = nf @ nt.T
    fi, ti = linear_sum_assignment(-cos)
    return cos[fi, ti]


def _x_to_pc_docs(X: np.ndarray):
    """(D, V) dense count matrix -> list[PCDocument] (label-free placeholders)."""
    from spark_vi.models.topic.types import PCDocument
    docs = []
    for row in X:
        idx = np.nonzero(row)[0].astype(np.int32)
        cnt = row[idx].astype(np.float64)
        docs.append(PCDocument(
            indices=idx, counts=cnt, length=int(cnt.sum()),
            y=np.zeros(1), label_mask=np.zeros(1),
        ))
    return docs


def _fit_beta(model_cls_kwargs, docs, spark, cfg_kwargs):
    """Fit a topic VIModel via VIRunner over a Spark RDD of docs; return beta."""
    from spark_vi.core import VIConfig, VIRunner
    rdd = spark.sparkContext.parallelize(docs, numSlices=4).persist()
    rdd.count()
    model = model_cls_kwargs()
    result = VIRunner(model, config=VIConfig(**cfg_kwargs)).fit(rdd)
    rdd.unpersist(blocking=False)
    lam = result.global_params["lambda"]
    return lam / lam.sum(axis=1, keepdims=True), result


@pytest.mark.slow
def test_pc_lda_recovers_planted_topics_and_parity_with_oracle(spark):
    """OnlinePCLDA(weight_y=0) recovers the planted topics, comparably to the
    exact PCTopicModel(weight_y=0) oracle."""
    from spark_vi.models.topic.pc import OnlinePCLDA
    from analysis.pc.model import PCTopicModel

    X, y = _make_corpus(SEED)
    n_te = int(0.3 * D)
    Xtr = X[:D - n_te]
    beta_true = _planted_topics()
    docs = _x_to_pc_docs(Xtr)

    # Full-batch fit (no sampling randomness) so the OnlineLDA-equivalence test
    # below can assert a tight match; K = K_true, disjoint blocks are recoverable.
    cfg = dict(max_iterations=60, learning_rate_tau0=10.0,
               learning_rate_kappa=0.6, random_seed=0, convergence_tol=1e-9)
    beta_pc, _ = _fit_beta(
        lambda: OnlinePCLDA(K=K_TRUE, vocab_size=V, C=1, weight_y=0.0,
                            random_seed=0),
        docs, spark, cfg,
    )
    cos_pc = _matched_cosine(beta_pc, beta_true)

    # Oracle: exact in-memory unsupervised LDA-MAP on the same train counts.
    oracle = PCTopicModel(K=K_TRUE, C=1, weight_y=0.0, alpha=1.1,
                          pi_iters=100, max_iter=150, seed=0).fit(Xtr, y[:D - n_te])
    cos_oracle = _matched_cosine(oracle.topics_, beta_true)

    print(f"\n[increment-1 recovery] OnlinePCLDA matched cosine: "
          f"min={cos_pc.min():.3f} mean={cos_pc.mean():.3f}  |  "
          f"oracle PCTopicModel(wy=0): min={cos_oracle.min():.3f} "
          f"mean={cos_oracle.mean():.3f}")

    # (a) OnlinePCLDA recovers the planted topics.
    assert cos_pc.mean() > 0.85, f"weak topic recovery: matched cosines {cos_pc}"
    assert cos_pc.min() > 0.60, f"a planted topic was missed: {cos_pc}"

    # (b) Recovery parity with the oracle: comparable mean matched cosine (not
    # numeric identity — CAVI vs NEF-MAP estimator gap).
    assert cos_pc.mean() > cos_oracle.mean() - 0.10, (
        f"OnlinePCLDA recovery {cos_pc.mean():.3f} not comparable to oracle "
        f"{cos_oracle.mean():.3f}"
    )


@pytest.mark.slow
def test_pc_lda_weight_y_zero_equivalent_to_online_lda(spark):
    """At weight_y == 0, OnlinePCLDA is OnlineLDA on the numbers (same code path,
    same seeds => same fitted topics up to matched cosine ~ 1.0)."""
    from spark_vi.models.topic.pc import OnlinePCLDA
    from spark_vi.models.topic.lda import OnlineLDA

    X, _ = _make_corpus(SEED)
    docs = _x_to_pc_docs(X[:300])
    cfg = dict(max_iterations=40, learning_rate_tau0=10.0,
               learning_rate_kappa=0.6, random_seed=0, convergence_tol=1e-9)

    # OnlinePCLDA consumes PCDocuments; OnlineLDA consumes the same objects
    # (it only reads .indices/.counts), so both see identical inputs.
    beta_pc, _ = _fit_beta(
        lambda: OnlinePCLDA(K=K_TRUE, vocab_size=V, C=1, weight_y=0.0,
                            random_seed=0),
        docs, spark, cfg,
    )
    beta_lda, _ = _fit_beta(
        lambda: OnlineLDA(K=K_TRUE, vocab_size=V, random_seed=0),
        docs, spark, cfg,
    )
    cos = _matched_cosine(beta_pc, beta_lda)
    print(f"\n[increment-1 equivalence] OnlinePCLDA vs OnlineLDA matched cosine: "
          f"min={cos.min():.4f} mean={cos.mean():.4f}")
    assert cos.min() > 0.999, (
        f"OnlinePCLDA(weight_y=0) diverged from OnlineLDA: matched cosines {cos}"
    )


# ---------------------------------------------------------------------------
# Increment 2 — internal grad-check of the VI-PC's OWN supervised gradient.
#
# This is NOT a comparison against analysis/pc's gradient (the VI-PC is a
# variational model with a different π-estimator; see design §"CRITICAL
# framing"). It validates INTERNAL consistency: the accumulated autograd
# supervised gradient (∂/∂ the topic representation CAVI reads = the λ
# correction, and ∂/∂w_CK = the head step) matches a central finite-difference
# of the VI-PC's OWN per-minibatch supervised loss on a tiny fixed batch with
# C > 1 and a non-trivial label_mask. No Spark, no fit — pure numpy, fast.
# ---------------------------------------------------------------------------

def _tiny_sup_batch(seed=0, C=2, V=12):
    """A tiny fixed PCDocument batch with C>1 and a non-trivial observed mask."""
    from spark_vi.models.topic.types import PCDocument
    rng = np.random.default_rng(seed)
    docs = []
    for _ in range(6):
        nnz = int(rng.integers(3, 7))
        idx = np.sort(rng.choice(V, size=nnz, replace=False)).astype(np.int32)
        cnt = rng.integers(1, 6, size=nnz).astype(np.float64)
        y = rng.integers(0, 2, size=C).astype(np.float64)
        mask = rng.integers(0, 2, size=C).astype(np.float64)   # some cells unobserved
        docs.append(PCDocument(indices=idx, counts=cnt, length=int(cnt.sum()),
                               y=y, label_mask=mask))
    # Guarantee at least one fully-observed doc so the batch is non-degenerate.
    docs[0] = PCDocument(indices=docs[0].indices, counts=docs[0].counts,
                         length=docs[0].length, y=np.ones(C),
                         label_mask=np.ones(C))
    return docs


def test_pc_supervised_gradient_matches_finite_difference():
    """The accumulated autograd supervised gradient (topic correction + head)
    matches a central finite-difference of the same per-minibatch supervised
    loss to max rel err <= 1e-5."""
    from spark_vi.models.topic.pc import (
        _supervised_batch_value_and_grad, _supervised_batch_value,
    )

    rng = np.random.default_rng(1)
    K, V, C = 4, 12, 2
    n_iters = 20
    alpha = np.full(K, 1.1)
    topics_repr = rng.random((K, V)) * 0.3 + 0.01   # expElogbeta-like, positive
    w_CK = rng.standard_normal((C, K)) * 0.5
    docs = _tiny_sup_batch(seed=0, C=C, V=V)

    _loss, grad_topics, grad_wCK = _supervised_batch_value_and_grad(
        topics_repr, w_CK, docs, alpha, K, n_iters,
    )

    def _fd(param, grad, evaluate, eps=1e-6):
        # Central difference at every coordinate carrying a nonzero analytic grad.
        coords = list(zip(*np.nonzero(grad)))
        rel = []
        for (i, j) in coords:
            pp = param.copy(); pp[i, j] += eps
            pm = param.copy(); pm[i, j] -= eps
            num = (evaluate(pp) - evaluate(pm)) / (2 * eps)
            ana = grad[i, j]
            rel.append(abs(num - ana) / max(abs(ana), abs(num), 1e-8))
        return max(rel)

    err_topics = _fd(
        topics_repr, grad_topics,
        lambda p: _supervised_batch_value(p, w_CK, docs, alpha, K, n_iters),
    )
    err_head = _fd(
        w_CK, grad_wCK,
        lambda p: _supervised_batch_value(topics_repr, p, docs, alpha, K, n_iters),
    )
    print(f"\n[increment-2 grad-check] max rel err: topic-correction={err_topics:.2e}, "
          f"head={err_head:.2e}")
    assert err_topics <= 1e-5, f"topic-gradient rel err {err_topics:.2e} > 1e-5"
    assert err_head <= 1e-5, f"head-gradient rel err {err_head:.2e} > 1e-5"


def test_supervised_lambda_gradient_matches_finite_difference():
    """The CORRECTION applied to λ must be ∂loss/∂λ, not ∂loss/∂expElogbeta.

    ``_grad_topics_to_lambda`` completes the chain rule from the autograd stat
    (taken w.r.t. ``expElogbeta``) back to the global parameter λ, through
    ``expElogbeta = exp(ψ(λ) − ψ(Σλ))``. This checks it against a central
    finite-difference of the supervised loss as a function of λ — and asserts the
    UN-transformed stat (what the code subtracted from λ before the fix) does NOT
    match, so the bug can't silently return.
    """
    from spark_vi.models.topic.pc import (
        _supervised_batch_value, _supervised_batch_value_and_grad,
        _grad_topics_to_lambda,
    )
    from scipy.special import digamma

    rng = np.random.default_rng(3)
    K, V, C = 4, 10, 2
    n_iters = 20
    alpha = np.full(K, 1.1)
    w_CK = rng.standard_normal((C, K)) * 0.5
    docs = _tiny_sup_batch(seed=2, C=C, V=V)
    lam = rng.gamma(3.0, 2.0, size=(K, V)) + 0.5           # Dirichlet counts, O(tens)

    def eb_of(l):
        return np.exp(digamma(l) - digamma(l.sum(axis=1, keepdims=True)))

    def loss_of_lambda(l):
        return _supervised_batch_value(eb_of(l), w_CK, docs, alpha, K, n_iters)

    _loss, grad_eb, _ = _supervised_batch_value_and_grad(
        eb_of(lam), w_CK, docs, alpha, K, n_iters)
    grad_lambda = _grad_topics_to_lambda(grad_eb, lam)

    # Central finite-difference dloss/dlambda at every cell with a nonzero analytic grad.
    coords = list(zip(*np.nonzero(grad_lambda)))
    eps = 1e-6
    rel_fixed, rel_raw = [], []
    for (i, j) in coords:
        lp = lam.copy(); lp[i, j] += eps
        lm = lam.copy(); lm[i, j] -= eps
        num = (loss_of_lambda(lp) - loss_of_lambda(lm)) / (2 * eps)
        rel_fixed.append(abs(num - grad_lambda[i, j]) / max(abs(grad_lambda[i, j]), abs(num), 1e-8))
        rel_raw.append(abs(num - grad_eb[i, j]) / max(abs(grad_eb[i, j]), abs(num), 1e-8))
    err_fixed, err_raw = max(rel_fixed), min(rel_raw)
    print(f"\n[lambda grad-check] transformed rel err={err_fixed:.2e}; "
          f"raw-dEb best-case rel err={err_raw:.2e}")
    assert err_fixed <= 1e-5, f"transformed λ-gradient rel err {err_fixed:.2e} > 1e-5"
    # The un-transformed ∂loss/∂expElogbeta is the WRONG λ-gradient — guard the fix.
    assert err_raw > 1e-2, "raw dEb unexpectedly matched dloss/dlambda; transform is a no-op?"


def _one_supervised_step(head_optimizer, weight_y=50.0, head_lr=0.1, rho=0.5,
                         seed=0):
    """Run initialize_global -> local_update -> update_global once; return
    (model, gp_before, gp_after) for a small supervised OnlinePCLDA."""
    from spark_vi.models.topic.pc import OnlinePCLDA
    K, V, C = 4, 12, 2
    docs = _tiny_sup_batch(seed=seed, C=C, V=V)
    model = OnlinePCLDA(K=K, vocab_size=V, C=C, weight_y=weight_y,
                        head_optimizer=head_optimizer, head_lr=head_lr,
                        random_seed=0)
    gp0 = model.initialize_global(None)
    stats = model.local_update(docs, gp0)
    gp1 = model.update_global(gp0, stats, learning_rate=rho)
    return model, gp0, gp1, stats


def test_adam_head_moves_head_and_maintains_moment_buffers():
    """The 'adam' head path updates w_CK off zero AND carries first/second-moment
    buffers in global_params (absent for 'sgd'). A second step keeps the buffer key
    set stable and the moments accumulate."""
    model, gp0, gp1, stats = _one_supervised_step("adam")
    assert set(gp0) == {"lambda", "alpha", "eta", "w_CK", "w_CK_m", "w_CK_v"}
    assert np.abs(gp1["w_CK"]).max() > 0.0            # head moved off the zero seed
    assert np.abs(gp1["w_CK_m"]).max() > 0.0          # first moment populated
    assert np.abs(gp1["w_CK_v"]).max() > 0.0          # second moment populated
    # sgd path: no moment buffers.
    _, gp0s, gp1s, _ = _one_supervised_step("sgd")
    assert "w_CK_m" not in gp0s and "w_CK_m" not in gp1s
    # A second adam step keeps the key set constant (safe for combine/resume).
    gp2 = model.update_global(gp1, stats, learning_rate=0.5)
    assert set(gp2) == set(gp1)


def test_adam_head_step_is_invariant_to_weight_y_but_sgd_is_not():
    """The two-timescale decoupling property: because Adam normalizes by the running
    gradient RMS, the FIRST adam head step (w seeded at 0, ridge = 0) is independent
    of weight_y — the head runs on its own rate, not the topics' ρ·weight_y schedule.
    The sgd step, by contrast, scales linearly with weight_y."""
    _, _, a_lo, _ = _one_supervised_step("adam", weight_y=50.0)
    _, _, a_hi, _ = _one_supervised_step("adam", weight_y=5000.0)
    # 100x weight_y -> identical adam head update: the head no longer rides the
    # topics' weight_y dial (the topic λ correction is a separate, capped path).
    assert np.allclose(a_lo["w_CK"], a_hi["w_CK"], atol=1e-10)

    _, _, s_lo, _ = _one_supervised_step("sgd", weight_y=50.0)
    _, _, s_hi, _ = _one_supervised_step("sgd", weight_y=5000.0)
    # sgd head step scales with weight_y -> the two are far apart (not a no-op head).
    assert np.abs(s_lo["w_CK"]).max() > 0.0
    assert not np.allclose(s_lo["w_CK"], s_hi["w_CK"])


def test_adam_update_lazy_inits_moments_on_warm_start_from_sgd_checkpoint():
    """A WARM START seeds global params from a saved checkpoint (replacing
    initialize_global's output), so an sgd phase-1 checkpoint carries no
    w_CK_m/w_CK_v. The adam update_global must lazy-init them (zeros) rather than
    KeyError. Regression for the phase-2 warm-start crash."""
    from spark_vi.models.topic.pc import OnlinePCLDA
    K, V, C = 4, 12, 2
    docs = _tiny_sup_batch(seed=1, C=C, V=V)
    # An sgd model produces a checkpoint WITHOUT the Adam moment buffers.
    sgd = OnlinePCLDA(K=K, vocab_size=V, C=C, weight_y=0.0, head_optimizer="sgd",
                      random_seed=0)
    warm_gp = sgd.initialize_global(None)
    assert "w_CK_m" not in warm_gp                       # sgd checkpoint lacks buffers
    # Warm-start an adam supervised model from those params (the phase-2 path).
    adam = OnlinePCLDA(K=K, vocab_size=V, C=C, weight_y=1000.0, head_optimizer="adam",
                       head_lr=0.05, random_seed=0)
    stats = adam.local_update(docs, warm_gp)
    gp1 = adam.update_global(warm_gp, stats, learning_rate=0.5)   # must NOT KeyError
    assert np.abs(gp1["w_CK"]).max() > 0.0               # head moved
    assert "w_CK_m" in gp1 and "w_CK_v" in gp1           # buffers now present
    # A second step (buffers now present) also works.
    adam.update_global(gp1, stats, learning_rate=0.5)


# ---------------------------------------------------------------------------
# Increment 2 — OUTCOME parity: a trained OnlinePCLDA(weight_y>0) reproduces the
# Prediction-Constrained advantage on heldout per-label AUC — it beats BOTH its
# own unsupervised (weight_y=0) representation AND a two-stage baseline (that same
# unsupervised representation + a downstream logistic regression). This is the
# Hughes-regime synthetic (K_fit < K_dom: an unsupervised fit spends its topics
# on the dominant structure and misses the low-mass predictive topic; the label,
# flowing through the topic correction, reshapes a topic onto the predictive
# direction). Validated as OUTCOME parity within a stochastic-SVI tolerance band,
# NOT numeric identity vs analysis/pc (design §"CRITICAL framing").
# ---------------------------------------------------------------------------

def _labeled_pc_docs(X, y):
    """(D,V) counts + (D,) labels -> list[PCDocument] with all cells observed."""
    from spark_vi.models.topic.types import PCDocument
    docs = []
    for i, row in enumerate(X):
        idx = np.nonzero(row)[0].astype(np.int32)
        cnt = row[idx].astype(np.float64)
        docs.append(PCDocument(
            indices=idx, counts=cnt, length=int(cnt.sum()),
            y=np.array([float(y[i])]), label_mask=np.array([1.0]),
        ))
    return docs


@pytest.mark.slow
def test_pc_supervised_beats_two_stage_on_heldout_auc(spark):
    """OnlinePCLDA(weight_y>0) reproduces the PC advantage: heldout per-label AUC
    beats both the weight_y=0 head and a two-stage baseline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models.topic.pc import OnlinePCLDA
    from analysis.pc.tests.test_synthetic_signal import (
        _make_corpus, SEED, D, V, K_FIT, TEST_FRAC,
    )

    X, y = _make_corpus(SEED)
    n_te = int(TEST_FRAC * D)
    n_tr = D - n_te
    Xtr, Xte, ytr, yte = X[:n_tr], X[n_tr:], y[:n_tr], y[n_tr:]
    docs = _labeled_pc_docs(Xtr, ytr)

    # Full-batch deterministic SVI (no minibatch sampling noise) so the gate is
    # reproducible; K_FIT < K_dom is the hard PC regime.
    cfg = dict(max_iterations=50, learning_rate_tau0=10.0,
               learning_rate_kappa=0.6, random_seed=0, convergence_tol=1e-12)

    def _fit(weight_y):
        model = OnlinePCLDA(K=K_FIT, vocab_size=V, C=1, weight_y=weight_y,
                            alpha=1.1, grad_cavi_iters=10, random_seed=0)
        rdd = spark.sparkContext.parallelize(docs, numSlices=4).persist()
        rdd.count()
        runner = VIRunner(model, config=VIConfig(**cfg))
        result = runner.fit(rdd)
        # theta for a held-out split via the SAME label-free CAVI (infer_local).
        te = _labeled_pc_docs(Xte, yte)
        tr = _labeled_pc_docs(Xtr, ytr)
        th_te = np.array([model.infer_local(d, result.global_params)["theta"] for d in te])
        th_tr = np.array([model.infer_local(d, result.global_params)["theta"] for d in tr])
        rdd.unpersist(blocking=False)
        return model, result.global_params, th_tr, th_te

    # PC (supervised) and its own unsupervised (weight_y=0) representation.
    # weight_y=1000: after the digamma-Jacobian fix the supervised topic gradient
    # is in true λ-space (~65x smaller than the old mis-scaled ∂/∂expElogbeta), so
    # the prediction weight must be ~an order of magnitude larger to shape topics
    # by the same amount. The fix makes this SAFE — a weight_y sweep on this corpus
    # is monotone in AUC (0.57@50 -> 0.91@1000 -> 0.92@3000) with Σλ pinned at
    # ~7.6e4 throughout (NO runaway at any weight_y), where the old code diverged.
    _pc, gp_pc, _pc_tr, pc_te = _fit(weight_y=1000.0)
    _un, gp_un, un_tr, un_te = _fit(weight_y=0.0)

    w_pc = gp_pc["w_CK"][0]
    pc_auc = roc_auc_score(yte, pc_te @ w_pc)              # trained head
    wy0_auc = roc_auc_score(yte, un_te @ gp_un["w_CK"][0]) if np.abs(gp_un["w_CK"]).max() > 0 else 0.5
    two_stage = LogisticRegression(max_iter=1000).fit(un_tr, ytr)
    two_auc = roc_auc_score(yte, two_stage.predict_proba(un_te)[:, 1])
    lr_codes = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    codes_auc = roc_auc_score(yte, lr_codes.predict_proba(Xte)[:, 1])

    print(f"\n[increment-2 outcome parity] heldout ROC AUC — "
          f"PC(wy=1000)={pc_auc:.4f}  weight_y=0 head={wy0_auc:.4f}  "
          f"two-stage={two_auc:.4f}  LR-on-codes={codes_auc:.4f}  "
          f"(pos rate te={yte.mean():.2f}, |w_pc|max={np.abs(w_pc).max():.3g})")

    # PC clears chance by a clear margin (the SVI-CAVI fit reaches a lower
    # absolute AUC than the reference's L-BFGS + 100-step NEF, but the PC
    # ADVANTAGE — the quantity under test — reproduces robustly).
    assert pc_auc > 0.75, f"PC heldout AUC {pc_auc:.3f} not clearly above chance"
    # PC beats its own unsupervised head (which sits at ~0.5, the zero seed).
    assert pc_auc > wy0_auc + 0.08, (
        f"PC {pc_auc:.3f} did not beat its weight_y=0 head {wy0_auc:.3f}"
    )
    # PC beats the two-stage baseline by a real margin (the Hughes result).
    assert pc_auc > two_auc + 0.05, (
        f"PC {pc_auc:.3f} did not beat two-stage {two_auc:.3f} by margin"
    )


# ---------------------------------------------------------------------------
# Increment 2 — AT-SCALE NUMERICAL STABILITY (the regression gate for the
# supervised-topic-correction divergence).
#
# The bug: the supervised topic correction subtracted ρ·weight_y·grad_topics
# from λ directly, where grad_topics is a corpus-SUMMED gradient in topic-
# PROBABILITY space (expElogbeta ~ O(1/V)) while λ is in Dirichlet-COUNT space
# (Σλ ~ corpus tokens). On a real 33k-doc run at weight_y=100 the correction
# dwarfed λ → Σλ doubled every iter → the (unsupervised-LDA) ELBO computed on
# the corrupted λ exploded to ±1e32 by iter 3. The increment-2 tests passed
# only because their synthetic corpora were small enough to stay stable — a
# TESTING GAP. This test fills it: a few-thousand-doc corpus is large enough
# that the corpus-summed correction is decades larger than λ, so WITHOUT the
# trust region λ diverges here; WITH it, everything stays bounded at
# weight_y ∈ {100, 1000} AND an aggressive tau0=64.
# ---------------------------------------------------------------------------

def _scale_stability_corpus(n_docs=2500, K=4, C=2, block=5, seed=0):
    """A few-thousand-doc labeled corpus: K disjoint vocab blocks, one predictive
    block driving a single OBSERVED label cell/row. C>1, exactly one observed cell
    per doc (semi-supervised asymmetry) — enough real supervised signal that the
    topic correction is non-trivial, so the trust region is genuinely exercised."""
    from spark_vi.models.topic.types import PCDocument
    rng = np.random.default_rng(seed)
    V = K * block
    sig = V - block           # first column of the last (predictive) block
    docs = []
    for _ in range(n_docs):
        t = int(rng.integers(0, K))
        favored = list(range(t * block, (t + 1) * block))
        counts = np.zeros(V)
        for w in rng.choice(favored, size=int(rng.integers(8, 16)), replace=True):
            counts[w] += 1.0
        # Label: does this doc load on the predictive block? (observed on cell 0.)
        label = 1.0 if t == K - 1 else 0.0
        # A little label noise so the head can't saturate to a trivial separator.
        if rng.random() < 0.1:
            label = 1.0 - label
        idx = np.nonzero(counts)[0].astype(np.int32)
        y = np.zeros(C); y[0] = label
        mask = np.zeros(C); mask[0] = 1.0        # exactly one observed cell/row
        docs.append(PCDocument(
            indices=idx, counts=counts[idx].astype(np.float64),
            length=int(counts.sum()), y=y, label_mask=mask,
        ))
    return docs, V, K, C


@pytest.mark.slow
@pytest.mark.parametrize("weight_y", [100.0, 1000.0])
def test_pc_supervised_at_scale_stability(spark, weight_y):
    """At corpus scale + large weight_y + aggressive tau0=64, the trust-region
    topic correction keeps the fit numerically bounded: ELBO stays finite and does
    not explode, Σλ does not run away geometrically, |w_CK|max stays bounded.

    This is the regression gate for the diverging supervised topic correction —
    without the trust region λ diverges on this corpus (ELBO → ±1e32, Σλ doubling
    per iter). Both weight_y values must stay bounded (weight_y is a robust dial)."""
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models.topic.pc import OnlinePCLDA

    docs, V, K, C = _scale_stability_corpus(n_docs=2500, K=4, C=2, seed=0)
    rdd = spark.sparkContext.parallelize(docs, numSlices=4).persist()
    rdd.count()

    model = OnlinePCLDA(K=K, vocab_size=V, C=C, weight_y=weight_y,
                        alpha=1.1, grad_cavi_iters=8, random_seed=0)

    # Aggressive tau0=64 (the task's stress setting) + full-batch so the
    # correction is summed over the whole corpus every iter — the exact
    # divergence condition. The trust region must make this safe on its own.
    cfg = VIConfig(max_iterations=40, learning_rate_tau0=64.0,
                   learning_rate_kappa=0.6, random_seed=0, convergence_tol=1e-12)

    sum_lam, wmax = [], []

    def _rec(_it, gp, _trace):
        sum_lam.append(float(np.asarray(gp["lambda"]).sum()))
        wmax.append(float(np.abs(np.asarray(gp["w_CK"])).max()))

    result = VIRunner(model, config=cfg).fit(rdd, on_iteration=_rec)
    rdd.unpersist(blocking=False)

    elbo = np.asarray(result.elbo_trace, dtype=np.float64)
    sum_lam = np.asarray(sum_lam)
    wmax = np.asarray(wmax)
    print(f"\n[increment-2 at-scale stability, weight_y={weight_y:g}, tau0=64] "
          f"ELBO first={elbo[0]:.4g} last={elbo[-1]:.4g} "
          f"max|ELBO|={np.abs(elbo).max():.4g} | "
          f"Sum_lambda first={sum_lam[0]:.4g} last={sum_lam[-1]:.4g} "
          f"max={sum_lam.max():.4g} | |w_CK|max last={wmax[-1]:.4g} "
          f"max={wmax.max():.4g}")

    # ELBO stays FINITE and does not blow up (no 1e32).
    assert np.all(np.isfinite(elbo)), f"ELBO went non-finite: {elbo}"
    assert np.abs(elbo).max() < 1e9, (
        f"ELBO exploded (max|ELBO|={np.abs(elbo).max():.3g}): {elbo}"
    )
    # Σλ stays bounded — no geometric runaway. Anchored to the first iterate:
    # a doubling-per-iter runaway over 40 iters would be 2^40x; a 50x cap is a
    # wide margin that still catches any geometric blow-up.
    assert np.all(np.isfinite(sum_lam)), f"Sum_lambda went non-finite: {sum_lam}"
    assert sum_lam.max() < 50.0 * sum_lam[0], (
        f"Sum_lambda ran away: first={sum_lam[0]:.4g} max={sum_lam.max():.4g}"
    )
    # |w_CK|max stays bounded.
    assert np.all(np.isfinite(wmax)), f"|w_CK|max went non-finite: {wmax}"
    assert wmax.max() < 1e3, f"|w_CK|max ran away: {wmax.max():.4g}"

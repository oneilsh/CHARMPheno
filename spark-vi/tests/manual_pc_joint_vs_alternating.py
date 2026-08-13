"""De-risk the joint-vs-alternating hypothesis in the FAITHFUL reference.

The realistic benchmark showed OnlinePCLDA's co-fit head tops out at topics-LR ~0.53
while the reference (joint L-BFGS over (topics, head)) reaches ~0.87. The confound
experiment (manual_pc_hughes_settings_experiment) ruled out pi-iters and weight_y. The
remaining hypothesis: it is the JOINT vs ALTERNATING optimization of (topics, head).

This settles it INSIDE the reference, holding everything else fixed. PCTopicModel now
has a fit_mode:
  * 'joint'       — L-BFGS over the concatenated (w_KV, w_CK) vector (Hughes / oracle).
  * 'alternating' — block-coordinate: L-BFGS the topics with head fixed, then the head
                    with topics fixed, repeat. SAME objective, pi-MAP T=100, L2, init,
                    full-batch, same solver — only the topic<->head coupling is severed.

If 'alternating' collapses toward OnlinePCLDA's ~0.53 while 'joint' holds ~0.87, the gap
IS the coupling (motivating a joint step in the online model). If 'alternating' also
reaches ~0.87, jointness is NOT the mechanism and we look elsewhere.

Run:  PYTHONPATH=<repo>/spark-vi:<repo> <venv>/bin/python \
      spark-vi/tests/manual_pc_joint_vs_alternating.py
"""
import importlib.util
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_spec = importlib.util.spec_from_file_location(
    "h", "spark-vi/tests/manual_pc_dag_case_finding_realistic.py")
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)


def main():
    from analysis.pc.model import PCTopicModel
    d = np.load(h.BETA_NPZ)
    beta, usage = d["beta"], d["usage_pct"]
    rng = np.random.default_rng(h.SEED)
    X, Y, mask = h.make_corpus(rng, beta, usage)
    DMAX, VMAX = 1200, 3000
    X, Y, mask = X[:DMAX], Y[:DMAX], mask[:DMAX]
    D = X.shape[0]
    freq = X.sum(0)
    keep = np.sort(np.argsort(freq)[::-1][:VMAX])
    keep = keep[freq[keep] > 0]
    X = X[:, keep]
    n_te = int(h.TEST_FRAC * D)
    Xtr, Xte = X[:-n_te], X[-n_te:]
    Ytr, Yte, Mtr = Y[:-n_te], Y[-n_te:], mask[:-n_te]
    nodes = list(range(1, h.C))
    print(f"corpus D={D} V(trim)={X.shape[1]} K={h.K_FIT} C={h.C} weight_y={h.WEIGHT_Y}",
          flush=True)

    def auc(P):
        return {l: roc_auc_score(Yte[:, l], P[:, l]) for l in nodes
                if 0 < Yte[:, l].sum() < len(Yte)}

    def m(a):
        return float(np.mean([a[k] for k in nodes if k in a])) if a else float("nan")

    def evaluate(ref, tag):
        head = auc(ref.predict_proba(Xte))
        pi_tr, pi_te = ref.transform(Xtr), ref.transform(Xte)
        S = np.zeros((n_te, h.C))
        for l in nodes:
            sel = Mtr[:, l] == 1
            if 0 < Ytr[sel, l].sum() < sel.sum():
                S[:, l] = LogisticRegression(max_iter=2000).fit(
                    pi_tr[sel], Ytr[sel, l]).predict_proba(pi_te)[:, 1]
        tlr = auc(S)
        print(f"  {tag:<12} HEAD={m(head):.3f}  topics-LR={m(tlr):.3f}  "
              f"|w|max={np.abs(ref.w_CK_).max():.3f}  final_obj={ref.final_obj_:.5f}  "
              f"iters={ref.n_iter_}", flush=True)

    # pi_iters reduced 100 -> 40 for tractable wall-clock; the confound experiment
    # (manual_pc_hughes_settings_experiment) proved pi-iters do not move the outcome,
    # and BOTH legs share it, so the joint-vs-alternating contrast is unaffected.
    import os
    PI = int(os.environ.get("EXP_PI", "40"))
    common = dict(K=h.K_FIT, C=h.C, weight_y=h.WEIGHT_Y, alpha=1.05, lambda_w=0.001,
                  doc_batch_size=128, seed=0, pi_iters=PI)
    print(f"\nFAITHFUL reference (pi_iters={PI}), SAME corpus/objective/pi/L2/init — "
          "only coupling differs:", flush=True)
    evaluate(PCTopicModel(max_iter=150, fit_mode="joint", **common).fit(
        Xtr, Ytr, label_mask=Mtr), "joint")
    evaluate(PCTopicModel(fit_mode="alternating", alt_rounds=25, alt_block_maxiter=50,
                          **common).fit(Xtr, Ytr, label_mask=Mtr), "alternating")
    print("\n(OnlinePCLDA online alternating, same corpus: co-fit topics-LR~0.53, "
          "head~0.52 at head_l2=1e-3)", flush=True)


if __name__ == "__main__":
    main()

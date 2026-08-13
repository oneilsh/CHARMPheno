"""Head-to-head: the FAITHFUL reference (analysis/pc PCTopicModel, full-batch L-BFGS
joint convergence, Hughes-exact) vs OnlinePCLDA (one-step Newton head) on the SAME
realistic corpus. Settles whether the co-fit head's failure is OnlinePCLDA's one-step
optimizer (reference should predict near its topics-LR ceiling) or PC/topics.

Reuses the realistic benchmark's generator. In-memory (no Spark). Vocab trimmed to the
concepts the corpus actually uses, to keep the reference's L-BFGS tractable.

Run:  PYTHONPATH=<repo>/spark-vi:<repo> <venv>/bin/python \
      spark-vi/tests/manual_pc_reference_comparison.py
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
    used = np.nonzero(X.sum(0))[0]                    # trim to concepts actually emitted
    X = X[:, used]
    n_te = int(h.TEST_FRAC * h.D)
    Xtr, Xte = X[:-n_te], X[-n_te:]
    Ytr, Yte, Mtr = Y[:-n_te], Y[-n_te:], mask[:-n_te]
    nodes = list(range(1, h.C))
    print(f"corpus D={h.D} V(used)={X.shape[1]} K={h.K_FIT} C={h.C} weight_y={h.WEIGHT_Y}", flush=True)

    ref = PCTopicModel(K=h.K_FIT, C=h.C, weight_y=h.WEIGHT_Y, alpha=1.05,
                       lambda_w=0.001, max_iter=200, doc_batch_size=1024, seed=0)
    ref.fit(Xtr, Ytr, label_mask=Mtr)

    def auc(P):
        return {l: roc_auc_score(Yte[:, l], P[:, l]) for l in nodes
                if 0 < Yte[:, l].sum() < len(Yte)}

    def m(a):
        return float(np.mean([a[k] for k in nodes if k in a])) if a else float("nan")

    head = auc(ref.predict_proba(Xte))               # reference's CONVERGED co-fit head
    pi_tr, pi_te = ref.transform(Xtr), ref.transform(Xte)
    S = np.zeros((n_te, h.C))
    for l in nodes:
        sel = Mtr[:, l] == 1
        if 0 < Ytr[sel, l].sum() < sel.sum():
            S[:, l] = LogisticRegression(max_iter=2000).fit(
                pi_tr[sel], Ytr[sel, l]).predict_proba(pi_te)[:, 1]
    tlr = auc(S)

    def line(name, a):
        return f"  {name:<16} mean={m(a):.3f}   " + " ".join(
            f"{l}:{a.get(l, float('nan')):.2f}" for l in nodes)

    print("\nREFERENCE (analysis/pc, L-BFGS joint convergence, lambda_w=0.001):", flush=True)
    print(line("co-fit HEAD", head), flush=True)
    print(line("topics-LR", tlr), flush=True)
    print(f"  |w_CK|max = {np.abs(ref.w_CK_).max():.3f}", flush=True)
    print("\n(OnlinePCLDA one-step Newton head, same corpus: co-fit HEAD=0.646, "
          "topics-LR=0.965, |w|max=3.4e11)", flush=True)


if __name__ == "__main__":
    main()

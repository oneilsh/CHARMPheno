"""Diagnostic: WHY does the co-fit head shape topics to be predictive (topics-LR 0.97)
yet fail to predict itself (head 0.65)? Distinguishes three mechanisms on ONE flat-PC fit:

  (1) train/test gap      — head good on TRAIN theta, bad on TEST theta (overfit).
  (2) under-convergence   — head bad on BOTH (lags the moving topics), w_CK MIS-AIMED
                            vs the post-hoc LR direction (low cosine).
  (3) theta-rep mismatch  — head trained against the 10-iter differentiable CAVI theta
                            but scored on the 100-iter infer_local theta; head is fine on
                            ITS OWN theta, mis-scored on the other.

Reports, per split and theta representation: co-fit head AUC, post-hoc LR AUC, and mean
cosine(w_CK[c], LR_coef[c]).

Run:
  JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
  PYSPARK_PYTHON=<venv>/bin/python PYSPARK_DRIVER_PYTHON=<venv>/bin/python \
  PYTHONPATH=<repo>/spark-vi:<repo> <venv>/bin/python \
  spark-vi/tests/manual_pc_head_shaping_diagnostic.py
"""
import importlib.util
import numpy as np
from scipy.special import digamma
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_spec = importlib.util.spec_from_file_location(
    "h", "spark-vi/tests/manual_pc_dag_case_finding_realistic.py")
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)


def main():
    import pyspark
    from spark_vi.models.topic.pc import _predict_proba_np, _cavi_theta_anp
    d = np.load(h.BETA_NPZ)
    beta, usage = d["beta"], d["usage_pct"]
    V = beta.shape[1]
    rng = np.random.default_rng(h.SEED)
    X, Y, mask = h.make_corpus(rng, beta, usage)
    n_te = int(h.TEST_FRAC * h.D)
    Xtr, Xte, Ytr, Yte, Mtr = X[:-n_te], X[-n_te:], Y[:-n_te], Y[-n_te:], mask[:-n_te]
    nodes = list(range(1, h.C))

    spark = (pyspark.sql.SparkSession.builder.master("local[4]")
             .appName("pc-head-diag").config("spark.ui.enabled", "false")
             .config("spark.sql.shuffle.partitions", "8").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    try:
        tr = h.to_docs(Xtr, Ytr, Mtr)
        te = h.to_docs(Xte, Yte, np.ones((len(Xte), h.C)))
        mB, gpB = h.fit(spark, tr, V, weight_y=h.WEIGHT_Y, head=None)
        w = np.asarray(gpB["w_CK"]); lam = gpB["lambda"]
        eb = np.exp(digamma(lam) - digamma(lam.sum(1, keepdims=True)))
        alpha = np.asarray(mB.alpha, dtype=np.float64)

        def th_infer(docs):                              # 100-iter infer_local (eval rep)
            return np.array([mB.infer_local(dd, gpB)["theta"] for dd in docs])

        def th_grad(docs, n):                            # n-iter differentiable CAVI (train rep)
            return np.array([np.asarray(_cavi_theta_anp(
                eb[:, dd.indices], np.asarray(dd.counts, np.float64), alpha, h.K_FIT, n))
                for dd in docs])

        th_tr, th_te = th_infer(tr), th_infer(te)
        th_te10 = th_grad(te, h.KG if hasattr(h, "KG") else 10)   # training rep on test

        def auc(theta, Ylab):
            S = np.array([_predict_proba_np(t, w, None) for t in theta])
            return {l: roc_auc_score(Ylab[:, l], S[:, l])
                    for l in nodes if 0 < Ylab[:, l].sum() < len(Ylab)}

        def m(a):
            return np.mean([a[k] for k in nodes if k in a]) if a else float("nan")

        # post-hoc LR (masked) + direction cosine vs w_CK.
        S_tr, S_te = np.zeros((len(tr), h.C)), np.zeros((len(te), h.C))
        lr_coef = np.zeros((h.C, h.K_FIT))
        cos = []
        for l in nodes:
            sel = Mtr[:, l] == 1
            if not (0 < Ytr[sel, l].sum() < sel.sum()):
                continue
            lr = LogisticRegression(max_iter=2000).fit(th_tr[sel], Ytr[sel, l])
            S_tr[:, l] = lr.predict_proba(th_tr)[:, 1]
            S_te[:, l] = lr.predict_proba(th_te)[:, 1]
            lr_coef[l] = lr.coef_[0]
            v, u = lr.coef_[0], w[l]
            den = np.linalg.norm(v) * np.linalg.norm(u)
            cos.append(float(v @ u / den) if den > 0 else 0.0)
        lr_tr = {l: roc_auc_score(Ytr[:, l], S_tr[:, l]) for l in nodes if 0 < Ytr[:, l].sum() < len(Ytr)}
        lr_te = {l: roc_auc_score(Yte[:, l], S_te[:, l]) for l in nodes if 0 < Yte[:, l].sum() < len(Yte)}

        print("\n=== head-shaping diagnostic (flat-PC, weight_y=%g) ===" % h.WEIGHT_Y, flush=True)
        print(f"co-fit HEAD  on TRAIN theta (100-it): mean={m(auc(th_tr, Ytr)):.3f}", flush=True)
        print(f"co-fit HEAD  on TEST  theta (100-it): mean={m(auc(th_te, Yte)):.3f}", flush=True)
        print(f"co-fit HEAD  on TEST  theta (10-it train-rep): mean={m(auc(th_te10, Yte)):.3f}", flush=True)
        print(f"posthoc LR   on TRAIN theta:          mean={m(lr_tr):.3f}", flush=True)
        print(f"posthoc LR   on TEST  theta:          mean={m(lr_te):.3f}", flush=True)
        print(f"mean cosine(w_CK, LR direction):      {np.mean(cos):+.3f}   per-node {np.round(cos,2)}", flush=True)
        # how far apart are the two theta representations the head trains-vs-eval on?
        drift = np.linalg.norm(th_te - th_te10, axis=1).mean()
        print(f"mean ||theta_100it - theta_10it|| on test: {drift:.3f}", flush=True)

        # === THE LAG TEST: freeze topics, keep stepping the head's OWN Newton update ===
        # If the co-fit head only failed because theta was moving, then with theta frozen
        # its cosine to the optimal direction should climb 0.64 -> ~1 and AUC 0.65 -> ~0.965,
        # using the SAME head machinery and the SAME 30% masked labels.
        from spark_vi.models.topic.pc import FlatLogisticHead
        hd = FlatLogisticHead()
        Kf, gci = h.K_FIT, 10

        def cos_lr(W):
            cs = []
            for l in nodes:
                u, v = W[l], lr_coef[l]
                den = np.linalg.norm(u) * np.linalg.norm(v)
                if den > 0:
                    cs.append(float(u @ v / den))
            return float(np.mean(cs))

        def head_te(W):
            S = np.array([_predict_proba_np(t, W, None) for t in th_te])
            return m({l: roc_auc_score(Yte[:, l], S[:, l]) for l in nodes
                      if 0 < Yte[:, l].sum() < len(Yte)})

        wf = w.copy()
        print("\nLAG TEST — freeze topics, keep stepping the head's Newton (from co-fit head):", flush=True)
        print(f"  step  0 (co-fit): cos={cos_lr(wf):+.3f}  headAUC={head_te(wf):.3f}", flush=True)
        for step in range(1, 16):
            _, _, g = hd.batch_value_and_grad(eb, wf, tr, alpha, Kf, gci)
            H = hd.batch_hessian(eb, wf, tr, alpha, Kf, gci)
            for c in range(h.C):
                Hc = H[c]
                ridge = 0.01 * (float(np.trace(Hc)) / Kf) + 1e-10
                wf[c] = wf[c] - np.linalg.solve(Hc + ridge * np.eye(Kf), g[c] + ridge * wf[c])
            if step in (1, 2, 3, 5, 8, 12, 15):
                print(f"  step {step:2d}: cos={cos_lr(wf):+.3f}  headAUC={head_te(wf):.3f}", flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

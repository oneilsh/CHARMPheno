"""REALISTIC local case-finding benchmark — the Mondo archetype planted on REAL EHR
topics (data/cache/sim_beta.npz: 300 cross-site LDA topics x 29,003 OMOP concepts).

Four EHR-hard knobs (test-honesty: this de-risks MECHANISM; only real AoU proves
transfer):
  * realistic beta overlap  — tokens ~ Multinomial(theta @ beta_real); phenotype topics
    share concepts with the background, so discrimination is NOT trivial.
  * Zipf background          — each patient's comorbidity mixture is drawn over the
    highest-usage real topics weighted by their real usage_pct.
  * weak low-mass rare signal— total signal mass is small (SIGNAL_FRAC), split over the
    closure; rare leaves appear in few patients, and K_FIT << 300 forces the fit to
    compress and DROP the low-mass rare phenotype topics (the toy-bars regime).
  * semi-supervised labels   — training label_mask is mostly 0 (P_OBS); the true frontier
    is retained ONLY for held-out scoring.

Methods compared, per-node held-out AUC + pc_topics_lr (topic quality vs head aim):
  (A) unsup LDA + post-hoc per-node LR on theta   — the current 'prediction machinery'.
  (B) flat PC head            — independent logistic per node, predict sigmoid(w.theta).
  (B') flat head, HIERARCHICAL prediction — closure product of the FLAT-trained sigmoids
       (the cheap 'train-flat / predict-hierarchical' option, no coupled training).
  (C) DAG-closure PC head     — closure-product head, co-fit.

Run:
  JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
  PYTHONPATH=<repo>/spark-vi:<repo> \
  <venv>/bin/python spark-vi/tests/manual_pc_dag_case_finding_realistic.py
"""
import numpy as np

SEED = 0
BETA_NPZ = "data/cache/sim_beta.npz"
PARENTS = [[], [0], [0], [1], [1], [2], [2]]     # 0 root; 1,2 mid; 3,4|5,6 leaves
C = len(PARENTS)
LEAVES = [3, 4, 5, 6]
LEAF_PREV = np.array([0.40, 0.38, 0.13, 0.09])   # node 2 subtree (5,6) rare
SIGNAL_FRAC = 0.20                                # weak total signal mass
N_BG_ACTIVE = 6                                   # comorbidity topics per patient
ALPHA_BG = 0.5
DOC_LEN = 160
P_OBS = 0.30                                      # semi-supervised: P(cell observed) in TRAIN
D = 2500
TEST_FRAC = 0.30
K_FIT = 20                                        # << 300 -> compresses, drops rare topics
WEIGHT_Y = 20.0
MAX_ITERS = 60


def closure(node):
    acc = {node}
    for p in PARENTS[node]:
        acc |= closure(p)
    return acc


CLOSURE = {n: closure(n) for n in range(C)}


def make_corpus(rng, beta, usage):
    """Plant the DAG on real topics. Returns (X counts DxV, Y DxC, mask DxC)."""
    K_real, V = beta.shape
    order = np.argsort(usage)[::-1]                # topics high->low usage
    bg_pool = order[:150]
    bg_p = usage[bg_pool] + 1e-6
    bg_p = bg_p / bg_p.sum()
    sig_topic = {n: int(order[200 + i]) for i, n in enumerate(range(1, C))}  # low-usage, distinct

    X = np.zeros((D, V))
    Y = np.zeros((D, C))
    mask = np.zeros((D, C))
    frontier = rng.choice(LEAVES, size=D, p=LEAF_PREV / LEAF_PREV.sum())
    for i in range(D):
        f = int(frontier[i])
        clo = CLOSURE[f]
        Y[i, list(clo)] = 1.0
        mask[i] = (rng.random(C) < P_OBS).astype(float)      # semi-supervised
        theta = np.zeros(K_real)
        active = rng.choice(bg_pool, size=N_BG_ACTIVE, replace=False, p=bg_p)
        theta[active] = (1 - SIGNAL_FRAC) * rng.dirichlet(np.full(N_BG_ACTIVE, ALPHA_BG))
        sig_nodes = [n for n in clo if n != 0]
        for n in sig_nodes:
            theta[sig_topic[n]] += SIGNAL_FRAC / len(sig_nodes)
        wp = theta @ beta
        X[i] = rng.multinomial(DOC_LEN, wp / wp.sum())
    return X, Y, mask


def to_docs(X, Y, mask):
    from spark_vi.models.topic.types import PCDocument
    docs = []
    for i, row in enumerate(X):
        idx = np.nonzero(row)[0].astype(np.int32)
        docs.append(PCDocument(indices=idx, counts=row[idx].astype(np.float64),
                               length=int(row[idx].sum()),
                               y=Y[i].astype(np.float64), label_mask=mask[i].astype(np.float64)))
    return docs


def fit(spark, docs, V, *, weight_y, head, head_l2=0.0):
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models.topic.pc import OnlinePCLDA
    model = OnlinePCLDA(K=K_FIT, vocab_size=V, C=C, weight_y=weight_y, alpha=1.05,
                        grad_cavi_iters=10, random_seed=0, head_optimizer="newton",
                        head_lr=0.7, weight_y_warmup_iters=10, head_l2=head_l2, head=head)
    cfg = VIConfig(max_iterations=MAX_ITERS, learning_rate_tau0=64.0,
                   learning_rate_kappa=0.6, random_seed=0, convergence_tol=1e-12)
    rdd = spark.sparkContext.parallelize(docs, numSlices=8).persist()
    rdd.count()
    gp = VIRunner(model, config=cfg).fit(rdd).global_params
    rdd.unpersist(blocking=False)
    return model, gp


def thetas(model, docs, gp):
    return np.array([model.infer_local(d, gp)["theta"] for d in docs])


def auc_table(Yte, scores, nodes):
    from sklearn.metrics import roc_auc_score
    a = {}
    for l in nodes:
        yl, sl = Yte[:, l], scores[:, l]
        if np.all(np.isfinite(sl)) and 0 < yl.sum() < len(yl):
            a[l] = roc_auc_score(yl, sl)
    return a


def posthoc_scores(model, gp, tr, te, Ytr, nodes, mask=None):
    """Per-node post-hoc classifier on the FITTED topics' theta. If `mask` is given,
    each node's classifier trains ONLY on that node's observed cells (apples-to-apples
    with the semi-supervised co-fit head); else on all train labels (the ceiling)."""
    from sklearn.linear_model import LogisticRegression
    th_tr, th_te = thetas(model, tr, gp), thetas(model, te, gp)
    S = np.zeros((len(te), C))
    for l in nodes:
        if mask is not None:
            sel = mask[:, l] == 1
            Xl, yl = th_tr[sel], Ytr[sel, l]
        else:
            Xl, yl = th_tr, Ytr[:, l]
        if len(yl) and 0 < yl.sum() < len(yl):
            S[:, l] = LogisticRegression(max_iter=2000).fit(Xl, yl).predict_proba(th_te)[:, 1]
    return S


def hier(S, M):
    """Hierarchical composition: P_hier[:,l] = prod_{a in closure(l)} S[:,a]."""
    return np.exp(np.log(np.clip(S, 1e-9, 1.0)) @ M.T)


def _mean(d, ks):
    v = [d[k] for k in ks if k in d]
    return float(np.mean(v)) if v else float("nan")


def main():
    import pyspark
    from spark_vi.models.topic.pc import DagClosureHead, _predict_proba_np
    d = np.load(BETA_NPZ)
    beta, usage = d["beta"], d["usage_pct"]
    K_real, V = beta.shape
    rng = np.random.default_rng(SEED)
    X, Y, mask = make_corpus(rng, beta, usage)
    n_te = int(TEST_FRAC * D)
    Xtr, Xte = X[:-n_te], X[-n_te:]
    Ytr, Yte = Y[:-n_te], Y[-n_te:]
    Mtr = mask[:-n_te]
    nodes = list(range(1, C))
    prev = Y.mean(0)
    obs_rate = (mask.sum(0) / D)
    print(f"beta=({K_real},{V}) D={D} K_fit={K_FIT}  prevalence "
          + " ".join(f"{l}:{prev[l]:.2f}" for l in nodes)
          + f"  (train obs-rate ~{obs_rate[1:].mean():.2f})", flush=True)

    spark = (pyspark.sql.SparkSession.builder.master("local[4]")
             .appName("pc-dag-realistic").config("spark.ui.enabled", "false")
             .config("spark.sql.shuffle.partitions", "8").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    try:
        tr = to_docs(Xtr, Ytr, Mtr)
        te = to_docs(Xte, Yte, np.ones((len(Xte), C)))
        clo_M = DagClosureHead(PARENTS)._closure_matrix

        def line(name, a):
            m = np.mean([a[k] for k in nodes if k in a]) if a else float("nan")
            rare = np.mean([a[k] for k in (5, 6) if k in a]) if any(k in a for k in (5, 6)) else float("nan")
            return (f"  {name:<22} mean={m:.3f} rare(5,6)={rare:.3f}   "
                    + " ".join(f"{l}:{a.get(l, float('nan')):.2f}" for l in nodes))

        # THE FIX: sweep the co-fit head's fixed L2 (head_l2). At 0.0 the relative ridge
        # vanishes on the separable shaped topics -> head stuck ~0.65; a positive L2 keeps
        # it finite -> the co-fit head itself should reach the post-hoc ceiling.
        m0, gp0 = fit(spark, tr, V, weight_y=0.0, head=None)
        S_uns_m = posthoc_scores(m0, gp0, tr, te, Ytr, nodes, mask=Mtr)

        # Distinguish "over-scaled ridge (w->0)" from "fundamental shape-vs-regularize
        # tension": report HEAD AUC, topics-LR (shaping health), and |w_CK| per l2. If
        # shaping (topics-LR ~0.96) survives at small l2 while the head improves, a sweet
        # spot exists; if topics-LR dies at ANY l2>0, the head genuinely can't shape while
        # regularized.
        unsup_lr = _mean(auc_table(Yte, S_uns_m, nodes), nodes)
        print(f"\nhead_l2 sweep (unsup posthoc-LR baseline={unsup_lr:.3f}):", flush=True)
        for l2 in (0.0, 1e-6, 1e-5, 1e-4, 5e-4):
            mB, gpB = fit(spark, tr, V, weight_y=WEIGHT_Y, head=None, head_l2=l2)
            thB = thetas(mB, te, gpB)
            hd = _mean(auc_table(Yte, np.array([_predict_proba_np(t, gpB["w_CK"], None)
                                                for t in thB]), nodes), nodes)
            tl = _mean(auc_table(Yte, posthoc_scores(mB, gpB, tr, te, Ytr, nodes, mask=Mtr),
                                 nodes), nodes)
            wmax = float(np.abs(gpB["w_CK"]).max())
            print(f"  l2={l2:<6g}  HEAD={hd:.3f}  topics-LR(shaping)={tl:.3f}  |w_CK|max={wmax:.2f}",
                  flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

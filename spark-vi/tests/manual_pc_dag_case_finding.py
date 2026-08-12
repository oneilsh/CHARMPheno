"""LOCAL synthetic end-to-end run of the DAG-closure PC head — the Mondo case-finding
archetype (the hierarchical analogue of toy_bars for the antidepressant work).

Generates a synthetic disease DAG with node-tied topics and RARE deep nodes, dilutes the
signal (most tokens are background comorbidity noise), then fits three models with the
REAL OnlinePCLDA + VIRunner in a local SparkContext and reports per-node held-out AUC:

  (A) unsupervised LDA + post-hoc per-node LogisticRegression on theta
      (the current case-finding 'prediction machinery' analogue),
  (B) flat PC head (C independent logistic heads),
  (C) DAG-closure PC head (P(node_l) = prod over the is-a closure of sigmoid(w.theta)).

The question: does the label-side hierarchy help, especially on the RARE deep nodes where
the flat head has few positives to learn from but the DAG head borrows strength from
better-estimated ancestors?

Run:
  JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
  PYTHONPATH=<repo>/spark-vi:<repo> \
  <venv>/bin/python spark-vi/tests/manual_pc_dag_case_finding.py
"""
import numpy as np

SEED = 0
# Label DAG: 0 root; 1,2 mid categories; 3,4 under 1; 5,6 under 2. Node 2's subtree
# (5,6) is RARE. Root owns no signal words (purely structural).
PARENTS = [[], [0], [0], [1], [1], [2], [2]]
C = len(PARENTS)                      # 7 label nodes
LEAVES = [3, 4, 5, 6]
LEAF_PREV = np.array([0.40, 0.40, 0.12, 0.08])   # -> node 2 subtree rare; 5,6 rarest
WPN = 15                              # signal words per non-root node
V = 6 * WPN + 40                      # 6 node blocks (90) + 40 background words = 130
K_FIT = 8
DOC_LEN = 120
SIGNAL_FRAC = 0.35                    # fraction of tokens from disease topics (rest background)
D = 3000
TEST_FRAC = 0.30
WEIGHT_Y = 100.0
MAX_ITERS = 60


def closure(node):
    acc = {node}
    for p in PARENTS[node]:
        acc |= closure(p)
    return acc


CLOSURE = {n: closure(n) for n in range(C)}
BG_START = 6 * WPN                    # background words start here


def node_block(k):                   # non-root node k in 1..6 owns a disjoint word block
    lo = (k - 1) * WPN
    return range(lo, lo + WPN)


def make_corpus(rng):
    """Return (X counts DxV, Y labels DxC, frontier D). Signal is dilute; deep nodes rare."""
    X = np.zeros((D, V))
    Y = np.zeros((D, C))
    frontier = rng.choice(LEAVES, size=D, p=LEAF_PREV)
    for i in range(D):
        f = int(frontier[i])
        clo = CLOSURE[f]
        Y[i, list(clo)] = 1.0
        signal_nodes = [n for n in clo if n != 0]     # root has no words
        n_sig = rng.binomial(DOC_LEN, SIGNAL_FRAC)
        for _ in range(n_sig):                        # signal tokens
            k = signal_nodes[rng.integers(len(signal_nodes))]
            X[i, rng.choice(list(node_block(k)))] += 1.0
        for _ in range(DOC_LEN - n_sig):              # background comorbidity noise
            X[i, BG_START + rng.integers(V - BG_START)] += 1.0
    return X, Y, frontier


def to_docs(X, Y):
    from spark_vi.models.topic.types import PCDocument
    docs = []
    for i, row in enumerate(X):
        idx = np.nonzero(row)[0].astype(np.int32)
        docs.append(PCDocument(
            indices=idx, counts=row[idx].astype(np.float64), length=int(row[idx].sum()),
            y=Y[i].astype(np.float64), label_mask=np.ones(C)))
    return docs


def fit(spark, docs, *, weight_y, head):
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models.topic.pc import OnlinePCLDA
    model = OnlinePCLDA(K=K_FIT, vocab_size=V, C=C, weight_y=weight_y, alpha=1.05,
                        grad_cavi_iters=10, random_seed=0, head_optimizer="newton",
                        head_lr=0.7, weight_y_warmup_iters=10, head=head)
    cfg = VIConfig(max_iterations=MAX_ITERS, learning_rate_tau0=32.0,
                   learning_rate_kappa=0.6, random_seed=0, convergence_tol=1e-12)
    rdd = spark.sparkContext.parallelize(docs, numSlices=8).persist()
    rdd.count()
    result = VIRunner(model, config=cfg).fit(rdd)
    rdd.unpersist(blocking=False)
    return model, result.global_params


def thetas(model, docs, gp):
    return np.array([model.infer_local(d, gp)["theta"] for d in docs])


def per_node_auc(Yte, scores):
    from sklearn.metrics import roc_auc_score
    aucs = {}
    for l in range(1, C):                             # skip root (always positive)
        yl, sl = Yte[:, l], scores[:, l]
        if np.any(~np.isfinite(sl)):                  # a degenerate method -> skip, don't crash
            continue
        if 0 < yl.sum() < len(yl):
            aucs[l] = roc_auc_score(yl, sl)
    return aucs


def topics_lr(model, gp, tr, te, Ytr, Yte):
    """pc_topics_lr: fresh per-node LogisticRegression on the FITTED topics' theta.
    Convergence-robust measure of TOPIC quality — separates 'topics carry the signal'
    from 'the co-fit head aims at it'."""
    from sklearn.linear_model import LogisticRegression
    th_tr, th_te = thetas(model, tr, gp), thetas(model, te, gp)
    S = np.zeros((len(te), C))
    for l in range(1, C):
        if 0 < Ytr[:, l].sum() < len(Ytr):
            S[:, l] = (LogisticRegression(max_iter=2000, C=1.0)
                       .fit(th_tr, Ytr[:, l]).predict_proba(th_te)[:, 1])
    return per_node_auc(Yte, S)


def _mean(d, ks):
    v = [d[k] for k in ks if k in d]
    return float(np.mean(v)) if v else float("nan")


def main():
    import os
    import sys
    # Executors must use THIS interpreter (the venv with numpy), not PATH's python3;
    # mirrors tests/conftest.py. Without this the workers ModuleNotFoundError on numpy.
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    import pyspark
    from spark_vi.models.topic.pc import DagClosureHead
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(SEED)
    X, Y, _ = make_corpus(rng)
    n_te = int(TEST_FRAC * D)
    Xtr, Xte, Ytr, Yte = X[:-n_te], X[-n_te:], Y[:-n_te], Y[-n_te:]
    prev = Y.mean(0)
    print(f"corpus D={D} V={V} C={C} K_fit={K_FIT}  node prevalence: "
          + " ".join(f"{l}:{prev[l]:.3f}" for l in range(1, C)), flush=True)

    spark = (pyspark.sql.SparkSession.builder.master("local[4]")
             .appName("pc-dag-case-finding").config("spark.ui.enabled", "false")
             .config("spark.sql.shuffle.partitions", "8").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    try:
        tr, te = to_docs(Xtr, Ytr), to_docs(Xte, Yte)

        nodes = list(range(1, C))

        def head_auc(model, gp):
            th = thetas(model, te, gp)
            S = np.array([model._head.predict_proba(t, gp["w_CK"]) for t in th])
            return per_node_auc(Yte, S)

        def line(name, auc):
            return (f"  {name:<16} mean={_mean(auc, nodes):.3f}   "
                    + " ".join(f"{l}:{auc.get(l, float('nan')):.2f}" for l in nodes))

        # (A) unsupervised topics + post-hoc per-node LR on theta (current machinery).
        m0, gp0 = fit(spark, tr, weight_y=0.0, head=None)
        auc_unsup = topics_lr(m0, gp0, tr, te, Ytr, Yte)

        # (B) flat PC head @ a moderate weight_y (ceiling reference).
        mB, gpB = fit(spark, tr, weight_y=20.0, head=None)
        auc_flat, lr_flat = head_auc(mB, gpB), topics_lr(mB, gpB, tr, te, Ytr, Yte)

        # (C) DAG-closure PC head over a weight_y SWEEP: head AUC vs pc_topics_lr
        # (topics-vs-head) to localize the earlier weight_y=100 failure.
        dag = {}
        for wy in (2.0, 10.0, 40.0):
            m, gp = fit(spark, tr, weight_y=wy, head=DagClosureHead(PARENTS))
            dag[wy] = (head_auc(m, gp), topics_lr(m, gp, tr, te, Ytr, Yte))

        print(f"\nper-node held-out AUC  (nodes 1..6; rare: 5,6 @ prev "
              f"{prev[5]:.2f}/{prev[6]:.2f})", flush=True)
        print(line("unsup+LR", auc_unsup), flush=True)
        print(line("flatPC head@20", auc_flat), flush=True)
        print(line("flatPC topics-LR", lr_flat), flush=True)
        for wy in (2.0, 10.0, 40.0):
            h, t = dag[wy]
            print(line(f"dagPC head@{int(wy)}", h), flush=True)
            print(line(f"dagPC topics-LR@{int(wy)}", t), flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

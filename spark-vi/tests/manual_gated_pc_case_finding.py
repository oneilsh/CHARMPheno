"""Gated-PC composition end-to-end: topic-side DAG gate x label-side DAG head.

Task #14. The supervised head is topic-engine-agnostic (reads only lambda/alpha/K), and
GatedOnlineLDA IS-A OnlineLDA, so OnlinePCLDA(topic_engine=GatedOnlineLDA(...)) composes
the two seams by construction (design: 2026-08-12-pc-supervised-head-seam-design.md 2x2).

This plants node-specific topics on a small Mondo-like DAG and compares, with the SAME
DagClosureHead and SAME docs, three points of the 2x2:
  (A) ungated + DAG head   — label-side supervision only (topic_engine=None).
  (B) GATED + flat head    — topic-side gate only (no closure coupling in the head).
  (C) GATED + DAG head     — the full Gated-PC composition.

Reports per-node held-out AUC of the head's P(node) and of node_affinity (topic-block
mass, the gate's native readout). The gate welds each node's topic block to its subtree's
docs; the head predicts closure membership from the (ungated, label-free) theta.

Run:
  JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
  PYSPARK_PYTHON=<venv>/bin/python PYSPARK_DRIVER_PYTHON=<venv>/bin/python \
  PYTHONPATH=<repo>/spark-vi:<repo> <venv>/bin/python \
  spark-vi/tests/manual_gated_pc_case_finding.py
"""
import numpy as np

SEED = 0
PARENTS = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}   # root 0; mid 1,2; leaves 3,4 | 5,6
C = 7                                             # label ids = DAG node ids 0..6
CLOSURE_PARENTS = [(), (0,), (0,), (1,), (1,), (2,), (2,)]
LEAVES = [3, 4, 5, 6]
LEAF_PREV = np.array([0.30, 0.30, 0.22, 0.18])
N_BG = 3
TPN = 1
V = 400
D = 1800
DOC_LEN = 120
SIGNAL_FRAC = 0.45
TEST_FRAC = 0.30
P_OBS = 0.5
WEIGHT_Y = 20.0
MAX_ITERS = 50


def build_layout():
    from spark_vi.models.topic.dag_placement import DagLayout
    return DagLayout(PARENTS, n_bg=N_BG, tpn=TPN)


def closure_ids(node):
    acc = {node}
    for p in ([PARENTS[node]] if node in PARENTS else []):
        acc |= closure_ids(p)
    return acc


def make_corpus(rng, lay):
    """Plant node-signature topics. topic block[u] (one topic) is node u's signature; the
    N_BG background topics are shared. Returns X, Y(DxC closure membership), frontier list."""
    K = lay.K
    # each topic gets a peaked word distribution over a distinct vocab slice
    beta = np.full((K, V), 0.5 / V)
    slice_w = V // K
    for k in range(K):
        beta[k, k * slice_w:(k + 1) * slice_w] += 1.0
    beta /= beta.sum(1, keepdims=True)

    X = np.zeros((D, V))
    Y = np.zeros((D, C))
    frontiers = []
    frt = rng.choice(LEAVES, size=D, p=LEAF_PREV / LEAF_PREV.sum())
    for i in range(D):
        leaf = int(frt[i])
        ids = closure_ids(leaf)
        Y[i, list(ids)] = 1.0
        frontiers.append(frozenset({leaf}))
        theta = np.zeros(K)
        theta[rng.choice(range(N_BG))] = 1 - SIGNAL_FRAC        # a background topic
        sig_topics = [lay.block[u][0] for u in ids if u != 0]   # node signature topics
        for t in sig_topics:
            theta[t] += SIGNAL_FRAC / len(sig_topics)
        wp = theta @ beta
        X[i] = rng.multinomial(DOC_LEN, wp / wp.sum())
    return X, Y, frontiers


def to_docs(X, Y, frontiers, mask):
    from spark_vi.models.topic.types import GatedPCDocument
    docs = []
    for i, row in enumerate(X):
        idx = np.nonzero(row)[0].astype(np.int32)
        docs.append(GatedPCDocument(
            indices=idx, counts=row[idx].astype(np.float64), length=int(row[idx].sum()),
            y=Y[i].astype(np.float64), label_mask=mask[i].astype(np.float64),
            frontier=frontiers[i]))
    return docs


def fit(spark, docs, lay, *, gated, head):
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models.topic.pc import OnlinePCLDA
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    engine = GatedOnlineLDA(lay, vocab_size=V, random_seed=0) if gated else None
    model = OnlinePCLDA(K=lay.K, vocab_size=V, C=C, weight_y=WEIGHT_Y, alpha=1.05,
                        grad_cavi_iters=10, random_seed=0, head_optimizer="newton",
                        head_lr=0.7, weight_y_warmup_iters=8, head_l2=1e-3,
                        head=head, topic_engine=engine)
    cfg = VIConfig(max_iterations=MAX_ITERS, learning_rate_tau0=64.0,
                   learning_rate_kappa=0.6, random_seed=0, convergence_tol=1e-12)
    rdd = spark.sparkContext.parallelize(docs, numSlices=8).persist()
    rdd.count()
    gp = VIRunner(model, config=cfg).fit(rdd).global_params
    rdd.unpersist(blocking=False)
    return model, gp


def main():
    import pyspark
    from spark_vi.models.topic.pc import DagClosureHead, FlatLogisticHead, _predict_proba_np
    from spark_vi.models.topic.gated_lda import node_affinity
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(SEED)
    lay = build_layout()
    X, Y, frontiers = make_corpus(rng, lay)
    n_te = int(TEST_FRAC * D)
    nodes = list(range(1, C))
    # test docs: ungated (empty frontier) fold-in, all cells scored
    mask = (rng.random((D, C)) < P_OBS).astype(float)
    print(f"Gated-PC: DAG nodes={list(range(1,C))} K={lay.K} V={V} D={D} weight_y={WEIGHT_Y}",
          flush=True)

    spark = (pyspark.sql.SparkSession.builder.master("local[4]")
             .appName("gated-pc").config("spark.ui.enabled", "false")
             .config("spark.sql.shuffle.partitions", "8").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    try:
        tr = to_docs(X[:-n_te], Y[:-n_te], frontiers[:-n_te], mask[:-n_te])
        te = to_docs(X[-n_te:], Y[-n_te:], [frozenset()] * n_te, np.ones((n_te, C)))
        Yte = Y[-n_te:]

        def evaluate(tag, model, gp, use_affinity):
            th = np.array([model.infer_local(dd, gp)["theta"] for dd in te])
            head_S = np.array([_predict_proba_np(t, gp["w_CK"],
                               getattr(model._head, "_closure_matrix", None)) for t in th])
            def auc(S):
                a = {l: roc_auc_score(Yte[:, l], S[:, l]) for l in nodes
                     if 0 < Yte[:, l].sum() < len(Yte)}
                return a, float(np.mean(list(a.values())))
            ha, hm = auc(head_S)
            line = f"  {tag:<22} head-AUC mean={hm:.3f}  " + " ".join(
                f"{l}:{ha.get(l, float('nan')):.2f}" for l in nodes)
            if use_affinity:
                aff = np.array([[node_affinity(t, lay).get(l, 0.0) for l in range(C)]
                                for t in th])
                _, am = auc(aff)
                line += f"   | node_affinity mean={am:.3f}"
            print(line, flush=True)

        dag_head = lambda: DagClosureHead(CLOSURE_PARENTS)
        mA, gA = fit(spark, tr, lay, gated=False, head=dag_head())
        evaluate("(A) ungated+DAGhead", mA, gA, use_affinity=False)
        mB, gB = fit(spark, tr, lay, gated=True, head=FlatLogisticHead())
        evaluate("(B) GATED+flathead", mB, gB, use_affinity=True)
        mC, gC = fit(spark, tr, lay, gated=True, head=dag_head())
        evaluate("(C) GATED+DAGhead", mC, gC, use_affinity=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

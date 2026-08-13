"""Gated-PC on the REALISTIC EHR-beta benchmark — does the topic-side gate help in the
hard regime? Toward the real AoU/Mondo run.

The clean-synthetic Gated-PC test (manual_gated_pc_case_finding) showed the gate transforms
case-finding when node topics are strongly planted. This asks the harder question on the
realistic generator (Mondo archetype planted on real cross-site LDA beta, Zipf background,
weak low-mass rare signal, semi-supervised): does gating the topic side beat the label-side
DAG head ALONE, on data whose difficulty mirrors real EHR?

Compares, per-node held-out AUC (co-fit head P(node) + node_affinity for the gated arms):
  (A) ungated PC + DAG head    — the current realistic best (label-side supervision only).
  (B) GATED PC + DAG head      — the full composition (topic-side gate + label-side head).
  (C) GATED PC + flat head     — gate + independent heads (closure-free).

Same head_l2=1e-3 (absolute, ADR 0041), newton head, K=20 both arms (ungated K_FIT=20;
gated n_bg=14 + 6 nodes = 20). Reuses the realistic generator.

Run:
  JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
  PYSPARK_PYTHON=<venv>/bin/python PYSPARK_DRIVER_PYTHON=<venv>/bin/python \
  PYTHONPATH=<repo>/spark-vi:<repo> <venv>/bin/python \
  spark-vi/tests/manual_gated_pc_realistic.py
"""
import importlib.util
import numpy as np

_spec = importlib.util.spec_from_file_location(
    "h", "spark-vi/tests/manual_pc_dag_case_finding_realistic.py")
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)

# topic-side gate layout: same DAG as the label side, K=20 (n_bg 14 + 6 nodes x 1)
GATE_PARENT = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
GATE_NBG, GATE_TPN = 14, 1
CLOSURE_PARENTS = [[], [0], [0], [1], [1], [2], [2]]


def frontier_of(y_row):
    """The planted frontier leaf = the LEAF whose closure-membership is set in y."""
    for l in h.LEAVES:
        if y_row[l] > 0.5:
            return frozenset({int(l)})
    return frozenset()


def to_gated_docs(X, Y, mask):
    from spark_vi.models.topic.types import GatedPCDocument
    docs = []
    for i, row in enumerate(X):
        idx = np.nonzero(row)[0].astype(np.int32)
        docs.append(GatedPCDocument(
            indices=idx, counts=row[idx].astype(np.float64), length=int(row[idx].sum()),
            y=Y[i].astype(np.float64), label_mask=mask[i].astype(np.float64),
            frontier=frontier_of(Y[i])))
    return docs


def fit(spark, docs, V, *, gated, head):
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models.topic.pc import OnlinePCLDA
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    from spark_vi.models.topic.dag_placement import DagLayout
    if gated:
        lay = DagLayout(GATE_PARENT, n_bg=GATE_NBG, tpn=GATE_TPN)
        K = lay.K
        engine = GatedOnlineLDA(lay, vocab_size=V, alpha=np.full(K, 1.0 / K),
                                eta=1.0 / K, random_seed=0)
    else:
        K, engine = h.K_FIT, None
    model = OnlinePCLDA(K=K, vocab_size=V, C=h.C, weight_y=h.WEIGHT_Y, alpha=1.05,
                        grad_cavi_iters=10, random_seed=0, head_optimizer="newton",
                        head_lr=0.7, weight_y_warmup_iters=10, head_l2=1e-3,
                        head=head, topic_engine=engine)
    cfg = VIConfig(max_iterations=h.MAX_ITERS, learning_rate_tau0=64.0,
                   learning_rate_kappa=0.6, random_seed=0, convergence_tol=1e-12)
    rdd = spark.sparkContext.parallelize(docs, numSlices=8).persist()
    rdd.count()
    gp = VIRunner(model, config=cfg).fit(rdd).global_params
    rdd.unpersist(blocking=False)
    return model, gp


def main():
    import pyspark
    from spark_vi.models.topic.pc import DagClosureHead, FlatLogisticHead, _predict_proba_np
    d = np.load(h.BETA_NPZ)
    beta, usage = d["beta"], d["usage_pct"]
    V = beta.shape[1]
    rng = np.random.default_rng(h.SEED)
    X, Y, mask = h.make_corpus(rng, beta, usage)
    n_te = int(h.TEST_FRAC * h.D)
    Xtr, Xte, Ytr, Yte, Mtr = X[:-n_te], X[-n_te:], Y[:-n_te], Y[-n_te:], mask[:-n_te]
    nodes = list(range(1, h.C))
    print(f"realistic Gated-PC: D={h.D} V={V} C={h.C} weight_y={h.WEIGHT_Y}  "
          f"rare=(5,6)  ungated K={h.K_FIT} gated K={GATE_NBG + 6}", flush=True)

    spark = (pyspark.sql.SparkSession.builder.master("local[4]")
             .appName("gated-pc-realistic").config("spark.ui.enabled", "false")
             .config("spark.sql.shuffle.partitions", "8").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    try:
        tr = to_gated_docs(Xtr, Ytr, Mtr)
        te = to_gated_docs(Xte, Yte, np.ones((len(Xte), h.C)))  # ungated fold-in at deploy

        def line(tag, model, gp):
            th = np.array([model.infer_local(dd, gp)["theta"] for dd in te])
            clo = getattr(model._head, "_closure_matrix", None)
            S = np.array([_predict_proba_np(t, gp["w_CK"], clo) for t in th])
            a = h.auc_table(Yte, S, nodes)
            m = h._mean(a, nodes)
            rare = np.mean([a[k] for k in (5, 6) if k in a]) if any(k in a for k in (5, 6)) else float("nan")
            print(f"  {tag:<22} head-AUC mean={m:.3f} rare(5,6)={rare:.3f}  "
                  + " ".join(f"{l}:{a.get(l, float('nan')):.2f}" for l in nodes), flush=True)

        mA, gA = fit(spark, tr, V, gated=False, head=DagClosureHead(CLOSURE_PARENTS))
        line("(A) ungated+DAGhead", mA, gA)
        mB, gB = fit(spark, tr, V, gated=True, head=DagClosureHead(CLOSURE_PARENTS))
        line("(B) GATED+DAGhead", mB, gB)
        mC, gC = fit(spark, tr, V, gated=True, head=FlatLogisticHead())
        line("(C) GATED+flathead", mC, gC)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

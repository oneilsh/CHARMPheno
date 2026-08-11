"""LOCAL (no-cluster) reproduction of the AoU 'head trained but MIS-DIRECTED' result.

On AoU the co-fit VI-PC head w_CK came out ~orthogonal (mean cos +0.08) to a batch LR
on the SAME final topics, while pc_topics_lr on those topics scored 0.61 (topics carry
signal; the head does not aim at it). Faithful serial head-SGD reproductions always
reached the LR direction, so the fault is something the JOINT distributed fit does
beyond head-SGD-on-final-topics. The passing toy test (C=1, FULL-BATCH, K_FIT<K_DOM)
does NOT show it. This script bisects the toy->AoU config axes with the REAL
OnlinePCLDA + VIRunner in a LOCAL SparkContext.

Run (poetry env, Spark):
  JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
  PYTHONPATH=<repo>:<repo>/spark-vi \
  poetry run python spark-vi/tests/manual_pc_head_direction_repro.py

Prints, per config: batch-LR heldout AUC on final theta (the pc_topics_lr analog),
the co-fit head heldout AUC, and mean cosine(w_CK[c], raw-theta LR coef on train).
mean cos ~1 => head aims right (toy-like); ~0 => head mis-directed (AoU-like).
"""
import numpy as np

SEED = 0
V = 300
K_DOM = 30           # structural (mostly label-irrelevant) topics on disjoint blocks
K_FIT = 30
D = 2500
DOC_LEN = 220
TEST_FRAC = 0.30


def make_corpus(seed, C=1):
    """K_DOM topics on disjoint word blocks; a WEAK predictive tilt from a small set
    of 'signal' topics -> logistic labels. LR on true theta is ~0.6-0.7 (recoverable
    but weak), mirroring AoU where pc_topics_lr=0.61."""
    rng = np.random.default_rng(seed)
    block = V // K_DOM
    beta = np.full((K_DOM, V), 1e-3)
    for k in range(K_DOM):
        beta[k, k * block:(k + 1) * block] += 1.0
    beta /= beta.sum(1, keepdims=True)
    alpha = np.full(K_DOM, 0.2)
    theta = rng.dirichlet(alpha, size=D)
    X = np.zeros((D, V))
    for i in range(D):
        wp = theta[i] @ beta
        X[i] = rng.multinomial(DOC_LEN, wp / wp.sum())
    # C weak predictive directions over the topic simplex (shared theta, C heads).
    Y = np.zeros((D, C))
    for c in range(C):
        w = rng.normal(size=K_DOM)
        w /= np.linalg.norm(w)
        logit = theta @ (w * 12.0)   # LR-on-true-theta ~0.68 (clean, AoU-like headroom)
        logit -= logit.mean()
        Y[:, c] = (rng.random(D) < 1 / (1 + np.exp(-logit))).astype(float)
    return X, Y


def labeled_docs(X, Y):
    from spark_vi.models.topic.types import PCDocument
    C = Y.shape[1]
    docs = []
    for i, row in enumerate(X):
        idx = np.nonzero(row)[0].astype(np.int32)
        docs.append(PCDocument(
            indices=idx, counts=row[idx].astype(np.float64),
            length=int(row[idx].sum()),
            y=Y[i].astype(np.float64), label_mask=np.ones(C),
        ))
    return docs


def run_config(spark, name, *, C, mini_batch_fraction, weight_y=1000.0,
               max_iters=40, grad_cavi_iters=20, head_optimizer="sgd",
               head_lr=0.05):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models.topic.pc import OnlinePCLDA

    X, Y = make_corpus(SEED, C=C)
    n_te = int(TEST_FRAC * D)
    Xtr, Xte, Ytr, Yte = X[:-n_te], X[-n_te:], Y[:-n_te], Y[-n_te:]
    docs = labeled_docs(Xtr, Ytr)

    model = OnlinePCLDA(K=K_FIT, vocab_size=V, C=C, weight_y=weight_y,
                        alpha=1.1, grad_cavi_iters=grad_cavi_iters, random_seed=0,
                        head_optimizer=head_optimizer, head_lr=head_lr)
    cfg = dict(max_iterations=max_iters, learning_rate_tau0=32.0,
               learning_rate_kappa=0.6, random_seed=0, convergence_tol=1e-12)
    if mini_batch_fraction is not None:
        cfg["mini_batch_fraction"] = mini_batch_fraction
    rdd = spark.sparkContext.parallelize(docs, numSlices=8).persist()
    rdd.count()
    result = VIRunner(model, config=VIConfig(**cfg)).fit(rdd)
    rdd.unpersist(blocking=False)
    gp = result.global_params

    tr = labeled_docs(Xtr, Ytr)
    te = labeled_docs(Xte, Yte)
    th_tr = np.array([model.infer_local(d, gp)["theta"] for d in tr])
    th_te = np.array([model.infer_local(d, gp)["theta"] for d in te])

    head_aucs, lr_aucs, cosines = [], [], []
    for c in range(C):
        w = gp["w_CK"][c]
        head_aucs.append(roc_auc_score(Yte[:, c], th_te @ w))
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(th_tr, Ytr[:, c])
        lr_aucs.append(roc_auc_score(Yte[:, c], lr.predict_proba(th_te)[:, 1]))
        v = lr.coef_[0]
        d = np.linalg.norm(v) * np.linalg.norm(w)
        cosines.append(v @ w / d if d > 0 else 0.0)
    print(f"[{name:<32}] C={C} mbf={mini_batch_fraction} opt={head_optimizer} "
          f"|w_CK|max={np.abs(gp['w_CK']).max():.3g}  "
          f"head-AUC={np.mean(head_aucs):.3f}  LR-on-theta={np.mean(lr_aucs):.3f}  "
          f"mean-cos={np.mean(cosines):+.3f}", flush=True)
    return np.mean(cosines)


def main():
    import pyspark
    spark = (pyspark.sql.SparkSession.builder
             .master("local[4]").appName("pc-head-repro")
             .config("spark.ui.enabled", "false")
             .config("spark.sql.shuffle.partitions", "8")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    try:
        # Harsh regime (C=10, mbf=0.05, 200 iters — the compounding AoU has that
        # 80-iter runs lack): does the RM-SGD head misdirect, and does Adam hold?
        print("SGD vs Adam head @ C=10, mbf=0.05, 200 iters (strong signal):\n")
        run_config(spark, "C=10 mbf=0.05 SGD", C=10, mini_batch_fraction=0.05,
                   max_iters=200, head_optimizer="sgd")
        for lr in (0.02, 0.05):
            run_config(spark, f"C=10 mbf=0.05 ADAM lr={lr}", C=10,
                       mini_batch_fraction=0.05, max_iters=200,
                       head_optimizer="adam", head_lr=lr)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

"""Head-L2 RECALIBRATION sweep — the fix the joint-vs-alternating de-risk pointed to.

manual_pc_joint_vs_alternating refuted "the gap is joint-vs-alternating": the faithful
reference's ALTERNATING fit reaches topics-LR 0.874 (== its joint 0.862) with |w|~105.
OnlinePCLDA's online alternating collapses to ~0.53 with |w|~4.76 — an ~840x = n_docs
OVER-regularization. Hughes's lambda_w=0.001 is a ridge on the SUMMED data gradient
(weight_y and /n_tokens cancel), so effective ridge ~ lambda_w. OnlinePCLDA applies
head_l2 PER-DOC, x n_docs (ridge = head_l2 * n_docs), so head_l2=1e-3 acts like Hughes's
lambda_w~0.84 — ~840x too strong. To MATCH Hughes: head_l2 ~ lambda_w / n_docs ~ 1e-6.

This sweeps head_l2 across that range on the ONLINE model (one-step Newton head). Expect
a sweet spot where |w| grows to ~50-100 (finite, not the 1e11 blowup at head_l2=0), the
co-fit HEAD becomes readable (~0.9), and topics-LR recovers toward the reference's ~0.87
— WITHOUT any joint step.

Run:
  JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
  PYSPARK_PYTHON=<venv>/bin/python PYSPARK_DRIVER_PYTHON=<venv>/bin/python \
  PYTHONPATH=<repo>/spark-vi:<repo> <venv>/bin/python \
  spark-vi/tests/manual_pc_head_l2_recalibration.py
"""
import importlib.util
import os
import numpy as np

_spec = importlib.util.spec_from_file_location(
    "h", "spark-vi/tests/manual_pc_dag_case_finding_realistic.py")
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)


def main():
    import pyspark
    from spark_vi.models.topic.pc import _predict_proba_np
    h.D = int(os.environ.get("EXP_D", "2500"))
    h.MAX_ITERS = int(os.environ.get("EXP_ITERS", "60"))
    d = np.load(h.BETA_NPZ)
    beta, usage = d["beta"], d["usage_pct"]
    V = beta.shape[1]
    rng = np.random.default_rng(h.SEED)
    X, Y, mask = h.make_corpus(rng, beta, usage)
    n_te = int(h.TEST_FRAC * h.D)
    Xtr, Xte, Ytr, Yte, Mtr = X[:-n_te], X[-n_te:], Y[:-n_te], Y[-n_te:], mask[:-n_te]
    nodes = list(range(1, h.C))
    n_tr = len(Xtr)
    print(f"D={h.D} n_train={n_tr} K={h.K_FIT} weight_y={h.WEIGHT_Y}  "
          f"Hughes-match head_l2 ~ lambda_w/n_docs = 1e-3/{n_tr} = {1e-3/n_tr:.2e}",
          flush=True)

    spark = (pyspark.sql.SparkSession.builder.master("local[4]")
             .appName("pc-head-l2-recal").config("spark.ui.enabled", "false")
             .config("spark.sql.shuffle.partitions", "8").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    try:
        tr = h.to_docs(Xtr, Ytr, Mtr)
        te = h.to_docs(Xte, Yte, np.ones((len(Xte), h.C)))
        print("\nhead_l2 sweep (flat one-step Newton head):  ref alternating=0.874 "
              "|w|~105", flush=True)
        for l2 in (0.0, 1e-6, 3e-6, 1e-5, 1e-4, 1e-3):
            mB, gpB = h.fit(spark, tr, V, weight_y=h.WEIGHT_Y, head=None, head_l2=l2)
            thB = h.thetas(mB, te, gpB)
            hd = h._mean(h.auc_table(Yte, np.array(
                [_predict_proba_np(t, gpB["w_CK"], None) for t in thB]), nodes), nodes)
            tl = h._mean(h.auc_table(Yte, h.posthoc_scores(mB, gpB, tr, te, Ytr, nodes,
                                                          mask=Mtr), nodes), nodes)
            wmax = float(np.abs(gpB["w_CK"]).max())
            print(f"  head_l2={l2:<7g}  HEAD={hd:.3f}  topics-LR={tl:.3f}  "
                  f"|w|max={wmax:.4g}", flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

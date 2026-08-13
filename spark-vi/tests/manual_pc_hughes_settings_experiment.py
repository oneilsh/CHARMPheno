"""Confound-isolation experiment (step 1 of the joint-vs-alternating question).

The realistic benchmark showed the co-fit head's SHAPED topics (topics-LR) plateau
at ~0.53, far below the faithful full-batch L-BFGS reference's 0.87. Before attributing
that gap to alternating-vs-joint optimization, close the two confounds the PRIMARY
sources (Hughes 1707.07341 §3.4 / 1712.00499 §3) reveal our OnlinePCLDA violates:

  * local-inference under-convergence — Hughes differentiates through a T≈100
    exponentiated-gradient MAP; OnlinePCLDA's supervised path unrolls only
    grad_cavi_iters=10. This is the θ the SHAPING gradient flows through.
  * weight_y scale — Hughes: "λ on the order of the number of tokens in the average
    document, though it may need to be much larger"; our docs ≈160 tokens but weight_y=20.

Grid: (grad_cavi_iters ∈ {10,100}) × (weight_y ∈ {20,160}), flat head, head_l2=1e-3
(blowup guard, Newton head UNCHANGED — this experiment does NOT touch the optimizer).
Reports per cell: co-fit HEAD AUC, topics-LR (shaping quality), |w|max. The ceiling is
the unsup post-hoc LR and the reference's topics-LR 0.868 / head 0.812.

If topics-LR climbs toward the reference as (π-iters, weight_y) -> Hughes's settings,
a chunk of the "joint gap" is really local-inference under-convergence, not jointness.

RESULT (D=1400, iters=50, pi_hi=50; unsup ceiling 0.483, reference 0.868) — BOTH
CONFOUNDS REFUTED:
    pi_iters=10  weight_y=20   HEAD=0.486  topics-LR=0.496
    pi_iters=50  weight_y=20   HEAD=0.486  topics-LR=0.496   (== pi=10, byte-identical)
    pi_iters=10  weight_y=160  HEAD=0.519  topics-LR=0.532
    pi_iters=50  weight_y=160  HEAD=0.518  topics-LR=0.524   (pi has no signal)
  * pi-iters: NO effect. Our CAVI is coordinate-ascent (fast); 10 iters is effectively
    converged where Hughes needs T~100 only because his exponentiated-gradient uses a
    tiny fixed step nu~0.005. Isolated check confirms theta DOES drift with n (||dtheta||
    0.073@10 -> 0.014@50 vs 100), but at head_l2=1e-3 the shaping correction
    (~ weight_y*|w|*dtheta) is too weak to move topics off the unsupervised solution, so
    the fit is insensitive to theta quality. The "10 vs 100" gap is a non-issue.
  * weight_y 20->160: small lift (+0.036 topics-LR), consistent with Hughes's "~tokens
    per doc" guidance beating 20, but nowhere near 0.868.
  => The gap to the reference survives closing both confounds. It is NOT local-inference
     under-convergence and NOT weight_y scale; it is the |w|-shaping-vs-L2 tension under
     ALTERNATING updates (shaping strength is dominated by |w|, which head_l2 throttles).
     Escaping it needs the reference's JOINT second-order step (L-BFGS over (topics,head)),
     which the online alternating natural-grad+Newton scheme does not reproduce. Motivates
     a joint step (note: the reference that reaches 0.868 is L-BFGS = Newton-family, not
     Adam — a joint fix need not reintroduce Adam).

Run:
  JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
  PYSPARK_PYTHON=<venv>/bin/python PYSPARK_DRIVER_PYTHON=<venv>/bin/python \
  PYTHONPATH=<repo>/spark-vi:<repo> <venv>/bin/python \
  spark-vi/tests/manual_pc_hughes_settings_experiment.py
"""
import importlib.util
import numpy as np

_spec = importlib.util.spec_from_file_location(
    "h", "spark-vi/tests/manual_pc_dag_case_finding_realistic.py")
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)


def main():
    import os
    import pyspark
    from spark_vi.models.topic.pc import _predict_proba_np
    # Reduced scale keeps the pi_iters=50 unrolled-autograd tape tractable while staying
    # trend-comparable; D=1400 also matches the reference's DMAX subset more closely.
    h.D = int(os.environ.get("EXP_D", "1400"))
    h.MAX_ITERS = int(os.environ.get("EXP_ITERS", "50"))
    PI_HI = int(os.environ.get("EXP_PI_HI", "50"))
    d = np.load(h.BETA_NPZ)
    beta, usage = d["beta"], d["usage_pct"]
    V = beta.shape[1]
    rng = np.random.default_rng(h.SEED)
    X, Y, mask = h.make_corpus(rng, beta, usage)
    n_te = int(h.TEST_FRAC * h.D)
    Xtr, Xte, Ytr, Yte, Mtr = X[:-n_te], X[-n_te:], Y[:-n_te], Y[-n_te:], mask[:-n_te]
    nodes = list(range(1, h.C))

    spark = (pyspark.sql.SparkSession.builder.master("local[4]")
             .appName("pc-hughes-settings").config("spark.ui.enabled", "false")
             .config("spark.sql.shuffle.partitions", "8").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    try:
        tr = h.to_docs(Xtr, Ytr, Mtr)
        te = h.to_docs(Xte, Yte, np.ones((len(Xte), h.C)))

        # ceilings
        m0, gp0 = h.fit(spark, tr, V, weight_y=0.0, head=None)
        unsup = h._mean(h.auc_table(Yte, h.posthoc_scores(m0, gp0, tr, te, Ytr, nodes,
                                                          mask=Mtr), nodes), nodes)
        print(f"\nceilings: unsup posthoc-LR={unsup:.3f}   "
              f"reference topics-LR=0.868 head=0.812 |w|=5.97", flush=True)
        print(f"grid — flat head, head_l2=1e-3, Newton, D={h.D} iters={h.MAX_ITERS} "
              f"(docs~160 tok):", flush=True)

        # (grad_cavi_iters, weight_y). Baseline, single knobs, both. PI_HI ~ Hughes's
        # T (reduced from 100 to keep the unrolled tape tractable; still 5x baseline).
        cells = [(10, 20.0), (PI_HI, 20.0), (10, 160.0), (PI_HI, 160.0)]
        for gci, wy in cells:
            mB, gpB = h.fit(spark, tr, V, weight_y=wy, head=None, head_l2=1e-3,
                            grad_cavi_iters=gci)
            thB = h.thetas(mB, te, gpB)
            hd = h._mean(h.auc_table(Yte, np.array(
                [_predict_proba_np(t, gpB["w_CK"], None) for t in thB]), nodes), nodes)
            tl = h._mean(h.auc_table(Yte, h.posthoc_scores(mB, gpB, tr, te, Ytr, nodes,
                                                          mask=Mtr), nodes), nodes)
            wmax = float(np.abs(gpB["w_CK"]).max())
            print(f"  pi_iters={gci:<3d} weight_y={wy:<5g}  HEAD={hd:.3f}  "
                  f"topics-LR={tl:.3f}  |w|max={wmax:.3g}", flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

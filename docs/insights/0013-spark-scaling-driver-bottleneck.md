# 0013 — Spark executor count past ~20 doesn't reduce iter time
**Date:** 2026-05-12
**Topic:** ops | svi
**Status:** Observed

Scaling Dataproc executors from ~10 to ~30 produced no meaningful
reduction in per-iter time on patient-year LDA/HDP runs. Adding more
workers wasn't bound by worker compute — it was bound by something
else.

The likely culprit is **driver-side aggregation**: each mini-batch's
sufficient statistics (λ updates summed across docs) get pulled to the
driver per iteration. With short docs and many of them, the per-batch
aggregation/serialization overhead is fixed regardless of executor
count, and at some executor count the gather phase dominates the
parallel E-step phase.

Confirming this would require profiling per-stage time — the
mini-batch E-step distributed phase vs the driver-side M-step phase —
which we haven't done yet. But the empirical observation stands:
throwing executors at the problem stops helping somewhere around 20.

**Implications.** Don't scale executor count past 20 for this workload
without first profiling. The lever to pull when iter time matters
isn't more workers, it's either (a) reducing per-iter driver-side
work (larger mini-batches, fewer total iters), or (b) shifting more
of the M-step into the distributed phase if the framework permits.

**Setting context.** Dataproc cluster, online LDA and HDP, AoU OMOP
patient-year condition-era docs. 4 vcores × 6 GB executors,
batch_fraction=0.1. Same observation across both models.

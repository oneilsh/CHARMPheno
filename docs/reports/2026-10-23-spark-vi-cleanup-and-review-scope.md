# spark-vi: cleanup scope and walkthrough timing (decisions, 2026-10-23)

**Status:** decided by the maintainer; work not started.

## Review state

The last maintainer walkthrough of spark-vi is **2026-08-12** (PC branch, 7 lessons).
The 2026-08-20 / 08-22 / 08-27 review-log entries are agent-recorded build notes, not
sign-offs, and are annotated as such. Since 2026-08-12, spark-vi's package code has had
53 commits and about 5,500 added lines (gross): multi-domain gated PC, the
supervised-head evolution, batched L-BFGS heads, runner broadcast/reduce changes, the
spectral init's scalable and guided forms, profile-η / count transform / alpha policy,
and the leaves-first gate schedule with its seeds. None of it has been walked through.

## Decisions

- **Cut the supervised-head machinery** (the `weight_y > 0` path: λ corrections, trust
  caps, the co-fit head and its provider seam, head-support variants used only by the
  co-fit). It never worked well, and the readout fits its heads post hoc. Removal gets
  an ADR; the post-hoc readout's batched L-BFGS is kept.
- **Keep the spectral init as an option**, even if the leaf seed (exp 0162) supersedes
  it as the default under the gate schedule. It has proved useful.
- **Keep the profile (HPO) seeds** as a comparator for the other seeding strategies.
- **No walkthrough yet.** It waits until the methods to keep are settled AND the user
  API is decided, including the split between the core (`spark_vi.models` /
  `spark_vi.core`) and the MLlib shim (`spark_vi.mllib`).

## Order of work

1. Settle the init question (exp 0162: leaf seed against spectral).
2. Cleanup: the supervised-head removal (ADR), plus anything else judged dead.
3. The user API: core against MLlib shim, and the two-engine design (Spark `VIRunner` and
   a local runner) with a home for the generic readout and evaluation pieces (heads over
   a label DAG, the closure product, gated NPMI, the identity census, the framings'
   generic cores); ADR + spec first.
4. The walkthrough of what remains, recorded in `docs/REVIEW_LOG.md` as sign-off.

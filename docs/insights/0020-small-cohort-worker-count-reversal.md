# 0020 — Below ~1k docs/worker/epoch, fewer Spark workers wins on wall time
**Date:** 2026-05-18
**Topic:** ops | svi
**Status:** Observed

Insight [0013](0013-spark-scaling-driver-bottleneck.md) noted that
per-iter wall time stops improving past ~20 executors on the full
patient-year corpus (the driver-side aggregation becomes the
bottleneck). On the dementia cohort (~9k docs at subsampling
rate 0.2 → ~1,800 docs/batch), we observed the symmetric phenomenon
at the *small* end: **dropping from 4 workers to 2 workers
*decreased* per-iter wall time.**

## Why

At 4 workers each iter splits 1,800 docs into ~450 docs/executor.
That's a handful of CountVectorizer sparse vectors and a small
λ-style update each — call it 50–100 ms of actual compute. The
fixed overhead per task (Spark task launch + python deserialize +
broadcast of the K×V variational params + reduce-tree of sufficient
statistics) is roughly constant in docs-per-task; it scales with
*number* of tasks. Once compute drops below ~100ms, that fixed
overhead exceeds the compute savings from parallelism and adding
workers strictly adds wall time.

## Rough heuristic

The two endpoints we now have:

| corpus            | docs/iter   | breakeven on this cluster                |
|-------------------|-------------|------------------------------------------|
| patient-year full | ~200k batch | adds value through ~20 executors ([0013](0013-spark-scaling-driver-bottleneck.md)) |
| dementia cohort   | ~1.8k batch | 2 workers > 4 workers                    |

Implied breakeven is around **a few hundred to ~1k docs per worker
per iter**. Cohort runs that filter the corpus down (cancer,
dementia, future small cohorts) likely all sit below that breakeven
on cluster shapes sized for the full run — worth right-sizing
executors per cohort instead of reusing the default.

## Implication

For cohort runs producing batches in the low-thousands, default to
2–4 workers, not the larger executor count that's optimal for the
patient-year full-corpus fits. Specifically: a cluster sized for
the patient-year run is *actively bad* for a 1–10k-doc cohort fit;
spinning down or reusing fewer slots is the cheaper and faster path.

**Setting context:** Online VI LDA, K=40, dementia cohort
(`first_dementia_year` ≈ 9k qualifying patients post-cohort filter),
condition_era doc-unit (one doc per patient's post-dx year),
subsampling_rate=0.2, full-corpus person_mod=1, τ_0=64, κ=0.7
(aggressive learning-rate schedule from the cohort Makefile
targets — gives ρ_t around 0.05 at iter 50 vs ~0.03 under the
defaults τ_0=1024 κ=0.51; doesn't affect per-iter compute or the
worker breakeven, but noted for reproducibility since the
observation was made on this specific schedule).
Observation made mid-run after restarting a driver crash from
4 workers down to 2; per-iter time at iter 50 with 2 workers came
in noticeably below the iter-10 / iter-47 cadence under 4.

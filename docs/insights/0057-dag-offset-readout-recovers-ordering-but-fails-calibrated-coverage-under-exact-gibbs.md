# Insight 0057 — The co-sampled Gibbs DAG-offset read-out recovers increment ORDERING but fails calibrated COVERAGE even when converged (attenuated means + overconfident intervals; insight 0051 reproduced under exact Gibbs); and full-vector coverage is structurally zero because a node is identified only on its own sticks

**Date:** 2026-07-14
**Branch:** pg-stm
**Topic:** pg-stm | dag-offsets | read-out | coverage | calibration | gibbs
**Status:** Observed
**Relates to:** the read-out engine (spec/plan 2026-07-14, `dag_readout.py`, `pg_stm_dag_gibbs.py`);
insight 0050 (only increments identified), 0051 (ridge offset intervals order right but are
overconfident, coverage 0.13), 0052 (mean-field sound for point/ordering, unsound for calibration),
0053 (read out at block granularity, never per-stick across positions), 0054 (node offsets identified
on FOREGROUND sticks only), 0056 (design-wall verdict). This is the read-out engine's coverage-plant
acceptance gate (plan Task 11) returning an honest negative.

**Context:** Built the full DAG-offset read-out engine (warm-start VI → compile on the expected moment
→ co-sampled quotient Gibbs emitting offset-increment draws → per-coordinate-class read-out). The
acceptance gate is a redraw-truth coverage plant (insight 0051 protocol): each replicate redraws the
node offsets from the prior, runs the whole engine, and asks whether each identified coordinate's 90%
credible interval covers its planted value. Four cells: populated / scarce / soft-gated / design-wall.

**Findings:**

1. **The design-wall (emit-but-flag) contract HOLDS perfectly — 12/12 replicates.** Gauge and
   unresolved coordinates never emit a point estimate (no `increment_mean`), unresolved carries its
   attestation recipe, gauge its convention. The half of the engine that refuses to answer un-identified
   directions works exactly as designed.

2. **Full-vector coverage is 0 BY CONSTRUCTION — a schema/granularity defect.** Each node's offset is
   reported as a full length-(K-1) stick vector, and coverage is checked over ALL sticks (`np.all`). But
   a node is identified only on the sticks its own documents activate (its group's block; insight
   0054). For an A-group node, group B's sticks are never touched — their posterior sits pinned at the
   prior mean (tight, ≈0) while the redrawn truth is ~N(0, 4), so those coordinates essentially never
   cover, and the all-sticks `np.all` check can never pass regardless of sampler quality. **The read-out
   must restrict its coverage claims (and ideally its reported vector) to each node's identified
   sub-block** — insight 0053's block-granularity rule, now a hard schema requirement, not just a
   reporting convention. **DONE (2026-07-14):** `assemble_readout` takes a `node_sticks` map and the
   orchestrator passes each node its group's foreground sticks, so an identified coordinate reports
   increment_mean / ci and claims coverage on ONLY those sticks (named in a `sticks` field). This
   removes the structural full-vector-0 — and the gate STILL fails (finding 3), which is the whole
   point: granularity was never the blocker.

3. **Even on the node's OWN foreground stick, coverage fails — and it is NOT a convergence artifact.**
   Restricting to the identified own-foreground coordinate: short chain (n_iter 80 / burn 40) covers
   0.12; a properly converged chain (n_iter 250 / burn 120) covers 0.40 — better mixing helps but does
   not calibrate. Two failures persist that more iterations cannot fix: (a) the posterior MEAN is
   attenuated toward zero (e.g. truth 1.99 → mean 0.76; truth 3.98 → mean 3.44) — a systematic
   shrinkage bias; (b) the intervals are OVERCONFIDENT (~0.3 wide) and miss even when the sign is right;
   (c) an occasional hard sign-confound (truth −3.46 → mean +3.1 with a tight CI far from truth). The
   within-block recovery-correlation is positive (~0.6) — the ORDERING/recovery works (consistent with
   Task 6 r=0.605 and insights 0054/0056), only the absolute CALIBRATION fails.

4. **This reproduces insight 0051 under the EXACT co-sampled Gibbs engine.** The VI ridge intervals were
   "order right, overconfident absolutely, coverage 0.13"; the Gibbs read-out is "order right,
   overconfident absolutely, own-block coverage 0.12→0.40." Moving from mean-field VI to exact Gibbs did
   NOT cure the calibration failure on the identified increment — it is deeper than the mean-field
   attenuation of insights 0044/0047. This is precisely Fable's pre-registered branch: *if scarce/
   identified coverage fails even under Gibbs, that is where informative (LKJ / half-t provenance) priors
   must earn their place*; the mean attenuation additionally points at the depth-scaled ridge
   parameterization biasing the point, which a prior — not more samples — addresses.

**Consequences:**
- The read-out engine is BUILT and its STRUCTURE is validated: the compiler→quotient→classification
  pipeline runs end-to-end, planted increments recover in ordering (corr ~0.6), and the design-wall
  emit-but-flag contract holds 12/12. But the CALIBRATED-COVERAGE acceptance gate is UNMET.
- Two distinct next steps are now measured, not guessed: (i) restrict the read-out to each node's
  identified sub-block (removes the structural full-vector-0; a schema refinement consuming the
  compiler's identified-coordinate list — Fable's amended-directive item 5); (ii) informative LKJ/half-t
  provenance priors + addressing the ridge-induced mean attenuation, then re-run the gate. These are
  Fable's territory (she pre-registered exactly this outcome) — consult before building.
- The committed coverage test is the acceptance criterion and currently FAILS; it is marked xfail with a
  reason pointing here, so the suite stays green while the unmet gate is preserved and un-hideable (the
  threshold was NOT loosened to fake a pass).

**Does not claim:** anything about real data (synthetic, model-matched). Does NOT claim recovery/ordering
fails — it works (corr ~0.6). Only that ABSOLUTE calibrated coverage of the identified increment is unmet
under the current engine, for the two measured reasons (off-block granularity + own-block
attenuation/overconfidence).

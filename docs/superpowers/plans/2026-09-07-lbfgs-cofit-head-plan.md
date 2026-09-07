# Plan: scalable strong co-fit head — matrix-free batched L-BFGS (exp 0120)

**Date:** 2026-09-07 · **Status:** approved, ready for agent execution
**Feeds from:** exps 0118 (Newton head O(C·K²) collect OOMs at K=1498) and 0119
(SGD head too weak, corr_relΔλ≈6e-6) — both stock solvers fail in opposite ways;
the 2026-08-20 PC closeout §6 prescribed a scalable STRONG head, and the
2026-08-20 distributed-readout plan already frames the co-fit head as "the same
[batched-L-BFGS treeAggregate] machinery warm-started and amortized."

## Goal

Give the gated-PC co-fit head a **matrix-free amortized L-BFGS** solver
(`head_optimizer: "lbfgs"`) that is O(C·K)-shuffle and Hessian-free — right-sized
shaping like Newton (targets corr_relΔλ in the healthy ~0.05 band) without the
O(C·K²) driver collect that OOMs at scale. Then re-run 0118's config with the
lbfgs head as **exp 0120** — the fair PC test the arc has been unable to run.

**The reuse is the point:** the solver (`analysis/pc/batched_lr.py::
solve_batched_lr` — per-node Armijo, node freezing, `fold_standardization`) and
the O(C·K) distributed stats seam (`distributed_readout.py::make_spark_stats_fn`,
ADR-0047-clean) are already built, tested, and in-tree. This is wiring + one
genuinely new problem (below), not a from-scratch solver.

## The one real risk (gate the whole build on it)

`solve_batched_lr` was built for the POST-HOC readout on **frozen** θ. The co-fit
head runs it against **moving** θ (θ drifts every SVI outer iter) — non-stationary
curvature the L-BFGS history has never faced. Open questions: inner L-BFGS passes
per outer iter, whether to reset the history each outer iter, and whether the EG
`head_trust_move` cap alone contains over-drive.

**This is validated on the LOCAL simulator BEFORE any cluster run** — the answer to
this arc's three cluster misses (0118 OOM, 0119a over-drive misread, 0119b too
weak). No cluster minute until the coupling is green locally: corr_relΔλ lands in
~0.02–0.05, no over-drive, and the head's per-node heldout AUC tracks the sklearn
oracle on simulator data.

## Normative design decisions (pre-registered)

- **D1 — one source of truth.** Lift the pure-numpy `solve_batched_lr` (+
  `fold_standardization` and its helpers) into spark-vi as the canonical module
  (`spark_vi/models/topic/batched_lr.py`, pure-Python/flat, no charmpheno/analysis
  import). Repoint `analysis/pc` / `distributed_readout` consumers to import it
  from spark-vi (analysis→spark_vi is the LEGAL dependency direction). The build
  MUST prove the readout is byte-identical after the repoint (its existing suites
  green, unchanged numbers). Two diverging copies of a numerically-critical solver
  is the hazard this avoids.
- **D2 — the seam.** The head is a two-method split in
  `spark_vi/models/topic/pc.py` (`OnlinePCLDA`): `local_update` (executor E-step,
  emits per-partition numpy stats — under `lbfgs` emit ONLY the gradient stats
  already emitted for sgd, NO Hessian) and `update_global` (driver M-step —
  branch on `head_optimizer`, replace the per-node Newton solve with an amortized
  batched-L-BFGS step over `self._topic_support[c]`, warm-started from current
  `w_CK`/`b_CK`, run `head_inner_iters` passes). The EG mass-preserving
  λ-correction and the `head_trust_move` trust cap stay UNCHANGED downstream —
  the lbfgs head only changes how w_CK is updated. ADR 0047: the head object never
  crosses a partition boundary; only unboxed numpy stats do (already true).
- **D3 — knobs.** New `head_inner_iters` (default a few, e.g. 3 — amortized
  passes per outer iter); reuse `head_l2`, `head_lr` (Armijo/step scale as the
  solver defines), `head_trust_move` (mandatory pairing — bigger heads over-drove
  to corr 0.72 without it), `head_standardize`, the localized `head_support`.
  Whether to reset L-BFGS history per outer iter is a knob decided by D-risk's
  simulator result (`head_history_reset` bool, default per what wins locally).
- **D4 — allowlist.** Add `"lbfgs"` to the head_optimizer allowlist at
  pc.py:~774, the mllib `headOptimizer` Param doc, and the gated_pc_cloud argparse
  `choices=`. `sgd`/`newton` paths stay byte-identical.

## Work packages (each: implement + tests + the gate)

- **WP0 — micro-scout + solver lift (D1).** Confirm the exact `solve_batched_lr`
  interface and every consumer/import site; lift it to spark-vi canonical; repoint
  consumers; prove readout byte-identity. Write the coupling micro-design as a
  comment block in the WP1 commit.
- **WP1 — spark-vi engine (D2, the seam + the risk).** The `lbfgs` branch in
  `local_update` (grad-only stats) and `update_global` (amortized batched-L-BFGS
  over the support, warm-started). Tests: (a) BYTE-IDENTITY when
  `head_optimizer∈{sgd,newton}` (existing stats/paths untouched); (b) **the
  coupling prototype on the local simulator** — corr_relΔλ reaches the healthy
  band, no over-drive under the trust cap, per-node heldout AUC tracks the sklearn
  oracle. WP1 is not done until (b) is green.
- **WP2 — mllib wiring.** `headOptimizer` Param doc lists `lbfgs`; thread
  `headInnerIters` (+ `headHistoryReset` if D3 keeps it) through the
  estimator→engine shim. Mirror the head_trust_move wiring pattern.
- **WP3 — driver + emission.** `lbfgs` in gated_pc_cloud argparse `choices`;
  `--head-inner-iters` (mirror `--head-trust-move`: argparse → estimator → manifest);
  emission in run_experiment `build_gated_pc_args`. **--py-files:** ensure the
  module carrying any executor-side L-BFGS kernel is on `--py-files` in every
  submit path (the batched-LR stats run executor-side).
- **WP4 — exp 0120.** Doc = 0118 front matter with `head_optimizer: lbfgs` (+
  head_inner_iters), profile-eta S=1.0, head_trust_move 0.03, weight_y 2.0/warmup.
  Fit + readout + `readout-ab ID=120 BASE=113` and `BASE=116` + `--profile-align`.
  ONLY after WP1(b) is green locally.

## Validation gate (every WP)

`PYTHONPATH=spark-vi poetry run pytest <new/touched suites>
tests/scripts/test_case_finding_cache_mondo.py -p no:randomly -q` all green — the
60-test tripwire byte-identical (this work touches only driver-owned + spark-vi
files; NO source-hashed module — cohorts/multi_domain/case_finding_assembly,
mondo_*, condition_dag, preindex_closure). `make test` in spark-vi. The readout
suites green and unchanged after the D1 repoint. No model names in artifacts.
Per-step timings "X.Xs", never cumulative "elapsed".

## Pre-registered acceptance for 0120 (decided BEFORE the run)

1. **Fit-health (local sim first, then cluster):** corr_relΔλ in ~0.02–0.05,
   grad_y SHRINKING (head converging, not chasing), ELBO stable, η_boost sentinel
   present, no O(C·K²) collect / OOM, per-iter time O(C·K) not O(C·K²).
2. **Primary (the fair PC test):** `readout-ab` — credited paired dAUC up vs BOTH
   0113 (flat) and 0116 (profile-eta, no shaping); macro ≥ 0.7813 − noise floor.
   The co-fit head arm's AUC is now a FAIR read against the 0.758 revival bar
   (a right-sized head, unlike SGD).
3. **Failure reads:** corr healthy but readout flat/down → controlled shaping does
   not help even with a strong aligned-target head → PC closes on the merits (write
   the closing insight; the record then points at representation — episode index /
   cascade); corr won't reach the band even with L-BFGS → the coupling is the
   blocker, revisit inner-iters/history-reset; over-drive under the cap → the EG
   cap needs the move-radius lowered.

## Out of scope (parked)

MI-selected dynamic bounded-support Newton (the other closeout option — zero
existing code, capped by support; only if L-BFGS underperforms). Whole-Mondo
scale-up of the lbfgs head (K≈3,800) — 0120 is the CV-branch proof first.

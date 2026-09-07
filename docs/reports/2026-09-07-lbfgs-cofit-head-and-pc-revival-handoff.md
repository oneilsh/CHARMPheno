# Handoff: L-BFGS co-fit head + PC-revival arc (2026-09-07)

Session handoff written before a context compaction. Branch:
`claude/gated-conditional-voi`. Everything below is committed and pushed unless
marked LIVE/pending. Read `docs/experiments/0120-*.md` and
`docs/superpowers/plans/2026-09-07-lbfgs-cofit-head-plan.md` alongside this.

## TL;DR — where things stand

The **matrix-free L-BFGS co-fit head is built, tested, and RUNNING on the
cluster** as exp 0120 — the fair Prediction-Constrained (PC) test the whole arc
has been building toward. As of last check it was ~iter 28/50, **healthy and
converging** (see "Reading the fit log"). The engineering goal is met; the open
question is the science one: **does controlled label-side shaping help
case-finding AUC** (pre-registered decision below). The remaining risk to the
run is executor OOM, not correctness.

## Standing constraints (do not violate — carried across compaction)

- Work ONLY on `claude/gated-conditional-voi`. Commit trailer:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` +
  `Claude-Session: https://claude.ai/code/session_01CWHmrp8SXQU127fTYksP1c`.
- NEVER commit patient-level data. Egress floor: any cell < 20 not disclosable;
  committed reports/log output carry pooled figures + counts-of-nodes only.
- NO AI model identifiers in committed content (chat only).
- Source-hashed modules NEVER edited (importing fine); the 60-test tripwire
  `tests/scripts/test_case_finding_cache_mondo.py` must stay byte-identical:
  `charmpheno/omop/{cohorts,multi_domain,case_finding_assembly}.py`,
  `analysis/cloud/{mondo_dag,mondo_native_dag,mondo_usage_core,mondo_collapse,
  mondo_to_omop_mapping,condition_dag,preindex_closure}.py`.
- spark-vi stays pure-Python/flat and NEVER imports charmpheno or analysis
  (dependency direction is analysis → spark_vi only).
- ADR 0047: nothing array-shaped or Spark-capturing rides a task closure;
  explicit `sc.broadcast`, None-sentinel treeAggregate zeros. (This arc's last
  crash was exactly an ADR-0047 violation — see below.)
- Every cluster command carries the preamble:
  `cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only`.
- User prefs: per-step timings ("X.Xs", never cumulative "elapsed"); plain-text
  chat (no LaTeX — λ, θ, ‖·‖ ASCII); judicious main-chat context (delegate heavy
  builds to background subagents, narrow reads); one step at a time; precision +
  owning wrong calls rewarded.

## The PC-revival arc (why we're here)

The eta-prior arc (insights 0083–0085) established that the HPO-profile word-side
prior buys topic-profile ALIGNMENT but **zero case-finding AUC** — the ceiling
(~0.78 macro on the CV branch) is not about how θ is *weighted* by a word-side
prior. Insight 0085's remaining θ-allocation lever was **PC** (a label-side pull
on λ), whose 2026-08-20 closeout parked it behind "co-fit head ≥ ~0.758" and
named the enabling fix. The arc:

- **exp 0118** (Newton head, `head_optimizer: newton`): ABANDONED. The per-node
  Hessian collect is O(C·K²); at K=1498 it OOM'd the 8g driver, steady 1450s/iter
  — the exact wall the closeout named. The MODEL was healthy (corr 0.05); the
  solver doesn't scale.
- **exp 0119** (SGD head): ABANDONED. Scales (O(C·K)) but too weak — corr_relΔλ
  ~6e-6, non-converging, grad_y climbing (chasing moving θ). Also surfaced a
  needed knob: `head_trust_move` (the scale-free trust cap) was in the engine but
  unwired — wired in commit 4392f24. Cranking head_lr_scale to compensate is the
  fragile per-scale hunt the closeout warned against.
- Conclusion: both stock solvers fail in opposite ways (Newton right-sized but
  doesn't scale; SGD scales but too weak). The closeout's prescribed fix is the
  **matrix-free amortized L-BFGS head** — curvature-aware like Newton, O(C·K) like
  SGD. User approved building it.
- **exp 0120** (L-BFGS head): the build (below), now LIVE on the cluster.

## What was built (the L-BFGS co-fit head)

Plan: `docs/superpowers/plans/2026-09-07-lbfgs-cofit-head-plan.md`. Key finding:
the branch already descends from the stash, so the matrix-free solver was
**already in-tree, tested** — this was reuse + wiring, not a from-scratch build.

- **32274fb (WP0/WP1):** lifted the pure-numpy `solve_batched_lr` (+
  `fold_standardization`) to canonical `spark_vi/models/topic/batched_lr.py`;
  `analysis/pc/batched_lr.py` is now a re-export shim (readout proven
  byte-identical). Added `head_optimizer="lbfgs"` to `OnlinePCLDA` (spark-vi
  `models/topic/pc.py`): `local_update` emits grad-only stats (Hessian gated on
  newton — lbfgs structurally cannot hit the O(C·K²) collect); `update_global`'s
  lbfgs branch (`_lbfgs_head_step`) runs an amortized batched-L-BFGS step over the
  localized support, warm-started, `head_inner_iters` passes; the EG λ-correction
  + `head_trust_move` cap downstream are UNCHANGED (lbfgs changes only how w_CK is
  learned). Knobs: `head_inner_iters` (default 3), `head_history_reset` (default
  True) — both sim-decided (insight 0086). **Coupling proof on the local
  simulator (the gate before cluster):** corr climbs to the cap then decays,
  grad_y shrinks 5.2→0.12, tracks the sklearn oracle within 0.005.
- **a826528 (WP2/WP3):** the DISTRIBUTED re-scoring provider in
  `analysis/cloud/distributed_readout.py` (`_cofit_theta_kernel` reproduces the
  head's label-free `_plain_cavi_theta` byte-for-byte — the PC faithfulness θ, NOT
  the readout transform's CAVI; `score_cofit_theta_df`;
  `make_cofit_head_stats_provider`). Scores θ once per outer iter (persisted),
  reuses `make_spark_stats_fn`'s O(C·K) treeAggregate for the inner L-BFGS.
  Injected driver-side via `OnlinePCLDAEstimator.setHeadStatsProviderFactory`
  (engine stays Spark-free), built in `gated_pc_cloud.py`'s pc-arm fit. Proven to
  match the in-memory provider on single- AND multi-domain (rtol 1e-9). Wiring:
  mllib Params + `gated_pc_cloud` argparse (`--head-inner-iters`,
  `--head-history-reset`) + run_experiment emission (mirrors `--head-trust-move`).
- **5415845 (the crash fix):** exp 0120's FIRST cluster attempt crashed ~22s in,
  before iter 1, with `[CONTEXT_ONLY_VALID_ON_DRIVER]` (SPARK-5063). Root cause:
  VIRunner ships the model into every E-step task closure (`_model=model`); the
  injected distributed provider closes over the SparkContext, so cloudpickling the
  model with it attached raised. The invariant was documented on
  `set_head_stats_provider` but never enforced. Fix: `OnlinePCLDA.__getstate__`
  strips `_head_stats_provider` from pickled copies (mirrors
  `GatedOnlineLDA.__getstate__`'s eta-boost exclusion) — driver instance keeps it
  (update_global consumes it there), executor copies get None. Regression test
  round-trips a provider-bearing model through cloudpickle. This was missed
  locally because the coupling test used the IN-MEMORY provider (picklable rows,
  no SparkContext).

**Pattern that bit us twice on the cluster** (both now fixed): failures came from
Spark-integration edges local small-fixture tests structurally can't reach (the
Newton head's real-scale cost; the provider's SparkContext capture). If a THIRD
surfaces, the plan says: stand up a local-Spark end-to-end fit of the whole lbfgs
path (real multi-partition, real provider) as a gate before any further cluster
spend, rather than patching reactively.

## Reading the fit log (diagnostics glossary — for the LIVE 0120 run)

- **corr_relΔλ** = ‖λ_sup − λ_unsup‖/‖λ_unsup‖: the relative size of the
  supervised head's correction to the topics per SVI step. `head_trust_move=0.03`
  CAPS it. Healthy = pinned at ~0.03 (cap governing = max allowed shaping every
  iter). It's the scale-free (K-invariant) shaping diagnostic — watch THIS, not
  raw weight_y. The 0.03 is the empirically-safe band ceiling (0098/0099: >1
  detonates; <0.01 does nothing), calibrated to the 50-iter/rho schedule — a
  guardrail, not a magic optimum.
- **grad_y**: the head-gradient inf-norm. **BIG ABSOLUTE VALUES ARE COSMETIC**
  under `head_standardize` (the 1/σ amplification — the Newton head hit 1.9e8 /
  |w_CK| 2.9e5 too). Judge by TREND: healthy = shrinks then plateaus. On 0120 it
  went 4.5e5 (it2) → 6.1e4 (it25) → plateau ~6e4 — the head converged then reached
  a moving equilibrium (θ still evolving at K=1498, so the head tracks it at the
  cap-limited rate; the sim decayed to rest only because small-K θ stops moving).
  A SUSTAINED climb (0119: 988→10k) is the failure mode; a flat plateau with noise
  is fine.
- **ELBO** rising (0120: −125.7M → −91.4M through it28), no −1e27 detonation = the
  EG mass-preserving correction holding.
- **η_boost[topics=132 nnz=28883 mass=440.1]**: the profile-eta prior sentinel
  (must be present — it's the aligned target the pull shapes toward).

## What to do next — by scenario

### If 0120 FINISHES (npz saved):
```
make -C analysis/cloud gated-pc-readout ID=120 GPR_ARGS="--readout-mode distributed"
make -C analysis/cloud readout-ab ID=120 BASE=113
make -C analysis/cloud readout-ab ID=120 BASE=116
make -C analysis/cloud inspect-topics ID=120 CREDITED=1 COMPARE=116 INSPECT_ARGS="--profile-align"
make -C analysis/cloud inspect-topics ID=120 CREDITED=1 RESOLVE_NAMES=1 INSPECT_ARGS="--digest --grep 'cardiomyopath'"
```
Then fill 0120 Results + write the closing insight, per the pre-registered
decision:
- **Credited paired dAUC UP vs BOTH 0113 (flat) and 0116 (profile-eta,
  no-shaping), macro ≥ 0.7813 − noise → PC REVIVED.** The co-fit head arm's AUC is
  now a fair read against the 0.758 revival bar.
- **corr was healthy but readout FLAT/DOWN → PC CLOSES ON THE MERITS.** Controlled
  shaping with a strong, aligned-target head didn't help → the record then points
  at REPRESENTATION as the frontier: the episode index (0111/0112 machinery, built
  and waiting) or the 0071 cascade. Write that insight; it's a clean, important
  negative, not a failure.

### If 0120 OOM-ABORTS (executor JVM OOM, exit 52 — seen once at iter 25, recovered):
It's memory pressure (the co-fit head's persisted scored-θ df on top of the cached
corpus, on 8g × only 8 workers), NOT a code leak (the provider's persist/unpersist
lifecycle is designed bounded — verified by read). Hardened re-run config:
- Full cluster: 20 workers (the `spark_conf` already requests it; user was running
  8). Spreads the cache (less heap/executor) + adds cores (also fixes the ~12
  min/iter — it's compute-bound, not IO-bound, with a tiny O(C·K)=3.6MB shuffle).
- `spark.executor.memory: 8g → 16g` (the OOM was JVM heap).
- `head_inner_iters: 3 → 1` (smaller working set + faster; sim said inner_iters
  1/3/5 are quality-equivalent).
- Optional due diligence: audit the scored-θ df unpersist for a
  blocking-unpersist. No smoking gun expected.
- No checkpoint/resume today (`--resume-from` is a no-op), so an abort loses all
  iters — the durable fix if this recurs is wiring resume, parked.

## Broader map (context that survives compaction)

- **Insights of record this session:** 0083 (spectral init costs case-finding),
  0084 (profile-eta S=1 null but legible), 0085 (eta dose buys alignment not AUC —
  ladder stopped, S=1 is the operating point), 0086 (lbfgs coupling passes the
  sim gate). Index in `docs/insights/README.md`.
- **Tooling added:** `inspect_topics --readout-auc` with `--credited-file` /
  `--compare-run` (paired credited-split readout — the primary PC read); the
  `readout-ab ID=N BASE=M` Makefile target + auto bundle-key discovery +
  `COMPARE=`/`CREDITED=` vars (91ab4af, e8c67e8); `--profile-align` scorecard +
  HPO-profile `*` markers in the digest (b88901e); ANSI color in driver logs
  (`term_colors`, 9fef90d — `CHARM_COLOR=0` disables).
- **profile-eta pipeline:** `hpoa_profile_survey.py` (stage-1 offline),
  `hpoa_stage2_probe.py --emit-eta` (the prior TSV), `profile_eta.py` (fit-side
  builder). TSV regenerated via `make -C analysis/cloud hpoa-stage2-probe ID=114
  GPR_ARGS="--emit-eta ~/repos/CHARMPheno/data/ontology/profile_eta_MONDO_0004995.tsv"`
  (probe needs a bundle-cache HIT; workspace-internal, never committed).

## Parked / out of scope

MI-selected dynamic bounded-support Newton head (the other closeout option — zero
existing code; only if L-BFGS underperforms). Whole-Mondo scale-up of the lbfgs
head (0120 is the CV-branch proof first). Checkpoint/resume for gated_pc. GCS
cache persistence (bucket name must come from the user). Episode-index frontier
(the alternative if PC closes).

## Commits of record (this arc, newest first)

5415845 (ADR-0047 provider closure fix) · f8f453b (0120 doc) · a826528 (WP2/WP3
provider+wiring) · 32274fb (WP0/WP1 seam+lift) · 0c8e6da (plan) · 7e4fe7e (0119
abandoned) · 4392f24 (head_trust_move wiring) · 8366792 (0118 abandoned + 0119) ·
e878ee2 (0118 doc) · e11cde6 (0117 + insight 0085).

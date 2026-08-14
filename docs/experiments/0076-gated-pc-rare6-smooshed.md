---
id: 76
slug: gated-pc-rare6-smooshed
status: pending
model_class: gated_pc
cohort: population_rare6
cohort_def: population_rare6
disease: rare6
# --- corpus / DAG (mirrors the rare6 dag_placement incumbent 0065 so the
#     pc_topics_lr comparison sits in the same regime) ---
person_mod: 1
prior_obs_days: 0
doc_min_length: 10
min_n: 20
holdout_frac: 0.2
vocab_size: 5000
min_df: 20
min_patient_count: 20
window_mode: lookback
lookback_days: 1825      # 5yr feature history (run 3): run 2's 1yr window starved
                         # K=170 (Σλ_k min ~31). Richer BOW → more mass to support
                         # the node topics + a fairer hidden-low-mass test.
label_window_days: 365
strip_mode: both
# per-node label observation policy (Step-A adapter). 'full' = every node
# observed, background is a true negative everywhere. See frontier_to_label.
label_mask_mode: full
# --- gate topic-block layout (same as 0065). K is emergent = n_bg + nodes*tpn,
#     shared by BOTH the gated_pc and unsup_gated arms (a fair internal A/B) ---
n_bg: 40
tpn: 5
optimize_doc_concentration: true
# --- PC head (inject the hierarchy ONCE: gate + FLAT head; ADR 0042) ----------
weight_y: 50.0            # PC prediction weight. Hughes ~ tokens/doc; rare6 1yr
                         # windows are short, so 50 is a starting point — TUNE on
                         # the delta vs unsup_gated (try 100/200 if the head is
                         # under-moved, i.e. |w_CK| still climbing at max_iter).
head_optimizer: newton   # settled convergent head (ADR 0039); no sgd/adam.
head_lr: 0.3             # DAMPED (insight 0075): 0.5 oscillated (|w_CK| bounced
                         # 19→7→14→60 on run 1). 0.3 makes the head an EMA of the
                         # per-minibatch optima → converges to their mean.
head_newton_ridge: 0.05  # 0.01→0.05: regularize the near-singular per-minibatch
                         # H_c that spikes the IRLS solve (only stabilizes it;
                         # AUC is scale-invariant to head magnitude).
head_l2: 0.01            # ABSOLUTE ridge = Hughes lambda_w (ADR 0041). Run 2 used
                         # 1e-3 and the head BLEW UP at iter ~151 (|w_CK| 45→1.25e6
                         # in 5 iters) once the converged gated topics went ~separable
                         # — 1e-3 is too weak to bound the singular-minibatch IRLS
                         # step there. 1e-2 (strong end of the recalibration's good
                         # basin ~1e-4..1e-2) caps |w| ~10x tighter; combined with the
                         # 100-iter cap (blow-up hit at 151) this should stay bounded.
grad_cavi_iters: 30      # differentiable CAVI unroll depth; must match scoring
                         # convergence (cavi_max_iter=100). 30 suffices for the
                         # short lookback docs (deeper = bigger autograd tape).
topic_trust: 0.05        # 0.1→0.05: run 1's Σλ_k max blew up 4.6e4→6.2e5 in 11
                         # iters (one topic hoovering mass) — the loose correction
                         # driven by the oscillating head. Tighter trust keeps the
                         # supervised topics near the unsup warm-start.
weight_y_warmup_iters: 25  # 10→25: run 1 spiked at iter 11 the moment full
                         # weight_y engaged; a longer ramp softens the onset.
# --- SVI schedule (comparable to 0065 / the pc-vi runs) -----------------------
max_iter: 100            # run 3: ~95% of the signal by 100 iters (run 2's ELBO
                         # was well into diminishing returns); halves wall-clock.
subsampling_rate: 0.1
tau0: 64.0               # RM offset; on smaller cohorts ~10-64 so the head moves.
kappa: 0.51
cavi_max_iter: 100
cavi_tol: 0.001
# --- arms + eval --------------------------------------------------------------
# unsup_gated (weightY=0 twin) runs by default as the controlled incumbent.
# with_dag_head (ungated + DAG-closure head) is OFF for this first run — it is
# the label-side alternative to the gate, added once the gate+flat baseline reads.
skip_unsup_gated: false
with_dag_head: false
baseline_max_iter: 100   # unsup topics converge faster than the head.
min_label_count: 20      # AoU small-cell floor: mask rare nodes from the macro.
# Parallelism: the cached bundle parquet has ~8 part-files, so dynamic allocation
# pins the fit to ~2 executors (8 tasks / 4 cores) while the cluster sits idle and
# the per-doc autograd serializes. Repartition ≈ total cluster executor cores so
# the fit demands + spreads across more executors. This ~14-worker × 4-core cluster
# has ~56 cores; 96 gives headroom + load balance. Pair with
# CHARM_SPARK_CONF='spark.locality.wait=0s' (see How to run).
num_partitions: 96
seed: 42
# Known-good in-cluster HDFS cache (same path the rare6 dag_placement runs use).
# HDFS is per-cluster ephemeral (a fresh cluster rebuilds the bundle), but writes
# reliably within a run. Point at a persistent GCS bucket only if you have a valid
# one for this workspace (a wrong bucket now just warns and rebuilds, not aborts).
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
---

# 0076 — Gated-PC rare-disease case-finding (smooshed vocab)

The forward test named in insight 0066: prediction-constrained topic-shaping
should help in the **hidden-low-mass** regime — a rare phenotype an unsupervised
fit spends its K topics missing. This is the first real-AoU Gated-PC run, on a
single smooshed (fused-vocab) BOW over the rare6 rare-disease forest, mirroring
the corpus/gate config of the dag_placement incumbent (0065) so the numbers sit
in the same regime.

**What it fits** (`analysis/cloud/gated_pc_cloud.py`, ADR 0042 — inject the
hierarchy ONCE), all on the same gated hierarchy + per-node label:

- **`gated_pc`** — gate + **FLAT** PC head (`weight_y=50`, Newton head,
  `head_l2=1e-3`). The topic-side gate welds each node's topic block to its
  subtree's docs; the label-side head shapes the ungated label-free θ.
- **`unsup_gated`** — the `weight_y=0` twin (identical gate / K / init): the
  controlled, `pc_topics_lr`-comparable incumbent.

**Headline metric** — `pc_topics_lr` (insight 0066): a fresh post-hoc
LogisticRegression on each arm's FINAL per-doc θ against the per-node label,
macro AUC over nodes. It isolates representation quality from any co-fit head's
own convergence and is directly comparable across arms. The `gated_pc` arm also
reports its co-fit head P(node) AUC (secondary).

**Read at the end** (the `[driver] HEADLINE:` line):
1. `gated_pc pc_topics_lr` vs `unsup_gated pc_topics_lr` — a **positive delta**
   in this rare regime is the thesis. A null is a *data* finding (à la 0066 on
   the high-mass antidepressant task), not a bug.
2. `gated_pc co-fit head` AUC vs its own `pc_topics_lr` — did the Newton head
   converge to the representation ceiling?

**Tune** `weight_y` (↑ if the head is under-moved) and `tau0` (↓ so the head
moves on the smaller cohort). `with_dag_head: true` adds the ungated+closure-head
arm once the gate+flat baseline reads.

**Scale note.** `pc_topics_lr` collects each arm's `D × K` θ + labels to the
driver and fits a per-node LR there (train+test); the driver runs at 8g (same as
the `pc` model). If it OOMs at `person_mod: 1`, raise `person_mod` (sub-sample
patients) or override `CHARM_DRIVER_MEMORY`. The distributed gated fit itself
streams on executors as usual.

## How to run (cluster)

The repo lives at `~/repos/CHARMPheno` on the workspace. One-liner that syncs the
branch and runs the experiment:

```bash
cd ~/repos/CHARMPheno && \
  git fetch origin claude/spectral-anchor-topic-k-200nqp && \
  git checkout claude/spectral-anchor-topic-k-200nqp && \
  git pull --ff-only origin claude/spectral-anchor-topic-k-200nqp && \
  CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=76
```

`CHARM_SPARK_CONF='spark.locality.wait=0s'` makes executors added mid-job pick up
work immediately instead of being starved by cache-locality on the original nodes
(pairs with `num_partitions: 96`). Without it the corpus stays pinned to the first
~2 executors even after the pool scales up.

Fit-only (skip the self-contained result's implicit eval step):
`make -C analysis/cloud exp ID=76 NO_EVAL=1`.

**Full case-finding readout from a FINISHED run** (no re-fit — reconstructs the
model from the saved globals, reloads the cached bundle, and prints pc_topics_lr +
precision@recall / recall@FDR / detection for the gated_pc arm):

```bash
make -C analysis/cloud gated-pc-readout ID=76 GPR_DOC_MIN_LENGTH=10
```

`GPR_DOC_MIN_LENGTH=10` is only needed for runs whose manifest predates the
doc_min_length record (run 2); the current driver records it, so later runs need
just `ID=76`. `GPR_CACHE_URI` defaults to the HDFS bundle cache; override it (or
pass `GPR_ARGS="--bundle-path <cache_uri>/<key>"`) if the key misses.

## Run log

### Run 1 (2026-08-13) — head oscillated + Σλ blow-up; retuned to damped head

DAG pruned to 27 nodes (C=27), K=170 (40 bg + 26 nodes × 5 tpn), V=5000. ELBO
improved cleanly (−7.7M → −5.95M over 11 iters), so the topics were learning, BUT:

- **Head oscillated:** `|w_CK|max` non-monotone 1.0 → 5.9 → **19.1** → 7.3 → 13.9
  → **60.7**, spiking at iter 11 as the `weight_y_warmup_iters=10` ramp completed
  and full `weight_y=50` engaged. The insight-0075 under-damped-Newton failure mode
  (`head_lr=0.5`, `head_newton_ridge=0.01` too loose).
- **Σλ_k blew up:** max 4.6e4 → **6.2e5** in 11 iters while min dropped to 1.3e3 —
  one topic hoovering mass, the degenerate drift the trust region (`topic_trust=0.1`,
  too loose) is meant to bound, driven by the oscillating head's correction signal.

**Retune (this config):** `head_lr 0.5→0.3`, `head_newton_ridge 0.01→0.05` (the
settled 0075 damped-head values), `topic_trust 0.1→0.05`, `weight_y_warmup_iters
10→25`. `pc_topics_lr` would have been valid even on run 1 (convergence-robust),
but the Σλ blow-up risked degrading the gated_pc topics themselves → a
misleadingly negative delta, so run 1 was not a clean test. If run 2 still drifts,
the next lever is `weight_y` (50→20–30, gentler correction).

### Run 2 (2026-08-13) — parallelism fixed; head spikes-and-recovers; Σλ stable

`num_partitions: 96` + `spark.locality.wait=0s` fixed the under-parallelization
(run 1 was pinned to ~2 executors / 8 partitions at ~90s/iter → **~12s/iter**).
ELBO smooth despite the 10% minibatch (−7.7M → −3.6M by iter 78).

- **Head `|w_CK|` spikes and RECOVERS, not diverges:** e.g. iters 67–78
  50→31→103→192→**286**→179→125→88→61→43→31→24. Occasional near-singular
  minibatch H_c → a big IRLS step, damped back by head_lr=0.3 + ridge=0.05.
  Bounded chasing of a still-moving θ, NOT runaway. And `|w_CK|`≈1e2 is far below
  the recalibration equilibrium (~1e4 at head_l2=1e-3), and `pc_topics_lr` doesn't
  read the head anyway — so this is cosmetic to the headline.
- **`Σλ_k min` is a non-issue (plateaued ~31–38):** the earlier apparent "collapse"
  (1.67e3→142) LEVELED OFF, not drained to zero. It's just under-supported topics
  under the sparse LDA prior with K > what short 1yr docs support (SO's call) — not
  degenerate drift. `Σλ_k max` stable ~5.8e5.

- **Head BLEW UP late (iter ~151):** after ~150 stable iters `|w_CK|` ran away
  45 → 726 → 4540 → 1.17e5 → 3.48e5 → **1.25e6**, and the ELBO stopped improving /
  started bouncing. Cause: the converged gated topics went ~separable, the per-node
  logistic MLE → ∞, and `head_l2=1e-3` was too weak to bound the singular-minibatch
  IRLS step. `Σλ_k` stayed flat (topic_trust held the TOPICS), so `pc_topics_lr`
  should survive but the co-fit head readout is garbage (saturated at |w|=1e6).

**Run 2 finished — full readout (`make gated-pc-readout`, 15/27 nodes scored):**

| arm | AUC | AP | node P@R0.5 | node P@R0.9 | R@FDR0.1 |
|---|---|---|---|---|---|
| gated_pc pc_topics_lr | 0.7947 | 0.0434 | 0.025 | 0.011 | 0.003 |
| unsup_gated pc_topics_lr | 0.7862 | 0.0447 | — | — | — |
| gated_pc co-fit head | 0.7644 | 0.0273 | 0.020 | 0.010 | 0.001 |

Detection (case vs bg, prevalence 0.045): gated_pc **AUC 0.727 / AP 0.141**,
P@R0.5 = 0.109; co-fit head AUC 0.678 / AP 0.082.

Two reads:
1. **PC supervision ≈ wash** (AUC Δ+0.0085, AP Δ−0.0013 vs unsup_gated) — insight
   0066 again: the label adds no representation-level signal over the unsupervised
   gated topics on this corpus.
2. **AUC flatters a low-prevalence problem** (insight 0064): AUC ~0.79 but per-node
   precision is ~2.5% at 50% recall — real but weak signal (~2.5–3× lift over base
   rate), NOT precise-case-finding usable. The full readout is what surfaces this;
   AUC alone would have read as "decent."
3. Co-fit head (0.764) < pc_topics_lr (0.795): the late blow-up degraded it, as
   expected → Firth (task #27) should recover this + give calibrated probabilities.

Caveat: this is the STARVED 1yr / blown-head run. Run 3 (5yr history, head_l2=1e-2,
100 iters) is the fair test of whether more history lifts the representation.

## Follow-ups (next run = run 3, this config)

- **Extend the feature history beyond 1 year** (`lookback_days: 365 → 1825`, 5yr):
  richer per-doc BOW → more mass to support K=170, and a fairer test of whether the
  rare phenotype is a hidden-low-mass signal. The short 1yr window is likely
  starving the node topics (Σλ_k min ~31). ✅ set for run 3.
- **Raise `head_l2` 1e-3 → 1e-2 + cap at 100 iters** to prevent the late-iter head
  blow-up. ✅ set for run 3.
- **DURABLE FIX (decided): Firth / Jeffreys-prior penalized logistic head.** The
  late-iter blow-up is *separation* — the converged gated topics make each node's
  logistic problem linearly separable, so the MLE runs to ∞. The current levers
  (`head_l2`, `head_newton_ridge`, `head_lr`) all bound `|w|` with a tunable knob;
  a step clamp would reintroduce a *timescale* knob (the two-timescale pathology we
  fought in 0065). Firth's penalty `+½·log det I(β)` is the **parameter-free** cure:
  it's the Jeffreys prior (determined by the Fisher information, nothing to tune),
  and `log det I → −∞` as `|w| → ∞`, so it self-regularizes *exactly* at separation
  and guarantees finite estimates. Unlike unit-norm (which discards the magnitude →
  squashed, uncalibrated logits), Firth keeps a **finite, bias-reduced, calibrated**
  probability — interpretable per-patient P(node), the deliverable we do want.
  Cost: predictively ~none (often a hair better on rare nodes with few positives —
  our regime); compute ~free (the head aggregation is a rounding error next to the
  per-doc CAVI E-step); real cost is code/test complexity (leverages
  `h_d = p_d(1−p_d)·θ_dᵀH⁻¹θ_d` reuse the Newton `H⁻¹`, folded into each inner IRLS
  step so `H` stays conditioned). **Sequencing:** implement AFTER run 3 so runs 2/3
  (the `head_l2` ridge stopgap) are the comparators. Then run 4 can drop `head_l2`.
  Rejected alternative: head trust-region step clamp (a timescale knob).

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
n_bg: 1                  # run 6: 40→1 (EXTREME). K drops 66→27 (1 bg + 26 nodes×1).
                         # Probe: how hard does starving the background topic space hit
                         # the representation? 1 bg topic must absorb ALL non-disease
                         # structure (comorbidity/common-disease); either it forces the
                         # node topics to stay disease-specific (helps) or it pollutes
                         # them (hurts). A/B vs run 4 (n_bg=40) isolates background capacity.
tpn: 1                   # run 4: 5→1. Unused topics are free UNSUPERVISED (they
                         # just die: Σλ_min ~31) but costly in the CO-FIT — the head
                         # reads all K, so tpn=5 → K=170 over-fits/separates the head
                         # (the blow-up), dilutes the correction across 5 topics/node,
                         # and inflates the pc_topics_lr LR. tpn=1 gives a clean 1:1
                         # node→topic→label chain, K=66, smaller head (less blow-up,
                         # orthogonal to Firth), and no dead node-topics. A/B vs run 3
                         # (5yr, tpn=5) isolates tpn. n_bg=40 held; it's the next lever.
optimize_doc_concentration: true
# --- PC head (inject the hierarchy ONCE: gate + FLAT head; ADR 0042) ----------
weight_y: 50.0            # PC prediction weight. Hughes ~ tokens/doc; rare6 1yr
                         # windows are short, so 50 is a starting point — TUNE on
                         # the delta vs unsup_gated (try 100/200 if the head is
                         # under-moved, i.e. |w_CK| still climbing at max_iter).
head_optimizer: newton   # settled convergent head (ADR 0039); no sgd/adam.
# Run 5 attempt 1 (head_penalty=firth, head_l2=0) DIVERGED on real data — |w_CK|→1e17
# at iter 1: the undamped inner IRLS overshoots on early-iter rank-deficient θ / logit
# saturation, and with head_l2=0 there's no floor. FIXED (step-halving, commit 4874bb7 /
# engine in 6d05a61): the Firth IRLS now ascends the penalized LL via step-halving —
# reproducing test shows raw |w|=7.99e10 → bounded 0.42. TRUE FIRTH is ready: flip
# head_penalty→firth + head_l2→0 to re-run (attempt 2).
#   Currently set to the PATH-B-NO-FIRTH CONTROL (head_penalty=none, head_inner_iters=25,
# head_l2=1e-2 — the ridge floors the solve): a stable run isolating the inner-loop
# head-fit PATH (vs run 4's aggregated one-step, Path A). firth-PathB vs this none-PathB
# then isolates the PENALTY; this vs run 4 isolates the PATH.
# Head = run 4's KNOWN-GOOD Path A (aggregated one-step Newton + head_l2 ridge). Firth
# on Path B was abandoned on the cluster (run 5b): (1) driver-bound + crawling — the
# step-halving line search (slogdet per halving × 27 labels × 25 inner iters) idles the
# cluster; (2) still blew up — the line search is toothless when the CURRENT point has a
# rank-deficient Fisher (PLL=−inf baseline ⇒ any finite step clears ≥−inf), which is the
# early-iteration real-data regime. Not worth fixing: the head does NOT move pc_topics_lr
# (0.79–0.80 every run regardless of head state; run 5a's dead head cost 0.006), and
# calibrated probabilities are unused on an information-limited task. Firth stays an
# implemented + unit-tested engine feature (works on well-conditioned data) for a future
# task where calibration is the deliverable and the design isn't rank-deficient.
head_penalty: none
head_inner_iters: 0      # Path A (one aggregated Newton step/iter — fast, bounded).
head_lr: 0.3
head_newton_ridge: 0.05
head_l2: 0.01            # absolute ridge = run 4 (|w_CK| stayed 14.6, no blow-up).
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
eval_every: 20           # log pc_topics_lr AUC + detection AP every 20 iters of the
                         # gated_pc fit (watch the shaping converge). Each eval = 2 CAVI
                         # transforms + an LR on the driver; raise to 50 (2 evals) or 0
                         # (off) if the overhead bites.
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

### Run 3 (5yr history, tpn=5, head_l2=1e-2, 100 iters) — head bounded; history didn't help

Head stayed bounded (`|w_CK|` ~17–20 through iter 100, NO blow-up) — `head_l2=1e-2`
+ the 100-iter cap held. Clean run.

| arm | AUC | AP | detection AUC / AP | det P@R0.5 |
|---|---|---|---|---|
| gated_pc pc_topics_lr | 0.7986 | 0.0338 | 0.738 / 0.127 | 0.104 |
| unsup_gated pc_topics_lr | 0.7957 | 0.0320 | 0.734 / 0.116 | 0.102 |
| gated_pc co-fit head | 0.7722 | 0.0272 | 0.704 / 0.093 | 0.082 |

HEADLINE Δ (gated_pc − unsup): AUC +0.0029, AP +0.0018, node P@R0.9 +0.0001,
**detection AP +0.0110** (the clearest PC signal yet). Prevalence 0.038.

Reads:
1. **5yr history ≈ flat vs 1yr.** Absolute AP looks lower (0.127 vs run 2's 0.141)
   but prevalence dropped 0.045→0.038; prevalence-adjusted detection LIFT went
   3.1× → 3.3× — essentially unchanged. The rare-disease signal is NOT
   history-limited; a longer window doesn't unlock it.
2. **PC still marginal, but detection AP Δ+0.011 (~9% rel) is the clearest win** —
   at the coarse case-vs-bg level, not fine node discrimination (node P@R flat).
3. **Head bounded** — head_l2=1e-2 stopgap works; the co-fit head (0.772) is
   meaningful this run (unlike run 2's blown 0.764).
4. Absolute precision still poor (det P@R0.5 ~0.10) — real but weak (~3× lift), not
   yet case-finding usable.

### Run 6 (n_bg=1, K=27) — the PC MECHANISM demonstrated: capacity↓ ⇒ absolute↓ but PC-benefit↑

| arm | AUC | AP | detection AUC / AP |
|---|---|---|---|
| gated_pc pc_topics_lr | 0.7734 | 0.0268 | 0.715 / **0.0957** |
| unsup_gated pc_topics_lr | 0.7646 | 0.0241 | 0.684 / **0.0795** |
| gated_pc co-fit head | 0.7328 | 0.0212 | 0.686 / 0.0878 |

HEADLINE Δ: AUC +0.0089, AP +0.0027, node P@R0.9 −0.0001, **detection AP +0.0162
(+20% rel)**. `|w_CK|` 13.8 (bounded, Path A). Σλ_k min 185 / max **1.5e7** (the lone bg
topic became a mega-topic absorbing all non-disease mass). eval trace: AUC 0.745(20) →
0.765(40) → 0.767(60) → 0.765(80) → 0.773(100) — mostly plateaued by ~40, still creeping.

**The finding (a genuine trade-off, first clean PC-mechanism evidence on real data):**
1. **Starving background HURTS absolute quality** — det AP 0.134→0.096, AUC 0.80→0.77 vs
   run 4 (n_bg=40). 1 bg topic can't hold the diversity of non-disease structure, so it
   leaks into the node topics. Background capacity does real work.
2. **…but 3.7×'s the PC BENEFIT** — det AP Δ went +0.0044 (run 4) → **+0.0162 (run 6)**,
   +20% relative, the clearest PC signal in the arc. Exactly insight 0066's mechanism:
   PC helps in proportion to how much the UNSUPERVISED fit misses. Large n_bg ⇒
   unsupervised already captures it ⇒ PC marginal; n_bg=1 ⇒ unsupervised drops (det AP
   0.130→0.080) ⇒ supervision RECOVERS a chunk (0.080→0.096).
3. **Trade-off, not a win:** low n_bg = worse absolute + bigger PC delta; high n_bg =
   better absolute + tiny PC delta. But it's the first real demonstration that the label
   does representational work when there's room — the Gated-PC thesis, shown.

## Domain value — SETTLED on the hybrid branch (do not re-derive)

The `hybrid-domain-reliability-review-ckn2bq` branch ran the full per-domain
ablation (odds-ratio / LR-over-codes readouts on unsupervised gated fits). Verdict
retires the naive "add meds+labs":

- **Condition is necessary + nearly sufficient** (only:condition ≈ or > all for 5/6
  rare6; insight 0071).
- **Observation is net-negative for all six** — dropping it *improves* placement;
  AoU survey/SDOH/admin/vitals vocab, high volume/low specificity; PPI strip
  insufficient (insights 0071, 0072). "Observation sucks" confirmed.
- **Drug is a mild disease-specific positive** — only signature-drug diseases (MG,
  lupus, EDS POTS meds); ~neutral else (0071).
- **Value-aware measurement (range-abnormality tokens) is the ONE lift** — rescues
  labs-dependent diseases condition can't find (Guillain-Barré, Marfan, Osler,
  sarcoid, CIDP, FH), but DEGRADES condition via modality interference (near-
  universal `[measured]` tokens pull shared θ off condition), and **no fixed combo
  beats condition-alone at macro** (insights 0078, 0079).
- **Supervised per-disease domain weighting already tried** (nested-CV): ~6% over
  fixed, condition-primary (~28/39 → condition=1.0), "not worth deploying" (0075,
  0076).
- **Case-finding here is INFORMATION-LIMITED** (0062): FPs are genuine code-level
  similarity (anemia+CKD background ≡ SLE on condition codes). Not a scoring/head/
  weighting problem — the information isn't there.

**Implications for this arc.** Runs 2–4 (~3× lift, condition-only, history-
insensitive) sit at that documented ceiling — we reproduced it. So:
1. **Multi-domain is retired for the macro** — flat cond+drug+obs is a dead-end;
   don't build it. The ONLY live thread is value-aware measurement for the labs-
   dependent SUBSET *through the per-node gate* (untested combo: the gate might let
   labs-dependent nodes go measurement-heavy without diluting condition-nodes — the
   shared-θ interference the branch hit is per-node-separable here). Speculative,
   low ceiling, ~6 diseases.
2. **Firth (task #27) is next** — contained, orthogonal; cleanup (calibration + drop
   head_l2), NOT a ceiling-raiser (the head was never the bottleneck).
3. **Strategic:** PC needs a hidden-low-mass signal that EXISTS; on condition-coded
   rare disease it mostly doesn't. The cleanest Gated-PC demonstration may be a
   present-but-buried task, not this present-but-absent one.

### Run 4 (tpn=1, K=66, 5yr) — CONFIRMED: quarter the topics, same quality; adopt tpn=1

| arm | AUC | AP | detection AUC / AP | det P@R0.5 |
|---|---|---|---|---|
| gated_pc pc_topics_lr | 0.8000 | 0.0320 | 0.734 / 0.134 | 0.105 |
| unsup_gated pc_topics_lr | 0.7974 | 0.0314 | 0.730 / 0.130 | 0.103 |
| gated_pc co-fit head | 0.7756 | 0.0257 | 0.709 / 0.112 | 0.085 |

HEADLINE Δ: AUC +0.0025, AP +0.0007, node P@R0.9 +0.0000, detection AP +0.0044.
`|w_CK|`final 14.6, Σλ_k min **76** (run 3 was 29), prevalence 0.038.

Reads:
1. **tpn=1 confirmed — K 170→66, quality held/better.** pc_topics_lr 0.800 ≈ run 3's
   0.799; detection AP 0.134 > run 3's 0.127; co-fit head 0.776 > 0.772; `|w_CK|` calmer
   (14.6 vs ~20); Σλ_min 29→76 (dead-topic starvation gone — each node's single topic
   owns its subtree's mass). The extra sub-topics were pure liability in the co-fit.
   **Adopt tpn=1.**
2. **PC still a wash — 5th confirmation.** detection AP Δ+0.0044 (< run 3's +0.011).
   Supervision adds no representation signal (insight 0066).
3. **Ceiling unchanged** — det AP 0.134, ~3.5× lift at prev 0.038; same ~3× every run.
4. Head already well-behaved at K=66 (no blow-up) → Firth's run-5 job is knob-removal +
   calibration + closing the 0.776→0.800 head gap, not blow-up prevention.

### Run 5a (Path-B-no-Firth control: none + inner_iters=25 + head_l2=1e-2) — head blew up UNGUARDED

| arm | AUC | AP | detection AUC / AP | co-fit head AUC |
|---|---|---|---|---|
| gated_pc pc_topics_lr | 0.7938 | 0.0278 | 0.738 / 0.116 | — |
| unsup_gated pc_topics_lr | 0.7922 | 0.0277 | 0.724 / 0.120 | — |
| gated_pc co-fit head | — | — | 0.500 / 0.038 | **0.500 (DEAD)** |

`|w_CK|` = **2.2e7** (prev 0.038). Reads:
1. **The `none` inner-loop path is UNSTABLE without step-halving.** 25 undamped Newton
   steps/M-step blew `|w|` to 2.2e7 even with `head_l2=1e-2` — the co-fit head saturated
   to chance (AUC exactly 0.500). The `none` branch has no trajectory guard (only firth
   does); Path B is intended for Firth. (Minor follow-up: guard `none`+inner_iters>0 too.)
2. **The blown head slightly HURT the representation** — pc_topics_lr 0.800→0.794,
   detection AP 0.134→0.116 vs run 4's Path A: the saturated head fed a bad correction.
3. **PC wash — 6th confirmation** (pc_topics_lr Δ+0.0016, detection AP Δ−0.0035).
4. **This is the motivation for Firth:** the inner-loop path (needed for calibrated
   probabilities) needs step-halving to be stable; Firth supplies exactly that.

### Run 5b (Firth attempt 2, step-halving) — ABANDONED on cluster

Killed. Two inherent Path-B-Firth problems: (1) **driver-bound + crawling** — the
step-halving line search (a `slogdet` per halving × 27 labels × 25 inner iters) is a
huge serial driver computation that idled the cluster (executors `active 0/4`,
`util 0%`); (2) **still blew up** — the line search accepts `w−step` if `PLL(cand) ≥
PLL(current)`, but early-iter random topics give a rank-deficient Fisher so
`PLL(current)=−inf`, and any finite step clears `≥−inf` (toothless exactly when needed;
the reproducing test's ridge masked this).

**Decision: abandon cluster Firth; keep it as a tested engine feature.** The head does
NOT move pc_topics_lr — 0.79–0.80 every run regardless of head state (bounded/dead/blown);
run 5a's dead head cost 0.006. Calibrated probabilities are unused on an
information-limited task. Reverted 0076 to run 4's Path A + head_l2=1e-2 (bounded, fast).
The head arc is practically closed: Newton → head_l2 → (Firth deferred). To make Firth
cluster-viable later: a robust NON-collapsing conditioner (absolute floor, no slogdet
line search) or Path-A Firth via a broadcast leverage pass — only if calibrated
probabilities become a real deliverable on a non-rank-deficient task.

## (removed) Run 5b config

Rationale above (frontmatter `tpn`): the co-fit head reads every topic, so the 5
sub-topics/node that are free unsupervised become a liability (head over-fit /
separation, diluted correction, dead node-topics). Run 4 = run 3 (5yr, head_l2=1e-2,
100 iters) with `tpn=1` (K 170→66), a clean A/B on tpn. Run AFTER run 3 so the
tpn=5 5yr result is the comparator. Watch: does the head stay bounded (smaller K),
and do pc_topics_lr / detection-AP / P@R hold or improve at a quarter the topics?
`n_bg=40` held (the next lever if the head still separates — 40 shared bg features
is a lot for a rare-node head). Note: changing tpn re-keys the bundle cache, so run
4 rebuilds the (identical, tpn-independent) 5yr bundle once — harmless.

## Follow-ups (after run 4)

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
  ✅ **IMPLEMENTED** (commit 3368926): opt-in `head_penalty="firth"`. Separable-fixture
  gate: plain-Newton |w|→325 (clip-saturated), Firth |w|→30 finite/stable (AUC=1);
  non-separable Firth ≈ MLE (cosine 1.0000). 50 PC tests green.

## Run 5 (planned) — Firth head, drop head_l2 (after run 4)

`head_penalty: firth`, `head_l2: 0` on the run-4 config (tpn=1, 5yr, 100 iters). The
parameter-free separation cure: no ridge knob, and a genuinely calibrated per-patient
P(node). **Read:**
- `|w_CK|` bounded through ALL iters with NO head_l2 (the proof Firth supersedes the
  ridge → run 6+ drops the knob for good).
- pc_topics_lr AUC/AP ≈ runs 3/4 (ranking is scale-invariant, so Firth ≈ same
  discrimination — calibration came free).
- **co-fit head AUC ≈ pc_topics_lr** (run 3 had a 0.772 vs 0.799 gap; a converged,
  finite, calibrated Firth head should close it — the clearest head-quality win).

**Interpretation nuance.** Firth lives on the *inner-loop* head path (Path B: driver-
side per-doc IRLS), whereas runs 2–4 used the *default aggregated one-step* path
(Path A). Path B also carries a ±50 logit clip, so its `none`-baseline `|w|` is already
partly bounded (~325 in the fixture) — unlike Path A's uncapped 1e6 cluster blow-up.
So run 5 changes BOTH the path and the penalty vs runs 3/4; the clean read is
"|w_CK| small/finite AND pc_topics_lr holds AND co-fit head ≈ pc_topics_lr." A
`head_penalty=none, head_inner_iters=25` control (Path B without Firth) would isolate
Firth exactly if the head-quality delta is ambiguous.

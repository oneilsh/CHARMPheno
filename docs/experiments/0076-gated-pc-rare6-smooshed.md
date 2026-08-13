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
lookback_days: 365
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
head_lr: 0.5
head_newton_ridge: 0.01
head_l2: 0.001           # ABSOLUTE ridge = Hughes lambda_w (ADR 0041). 0.0 blows
                         # up on the separable topics PC creates.
grad_cavi_iters: 30      # differentiable CAVI unroll depth; must match scoring
                         # convergence (cavi_max_iter=100). 30 suffices for the
                         # short lookback docs (deeper = bigger autograd tape).
topic_trust: 0.1
weight_y_warmup_iters: 10
# --- SVI schedule (comparable to 0065 / the pc-vi runs) -----------------------
max_iter: 200
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
  make -C analysis/cloud exp ID=76
```

Fit-only (skip the self-contained result's implicit eval step):
`make -C analysis/cloud exp ID=76 NO_EVAL=1`.

## Run log

_(pending first run)_

---
id: 134
slug: mondo-cardiovascular-blend-bg1200-tpn1-stacked
status: done
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# THE BLEND: shared strata + one signature per node, under the stacked head (insight 0094).
# 0133's config with ONE knob: n_bg 8 -> 1200, so K = 1200 + 298 = 1498 — the SAME width as
# 0123 (tpn=5) and 0132 (flat), the non-arbitrary match. Every document sees the 1,200
# background topics plus its closure's one-topic blocks.
#
# WHY. 0133 (tpn=1, K=306) holds the gate's gap on ranking (+0.008 vs tpn=5, +0.038 vs
# flat), has the best de-novo AUC (0.855), 0% starved, and the most legible per-node topics
# of the series — and loses the ROOT head (0.781 vs 0.807/0.798) and with it de-novo AP
# (0.198 vs 0.236). The root head reads population strata and K=306 offers eight; 0132's
# 1,250 live flat strata gave it 0.798. This run adds the strata back WITHOUT touching the
# per-node unit. HSLDA's shared profiles + the gate's one signature per node.
#
# SEED. A 1,200-topic background is NOT anchorable: the hull-vertex greedy and the per-word
# NNLS recovery both scale with the anchor count (the first two launches died in each in
# turn — see the run log). And it should not be: 0132 showed a flat background forms its
# ~1,250 live strata from a random start on its own. So spectral_bg_anchors: 8 anchors
# the first eight background rows exactly as 0133 did (same deflation seeds for every
# node), and the other 1,192 background rows take the engine's random Gamma init — the
# one 0132's flat fit started from. The only difference from 0133 is +1,192 free
# background topics. spectral_d stays at 0133's 768 (8 + depth anchors per node).
#
# READ (ridge 100, both readouts; ABs vs 0133, 0123, 0132; named digest):
#   - root head and de-novo AP: back to ~0.80 / ~0.24 (0123's) or not.
#   - within-cohort macro + paired vs 0133 (0.8006): do the per-node topics keep their
#     signal next to 1,200 strata, or does the background absorb it (the head can read
#     either; the digest says which it is).
#   - digest: are the per-node topics still the 0133 textbook ones? Does the background
#     carry the cohort strata (pregnancy, diabetes, asthma) that 0133's same-vocabulary
#     children fell into — i.e. does DCM's one topic get its own words back once the
#     young-women stratum has a background home?
# OUTCOMES:
#   - root ~0.80, per-node ≈ 0133, digest as legible or better: the unit is found — one
#     signature per node over shared strata, closure product as decoder. tpn closed at 1.
#   - root ~0.80 but per-node topics emptied into the background (paired vs 0133 down,
#     digest generic): the background competes with the blocks; n_bg is then a real knob
#     and the answer is somewhere between 8 and 1200 — stop and think, do not sweep.
#   - root unchanged (~0.78): width was not it; the root head wants something else
#     (e.g. the readout's top-256 truncation at K=1498 — check theta top-m mass).
# COST: fit between 0133's and 0132's (a gated doc sees 1,200 + depth topics); seed as
# above; bundle HIT on a warm cluster (same key as 0123/0132/0133).
dag_source: mondo_native
mondo_branch: MONDO:0004995
tpn: 1
max_iter: 50
diag_only: true
# --- the ONLY change from 0113: spectral block-aligned seed for the gated engine ---
init: spectral
spectral_method: scalable   # concatenated V ~11.6k >= 8000 threshold; dense = driver wall
spectral_d: 768             # 0133's; only 8 + depth anchors per node (see SEED)
anchor_scope: closure       # node trained from its whole closure; ancestors deflated by topo order
spectral_topo_order: forward  # ancestors-first: each node's seed = its increment over ancestors
count_transform: none       # THE ONLY CHANGE vs 0115: raw per-visit counts (0114's representation)
# ------------------------------------------------------------------------------------
preindex_closure: false
readout_mode: distributed
readout_theta_topm: 256
weight_y: 0.0
weight_y_warmup_iters: 0
skip_unsup_gated: true
min_positives: 100
mondo_version: 2026-06-02
mondo_cache_dir: data/mondo
extra_domains: measurement,drug
label_mask_mode: closure
localize_head: true
head_support: path_cousins_kids
head_intercept: true
head_standardize: true
doc_concentration: 0.5
head_lr: 1.0
person_mod: 1
prior_obs_days: 0
doc_min_length: 10
min_n: 0
holdout_frac: 0.2
vocab_size: 5000
min_df: 20
min_patient_count: 20
window_mode: lookback
lookback_days: 1825
label_window_days: 365
strip_mode: both
n_bg: 1200                  # THE CHANGE vs 0133: K = 1200 + 298 = 1498
spectral_bg_anchors: 8      # anchor 0133's eight; the other 1,192 bg rows start random (see SEED)
# LANDMINE (2026-09-11): 0113-0120 all say `optimize_doc_concentration: true` but the flag
# was INERT on the PC path until 0121 wired it (pc.py `set_alpha_policy`); their alpha was
# FIXED at 0.5. Copying that line now turns the empirical-Bayes alpha ON, and learned alpha
# collapses to the floor and re-starves depth (insight 0091; 0121: 79% starved). The first
# 0123 launch did exactly that (alpha mean 0.5 -> 0.21 by iter 19, background block 0.28
# and falling ~2.5%/iter) and was killed. `false` here reproduces 0115's EFFECTIVE alpha.
optimize_doc_concentration: false
head_optimizer: newton
head_newton_ridge: 0.05
head_l2: 0.01
grad_cavi_iters: 15
topic_trust: 0.05
subsampling_rate: 0.1
tau0: 64.0
kappa: 0.51
cavi_max_iter: 100
cavi_tol: 0.001
with_dag_head: false
baseline_max_iter: 100
min_label_count: 20
eval_every: 0
num_partitions: 96
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
spark_conf:
  # Same geometry as 0113 (from 0110, the whole-Mondo survivor). Over-provisioned and
  # safe at this branch's K; the scalable spectral seed's sequential passes are the new
  # cost, not the fit.
  spark.executor.cores: 2
  spark.executor.memory: 8g
  spark.executor.memoryOverhead: 3g
  spark.dynamicAllocation.enabled: "false"
  spark.executor.instances: 20
  spark.excludeOnFailure.timeout: 10m
  spark.cleaner.periodicGC.interval: "5min"
---

# 0134 — shared strata + one signature per node (n_bg=1200, tpn=1, K=1498) under the stacked head

0133 with 1,200 background topics. Insight 0094: one topic per node holds the gate's gap
and is the most legible topic side of the series; what it lacks is strata for the root
head to read. This adds them at matched width.

## What it does

`make -C analysis/cloud exp ID=134` → spectral-init gated fit with a 1,200-topic shared
background and one topic per node; the ridge-100 readout (record); the stacked readout;
paired ABs vs 0133, 0123 and 0132; a named digest of the cardiomyopathy neighbourhood.

## Acceptance criterion

The three-way split in the front matter: root head back to ~0.80 with per-node topics
intact (the unit), root back but blocks emptied (n_bg is a knob — stop), or root
unchanged (width was not it).

## Run

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0134-mondo-cardiovascular-blend-bg1200-tpn1-stacked
mkdir -p "$RUN"
nohup bash -c '
  make -C analysis/cloud exp ID=134 || exit 1
  echo "=== readout START $(date)"
  make -C analysis/cloud gated-pc-readout ID=134 GPR_ARGS="--readout-mode distributed --readout-l2 100"
  echo "=== stacked START $(date)"
  make -C analysis/cloud gated-pc-readout ID=134 GPR_ARGS="--readout-mode distributed --readout-l2 100 --readout-stacked"
  for b in 133 123 132; do make -C analysis/cloud inspect-topics ID=134 COMPARE=$b INSPECT_ARGS="--readout-auc"; done
  make -C analysis/cloud inspect-topics ID=134 RESOLVE_NAMES=1 INSPECT_ARGS="--digest --redundancy 30 --grep '"'"'cardiomyopathy|atrial fibrillation|heart failure|pregnan'"'"'"
  echo "=== DONE $(date)"
' > "$RUN"/sweep_log.md 2>&1 &
```

Early check: the corpus line must read `K=1498 gated topics (1200 bg + 298 nodes x 1 tpn)`.

Pull the numbers with:

```bash
grep -E "^=== |corpus: V=|starved|top-m mass|gated_pc(_stacked)? \(pc_topics_lr\): (macro|detection)|root head alone|stacked: max|ranking \(within|paired per-node|paired delta|depth [0-9]+:|flat sigma|stacked P_stack|^## paired|^all: n=|^by depth" "$RUN"/sweep_log.md
```

## Run log

**2026-10-07 — first launch killed in the seed.** 54 minutes of silence after
`>>> gated_pc fit` with the driver at ~10% CPU: the background anchor greedy
(`find_anchors_projected`) re-projected every vocabulary row against the whole basis at
every step — O(n² · V · d) in Python-level dots — fine for 8 background anchors, days for
1,200. Rewrote both greedies (dense and projected) onto one basis-form search
(`spectral_init.greedy_anchors`: basis as state, one GEMV per chosen anchor, incremental
residual norms, seeds spanned in one SVD batch); same anchors up to floating-point ties,
pinned against the old loop by `spark-vi/tests/test_anchor_greedy_residual_form.py`.
Relaunched on the fixed code.

**2026-10-07 — second launch killed in the seed, one step later.** The greedy took seconds;
the cluster then sat idle for 25+ minutes after the pooled-sketch stages: the background
β RECOVERY (`recover_beta_projected`) is one NNLS per vocabulary word against the anchor
rows, and 1,200 anchors makes each solve seconds — hours for the vocabulary, and the same
again per node. The honest conclusion is that a wide background is not anchorable and
should not be: the design changes to `spectral_bg_anchors: 8` (0133's eight anchored
background rows, identical deflation seeds) + 1,192 random-init background rows (0132's
start), `spectral_d` back to 768. Third launch on that.

**2026-10-07 — third launch (spectral_bg_anchors 8, spectral_d 768) ran to the end.** Corpus
line `K=1498 gated topics (1200 bg + 298 nodes x 1 tpn)`; seed as fast as 0133's; the
named digest landed from the chain itself. The readout numbers were not pulled into this
doc before the thread moved; pulled afterwards (Results).

## Results

**Digest (named; 1,200 background + one topic per node), paired with 0133's by eye:**

| | 0133 (8 bg + 1/node, K=306) | **0134 (1200 bg + 1/node, K=1498)** |
|---|--:|--:|
| starved | 0% | 1% |
| evidence: min / median / p90 / max | 43.5 / 3.8e3 / 9.6e4 / 1.65e6 | 10.7 / 1.69e3 / 5.3e4 / 5.7e5 |
| median evidence by depth 3 / 5 / 7 | 1.85e4 / 3.8e3 / 1.14e3 | 5.6e3 / 1.69e3 / 802 |
| parents with ≥2 fed children / collapsed / worst sibling cosine | 80 / 0 / 0.51 | 81 / 0 / 0.41 |

**The per-node topics are 0133's, next to 1,200 strata.** Every textbook block of 0133 is
still a textbook block here, often sharper: atrial fibrillation (AF, paroxysmal, chronic,
persistent, flutter // warfarin, metoprolol, diltiazem), paroxysmal AF (// apixaban,
rivaroxaban), cardiomyopathy d3 (cardiomyopathy, primary CM, HF, DCM, CHF, chronic
systolic HF, LBBB), systolic HF (// carvedilol, spironolactone, valsartan), diastolic HF,
congestive HF (// furosemide), hypertrophic CM (HCM, HOCM, cardiomegaly, ventricular
tachycardia // metoprolol, verapamil), extrinsic CM (amyloidosis, "cardiomyopathy
associated with another disorder"), alcoholic CM (dependence, withdrawal, subdural
haemorrhage // thiamine), rheumatic CHF, Takotsubo (Takotsubo CM, NSTEMI, breast
carcinoma in situ — older women). The background did not empty the blocks: node evidence
is roughly halved (the shared strata take the generic mass the node topics no longer
have to carry), but the identity of every block that had one is intact, and sibling
distinctness improved (worst cosine 0.41 vs 0.51).

**The background did NOT give the same-vocabulary children their words back.** The one
failure mode of 0133 (insight 0094) is unchanged, node for node: dilated CM is still the
young-women primary-care stratum (unplanned pregnancy, pharyngitis, myopia, acne);
intrinsic CM still PCOS / irregular periods; persistent AF still a diabetes stratum;
non-familial restrictive CM still asthma; heart failure d3 still the CKD / COPD / OSA
comorbidity stratum; toxemia of pregnancy still generic ED symptoms. A 1,200-topic shared
background with those strata in it does not stop forward deflation from leaving the
child's residual AS that stratum: the child's one topic is seeded from its own closure
documents after its ancestors claimed the shared vocabulary, and the ancestors' claim is
the problem, not the absence of a stratum to absorb the cohort. That closes the
"background as a home for the cohort" hypothesis from the front matter.

**Readout numbers (ridge 100; 193 within-cohort / 224 de-novo nodes; 54,753 persons):**

| read | 0123 tpn=5 | 0132 flat | 0133 tpn=1 | **0134 blend** |
|---|--:|--:|--:|--:|
| flat head, within-cohort macro AUC / AP | 0.7946 / 0.533 | 0.7612 / 0.476 | **0.8006** / 0.517 | 0.7823 / 0.512 |
| paired vs 0133 (193) | — | — | — | −0.018 (55/138); d2 +0.021, d3 −0.007, d4–d7 −0.017…−0.024 |
| paired vs 0123 / vs 0132 | — | — | +0.008 / +0.038 | −0.009 (60/133) / +0.017 (144/49) |
| root head alone (= stacked detection) | 0.8065 / AP 0.886 | 0.7983 / 0.881 | 0.7807 / 0.866 | **0.8113 / 0.890** |
| stacked de-novo macro AUC / AP (224) | 0.8497 / **0.236** | 0.8362 / 0.185 | **0.8546** / 0.198 | 0.8456 / 0.224 |
| de-novo paired (stacked − own flat) | +0.116 | +0.118 | +0.096 | +0.117 (221/2) |
| stacked within-cohort paired (stacked − flat) | −0.041 | −0.032 | −0.063 | −0.036 |
| marginal ECE, stacked, by depth | 0.003–0.015 | 0.003–0.016 | 0.002–0.015 | 0.003–0.016 |

**Read.** The blend did what it was built to do and paid for it on the other axis:

- **The root head came back, and then some**: 0.811 / AP 0.890, the best of the series
  (0133's 0.781 was the width effect; 1,200 strata fixed it). De-novo AP followed: 0.224,
  within 0.012 of 0123's 0.236, up from 0133's 0.198. The product's within-cohort tax
  also eased (−0.036 vs 0133's −0.063), as a stronger root factor predicts.
- **The per-node heads lost a little next to the strata**: within-cohort 0.782, paired
  −0.018 vs 0133 (55 up / 138 down, uniform from depth 3 down; depth 2 up). Still above
  0132 (+0.017) and within 0.009 of 0123. The digest says the blocks themselves are
  intact; two candidate mechanisms for the head loss, both cheap to test: (i) node
  evidence per document is halved (the strata carry the generic mass), so each node
  topic is a smaller θ entry for the head to read; (ii) the readout's top-256 θ
  truncation, which at K=306 (0133) kept 84% of the topics and at K=1498 keeps 17% — a
  document's top 256 can now be strata, dropping its node entries. (ii) is one
  re-readout: `gated-pc-readout ID=134 GPR_ARGS="--readout-mode distributed
  --readout-l2 100 --readout-theta-topm 0"` (tagged `topm0`, ~15 min, no refit).
- **Across the four topic sides, no arm dominates and every read is within ~0.02**:
  0133 wins within-cohort and de-novo AUC, 0123 wins de-novo AP by 0.012, 0134 wins the
  root head, 0132 wins nothing. The decoder-side changes of this week (a root head that
  saw the background: 0.63 → 0.81; the closure product de novo: +0.11) are five to ten
  times the size of anything the topic side moves. Insight 0095.

**Outcome among the front matter's three:** the second, softened — the root returned and
the blocks kept their identity, but the heads read them slightly less well next to
1,200 strata. Whether that is truncation (fixable, one re-readout) or mass sharing
(structural) is the one open number.

**Where this leaves the program (for the next thread).** Three topic sides are now
decodable to within two hundredths of each other under the stacked head. The choice
between them is an interpretability choice, and on that axis 0133/0134 (one legible
topic per node, textbook wherever a node's vocabulary is its own) are not close to
0123 (five-way strata splits) or 0132 (no node topics). The open legibility item is the
same-vocabulary child (DCM → its parent's cohort), unchanged by any of 0127–0134; the
open decoding item is the top-m check above.

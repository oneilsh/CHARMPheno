---
id: 78
slug: multidomain-gated-pc-measurement
status: pending
model_class: gated_pc
cohort: population_rare6
cohort_def: population_rare6
disease: rare6
# MULTI-DOMAIN: conditions (domain 0) + value-aware MEASUREMENT (domain 1). The
# stronger thesis test — measurement is the ONE non-condition domain that carried
# rare-disease signal in the hybrid branch (Marfan/GBS/EDS via labs, 0078/0079).
# Measurement uses value-aware synthetic tokens (concept_id*100+state: range
# low/normal/high, coded qualitative, presence) with per-document BINARY presence
# (bursty, no era rollup). Built fresh (no cache). Companion to 0077 (cond+drug).
extra_domains: measurement
# --- corpus / DAG: IDENTICAL to 0077 / 0076 run 7 (only the domain differs) ---
person_mod: 1
prior_obs_days: 0
doc_min_length: 10
min_n: 20
holdout_frac: 0.2
vocab_size: 5000
min_df: 20
min_patient_count: 20
window_mode: lookback     # REQUIRED for multi-domain
lookback_days: 1825
label_window_days: 365
strip_mode: both
label_mask_mode: full
n_bg: 8
tpn: 1
optimize_doc_concentration: true
# --- PC head: run 7's Path A (aggregated one-step Newton + ridge) ---
weight_y: 50.0
head_optimizer: newton
head_penalty: none
head_inner_iters: 0
head_lr: 0.3
head_newton_ridge: 0.05
grad_cavi_iters: 30
topic_trust: 0.05
weight_y_warmup_iters: 25
max_iter: 100
subsampling_rate: 0.1
tau0: 64.0
kappa: 0.51
cavi_max_iter: 100
cavi_tol: 0.001
skip_unsup_gated: false
with_dag_head: false
baseline_max_iter: 100
min_label_count: 20
eval_every: 20
num_partitions: 96
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache   # ignored on the fresh multi-domain path
---

# 0078 — Multi-domain Gated-PC (conditions + value-aware measurement), rare6

The stronger of the two multi-domain PC runs. Where 0077 adds drug (MG signal),
this adds **value-aware measurement** — the domain the hybrid branch found carried
the rare-disease signal (Marfan/GBS/EDS via labs, insights 0078/0079). Same thesis
(per-node supervised domain shaping), same config as 0077 bar the domain.

## The test

Same three bars as 0077 (vs condition-only run 7, vs the unsupervised multi-domain
twin, vs the ~6% fixed-readout ceiling), plus the direct `per_node_domain_mass`
readout: **do Marfan's / GBS's / EDS's nodes go measurement-heavy** while
condition-driven nodes stay condition-heavy? This is the run the whole thesis was
pointed at — if supervised per-node shaping can pull the lab specialists into the
macro, it shows here.

Measurement tokenization: value-aware synthetic tokens (`measurement_tokens`,
verified by `test_bigquery_measurement.py`) with per-document binary presence.

## Run

```bash
cd ~/repos/CHARMPheno && \
  git fetch origin claude/spectral-anchor-topic-k-200nqp && \
  git checkout claude/spectral-anchor-topic-k-200nqp && \
  git pull origin claude/spectral-anchor-topic-k-200nqp && \
  CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=78
```

Note: builds the corpus FRESH from BigQuery (measurement is a large, bursty table —
the assemble phase will be slower than the condition-only cached runs).

## Run log

### Run 1 (n_bg=8, tpn=1, measurement, 100 iters) — first REAL per-node specialization, hierarchy-aligned; big PC lift over the unsupervised twin; ties condition-only at the macro

Detection AP (case vs bg):

| arm | det AP |
|---|---|
| **gated_pc (multi-domain, supervised)** | **0.1472** |
| unsup_gated (multi-domain, weightY=0) | 0.1216 |
| condition-only Gated-PC (0076 run 7) | 0.148 |

- **Biggest PC delta of the arc:** gated_pc vs unsup_gated detection AP **Δ+0.0257
  (+21% rel)**; pc_topics_lr AP Δ+0.0071, AUC Δ+0.0057. Supervision does serious
  work once measurement is in — much more than drug (0077: Δ+0.0112).
- **Ties condition-only at the macro** (0.1472 ≈ run 7's 0.148; was 0.1497 at
  iter 40). So measurement adds information the supervised fit exploits *over the
  unsupervised twin*, but aggregate detection lands at condition-level — the
  information-limit caveat (0076/0079) still holds for the macro. Note θ_contrib
  flipped to 0.47/0.53: measurement carries MORE of the representation than
  conditions (a big domain even under binary presence).
- **The per-node λ-mass shows the thesis signal — and it tracks DAG DEPTH.** Unlike
  drug's flat ~0.76/0.24, measurement gives real, directional, depth-correlated
  specialization: the ANCHOR/parent nodes are the most condition-heavy, the fine
  subtypes balanced or measurement-leaning:

  | node | cond | meas | level |
  |---|---|---|---|
  | Amyloidosis | 0.87 | 0.13 | anchor |
  | SLE (anchor) | 0.81 | 0.19 | anchor |
  | Sarcoidosis | 0.79 | 0.21 | anchor |
  | Ehlers-Danlos (anchor) | 0.71 | 0.29 | anchor |
  | Myasthenia gravis (anchor) | 0.59 | 0.41 | anchor |
  | Cardiac sarcoid / AL amyloid / cerebral amyloid / … (leaves) | ~0.50 | ~0.50 | leaf |

  The clinically-right pattern: **coarse categories are condition-coded; fine
  subtypes reach toward labs to make the distinction.** This is the per-node
  domain specialization the whole multi-domain PC thesis predicted — first clear
  sighting, and it correlates with hierarchy level.

**Read:** measurement is the domain where the mechanism visibly works — strong
supervised lift over the unsupervised twin AND real hierarchy-aligned per-node
specialization. What it does NOT (yet) do is beat condition-only on aggregate
detection. But aggregate detection is the wrong yardstick for this specialization:
the fine subtypes are exactly where measurement leans in, so the payoff should show
in the CONDITIONAL sharpening readout (P(child|parent)), which this run predates
(built in commit 24cf434, mid-flight). **Next: re-run 0078 to get the conditional
table — the hypothesis is that measurement sharpens SUBTYPING where it can't move
de-novo detection.**

### Run 2 (same config + conditional readout) — sharpening hypothesis REFUTED; conditional prediction validated; dichotomy points at closure-mask

Re-ran with the conditional "sharpening" readout (P(child|parent-cohort)).

**Conditional sharpening — gated_pc vs unsup_gated (weightY=0):**

| metric | gated_pc | unsup_gated | Δ |
|---|---|---|---|
| cond AP (child\|parent) | 0.2681 | 0.2808 | **−0.0127** |
| cond AUC | 0.6203 | 0.6232 | −0.0030 |
| multiclass top-1 | 0.7783 | 0.7867 | −0.0084 |

- **Supervision does NOT help sharpening — it's marginally WORSE.** The hypothesis
  (measurement sharpens subtyping under PC) is refuted: the unsupervised gated fit
  is slightly better at P(child|parent). The conditional metrics AND θ_contrib
  (0.47/0.53) are near-identical across arms, so the hierarchy-aligned per-node
  specialization from run 1 is largely a property of the GATED MULTI-DOMAIN
  representation itself, not a PC-supervision effect (run 1's attribution corrected).
- **But the conditional FRAMING is strongly validated.** The sharpening lift is huge
  — conditioning on the parent gives a 5–29× AP jump over marginal (depth-0 cond_AP
  0.274 vs marg 0.051; depth-2 cond_AP 0.577 vs marg 0.020). Multiclass top-1 is
  clinically real: **Sarcoidosis→subtype 86%, SLE→subtype 100%, Amyloidosis→subtype
  75–79%.** "Sharpen a loose dx" is far more tractable than de-novo detection.
- **The dichotomy is the finding: supervision helps DETECTION (det AP 0.1473 vs
  0.1225, Δ+0.0247) but not SHARPENING** — because `label_mask_mode=full` trains
  each node against BACKGROUND (a detection objective), never P(child|parent), and
  by pulling a parent's subtypes toward "parent vs bg" it may blur within-parent
  distinctions.

**Read + next.** Conditional/hierarchical prediction is a real, valuable capability
(high top-1, huge lift) — already good in the unsupervised gated fit. PC as
configured optimizes the wrong thing for it. The clear test: `label_mask_mode=closure`
trains each node vs its SIBLINGS (the conditional objective) — exp 0079. Hypothesis:
closure-mask supervision improves the conditional sharpening metric where full-mask
supervision didn't.

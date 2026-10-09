---
id: 136
slug: dismech-ribbon-profile-census
status: done
model_class: gated_pc
# NOT A FIT. A readout of exp 0135's saved fit (spec 2026-10-07 §D3, WP-C):
# `inspect_topics.py --profile-census`, off-YARN, pure numpy over gated_pc_result.npz
# + the bundle meta. Decides tpn_max for the record run (0137).
#
# QUESTION. tpn=3 in 0135 is a CEILING. How many coherent, disease-specific profiles
# does each ribbon disease actually carry? Each block topic is classified
#   starved    support_frac > 0.5 (prior floor; nothing learned)
#   stratum    fed, cosine >= 0.8 to a background topic (a care-context split a shared
#              topic already explains — exps 0127-0131's "four strata" read)
#   duplicate  fed, specific, cosine >= 0.8 to a higher-evidence signature of its block
#   signature  fed, specific, distinct
# and n_profiles(d) = number of signatures.
#
# PRE-REGISTERED READS:
#   - the n_profiles histogram over nodes (0..3), overall and by depth;
#   - stratum rate (does a flat forest still grow strata inside blocks, or did 0134's
#     1,200 shared strata absorb them?) and duplicate rate (wasted capacity);
#   - exemplar multi-signature nodes WITH WORDS: are two signatures of one disease
#     clinically distinct sub-phenotypes (the EDS precedent, insight 0035: POTS / MCAS /
#     joint instability / vascular / GI), or the same profile at two evidence levels?
#   - --grep on Ehlers-Danlos, cardiomyopathy, Marfan, Gaucher, immunodeficiency.
# VERDICT RULE (printed by the tool): p90 of n_profiles < 3 -> tpn_max drops to p90
# (floor 1) for 0137; a tail AT 3 -> raise tpn_max (try 5) before the record.
#
# HPO ALIGNMENT is reported when a --credited-file covers the ribbon; the only emitted
# profile TSV today is the CV branch's (profile_eta_MONDO_0004995.tsv), so the first
# census runs without it (or with it, restricted to the ~300 CV members it covers —
# CREDITED=1 picks that file). A ribbon-wide emit-eta is a WP-C' follow-up.
#
# COST: minutes on the master, no YARN; needs the 0135 bundle meta (INSPECT_KEY
# auto-selects the newest bundle in the cache — on a cluster that also holds 0134's
# bundle, pass INSPECT_KEY explicitly if the auto-pick is wrong; the report flags a
# shape mismatch).
fit_of: 135
---

# 0136 — profile census of the ribbon fit (what is `n_profiles(d)`, what should `tpn_max` be)

Reads 0135. No cluster fit.

## Run

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0135-dismech-ribbon-smoke-tpn3-bg1200
make -C analysis/cloud inspect-topics ID=135 RESOLVE_NAMES=1 INSPECT_ARGS="--profile-census --digest-exemplars 12 --grep 'Ehlers|cardiomyopathy|Marfan|Gaucher|immunodeficiency'" | tee "$RUN"/profile_census_log.md
# optional, CV-branch members only:
make -C analysis/cloud inspect-topics ID=135 CREDITED=1 INSPECT_ARGS="--profile-census" > "$RUN"/profile_census_credited.md
```

The report is also written to `<run>/profile_census.md`.

**WP-C′ (built 2026-10-08) — the honest census.** Once `<run>/code_map.tsv` exists
(written by the fit driver on every native-Mondo run from now on, and by the first
re-readout of an older run — 0135 launch 3 provides it):

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0135-dismech-ribbon-smoke-tpn3-bg1200
# a fresh cluster has no HDFS bundle: write the meta + code map into the run dir first
# (rebuilds the bundle, ~20 min; no transform, no solve). Later fits write both themselves.
make -C analysis/cloud gated-pc-readout ID=135 GPR_ARGS="--bundle-only"
make -C analysis/cloud inspect-topics ID=135 RESOLVE_NAMES=1 INSPECT_ARGS="--profile-census --own-codes --digest-exemplars 12 --grep 'Ehlers|cardiomyopathy|Marfan|Gaucher|immunodeficiency|insomnia|hyperlipid'" | tee "$RUN"/profile_census_own_log.md
```

A fed topic with none of its node's own attesting codes in its top-15 condition
tokens is a `stratum` whatever its background cosine; own-code words are marked `†`.
The verdict line then decides `tpn_max` for 0137 honestly (the 2026-10-08 read below
was eyeballed under this rule).

## Results (2026-10-08, on 0135 launch 2)

`block topics 1173 · signature 1125 (96%) · stratum 7 (1%) · duplicate 4 · starved 37
(3%)`; `n_profiles`: 0: 6 · 1: 2 · 2: 26 · 3: 357 (91%); verdict "ceiling binding".

**Read with 0135's digest, the census over-counts signatures.** The 0.8
cosine-to-background threshold passes generic-symptom strata (insomnia / OSA / aortic
stenosis / obesity: nausea · SOB · pain // saline · ondansetron; bg cos 0.3–0.77) and
would, at a looser threshold, fail true signatures of common diseases that are
themselves population strata (hyperlipidemia 0.71, DCM 0.50). Cosine to the background
is not the discriminator; **identity is** — whether the topic's top words include the
node's own attesting codes. Multi-signature exemplars that ARE real under that rule:
EDS (POTS-dysautonomia-GI / MCAS-immune / musculoskeletal-migraine), T2D (foot-ulcer-
retinopathy-insulin / uncomplicated-metformin / vitamin-D-anaemia), heart failure
(systolic / acute / CKD-comorbid), CVID (3), Marfan (3), DCM (signature + acute
decompensation), HCM (2). So the ceiling IS binding where the disease is well coded,
and `tpn_max` 5 is the record setting — once the census counts honestly.

**WP-C′ (next):** the fit driver writes the native code map `(std_cid, node_cid)` to
the run dir (small; it is already in hand at build time); `--profile-census` gains
`--own-codes` marking own-code tokens (`†`) in the words and classing a fed topic with
no own-code token in its top-m as `stratum` regardless of background cosine. The
0.8-cosine classes stay as secondary flags.

## Results, own-code rule (2026-10-09, on 0135 launch 2)

`5881 (code, node) pairs; 283/391 nodes have an own code in the condition vocab` ·
`signature 578 (49%) · stratum 554 (47%) · duplicate 4 · starved 37 (3%)` ·
n_profiles over all nodes 0: 113 · 1: 75 · 2: 106 · 3: 97 (25% at the ceiling).

- **The rule sorts the cases cosine could not.** Insomnia's nausea·SOB·pain topic,
  hyperlipidemia's skin topic, asthma's and osteoarthritis's ED-symptom topics, COPD's
  chronic-back-pain topic and GERD's hypothyroid topic are strata; every disease's
  coded profile (DCM's carvedilol·furosemide·spironolactone, asthma's step-therapy,
  CKD by stage, T2D complicated vs uncomplicated) is a signature.
- **Two artefacts of the first rule, fixed in the tool the same day.** (i) The 108
  nodes with no own code in the vocab (their attesting codes fall under min_df or the
  5,000 cap) were classed all-stratum and inflated the 0-profile bin (113); they are
  now reported apart and fall back to the cosine rule, and the verdict reads the
  decidable nodes. (ii) The cosine still overrode identity: atrial fibrillation's
  best own-code topic (bg cos 0.81 — the background duplicated AF, not the reverse)
  was called a stratum; own codes now decide and the cosine is a flag.
- **Verdict holds: the ceiling is binding.** On the 283 decidable nodes roughly a third
  sit at 3; `tpn_max` 5 for 0137.
- **Common diseases split by comorbidity context, rare ones by phenotype.** Hypertension
  (diabetic-CAD / metabolic / musculoskeletal), T2D (complicated / uncomplicated /
  vitamin-anaemia), CKD (by stage / ESRD-dialysis / transplant). Marfan (aortic /
  ocular-valvular / skeletal), CVID (3), HCM (obstructive / arrhythmic), DCM (chronic /
  decompensated). EDS proper carries one signature next to hEDS's two (R1a working:
  the "EDS type 3" code is hEDS's own code, not EDS's).
- **Peripartum cardiomyopathy's own codes are pregnancy codes.** All three topics are
  pregnancy (Finding related to pregnancy†, High risk pregnancy†, trimester codes†,
  Gestation period 8 weeks†): generic pregnancy concepts are in its code map, which is
  why its evidence (6e4) is four times DCM's. A code-map defect (a Mondo xref or a
  climb landing), not a topic-model one; to trace before 0137 (list the node's own codes from `<run>/code_map.tsv` + `bundle_meta.json`).
- Restrictive CM 0 profiles (unfed, ev ≤ 533) and Tako-tsubo 1 match their weak
  de-novo AUCs (0.738 / 0.755).


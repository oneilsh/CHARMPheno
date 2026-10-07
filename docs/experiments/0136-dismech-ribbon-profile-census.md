---
id: 136
slug: dismech-ribbon-profile-census
status: planned
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

## Results

(pending 0135)

---
id: 138
slug: wpd-conditional-diagnosis-voi-on-0137
status: pending
model_class: gated_pc
# NOT A FIT. WP-D (spec 2026-10-07 §D4) on the ribbon record 0137, no refit:
#   part 1 — code-level VOI tables over Mondo candidate sets (off-cluster, numpy over the
#            saved λ + census signatures + readout_dag.json; `inspect_topics --voi`);
#   part 2 — conditional diagnosis on held-out patients: per readout-DAG edge P -> c, the
#            within-P-cohort cond AUC (the sober column), top-1 vs majority and ECE, for
#            the flat heads and the stacked product (`gated_pc_readout
#            --conditional-from-heads stacked_mass99`: test split only, saved heads, no
#            solve).
# CANDIDATE SETS (decided 2026-10-07): Mondo ancestors — first two: connective tissue
# disorders and cardiomyopathies; the full per-edge table comes free from part 2.
# PRE-REGISTERED READS:
#   part 1: are the top codes clinically discriminating (e.g. aortic dilatation/dissection
#           for Marfan vs EDS; LBBB / systolic HF for DCM vs HCM's outflow obstruction),
#           or care-context codes? Own codes are excluded from the ranking by default.
#   part 2: cond AUC by depth for flat vs stacked; top-1 above the majority baseline for
#           how many parents; ECE (calibration is the prerequisite for VOI to mean bits).
# NEXT (part 3, not built): simulated acquisition on held-out patients — hide codes,
# reveal in EIG order vs random vs information-content order, measure posterior
# concentration (the 2026-08-14 report's evaluation of VOI).
fit_of: 137
---

# 0138 — WP-D on 0137: code-level VOI over Mondo groups, and conditional diagnosis

## Run

Part 1 (minutes, master only; names resolved off-YARN):

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0137-dismech-ribbon-record-tpn5-v2codemap
make -C analysis/cloud inspect-topics ID=137 RESOLVE_NAMES=1 INSPECT_ARGS="--voi '^cardiomyopathy\$|^connective tissue disorder\$' --own-codes --relevance-lambda 0.6 --own-top 5 --results-file results_readout_stacked_mass99.json --voi-top 12" 2>/dev/null | grep -vE "^\[inspect-topics\]|^make" | cut -c1-170
```

Census re-read with the usage dashboard's interpretation tools (relevance-ranked words,
KL from the corpus marginal, ANCHOR flag; own code must rank in the top 5 by RELEVANCE,
which is the strict rule for common diseases), before reading VOI:

```bash
make -C analysis/cloud inspect-topics ID=137 RESOLVE_NAMES=1 INSPECT_ARGS="--profile-census --own-codes --relevance-lambda 0.6 --own-top 5 --top-words 4 --digest-exemplars 4 --grep 'Ehlers-Danlos syndrome,|Marfan|Dilated cardio|thrombophilia'" 2>/dev/null | grep -vE "^\[inspect-topics\]|^make" | cut -c1-170 | sed -n '1,70p'
```

Part 2 (bundle rebuild on a fresh cluster ~20 min, test-split transform, one collect; no
solve):

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0137-dismech-ribbon-record-tpn5-v2codemap
nohup make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--conditional-from-heads stacked_mass99" > "$RUN"/conditional_log.md 2>&1 &
grep -A 40 "conditional diagnosis" "$RUN"/conditional_log.md | cut -c1-170
```

## Run log

**2026-10-12 — census re-read with relevance ranking and the strict own-code rule**
(`--relevance-lambda 0.6 --own-top 5`, thin < 0.1). Block topics 1,945: signature 1,044
(54%) · stratum 387 (20%) · thin 436 (22%) · duplicate 8 · starved 70 (4%). Decidable nodes
(280): n_profiles 0: 5 · 1: 60 · 2: 96 · 3: 68 · 4: 42 · 5: 9 — median 2, p90 4, 3% at the
ceiling (was 16% under the lenient rule). Rare diseases now read cleanly: Marfan = aortic
(ectasia, atenolol, warfarin) + dissection/pseudoaneurysm + skeletal (pectus, scoliosis),
the evidence-36 carotid/goiter topic thin; hEDS = EDS type 3 + mast cell + somatic
dysfunction, and a musculoskeletal profile, the hearing-loss topic thin and the
hypermobility-POTS topic a stratum (no hEDS own code in its top 5 — arguably a real hEDS
profile coded as hypermobility; watch); DCM = DCM/LBBB/chronic systolic HF // sacubitril,
carvedilol + a primary-CM/PVC profile, the decompensated-HF topic a stratum. Common
diseases still carry 3–4 (hypertension: diabetic-neuropathy, metabolic, lumbar-pain —
own code ranked by relevance still clears top 5 there); the ANCHOR flag marks the
own-code-dominated topics. **`tpn_max` 5 confirmed with headroom**; a per-disease tpn
from this census is the efficiency option for the next fit.

## Results

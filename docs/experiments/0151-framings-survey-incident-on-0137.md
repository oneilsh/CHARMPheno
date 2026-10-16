---
id: 151
slug: framings-survey-incident-on-0137
status: done
model_class: gated_pc
# NOT A FIT. Insight 0103: the survey's best which-member groups are called from the
# diagnosis in other words (synonym codes, confirmatory tests, treatments), because
# most label-year positives are prevalent and their pre-index record documents the
# diagnosis. This repeats 0150 exactly (same groups, engines, caps, cross-fit, framing
# 2 on E5) with ONE change: test patients are restricted to INCIDENT ones — no
# attestation of any member, or of the group node, or of anything under them, anywhere
# before the index (E4 sidecar + Mondo is-a). Their pre-index record is a
# pre-diagnosis record. Training unchanged (label-year, single member; 0144's lesson).
# PRE-REGISTERED READS: (1) kappa distribution vs 0150 (quartiles 0.07 / 0.18 / 0.34;
# 75 groups >= 0.3) — expected to fall; how many groups keep kappa >= 0.3 on >= 40
# incident patients; (2) whether the surviving groups' E5 reasons are clinical
# features rather than synonyms / tests / treatments; (3) framing 2 (gap closed, VOI −
# table) under the same restriction; (4) how many groups drop below 40 patients.
fit_of: 137
---

# 0151 — framings survey on incident (pre-diagnosis) patients (0137)

## Run

Rebuild the sidecar on this cluster first (~4 min), then the survey (as 0150, several
hours), then the analysis.

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0137-dismech-ribbon-record-tpn5-v2codemap
nohup bash -c '
  make -C analysis/cloud build-conversion-sidecar ID=137 || exit 1
  make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--differential . --diff-survey 25 --diff-incident --diff-max-patients 1500 --diff-crossfit --diff-next-question --diff-nq-hide 0.8 --acq-tag survey_inc"
' > "$RUN"/survey_inc_log.md 2>&1 &
```

Report back:

```bash
python3 analysis/cloud/survey_analysis.py "$RUN" survey_inc | cut -c1-220
```

## Run log

**2026-10-15/16 — run** (launch 1 killed after two groups at ~440 s each: the anti-join
shuffled the test frame; fixed by broadcasting the excluded persons, then 30–90 s per
group). 326 candidate sets, 198 distinct member sets scored (>= 40 incident patients).

## Results

| | 0150 (all label-year positives) | **0151 (incident only)** |
|---|--:|--:|
| distinct member sets scored | 243 | 198 |
| kappa quartiles (best engine) | 0.07 / 0.18 / 0.34 | **0.04 / 0.09 / 0.22** |
| groups with kappa >= 0.3 / >= 0.5 | 75 / 29 | **32 / 6** |
| E6 − E3 kappa quartiles | −0.001 / +0.016 / +0.056 | +0.000 / +0.017 / +0.058 |
| Spearman kappa vs F2 gap closed / VOI − table | +0.54 / +0.46 | +0.37 / +0.37 |
| Spearman kappa vs F3 AUC θ / raw | +0.11 / +0.21 | +0.09 / +0.12 |

1. **Read (1): skill halves on pre-diagnosis records.** Median kappa 0.18 -> 0.09;
   groups at >= 0.3 fall from 75 to 32 (16% of the scored sets), at >= 0.5 from 29 to 6.
2. **Read (2): what carries the survivors is now mostly plausible pre-diagnosis
   signal, plus base-rate proxies.** Precursors and early findings: Sjogren syndrome
   (tear film insufficiency, fibromyalgia), thoracic aortic aneurysm (ectasia of the
   thoracic aorta), cutaneous melanoma (actinic keratosis, neoplasm of uncertain
   behaviour of skin), HCV (HCV antibody coded positive and high AST BEFORE any HCV
   diagnosis code — the lab precedes the coded diagnosis). Base-rate / comorbidity
   proxies: hypertension, hyperlipidemia, type 2 diabetes (Bell's palsy, AAA), generic
   symptoms (hepatitis A). The synonym and treatment codes of 0150 are gone.
3. **Read (3): framing 2 is still ~ the population table** (VOI − table near 0 for most
   groups; readable only in a few, e.g. gram-negative bacterial infections +0.063,
   parasitic infections +0.046).
4. **Read (4): sample size.** Many surviving groups rest on 44–100 incident patients;
   kappa correlates −0.44 with n — part of the top of the list is small-sample noise.

**Read.** An honest "early which-member" signal exists for a minority of groups and
is clinically legible where it exists (precursor findings, a lab before the coded
diagnosis), but it is modest, and asking further questions adds nothing over a
population table. Insight 0103 updated.

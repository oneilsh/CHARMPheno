---
id: 143
slug: prevalence-check-on-0137
status: done
model_class: gated_pc
# NOT A FIT. Insight 0100's measurement: how much of 0137's case-finding AUC is
# prevalent recall? Per readout node X, the saved stacked_mass99 heads (flat and stacked)
# scored over ALL test patients vs with PRIOR-X patients set aside — prior = a mapped
# attestation code of X or a label descendant in the 5-year pre-index condition
# features (`prevalence_check.py`; a lower bound: older diagnoses count as incident).
# PRE-REGISTERED READS: (1) macro AUC all → incident, flat and stacked; (2) the gap by
# the prior share of positives (expected: grows with it; chronic common diseases worst);
# (3) how many nodes lose > 0.05; (4) the named worst offenders (RA, SLE, leiomyoma
# predicted from 0142). Decides whether the label redesign (incident arm for heads and
# evaluation; all-ever attestation for topic learning) precedes everything else.
fit_of: 137
---

# 0143 — prevalence check: AUC with prior-diagnosis patients set aside (0137)

## Run

Test-split transform + two collects (scores; one sparse condition-code list per patient).
Bundle rebuild first if this cluster has no cache (~20 min).

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0137-dismech-ribbon-record-tpn5-v2codemap
nohup make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--prevalence-check" > "$RUN"/prevalence_log.md 2>&1 &
grep -E "^## |^\| 0|^- " "$RUN"/prevalence_check.md | cut -c1-170
```

## Run log

**2026-10-13 — run.** 535 readout nodes with >= 20 positives both ways.

## Results

| arm | macro AUC all | macro AUC incident | gap median | gap p90 | nodes losing > 0.05 |
|---|--:|--:|--:|--:|--:|
| stacked | 0.8176 | **0.7022** | +0.120 | +0.195 | 439 / 535 |
| flat | 0.7269 | **0.6163** | +0.110 | +0.205 | 399 / 535 |

By the share of a node's label-year positives that had a prior code (stacked):

| prior share | nodes | AUC all | AUC incident | median gap |
|---|--:|--:|--:|--:|
| 0.0–0.1 | 27 | 0.715 | 0.715 | +0.000 |
| 0.1–0.3 | 59 | 0.759 | 0.724 | +0.035 |
| 0.3–0.6 | 251 | 0.803 | 0.700 | +0.104 |
| 0.6–1.0 | 198 | 0.867 | 0.697 | +0.166 |

Median prior share of positives: 0.53. Largest stacked gaps: cardiovascular disorder
0.870 → 0.623, celiac 0.796 → 0.551, multiple sclerosis 0.915 → 0.675, pituitary tumour
0.852 → 0.619, thoracic cancer 0.901 → 0.671, hypertensive disorder 0.884 → 0.658,
breast carcinoma 0.905 → 0.682, vitiligo 0.858 → 0.625.

**Read.** About half of every node's label-year positives are prevalent re-codes, and the
readout's AUC above ~0.70 is almost entirely recognizing them: the incident AUC is ~0.70
(stacked) / ~0.61 (flat) in EVERY prior-share band, and the "all" AUC rises with the
prevalent share. Genuine new-diagnosis case-finding on 0137 is ~0.70 stacked — and that is
a lower bound only on the gap (5-year lookback). Two caveats in the model's favour: these
heads were TRAINED on the mixed labels, and the topics were gated on label-year
attestation only (prevalent-not-recoded patients never taught X's topics). So the label
redesign (insight 0100) precedes everything else; its first cheap step is readout-side —
refit the heads with prior cells masked (incident training) on the saved fit.

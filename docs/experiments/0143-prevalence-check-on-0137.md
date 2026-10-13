---
id: 143
slug: prevalence-check-on-0137
status: planned
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

## Results

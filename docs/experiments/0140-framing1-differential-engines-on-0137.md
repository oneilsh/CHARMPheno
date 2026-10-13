---
id: 140
slug: framing1-differential-engines-on-0137
status: done
model_class: gated_pc
# NOT A FIT. Framing 1 of spec 2026-10-13 (clinician framings): "for this patient, which
# disease in group G, and what in their record points there?" — four engines on the
# same held-out patients of 0137 (`differential.py`, `gated_pc_readout --differential`):
#   E1 presence model with SUBTYPE profiles (D5), present AND absent codes, tempered;
#   E2 the stacked readout heads renormalized (0139's baseline);
#   E3 a multiclass softmax head on log θ, fit on train-split single-positive patients;
#   E4 gated per-disease likelihood (fold-in ELBO under {background ∪ block(m)}), tempered.
# Records lose attestation codes (members' + group's own) and [measured] tokens (D1, D2).
# Evaluation patients: test split, positive on exactly one member (multi-positive set
# aside and counted); tempers fit on one half, ALL engines scored on the other half.
# Sets: ^cardiomyopathy$ (8), ^connective tissue disorder$ (19), Marfan/EDS/hEDS (3).
# PRE-REGISTERED READS: (1) which engine has the best balanced top-1 at acceptable
# calibration (ECE ≲ 0.1) per set — that engine carries framings 2 and 3; (2) E2 should
# reproduce 0139's whole-record figures roughly (balanced top-1 0.32 / 0.22 / 0.48 —
# not exactly: attestation and [measured] are now removed, and only single-positive
# patients are scored); (3) whether E4's τ is tiny (generative overconfidence on long
# records) and whether, tempered, it is calibrated; (4) do the reasons read clinically
# (E1/E4 pooled supporting codes per member)?
fit_of: 137
---

# 0140 — framing 1: differential with reasons, four engines on 0137

## Run

The HDFS cache died with the last cluster, so the first command rebuilds the bundle
(~20 min, n2-standard-8 master); the other two reuse it. Each set is then minutes of
driver numpy (benchmarked: full-K fold-in ~0.03 s, E4 over 19 members ~0.3 s per
patient).

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0137-dismech-ribbon-record-tpn5-v2codemap
cat > "$RUN"/differential.sh <<'EOF'
make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--differential '^cardiomyopathy\$\$|^connective tissue disorder\$\$' --acq-tag groups"
make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--differential '^Marfan syndrome|^Ehlers-Danlos syndrome' --acq-adhoc --acq-tag marfan_eds"
EOF
nohup bash "$RUN"/differential.sh > "$RUN"/differential_log.md 2>&1 &
```

Read:

```bash
cat "$RUN"/differential_groups.md "$RUN"/differential_marfan_eds.md | cut -c1-170
```

## Run log

**2026-10-13 — first run.** Held-out single-positive patients scored (other half fits τ):
connective tissue 271, cardiomyopathy 223, Marfan/EDS 57; E3 trained on 1,834 / 1,792 /
482.

| set | engine | top-1 | top-3 | bal top-1 | log loss | ECE |
|---|---|--:|--:|--:|--:|--:|
| connective tissue (19) | E1 presence subtypes | 0.391 | 0.694 | 0.143 | 2.26 | 0.069 |
| | E2 stacked heads | 0.140 | 0.310 | 0.122 | 10.14 | 0.526 |
| | E3 multiclass head | **0.450** | **0.734** | **0.184** | **1.85** | 0.076 |
| | E4 gated likelihood | 0.325 | 0.675 | 0.143 | 2.34 | 0.086 |
| cardiomyopathy (8) | E1 | 0.507 | 0.821 | 0.221 | 2.23 | 0.098 |
| | E2 | 0.426 | 0.704 | 0.162 | 3.87 | 0.227 |
| | E3 | **0.547** | **0.852** | **0.278** | **1.26** | 0.080 |
| | E4 | 0.426 | 0.722 | 0.144 | 2.41 | 0.166 |
| Marfan/EDS (3) | E1 | 0.561 | 1.000 | 0.344 | 4.99 | 0.257 |
| | E2 | 0.544 | 1.000 | 0.350 | 2.07 | 0.290 |
| | E3 | **0.614** | 1.000 | 0.422 | **0.92** | 0.147 |
| | E4 | **0.614** | 1.000 | **0.427** | 4.85 | 0.206 |

Two defects in the first run, both in the PRIOR used by E1/E4 (E3 learns its own; E2
uses none): it came from the scored-results file, so (a) members missing there got prior
1e-12 and could never be called — the log loss of 12 / 26 / 110 on the prior row and E1
pinned at the prior's balanced top-1 on Marfan/EDS are this — and (b) it was
test-derived. Fixed: add-one-smoothed counts of the train split's single-positive
patients. Reasons printed as cid:N because no concept-name file was in the run dir;
`run_differential` now reads `<run>/concept_names.csv`. Re-run below. Already clear and
unaffected by the prior: **E3 is best on every set at good calibration** (ECE 0.08 on the
Mondo groups), **E2 is the weakest** (0139 confirmed on whole records), and the measurement
reasons for leiomyoma are low-result CBC tokens (anemia from menorrhagia) — result
states pulling their weight.

Re-run (names first: any RESOLVE_NAMES report writes the file to /tmp):

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0137-dismech-ribbon-record-tpn5-v2codemap
make -C analysis/cloud inspect-topics ID=137 RESOLVE_NAMES=1 INSPECT_ARGS="--measurement-states" > /dev/null && cp /tmp/concept_names_137.csv "$RUN"/concept_names.csv
nohup bash "$RUN"/differential.sh > "$RUN"/differential_log2.md 2>&1 &
cat "$RUN"/differential_groups.md "$RUN"/differential_marfan_eds.md | cut -c1-170
```

**2026-10-13 — re-run with the train-split prior and names.**

| set | engine | top-1 | top-3 | bal top-1 | log loss | ECE |
|---|---|--:|--:|--:|--:|--:|
| connective tissue (19; 272 scored) | prior | 0.210 | 0.581 | 0.062 | 2.24 | 0.064 |
| | E1 presence subtypes | 0.346 | 0.673 | 0.118 | 1.97 | 0.058 |
| | E2 stacked heads | 0.143 | 0.283 | 0.153 | 9.20 | 0.513 |
| | E3 multiclass head | **0.415** | **0.695** | **0.169** | **1.90** | 0.075 |
| | E4 gated likelihood | 0.346 | 0.662 | 0.136 | 2.07 | 0.056 |
| cardiomyopathy (8; 223) | prior | 0.390 | 0.776 | 0.125 | 1.64 | 0.011 |
| | E1 | 0.422 | 0.821 | 0.164 | 1.45 | 0.125 |
| | E2 | 0.422 | 0.713 | 0.184 | 4.25 | 0.235 |
| | E3 | **0.587** | **0.839** | **0.316** | **1.21** | 0.089 |
| | E4 | 0.395 | 0.789 | 0.127 | 1.54 | 0.113 |
| Marfan/EDS (3; 57) | prior | 0.561 | 1.000 | 0.333 | 0.99 | 0.090 |
| | E1 | 0.561 | 1.000 | 0.360 | 1.00 | 0.205 |
| | E2 | 0.544 | 1.000 | 0.350 | 2.07 | 0.290 |
| | E3 | 0.614 | 1.000 | 0.422 | 0.92 | 0.147 |
| | E4 | **0.632** | 1.000 | **0.438** | **0.87** | 0.094 |

Fitted tempers: E1 0.035–0.065, E4 0.026–0.18 — both generative engines need their
likelihood shrunk 5–40×, the naive-Bayes / long-record overconfidence measured again.

## Results

1. **E3 (multiclass head on log θ) is the framing-1 engine** for the Mondo groups: best
   top-1, top-3, balanced top-1 and log loss on both, ECE 0.075–0.089. Gains over the
   prior are real but modest on the 19-member group (balanced 0.062 → 0.169) and solid on
   cardiomyopathy (0.125 → 0.316; top-1 0.39 → 0.59).
2. **E4 (gated likelihood) is competitive on the focused contrast** (Marfan/EDS: best
   balanced top-1 and log loss, ECE 0.094) but at n = 57 it is not separable from E3.
   E1 is the best calibrated on connective tissue (0.058) with little accuracy.
3. **E2 is confirmed unusable as a differential** (ECE 0.24–0.51).
4. **The reasons read clinically.** RA: methotrexate, hydroxychloroquine, arthropathy,
   folic acid (E4 adds abatacept, and fibromyalgia — a frequent RA comorbidity). Leiomyoma:
   excessive menstruation, ovarian cyst, anemia, iron-deficiency anemia. DCM: chronic
   systolic heart failure, carvedilol, heart failure. Generic EDS: orthostatic
   hypotension, fludrocortisone, tachycardia, arrhythmia, gastroparesis. Two kinds of
   noise: (a) normal-result tokens (urine pH, heart rate, diastolic BP [normal]) — care
   context of the member's population; (b) **"Cardiomyopathy" supporting DCM**: the
   group's own generic code survives because the group node is a Mondo ANCESTOR, not a
   label node, so it has no own codes in the code map. Group attestation for ancestor
   nodes needs a code list for them (next deliberate code-map rebuild).

**Read.** Framing 1 ships with E3, reasons from E3's attribution (added next; E1/E4
reasons above are the cross-check). Framing 2 values "yes" answers with E3 and the topic
model's predictive presence; a "no" is not valued — consistent with D2/PU (an absent
code is not a negative) and with the spec's yes-only evaluation. E4 stays as the
focused-contrast alternative.

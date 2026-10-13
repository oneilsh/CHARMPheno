---
id: 140
slug: framing1-differential-engines-on-0137
status: running
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

## Results

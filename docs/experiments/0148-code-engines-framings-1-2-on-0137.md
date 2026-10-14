---
id: 148
slug: code-engines-framings-1-2-on-0137
status: planned
model_class: gated_pc
# NOT A FIT. Insight 0101 (raw codes beat θ by ~0.04 on new diagnoses, most for the
# rarest nodes) carried to the framings that matter (1 and 2). Two engines beside E1-E4,
# on the SAME filtered records (set attestation codes, name-matched concepts and
# [measured] removed), the same train / test patients and the same metrics as 0140/0141:
#   E5 — softmax on the record's codes: binary presence of tokens carried by >= 5
#        training patients (top 3,000) + log(1 + distinct codes);
#   E6 — the same plus standardized log θ over E3's features (bg + member blocks).
# Framing 2 (20% visible, 0141's informative regime) adds "VOI (codes)" — P(yes) from
# the code head's per-member presence rates, the differential after a YES by flipping
# that input (exact; no θ re-fold for E5) — and scores every strategy's reveals under
# the code differential too. The code engine for framing 2 is E5 or E6, whichever has
# the lower log loss on the τ-fit half (never the evaluation half).
# PRE-REGISTERED READS: (1) framing 1 — balanced top-1 and calibration of E5/E6 vs E3
# (0140: connective tissue 0.169, cardiomyopathy 0.297, Marfan/EDS E4 0.438); a gain
# > ~0.03 balanced top-1 moves the framing-1 engine off θ-only; (2) does E6 beat E5
# (θ adds to codes)? (3) framing 2 — P(true)/top-1 under the code differential, and
# whether VOI (codes) beats the population table (0141: patient-specific VOI ~= table
# under E3); (4) the visible-to-full gap under codes vs E3.
fit_of: 137
---

# 0148 — code engines for framings 1 and 2 (0137)

## Run

Driver-side numpy after one small collect per set (as 0140/0141). Both sets, framing 1
and framing 2 (80% of each record hidden) in one pass each.

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0137-dismech-ribbon-record-tpn5-v2codemap
cat > "$RUN"/code_engines.sh <<'EOF2'
make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--differential '^cardiomyopathy\$\$|^connective tissue disorder\$\$' --diff-next-question --diff-nq-hide 0.8 --acq-tag groups_codes"
make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--differential '^Marfan syndrome|^Ehlers-Danlos syndrome' --acq-adhoc --diff-next-question --diff-nq-hide 0.8 --acq-tag marfan_eds_codes"
EOF2
nohup bash "$RUN"/code_engines.sh > "$RUN"/code_engines_log.md 2>&1 &
```

Report back:

```bash
for f in groups_codes marfan_eds_codes; do grep -E "^## |^\| (E[0-9]|prior)|^code differential|^\| (visible|VOI|population|likely|random|full)" "$RUN"/differential_$f.md | cut -c1-200; done
```

## Run log

## Results

---
id: 149
slug: differential-more-signal-on-0137
status: planned
model_class: gated_pc
# NOT A FIT. Insight 0102 (which-member differentials are limited by the record; engines
# span ~0.04) — three levers on the same engines (E1-E6), same filtered records:
#   (3) MORE SCORED PATIENTS: the test cap raised (600 -> 5,000, i.e. all) and CROSS-FIT
#       (tempers fit on each half in turn, every patient scored once);
#   (1) EVER-CODED TRAINING: a training patient's members = every member ever coded in
#       the whole record (E4 sidecar, Mondo is-a roll-up), not only the label year;
#       train cap 2,000 -> 8,000. Test truth stays the label year (comparable to 0148);
#   (2) MULTI-POSITIVE PATIENTS: coded with 2+ members, now soft targets in training
#       (unit weight split evenly) and scored apart on test (top-1/top-3 = any true
#       member; all-in-top-3).
# Run A = (3) alone (the new, larger baseline); run B = (3) + (1) + (2), plus framing 2
# (20% visible). A vs 0148 = noise check; B vs A = what more training signal buys.
# PRE-REGISTERED READS: (i) A's balanced top-1 within ~0.03 of 0148 (same engines, more
# patients) — and the E6-vs-E3 gap, now readable; (ii) B vs A: a rise > ~0.03 balanced
# top-1 for the multiclass engines = training signal was a limit (insight 0102's lever
# 1); flat = the record is the limit; (iii) how many train patients ever-coding and
# multi-positive add; (iv) multi-positive test patients: does the true pair land in the
# top 3?
fit_of: 137
---

# 0149 — differential with more signal: bigger evaluation, ever-coded and multi-positive training (0137)

## Run

Driver-side, as 0148, with more patients (expect a few times 0148's time; E4 is one
fold-in per member per test patient). Needs the sidecar on this cluster (rebuilt
2026-10-14).

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0137-dismech-ribbon-record-tpn5-v2codemap
cat > "$RUN"/diff_signal.sh <<'EOF2'
A="--diff-max-patients 5000 --diff-crossfit"
B="$A --diff-max-train 8000 --diff-train-ever --diff-multi --diff-next-question --diff-nq-hide 0.8"
make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--differential '^cardiomyopathy\$\$|^connective tissue disorder\$\$' $A --acq-tag groups_A"
make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--differential '^Marfan syndrome|^Ehlers-Danlos syndrome' --acq-adhoc $A --acq-tag marfan_eds_A"
make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--differential '^cardiomyopathy\$\$|^connective tissue disorder\$\$' $B --acq-tag groups_B"
make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--differential '^Marfan syndrome|^Ehlers-Danlos syndrome' --acq-adhoc $B --acq-tag marfan_eds_B"
EOF2
nohup bash "$RUN"/diff_signal.sh > "$RUN"/diff_signal_log.md 2>&1 &
```

Report back:

```bash
grep -E "single-positive test patients" "$RUN"/diff_signal_log.md | cut -c1-220
for f in groups_A marfan_eds_A groups_B marfan_eds_B; do echo "### $f"; grep -E "^## |^\| (E[0-9]|prior)|^multi-positive|^code differential|^\| (visible|VOI \(codes\)|population|full)" "$RUN"/differential_$f.md | cut -c1-200; done
```

## Run log

## Results

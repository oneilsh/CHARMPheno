---
id: 149
slug: differential-more-signal-on-0137
status: done
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

**2026-10-14 — run.** Test (all, cross-fit): connective tissue 3,478 single-member + 354
multi; cardiomyopathy 447 + 28; Marfan/EDS 115 + 1 (0148's cap of 600 had bound on
connective tissue). Train: A 1,852 / 1,792 / 482 (label year, single); B 8,000 (cap
binding) / 5,857 / 1,070 with 1,867 / 1,570 / 313 multi-positive.

## Results

**Balanced top-1 (log loss):**

| engine | CT A | CT B | CM A | CM B | M/EDS A | M/EDS B |
|---|--:|--:|--:|--:|--:|--:|
| prior | 0.053 | 0.053 | 0.125 | 0.125 | 0.333 | 0.333 |
| E3 θ | 0.158 (1.783) | 0.136 (1.781) | 0.290 (1.246) | 0.244 (1.319) | 0.452 (0.787) | 0.389 (0.872) |
| E4 gated | 0.125 | 0.117 | 0.132 | 0.136 | 0.479 | 0.500 |
| E5 codes | 0.153 (1.792) | 0.147 (1.699) | 0.286 (1.213) | 0.224 (1.306) | 0.400 (0.812) | 0.444 (0.825) |
| **E6 θ + codes** | **0.171 (1.741)** | **0.175 (1.632)** | **0.299 (1.190)** | 0.232 (1.284) | **0.500 (0.717)** | 0.456 (0.822) |

Connective tissue top-1 (A): E6 0.491, E3 0.463, E5 0.476; top-3 E6 0.754.

1. **Read (i): A reproduces 0148** within noise, now on 3,478 patients for connective
   tissue: E6 is the best engine (+0.028 top-1, −0.04 log loss vs E3 at n = 3,478 —
   readable), codes alone ≈ θ alone.
2. **Read (ii): more training signal does NOT help.** B vs A: connective tissue
   balanced top-1 flat for E6 (+0.004) though log loss improves (1.741 → 1.632);
   cardiomyopathy and Marfan/EDS get WORSE (E6 −0.07, −0.04). The prior row's ECE
   jumps (CT 0.022 → 0.126): the ever-coded training population has a different member
   mix from the label-year test truth — training on it shifts the heads' priors away
   from the population they are scored on. "More data" here is also "different data".
3. **Read (iv): multi-positive patients** (CT, 354): E6 puts a true member first for
   63% (prior 24%), one in the top 3 for 92%, BOTH/all in the top 3 for 41% (prior
   30%). Cardiomyopathy's 28 are too few to read.
4. **Framing 2 (B): unchanged** — VOI (codes) = population table (CT k = 5: 0.253 vs
   0.248 P(true); CM 0.295 vs 0.290); Marfan/EDS does not move.

**Read.** The engine question is settled for these sets: E6 (θ + codes, label-year
training) is the framing-1 engine. Neither more scored patients nor ever-coded /
multi-positive training lifts the ceiling; training on ever-coded members mostly adds a
population shift. Insight 0102 stands, with tighter numbers.

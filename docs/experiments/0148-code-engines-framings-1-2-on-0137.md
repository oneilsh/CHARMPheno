---
id: 148
slug: code-engines-framings-1-2-on-0137
status: done
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

**2026-10-14 — run.** Connective tissue 272 scored (1,852 train), cardiomyopathy 223
(1,792), Marfan/EDS 57 (482). Framing-2 code engine chosen on the τ-fit half: E6 in all
three sets.

## Results

**Framing 1 — balanced top-1 (log loss, ECE):**

| engine | connective tissue (19) | cardiomyopathy (8) | Marfan / EDS (3) |
|---|--:|--:|--:|
| prior | 0.059 | 0.125 | 0.333 |
| E3 θ multiclass | 0.163 (1.877, 0.074) | 0.282 (1.236, 0.075) | 0.396 (0.822, 0.108) |
| E4 gated likelihood | 0.136 | 0.139 | 0.460 (0.752, 0.081) |
| **E5 codes** | 0.178 (1.911, 0.102) | 0.279 (1.158, 0.099) | 0.375 (0.802, 0.061) |
| **E6 θ + codes** | **0.180** (1.876, 0.106) | **0.299** (1.140, 0.093) | **0.472** (0.752, 0.071) |

1. **Read (1): not met.** E6 beats E3 by +0.017 / +0.017 balanced top-1 on the Mondo
   groups (+0.076 on Marfan/EDS, where E4 was already 0.460) — under the 0.03 bar, and
   within noise at 57–272 scored patients. The 0147 gap (codes +0.04 on new-diagnosis
   AUC) does NOT carry over to "which member": with the members' own codes removed, θ
   and codes are about equal here.
2. **Read (2): θ adds to codes.** E6 ≥ E5 on balanced top-1 and log loss in all three
   sets, and E6 has the best log loss everywhere (ties E4 on Marfan/EDS). Codes are a
   little less calibrated (ECE 0.06–0.11 vs E3's 0.07–0.11).

**Framing 2 — 20% of each record visible, P(true) / top-1 under each differential:**

| | CT E3 | CT codes (E6) | CM E3 | CM codes (E6) | M/EDS codes (E6) |
|---|--:|--:|--:|--:|--:|
| visible record | 0.201 / 0.316 | 0.238 / 0.383 | 0.315 / 0.491 | 0.300 / 0.491 | 0.513 / 0.632 |
| own VOI, k = 5 | 0.213 / 0.357 | **0.262 / 0.409** | 0.329 / 0.537 | **0.325 / 0.518** | 0.515 / 0.632 |
| population table, k = 5 | 0.216 / 0.361 | 0.257 / 0.401 | 0.343 / 0.518 | 0.316 / 0.495 | 0.514 / 0.632 |
| random hidden, k = 5 | 0.207 / 0.338 | 0.250 / 0.383 | 0.319 / 0.491 | 0.303 / 0.495 | 0.519 / 0.614 |
| full record | 0.286 / 0.439 | 0.354 / 0.465 | 0.425 / 0.587 | 0.428 / 0.606 | 0.582 / 0.702 |

3. **Read (3): patient-specific VOI ≈ the population table under every engine.** VOI
   (codes) at k = 5 edges the table by +0.005 / +0.009 P(true) — noise. Five questions
   close ~20% of the visible-to-full gap under codes (CT 0.238 → 0.262 of 0.354; CM
   0.300 → 0.325 of 0.428), as 0141 found under E3; Marfan/EDS does not move.
4. **Read (4): the code differential has a higher ceiling on Connective tissue** (full
   record P(true) 0.354 vs 0.286), equal on cardiomyopathy.
5. **The ceiling is the record, not the engine.** Even the full record (attestation codes
   removed) puts P(true) at 0.29–0.43 and top-1 at 0.44–0.61 on the Mondo groups, for
   every engine. Representation choices move this by ~0.02.

**Consequences.** Framing 1's engine becomes E6 (best log loss in all three sets, best or
tied balanced top-1) — a modest, noisy improvement. Framing 2's deliverable stays the
population VOI table (now confirmed under four engines). What would move both is more
signal per patient: more training patients (ever-coded members, the spec revision),
handling the multi-positive patients now set aside, and features the record lacks.
Insight 0102.

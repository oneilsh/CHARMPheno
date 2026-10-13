---
id: 141
slug: framing2-next-question-on-0137
status: running
model_class: gated_pc
# NOT A FIT. Framing 2 of spec 2026-10-13 (clinician framings): "given everything
# already in this record, which further fact — a condition, a medication, a test RESULT —
# would most change the differential?", on the framing-1 engine of record (E3, the
# multiclass head; 0140). Also adds E3's reasons (attribution through θ).
# PROTOCOL (R2.3, positive-unlabeled aware): on 0140's evaluation patients, hide a random
# 20% of each record (attestation and [measured] already removed); rank the pool's
# not-yet-visible candidates (population-EIG table: 100 condition + 100 drug + 100
# measurement-result codes) four ways — VOI = P(yes) · KL(E3 differential after a yes ||
# now); most likely present; the population table order; random — and for the top k
# (1, 3, 5): YIELD = share that are among the hidden codes (known yes; an absent
# candidate is unlabeled, never a no), and the differential after revealing them
# (P(true), top-1). References: the visible record, k random HIDDEN codes (an oracle that
# only reveals real facts), the full record.
# R2.2: candidates whose presence alone gives one member >= 90% are flagged † (likely
# treatment / diagnostic consequence), shown not hidden.
# PRE-REGISTERED READS: (1) does VOI beat the population table and random on P(true)
# after k = 3; (2) yield — VOI trades yield for informativeness, "likely present"
# maximizes yield; which reveals more of the truth per question; (3) how much of the
# top 3 is flagged † (the circularity share); (4) E3 reasons vs E1/E4 reasons (0140).
fit_of: 137
---

# 0141 — framing 2: next question, on the E3 differential (0137)

## Run

Same driver as 0140 (`--differential`) with `--diff-next-question`; reuses the bundle
cache if this cluster built it for 0140, else ~20 min rebuild. Then roughly 1.5 s per
evaluation patient for the next-question pass.

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0137-dismech-ribbon-record-tpn5-v2codemap
cat > "$RUN"/next_question.sh <<'EOF'
make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--differential '^cardiomyopathy\$\$|^connective tissue disorder\$\$' --diff-next-question --acq-tag groups_nq"
make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--differential '^Marfan syndrome|^Ehlers-Danlos syndrome' --acq-adhoc --diff-next-question --acq-tag marfan_eds_nq"
EOF
nohup bash "$RUN"/next_question.sh > "$RUN"/next_question_log.md 2>&1 &
cat "$RUN"/differential_groups_nq.md "$RUN"/differential_marfan_eds_nq.md | cut -c1-170
```

(`concept_names.csv` from 0140 is reused from the run dir.)

## Run log

**2026-10-13 — first run.** Framing-1 numbers reproduce 0140 exactly (same seed).

Next question, P(true) on the E3 differential (visible 80% → after the top 3 → full
record):

| set | visible | VOI@3 | likely present@3 | table@3 | random@3 | 3 random hidden | full record |
|---|--:|--:|--:|--:|--:|--:|--:|
| connective tissue (267) | 0.262 | 0.264 | 0.263 | 0.267 | 0.262 | 0.263 | 0.278 |
| cardiomyopathy (221) | 0.393 | 0.402 | 0.395 | 0.403 | 0.393 | 0.397 | 0.422 |
| Marfan/EDS (56) | 0.526 | 0.527 | 0.527 | 0.526 | 0.526 | 0.530 | 0.540 |

Yield@3: VOI 0.01 / 0.09 / 0.05; likely present 0.13 / 0.45 / 0.33; table 0.02 / 0.06 / 0.00.
Flagged † in the VOI top 3: 22% / 29% / 36%.

**Read.** No strategy separates, because nothing moves the differential: revealing ALL
of the hidden 20% (oracle) raises P(true) by only 0.014–0.029. One more fact is worth
almost nothing to E3 on a long record — the same arithmetic as insight 0098, now on full
records: a code's pull on θ is ~ its count / (α·K + record tokens), α·K ≈ 1,570. The
protocol cannot discriminate strategies when the ceiling itself is ~0.02. The CONTENT of
the recommendations is clinically sensible (Postpartum finding → peripartum CM; aortic
valve disorder and RETINAL DETACHMENT → Marfan; hyperprolactinemia → pituitary tumour;
methotrexate / psoriasis / hydroxyurea for the connective-tissue split), with † on
roughly a quarter to a third.
Leak found: E3's reasons for HCM lead with "Hypertrophic cardiomyopathy" and VOI
recommends it († flagged) — a disease-naming code that is not among the member's
attestation codes in the code map (to check: its concept vs code_map.tsv). DCM's
reasons likewise carry the group-level "Cardiomyopathy" / "Primary cardiomyopathy".

## Results

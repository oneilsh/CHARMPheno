---
id: 141
slug: framing2-next-question-on-0137
status: planned
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

## Results

# Framing 2 as workup: which fact would most move this patient's differential?

**Date:** 2026-10-28
**Status:** draft for review
**Supersedes:** framing 2's evaluation in the clinician-framings spec (2026-10-13, R2):
"hide 80% of the pre-index record at random, ask YES-only questions, score P(true)@k".
That protocol stays in the record (exps 0148-0151, 0166); it is not the deliverable.
**Builds on:** framing 1's engines and protocol (E6 = codes + θ, cross-fit, incident test
patients); the conversion sidecar (first-attestation dates, index horizon); the
measurement result tokens (insight 0099).

## Why

The clinician's question is: *this patient has some hepatitis; given what I already
believe (a prior, adjusted by their record), what fact would most change my belief
about which one?* That is a ranking of candidate facts by how much their answer is
expected to move the differential **in any direction**: at 95% hepatitis C, the most
valuable fact is likely one whose rare answer would knock C down. It is not "which
fact maximizes top-1", and it is not a question about the patient's past.

Exp 0166 and the framing-2 review (2026-10-28) found four defects in the current
protocol that make it unable to answer that question:

1. **Answers come from the past.** Questions are answered from a randomly hidden 80% of
   the pre-index record. The facts that settle a differential (serology, imaging, the
   findings of a workup) are recorded after the index and are not in the bundle at all.
2. **A NO is never valued.** The code head reads an unasked code and an absent one as
   the same 0, so only YES answers can move it.
3. **The pool and the scorer are different models.** The pool is ranked by population
   information gain under the topic profiles; the answers are scored by E5's weights.
4. **The score is a noisy proxy.** P(true)@k from a 5-question policy on 30-130
   patients per group, with a full-record ceiling only 0.05-0.17 above the start.

## Definitions

**Patients.** Framing 1's incident patients for a group G with members M: no member or
group diagnosis before the index, exactly one member first diagnosed in the label year
`[index, index + 365)`. Their diagnosis date `t_dx` is the first attestation of that
member's closure in the conversion sidecar.

**The workup window** `W = [index, t_dx)` (strict, default) or `[index, t_dx]`
(inclusive of the diagnosis day; serology often lands the same day as the code). Facts
after the diagnosis are its consequences (confirmatory tests, treatment) and are never
answers. W's length is anywhere from 0 to 365 days under the population index.

**Current belief** `q0` over M: E6 (codes + θ, pre-index record, attestation codes
removed) fit on the training split, as in framing 1. Any framing-1 engine can supply
it; E6 is the default because it is best or tied in most groups (exp 0163).

**Candidate facts** (any domain, D3 of the 2026-10-13 spec):

- a **condition or drug** concept v: answer in {recorded in W, not recorded in W};
- a **measurement** concept t: answer in {not done, done with no result state, normal /
  negative, low, high / positive / abnormal}. When W holds several states of t, the
  answer is the most abnormal (abnormal > normal > done-without-result). The
  "done, no result" category is the order-only `[measured]` token D1 kept out of the
  framings; it is kept here as an answer category because "was the test ordered" is
  part of a workup, with a sensitivity run that drops it.

Attestation codes of the members and the group are not candidates. Candidates are the
facts whose non-default answer occurs for at least 20 training patients of the group
(the egress floor, which also keeps likelihoods estimable).

**Likelihoods** `P(a | m)`: the rate of each answer among training patients of member
m, measured in THEIR workup windows (same window rule), add-half smoothed over the
answer categories. "Not recorded" is a recording outcome, not a clinical negative:
it is evidence exactly as far as recording rates differ between members.

**Update** (naive Bayes over the answers actually given, nothing for unasked facts):
`q(m | a) ∝ q0(m) · P(a | m)^τ`, with the temper τ fit on the cross-fit fitting half
(as E1 / E4 are) to correct for the independence assumption.

**Expected information gain** of fact v under the patient's own belief:

    EIG(v) = H(q0) − sum_a P(a) H(q(· | a)),   P(a) = sum_m q0(m) P(a | m)

equal to the expected KL from q0 to the updated belief: how far the answer is expected
to move the differential, in any direction.

## Evaluation

Every candidate's true answer is known for every patient, so every (patient, fact) pair
is scored. Nothing depends on a sequential policy.

- **Realized gain** of fact v for a patient: `g(v) = log q(true | a_true) − log q0(true)`,
  using the fact's actual answer. A proper score: negative when the answer moved the
  belief the wrong way; its expectation under a calibrated model is EIG(v).
- **R1 — ranking accuracy.** Per patient, the Spearman correlation between EIG and g
  over the candidates; median and quartiles per group, pooled over groups.
- **R2 — value of the top fact.** Mean g of the top-1 / top-3 facts by EIG against
  (a) a random candidate, (b) the population ranking (EIG under the training prior
  instead of q0: one list per group), (c) the best fact in hindsight (the oracle
  ceiling, max g).
- **R3 — calibration of EIG.** (Patient, fact) pairs binned by EIG; mean g per bin.
  Slope near 1 means the values are trustworthy, not just the order; below 1 is the
  naive-Bayes overconfidence τ did not remove.
- **R4 — decisions.** How often the top fact's actual answer changes the top-1 call,
  and how often that change is toward the truth.
- **Coverage, reported first** (the short-window worry): per group, the distribution of
  W's length, the share of test patients with at least one non-default answer among
  the candidates, and the median number of non-default answers. A group with too
  little workup to rank is reported as such, not scored.

All per-group results need >= 20 scored patients; candidate-level lists (the top facts
of a group) show only facts with >= 20 patients in every reported cell.

## The workup sidecar

The bundle cannot answer these questions: the assembler keeps the post-index window
for conditions only, and only as label nodes; post-index drugs and measurements are
discarded at windowing (`multi_domain.lookback_feature_frames`). So the facts are
captured the way the conversion sidecar captures first-attestation dates: where the
full-history frames still exist, in driver-owned code, as a separate artifact.

- **Module:** `analysis/cloud/workup_sidecar.py`, `make build-workup-sidecar ID=N`.
  Driver-owned: it imports the loaders (`load_omop_bigquery`) and the measurement
  tokenizer, edits no hashed module, and moves no bundle, corpus or covariate key.
- **Contents:** one row per (person, token) with the token's FIRST date in
  `[index, index + 365)`, for every person in the run's sample (train and test: the
  likelihoods come from training patients' windows). Tokens are the corpus's own
  (condition_era concepts, drug_era concepts, measurement result tokens), restricted
  to the frozen vocabularies, so answers live in the space the engines read. The
  per-group window `[index, t_dx)` is applied at analysis time, by joining the
  member's first-attestation date from the conversion sidecar.
- **Index:** the conversion sidecar's index horizon (the corpus's own index,
  re-derived), so both sidecars agree on every person's index by construction.
- **Key:** `(cdr, person_mod, index identity (the conversion sidecar key), vocab
  identity, label_window_days, this module's source)`. Its own key, like the
  conversion sidecar's: a bundle-key move does not orphan it.
- **Size:** one year per person, distinct tokens with a first date, which is much
  smaller than the events. The build logs the row and person counts and the coverage
  numbers above.
- **Location:** HDFS by default like the others (rebuilt per cluster; the conversion
  sidecar must exist on the cluster first), or `GPR_SIDECAR_URI` for a persistent
  in-boundary bucket.

## Code

- `differential.py` (driver-owned): `workup_answers` (the categorical answer of every
  candidate for one patient's window), `answer_likelihoods` (per-member smoothed
  rates), `eig_categorical`, `nb_update`, `evaluate_workup` (R1-R4 and coverage).
  Pure numpy; unit-tested against a planted toy where one test separates the members
  and EIG must rank it first and be calibrated.
- `gated_pc_readout.py --diff-workup [--diff-workup-window strict|inclusive]
  [--diff-workup-no-measured]`: framing 2 on the workup sidecar in place of the hidden-
  record protocol, inside the existing survey (`--diff-survey`), so the same groups and
  patients as framing 1 are scored.
- `survey_analysis.py`: the coverage and R1-R4 columns, single-run and paired.

## Open questions for review

1. **Window end.** Strict `[index, t_dx)` by default with an inclusive sensitivity
   run, or the reverse?
2. **"Done, no result" as an answer category.** Kept by default (a test being ordered
   is workup information), with a run that drops it?
3. **Which q0.** E6 by default; also run E3 (θ only) so framing 2 says something about
   the model's representation rather than the raw codes?
4. **Sensitivity of the pre-index record** (the maintainer's note): q0 computed from a
   shortened lookback (the last 1 or 2 years) or a random share of pre-index codes, to
   see how the ranking changes as the starting belief weakens. Proposed as a later
   experiment, once R1-R4 exist.

# Case-finding FDR readout — design (sub-project 1 of 2)

**Status:** approved (brainstorm 2026-07-20). Exploratory research build; no
production target.

## Scope and decomposition

This is **sub-project 1** of two:

- **SP1 (this spec): in-enclave FDR readout.** A background-relative,
  multiple-testing-corrected scoring of the fitted gated-LDA placement
  profiles, computed inside `evaluate`. Ships new metrics on exps 0061/0062
  with near-zero egress (a handful of summary numbers). Includes a
  zero-inflated-Beta-vs-empirical diagnostic whose result chooses SP2's export
  form.
- **SP2 (later, separate spec): exportable per-node null + portable scorer.**
  Freezes SP1's per-node background null into a small, privacy-safe artifact so
  a new cohort/site can score locally. Its export form (parametric ~3KB vs
  tail-dense empirical grid) is decided by SP1's diagnostic — we measure before
  we size. Out of scope here.

## Motivation

The current detection readout scores a patient by `case_score = P.max(axis=1)`
(strongest single node-block mass) and `disease_mass = P.sum(axis=1)` (total
node-block mass), both raw shares of the theta simplex (dag_placement.py
`evaluate`). Two problems:

1. **Multimorbidity is penalized by the simplex.** theta sums to 1, so a
   patient whose record is dominated by an unrelated condition (its mass parked
   in the background blocks) has less simplex left for a rare node; a patient
   with two rare diseases splits mass between the two node blocks, diluting
   both. Raw-mass scoring makes exactly the richest-truth patients hardest to
   flag.
2. **The score is not background-relative and the threshold is arbitrary.** The
   @sensitivity operating points are hand-set; there is no calibrated notion of
   "how surprising is this mass relative to a non-case."

Background-relative, per-node testing fixes both: each (patient, node) is its
own hypothesis, so a patient can be a discovery on several nodes independently
(no simplex tug-of-war), and a p-value + FDR turn the operating point into the
clinically meaningful knob — "of the patient-disease leads surfaced, at most q
are false."

## The statistic, null, and test

For each test patient i and node u, the statistic is patient i's **node-u block
mass** `P[i][u]` (already produced by `profile`), **conditioned on record
length**. Record length (token count per document) is stratified into bins; a
patient's null is drawn from their own length bin, because longer records
concentrate theta differently (more evidence overrides the alpha prior) and
would otherwise confound the lift. **Binning default:** quantile bins of the
background length distribution, a small configurable count (default 4); a
patient is assigned to the bin its length falls in; a count of 1 recovers the
unconditioned null (the fallback when length is shown not to confound).

- **Null (H0):** patient i is background w.r.t. u — their node-u mass is drawn
  from the background distribution of node-u mass in their length bin.
- **Empirical null.** The background patients (empty frontier) are present in
  the same test set (n_bg ~ 29k-43k for rare6), so per (node, length-bin) their
  node-u masses form the empirical null directly inside `evaluate`. The p-value
  is the right-tail fraction of background masses >= observed (with the standard
  +1 / (n+1) plug so p is never exactly 0; self-inclusion of a scored background
  patient in its own reference is negligible at this n, leave-one-out optional).
  This calibrates
  externally to the model: the model output is used only as a statistic, and
  significance comes from real background patients — the project-consistent
  choice given the known variational overconfidence (insights 0051/0057).
- **Two-groups framing** (Efron 2004, 2008, empirical Bayes / local FDR): the
  collection of per-node statistics is a null-mixture f0 + alternative f1 with
  null proportion pi0; pi0 is estimated from the bulk (center) of the
  distribution, which is robust to the tail contamination described below.
- **Correction.** Per-node **Benjamini-Hochberg** (Benjamini & Hochberg 1995,
  JRSS-B) across patients: "for rare disease u, the candidate patient list at
  FDR <= q." This is the interpretable case-finding deliverable and stays well
  within the empirical null's tail resolution. **Benjamini-Yekutieli**
  (Benjamini & Yekutieli 2001, Annals of Statistics) is available as the
  dependence-conservative fallback — within a patient the node masses are
  compositionally negatively correlated, violating BH's PRDS assumption, though
  BH is robust in practice.

The full (patient x node) grid (global FDR over all pairs) is NOT the default:
its large m demands deeper-than-floor p-values that the empirical null cannot
resolve. Per-node BH is the default; the full grid is deferred to SP2/EVT if
ever needed.

## Architecture — where it lives

All in the engine, `spark-vi/spark_vi/models/topic/dag_placement.py`, which is
domain-neutral (node-block mass, no concept-ids) — consistent with the
engine/domain layer split. Structure:

- **Pure-numpy helpers** (unit-testable in isolation): an empirical right-tail
  p-value against a reference sample; BH and BY q-value / rejection given
  p-values; a length-binning helper; a per-node discovery routine that ties the
  above together over `P`, `is_fg`, and `doc_lengths`.
- **A new `fdr` block in `evaluate`.** `evaluate` already has `P`, derives
  `is_fg`, and has the background patients in-set; it gains one parameter,
  `doc_lengths` (per-doc token count, aligned to `profiles`), and returns the
  new block alongside the existing `detection`. When `doc_lengths is None`,
  length-conditioning collapses to a single bin (all patients) so existing
  callers/tests keep working unchanged.
- **Driver threading.** `analysis/cloud/dag_placement_cloud.py` selects the doc
  `features` (BOW) alongside `nodeAffinity`/`frontier` in the inline eval,
  computes per-doc token count (sum of BOW counts), and passes `doc_lengths`
  into `evaluate`. The new metrics are printed and folded into the manifest.

**Estimation is in-sample** for SP1 (the null is estimated from the same test
set's background). That is correct and standard for a two-groups *readout*; SP2
freezes the null for out-of-sample scoring.

## Outputs (the `fdr` block)

Reported next to `detection`, all aggregates:

- Per q in a small grid (0.05, 0.10, 0.20): total discoveries, precision among
  discoveries (fraction whose (patient, node) is truly in the patient's
  frontier subtree), and recall of the true scoreable frontier.
- **Multimorbidity payoff:** mean discoveries per truly-multimorbid patient
  (truth frontier size >= 2), vs the raw-mass argmax baseline — the direct
  measure of the simplex fix.
- **Saturation flag:** whether discoveries pile at the empirical p-floor
  (1/n_bg), i.e. whether the tail needed finer resolution than the empirical
  null provides (the signal that SP2 would need EVT/GPD rather than a quantile
  grid).
- **ZIB diagnostic:** per (node, length-bin), the max CDF gap between a fitted
  zero-inflated Beta (point mass at ~0 + Beta on the positive part) and the
  empirical CDF, summarized (e.g. mean and worst-case gap). Reported only — the
  p-values never depend on it. This is the number that decides SP2's export
  form: small gaps -> parametric null is faithful (~3KB export); large gaps ->
  the null is non-Beta and SP2 must ship the tail-dense empirical grid.

## Caveats (surfaced in the reported output, not only in prose)

- **Contaminated null.** The background arm is "no attested rare6 node in the
  label window" — exactly where undiagnosed true cases hide. The empirical null
  is therefore inflated toward the alternative, making p-values **conservative**
  (under-calling). Efron's bulk-based pi0/f0 estimation mitigates (contamination
  lives in the tail; the bulk is genuine background), but the effect is real and
  inherent to case-finding. Flagged in the block.
- **In-sample null.** SP1 estimates the null on the same test background it
  scores. Appropriate for a readout; SP2 addresses out-of-sample.

## Testing

Engine unit tests (`spark-vi/tests`, pure numpy, no Spark):

- **Calibration:** null-only synthetic `P` (no planted signal) -> realized FDR
  <= q within Monte-Carlo tolerance across the q grid.
- **Power / recovery:** planted-signal synthetic `P` (a known subset elevated on
  known nodes) -> those (patient, node) pairs recovered as discoveries;
  multimorbid planted patients recovered on multiple nodes.
- **Length conditioning:** synthetic where length confounds node mass ->
  unconditioned scoring miscalibrates, length-conditioned scoring restores FDR
  control.
- **BH vs BY:** BY rejects a subset of BH at the same q (ordering/monotonicity).
- **Backward compat:** `evaluate` with `doc_lengths=None` returns the prior
  metrics unchanged plus a single-bin `fdr` block.
- Driver: an arg-surface / smoke test that the inline eval selects `features`
  and threads `doc_lengths` (the BQ transform body stays cluster-covered).

The real read is the cluster rerun of 0061/0062's `evaluate` on the *identical*
fitted profiles (no model re-fit): does the FDR readout surface multimorbid
cases the raw-mass argmax misses, and where does the FDR operating point land?

## Out of scope

- **SP2:** the exportable per-node null artifact and the portable/federated
  scorer (separate spec; form chosen by SP1's ZIB diagnostic).
- The **per-patient code->topic contribution viewer** (in-enclave, reuses the
  dashboard `codeComposition` formula) — a related but separate diagnostic tool,
  not part of this readout.
- Any change to the model, gate, anchor_scope, or windowing (all orthogonal;
  this is post-hoc on the fitted profiles).
- Full (patient x node) global-grid FDR and EVT/GPD deep-tail nulls (revisited
  in SP2 only if SP1's saturation flag fires).

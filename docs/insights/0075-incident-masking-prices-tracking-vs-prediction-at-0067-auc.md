# 0075 — Incident masking prices the tracking-vs-prediction gap at ~0.067 AUC on shared nodes; AP halves for prevalence reasons, not skill reasons

**Date:** 2026-09-02
**Topic:** evaluation, incident cohorts, case-finding, conditional prediction

**Status:** Confirmed on exp 0110's record run (native Mondo, C=2,714), the
incident-episode eval program's E2 payoff. Shared-node-set discipline (R2.2):
both numbers computed over the same 1,599 nodes, same scan, different cells.

**Setting.** The same fitted model, the same test documents, two evaluation
cohorts per node c: PREVALENT (every eligible document) vs INCIDENT (documents
with `c ∉ R_d` — no pre-index attestation of c's closure, spec D2). The claim
naming discipline (D7) matters: this is *a prevalent-fit model evaluated on an
incident cohort*, not an incident-fit model. Nothing about training changed.

**The numbers (shared 1,599 nodes, macro):**

| cohort | AUC | AP |
|---|---|---|
| prevalent | 0.7412 | 0.4361 |
| incident  | 0.6741 | 0.2429 |

**Finding 1 — ~0.067 AUC of the apparent skill is tracking, not prediction.**
Prevalent evaluation lets the model "predict" diagnoses already in the chart —
the θ features contain the disease's own codes, so part of the AUC is reading
the record back. Removing prior carriers removes exactly that channel, and the
residual 0.6741 is the honest forward-prediction signal: well above chance, and
the defensible headline for any conditional-prediction claim. Every prior
experiment's prevalent-only numbers (0104's 0.6978/0.4845 included) carry this
inflation and should be read with it in mind.

**Finding 2 — the AP halving is prevalence-mechanical, not a second skill
loss.** AP is prevalence-sensitive where AUC is not; incident cohorts are far
lower-prevalence once prior carriers (the bulk of positives) leave. The AUC
gap measures lost signal; the AP gap mostly measures the changed base rate.
Reading both drops as "the model got worse twice" double-counts.

**Finding 3 — the skipped-column census is a finding, not bookkeeping.**
Incident evaluation dropped 923 nodes as <20 incident positives (vs 192
degenerate, 0 constant — the census-predicted-empty C2.1 population confirmed,
R2.1's guard firing zero times). A population-random index catches only each
person's one sampled pre-onset year, starving a third of the DAG. This number
is what redirected exp 0111 to episode-anchored sampling (insight 0078).

**Practice.** Report both cohorts, always named per D7; treat prevalent-only
metrics as upper bounds contaminated by tracking; expect AP to move with any
cohort change that moves prevalence, and say so rather than narrating it as
skill.

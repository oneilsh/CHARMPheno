# Session Handoff: Hybrid domain reliability and scalable supervised placement

**Date:** 2026-07-30
**Project:** `/Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno`
**Branch:** `hybrid-domain-reliability`
**Base:** `multidomain-spectral-init` at `5769f74`
**Phase:** empirical interpretation and next-architecture exploration
**Intended reader:** Claude or another agent resuming the rare-disease
case-finding arc

---

## Current state

The hybrid domain-reliability readout is implemented, reviewed, pushed, and run
successfully on two freshly attested rare6 artifacts:

- exp 0073: mini-batch replication of exp 0072;
- exp 0074: corrected full-batch replication of exp 0071.

Both fits and both weighting readouts exited successfully. The immediate
implementation arc is complete. The next task is **not** another rare6
hyperparameter tweak: it is deciding how to generalize supervised domain
placement across hundreds of disease anchors and future domains without
building hundreds of unrelated classifiers.

The user is transferring the discussion back to Claude because of usage limits.
Do not begin with a code walkthrough; the user explicitly said they are not
ready for one.

## Goal and scope

The active scientific goal has moved beyond the earliest architecture docs:

- Find uncoded or undiagnosed rare-disease patients using an
  ontology-guided, gated, multidomain topic model.
- Maximize precision/recall on the current rare6 development benchmark using
  condition, drug, and observation before adding measurement.
- Later scale to hundreds or thousands of rare-disease anchors.
- Eventually replace the current SNOMED stand-in with Mondo↔OMOP mappings and
  use the Mondo DAG.
- Return to temporal modeling later; the early OU/topic-state architecture is
  still relevant but not the present arc.

The user strongly prefers general methods over rare6-specific policies or
feature engineering. Future content domains include procedure and measurement.
Visits, providers, and observation periods are currently viewed as linking,
exposure, or conditioning modalities rather than competing content domains.

## Work completed in this session

### 1. Hybrid readout designed and implemented

The approved design and plan are:

- `docs/superpowers/specs/2026-07-29-hybrid-domain-reliability-readout-design.md`
- `docs/superpowers/plans/2026-07-29-hybrid-domain-reliability-readout.md`

Implementation commits on this branch:

- `64d6afd` — fixed domain-score combination primitives;
- `a356716` — background edge-case hardening;
- `3123b79` / `4e8f553` — label-free λ reliability and boundary validation;
- `35d259a` / `2026923` — nested domain-weight evaluation and semantic tests;
- `71987ad` — comparison with model-derived weights;
- `fb13d94` / `fcd66ef` — CLI plus artifact/identity contract hardening;
- `404c3a7` / `7c8b4b7` — attested rerun registration and handoff corrections;
- `36fad21` — full-batch attested replication registration.

Key implementation surfaces:

- `spark-vi/spark_vi/models/topic/dag_placement.py`
  - fixed LR background;
  - domain score scales and weighted combination;
  - λ-derived distinctiveness, ownership, and viability.
- `analysis/cloud/multidomain_weighting.py`
  - repeated nested stratified CV;
  - fold-local backgrounds and score scales;
  - discrete policy selection;
  - nonnegative continuous simplex search;
  - AP and precision-at-recall reporting.
- `analysis/cloud/multidomain_weighting_readout.py`
  - artifact loading and report CLI.
- `analysis/cloud/Makefile`
  - `multidomain-weighting-readout`.
- `analysis/cloud/multidomain_cloud.py`
  - privacy-safe one-row-per-person attestation in persisted test artifacts.
- `docs/decisions/0038-supervised-multidomain-readout-identity-attestation.md`
  - row identity and semantic domain-name contract.

The readout refuses legacy artifacts that cannot prove one row per person.
Person IDs are not persisted; only row count, unique-person count, and the true
attestation are saved. Domain names, not ordinal positions, define
`fixed:condition_drug`.

### 2. Cluster execution

The cluster workflow was deliberately made pull-and-run friendly and resilient
to a closed laptop:

```bash
cd /home/dataproc/repos/CHARMPheno
git pull

nohup bash -c '
  set -x
  make -C analysis/cloud exp ID=73 \
    && make -C analysis/cloud multidomain-weighting-readout ID=73
  make -C analysis/cloud exp ID=74 \
    && make -C analysis/cloud multidomain-weighting-readout ID=74
' > ~/hybrid-73-74.log 2>&1 &
```

Both runs reported `JOB_EXIT_CODE=0`.

Exp 0074 full batch converged in 13 iterations with `rho=1.0000`; the fit phase
took about 540 seconds. It reported 3/180 fully starved topics, all duplicate
topics for “Lung disease with systemic lupus erythematosus.” This is reported
honestly by `starved_topic_report`; the node-level init report remained empty
because other topics in that node were alive.

### 3. Empirical results

Macro median average precision:

| Artifact | fixed condition+drug | discrete selector | continuous weights | distinctiveness | ownership | product |
|---|---:|---:|---:|---:|---:|---:|
| 0073 mini-batch | 0.082 | 0.054 | 0.087 | 0.049 | 0.058 | 0.058 |
| 0074 full-batch | 0.072 | 0.055 | 0.079 | 0.043 | 0.049 | 0.046 |

Interpretation:

- Continuous nonnegative disease-specific weights offer real but modest
  aggregate headroom over fixed condition+drug: approximately 6% relative in
  0073 and 10% in 0074.
- The discrete selector is unstable and substantially worse than the fixed
  baseline. Do not interpret it as evidence against supervision generally; it
  is a high-variance choice among discontinuous policies with few positives.
- Simple λ-derived reliability is not a task-utility oracle. Distinctiveness,
  ownership, and product all trail the fixed baseline. Ownership is the
  strongest of them and is useful as a prior/input, not as the final answer.
- Observation receives zero median continuous weight for every rare6 disease
  in both artifacts.
- Five diseases choose condition-only median continuous weights.
- Myasthenia gravis is the exception:
  - 0073: condition 0.55, drug 0.45;
  - 0074: condition 0.45, drug 0.55.
- These findings must **not** become a universal “condition first” policy.
  rare6 is a development benchmark and mechanism stress test. Procedures,
  measurements, other diseases, and larger anchor sets may behave differently.

Per-disease fixed/continuous AP:

| Disease | 0073 fixed | 0073 continuous | 0074 fixed | 0074 continuous |
|---|---:|---:|---:|---:|
| Ehlers-Danlos | 0.126 | 0.146 | 0.106 | 0.129 |
| Sarcoidosis | 0.073 | 0.069 | 0.063 | 0.063 |
| Systemic lupus | 0.135 | 0.139 | 0.137 | 0.138 |
| Scleroderma | 0.080 | 0.098 | 0.075 | 0.081 |
| Myasthenia gravis | 0.043 | 0.040 | 0.016 | 0.027 |
| Amyloidosis | 0.032 | 0.032 | 0.035 | 0.036 |

This result is recorded as:

- `docs/insights/0075-hybrid-domain-weighting-headroom-is-small-and-not-identified-by-lambda-reliability.md`

### 4. Data-assembly caching finding

The user noticed that data assembly remained annoyingly slow. The old caching
system exists, but the N-domain driver is **not using it**:

- `analysis/cloud/_case_finding_cache.py` supports cached single-domain
  `CaseFindingBundle` assembly.
- `analysis/cloud/multidomain_cloud.py` directly loads every BigQuery domain,
  windows it, and calls `assemble_multidomain_from_events`.
- The multidomain driver has no `--cache-uri` seam and the Makefile does not
  pass one.
- Spark `.cache()` calls inside assembly only avoid recomputation within the
  current application; they do not persist a reusable build across fits.

Therefore 0073/0074's roughly 196-second window+assemble phase was a fresh
assembly. Wiring a content-addressed N-domain bundle cache is a valid future
operations improvement, especially before larger anchor sweeps, but it was not
implemented in this session.

## Design discussion and decisions

### Do not overgeneralize rare6

The initial interpretation leaned too hard toward condition-only weights and a
label-free fallback. The user corrected this:

- future models may contain hundreds of rare-disease anchors;
- future domains may have very different reliability;
- observation is an OMOP dumping ground and may remain harmful, but this is not
  established for every disease or future representation;
- procedures and measurements remain important;
- models themselves may become more sophisticated.

The design target is a general strategy across diseases and domains, not a
fixed rare6 recipe.

### Completely unseen diseases are not required

“Completely unseen disease” was clarified to mean an anchor with no usable
positive examples during fitting. The user considers that practically
irrelevant. Assume every modeled anchor has some known coded patients.

Operational setting:

- known coded patients establish the gate/topics and may contribute
  supervision;
- defining marker codes remain stripped from features;
- held-out/unlabeled patients are ranked to find additional uncoded cases.

Generalization means sharing statistical strength across diseases, not zero-shot
disease learning. A disease-held-out analysis may still be useful as a stress
test of a shared weighting mechanism, but it is not a deployment requirement.
The hybrid design spec has been amended accordingly.

### Preferred next architecture

The user reacted positively to a shared hierarchical placement mechanism:

- global domain or domain-family tendencies;
- partially pooled disease-specific deviations;
- ontology-conditioned pooling so related diseases share strength;
- one joint mechanism across hundreds of anchors, not one unrelated classifier
  per disease;
- λ-derived ownership/distinctiveness as priors or inputs rather than truth;
- patient-level held-out evaluation for every disease;
- optional disease/subtree-heldout evaluation as a generalization diagnostic.

Because non-cases are mostly unlabeled rather than confirmed negatives, a
positive-unlabeled or ranking objective may be more faithful than ordinary
binary cross-entropy. This is a design hypothesis, not yet approved in detail.

The current recommendation is to prototype a shared hierarchical readout first.
It can test partial pooling and generalization without changing variational
inference. If it shows durable value across many anchors/domains, consider
moving the predictive signal inside the topic model.

### Supervision inside versus after the topic model

The user dislikes secondary held-out supervised fitting but is willing to use
it if necessary. The relevant literature shows a real spectrum:

- MixEHR: unsupervised multimodal representation plus downstream disease and
  mortality classifiers.
- sLDA: response likelihood jointly shapes topics.
- MedLDA: max-margin prediction jointly shapes topics.
- Semi-supervised prediction-constrained topic models: balance generative fit
  and prediction, including EHR experiments.
- Prediction-focused topic models: suppress task-irrelevant features.
- MixEHR-S: supervised specialist/disease topic inference.
- MixEHR-SurG: survival supervision inside the MixEHR lineage.
- MixEHR-SAGE: guided diagnoses/procedures/medications and more than 1,000
  PheCode topics.

These are now represented in `docs/references.md`. Do not portray the literature
as proving that integrated supervision is automatically better here; it
provides design precedents and tradeoffs.

### Domain adapters versus weights

Keep separate:

- **domain representation/likelihood** — how a domain emits evidence;
- **domain reliability/placement** — how its evidence affects case ranking.

Condition, drug, and procedure can initially use event/count views.
Measurement probably needs a value-aware model involving result, unit,
abnormality/reference range, and time. Treating “test measured” as an ordinary
token discards much of the clinical signal and risks severe repetition
overcounting.

## Correlated evidence: important unresolved objection

The user's colleague raised LIRICAL's conditional-independence problem:
correlated phenotype observations can be multiplied as if they were independent
and overstate evidence.

Our model has a related limitation:

```text
z_token ~ θ_patient
code_token ~ β_domain,topic
```

Given θ/topic assignments, token emissions are conditionally independent and
exchangeable. Shared θ induces marginal co-occurrence and lets a topic absorb a
common bundle, so this is less naïve than multiplying independent feature LRs.
It does **not** fix confidence inflation: correlated conditions, a condition
plus its treatment, or several codes from one encounter still add separate mass
to γ and separate LR evidence.

Existing mechanisms do not solve this:

- presence/count transforms suppress repeats of the same concept;
- background topics absorb common content;
- ω controls whole-domain volume;
- the ontology gate controls allowed disease topics;
- none models dependence among distinct observations.

Three initial suggestions were discussed and the user rejected all as primary
directions:

1. broad compression/feature engineering — some repetition suppression is
   acceptable, but not extensive hand-engineering;
2. learned “evidence groups” — these sound like topics again and risk merely
   duplicating the representation;
3. explicit care-process/manifestation latent variables — fancier models are
   likely eventually, but reliably learning care-process variables is doubtful.

Do not resume by re-proposing those three unchanged. The open question is:
**Can the probabilistic model represent burstiness/redundancy or calibrate an
effective evidence count in a general, learned way without hand-built groups or
speculative care-process latents?** This deserves a focused literature/modeling
brainstorm before measurements are added.

## User preferences and constraints

- General, reusable, literature-grounded solutions over disease-specific
  patches.
- Rare6 iteration is acceptable; external validation on other diseases can
  happen when hyperparameter overfitting becomes a serious concern.
- New fits are reasonably fast, so do not avoid one when it genuinely answers
  a modeling question.
- Cluster workflow is: agent edits and pushes; user pulls and runs; user pastes
  privacy-safe results back.
- Cluster commands should be Make-based, guarded against stale code, and safe
  under `nohup`.
- Judge rare-disease case finding primarily with PR/AP and precision at recall,
  not ROC alone.
- Do not commit patient-level data or identifiers.
- The user is tired at the end of this session; resume with a concise
  synthesis, not a large menu of speculative mechanisms.

## Open questions

- [ ] What exact shared hierarchical domain-placement model should follow the
      diagnostic continuous-weight ceiling?
- [ ] Should its objective be PU classification, pairwise/listwise ranking, or
      another case-finding loss?
- [ ] How should ontology pooling work over a future Mondo DAG, especially
      multi-parent nodes?
- [ ] Should the shared readout be proven across a larger anchor set before any
      supervision moves inside VI?
- [ ] How can correlated evidence be handled without heavy feature engineering,
      redundant evidence-group topics, or weakly identified care-process
      variables?
- [ ] When measurement is added, what value-aware likelihood and repetition
      semantics should it use?
- [ ] Should the multidomain assembly path gain a persistent content-addressed
      cache before at-scale anchor sweeps?

## Recommended next steps

1. [ ] Read this handoff and the 2026-07-29 parent handoff before proposing
       implementation.
2. [ ] Briefly restate the empirical constraint from insight 0075: small
       continuous headroom, weak λ-only fallback, no universal domain policy.
3. [ ] Do a focused design/literature pass on two connected questions:
       hierarchical multitask supervised topic placement at anchor scale, and
       dependence-aware/bursty evidence models.
4. [ ] Present two or three general architectures with explicit inference,
       scaling, interpretability, and leakage/PU tradeoffs.
5. [ ] Get user approval on the architecture before writing another
       implementation plan.
6. [ ] Treat measurement and multidomain caching as subsequent arcs unless the
       chosen design makes one a prerequisite.

## Files to review on resume

- `.claude/handoffs/2026-07-29-multidomain-case-finding-arc-handoff.md` — parent
  arc, engine findings, and operational history.
- `docs/insights/0071*` through `docs/insights/0075*` — empirical sequence.
- `docs/superpowers/specs/2026-07-29-hybrid-domain-reliability-readout-design.md`
  — implemented experiment and 2026-07-30 scope amendment.
- `docs/decisions/0038-supervised-multidomain-readout-identity-attestation.md`
  — honest nested-CV artifact contract.
- `analysis/cloud/multidomain_weighting.py` — nested readout mechanics.
- `analysis/cloud/multidomain_weighting_readout.py` — artifact/report boundary.
- `spark-vi/spark_vi/models/topic/dag_placement.py` — LR and λ reliability.
- `analysis/cloud/multidomain_cloud.py` — current uncached N-domain assembly and
  fitted model driver.
- `docs/references.md` — MixEHR and predictive topic-model lineage.
- `docs/architecture/TOPIC_STATE_MODELING.md` — early temporal and richer-model
  possibilities; useful vision, not current operational truth.

# ADR 0050 — One DAG-gated LDA estimator; the supervised head and the DAG spectral init are removed

**Date:** 2026-10-27
**Status:** Proposed (maintainer review)

## Context

The prediction-constrained (PC) arc wrapped `GatedOnlineLDA` in `OnlinePCLDA` (ADR 0042)
and grew a co-fit supervised head: λ corrections, trust caps, Newton/L-BFGS heads,
localized head support and a distributed co-fit seam (ADRs 0039–0045). It never helped
case-finding: the shaped topics decoded worse than a post-hoc readout on the same θ
(insight 0087, exp 0120), and every run since ~0121 fits at `weight_y = 0`, where the
wrapper is a pass-through. Readout heads are fit after the fit (ADR 0046). Around the
head sit options no run uses any more (audit report 2026-10-26 §2). Two MLlib
estimators, `GatedLDAEstimator` and `OnlinePCLDAEstimator`, now duplicate the init
Params, the spectral routing, the per-domain λ conversion and the domain-bounds checks.

## Decision

1. **Remove the supervised head**: `OnlinePCLDA` and `models/topic/pc.py`, the
   `OnlinePCLDAEstimator/Model` shim, `PCDocument` / `GatedPCDocument`, the
   `weightY`, `head*`, `topicTrust`, `gradCaviIters`, `lambdaW`, `labelCol` /
   `labelMaskCol` / `numLabels` / `probabilityCol` and `localizeHead` / `headSupport`
   Params, the localized-support helpers in `dag_placement.py` (`cost_report` keeps
   only its K / fan-out part), the co-fit stats provider in `distributed_readout.py`, the
   driver flags, and the `analysis/pc` in-memory reference (ADR 0038) with its tests.
2. **Remove the fresh-counter warm start** (`VIRunner.fit(warm_start_from=)`,
   `warmStartFrom`): its only use was the supervised two-phase protocol. **`resume_from`
   stays**: continuing a fit is more iterations on the same Robbins–Monro counter, for
   every model.
3. **One estimator, named for its gate**: the engine becomes `DagGatedOnlineLDA`
   (`models/topic/dag_gated_lda.py`) and the MLlib pair `DagGatedLDAEstimator` /
   `DagGatedLDAModel`; the name says the gate is a DAG closure, which distinguishes it
   from a flat labelled-LDA gate (a depth-1 DAG is that special case). It carries the
   union of the live Params: the gate (`frontierCol`, `gateParent`, `gateNBg`,
   `gateTpn`), the schedule and seeds (`gateWarm`, `gateOpenSeed`, `gateSeedMode`
   default **balanced**, `gateSeedOwnCodes`, `gateSeedLeaves`, `gateSeedProfile`),
   `docConcentration`, `optimizeDocConcentration`, and multi-domain `featuresCols` /
   `domainBounds`. Initialization is random, with the data-driven gate seeds.
4. **Remove the DAG spectral init** (`gated_init.py`'s block-aligned, multi-domain and
   scalable anchor seeds, ~1,000 lines; the `init` / `spectral*` / `anchorScope` /
   `spectralTopoOrder` / `spectralMarginalFloor` / `spectralBgAnchors` /
   `spectralAnchorCandidates` Params; the anchor dump, `spectral_anchors.json`, the
   digest's ⚓ marks and `name_spectral_anchors.py`). Last used in exp 0161; the leaf and
   balanced seeds replaced it at equal or better quality (exps 0162, 0163), and its
   DAG-specific variants (anchor scope, topological order, marginal floor, background
   anchors) are complexity with no remaining use. The flat anchor-word init
   (`spectral_init.py`, `spectral_init_scalable.py`) stays: it is the STM's validated
   default (insight 0030) and predates this work.
5. **Remove other unused options**: the profile-η prior (`etaBoost`, `profileEta*`; the
   profile *seed* stays), `countTransform`, `alphaInit=equalized`, the ω modality
   weight and per-domain η, and the CAVI θ twins that only the head used.
6. **Remove the `dag_placement` line** (exps 0052–0069, the first gated model, scored
   by node placement before `gated_pc` replaced it with readout heads): the
   `dag_placement` model class and `dag_placement_cloud.py`, `lr_readout.py`, the
   placement scoring and evaluation in `models/topic/dag_placement.py` (likelihood-ratio
   and explain-away scores, FDR discovery report, `evaluate`), the estimator's
   `nodeAffinity` output, and their tests and defaults. `DagLayout` stays (it moves to
   `spark_vi/dag/` with ADR 0051); the collapsed-Gibbs `fit_gated` / `profile` stay
   only as a test oracle for the engine. Hashed charmpheno modules import `DagLayout`,
   so its move is a separate, announced cache-drop commit.
7. **Housekeeping in the same pass**: one `_destroy_broadcast` in `spark_vi.core`
   (six copies today), one dict-λ → `expElogbeta` helper (three today), `DagLayout`
   built once per fit, and domain-neutral wording: the profile seed and own-code seed
   take the domain index as a parameter instead of assuming domain 0 is "conditions".

`batched_lr.py` stays; where it lives is ADR 0051's business (API and two engines).

## Alternatives considered

- **Keep `OnlinePCLDA` as a thin alias** at `weight_y = 0`: keeps two names for one
  model and every head Param inert on the API. Rejected.
- **Keep the head as an option**: three ADRs of engineering made it scale, and it is
  still a negative result; it stays recoverable from git (commit `fff7bf4067`).
- **Keep the DAG spectral init as an option** (the 2026-10-23 decision): reversed on
  2026-10-27 once the leaf seed made it redundant; it stays recoverable from git.
- **Keep ω as an option**: a tempering weight tuned for a downstream task, never set by
  any run; observed volume cannot identify it. A learned per-topic domain mix is a
  different model question (noted below), not ω.

## Consequences

Supersedes ADRs 0038–0045 (the PC reference, head optimizer, head regularization, the
injectable-engine composition and the supervised λ correction), and the DAG uses of the
spectral init in ADRs 0032 and 0037 (both stand for the flat STM path). Amends ADR 0049: the
balanced open-time seed becomes the estimator default (`gateWarm` stays 0 by default;
the central configuration sets it). `gated_pc_readout` reconstructs the model as a
`DagGatedLDAModel` from the run's npz, so every existing run dir stays readable; models
persisted with the old MLlib PC writer do not load (none are used). The older
`dag_placement` model class (exps up to 0069) builds on the same engine and loses its
spectral option with this change. The fit driver keeps
its name for now; renaming the `gated_pc` model class and files is part of ADR 0051.

Open model question, not decided here: each domain is a separate multinomial per topic,
normalized on its own (MixEHR-style), so a topic has no learned share of conditions vs
drugs vs measurements, and a domain's pull on θ is its token count. A pooled
normalization over the concatenated vocabulary would learn that per-topic domain mix;
it is a one-switch experiment if wanted.

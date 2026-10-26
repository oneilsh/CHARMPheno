# ADR 0050 — One gated-LDA estimator; the supervised head and its satellites are removed

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
3. **One estimator**: `spark_vi.mllib.topic.GatedLDAEstimator` / `GatedLDAModel` carry
   the union of the live Params: the gate (`frontierCol`, `gateParent`, `gateNBg`,
   `gateTpn`), the schedule and seeds (`gateWarm`, `gateOpenSeed`, `gateSeedMode`
   default **balanced**, `gateSeedOwnCodes`, `gateSeedLeaves`, `gateSeedProfile`), the
   init (`random` | `spectral` with `spectralMethod`, `spectralD`, `spectralMaxVocab`,
   `spectralMinDocFreq`, `spectralBgAnchors`), `docConcentration`,
   `optimizeDocConcentration`, and multi-domain `featuresCols` / `domainBounds`.
4. **Remove unused options**: profile-η prior (`etaBoost`, `profileEta*`; the profile
   *seed* stays), guided anchors (`spectralAnchorCandidates`), `anchorScope=frontier`,
   `spectralTopoOrder=reverse`, `spectralMarginalFloor`, the dense multi-domain spectral
   path (multi-domain routes to the scalable one), `countTransform`, `alphaInit=
   equalized`, the ω modality weight and per-domain η, and the CAVI θ twins that only
   the head used.
5. **Housekeeping in the same pass**: one `_destroy_broadcast` in `spark_vi.core`
   (six copies today), one dict-λ → `expElogbeta` helper (three today), `DagLayout`
   built once per fit, and domain-neutral wording: the profile seed and own-code seed
   take the domain index as a parameter instead of assuming domain 0 is "conditions".

`batched_lr.py` stays; where it lives is ADR 0051's business (API and two engines).

## Alternatives considered

- **Keep `OnlinePCLDA` as a thin alias** at `weight_y = 0`: keeps two names for one
  model and every head Param inert on the API. Rejected.
- **Keep the head as an option**: three ADRs of engineering made it scale, and it is
  still a negative result; it stays recoverable from git (commit `fff7bf4067`).
- **Keep ω as an option**: a tempering weight tuned for a downstream task, never set by
  any run; observed volume cannot identify it. A learned per-topic domain mix is a
  different model question (noted below), not ω.

## Consequences

Supersedes ADRs 0038–0045 (the PC reference, head optimizer, head regularization, the
injectable-engine composition and the supervised λ correction). Amends ADR 0049: the
balanced open-time seed becomes the estimator default (`gateWarm` stays 0 by default;
the central configuration sets it). `gated_pc_readout` reconstructs the model as a
`GatedLDAModel` from the run's npz, so every existing run dir stays readable; models
persisted with the old MLlib PC writer do not load (none are used). The fit driver keeps
its name for now; renaming the `gated_pc` model class and files is part of ADR 0051.

Open model question, not decided here: each domain is a separate multinomial per topic,
normalized on its own (MixEHR-style), so a topic has no learned share of conditions vs
drugs vs measurements, and a domain's pull on θ is its token count. A pooled
normalization over the concatenated vocabulary would learn that per-topic domain mix;
it is a one-switch experiment if wanted.

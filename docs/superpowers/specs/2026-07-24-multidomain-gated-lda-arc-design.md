# Multi-domain (MixEHR-style) gated DAG LDA for case-finding — arc design

**Date:** 2026-07-24
**Branch:** `multidomain-spectral-init` (off merged `main`)
**Status:** Overarching arc design. Covers the full MixEHR-style multi-domain capability *through the model*, not just the spectral init. SP1/SP2 are detailed here; SP3/SP4 are scoped stubs to be expanded just-in-time.

## Goal

Incorporate multiple token domains (v1: conditions + drugs; designed for N) into the gated DAG topic model *end to end* — the model learns a per-domain topic-word distribution with a shared, DAG-gated document-topic mixture, so a corroborating domain (a drug attesting a diagnosis) sharpens node-vs-background contrast and raises case-finding **specificity** (lower per-node FDR at fixed sensitivity), concentrated at leaf/subtype nodes where conditions alone are ambiguous.

This is the MixEHR idea (Li et al. 2020, Nat. Commun. — per-modality topic-word distributions, shared patient-topic mixture) composed with this project's existing hard **DAG gate** (topics tied to a node's subtree via `allowed_set(frontier)`).

## The one idea, and why gate ⟂ domain

- **Gate** acts on **θ**: the allowed topic set A_d = `allowed_set(F_d)` = background block ∪ closure blocks of the frontier restricts *which topics* a document may use. (Unchanged from today.)
- **Domains** act on **β**: each domain m has its own topic-word distribution β^m, normalized *within* domain m. Each token is drawn from its own domain's β given the shared θ.
- These are **orthogonal axes** — the gate never touches β, domains never touch θ — which is why multi-domain drops onto the existing gated machinery instead of fighting it.

**Shared θ is the whole point.** A drug token and a condition token in the same visit are drawn from the *same* θ_d. That shared mixture is exactly what ties the two modalities and lets a drug corroborate a diagnosis.

## Model core

**Generative story (gated, MixEHR-style).** For document d with frontier F_d:
- θ_d ~ Dirichlet(α restricted to A_d) — one mixture, support on allowed topics only.
- For each domain m and each token in it: z ~ Categorical(θ_d), then code w ~ Categorical(β^m_z). Each β^m_k ~ Dirichlet(η_m), independent per (topic, domain).

**Variational inference — the single mechanism that changes.** In today's gated LDA, `expElogbeta = exp(digamma(λ) − digamma(λ.sum(axis=1)))` uses a **full-row** normalizer (spark_vi/models/topic/lda.py). Multi-domain replaces that one `sum(axis=1)` with a **per-domain-block** normalizer: expElogβ^m[k,w] = exp(ψ(λ^m_{k,w}) − ψ(Σ_{w′∈domain m} λ^m_{k,w′})). That *is* MixEHR's per-modality Dirichlet. Then:
- Per-token responsibility φ for a domain-m token: ∝ expElogθ_d[k] · expElogβ^m[k,w], over k ∈ A_d — **same loop shape** as today, indexing the right domain's β.
- γ_d = α[A_d] + Σ_m **ω_m** · Σ_{tokens in m} count·φ — the doc-topic posterior accumulates across all domains (the shared θ), each domain scaled by a modality weight ω_m (see "Domain heterogeneity").
- sstats and the SVI M-step blend run **per domain**: λ^m ← (1−ρ)λ^m + ρ(η_m + expElogβ^m · sstats^m). η_m is a per-domain prior.
- ELBO = Σ_m (domain-m token likelihood) + shared θ-KL + Σ_m β^m-KL.

**Hybrid storage/inference (decided).** Store λ **explicitly per-domain** — `{0: K×V_cond, 1: K×V_drug, …}`, each a proper Dirichlet — because that is the self-documenting, extensible, per-modality object the readouts, per-domain priors/weights, and the v2 extension all want. But keep **one** shared CAVI E-step: internally assemble the per-domain-normalized expElogβ^m blocks and feed the *existing* gated inference loop; do not write a second inference path, and do not refactor the base `OnlineLDA` (single-domain stays the degenerate N=1 case). The engineering cost is paid once, in the model core.

Why explicit-per-domain over concatenated-single-λ (both are the same model): explicit is type-safe on the domain boundary (a mis-normalization becomes unrepresentable rather than a silent mass-leak across ≥3 expElogβ recompute sites), naturally holds per-domain η_m/ω_m/diagnostics, and — see below — makes v2 a one-factor addition. The base-class blast radius that concatenation would have avoided is sidestepped by the hybrid: shared *inference*, explicit *storage*.

## Domain heterogeneity — how domains differ, and where each is handled

"Different usages" is several axes, not one:

| Axis | Example | Handled by |
|---|---|---|
| Vocab size / smoothing | conditions vocab huge, drugs smaller | per-domain β^m + per-domain Dirichlet prior η_m |
| **Token volume per doc** | 20 condition codes but 3 drug codes | **ω_m** modality weight (see below) — the load-bearing knob |
| Count semantics | a refilled drug appears 12×; a condition coded once | per-domain count transform (presence / log1p / raw) at **doc assembly** (L3), not the model |
| Whole-domain missingness | no meds recorded | per-domain token arrays: an absent domain contributes nothing to γ; θ from present domains (MCAR default = absence is no-evidence, stated) |
| Generic high-frequency codes | PPIs/analgesics co-prescribed everywhere (insight 0021 universal anchors) | background block + per-domain anchor floor; **measured** by the FDR-delta ablation, never assumed helpful |

### The volume axis: ω_m, and the β / π / ω three-way distinction

Because γ pools token counts across domains, the higher-volume domain **dominates θ**. Drugs (scripts, refills) are often that domain, so a patient with a rare disease + 100 SSRI scripts gets θ dragged to an anxiety/depression topic and the rare signal drowns. This bites hardest at **case-finding scoring**, where the fold-in is intentionally *ungated* (full-K, to surface unattested cases) — the gate mitigates the pull during training but not at the readout the whole method depends on.

The fix is a first-class per-domain modality weight ω_m:

  γ_d = α[A_d] + Σ_m **ω_m** · Σ_{tokens in m} count·φ

Three distinct objects are easy to conflate here; keeping them separate is the core rationale of this design:

- **β^m_k** — within-modality *shape* (which codes, how strongly), per topic per domain. Learned. Sums to 1, so it carries *which* drugs indicate a topic and how discriminatively — this already captures "anxiety is drug-corroborated, an untreatable rare disease is not" via the peakiness/distinctiveness of β^m_k. It **cannot** carry how drug-heavy a topic is (that magnitude is normalized away).
- **π_{k,m}** — topic k's *modality proportion* (how drug-heavy vs condition-heavy). A **generative** parameter that does NOT exist in MixEHR (there, modality is *observed*, not generated) and is orthogonal to β. Learnable in a fuller model (v2), but it makes modality volume *more* informative for θ (amplifies the pull) and invites the utilization confound into topic definitions. Not a de-bias.
- **ω_m** — a deliberate **de-bias / tempering** weight. This is what corrects the volume pull. It is **not** a generative parameter (tempering does not normalize into a joint over data), so it is **not learnable from the fit** — the likelihood always prefers "use all the tokens" and drags a learned weight back toward volume or a boundary. This is exactly why MixEHR and its whole lineage (MixEHR-Guided, MixEHR-SurG, MixEHR-SAGE) never learn a per-modality θ-volume weight: their objectives (phenotype discovery, code prediction, lab imputation, held-out predictive likelihood) *reward* using all evidence.

**ω is tuned against a held-out downstream objective, not learned from the fit.** "Get it from data" is right — but from the *task* (labeled validation cases → FDR/AUC), not the generative fit, because volume-per-modality is confounded by utilization/billing (non-phenotype) and the fit cannot see the counterfactual "balanced-utilization" world. MixEHR's deliberate observe-don't-generate-modality choice protects the topics from that confound; we want to *correct* the confound's residual pull on θ, and correcting a confound the data can't self-identify is precisely a tuned knob's job.

### The ω dial

ω is a dial with two named anchors, swept against the FDR-delta ablation:
- **ω = 1** — raw / faithful-MixEHR (volume speaks).
- **ω = inverse-volume (corpus-level)** — ω_m = 1 / (modality m's share of all corpus tokens); parity, every modality equal say. Cheap (counts only, no π). This is the natural "de-bias" anchor.

Inverse-volume is an **anchor, not the optimum**: it is informativeness-blind (it discounts a modality for being data-rich even when the data is good, so it cannot distinguish a drown-worthy universal SSRI from a keep-worthy node-specific drug) and is the correct correction only under "all volume excess is confound." The task-swept optimum sits somewhere between raw and parity. Corpus-level, not per-patient (patients are too noisy — a single incidental drug token would swing an equalized θ). A per-modality **log1p** count transform (at L3) is a gentle, parameter-free partial compressor available alongside ω.

## v2 — generative π_{k,m}, and why it stays cheap and deferred

v2 generates each token's modality: z ~ θ, **m ~ Categorical(π_z)**, x ~ φ^m_z. Two facts make it a clean *future* extension, not something to bake in:

1. **v2 nests MixEHR exactly.** MixEHR = v2 with π tied across topics (π_{k,m} = π_m). When modality proportion carries no topic information, p(m|z) is a topic-independent constant that drops out of the z-posterior, recovering MixEHR's posterior over (θ, z) exactly. v2 departs from MixEHR only insofar as learned π_k genuinely fan out — a testable hypothesis (do they, or collapse to a common π_m?).
2. **v2 is a one-factor add on the v1 core**, *because* v1 stores λ per-domain and routes each token by its observed modality: E-step gains φ_k ∝ expElogθ_k · expElogβ^m_k[w] · **expElogπ_k[m]** (modality already in hand); one new K×M sstat n_{k,m}; a K×M Dirichlet π M-step blended by the same ρ; one ELBO term. Fully conjugate — no new inference machinery, no doc/shim/gate/SVI-loop change.

**v1 invariant (the "v2 seam"):** form the per-token responsibility as `expElogθ · (per-domain β factor)` with the modality index live at that point; do not optimize the E-step into a domain-agnostic concatenated gather that discards which modality a token came from. Written as a named invariant so an implementer does not simplify it away.

**Caveats keeping v2 deferred:** it is a richer *faithful* model that *amplifies* volume (not a de-bias — ω is still needed on top, and ω can never come for free from any generative model), and learning topic-varying π reimports the utilization confound. v2 is a research extension evaluated by whether π_k actually fan out, after SP2 proves the ω dial moves the FDR needle.

## Representation across the three layers

- **L1 engine (integer ids only).** Doc = `frontier` (frozenset of DAG node ids) + **per-domain token arrays** (domain d's indices in its own [0, V_d)). Model stores per-domain λ_m. The engine proper never sees a concatenated vector.
- **L2 mllib shim.** One `featuresCol` = **concatenated SparseVector** over [0, ΣV_m) + a **`domainBounds` Param** (cumulative offsets [0, V_cond, …, ΣV_m]); the shim **splits on ingest** into per-domain arrays before building the engine doc. Reuses the single-featuresCol convention, `VIRunner`, and `labelCol`=frontier untouched; N-domain-general, no schema churn. Concatenation is a wire-format detail, not the model's storage. Domains are **code-defined** (a code is intrinsically a condition/drug/lab); a context modality (inpatient vs outpatient) is handled by minting distinct codes in a new domain, not by a per-occurrence tag. The Model emits per-domain λ_m (and v2 π).
- **L3 charmpheno (concept-ids).** Owns the **load-bearing prerequisite** (below): doc-unit bundling. Maps concept-ids → per-domain integer vocab (contiguous per-domain blocks), sets `domain_bounds`, applies the per-modality count transform, carries ω_m config. Engine stays id-agnostic.

**Init → model handoff (SP1 → SP2 tie).** The spectral init works **concatenated internally** — it must, because the cross-block co-occurrence Q_CD is the entire signal — then `split_domains` cuts the joint β into per-domain seeds, and `initialize_global` sets **λ_m = scale·β^m + prior** per domain. The split is not merely a readout; it is the initializer for each λ_m.

## Load-bearing prerequisite

The cross-domain tie is Q_CD[i,j] = Σ_k A_k · β^C_ki · β^D_kj = (B_C)ᵀ A (B_D), which exists only from **within-document** cross-domain co-occurrence. If drugs and conditions live in separate documents, Q_CD = 0 and the two anchor hulls disconnect. The doc-unit (decision 0018 seam) MUST bundle co-occurring drugs and conditions into one document (a visit / condition-era window). Non-negotiable; stated in every layer's docstrings.

## Sub-project decomposition and sequencing

Sequence SP1 → SP2 → SP3 → SP4; each its own spec → plan → build.

- **SP1 — Multi-domain spectral init.** Per-domain candidate floor in `find_anchors`; `split_domains` post-recovery; two-domain planted generator; joint-recovery + FDR-delta acceptance via the Gibbs oracle. De-risks anchor-basis alignment cheaply. Plan drafted at `docs/superpowers/plans/2026-07-24-multidomain-spectral-init.md` — **light revision:** reframe the split as the *seed for SP2's λ_m* (not just a readout); otherwise as written.
- **SP2 — Multi-domain gated LDA core.** Per-domain λ_m (block-Dirichlet expectation), shared gated θ with **ω_m** + per-modality θ-contribution instrumentation, per-domain η_m, per-domain sstats/M-step, multi-domain ELBO, the **v2-seam invariant**; multi-domain Gibbs oracle + planted per-domain recovery + SVI≈Gibbs equivalence. Seeded by SP1. The statistical heart.
- **SP3 — Doc representation + mllib shim.** **EXPANDED 2026-07-25 and SPLIT IN TWO** (it spanned three layers with different constraints, and the third depends on the first two):
  - **SP3a — persistence + shim**, `spark-vi` only: `docs/superpowers/specs/2026-07-25-sp3a-multidomain-persistence-and-shim-design.md` — **DONE (2026-07-26)**: dict-λ persistence, `get_metadata`, `featuresCols` shim, ω/η, scalable-init recovery gate; whole-branch review clean; insight 0070.
  - **SP3b — drug domain + cloud driver**, `charmpheno` + `analysis/cloud`: `docs/superpowers/specs/2026-07-25-sp3b-drug-domain-and-cloud-driver-design.md` — **BUILD DONE (2026-07-26)**: `drug_era` loading, two-domain BOW + bundle + per-domain strip, the SP3b↔SP3a seam gate (green), the cloud driver with explicit seed + dead-node read; whole-branch review clean + fix wave. **Cluster smoke (`make -C analysis/cloud multidomain-bq-smoke`) is USER-RUN on Dataproc — the first real-data two-domain fit — not yet run.** Parked for SP4: window both domains against one `case_finding_index_table` (robust drug-windowing + lookback mode), drug-only-doc `source_cohort`.

  Two amendments the expansion made to this stub:
  - **The shim takes SEPARATE per-domain feature columns** (`featuresCols`), not the concatenated `featuresCol` + `domainBounds` Param this stub proposed. User decision, 2026-07-25. Bounds are derived from the per-column vector sizes and every row validated against them, since a mis-sized vector could otherwise re-lay-out the vocabulary silently.
  - **Blocker 1 — answered YES (adequate), NOT "immune". CORRECTED 2026-07-26.** A 3-seed probe (2026-07-25) first suggested the scalable path "matched or beat" the dense+floor seed and the immunity hypothesis "held" — that was a lucky-draw reading. An 8-projection-seed test then showed the scalable SEED is fragile (an exclusive-node signal can be 0.0 at init on some draws), and a naive dense-vs-scalable comparison also tripped a metric artifact on the b_only node's non-identifying shared-support cell. The settled, honest result: `test_scalable_init_recovers_every_identifying_signal` fits the scalable seed across 8 draws and confirms the gated EM recovers every node's IDENTIFYING per-domain signal above a floor — so the production init path is **adequate**, and no `domain_bounds` plumbing is needed there. What is NOT claimed: seed-quality parity with dense, or that the dense floor is redundant in general. Recovery is carried by EM, not the seed; consistent with insight 0067 (the dense floor's apparent value is largely a degenerate-plant artifact). The stronger "immune" framing is withdrawn.

  **SP3 BLOCKERS carried over from SP2's final review — read before planning SP3:**

  1. **The per-domain candidate floor exists ONLY on the dense driver init path.** SP1 threaded `domain_bounds` into `find_anchors` → `spectral_init.spectral_init_beta`, `gated_init.spectral_block_aligned_lambda` and `dag_placement.fit_gated`. Its production twin `gated_init.scalable_block_aligned_lambda` and the primitive it calls, `spectral_init_scalable.find_anchors_projected`, **did not get it** — and the mllib shim routes to the *scalable* path at scale (`spark_vi/mllib/topic/gated_lda.py`, `resolve_spectral_method` → `"scalable"` above `spectralMaxVocab`). On the dense path that floor is the difference between per-domain recovery 0.005 and 0.675 at `random_seed=0` (insight 0066), so a multi-domain fit at production vocabulary size is currently **unvalidated** on the code path it will actually take.
     **IMMUNITY HYPOTHESIS (untested — this is the thing to test first, not to assume).** The two floors are not the same rule. Dense `find_anchors` uses a *mean-relative* floor (`marginal ≥ min_marginal_frac × mean nonzero marginal`), and that is exactly what a denser domain can dominate. `find_anchors_projected` instead uses an **absolute document-frequency floor** (`df_w ≥ min_doc_freq`, default 5), chosen under ADR 0032 precisely because the mean-relative rule "over-excludes rare-but-pure words" in the sketch setting. An absolute per-word threshold has no pooled mean to be swamped by, so the scalable path may be **structurally immune to the specific pooled-mean failure** — a sparse domain's real anchor clears `df ≥ 5` on its own. If that holds, no `domain_bounds` plumbing is needed there at all and the right SP3 deliverable is a TEST plus a docstring stating the equivalence. If it does not hold (e.g. because the sparse domain's anchors are genuinely low-df, not merely low-marginal), the scalable path needs its own per-domain rule and threading `domain_bounds` for symmetry would be the wrong fix anyway.
     **SP3 acceptance:** run the multi-domain planted two-domain corpus through `scalable_block_aligned_lambda` and compare per-domain recovery against the dense `domain_bounds` seed. `spectral_block_aligned_lambda`'s docstring points here.
  2. **Multi-domain fits cannot be checkpointed, exported or resumed.** `io.export.save_result` now raises `UnsupportedGlobalParamError` on the per-domain dict λ instead of silently writing an unreadable 0-d pickled object array (which `load_result`, reading with `allow_pickle=False`, could never load). SP3 owns the real writer — one `params/lambda_<m>.npy` per domain plus the domain sizes in the manifest, and the matching load/resume path. Pinned by `tests/test_gated_lda_integration.py::test_multidomain_gated_fit_with_checkpointing_fails_loudly`.
  3. **`get_metadata` omits `omega` / `domains` / `eta_m`**, so a saved multi-domain result is not reconstructable even once (2) is fixed. Same task.
  4. **η provenance differs by mode:** multi-domain `update_global` / `compute_elbo` read the per-domain η from instance state (`self._eta_domains`), while the single-domain path reads `global_params["eta"]`. A resumed fit therefore takes η from the reconstructed *model*, not from the checkpoint. Decide and document this when (2) is built.
- **SP4 — charmpheno cloud driver + real-cohort assembly + ω-swept FDR-delta (stub).** Doc-unit bundling (Q_CD prereq), concept-id→per-domain vocab, per-modality count transform, ω config; the real-data specificity green light. Depends on the separately-spec'd condition/drug DAG builders. Expand just-in-time.

**v2 (generative π)** — research extension after SP2; out of the main line.

## Validation strategy (mirrors the branch's oracle-validates-SVI convention)

1. **Multi-domain Gibbs oracle** (extend `fit_gated`): per-token conditional uses the per-domain word-topic factor (n_kw + η_m)/(n_{k,·∈m} + V_m·η_m). The reference for the SVI core.
2. **Planted per-domain recovery**: β^C and β^D recover the planted phenotypes, incl. topics anchored from the drug domain alone.
3. **SVI ≈ Gibbs equivalence** on a small multi-domain corpus: multi-domain SVI fold-in `node_affinity` matches the oracle.
4. **ω-swept FDR-delta** (specificity green light): sweep ω_drug 1 → inverse-volume; per-node FDR at fixed sensitivity; node-specific drug lowers leaf FDR, generic drug does not (control); leakage held fixed (node-marker codes stripped in both arms); **measure where the mass LANDED, per modality** — each modality's marginal contribution to the *fitted* θ (e.g. the per-domain decomposition of Σ_d θ_d under the fitted λ_m), or a leave-one-domain-out refit at each ω.

   **AMENDED (insight 0069, SP2).** This criterion originally read "instrument per-modality θ-contribution", meaning SP2's `theta_contribution_by_domain` stat. That stat is **refuted as a diagnostic**: it equals ω_m × (domain-m token volume) *bit-exactly, for every ω*, because a partition of the γ increment is a partition of the evidence and the CAVI γ-update conserves evidence mass across topics. It is computable without fitting anything and can never show whether a modality dominates the shared θ. Keep emitting it — it is an exact trace that ω was applied, to which domains and in what proportion, and the cheapest regression guard on "ω weights θ" — but **do not accept it as the ω-tuning read**, and do not let it satisfy this acceptance item. Tuning ω against a downstream task metric remains the only validated route.
5. **v2-seam invariant test**: assert the E-step keeps each token's modality live at φ-formation.

## Constraints (branch conventions)

- **Engine domain-agnostic:** integer token/domain-boundary ids only; NO clinical/OMOP/EHR vocabulary in `spark_vi/**` or its tests. The domain edge lives in `analysis/cloud`.
- **Cite methods in docstrings:** anchor-word = Arora, Ge, Halpern, Mimno, Moitra, Sontag, Wu, Zhu 2013 (ICML). Multi-modal shared-θ / per-modality β = Li, Nair, Lu et al. 2020, Nat. Commun. (MixEHR). Corroboration / anchor-and-learn phenotyping = Halpern, Horng, Choi, Sontag 2016 (JAMIA). Modality weighting as pseudo-likelihood tempering: cite as such, not as the vanilla generative likelihood.
- **No LaTeX; Unicode Greek only** (α, β, θ, Σ, η, λ, ω, π, ψ, ρ).
- **TDD** (superpowers:test-driven-development).
- This branch does **not** auto-push; push only when asked.
- Commit trailer EXACTLY:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  ```

## References

- MixEHR: Li et al. 2020, Nat. Commun. 11:2536 — per-modality topic-word distributions, shared patient-topic mixture; modality *observed* not generated; no θ-volume weighting (objective is phenotype discovery / prediction).
- Anchor-word: Arora et al. 2013, ICML.
- Anchor-and-learn phenotyping: Halpern, Horng, Choi, Sontag 2016, JAMIA.
- SP1 brief: `docs/superpowers/specs/2026-07-24-multidomain-spectral-init-case-finding.md`; SP1 plan: `docs/superpowers/plans/2026-07-24-multidomain-spectral-init.md`.
- Decision 0018 (doc-unit abstraction — the Q_CD prerequisite). Insight 0021 (universal-anchor mass concentration). FDR readout plan: `2026-07-21-case-finding-fdr-readout.md`.

# Case-finding lever retrospective — are we chasing our tail?

**Date:** 2026-07-23 · **Branch:** case-finding · **Author:** review with Claude

A collation of every lever tried on the rare-disease case-finding task, to confirm the
"binding constraint is information, not the model" conclusion is *earned* and to surface any
untested path. Short answer: **we are not chasing our tail — the model side is systematically
ruled out (six nulls), the one durable win is a readout lens, and the two un-chased levers are
named.**

## The task

Place held-out patients in a rare-disease DAG (rare6: amyloidosis / sarcoidosis / SLE / EDS /
scleroderma / myasthenia families, 26 scoreable nodes) from their condition-code history, to find
patients whose rare disease is uncoded. Gated topic model: each node gets a topic block welded to
its subtree's documents by a hard DAG gate; a held-out patient's per-node affinity profile is the
placement readout. Detection = separate the 1382 true rare cases from ~29595 background at 4.5%
prevalence.

## The levers (chronological)

| # | Lever | What it tested | Exp(s) | Result | Evidence |
|---|-------|----------------|--------|--------|----------|
| 0 | **theta-mass placement** | node-block posterior mass as the score | 0055-0062 | baseline | ROC ~0.65; FDR found ZERO discoveries (signal buried) |
| 1 | **LR readout (alpha->inf lift limit)** | read learned lambda as a per-node Naive-Bayes detector vs the flat base rate | 0061+ | **WIN** | **ROC 0.78 = +0.11-0.13 over theta-mass; 2.6x PR-AUC** |
| 2 | Fixed asymmetric alpha (node_alpha_scale 0.1) | a peaked doc-topic prior on disease nodes | 0063/0064 | null | <= 0.003 on every detection metric (0060) |
| 3 | Learned per-node alpha (optimize_alpha) | let the fit choose each node's Dirichlet concentration | 0065/0067 | null | LR 0.777 vs symmetric 0.778; shifts argmax but not discrimination (0061) |
| 4 | Deployment-alpha = symmetric | decouple the fit-aid alpha from the fold-in prior | 0067 | **fix, not gain** | recovered mrr 0.442 -> 0.596 (undid the fitted-prior argmax bias) |
| 5 | Explain-away routing | responsibility-weight each code so comorbid codes route to background | 0067 | null | LR ROC -0.011; the FN comorbidity-drag class it targeted was UNMOVED (0062) |
| 6 | Background capacity (n_bg 40 -> 80) | more background topics to absorb comorbid clusters | 0068 | null | FP class 13158 -> 12992 (noise); bg_fpr flat (0062) |
| 7 | count-mode log1p | saturate repeated codes | 0067 | null | PR-AUC 0.222 -> 0.233 |
| 8 | **Reverse-topo spectral init** | recover leaves-first, deflate against descendants, to sharpen leaf topics | 0069 | null | LR 0.779 vs 0.778; FN class 276 = 276 = 276 (0063) |

## The pattern

**One durable win, and it is a READOUT LENS, not a model change.** The LR readout (lever 1) reads the
same learned lambda through a background-relative Naive-Bayes lens and beats theta-mass by +0.11-0.13
ROC. It is parameter-free at the alpha->inf lift limit. Everything else on the table changes the MODEL
(prior, scoring reweight, init geometry, capacity) and every one is null.

**Both error classes are information-bound, and no model lever moves either.**
- **False negatives (rare called background) = 276, byte-identical across 0067/0068/0069.** These are
  data-starved patients: few codes, or a signature that is common (e.g. a Sarcoidosis patient whose only
  code is Rheumatoid arthritis). Routing (5), capacity (6), and init (8) all left this class exactly
  unchanged — the evidence is not there to reweight.
- **False positives (background called rare) ~13000 = 44% of background @80% sens, barely moved by
  capacity.** These are genuinely-disease-like background patients: anemia + CKD really are SLE-associated
  (anemia of chronic disease, lupus nephritis), so from condition codes alone they are indistinguishable
  from SLE. More background topics cannot manufacture a distinction the codes do not carry.

**Confounds we ruled out along the way:** the forward-mode "asymmetric alpha helps" (0.709 vs 0.660) was
an init/epochs confound that did not replicate on lookback (0060); the learned-alpha "hurts placement"
was a deployment-prior argmax bias, not a worse fit (0061, fixed by lever 4); single-seed alpha orderings
are multimodal (0059), so we read gross structure not per-node points.

## Are we chasing our tail? No.

The six model-side nulls are not random flailing — they span the *entire* model-side design space
systematically: the **prior** (levers 2,3), the **scoring scheme** (levers 5,7), the **init geometry**
(lever 8), and **model capacity** (lever 6). They converge, with pre-registered A/Bs and stable
error-class counts, on one conclusion: **the condition-code stream does not contain the information to
separate these cases, so no transform of it will.** That is a real, defensible negative result, and the
LR lens is a genuine methodological contribution (read a gated topic model as a per-node NB detector at
the parameter-free lift limit).

## The un-chased levers (so nothing is missed)

1. **Richer features (meds / labs) — the MixEHR multi-domain direction.** The ONLY lever that attacks the
   actual binding constraint (adds separating information the condition codes lack). Not tested. This is
   the strategic next step.
2. **LR-FDR readout.** The Efron two-groups empirical-null FDR machinery (per_node_discoveries) is
   currently applied ONLY to theta-mass, where it found zero discoveries because theta-mass buries the
   signal. It is score-agnostic; wiring it onto the LR / explain-away scores tests whether LR's +0.12-ROC
   edge yields actual FDR-controlled discoveries. Cheap, post-hoc, no re-fit. Queued (spec pending).
3. **Held-out log-likelihood.** The fair "is the learned alpha a better model" test (the objective
   optimize_alpha actually maximizes) was never measured — only ranking/detection were. Would settle the
   placement question the alpha work left open, independent of the detection nulls.

## Dead / orphaned code surfaced during the walkthrough

- `identifiability_annotation` (dag_placement.py) — tested but ZERO production consumers; a pg-stm
  identifiability-arc vestige. Either wire in (flag structurally-unidentifiable nodes in the readout,
  connecting to the pg-stm compiler) or remove + its tests.
- `fit_gated` / `profile` (Gibbs oracle) — validator/test-only; production uses the SVI GatedOnlineLDA +
  evaluate. Legitimate, but label as oracle so it is not mistaken for the live path.
- `_zib_empirical_gap` / `zib_gap` — printed + saved to the manifest, but its design consumer (a
  parametric-vs-empirical exportable null for an unbuilt "sub-project 2") does not exist; low-value.

## Bottom line

The gated ontology placement engine is sound and reviewed (walkthrough Lessons 1-7; one real bug — the
sparse/dense log1p discriminator — caught and fixed). On the rare6 condition-code task the model side is
exhausted and the answer is information. The publishable story is: a hard-gated DAG topic model + an
alpha->inf background-relative LR readout that surfaces mass-starved node signal theta-mass buries,
together with a clean negative result that scoring/prior/init/capacity do not move an information-limited
task — motivating the multi-domain (meds/labs) direction.

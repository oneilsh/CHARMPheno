# 0079 — Gated-PC per-node topics starve at a hard depth-5 cliff; label coverage is not topic learnability

**Date:** 2026-09-05
**Topic:** lda
**Status:** Observed, MECHANISM UNRESOLVED — **the "uniform depth-5 cliff" framing
(body) and the later "information ceiling / init-refuted" reading (Refinement) are
BOTH over-reads. What is established: the fed/starved split tracks code-separability
after the leakage strip. What is NOT: whether the starved-but-populated nodes starve
from genuine non-differentiability, the aggressive global strip, hard deflation, or a
flat-start init trap — these are separable by experiment (see the Correction). A
basic K≈80 LDA on one subDAG is the decisive next test.**

**Relates to:** 0078 (episode anchoring un-starves the LABEL space — this is the
orthogonal *topic* view), 0071 ("as ONE fit K≈3,800 is not viable — decompose by
body system into a cascade" — the monolith 0111 fits is exactly what starves),
the 0111 scouting report (2026-09-02, Finding 2: case-finding signal attenuates
monotonically with depth; deep nodes are thin), 0080 (the fed shallow topics).
This is a cleaner *quantification* (topic λ-evidence by depth) of a known
depth-thinness, not a new failure mode — its one novel claim is the label↔topic
split below.

## Observation

Inspecting the fitted 0111 episode-arm globals (`gated_pc_result.npz` λ +
`readout_heads_gated_pc.npz`, off-cluster via `analysis/cloud/inspect_topics.py`)
shows a **hard starvation cliff at depth 5** in the per-node topic blocks. Per-topic
*evidence* (λ.sum(), the posterior pseudo-count mass a topic accrued) by DAG depth:

| depth | n nodes | median evidence | median eff.support / V |
|--:|--:|--:|--:|
| 1–4 | 182 | 1.0e3 – 7.9e6 | sharp (0.04–0.07) |
| 5 | 419 | **5.15** | **1.00 (flat = prior)** |
| 6–16 | 2349 | ~5.1 | 1.00 |

Evidence falls ~200× between depth 4 (~1.0e3) and depth 5 (~5.15 = the Dirichlet
prior floor = essentially zero tokens). 1970/2713 node topics (73%) sit at
`support_frac > 0.5` (near-prior); 177 heads are degenerate. Only the top four
Mondo levels (182 nodes, 7% of the label space) are actually fit; the 8 shared
background topics absorb the bulk of the corpus (evidence 4–7e6 each). The
topic *words* confirm this is real, not a labelling artifact: every depth-5+
topic's top concepts are the corpus-marginal common codes at ~uniform mass
(`cherry hemangioma` → "Essential hypertension · Pain · Hyperlipidemia";
lymphomas → generic *normal* lab panels) — no disease-specific signal.

## Interpretation

This is the closure-gated topic architecture, not (only) rarity. Each doc's
tokens are allocated across `allowed(v)` = background ∪ ancestor blocks ∪ the
node's own block; the node's own block receives only tokens **specific to it and
not already explained by an ancestor or background topic**. With the label code
stripped (`strip_mode both`), a fine-grained Mondo leaf contributes almost no
vocabulary its parent category doesn't, so its block starves even when the node
has adequate labelled documents.

This also bounds the readout directly at DEPLOYMENT. Training gates each doc's
CAVI to `allowed_set(frontier)`, but `transform` folds held-out docs in UNGATED
(full-K) — you can't gate by the label you're predicting (`gated_lda.py:4`). A
starved topic (β ≈ uniform) cannot win words in the fold-in responsibility
softmax, so its θ dimension is **inert at test** — θ_c ≈ 0 for every doc, disease
or not — and node c's localized head has nothing to read but ancestors/background
(population context). The ungated fold-in adds no noise (flat topics stay ~0, they
don't steal mass) and the head is localized anyway; the killer is the starvation,
not the ungating. Fix the topics and the ungated deployment works as designed
(that IS the placement task).

Critically, this means **label coverage ≠ topic learnability**. Insight 0078
showed episode anchoring un-starves the *label* space (2583/2714 nodes reach ≥20
gated onset episodes as positives). 0079 is orthogonal: those same deep nodes
still have flat *topics*, because label positives feed the readout, but the
gated allocation starves the node's own topic-word posterior. The macro AUC
(0.72) and cond_AUC survive because the readout decodes deep nodes through their
*ancestors'* sharp topics + background (the "borrowed topics" column), never
through the node's own empty block — which is exactly why cond_AUC slides with
depth and top-1 loses on deep small parents.

## Implications

- The interpretable-per-node-topic goal is not met past depth ~4 with this
  MONOLITHIC whole-Mondo fit, regardless of sampling. Episode anchoring cannot fix
  a depth-5 cliff; the 0111-vs-0112 comparison must be judged on **deep-node
  evidence** (does anchoring move the depth-5 floor at all?), not only on AUC.
- The known-good direction is already on record: 0071 concluded whole-Mondo "as
  ONE fit is not viable — decompose by body system into a cascade," each branch fit
  at K~few-hundred (the regime where 0019-style phenotype emergence works). The
  depth-5 cliff is fresh evidence *for* that conclusion, seen in the fitted topics
  rather than only in the head-cost argument.
- Other candidate model levers (each its own experiment + ADR, and note the priors
  below): tpn>1 or shared subtree topics; relaxing the closure gate so a leaf
  competes for its ancestors' tokens; larger n_bg. NOT `weight_y>0` (PC is a
  standing-constraint dead end, 0066), and init is a null lever historically (0063)
  though untested at this scale.
- The readout's raw-θ head coefficients EXPLODE (±1e4–1e5) for these low-variance
  starved topics (V = W_std/σ, σ→0); read standardized W_std, not raw V, for any
  loadings interpretation (`inspect_topics` prefers the ckpt W_std when present).

## Refinement (2026-09-05, with `--grep` on specific high-n deep nodes) — the cliff is code-separability after the leakage strip, not depth/prevalence/init/sampling

A `--grep` lookup of well-POPULATED deep nodes settles the mechanism, and refutes
the depth/prevalence framing above and the flat-start-init-trap hypothesis floated
in the same session:

| deep node | depth | n | evidence | frac | verdict |
|---|--:|--:|--:|--:|---|
| hemorrhoid | 8 | 2138 | 1.5e4 | 0.01 | fed, sharp |
| varicella zoster | 8 | 2222 | 3641 | 0.01 | fed |
| disorder of facial skeleton | 7 | 2482 | 2606 | 0.01 | fed |
| ischemic stroke | 8 | 2618 | 744 | 0.00 | fed, sharp |
| female breast carcinoma | 10 | 1161 | **5.2** | **1.00** | starved |
| invasive ductal / lobular / in-situ / male breast ca | 9–11 | — | **5.1** | **1.00** | starved |
| strep sore throat / pneumonia (parent "strep infection" is fed, ev 143) | 8–9 | — | **5.1** | **1.00** | starved |

Same depths, same-or-higher n, opposite outcomes — so NOT prevalence, NOT depth,
NOT a random init trap (systematic, not scattered), NOT minibatch sparsity
(ischemic stroke n=2618 is fed). **The discriminator is code-separability after the
leakage strip.** `strip_mode="both"` drops EVERY DAG-node code from the BOW, train
included (`case_finding_assembly.py:449-452`: `drop_idxs = {vocab_map[c] for c in
before_dag.nodes()}`). So a node's topic is built from a vocabulary with its own
defining code — and all Mondo codes — removed; it learns only if it can WIN its
distinctive tokens in the UNSTRIPPED domains (drugs, labs, non-Mondo symptoms)
against its sharper ancestors. Ischemic stroke keeps antithrombotics + distinctive
management → fed. The breast-carcinoma subtypes and strep syndromes are flat.

**Correction (2026-09-05, same-session pushback) — do NOT read this as a clean
information ceiling, and do NOT read it as refuting init. Both were over-reads.**
The fed/starved split *correlates with* code-separability, but that is consistent
with several mechanisms this data does NOT separate:

1. **Deflation IS the flat-start trap (so spectral init is a LEADING lever, not
   refuted).** The gate subtracts nothing; a child topic competes against its sharp
   high-mass ancestors for the shared tokens in the CAVI responsibility and only
   wins what the ancestors don't already explain. A child that starts FLAT (random
   init) cannot win any token against a sharp ancestor → stays flat, self-
   reinforcing; a child seeded SHARP on its anchor words (spectral init) can win its
   distinctive tokens. This bites *harder with depth* (a taller ancestor stack to
   lose to), so 0063's "init is null" — measured on a 170-block SHALLOW DAG — does
   not transfer; init at whole-Mondo depth is genuinely untested. The fed deep nodes
   carrying sharp signal are the existence proof that sharp deep topics are
   achievable, i.e. the trap is escapable.
2. **The signal probably EXISTS for many "starved" nodes.** A basic K≈80 LDA on one
   subDAG's patients would likely separate e.g. breast-ca subtypes by treatment
   intensity / follow-up — which live in the (UNSTRIPPED) drug/lab domains. So the
   starvation is likely a *modeling artifact* (deflation + random init + the
   aggressive GLOBAL strip removing comorbidity codes) that a flat LDA sidesteps —
   NOT an information vacuum. Only genuinely-identical siblings (same disease, two
   names) are a true ceiling, and they are a subset, not the whole tail.
3. **Strip SCOPE is a lever.** The global strip drops ALL DAG-node codes, so
   comorbidity DISEASE codes vanish too (surviving only as drug/lab proxies). A
   closure-only strip (mask a node's own+ancestor codes per-node in the gated
   E-step) would keep comorbidity signal; leaving codes in is not an option (a topic
   anchored on its own code is a leaky SINK that defeats the uncoded-discovery goal
   — strip is load-bearing, only its scope is negotiable).

**The decisive cheap test:** a basic K≈80 LDA on one subDAG's patients. Clean
subtype signal ⇒ modeling artifact ⇒ spectral init / closure-strip / softer gate
become A/B-able levers. No signal ⇒ that slice is a genuine ceiling. Until that
runs, the mechanism is UNRESOLVED — the earlier "information ceiling / init
refuted" reading is retracted. The cascade (0071) and multi-domain (0062) points
still stand.

**Setting context:** exp 0111 episode arm — gated-PC (weight_y 0, unsupervised
topics + separate L-BFGS readout), whole-Mondo native DAG (C=2714, K=2721,
n_bg=8, tpn=1), 3 domains (condition/measurement/drug), episode index (gap 90d,
cap 3, 365d label), lookback 1825d, doc_concentration 0.5. Read from the fit's
own saved globals via `inspect_topics.py --grep`; the matched-random control
(0112) had not yet been fit.

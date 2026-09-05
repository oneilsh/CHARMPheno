# 0079 — Gated-PC per-node topics starve at a hard depth-5 cliff; label coverage is not topic learnability

**Date:** 2026-09-05
**Topic:** lda
**Status:** Observed

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

**Setting context:** exp 0111 episode arm — gated-PC (weight_y 0, unsupervised
topics + separate L-BFGS readout), whole-Mondo native DAG (C=2714, K=2721,
n_bg=8, tpn=1), 3 domains (condition/measurement/drug), episode index (gap 90d,
cap 3, 365d label), lookback 1825d, doc_concentration 0.5. Read from the fit's
own saved globals; the matched-random control (0112) had not yet been fit.

# Explain-away (responsibility-weighted) LR placement scorer

**Status:** design (brainstormed 2026-07-23, branch `case-finding`)

## Problem

The current per-node likelihood-ratio (LR) placement score scores every code
against every node independently, vs the flat corpus base rate:

    s(u) = Σ_w cnt(w) · log[P(w|u)/bg(w)]

Pixel-peeping the false-positive / false-negative cases (exp 0067) showed the
failure mode: a comorbidity-heavy patient accumulates many small NEGATIVE
contributions (each generic code is a little less likely under a rare-disease
topic than the base rate), which drag the true node below threshold (false
negative), while a background patient with a code that happens to be elevated in
a node's learned topic gets a spurious positive (false positive). Dampening
repeated codes (`--count-mode log1p`) was ~neutral on 0067 (PR-AUC 0.222->0.233),
so the issue is structural, not repeat-counting: **it is unfair for a code that a
node did not "claim" to count against (or for) that node.**

## Idea: explain-away routing

Let each code compete for a home across ALL topics first (the mixture E-step's
"explain-away", Pearl 1988), then weight its evidence for a node by how much it
belongs there:

    s(u) = Σ_w cnt(w) · r(u|w) · log[P(w|u)/bg(w)]

where r(u|w) is the code's soft responsibility on node u's topic block. A comorbid
code routes to a background topic -> r(foreground|w) ~ 0 -> its negative log-ratio
is suppressed to ~0 (fixes the FN drag) and it does not spuriously support a
foreground node (fixes the FP, IFF a background topic wins the code -- so the FP
improvement scales with background capacity n_bg). A distinctive rare code routes
to its node -> full positive contribution, unchanged.

This is a third scoring scheme between theta-mass (codes compete on the simplex
but node topics are mass-starved so signal is buried) and the current LR (evidence
weighting but no competition, so comorbidity penalizes).

## Design decisions (locked in brainstorming)

1. **Soft (responsibility-weighted) routing**, not hard argmax: r(u|w) is the
   fraction of code w's total topic-probability that lands on node u's block. No
   brittle ties; partial evidence is kept.
2. **Uniform topic prior in the routing** (responsibility ∝ P(w|topic), NOT
   mass-weighted): mass-weighting would let the 40 background topics swamp the 130
   node topics and route everything to background.
3. **Routing uses the raw normalized topic-word distributions** P(w|topic) =
   λ[k]/Σλ[k]; the **evidence term keeps the existing α handling** (including the
   parameter-free α->∞ lift limit). Two separate uses of λ.
4. **Report both scores side-by-side** in one lr-readout run (plain LR + explain-
   away detection blocks); the α sweep + AUC table stay on plain LR (the tuning
   axis), explain-away reported at the α->∞ limit.
5. **No engine re-fit, no new experiment**: a post-hoc readout on exp 0067's saved
   λ + cached bundle.

## The math

Notation: λ [K x V] learned topic-word counts; block(u) the tpn topic rows of node
u; bg(w) the corpus base rate; epsilon floor.

**Evidence (unchanged, from `_lr_logratio_rows`):**

    logratio[u,w] = log[P(w|u)/bg(w)],
      P(w|u) = (Σ_{k in block(u)} λ[k,w] + α·bg(w)) / (Σλ(u) + α)   (finite α)
      logratio[u,w] = nc[u,w]/bg(w) - Σ_w nc[u,w]                   (α = inf limit)

**Routing (new):**

    Ptopic[k,w] = λ[k,w] / max(Σ_w λ[k,w], epsilon)        # P(w|topic k), per-topic norm
    Rtopic[k,w] = Ptopic[k,w] / max(Σ_j Ptopic[j,w], epsilon)   # responsibility, uniform prior
    Rnode[u,w]  = Σ_{k in block(u)} Rtopic[k,w]            # node-block responsibility, [n_nodes x V]

A code w unseen in every topic (column all zero) -> Rtopic[:,w] = 0 -> contributes
nowhere (no evidence). Background topics are in the softmax denominator, so
comorbid codes they explain get Rnode(foreground) ~ 0. Σ over ALL blocks
(background + nodes) of the per-block responsibility = 1 for any seen code.

**Score:**

    W[u,w]  = Rnode[u,w] · logratio[u,w]                  # routing-weighted evidence, [n_nodes x V]
    s(i,u)  = Σ_w cnt(i,w) · W[u,w]  =  cnt(bow) @ Wᵀ     # same matmul shape as plain LR

`count_mode` (raw|log1p) and `length_normalize` apply to cnt(bow) exactly as in
`lr_placement_scores`.

## Components

### Engine — `spark_vi/models/topic/dag_placement.py` (id-agnostic)

- `_routing_rows(lam, lay, *, epsilon=1e-9) -> np.ndarray`  [n_nodes x V]
  Pure. Builds Rnode as above. Unit-testable in isolation.

- `explain_away_placement_scores(bow, lam, lay, *, alpha, background=None,
  epsilon=1e-9, count_mode="raw", length_normalize=False) -> np.ndarray`
  [n_docs x n_nodes], columns in lay.nodes order. Mirrors
  `lr_placement_scores`: bg = `_lr_base_rate`; logratio = `_lr_logratio_rows`;
  W = `_routing_rows` ⊙ logratio; scores = cnt(X) @ Wᵀ; optional length_normalize.

- `explain_away_decompose(bow_row, lam, lay, u, *, alpha, background,
  epsilon=1e-9, count_mode="raw") -> list[(w, count, r_u_w, contribution)]`
  Like `lr_decompose` but each row also carries r(u|w) (the routing weight), and
  contribution = cnt · r(u|w) · logratio[u,w]. Σ contribution == the score. Sorted
  by |contribution| desc. Lets the viewer show WHERE each code routed.

### Readout — `analysis/cloud/lr_readout.py`

- `detection_report`: after the plain-LR block, compute and print an EXPLAIN-AWAY
  block (ROC-AUC, PR-AUC, precision@{80,90,95}% sens) using
  `explain_away_placement_scores(..., alpha=inf, count_mode, length_normalize)`
  max-over-nodes as the case score, via the same `_detection_metrics`. One extra
  labeled block; plain LR unchanged.

- `render_decompose_rows`: extend to render the optional r(u|w) column when
  present (a 4-tuple), else the existing 3-tuple. Keep concept-name rendering.

- Error-class viewer (`write_case_viewer_by_class` / `_render_case`): add a
  `score_mode` ("lr" | "explain_away", default "lr"). When "explain_away", the
  per-doc ranking + classification use `explain_away_placement_scores` and the
  decompose uses `explain_away_decompose` (showing r(u|w)); the plain-LR max score
  is printed alongside per patient for contrast. New flag `--viewer-score-mode`.

## Validation

**Primary unit test (pure, synthetic)** in `test_dag_placement.py`: construct a
tiny λ where node u's topic block strongly emits a distinctive code d and ~0 for
generic codes, and a background topic strongly emits generic codes g1..gm. A doc
= {d:1, g1..gm}. Assert:
- explain_away score(u) >= plain LR score(u) for the same doc (the comorbid
  negatives are suppressed, not counted against u);
- in `explain_away_decompose(doc, u)`, r(u|gi) ~ 0 and contribution(gi) ~ 0 for
  each generic code, while r(u|d) ~ 1 and contribution(d) > 0;
- `_routing_rows` responsibilities over all blocks (background + nodes) sum to 1
  for each seen code (a routing-conservation check).

**ELBO/idempotence guards:** explain_away score with a single background-only
corpus and no distinctive codes -> ~0 for every node (no spurious signal).

**Cluster (post-hoc, no re-fit):** run `make lr-readout ID=67` and read the new
explain-away detection block vs plain LR (ROC/PR-AUC/precision@sens), and the
error-class viewer under `--viewer-score-mode explain_away` to confirm the FN
comorbidity-drag cases recover and inspect whether FPs shrink.

## Interactions & risks

- **FP fix is capacity-bounded:** explain-away only removes a comorbid code's
  spurious support if a BACKGROUND topic out-competes the node for it. With
  n_bg=40 this may be partial; a follow-on n_bg sweep is the lever if FPs persist.
  This is a known limit, not a bug -- state it in the insight.
- **Information-limited errors are untouched:** few-code patients and
  non-distinctive-signature patients (e.g. a Sarcoidosis patient whose only code
  is Rheumatoid arthritis) are not fixed by routing -- the evidence itself is
  weak. Explain-away targets ONLY the comorbidity-drag class.
- **Mass-starved node topics** have small Σλ, so their per-topic P(w|topic) can be
  noisy; the uniform-prior responsibility partly mitigates (a node only wins codes
  it emits distinctively). No special handling in v1; observe in the readout.
- **α=inf routing:** routing is independent of α (uses raw λ), so the α->∞ limit
  applies only to the evidence term, exactly as for plain LR.

## Out of scope

- Hard/argmax routing and a soft<->hard temperature knob (soft only in v1).
- Changing the α sweep / AUC table (stays plain LR).
- A background-conditioned base rate (bg(w) stays the flat corpus rate; only the
  routing, not the evidence denominator, becomes background-aware in v1).
- Any engine re-fit or new experiment.

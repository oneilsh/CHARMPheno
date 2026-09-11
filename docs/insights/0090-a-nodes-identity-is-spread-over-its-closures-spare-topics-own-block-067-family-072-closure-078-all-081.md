# 0090 — A node's case-finding identity is spread over its closure's SPARE topics, not held in its own block: own+bg 0.67 → +siblings 0.72 → +ancestors 0.78 → everything 0.81. HPO profile tokens are PRESENT in 76% of positives but not SPECIFIC (ancestors own them), the learned α was on all along and did not prevent capture, and the residual a node does keep is its clinical or demographic context

**Date:** 2026-09-11
**Topic:** gated-pc, decoder, ablation, deflation, ancestor-capture, tpn, profile-eta, HPOA, case-finding
**Status:** Confirmed on exp 0116 (CV branch, tpn=5, profile-eta S=1.0; `optimize_doc_concentration: true` in front matter but INERT on the PC path — α fixed at 0.5, see point 4), readout at ridge 100 with `--readout-feature-mask` (insight 0089's instrument), `--profile-support`, and the stage-2 probe re-run. Pooled figures and counts-of-nodes only.

## Observation

**1. The feature ablation.** The per-node readout head restricted to a per-node topic mask, same θ, same split, ridge 100:

| head may load on | topics per node | macro AUC | lift@1% |
|---|--:|--:|--:|
| background only | 8 | 0.6002 | 2.1 |
| own block + background | 13 | 0.6688 (0.6740 at full K, so not truncation) | 6.1 |
| own + siblings + background (`family`) | tens | 0.7192 | |
| own + siblings + ancestors + background (`family-closure`) | tens to ~100 | 0.7768 | |
| everything except own block | 1,493 | 0.7964 | 8.2 |
| everything (record) | 1,498 | 0.8098 | 9.7 |

The own block is real but small (+0.07 over background) and almost entirely redundant
(dropping it costs 0.013). Siblings add +0.05, ancestors a further +0.06, and the ~1,400
unrelated topics the last +0.03. **No compact structural decoder recovers the record**;
the closest, the closure plus siblings, sits 0.033 below it. Sibling contrast alone is
not the mechanism (0.72). Ancestor blocks matter as much as siblings for telling a node
from its siblings — which only makes sense if the ancestors' five topics each split the
parent cohort into sub-presentations that line up with its children. **Deflation put the
child's identity into the parent's spare topics.** (Insight 0087's `self-w ≈ 0` is this,
mechanically: the head reads a node out of the closure's spare capacity.)

**2. The profile tokens are there; they are not the node's.** Stage-2 probe, per credited
node with ≥100 positives: a median **76%** of positive pre-index documents carry ≥1 profile
token (14/15 nodes ≥50%). Yet the boosted topics hold λ mass 64 against a prior floor of
61, and 29/36 credited nodes have **no fed data-only sibling topic at all** — after
ancestors and background take their share there is no node-specific residual for any word
prior to attach to. The HPO vocabulary for these nodes IS the ancestors' vocabulary; a
one-document-strength prior on a child block cannot win words from a block fit on
thousands of documents that contain them. Presence ≠ specificity.

**3. The residual a node does keep is context, not phenotype.** For the 7/36 nodes with
a fed data-only topic (2× the boosted topic's evidence, profile overlap 0.07 — ten times
the flat baseline, still small), the named words are of two kinds: the disease's own code
plus its clinical context (central retinal vein occlusion → the CRVO code, retinal edema,
ocular hypertension, hypertension, diabetes; pulmonary arterial hypertension → the PAH
code, cor pulmonale, VSD, dyspnea, ILD, and tadalafil / treprostinil in the drug domain),
or the cohort's demographic signature (preeclampsia and peripartum cardiomyopathy →
pregnancy; varicose disease → osteoporosis, vitamin D/B12 deficiency, atrophic vaginitis).
Both separate the node from its siblings; neither is what HPO describes. The IDF weighting
in the prior made it worse: PAH's boosted topic is juvenile rheumatoid arthritis, leukemias
and autism — the rare syndromic end of the profile, least likely in a pre-index chart.

**4. CORRECTION (2026-09-11): the learned α was NEVER on in this arc — the flag was
inert.** Every 0113+ front matter says `optimize_doc_concentration: true`, and the driver
passes it to the estimator; but the Gated-PC estimator builds its gated engine and INJECTS
it into `OnlinePCLDA`, which treats an injected engine's LDA kwargs as the engine's own —
`optimize_alpha` never reached `GatedOnlineLDA`, and no frontier histogram was ever
computed on that path (`mllib/topic/pc.py`, engine construction vs `OnlinePCLDA.__init__`).
**α was fixed at `doc_concentration` = 0.5 on every gated PC fit, 0113–0120 included.**
The per-node tied empirical-Bayes α (insight 0059) is therefore UNTESTED at this scale on
this path, not null. Wired properly in the 0121 build (`set_alpha_policy` on the engine,
histogram from the document RDD); the front-matter flag now does what it says, which
means every fit after this date learns α unless it turns the flag off — read
comparisons across that date with this in mind.

## Why it matters

- **The readout's diffuse weights (0089) are not smearing; they are where the identity
  is.** Any compact decoder at whole-Mondo scale must be closure ∪ siblings, and will pay
  ~0.03 for it on this fit. That is the tractable design, and it is honest about what the
  gated model built.
- **The competition, not the budget, is the lever to test first.** Spare topics on
  ancestors are where children's words land, but reducing tpn only relocates the same
  competition (the user's read; agreed). The untested lever is the PRIOR ASYMMETRY that
  seeds capture: an ancestor block is visible to every document under it, a leaf's to few,
  so a uniform α hands the ancestor N_anc·α of prior mass and the leaf N_leaf·α. Exp 0121
  inverts that with a derived, knob-free init (equal TOTAL prior pseudo-count per block,
  α ∝ 1/N_docs_seeing_block) and then lets the empirical-Bayes α run; exp 0122 is the
  uniform-init control that isolates the init from the optimizer (which, per point 4, has
  never actually run on this path).
- **Profile-eta closes with a sharper reason than 0084/0085 gave.** Not "the prior can't
  move θ" in the abstract: the prior's words are present but owned by the ancestors, and
  the residual the child can own is context. A phenotype prior aimed at own blocks in this
  architecture has nothing to hold. Legibility should be read off outputs as log-lift
  against the PARENT cohort (SAGE's metric), which is also what keeps a "diabetes because
  they all have diabetes" topic from reading as a phenotype.
- **The structural fix is the cascade** (`docs/references.md`, SAGE regime b:
  β_node ∝ exp(m + Σ_anc dev + dev_node)): a child is its parent plus a deviation, so
  ancestor capture is impossible by construction and identity IS the deviation — which is
  exactly the residual measured here. The PG-STM engine that implemented it was excised in
  `ddcf52d` and is recoverable. Hold it until 0121 says whether tpn alone moves identity.

## Reproduce

```bash
make -C analysis/cloud gated-pc-readout ID=116 GPR_ARGS="--readout-mode distributed --readout-l2 100 --readout-feature-mask family-closure"   # any mode
make -C analysis/cloud inspect-topics ID=116 CREDITED=1 RESOLVE_NAMES=1 INSPECT_ARGS="--profile-support"
```

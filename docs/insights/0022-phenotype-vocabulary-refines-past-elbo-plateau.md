# 0022 — Topic-word concentration on small-α tail topics continues refining well past mass-distribution convergence
**Date:** 2026-05-18
**Topic:** lda | svi | diagnostics

The ELBO trace stops moving meaningfully somewhere around iter 40–50
on a small-cohort SVI LDA run, and the topic mass distribution
(E[β], α) settles around the same time. A naive read says "converged,
stop." But the **per-topic vocabulary concentration on small-mass
tail topics keeps refining for at least another 50–150 iterations.**

On the dementia first-year cohort (~9k docs, K=40, τ_0=64, κ=0.7),
ELBO at iter 50 was already oscillating in a ~4% band around its
asymptote, and the two-anchor mass concentration ([0021](0021-cohort-corpora-two-anchor-mass-concentration.md))
was within 1–2 pp of its final value. Yet topic-word distributions
at iter 200 were qualitatively different from iter 100 for many
phenotype topics:

| topic | iter 100 top-N | iter 200 top-N |
|---|---|---|
| t0 | ITP / nummular eczema / lipoatrophy / ILD / pancreas cancer (mixed) | **Breast cancer** (primary malignant neoplasm of breast 0.24, fibrocystic dz, ER+ tumor, lymphedema) |
| t9 | discharge from nipple / galactorrhea / breast finding | **Prolactinoma** (breast symptoms + pituitary adenoma 0.06 + galactorrhea + retraction of nipple) |
| t7 | autism + claudication + vocal cord paralysis (incoherent) | **PVD/AAA cluster** (peripheral vascular disease + atherosclerosis of extremity arteries + AAA + claudication). Autism gone. |
| t34 | ADHD + postconcussion + TBI | sharper postconcussion 0.136 + PTSD added |
| t16 (LBD) | RBD 0.09 + diffuse Lewy body | RBD 0.138, neurocognitive disorder, impaired cognition, coordination problem — full LBD prodromal-to-clinical picture |

The shift isn't subtle: t0 went from incoherent to a textbook
clinical entity. Same checkpoint family, same hyperparameters, just
+100 iters past the apparent ELBO plateau.

## Why this happens

ELBO is dominated by per-doc topic-loading entropy and topic-word
marginal coherence on the **mass-bearing topics**. Once the anchors
absorb ~85–95% of mass (typical for cohort corpora), they pin down
the ELBO contribution. The remaining ~5–15% of mass distributed
across the tail of small-α topics moves the ELBO trivially, even
when individual tail topics undergo substantial per-word
re-concentration. The optimizer is correctly minimizing ELBO; the
trace just isn't sensitive to the refinement that matters most for
phenotype interpretation.

A diagnostic that *does* track this refinement: per-topic
**peak-word probability** (max_w β[k,w]) on small-mass topics.
At iter 50 the dementia run's small-mass topics had peaks in the
0.02–0.05 range; at iter 200 the same topics had peaks in the
0.05–0.27 range (3–10× sharper). Watching that distribution
plateau is a better stop criterion than ELBO alone.

## Relationship to [0004](0004-lda-asymmetric-alpha-settles-late.md)

[0004](0004-lda-asymmetric-alpha-settles-late.md) is about
asymmetric α converging later than topic-word distributions on
large/full-corpus runs. This insight is **the opposite case on a
cohort**: α converges relatively early (anchor amplification finishes
around iter 50–100), but per-topic word concentration on the small-α
tail keeps going. Both findings imply that "stop when ELBO plateaus"
is the wrong rule; the right rule depends on which downstream use
case you care about. For dashboards/labeling that need crisp
phenotype labels, run past the ELBO plateau until the **peak-word
probability distribution** for small-α topics stops moving.

## Implications

- **Don't stop at ELBO plateau for small-cohort LDA fits intended
  for downstream phenotype labeling.** Plan for ~2× the iter count
  that ELBO seems to demand.
- **Add a `peak_β` trace to runner diagnostics.** Watching the
  distribution of `max_w β[k,w]` over the small-α tier converge is
  the operationally-useful stop signal.
- **Mid-iter snapshots are misleading for K-sizing decisions.** In
  this session I recommended K=20 at iter 10 (looked over-sized),
  then reversed at iter 47 (rare phenotypes had begun to emerge),
  then realized at iter 200 that the same K=40 had even more
  structure than visible at iter 100. The mass distribution is
  stable early; phenotype identity is not.
- **The cancer and gen-pop fits may benefit from longer training
  too.** Both were trained to ~50 iters per their Makefile targets.
  If they show late-refinement signal similar to dementia, a fresh
  100-iter continuation could improve their phenotype labels
  without retraining from scratch.

**Setting context:** Online VI LDA, K=40, asymmetric-α optimization
on, condition_era doc-unit, `first_dementia_year` cohort (~9k docs
post-cohort filter), person_mod=1, vocab_size=10000, min_df=10,
batch_fraction=0.2, τ_0=64, κ=0.7. Comparison made between
iter 50 / iter 100 / iter 200 snapshots from the same training
trajectory (resumed runs). NPMI eval against full-corpus reference.

# 0038 — Faithful in-memory PC reference lives in `analysis/pc/`, separate from the VI core

**Status:** Accepted

> **Numbering note.** This ADR was authored on the `claude/faithful-flat-pc`
> branch, in parallel with other branches that also number ADRs/insights/experiments
> from the same counter. At authoring time `main` was at ADR 0037, but a sibling
> branch had already claimed 0038–0039 with different slugs. If that branch merges
> first, renumber this file (`git mv` + retitle) to the next free ADR number — the
> slug (`faithful-in-memory-pc-reference-in-analysis-pc`) disambiguates it. The
> number is cosmetic; the decision is the content.

## Context

We needed a *trusted* implementation of the Prediction-Constrained (PC) topic
model of Hughes, Hope, Weiner, McCoy, Perlis, Sudderth & Doshi-Velez (2017/2018)
— the supervised-LDA variant that constrains the topics to stay predictive of a
label — before applying it to anything real (the Phase C antidepressant-stability
study on All-of-Us OMOP). "Trusted" here means *provably faithful to the
reference method*, not merely "a supervised topic model that runs."

The existing modeling core in this repo is Spark/`mllib`-shaped and built around
variational inference (VI) with a mean-field readout (see ADRs 0022–0037). That
core is the production target, but it is the wrong place to *first* establish PC
faithfulness: its VI approximations are exactly the thing we would need to hold
fixed and above suspicion while checking the objective, and its Spark surface
makes a tight grad-check / oracle loop slow and heavy.

Three sub-questions had to be decided:

1. **Where does the reference live**, given the VI/Spark core already exists?
2. **Which π (per-document topic proportions) inference** does the reference use —
   the *faithful* per-document MAP that Hughes' objective actually optimizes, or
   the cheaper free-variational-π shortcut that the eventual VI port will want?
3. **What makes it trusted** — what is the acceptance gate?

## Decision

**1. A separate, in-memory reference in `analysis/pc/`.** The faithful PC model is
implemented from scratch in `analysis/pc/` using **numpy / scipy / autograd only**
— deliberately *not* inside the Spark/VI core. `autograd` is a **reference-only
dependency**: it exists to grad-check the hand-derived objective and must not leak
into the VI core (isolated to the `.venv-pc` scratch venv; see `.gitignore`). The
reference is the source of truth for *what the PC objective is*; the VI core stays
the source of truth for *how we scale it*.

**2. Faithful per-document π-MAP inference, with the free-π variant parked.** The
reference infers each document's π by MAP under the Dirichlet prior (the quantity
Hughes' PC loss is actually a function of), rather than treating π as a free
variational parameter. This is what lets the reference reproduce the paper's loss
at the paper's own parameters (see gate below). The free-variational-π variant —
which the *future VI-native port* will build on — is preserved as a parked fork,
not deleted: the two disagree in ways that matter, and the VI port needs the
faithful reference to validate against, not a second approximation.

**3. Trust is gated on a reference oracle, not on internal self-consistency.**
`analysis/pc/` is considered faithful because, evaluated at Hughes' *own published*
optima on the vendored `toy_bars_3x3` fixture, our objective returns his reported
trade-off (near-generative `loss_x` with perfect predictive `loss_y`/AUC) **and**
correctly separates his `good_loss_pc` regime from his `good_loss_y` regime (which
wrecks the topics). The `toy_bars` fixture is vendored under `analysis/pc/tests/`
with MIT attribution so the oracle runs anywhere, not only in the container that
first built it. Two lower gates back it up: a ~1e-9 finite-difference grad-check of
the objective, and an 8/8-seed synthetic known-signal test where faithful PC beats
the unsupervised two-stage baseline.

**Eval layer may use scikit-learn; the model core may not.** `analysis/pc/evaluate.py`
(the id-agnostic PC-vs-baselines harness the Phase C driver calls) is allowed
`scikit-learn` for the logistic-regression baselines and AUC/AP metrics — it is
measurement, not the constrained objective. The model core stays numpy/scipy/autograd.

## Consequences

- The reference is fast to iterate on and independently checkable, at the cost of
  not being scalable — by design. It is the correctness anchor, not the deployment
  path.
- A future **VI-native PC port** into the Spark core is a known follow-on. It must
  validate against this reference (both the oracle and the synthetic gate), and it
  inherits the parked free-π variant as its starting point.
- Phase C plugs real AoU data into `evaluate_pc_vs_baselines(...)` without touching
  the model core — the eval harness is the stable API boundary.
- The `analysis/` tree now hosts a from-scratch model, not just analysis scripts;
  contributors should not assume everything under `analysis/` is throwaway.

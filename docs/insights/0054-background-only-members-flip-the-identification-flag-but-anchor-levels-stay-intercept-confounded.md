# Insight 0054 — Background-only members flip the anchor identification flag, but anchor LEVELS stay intercept-confounded on foreground sticks; only branching-evidence increments recover, and no-branching nodes are collapsible

**Date:** 2026-07-13
**Branch:** pg-stm
**Topic:** pg-stm | dag-offsets | identifiability | background-only-members | ontology-structure
**Status:** Observed
**Relates to:** the DAG background-only member support step (Task 4); insight 0050 (anchor offsets un-identified under a partitioning gate); insight 0052 (identification flag vs point-estimate recovery — two-engine); the deferred ontology node-collapsing design item.

**Context:** The gate-change step added background-only members (documents with no foreground
group, routed to a flat non-gated E-step) specifically to break the anchor-partition dummy trap
of insight 0050 — the hope being that, with some documents attesting only the root, the anchor
levels become identified and the leaf path-sums recover. Task 4 validated this with a two-anchor
DAG: anchor A with anchor-level docs + a subtype (A1), anchor B with ONLY a subtype (B1, no
anchor-level docs), plus 600 background-only members. The first path-sum test failed with
NEGATIVE correlations (r3=-0.15, r4=-0.35).

**Finding — three distinct effects, none a bug:**

1. **Recovery must be measured on FOREGROUND (node-specific) sticks, not all active sticks.** A
   node's offset on the SHARED background sticks is weakly identified (the shared block plus the
   lightly-penalized global intercept absorb it). Correlating a node's recovered vs planted offset
   over all active sticks (5 shared background + 4 group foreground, in the test config) lets the
   5 noisy background dims dominate the 4 clean foreground dims, so the aggregate correlation is
   noisy and sign-flippable. Measured split for the identified subtype A1: r(all-active)=+0.20 but
   r(foreground-only)=+0.89. The reparam step's earlier "r=0.76" on all-active dims was seed luck
   on this same noisy aggregate. The foreground-only correlation is the honest measure and is
   robust: subtype A1 foreground increment r ranged +0.54 to +0.99 across four configs/seeds.

2. **Background-only members flip the identification FLAG but do NOT restore anchor-LEVEL
   point recovery.** On the degenerate corpus (anchor B has no direct docs), the anchor A
   increment's identified flag goes False -> True when background-only members are added
   (ident=[_,0,...] -> [_,1,...], deterministic). But the anchor LEVEL's point estimate on the
   group FOREGROUND sticks stays confounded with the global covariate intercept Gamma: every
   group-A document has intercept=1 AND closure-indicator z_A=1 (collinear within the group), and
   background-only documents never activate the group-A foreground sticks (their allowed set is
   background only), so they cannot break that foreground confound. With gamma_ridge (1e-6) much
   lighter than the offset penalty (lam_base 1e-3), the intercept absorbs the group foreground
   offset and the anchor's B[u] shrinks toward zero. Background-only members ground the SHARED
   background block, not the group-specific foreground levels. This is a concrete instance of
   insight 0052: the identification flag (a variance-reduction signal) can flip while the
   mean-field point estimate remains unrecovered.

3. **A no-direct-docs single-child anchor has a residual collinearity that background grounding
   cannot break — such a node is collapsible.** When anchor B has no anchor-level docs, every
   group-B document attests {B, B1} together, so B[B] and B[B1] are collinear (a second dummy
   trap): no document has z_B=1, z_B1=0 to separate them. Background-only members do not help
   (they carry neither node). Give anchor B direct docs (branching evidence) and its subtype
   increment recovers (foreground r -0.67 -> +0.93). The general rule: a node's increment is
   identified only with BRANCHING evidence — either documents at the node alone, or a sibling
   subtype. A node with a single child and no own-level evidence carries no separable signal and
   should be COLLAPSED into its child rather than expecting the model to recover an
   un-identifiable split. Open wrinkle (for the deferred node-collapsing design): a single-child
   node with MULTIPLE parents cannot be trivially collapsed, because its offset is shared across
   more than one parent path.

**Consequences:**
- The honest, recoverable DAG-offset deliverable is the **subtype foreground increment given
  branching evidence**, measured on foreground sticks — not anchor levels, not path-sums over
  shared-background dims. This tightens insight 0050 (only identified increments carry signal) with
  the foreground-vs-shared-background distinction and the intercept-absorption mechanism.
- Background-only members' concrete contribution is (a) breaking the root-vs-anchor
  identification-FLAG trap and (b) grounding the shared background block — NOT restoring
  point-estimate recovery of group-foreground anchor levels (that needs a heavier intercept
  penalty or a debiased read-out engine, insight 0052) and NOT identifying no-branching nodes.
- Motivates an ontology preprocessing step: **collapse single-child nodes that have no own-level
  evidence** (tree case), leaving the multi-parent case as an open design question. Deferred.

**Does not claim:** anything about real-data recovery or transfer (synthetic proves
math-correctness only); nor that anchor levels are un-recoverable in principle (a heavier
intercept penalty or a debiased/Gibbs read-out per insight 0052 may recover them) — only that
background-only members alone, under mean-field VI with a light intercept ridge, do not restore
their point estimate.

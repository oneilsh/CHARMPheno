# 0104 — A leaves-first, seeded gate schedule raises block identity at every depth

**Setting:** exps 0155 / 0156, the cardiovascular Mondo branch (294 nodes, heights 0..7),
tpn 1, 200 background topics, spectral init, gate-ever, 60 iterations. Single-round gate
against `gate_warm: 3` with the subtree-mean seed at opening (ADR 0049).

**Observation.** The profile census with the own-code rule (a block whose top 10
condition words hold none of its node's attesting codes is a stratum):

| | single round | leaves-first + seed |
|---|---|---|
| signature blocks | 206 / 294 (70%) | 245 / 294 (83%) |
| own-code-decidable nodes with an own-code identity | 86 / 172 (50%) | 126 / 172 (73%) |
| stratum share, depth 2 / 3 / 4 / 5 / 6 / 7 / 8 | 75 / 36 / 34 / 23 / 26 / 29 / 25% | 25 / 21 / 14 / 14 / 16 / 21 / 0% |

Stratum share falls at every depth below the root (the root is a stratum in both).
Evidence moves down the tree: median evidence at depth 1–3 falls 3–10×, at depth 7–8 it
rises. On new-diagnosis prediction the stacked product gains a small, robust +0.007
(172 of 237 nodes, sign test p ≈ 1e-12); the per-node flat heads are unchanged. The
cardiomyopathy subtree (exps 0153 / 0154, 17 nodes) gave the same direction.

**Reading.** In a single-round nested gate, a parent and its children compete for the
children's documents from iteration 0, and the parent, admitted for every child's
document, often wins a child's words or a generic stratum. Opening by height lets each
child claim its own codes first, and the seed at opening keeps a late parent competitive
instead of dead on arrival. The schedule changes what the blocks HOLD, which the stacked
product reads and a node's own head barely does.

**Known costs.** During the leaf rounds, content that belongs to no single disease (the
inpatient-encounter stratum: pain, nausea, vomiting; sodium chloride, fentanyl) has only
leaves and the background to go to, and a few leaves keep it (aortic valve stenosis,
abdominal aortic aneurysm in 0156). Intrinsic cardiomyopathy took HCM's words in 0154: the
subtree-mean seed weights children by size.

**Follow-ups (exps 0157 / 0158).** A 1,200-topic background does NOT fix the leaf sink
(census unchanged) and costs about 0.03 new-diagnosis AUC (0157 better on 6 of 237 nodes):
keep 200. An HPO profile seed (35 of 294 nodes profiled) gives +0.003 stacked (147 of 237)
and slightly lower own-code identity (73% → 70%).

**Balanced seed (exp 0160).** Seeding a late block with each child's subtree share
weighted equally, plus the node's own coded documents, with descendants' own codes
removed, lifts own-code identity further, 73% → 77% (signature 83% → 86%; stratum share
at depths 3–5 down 3–4 points). New-diagnosis AUC (−0.0015 stacked) and NPMI are
unchanged. Best configuration so far: leaves-first + balanced seed + 200 background
topics.

**Coherence (exp 0159).** Gated NPMI (each block on the documents its gate admits) is
flat across the schedule, with or without the blocks' own codes in the top words; only
the parents' drug coherence rises. The schedule changes which words lead a block, not
how well the rest co-occur.

**Whole Mondo (exp 0161).** The central configuration (leaves-first at 2 iterations per
level, balanced seed, 200 background topics) on all 2,694 nodes, against 0152: own-code
identity 54% → 78%, stratum share down at every depth from 2 to 13, stacked
new-diagnosis AUC +0.016 on 827 ribbon nodes (707 better), drug NPMI +0.019, framings
unchanged within noise. The parent-holds-child failures of 0152 (diabetes, heart failure,
cardiomyopathy, IBD, breast carcinoma) are gone in the named-node read.

**Leaf seed, random init (exp 0162).** Starting every leaf block from its own documents'
word counts, with no spectral init, raises own-code identity on the cardiovascular branch
77% → 90% (signature 86% → 92%; stratum share at depths 6–7 from 15–21% to 3–7%), leaves
new-diagnosis AUC unchanged (stacked −0.0001) and condition NPMI flat to slightly up. It
also drops the spectral seeding, about 6.5 h of a whole-Mondo fit. Central configuration:
leaves-first + balanced seed + leaf seed + random init + 200 background topics.

**Open.** Whole Mondo with the leaf seed; a shorter warm-up against the leaf sink; the child-balanced, own-code-excluded
seed against the dominant-child start; a second seed of the 0155 / 0156 pair.

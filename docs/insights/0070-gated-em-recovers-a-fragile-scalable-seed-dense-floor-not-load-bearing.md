# 0070 — Gated EM recovers a fragile scalable random-projection seed; the dense per-domain floor is not load-bearing on a well-specified plant

**Date:** 2026-07-26
**Topic:** svi | lda | diagnostics
**Status:** Confirmed
**Context:** SP3a of the multi-domain gated DAG LDA arc (branch `multidomain-spectral-init`), building the production-path recovery gate (`test_scalable_init_recovers_every_identifying_signal`). Supersedes the premature "immunity" reading of arc SP3 blocker 1.

## Finding

The MLlib shim routes multi-domain spectral init through the **scalable** random-projection path (`scalable_block_aligned_lambda`, ADR 0032) above `spectralMaxVocab`, and a concatenated two-domain vocabulary crosses that threshold easily. That path never received the per-domain candidate floor the dense path has, which raised the question of whether it needs one.

Across **8 random-projection seeds** on the well-specified `b_only`-node two-domain plant (`bg_frac=0.2`, `ancestor_signature_decay=0.5`), two things are both true:

1. **The scalable seed is genuinely fragile.** Measured pre-EM, an exclusive node's per-domain signal reaches **0.000** at initialization on several draws (e.g. node 3's domain-1 signal is dead at seeds 1 and 2; node 2's at seeds 3 and 7). The random projection is a real lucky-draw risk — a single-seed test would have passed or failed on luck.

2. **The gated EM recovers every one.** Post-EM (50 full-batch iterations), every node's *identifying* per-domain signal is ≥ 0.55 on all 8 seeds, well above the dead-topic floor, and every cell clears 1.5× uniform. The initializer's fragility is fully absorbed by the E/M refinement.

So on a well-specified plant, **recovery is carried by the EM, not by the seed's quality and not by a candidate floor.** The scalable path is *adequate* — a multi-domain fit through it recovers every node — without a floor analogue. This is not because the scalable init is a good seed (it isn't, reliably); it is because the gated E/M is robust enough to climb out of a bad one.

This directly connects to [0067](0067-background-starved-plants-frame-a-correct-spectral-seed-as-broken.md): the dense per-domain floor's headline value ("recovery 0.005 vs 0.675 at random_seed=0", insight 0066) was measured on a **background-starved** plant, and 0067 showed that degeneracy — not the floor's absence — was what killed recovery. On a plant with a real background pool, neither the floor nor a lucky seed is load-bearing.

## Why it matters

1. **The "immunity" framing was wrong, and for an instructive reason.** A 3-seed comparison of the scalable seed against the dense+floor seed passed and was read as "the scalable path is immune / needs no floor, proven." That was a lucky draw: the loop held the projection seed fixed and varied only the EM seed (which washes out over 50 iterations), so it never sampled the risk it existed to rule out. Sampling the projection seed 8 ways refutes the parity claim (the scalable seed is *worse*, sometimes dead) while confirming the thing that actually matters (EM recovers anyway). **"Adequate because EM is robust" is a different, weaker, and true claim; "immune because the seed is as good" is false.**

2. **A recovery threshold is only meaningful on an identifying cell.** The same 8-seed comparison threw a false failure at one cell — the `b_only` node's domain-0 block, whose planted support is *identical to its parent's* by construction, so mass there identifies neither node and a threshold on it scores a coin-flip tie. This is the third time this arc has been misled by a plant/metric interaction (cf. 0067 background starvation, 0068 even-`closure` labelling). The durable fix is a discipline: **thresholds go only on cells whose planted support is exclusive to their node** — derivable from the supports (`_identifying_cells`), which reproduces from first principles the hand-worked "symmetric skip" a prior recovery test needed.

3. **Production consequence for the cloud driver.** The shim fits with `seed=(seed or 0)`, so a multi-domain fit with no seed set always uses projection seed 0's sketch — a fixed, unvalidated draw. Because different draws give materially different *seed* quality (even though EM recovers on this synthetic plant), a real corpus with a harder identifiability structure could expose a draw the EM does not fully rescue. SP3b/SP4 should treat the projection seed as a knob to pin and validate, not an invisible default.

**Bottom line:** the production (scalable) multi-domain init is adequate because the gated EM recovers a fragile seed, not because the seed is good — so recovery gates belong on the fit, thresholded only on identifying cells, and the dense per-domain floor is a degenerate-plant artifact rather than a load-bearing component on a well-specified corpus.

# 0106 — η = 1/V_m per domain is a weaker prior than 1/K here, and starves ~10% more blocks
**Date:** 2026-11-24
**Topic:** svi
**Status:** Supported (exp 0176)

Replacing the DAG-gated fit's η = 1/K (MLlib's default) with η_m = 1/V_m per domain
left NPMI unchanged (paired medians within 0.007) and the census about the same, but
raised the blocks the data does not sustain (supportFraction > 0.9) from 176 to 195 of
3,229, all in nodes with 100-1,000 supporting documents.

The two choices differ in strength as well as form. η_m · V_m is a topic row's total
prior mass: 1 under 1/V_m, V_m / K under 1/K (about 1.5 for 5,000-term domains at
K ≈ 3,250). A weaker prior makes a weakly fed row sharper, so it claims fewer of its
documents' words and decays toward the prior faster: a rich-get-richer effect that
falls on the marginal blocks. The form η ∝ 1/V_m is the principled part (no coupling
to K, which moves with the kept node set; per-domain scale); the strength c in
η_m = c / V_m is a hyperparameter with no derivation, like α.

Where 1/K comes from: Hoffman, Blei & Bach (NIPS 2010, §4) fixed "α and η ... at 0.01
and the number of topics K = 100", with no justification; their reference demo
(`onlinewikipedia.py`) writes it as `1./K`, and MLlib's online optimizer inherited that
form. The paper's own principled option is to learn η (its eq. 9).

First recorded as tentative (the flat count's run-to-run spread was unmeasured).
Exp 0176 repeated 0175 with seed 43: 204 flat blocks (195 with seed 42), so the increase
over 1/K's 176 holds across seeds, and the 1/V_m spread (9) is smaller than the gap
(19-28). Still open: only one 1/K seed, and whether a larger c (a sweep over c in
0.5-4) recovers 1/K's count while keeping the per-domain form.

**Setting context:** exp 0175 vs 0174, v9 CDR, population_mondo_all, native Mondo
(minDocsPerNode 100, 3,229 kept nodes), three domains (conditions, drugs,
measurements), gate_warm 4, 150 iterations, α = 0.5 fixed, seed 42, flatness by the
dominant-domain rule; exp 0176 the same with seed 43 (2 executors).

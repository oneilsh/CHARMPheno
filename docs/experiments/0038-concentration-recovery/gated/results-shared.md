# Gated concentration-recovery experiment (CR-4) results -- shared vocabulary

Seed: 0. beta_mode=shared (mean topic-support Jaccard=0.593; 0=disjoint vocab, ->1=full overlap). Config: K=4 (bg_k=2, groups=['A', 'B'], fg_per_group=1), V=100, D=300, doc_len=55, holdout_frac=0.3, Sigma=identity (R = I; c IS the generative eta-variance), c_grid=[1, 2, 3, 5, 8, 12, 20].

| s | planted_top_mass | GATED_argmax_c | GATED_recovered_top_mass | GATED_abs_err | NONGATED_argmax_c | NONGATED_recovered_top_mass | NONGATED_abs_err |
|---|---|---|---|---|---|---|---|
| 1.0 | 0.6386 | 1 | 0.6109 | 0.0277 | 20 | 0.6280 | 0.0106 |
| 2.0 | 0.7190 | 2 | 0.7051 | 0.0139 | 20 | 0.7121 | 0.0069 |
| 5.0 | 0.8347 | 5 | 0.8296 | 0.0051 | 20 | 0.8338 | 0.0009 |
| 10.0 | 0.9175 | 12 | 0.9066 | 0.0109 | 20 | 0.9073 | 0.0102 |

Boundary-widened cells (argmax was not an interior peak on the base grid):
- s=1.0, gated: argmax hit a grid boundary and the grid was widened to [0.5, 1, 2, 3, 5, 8, 12, 20] (final argmax_c=1)
- s=1.0, nongated: argmax hit a grid boundary and the grid was widened to [1, 2, 3, 5, 8, 12, 20, 40] (final argmax_c=20)
- s=2.0, nongated: argmax hit a grid boundary and the grid was widened to [1, 2, 3, 5, 8, 12, 20, 40] (final argmax_c=20)
- s=5.0, nongated: argmax hit a grid boundary and the grid was widened to [1, 2, 3, 5, 8, 12, 20, 40] (final argmax_c=20)
- s=10.0, nongated: argmax hit a grid boundary and the grid was widened to [1, 2, 3, 5, 8, 12, 20, 40] (final argmax_c=20)

## Summary

[beta_mode=shared, mean topic-support Jaccard=0.593] Under GATING, held-out predictive-LL DOES recover the planted generative scale across all 4 planted scales (worst-case gated abs error 0.0277, tolerance 0.08); the gated argmax_c does NOT always land on the grid value nearest the planted s. The non-gated sweep's argmax hit a grid boundary (needing the grid widened) in 4/4 cells vs 1/4 for gated -- the non-gated held-out-LL curve is comparatively FLAT across a wide range of c, so its argmax is a much less sharply-identified optimum than the gated one, even where the two regimes' RECOVERED concentrations end up close. Comparing the SAME documents run through both regimes, gating HURTS recovery relative to non-gated on these same documents (gated=0.0144 vs non-gated=0.0072). Finally, non-gated inference does NOT meaningfully leak mass onto non-allowed topics here (median leaked mass=0.0053; eff_topics non-gated=1.590 vs gated=1.598). Sigma was fixed to identity (R = I) throughout, so this isolates GATING from correlation-structure recovery.

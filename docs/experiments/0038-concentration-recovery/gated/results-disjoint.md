# Gated concentration-recovery experiment (CR-4) results -- disjoint vocabulary

Seed: 0. beta_mode=disjoint (mean topic-support Jaccard=0.000; 0=disjoint vocab, ->1=full overlap). Config: K=4 (bg_k=2, groups=['A', 'B'], fg_per_group=1), V=100, D=300, doc_len=55, holdout_frac=0.3, Sigma=identity (R = I; c IS the generative eta-variance), c_grid=[1, 2, 3, 5, 8, 12, 20].

| s | planted_top_mass | GATED_argmax_c | GATED_recovered_top_mass | GATED_abs_err | NONGATED_argmax_c | NONGATED_recovered_top_mass | NONGATED_abs_err |
|---|---|---|---|---|---|---|---|
| 1.0 | 0.6386 | 1 | 0.6384 | 0.0002 | 160 | 0.6542 | 0.0156 |
| 2.0 | 0.7190 | 2 | 0.7084 | 0.0105 | 80 | 0.7086 | 0.0104 |
| 5.0 | 0.8347 | 5 | 0.8308 | 0.0040 | 20 | 0.8339 | 0.0008 |
| 10.0 | 0.9175 | 12 | 0.9233 | 0.0058 | 40 | 0.9255 | 0.0080 |

Boundary-widened cells (argmax was not an interior peak on the base grid):
- s=1.0, gated: argmax hit a grid boundary and the grid was widened to [0.5, 1, 2, 3, 5, 8, 12, 20] (final argmax_c=1)
- s=1.0, nongated: argmax hit a grid boundary and the grid was widened to [1, 2, 3, 5, 8, 12, 20, 40, 80, 160, 320] (final argmax_c=160)
- s=2.0, nongated: argmax hit a grid boundary and the grid was widened to [1, 2, 3, 5, 8, 12, 20, 40, 80, 160] (final argmax_c=80)
- s=5.0, nongated: argmax hit a grid boundary and the grid was widened to [1, 2, 3, 5, 8, 12, 20, 40] (final argmax_c=20)
- s=10.0, nongated: argmax hit a grid boundary and the grid was widened to [1, 2, 3, 5, 8, 12, 20, 40, 80] (final argmax_c=40)

## Summary

[beta_mode=disjoint, mean topic-support Jaccard=0.000] Under GATING, held-out predictive-LL DOES recover the planted generative scale across all 4 planted scales (worst-case gated abs error 0.0105, tolerance 0.08); the gated argmax_c does NOT always land on the grid value nearest the planted s. The non-gated sweep's argmax hit a grid boundary (needing the grid widened) in 4/4 cells vs 1/4 for gated -- the non-gated held-out-LL curve is comparatively FLAT across a wide range of c, so its argmax is a much less sharply-identified optimum than the gated one, even where the two regimes' RECOVERED concentrations end up close. Comparing the SAME documents run through both regimes, gating does NOT materially change recovery error (gated=0.0051 vs non-gated=0.0087). Finally, non-gated inference does NOT meaningfully leak mass onto non-allowed topics here (median leaked mass=0.0015; eff_topics non-gated=1.549 vs gated=1.571). Sigma was fixed to identity (R = I) throughout, so this isolates GATING from correlation-structure recovery.

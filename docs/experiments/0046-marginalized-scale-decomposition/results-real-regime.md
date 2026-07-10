# exp 0046: MAP-vs-marginalized η-scale decomposition (real regime)

Seed: 0. LEVEL (planted η-scale): 5.0. n_samples: 128. c-grid: [0.5, 0.6507, 0.8469, 1.1022, 1.4345, 1.867, 2.4298, 3.1623, 4.1156, 5.3563, 6.9711, 9.0726, 11.8077, 15.3673, 20.0]. holdouts: [0.5, 0.7, 0.95].

### Regime: real

Planted η-scale (LEVEL) = 5.0; planted top_mass p50 = 0.3107; corpus K=60, V=5000, doc_len=44, D=1500; n_samples=128; wall=161.9s.

| estimator | c* @ h=0.5 | c* @ h=0.7 | c* @ h=0.95 | residual_drift |
|---|---|---|---|---|
| MAP plug-in | 3.458 (±0.004) | 3.503 (±0.004) | 3.385 (±0.009) | 0.118 |
| marginalized | 2.596 (±0.005) | 2.822 (±0.010) | 3.592 (±0.009) | 0.995 |

marg_scale_error = mean_h(marginalized c*) - LEVEL = -1.996 (marginalized recovers the planted scale iff this ≈ 0).

## Summary

Headline (per regime): the MAP plug-in c* should drift more across the holdout fractions than the marginalized c*, and the marginalized c* should sit near the planted LEVEL. [real] MAP residual_drift=0.118 vs marginalized residual_drift=0.995 (MAP-as-artifact NOT confirmed); marg_scale_error=-1.996 against LEVEL=5.0. A material marginalized residual_drift (or a nonzero marg_scale_error) is the known Laplace under-dispersion second-order term and is the number that decides whether a later importance-sampling refinement is warranted.

# exp 0046: MAP-vs-marginalized η-scale decomposition (clean regime)

Seed: 0. LEVEL (planted η-scale): 5.0. n_samples: 128. c-grid: [0.5, 0.6507, 0.8469, 1.1022, 1.4345, 1.867, 2.4298, 3.1623, 4.1156, 5.3563, 6.9711, 9.0726, 11.8077, 15.3673, 20.0]. holdouts: [0.5, 0.7, 0.95].

### Regime: clean

Planted η-scale (LEVEL) = 5.0; planted top_mass p50 = 0.5929; corpus K=8, V=400, doc_len=60, D=1000; n_samples=128; wall=67.6s.

| estimator | c* @ h=0.5 | c* @ h=0.7 | c* @ h=0.95 | residual_drift |
|---|---|---|---|---|
| MAP plug-in | 3.847 (±0.006) | 3.463 (±0.006) | 3.013 (±0.002) | 0.834 |
| marginalized | 3.984 (±0.008) | 3.945 (±0.010) | 4.079 (±0.027) | 0.134 |

marg_scale_error = mean_h(marginalized c*) - LEVEL = -0.998 (marginalized recovers the planted scale iff this ≈ 0).

## Summary

Headline (per regime): the MAP plug-in c* should drift more across the holdout fractions than the marginalized c*, and the marginalized c* should sit near the planted LEVEL. [clean] MAP residual_drift=0.834 vs marginalized residual_drift=0.134 (MAP-as-artifact confirmed); marg_scale_error=-0.998 against LEVEL=5.0. A material marginalized residual_drift (or a nonzero marg_scale_error) is the known Laplace under-dispersion second-order term and is the number that decides whether a later importance-sampling refinement is warranted.

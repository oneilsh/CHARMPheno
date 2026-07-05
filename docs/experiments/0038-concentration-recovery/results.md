# Concentration-recovery experiment (CR-3) results

Seed: 0. Config: K=8, V=400, D=300, doc_len=60, holdout_frac=0.3.

| mechanism | level | planted_top_mass | STM_argmax_c | STM_top_mass | STM_abs_err | LDA_argmax_alpha | LDA_top_mass | LDA_abs_err | LDA_opt_top_mass | LDA_opt_err |
|---|---|---|---|---|---|---|---|---|---|---|
| logistic_normal | 1 | 0.3297 | 1 | 0.3121 | 0.0175 | 1.0 | 0.3126 | 0.0171 | 0.3000 | 0.0297 |
| logistic_normal | 3 | 0.4716 | 3 | 0.4647 | 0.0069 | 1.0 | 0.4235 | 0.0480 | 0.4711 | 0.0005 |
| logistic_normal | 5 | 0.5527 | 3 | 0.5450 | 0.0077 | 0.3 | 0.5614 | 0.0087 | 0.5636 | 0.0109 |
| logistic_normal | 9 | 0.6625 | 8 | 0.6657 | 0.0032 | 0.3 | 0.6578 | 0.0047 | 0.6756 | 0.0131 |
| dirichlet | 3.0 | 0.2361 | 1 | 0.2453 | 0.0093 | 3.0 | 0.2100 | 0.0261 | 0.1714 | 0.0647 |
| dirichlet | 1.0 | 0.3273 | 1 | 0.3027 | 0.0246 | 1.0 | 0.3015 | 0.0258 | 0.2935 | 0.0338 |
| dirichlet | 0.3 | 0.4769 | 5 | 0.4871 | 0.0101 | 0.3 | 0.4882 | 0.0113 | 0.4933 | 0.0164 |
| dirichlet | 0.1 | 0.7308 | 12 | 0.7229 | 0.0079 | 0.1 | 0.7325 | 0.0017 | 0.7367 | 0.0059 |

## Summary

Held-out predictive-LL DOES recover the planted concentration across all 8 (mechanism, level) cells for both families (worst-case abs error 0.0480, tolerance 0.08). Overall, STM has the lower mean absolute error (STM mean=0.0109 vs LDA mean=0.0179). Split by planting mechanism, logistic_normal-planted: STM err=0.0088, LDA err=0.0196; dirichlet-planted: STM err=0.0130, LDA err=0.0162 -- the 'matched prior' effect (STM wins on logistic_normal-planted data, LDA wins on dirichlet-planted data) does NOT hold in this run. Finally, LDA's own alpha-optimization tracks the held-out-LL optimum closely (mean top_mass delta +0.0022).

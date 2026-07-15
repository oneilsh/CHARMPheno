# Concentration-recovery experiment (CR-3) results

Seed: 0. Config: K=60, V=5000, D=300, doc_len=44, holdout_frac=0.3.

| mechanism | level | planted_top_mass | STM_argmax_c | STM_top_mass | STM_abs_err | LDA_argmax_alpha | LDA_top_mass | LDA_abs_err | LDA_opt_top_mass | LDA_opt_err |
|---|---|---|---|---|---|---|---|---|---|---|
| logistic_normal | 1 | 0.0976 | 1 | 0.0735 | 0.0242 | 1.0 | 0.0556 | 0.0421 | 0.0300 | 0.0676 |
| logistic_normal | 3 | 0.2106 | 2 | 0.1753 | 0.0353 | 0.3 | 0.1744 | 0.0362 | 0.0483 | 0.1623 |
| logistic_normal | 5 | 0.2988 | 3 | 0.2671 | 0.0318 | 0.3 | 0.2312 | 0.0676 | 0.3120 | 0.0132 |
| logistic_normal | 9 | 0.4300 | 5 | 0.3964 | 0.0336 | 0.1 | 0.4000 | 0.0300 | 0.4320 | 0.0020 |
| dirichlet | 3.0 | 0.0481 | 1 | 0.0580 | 0.0098 | 3.0 | 0.0300 | 0.0181 | 0.0137 | 0.0344 |
| dirichlet | 1.0 | 0.0753 | 1 | 0.0685 | 0.0068 | 1.0 | 0.0526 | 0.0227 | 0.0300 | 0.0453 |
| dirichlet | 0.3 | 0.1386 | 2 | 0.1310 | 0.0076 | 0.3 | 0.1385 | 0.0001 | 0.0300 | 0.1086 |
| dirichlet | 0.1 | 0.2636 | 5 | 0.2600 | 0.0036 | 0.3 | 0.2131 | 0.0505 | 0.2860 | 0.0224 |

## Summary

Held-out predictive-LL DOES recover the planted concentration across all 8 (mechanism, level) cells for both families (worst-case abs error 0.0676, tolerance 0.08). Overall, STM has the lower mean absolute error (STM mean=0.0191 vs LDA mean=0.0334). Split by planting mechanism, logistic_normal-planted: STM err=0.0312, LDA err=0.0440; dirichlet-planted: STM err=0.0070, LDA err=0.0228 -- the 'matched prior' effect (STM wins on logistic_normal-planted data, LDA wins on dirichlet-planted data) does NOT hold in this run. Finally, LDA's own alpha-optimization UNDER-concentrates relative to the held-out-LL optimum (mean top_mass delta -0.0142, i.e. reads cooler/more diffuse).

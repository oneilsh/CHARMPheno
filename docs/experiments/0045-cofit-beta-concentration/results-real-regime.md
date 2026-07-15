# Co-fit-beta concentration-recovery (CR-4) results -- real regime

Seed 0. Config: {'regime': 'real', 'K': 60, 'V': 5000, 'D_train': 600, 'D_test': 300, 'doc_len': 44, 'n_em_iter': 50, 'top_k': 10, 'holdout_frac': 0.3, 'seed': 0, 'mechanism_levels': {'logistic_normal': [1, 3, 5, 9], 'dirichlet': [3.0, 1.0, 0.3, 0.1]}, 'c_grid': [1, 2, 3, 5, 8], 'alpha_grid': [0.05, 0.1, 0.3, 1.0, 3.0]}.

| mechanism | level | planted | FROZEN STM tm | FROZEN LDA tm | COFIT STM tm | COFIT LDA(HO) tm | COFIT LDA(aopt) tm | STM betasharp topk | LDA(HO) betasharp topk | LDA(aopt) betasharp topk | true betasharp topk | STM betacos | LDA(HO) betacos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| logistic_normal | 1 | 0.098 | 0.073 | 0.055 | 0.053 | 0.030 | 0.034 | 0.207 | 0.211 | 0.212 | 0.211 | 0.906 | 0.912 |
| logistic_normal | 3 | 0.216 | 0.177 | 0.178 | 0.055 | 0.030 | 0.036 | 0.208 | 0.212 | 0.213 | 0.211 | 0.878 | 0.906 |
| logistic_normal | 5 | 0.309 | 0.282 | 0.239 | 0.080 | 0.030 | 0.039 | 0.161 | 0.213 | 0.214 | 0.211 | 0.811 | 0.899 |
| logistic_normal | 9 | 0.443 | 0.407 | 0.410 | 0.099 | 0.050 | 0.049 | 0.171 | 0.216 | 0.217 | 0.211 | 0.734 | 0.813 |
| dirichlet | 3.0 | 0.049 | 0.060 | 0.030 | 0.053 | 0.030 | 0.031 | 0.205 | 0.210 | 0.210 | 0.211 | 0.910 | 0.911 |
| dirichlet | 1.0 | 0.073 | 0.070 | 0.052 | 0.053 | 0.030 | 0.033 | 0.208 | 0.213 | 0.214 | 0.211 | 0.910 | 0.913 |
| dirichlet | 0.3 | 0.147 | 0.135 | 0.143 | 0.055 | 0.030 | 0.032 | 0.212 | 0.216 | 0.217 | 0.211 | 0.895 | 0.907 |
| dirichlet | 0.1 | 0.271 | 0.264 | 0.217 | 0.056 | 0.030 | 0.036 | 0.211 | 0.215 | 0.216 | 0.211 | 0.876 | 0.901 |

## Summary

[real regime] VERDICT: PARTIAL. theta top_mass at the held-out-LL-calibrated knob (mean over 8 cells; planted 0.201): FROZEN STM 0.183 / LDA 0.166 (gap -0.018); CO-FIT STM 0.063 / LDA-heldout 0.032 (gap -0.030) / LDA-alpha-opt 0.036. beta-sharpness top_k mass (true 0.211): STM 0.198 vs LDA 0.213 (eff_vocab STM 103 vs LDA 86); LDA beta IS sharper than STM. Co-fitting beta does NOT widen the STM-vs-LDA peakiness gap vs frozen beta.

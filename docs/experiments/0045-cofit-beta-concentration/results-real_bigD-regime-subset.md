# Co-fit-beta concentration-recovery (CR-4) results -- real_bigD regime

Seed 0. Config: {'regime': 'real_bigD', 'K': 60, 'V': 5000, 'D_train': 3000, 'D_test': 1000, 'doc_len': 44, 'n_em_iter': 50, 'top_k': 10, 'holdout_frac': 0.3, 'seed': 0, 'mechanism_levels': {'logistic_normal': [5], 'dirichlet': [0.1]}, 'c_grid': [1, 2, 3, 5, 8], 'alpha_grid': [0.05, 0.1, 0.3, 1.0, 3.0]}.

| mechanism | level | planted | FROZEN STM tm | FROZEN LDA tm | COFIT STM tm | COFIT LDA(HO) tm | COFIT LDA(aopt) tm | STM betasharp topk | LDA(HO) betasharp topk | LDA(aopt) betasharp topk | true betasharp topk | STM betacos | LDA(HO) betacos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| logistic_normal | 5 | 0.311 | 0.277 | 0.240 | 0.150 | 0.123 | 0.030 | 0.178 | 0.185 | 0.207 | 0.211 | 0.436 | 0.433 |
| dirichlet | 0.1 | 0.266 | 0.263 | 0.215 | 0.158 | 0.131 | 0.030 | 0.062 | 0.131 | 0.204 | 0.211 | 0.678 | 0.497 |

## Summary

[real_bigD regime] VERDICT: PARTIAL. theta top_mass at the held-out-LL-calibrated knob (mean over 2 cells; planted 0.288): FROZEN STM 0.270 / LDA 0.227 (gap -0.043); CO-FIT STM 0.154 / LDA-heldout 0.127 (gap -0.027) / LDA-alpha-opt 0.030. beta-sharpness top_k mass (true 0.211): STM 0.120 vs LDA 0.158 (eff_vocab STM 408 vs LDA 227); LDA beta IS sharper than STM. Co-fitting beta does NOT widen the STM-vs-LDA peakiness gap vs frozen beta.

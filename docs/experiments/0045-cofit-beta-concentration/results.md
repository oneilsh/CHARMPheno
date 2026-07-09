# Co-fit-beta concentration-recovery (CR-4) results -- clean regime

Seed 0. Config: {'regime': 'clean', 'K': 8, 'V': 400, 'D_train': 300, 'D_test': 200, 'doc_len': 60, 'n_em_iter': 60, 'top_k': 10, 'holdout_frac': 0.3, 'seed': 0, 'mechanism_levels': {'logistic_normal': [1, 3, 5, 9], 'dirichlet': [3.0, 1.0, 0.3, 0.1]}, 'c_grid': [1, 2, 3, 5, 8], 'alpha_grid': [0.05, 0.1, 0.3, 1.0, 3.0]}.

| mechanism | level | planted | FROZEN STM tm | FROZEN LDA tm | COFIT STM tm | COFIT LDA(HO) tm | COFIT LDA(aopt) tm | STM betasharp topk | LDA(HO) betasharp topk | LDA(aopt) betasharp topk | true betasharp topk | STM betacos | LDA(HO) betacos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| logistic_normal | 1 | 0.330 | 0.316 | 0.316 | 0.265 | 0.263 | 0.194 | 0.318 | 0.316 | 0.290 | 0.322 | 0.116 | 0.114 |
| logistic_normal | 3 | 0.478 | 0.465 | 0.438 | 0.378 | 0.377 | 0.427 | 0.318 | 0.318 | 0.311 | 0.322 | 0.051 | 0.049 |
| logistic_normal | 5 | 0.577 | 0.560 | 0.509 | 0.489 | 0.496 | 0.553 | 0.324 | 0.319 | 0.302 | 0.322 | 0.034 | 0.033 |
| logistic_normal | 9 | 0.695 | 0.677 | 0.688 | 0.578 | 0.650 | 0.700 | 0.325 | 0.342 | 0.283 | 0.322 | 0.047 | 0.151 |
| dirichlet | 3.0 | 0.232 | 0.239 | 0.208 | 0.206 | 0.182 | 0.150 | 0.247 | 0.283 | 0.222 | 0.322 | 0.601 | 0.606 |
| dirichlet | 1.0 | 0.323 | 0.313 | 0.313 | 0.274 | 0.218 | 0.189 | 0.306 | 0.314 | 0.275 | 0.322 | 0.166 | 0.259 |
| dirichlet | 0.3 | 0.477 | 0.483 | 0.484 | 0.418 | 0.428 | 0.468 | 0.326 | 0.324 | 0.310 | 0.322 | 0.067 | 0.060 |
| dirichlet | 0.1 | 0.725 | 0.710 | 0.704 | 0.660 | 0.653 | 0.686 | 0.321 | 0.321 | 0.317 | 0.322 | 0.023 | 0.022 |

## Summary

[clean regime] VERDICT: PARTIAL. theta top_mass at the held-out-LL-calibrated knob (mean over 8 cells; planted 0.480): FROZEN STM 0.471 / LDA 0.458 (gap -0.013); CO-FIT STM 0.408 / LDA-heldout 0.408 (gap -0.000) / LDA-alpha-opt 0.421. beta-sharpness top_k mass (true 0.322): STM 0.311 vs LDA 0.317 (eff_vocab STM 63 vs LDA 61); LDA beta IS sharper than STM. Co-fitting beta does NOT widen the STM-vs-LDA peakiness gap vs frozen beta.

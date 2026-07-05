# Seed-panel synthetic method-validation (exp 0039)

Known-ground-truth check that the seed-panel acceptance test (spark_vi.eval.topic.seed_panel) DETECTS generative-scale over-commitment: a planted, disjoint-vocabulary beta and a planted unit-diagonal correlation R (tests/_stm_synth.py:gated_ln_corpus; groups=('A','B'), fg_per_group=4, bg_k=3, V=400, doc_len=40) are fed straight into seed_panel_sweep with a zero Gamma and reference=None. No STM fit is run -- the sweep is a pure function of (beta, Gamma, R, partition), so the planted values ARE the model under test. s_true=5 is a reference point for interpretation (the value planned for the real corpus), not a scale baked into doc generation (gated_ln_corpus draws eta ~ N(0, Sigma_true), Sigma_true already unit-diagonal, i.e. implicit scale 1); this script does not use the corpus's generated documents at all.

| c | median top_mass | median eff_topics | median second_mass | recover-self rate |
|---|---|---|---|---|
| 2 | 0.4757 | 3.6754 | 0.0886 | 1.0000 |
| 3 | 0.5712 | 2.8018 | 0.0727 | 1.0000 |
| 4 | 0.6362 | 2.3427 | 0.0617 | 1.0000 |
| 5 | 0.6831 | 2.0688 | 0.0539 | 1.0000 |
| 8 | 0.7686 | 1.6677 | 0.0394 | 1.0000 |

Median top_mass is non-decreasing in c: **True**. Median eff_topics is non-increasing in c: **True**. Secondary structure collapses materially between c=3 and c=8 (eff_topics collapse=True and/or second_mass collapse=False). Self-recovery rate at c in {3,4,5} is >= 0.8 for every value: {3: 1.0, 4: 1.0, 5: 1.0}.

**Interpretation.** As c grows, the prior precision (1/c)*R^-1 over the allowed topic set weakens, so a 1-2 token seed's likelihood term dominates the posterior mode more completely -- top_mass rises and eff_topics/second_mass fall. This is exactly the over-commitment failure mode the reviewer described, reproduced on a corpus where the truth is known: the acceptance test's summary statistics move in the expected direction and the recover-self rate confirms seeds still land on their planted source topic through c=5. This validates the METHOD (the statistics DETECT over-commitment as designed) -- it does not, by itself, decide whether c=5 over-commits on the REAL corpus; see real_results.md for that.

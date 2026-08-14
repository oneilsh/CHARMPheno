# Multi-domain PC — compaction handoff (2026-08-14)

**Branch:** `claude/spectral-anchor-topic-k-200nqp`. Read alongside
`2026-08-14-multidomain-pc-plan.md` (the design + merge assessment). This is the
resume point after a compaction.

## Where we are

**Engine cherry-pick from the hybrid branch is DONE and GREEN** (commits `7083bda`,
`43689f3`). The multi-domain `GatedOnlineLDA` (per-domain λ dict `{m:(K,V_m)}`, fuse/
split bridge) now lives on this branch, and our OnlinePCLDA / Gated-PC / Firth all pass
against it (`test_gated_lda.py` + `test_pc_lda_shim.py`: 64 passed / 0 failed).

Files pulled (all were untouched-on-ours → clean-takes; NO merge conflicts):
- `models/topic/gated_lda.py` (multi-domain λ), `models/topic/domains.py` (domain_bounds).
- `models/topic/lda.py` — ADDITIVE `gamma_count_weight=None` on `_cavi_doc_inference` (the
  ω tempering hook; `None` = bit-for-bit unchanged → our PC byte-identical).
- `models/topic/gated_init.py` + `models/topic/spectral_init.py` (multi-domain spectral
  init) + `tests/_stm_synth.py` (`two_domain_dag_corpus` fixture) + hybrid's
  `test_gated_lda.py` (matches the extended engine).
- **Deliberately NOT pulled:** hybrid's `dag_placement.py` (+587, would perturb our
  run 4–8 baseline) and `case_finding_assembly.py` (kept OUR emit_labels version). Our
  `dag_placement._as_counts` satisfies hybrid's gated_init dep.

## What multi-domain PC needs (the design — see plan §3/§4)

- **Head: NO change** (reads K-dim θ, domain-agnostic).
- **Per-domain λ: DONE** (the cherry-picked engine).
- **THE ONE NEW ENGINE BIT (unbuilt):** make OnlinePCLDA's supervised **topic correction**
  (∂loss_y/∂λ, a `(K,V)` grad over the concatenated vocab) domain-aware by scattering it
  through `GatedOnlineLDA._split_to_domains` into the right per-domain block before the
  `(1-ρ)·old + ρ·target` blend. Verify our correction path in `pc.py` handles the dict-λ
  (`global_params["lambda"]` is a dict `{m:(K,V_m)}` when multi-domain, a single array when
  not). Unit-test on a 2-domain synthetic (correction lands in the right block).
- **Corpus:** bring `charmpheno/omop/multi_domain.py` + `measurement_tokens.py` from
  hybrid (value-aware measurement: `concept_id*100+state`, low/normal/high/coded/presence),
  adapt `assemble_multidomain` to call OUR `attach_labels` (emit_labels) so the multi-domain
  case-finding corpus carries per-node label/labelMask. Start with **conditions + value-aware
  measurement** (the ONE non-condition domain that carried signal, insights 0078/0079).
- **Shim + driver:** extend the gated-PC shim (`mllib/topic/pc.py`) to accept multi-domain
  features (featuresCols + domain sizes → domain_bounds, mirror `mllib/topic/gated_lda.py`'s
  `_concat_domain_features`); add a driver arm.
- **ω is NOT needed** — PC replaces the tuned per-domain weight with a LEARNED per-node
  domain emphasis (keep ω=1). That's the whole thesis.

## The thesis (why this is the untested "right way")

The hybrid branch exhausted FIXED + POST-HOC-supervised domain combination (~6% ceiling,
"not worth deploying") — but that reweighted per-domain LR SCORES off a FROZEN unsupervised
fit; it never shaped the fit. PC shapes the fit (∂loss_y/∂λ) with a per-node gate, so each
disease node can specialize toward its predictive domain (Marfan→measurement, MG→drug) —
the disease-specific balance no fixed global rule could express. NOT refuted by the 6%
(different, weaker mechanism). **Honest caveat:** 0076/0079 say aggregate case-finding is
INFORMATION-limited under fixed readout; the bet is supervised per-node shaping extracts the
specialist signal (rescues are demonstrably present) into the macro. Run 6 showed PC does
this kind of work when the unsupervised fit misses.

**The headline test:** does supervised multi-domain beat (a) condition-only Gated-PC (our
runs), (b) unsupervised multi-domain gated, (c) the 6% fixed readout? And the DIRECT test:
inspect per-node per-domain λ mass — does Marfan's/GBS's topic go measurement-heavy while
EDS's stays condition-heavy?

## Resume steps (in order)

1. Bring `multi_domain.py` + `measurement_tokens.py` onto this branch; adapt
   `assemble_multidomain` to emit_labels. (Check: does multi_domain.py call only helpers
   present in OUR case_finding_assembly? It reuses split/frontier/prune/strip verbatim.)
2. Domain-aware topic correction in `pc.py` (the scatter) + 2-domain unit test.
3. Multi-domain gated-PC shim path (featuresCols → domain_bounds).
4. Driver arm + run: multi-domain Gated-PC vs the 3 baselines + per-node λ inspection.

## Env / commands
- venv `/home/user/CHARMPheno/.venv-pc/bin/python`; pyspark tests need
  `JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 PYSPARK_PYTHON=<venv> PYSPARK_DRIVER_PYTHON=<venv>
   PYTHONPATH=spark-vi:.:charmpheno`.
- Cluster run: `make -C analysis/cloud exp ID=76` (config = 0076 frontmatter).
- hybrid branch fetched as `origin/claude/hybrid-domain-reliability-review-ckn2bq`;
  cherry-pick more files via `git checkout <that-ref> -- <path>` (they're untouched-on-ours).

## Parked (not multi-domain)
- Firth on the cluster ABANDONED (task #29 = cheap resurrection: damped inner Newton +
  absolute-floor conditioner, no slogdet line search — for calibrated P(node) later).
- alpha tuning is OFF on the gated-PC path (shim never threads optimizeDocConcentration +
  frontier_histogram to the gated delegate) — a real bug, deferred per user.
- Spectral init as a "does it help" future item.
- Auto-K (HDP was crappy); the co-fit makes K matter — unsolved, parked.

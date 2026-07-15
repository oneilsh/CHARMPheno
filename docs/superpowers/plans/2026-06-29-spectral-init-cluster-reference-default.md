# Next-Experiment Plan + Context: dense spectral init on the cluster + reference default

> **Status:** Pre-implementation plan + compaction-survival context. The task
> list below is a "concept of a plan" — a couple of scope decisions (flagged) and
> the no-placeholder writing-plans pass remain before SDD execution.

**Goal:** Land the pending fixes needed to run the *next* cancer experiment with
**reference + spectral init**, expecting it to fix BOTH pathologies seen so far
(σ_init=1 topic under-resolution AND the Σ→10^10 blowup) on real data — without
building the large-V scalable rewrite yet.

---

## The story so far (read first on resume)

We are hardening `OnlineSTM` (prevalence-only Structural Topic Model, logistic-
normal prior, online VI in Spark) on the unmerged `stm` branch. Three published-
STM stabilizers (insight 0029): (1) K−1 reference identifiability, (2) Σ
shrinkage prior, (3) spectral/anchor-word init. All three are IMPLEMENTED opt-in
on the engine; (1) and (2) are now THREADED to the cluster; (3) is not.

**Empirical findings from the cancer-cohort experiments (the load-bearing context):**

The cancer cohort: `first_cancer_year`, prior_obs_days 0, person_mod 4, doc_unit
patient, `~ C(sex) + age`, K=40, seed 42, V=3691, ~10.8k patient docs,
condition_era.

- **exp 0008** — full-K, σ_init=1 (default): **collapse**. 2 catch-alls + ~36
  marginal clones. NPMI +0.055.
- **exp 0010** — full-K, σ_init=5: ~28 crisp phenotypes, NPMI +0.216, but **Σ→~10^10**.
- **exp 0012** — **reference**, σ_init=1: **escapes the collapse** — ~14 real
  phenotypes (breast/ER+, thyroid, lung, lymphoma, …), NPMI mean +0.084. BUT only
  ~14 of 40 topics active (~26 dead marginal clones), 3 topics MIXED
  (prostate+thyroid, pregnancy+neuro, ovary+scleroderma) = under-resolution, and
  **Σ still improper (~5×10^10)**.
- **exp 0013** — **reference**, σ_init=5 (mid-fit, looked decisive): **~28 rich,
  CLEAN phenotypes** — the mixed topics split (clean prostate AND thyroid; new
  clean HIV-lymphoma, HCV→cirrhosis→HCC, CHF, esophageal/Barrett's, ENT, MSK,
  …). Matches full-K@σ=5. BUT **Σ still blows up (~2.85×10^10)**.
- **exp 0014** — reference, σ_init=10: queued, lower-stakes (confirms high end).

**Two conclusions locked by 0012 vs 0013:**

1. **`reference_topic` should DEFAULT ON.** It strictly helps or ties full-K at
   every σ_init (rescues σ=1's catastrophic collapse; ties at σ=5). Keep the
   toggle only for research/repro. (User agreed: test first — now tested.)
2. **Reference is NOT enough by itself.** It fixes the *level/collapse*
   degeneracy but neither the σ=1 *under-resolution* (σ=1→14 vs σ=5→28 topics —
   σ_init still matters) nor the *Σ blowup* (improper at both σ=1 and σ=5).

**Why spectral init is the fix for BOTH remaining pathologies (the key insight):**

The Σ blowup and the under-resolution are the SAME root cause — **random β
initialization**. With random β, the only way topics differentiate is by pushing
η to extremes (θ toward simplex corners), because β starts near-uniform. Extreme
η → softmax saturation → huge residuals → the residual-variance M-step inflates
Σ without bound. So *crisp topics and bounded Σ are in direct tension under
random init*: a Σ-prior strong enough to bound Σ also bounds η, which re-collapses
differentiation back toward the σ=1 regime. The Σ-prior **trades blowup for
collapse**; it cannot give both. Spectral init breaks the tension by putting the
differentiation in **β** from iteration 0 (anchor words → distinct topic-word
distributions). Then θ/η can be moderate while topics stay distinct → Σ never
runs away. This is how the reference `stm` package keeps Σ proper *even with*
`sigma.prior=0` (its default). Synthetic confirms: spectral+reference at σ=1 gave
recovery with **Σ≈3.7**, not 10^10. So spectral is the necessary third piece;
reference is the identifiability prerequisite.

**Crucial scoping fact:** the Σ blowup does NOT block the demo. The dashboard uses
Γ (covariate-prevalence regression) + the topics, not Σ (ADR 0028 ignores Σ
scale). exp 0013, once converged, is already a shippable demo (~28 clean
phenotypes). The blowup only matters for a *proper* logistic-normal fit
(interpretable / sample-able Σ) — which is the spectral-init goal.

---

## What is already shipped / threaded (do not redo)

- **K−1 reference** (`OnlineSTM(reference_topic=True)`, ADR 0031): pins topic 0's
  η≡0, clamped-K. Commits 4793e7b..4ced1ae.
- **Σ-prior** (`sigma_prior_scale`, `sigma_prior_count`) on the engine.
- **Cluster threading of reference + Σ-prior** (the just-completed SDD arc,
  ready-to-merge): experiment frontmatter → `run_experiment.build_stm_args` →
  cloud driver → `StreamingSTM` → `OnlineSTM`; provenance in
  `metadata["stm_hardening"]`. Commits 41d48f0, 4a062c5, 5f9d740, f1f9af6.
- **allow_pickle gated-reload fix** (295a6e7) + **ADR 0032** (741813a, scalable
  spectral = random projection over maxV). Both local, **unpushed** as of writing.
- The **dense `spectral_init.py`** prototype (block-aware anchor-word init,
  `spectral_init_beta(docs, partition, V) -> KxV`): exists, unit-tested, NOT wired
  to any cluster driver. Runs in ~18s / 109 MB at V=3691 (benchmarked).
- **Wiring path confirmed:** `VIRunner.fit(..., data_summary=...)` already
  forwards `data_summary` to `model.initialize_global` ([runner.py:160](../../spark-vi/spark_vi/core/runner.py#L160)),
  and `OnlineSTM.initialize_global` already consumes `{"spectral_beta": KxV}`.
  `StreamingSTM.fit` ([mllib/stm.py:318](../../spark-vi/spark_vi/mllib/topic/stm.py#L318))
  just calls `runner.fit(...)` without `data_summary` today.

---

## Pending fixes — inventory

| # | Fix | Size | In next experiment? |
|---|---|---|---|
| A | Flip `reference_topic` default → True; keep toggle for research | small | yes (or set in frontmatter) |
| B | Wire **dense** spectral init into the cluster path (pre-fit stage + flag + driver CLI + run_experiment) | medium | **yes — the main piece** |
| C | `min_marginal_frac` → absolute document-frequency floor in `spectral_init.py` (ADR 0032; rare-phenotype-friendly) | small | yes (folds into B) |
| D | Push the 2 local commits (allow_pickle, ADR 0032) | trivial | n/a (housekeeping) |
| E | Σ-prior in the experiment config — secondary knob; spectral should bound Σ on its own, Σ-prior is at most a top-up | n/a | optional cell |
| — | **Scalable** spectral rewrite (distributed co-occ + random projection + distributed NNLS, ADR 0032) | LARGE | **NO — separate future arc** |

---

## Scope decision (RECOMMENDED — confirm on resume)

**Dense-first, scalable-later.** Wire the existing dense `spectral_init.py` into
the cluster path now (item B), which is enough for the V=3691 cancer cohort, and
defer the scalable rewrite (ADR 0032) to its own arc.

Rationale: (1) cancer is small — dense is ~18s/109MB on the driver, no distributed
machinery needed; (2) **de-risking** — running dense spectral on REAL cancer data
PROVES (or refutes) that spectral+reference fixes the Σ blowup + under-resolution
before we invest weeks in the distributed build; if real data behaves unlike
synthetic, we want to learn that cheaply first; (3) the dense and scalable
versions compute the *same* β seed — the scalable one is performance engineering,
not a different algorithm, so the experiment's scientific conclusion transfers.

The one thing to confirm: are we OK demoing/validating on the dense path now, with
the scalable rewrite as a tracked follow-on? (If the priority is the general
large-V library over the cancer demo, we'd instead build scalable first — bigger,
slower to first result.)

---

## The plan (concept — needs the writing-plans no-placeholder pass)

**Task 1 — `min_marginal_frac` → document-frequency floor (item C).**
In `spark-vi/spark_vi/models/topic/spectral_init.py`, replace the mean-relative
candidate floor in `find_anchors` with an absolute doc-frequency floor (a word may
anchor only if it appears in ≥ `min_doc_count` documents, default tunable ~25).
Document frequency is computed alongside the co-occurrence. Keeps rare-but-pure
phenotype words eligible (ADR 0032). TDD: a synthetic case where a rare-but-pure
word must remain an anchor candidate while a 2-doc degenerate word is excluded.

**Task 2 — dense spectral pre-fit stage in `StreamingSTM` (item B core).**
Add `spectral_init: bool = False` to `StreamingSTM.__init__` (thread + store, like
the existing knobs). In `fit`, when on: after the doc RDD is built/persisted
([mllib/stm.py:295-305](../../spark-vi/spark_vi/mllib/topic/stm.py#L295-L305)),
collect the docs to the driver (`rdd.collect()` — fine at ~10.8k docs; a sample
cap is a future option), build the partition (`self.topic_blocks` or the implicit
all-background one), call `spectral_init_beta(docs, partition, vocab_size)`, and
pass `data_summary={"spectral_beta": beta}` to `runner.fit(...)`. Record
`spectral_init` in `metadata["stm_hardening"]`. TDD: a tiny end-to-end fit with
`spectral_init=True` yields global params seeded from the spectral β (e.g. λ row
mass reflects the anchor-word seed, not the random-gamma default).
- OPEN DETAIL for the writing-plans pass: confirm `OnlineSTM.initialize_global`'s
  spectral branch returns the full global-params dict the runner expects, and that
  collecting `STMDocument`s (with `.groups`) round-trips for the gated case.

**Task 3 — thread `--spectral-init` through the drivers + runner (item B).**
Same pattern as the just-completed config-flag arc: `--spectral-init` (store_true)
on `analysis/cloud/stm_bigquery_cloud.py` and `analysis/local/fit_stm_local.py`,
forwarded to `StreamingSTM`; and `build_stm_args` in `scripts/run_experiment.py`
emits `--spectral-init` when `effective.get("spectral_init")` is truthy. TDD:
arg-parse + build_stm_args tests mirroring the existing ones.

**Task 4 — flip `reference_topic` default (item A).**
Change `OnlineSTM` (and `StreamingSTM`) `reference_topic` default False → True;
keep the param (research/repro toggle). Update the byte-identical-default tests
(the default is no longer "off"). Audit: any existing STM test that assumed
full-K behavior by default. Decide whether to keep the experiment frontmatter
explicit (`reference_topic: true`) regardless, so docs are self-documenting.
- OPEN: do this as part of this arc, or as a separate deliberate commit after the
  spectral experiment confirms? (Recommended: flip AFTER exp 0015 confirms
  spectral+reference is the good config, so the default reflects a validated
  stack. Until then, frontmatter sets it explicitly.)

**Task 5 — exp 0015 (+0016): the validating experiment.**
Write `docs/experiments/0015-stm-cancer-reference-spectral-sigma1.md`: cancer
cohort, K=40, reference_topic true, spectral_init true, **σ_init=1**, max_iter 300.
The decisive test: does spectral+reference at the DEFAULT σ_init give (a) the ~28
rich topics of exp 0013 AND (b) a **bounded, proper Σ** (O(1–100), not 10^10)?
Optionally 0016 adds a moderate Σ-prior (`sigma_prior_scale ~20`, count to bind)
as a top-up cell. Compare against 0012 (reference-only σ=1) and 0013 (reference
σ=5). Success = rich topics + bounded Σ at σ=1 = spectral closes both gaps =
green-light to (i) flip the reference default and (ii) invest in the scalable arc.

---

## Deferred (tracked, NOT in this arc)

- **Scalable spectral rewrite (ADR 0032):** distributed co-occurrence + random
  projection of word rows to V×d (d ≈ max(K, ε⁻²·log V), O(V·K) ≈ ~1 GB at
  V=100k/K=1000) + distributed NNLS recovery; never materialize V×V (dense is
  80 GB at V=100k). Its own brainstorm → spec → plan. The dense path proves the
  science; this is the performance engineering for the general large-V library.
- **Σ-prior as a primary fix:** ruled out (trades blowup for collapse). Keep as a
  secondary top-up knob only.
- **K−1 reference removed as the only mode:** not now — keep the toggle for
  research/ablation/repro even after flipping the default.

---

## Resume actions (post-compaction)

1. Confirm the scope decision (dense-first vs scalable-first).
2. Check exp 0013/0014 converged results (the σ=5/σ=10 reference data points) and,
   if not yet written, draft the **insight** pairing 0012 (under-resolution) +
   0013 (σ=5 recovers richness) + the "reference rescues collapse but doesn't kill
   the σ_init knob / Σ blowup → spectral is the fix" conclusion.
3. Run the writing-plans no-placeholder pass on Tasks 1–5 (resolve the two OPEN
   details), then execute via subagent-driven-development like the threading arc.
4. Housekeeping: push the 2 local commits (allow_pickle 295a6e7, ADR 0032 741813a)
   when ready.

## Pointers

- Insight: `docs/insights/0029-stm-sigma-init-collapse-blowup-missing-stabilizers.md`
- ADRs: 0031 (K−1 reference), 0032 (scalable spectral = random projection over maxV)
- Engine: `spark-vi/spark_vi/models/topic/stm.py` (reference, Σ-prior),
  `spark-vi/spark_vi/models/topic/spectral_init.py` (dense prototype)
- Shim/wiring: `spark-vi/spark_vi/mllib/topic/stm.py`, `core/runner.py:160`
- Drivers: `analysis/cloud/stm_bigquery_cloud.py`, `analysis/local/fit_stm_local.py`,
  `scripts/run_experiment.py::build_stm_args`
- Just-completed threading plan: `docs/superpowers/plans/2026-06-27-stm-hardening-cluster-threading.md`
- Memory: `[[project_stm_hardening]]`

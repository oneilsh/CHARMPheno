# Topic Coherence Evaluation — Design Spec

**Date:** 2026-05-11
**Status:** Draft, pending user review
**Scope:** First piece of the evaluation/assessment surface for the framework — held-out topic coherence via normalized pointwise mutual information (NPMI). Lands as a new `spark_vi.eval.topic` subpackage in `spark-vi`, a domain-side BOW split helper in `charmpheno/omop/`, and a thin analysis driver. Does **not** cover term-relevance computation, pyLDAvis adaptation, simulation/recovery testing, or quantitative comparison frameworks (OCTIS-style). Those are explicit follow-ons.

A small prerequisite refactor lands first in its own branch: `spark_vi.models.{lda,online_hdp,counting}` → `spark_vi.models.topic.{lda,online_hdp,counting}`. The eval branch builds on top.

---

## Context

The framework now has two working topic models (`OnlineLDA`, `OnlineHDP`), MLlib-style estimator/model shims for both, persistence (ADR 0006), and checkpoint/save/load on the shim side (ADR 0015). What's missing is any *quantitative* assessment of topic quality. K-selection for LDA, T-selection for HDP, η/α/γ hyperparameter comparisons, model-class comparisons — none of these are decidable from ELBO alone (ELBO doesn't compare across K, and only weakly tracks interpretability). The standard answer is held-out coherence, computed from word-pair co-occurrence statistics on documents the model didn't see during training.

The prior screenshot from the user's earlier project used a custom modified-UCI form: `Σ log₂((1 + p(t_i, t_j)) / (p(t_i)·p(t_j)))` summed over the top-N term pairs per topic, then z-scored across topics. That works, but the brainstorming session settled on NPMI as the established alternative — bounded in `[-1, 1]`, no smoothing constant to tune, no normalization required to interpret, and directly comparable across runs.

The "sliding window" of textbook NPMI is a token-ordering construct over running text. OMOP records *are* temporally ordered (every event has a timestamp), but for v1 we treat each patient bag as unordered for co-occurrence purposes. The reason isn't that order doesn't matter — it's that choosing the right window for clinical timelines is itself a non-trivial question (1-year rolling window? episode-bounded? observation-period-relative?), and patient timelines have variable density, gaps, and episode structures that don't map cleanly onto NPMI's sliding-window formulation. We leave temporal-window coherence variants to a follow-on and treat the whole-bag formulation as the principled v1 baseline.

## Goals

1. **Ship a generic NPMI coherence metric** in `spark_vi.eval.topic` that takes a topic-term matrix and a held-out BOW RDD, returns per-topic scores plus a summary, and works identically for OnlineLDA and OnlineHDP.
2. **Provide a deterministic, reproducible BOW split helper** in `charmpheno/omop/` so analysis drivers can produce held-out partitions without depending on Spark's `randomSplit` partition sensitivity. Splits at the application layer, not inside any estimator.
3. **Wire an end-to-end coherence driver** at `analysis/local/eval_coherence.py` that loads a saved `VIResult`, splits a BOW, computes NPMI, and prints a ranked-topic report with concept names. Cloud variant deferred to v2.
4. **Land ADR 0017** documenting the metric choice (NPMI over modified-UCI), the layering decision (generic in spark-vi, split in charmpheno, driver in analysis), and the v1/v2 split.
5. **Prerequisite: refactor models into `spark_vi.models.topic.*`** so the eval namespace's `eval.topic.*` mirrors a `models.topic.*` on the model side. Separate branch, separate PR, no scope creep.

## Scope

### Prerequisite branch (`refactor/models-topic-namespace`)

- Move `spark_vi/models/{lda,online_hdp,counting}.py` → `spark_vi/models/topic/`.
- Add `spark_vi/models/topic/__init__.py` re-exporting `OnlineLDA`, `OnlineHDP`, the counting helpers.
- Update imports across `spark_vi/mllib/`, `spark_vi/io/`, tests, `analysis/`, and ADR/spec cross-references.
- ADR (small, ~one page) recording the rename rationale — eval-namespace symmetry, room for future non-topic-model VI applications.
- Single mechanical PR; merge before starting eval work.

### In v1 (`feat/eval-topic-coherence`)

- `spark_vi/eval/__init__.py` — package marker, no public surface yet.
- `spark_vi/eval/topic/__init__.py` — re-exports public coherence API.
- `spark_vi/eval/topic/coherence.py` — NPMI computation:
  - `compute_npmi_coherence(topic_term: np.ndarray, holdout_bow: RDD, top_n: int = 20, hdp_topic_mask: np.ndarray | None = None) -> CoherenceReport`
  - Returns a frozen `CoherenceReport` dataclass.
- `spark_vi/eval/topic/types.py` — `CoherenceReport` dataclass (per-topic NPMI, top-N term indices per topic, descriptive summary stats).
- `charmpheno/charmpheno/omop/split.py` — `split_bow_by_person(bow_df, fraction, seed)`:
  - Deterministic SHA-256 hash of `person_id`, modulo a fixed bucket count, threshold defines holdout.
  - Returns `(train_bow_df, holdout_bow_df)`.
  - Reproducible regardless of partition state (unlike Spark's `randomSplit`).
  - The SHA-256-on-`person_id` pattern already exists in `analysis/cloud/lda_bigquery_cloud.py` for ID hashing; we promote it to a reusable, BOW-shaped split helper.
- `analysis/local/eval_coherence.py` — driver script:
  - Loads a `VIResult` checkpoint.
  - Rebuilds the BOW via charmpheno (or loads cached) and applies `split_bow_by_person`.
  - Computes `E[β]` from `lambda_`, runs `compute_npmi_coherence`, prints ranked topic table with concept names.
- Unit tests on the metric math (tiny synthetic corpora with known co-occurrences), the split helper (determinism + no overlap), and the HDP topic-mask filter.
- Integration test using a saved small-corpus checkpoint to verify end-to-end wiring.
- ADR 0017 documenting metric choice, layering, scope.

### Not in v1 (explicit)

- **Term relevance (Sievert–Shirley) and concept-named top-N tables.** Closely adjacent (same `E[β]`, same vocab + concept-name plumbing), but its own deliverable. Roll into a v2 spec.
- **pyLDAvis adapter.** Requires materializing a full `doc_topic_dists` matrix, which is its own plumbing exercise (subsampling strategy, `transform_corpus` helper on the shim).
- **Cloud driver.** `analysis/cloud/eval_coherence_cloud.py` mirroring the LDA/HDP cloud drivers. Defer until local works and we have a real model at AoU scale to evaluate.
- **Gensim adapter (`spark_vi/eval/gensim_adapter.py`).** Reserved for if/when we want UMass/c_v/c_uci/c_npmi via gensim. Hand-rolled NPMI is enough for v1.
- **OCTIS integration.** Reserved for the systematic multi-metric / multi-model comparison phase. Coherence is a prerequisite, not a substitute.
- **Simulation / synthetic recovery testing.** Generate ground-truth θ/β, fit, measure Hellinger on best-matched topics. Useful as an inference sanity check; separate spec.
- **Z-score normalization.** NPMI is already bounded and interpretable in absolute terms (Röder et al. 2015 report good topics in ~0.1–0.3 range on held-out), so we don't normalize. Drivers that want a "relative to this run's mean" view can compute it themselves from the per-topic array; not worth a built-in.
- **Topic-level confidence intervals via bootstrap.** Plausible follow-on; out of scope for v1.

### Never (dropped from scope)

- **Modified-UCI with `1 + p(t_i, t_j)` smoothing.** Replaced by NPMI. Rationale: bounded, established, no smoothing constant to tune, and the patient-bag co-occurrence shape we want is already the textbook NPMI shape modulo the sliding-window question (which doesn't apply to bag data).
- **Coherence-as-Param on estimators.** The eval surface lives outside the estimator. Estimators don't split; drivers do. MLlib idiom.

## The Metric

For each topic `t`, let `T_t = {w_1, ..., w_N}` be the top-N term indices by `E[β_t]`. Over the held-out BOW (D documents, each a set of term indices ignoring counts):

```
p(w_i)        = (# held-out docs containing w_i) / D
p(w_i, w_j)   = (# held-out docs containing both w_i and w_j) / D

NPMI(w_i, w_j) = log[ p(w_i, w_j) / (p(w_i) * p(w_j)) ]  /  -log p(w_i, w_j)

NPMI(w_i, w_j) = -1   when p(w_i, w_j) == 0   (Röder et al. 2015 convention)

Coherence(t) = mean over all unordered pairs (w_i, w_j) in T_t with i < j of NPMI(w_i, w_j)
```

Summary statistics over topics: mean, median, stdev, min, max — descriptive only, not used to normalize the per-topic scores.

Notes on the formulation:

- **Whole-document (bag) co-occurrence**, not sliding window. Patient events are temporally ordered, but choosing the right window for medical timelines (1-year rolling? episode-bounded? observation-period-relative?) is its own research question; v1 treats each patient bag as unordered for co-occurrence purposes. A temporal-window variant is a plausible v2 follow-on if topic-quality assessment turns out to be sensitive to it.
- **Binary doc-presence**, not count-weighted. A patient who has condition A 10 times and condition B once is the same co-occurrence signal as a patient with one of each. This matches the screenshot's "any record at any time."
- **N = 20 default** (matches the screenshot and the Röder et al. reproducibility convention). Configurable.
- **Mean over pairs**, not sum. The pair count is `N*(N-1)/2 = 190` for N=20; this is implicit in the formula either way, but using *mean* keeps the score in a comparable scale across choices of N. The screenshot's modified-UCI used a sum; we deviate here intentionally because NPMI is already pre-normalized per-pair, so mean is the natural aggregator.

## Spark-Side Computation

For a held-out corpus with D ≈ 5k–500k docs and a vocabulary of V ≈ 10k:

1. **Build the interest set.** Union of top-N term indices across all (filtered) topics. Size `M ≤ K * N`; in practice `M < K * N` due to overlap. For K=100, N=20: `M ≤ 2000`.
2. **Compute per-term doc-frequency** on the held-out BOW: broadcast the interest-set, map each doc to `[(w, 1) for w in indices if w in interest_set]`, reduceByKey, collect to driver. Output: dict `{w: doc_count}` of size ≤ M.
3. **Compute pairwise co-occurrence** on the held-out BOW: map each doc to the unordered pairs of interest-set terms it contains (`itertools.combinations` over `sorted(set(indices) & interest_set)`), emit `((w_i, w_j), 1)`, reduceByKey, collect to driver. Output: dict `{(w_i, w_j): pair_count}`.
4. **Compute NPMI driver-side** for each topic's top-N pair set using the collected dicts.

Memory budget for collected stats: at K=100, N=20, M≈2000, the pair-count dict has at most `M*(M-1)/2 ≈ 2M` entries, but only pairs that actually co-occur in at least one held-out doc are emitted — usually far fewer. In practice ≪ 1M entries, a few MB on the driver.

If profiling later shows this needs to be even more efficient, the obvious tightening is: only emit pairs that appear in *some topic's top-N* (a subset of all M*(M-1)/2 pairs), via a second broadcast of the actual pair set of interest. Defer until measured.

## API Surface

### Coherence

```python
# spark_vi/eval/topic/types.py
@dataclass(frozen=True)
class CoherenceReport:
    """Per-topic NPMI scores and summary statistics from a coherence evaluation run."""
    per_topic_npmi: np.ndarray         # shape (K,) or (K_used,) for HDP-filtered
    top_term_indices: np.ndarray       # shape (K, N) — top-N term indices per topic
    topic_indices: np.ndarray          # shape (K,) — original topic indices (identity for LDA; mask-filtered for HDP)
    n_holdout_docs: int
    top_n: int
    mean: float
    median: float
    stdev: float
    min: float
    max: float

# spark_vi/eval/topic/coherence.py
def compute_npmi_coherence(
    topic_term: np.ndarray,
    holdout_bow: RDD,
    *,
    top_n: int = 20,
    hdp_topic_mask: np.ndarray | None = None,
) -> CoherenceReport: ...
```

Inputs:

- `topic_term`: `(K, V)` array of `E[β]` (rows sum to 1). Caller computes this from `lambda_ / lambda_.sum(axis=1, keepdims=True)`. Generic for LDA and HDP.
- `holdout_bow`: RDD of `BOWRow` (the existing `spark_vi/core/types.py` type with `indices`, `counts`). NPMI ignores counts and uses `set(indices)` per doc.
- `top_n`: number of top terms per topic. Default 20.
- `hdp_topic_mask`: optional boolean mask of length K. When provided, only `True`-masked topics are scored. The caller produces this for HDP by ranking corpus stick weights `E[β]` and selecting top-K-by-usage (see below). For LDA, omit.

Outputs:

- `CoherenceReport.per_topic_npmi` is length equal to the number of scored topics.
- `topic_indices` carries the original topic identity so the driver can label rows correctly.

### HDP topic filtering

The caller produces `hdp_topic_mask` from corpus-level stick weights:

```python
# In the driver, not the eval function:
from spark_vi.eval.topic import top_k_used_topics  # convenience helper

mask = top_k_used_topics(u, v, k=50)  # length-T bool array
report = compute_npmi_coherence(topic_term, holdout_bow, hdp_topic_mask=mask)
```

`top_k_used_topics(u, v, k)` lives in `spark_vi/eval/topic/__init__.py` (or a small `hdp_helpers.py`). Computes `E[β_t]` from the GEM stick parameters, returns a mask selecting the top-K by descending corpus weight. Default `k=50`; threshold-based variant (`min_weight=0.01`) reserved for v2 if needed.

### Split helper

```python
# charmpheno/charmpheno/omop/split.py
def split_bow_by_person(
    bow_df: DataFrame,
    *,
    holdout_fraction: float,
    seed: int,
    person_id_col: str = "person_id",
) -> tuple[DataFrame, DataFrame]:
    """Deterministic SHA-256-hash split. Returns (train_df, holdout_df).

    The split is reproducible across runs (same seed + same person_id → same bucket)
    and across partition layouts (unlike DataFrame.randomSplit). holdout_fraction
    should be in (0, 1); a typical eval value is 0.1.
    """
```

Implementation: `sha2(concat(person_id, lit(seed)), 256)`, take the low 32 bits as an integer modulo a large bucket count (e.g., 10000), threshold by `int(holdout_fraction * 10000)`. Two boolean masks, two `.filter()` calls, return both DataFrames.

## Driver Shape

```python
# analysis/local/eval_coherence.py (skeleton)
result = load_result(checkpoint_path)
lambda_ = result.global_params["lambda_"]
topic_term = lambda_ / lambda_.sum(axis=1, keepdims=True)

bow_df = build_bow(...)  # existing charmpheno path
train_df, holdout_df = split_bow_by_person(bow_df, holdout_fraction=0.1, seed=42)
# (note: train_df is unused here; we assume the saved checkpoint was trained on the
#  same split with the same seed. The eval driver and the training driver must agree
#  on holdout_fraction + seed. We will document this as a contract in the driver header
#  and ADR 0017, and consider stamping it into VIResult.metadata as a v2 hardening.)

holdout_bow = holdout_df.rdd.map(_to_bow_row)

if model_class == "OnlineHDP":
    mask = top_k_used_topics(result.global_params["u"], result.global_params["v"], k=50)
else:
    mask = None

report = compute_npmi_coherence(topic_term, holdout_bow, top_n=20, hdp_topic_mask=mask)
_print_ranked_report(report, vocab, concept_names)
```

Open contract issue: the eval driver assumes the checkpoint was trained on the matching `(holdout_fraction, seed)` split. v1 documents this in the driver header and the ADR; v2 considers stamping the split provenance into `VIResult.metadata` so eval can verify (or refuse to run on mismatched config).

## Testing

### Unit (fast tier)

- **NPMI math.** Tiny hand-built corpus (5 docs × 4 terms), pre-computed expected co-occurrence and expected NPMI per pair. Verify the pair-aggregation step.
- **Zero co-occurrence convention.** Pair that never co-occurs returns NPMI = -1, not NaN/Inf.
- **`top_k_used_topics`.** Synthetic (u, v) with hand-computed expected `E[β]` ordering.
- **Top-N selection.** Topic-term row with known argpartition output; verify ties broken deterministically.
- **Split determinism.** Same seed → identical buckets. Different seeds → different buckets. Train ∩ holdout = ∅. Union covers the input.

### Integration (slow tier)

- **Driver smoke.** `analysis/local/eval_coherence.py` runs to completion on a fixture checkpoint and emits a non-empty ranked table. Also serves as the property-check vehicle: asserts all NPMI values in `[-1, 1]`, `per_topic_npmi` length matches mask sum (HDP) or K (LDA), and summary stats match `numpy` equivalents computed directly from `per_topic_npmi`.

## Migration / Compatibility

- The `models.topic` refactor is breaking for any external code importing `spark_vi.models.lda` directly. Mitigation: ship the refactor as its own PR with a clear ADR and announce in `docs/REVIEW_LOG.md`. No deprecation shim — we're early enough that "break and rename" is cheaper than carrying a redirect.
- The eval module is purely additive; no compatibility surface.
- The split helper is additive; no existing charmpheno code depends on it.

## ADR

ADR 0017 lands with this branch and records:

- Choice of NPMI over modified-UCI (bounded, established, no smoothing constant).
- Whole-document co-occurrence over sliding-window (justified by patient-bag semantics).
- Layering: generic metric in spark-vi, domain split in charmpheno, orchestration in analysis.
- Why splitting is a driver-layer concern (MLlib idiom; estimators don't split).
- The driver-side contract that holdout config must match training; v2 hardening via VIResult.metadata stamping.
- Deferral of term-relevance, pyLDAvis, gensim adapter, OCTIS, simulations.

## Open Questions (carrying into the plan)

None — the brainstorming session settled all major forks. Two minor items the implementation plan will need to nail down but don't require user input:

- Exact dtype for `per_topic_npmi` (np.float64) and `top_term_indices` (np.int32).
- Concrete bucket count for the SHA-256-modulo split (10000 is a starting proposal; the plan can re-examine).

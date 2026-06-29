# STM Hardening Cluster-Threading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thread the three opt-in `OnlineSTM` hardening knobs — `reference_topic`, `sigma_prior_scale`, `sigma_prior_count` — through the `StreamingSTM` mllib estimator, its persisted metadata, and both fit drivers, so the cluster fit path can actually use them.

**Architecture:** All three are already validated opt-in flags on `OnlineSTM` (insight 0029, ADR 0031 for the reference; ADR for the Σ prior). They are pure config flags — no new computation. `StreamingSTM.fit` constructs `OnlineSTM` but currently omits all three, so they are reachable only via the direct `OnlineSTM` API + the local ablation. This plan adds them at three layers: the estimator constructor + `OnlineSTM` construction in `fit`, the persisted metadata (provenance round-trip), and the two driver CLIs (`stm_bigquery_cloud.py`, `fit_stm_local.py`).

**Tech Stack:** Python, NumPy, PySpark (estimator + drivers), pytest.

## Scope

**In scope — the three config-flag knobs:** `reference_topic: bool`, `sigma_prior_scale: float | None`, `sigma_prior_count: float`. They share identical wiring (constructor param → `OnlineSTM` kwarg → metadata → CLI arg), so they are one coherent job, and the validated best configuration (spectral+reference+Σ-prior, insight 0029 Ablation 2) needs the Σ-prior pair alongside the reference. `sigma_prior_scale`/`sigma_prior_count` were left unthreaded by the earlier hardening arc — this closes that gap too.

**Out of scope — spectral initialization (separate SCALABLE-REDESIGN arc, not a deferred minor).** The current `spectral_init.py` materializes a dense V×V co-occurrence matrix on the driver — O(V²) memory: ~80 GB at V=100k, untenable past ~V=10–20k. spark-vi targets very large datasets (V up to 10⁵–10⁶, K up to ~10³), so wiring the prototype to the cluster as-is would just move an 80 GB allocation onto the driver. Making spectral init scale requires the standard Arora et al. 2013 scalable form — distributed co-occurrence + random projection of word rows to V×d (d ≈ max(K, ε⁻²·log V), O(V·K) memory ≈ ~1 GB at V=100k/K=1000) + distributed NNLS recovery — never materializing V×V. The `min_marginal_frac` decision (→ absolute document-frequency floor, computed in the distributed pass) folds into that arc. It deserves its own brainstorm → spec → plan. Threading the config flags first lets the GATED rare-arm runs (which recover with reference alone — insight 0029 Ablation 2) proceed on the cluster without it.

## File Structure

- `spark-vi/spark_vi/mllib/topic/stm.py` — `StreamingSTM.__init__` (add 3 params + store), `StreamingSTM.fit` (pass to `OnlineSTM`; persist in metadata). `STMModel` needs no change: it has no `OnlineSTM`-reconstructing inference path (unlike HDP), and the dashboard prevalence helpers use softmax(Γᵀx), which is already correct under a reference fit because Γ[:, 0] = 0. Metadata persistence is provenance + future-proofing.
- `analysis/cloud/stm_bigquery_cloud.py` — `parse_args` (make argv-injectable + 3 new flags), `main` (pass to `StreamingSTM`).
- `analysis/local/fit_stm_local.py` — parser (3 new flags), `StreamingSTM` construction.
- `scripts/run_experiment.py` — `build_stm_args` maps experiment frontmatter → cloud-driver flags; add the three so `make exp ID=N` can drive them from an experiment doc's frontmatter (`reference_topic`, `sigma_prior_scale`, `sigma_prior_count`). This is what makes exp 0012–0014 runnable.
- Tests: `spark-vi/tests/test_mllib_stm.py`, `spark-vi/tests/test_mllib_stm_persistence.py`, `analysis/cloud/tests/test_stm_driver_partition.py`, `tests/scripts/test_fit_stm_local.py`, `scripts/tests/test_run_experiment.py`.

## Global Constraints

- **Domain-agnostic engine:** `spark-vi` never sees OMOP concept ids/names; the estimator change stays integer-token / covariate-vector only. (Driver code in `analysis/` is allowed to touch OMOP — that boundary is unchanged.)
- **Opt-in, default-off, byte-identical default path:** all three default to off (`reference_topic=False`, `sigma_prior_scale=None`, `sigma_prior_count=0.0`); with defaults, the constructed `OnlineSTM` and the fit are numerically identical to today. The full existing test suite must stay green.
- **No LaTeX in prose/docstrings/comments:** plain text + Unicode Greek (η, Σ, Γ, λ).
- **Markdown-linkable code refs** in any prose.
- **TDD throughout:** failing test first, watch it fail, minimal code, watch it pass, commit.
- **Exact default values:** `reference_topic=False`, `sigma_prior_scale=None`, `sigma_prior_count=0.0` — matching `OnlineSTM`'s own defaults verbatim ([stm.py:295-296](spark-vi/spark_vi/models/topic/stm.py#L295-L296), [stm.py:301](spark-vi/spark_vi/models/topic/stm.py#L301)).

---

## Task 1: Thread the knobs through StreamingSTM (estimator → OnlineSTM → metadata)

**Files:**
- Modify: `spark-vi/spark_vi/mllib/topic/stm.py` — `StreamingSTM.__init__` ([stm.py:122-175](spark-vi/spark_vi/mllib/topic/stm.py#L122-L175)), `StreamingSTM.fit` `OnlineSTM` construction ([stm.py:260-270](spark-vi/spark_vi/mllib/topic/stm.py#L260-L270)) and metadata block ([stm.py:315-317](spark-vi/spark_vi/mllib/topic/stm.py#L315-L317)).
- Test: `spark-vi/tests/test_mllib_stm.py` (extend), `spark-vi/tests/test_mllib_stm_persistence.py` (extend).

**Interfaces:**
- Consumes: `OnlineSTM(..., sigma_prior_scale, sigma_prior_count, reference_topic)` (already exists on the engine).
- Produces:
  - `StreamingSTM.__init__(..., sigma_prior_scale: float | None = None, sigma_prior_count: float = 0.0, reference_topic: bool = False)` storing `self.sigma_prior_scale`, `self.sigma_prior_count`, `self.reference_topic`.
  - `StreamingSTM.fit` passes all three to `OnlineSTM` and writes `metadata["stm_hardening"] = {"reference_topic": bool, "sigma_prior_scale": float|None, "sigma_prior_count": float}`.

- [ ] **Step 1: Write the failing tests**

Append to `spark-vi/tests/test_mllib_stm.py` (inside the file, after the existing `TestStreamingSTMPathA` class):

```python
class TestStreamingSTMHardeningThreading:
    """The opt-in OnlineSTM hardening knobs reach the engine and the metadata."""

    def _toy_df(self, spark):
        from pyspark.ml.linalg import SparseVector, DenseVector
        rows = [
            (SparseVector(8, [0, 2], [3.0, 1.0]), DenseVector([1.0, 0.0])),
            (SparseVector(8, [1, 3], [2.0, 2.0]), DenseVector([0.0, 1.0])),
            (SparseVector(8, [0, 4], [1.0, 2.0]), DenseVector([1.0, 0.5])),
            (SparseVector(8, [5, 6], [1.0, 1.0]), DenseVector([0.0, 1.0])),
            (SparseVector(8, [2, 7], [2.0, 1.0]), DenseVector([1.0, 0.2])),
            (SparseVector(8, [1, 6], [1.0, 3.0]), DenseVector([0.0, 0.8])),
        ]
        return spark.createDataFrame(rows, ["features", "covariates"])

    def test_reference_topic_reaches_engine(self, spark):
        """A reference fit drives the reference topic's Gamma column to 0 — the
        end-to-end signature that reference_topic took effect through the shim."""
        from spark_vi.mllib.topic.stm import StreamingSTM
        est = StreamingSTM(
            K=4, features_col="features", covariates_col="covariates",
            covariate_names=["a", "b"], random_seed=0, reference_topic=True)
        model = est.fit(self._toy_df(spark), max_iter=3, subsampling_rate=1.0)
        assert np.allclose(model.global_params["Gamma"][:, 0], 0.0)

    def test_hardening_knobs_persisted_in_metadata(self, spark):
        from spark_vi.mllib.topic.stm import StreamingSTM
        est = StreamingSTM(
            K=4, features_col="features", covariates_col="covariates",
            covariate_names=["a", "b"], random_seed=0,
            reference_topic=True, sigma_prior_scale=2.0, sigma_prior_count=500.0)
        model = est.fit(self._toy_df(spark), max_iter=2, subsampling_rate=1.0)
        assert model.metadata["stm_hardening"] == {
            "reference_topic": True,
            "sigma_prior_scale": 2.0,
            "sigma_prior_count": 500.0,
        }

    def test_defaults_off_and_recorded(self, spark):
        from spark_vi.mllib.topic.stm import StreamingSTM
        est = StreamingSTM(
            K=4, features_col="features", covariates_col="covariates",
            covariate_names=["a", "b"], random_seed=0)
        assert est.reference_topic is False
        assert est.sigma_prior_scale is None
        assert est.sigma_prior_count == 0.0
        model = est.fit(self._toy_df(spark), max_iter=2, subsampling_rate=1.0)
        assert model.metadata["stm_hardening"] == {
            "reference_topic": False,
            "sigma_prior_scale": None,
            "sigma_prior_count": 0.0,
        }
        # Default fit does NOT force the reference column to zero.
        assert not np.allclose(model.global_params["Gamma"][:, 0], 0.0)
```

Append to `spark-vi/tests/test_mllib_stm_persistence.py` (inside `TestSTMModelPersistence`):

```python
    def test_stm_hardening_metadata_roundtrips(self, tmp_path: Path):
        from spark_vi.mllib.topic.stm import STMModel
        from spark_vi.models.topic.stm import OnlineSTM

        model = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp = model.initialize_global(None)
        stm_model = STMModel(
            global_params=gp,
            metadata={"K": 3, "V": 10, "P": 2, "stm_hardening": {
                "reference_topic": True,
                "sigma_prior_scale": 2.0,
                "sigma_prior_count": 500.0,
            }},
            model_spec=_FakeSpec(),
            covariate_names=["intercept", "cohort_b"],
        )
        out_dir = tmp_path / "stm_model_hardening"
        stm_model.save(out_dir)
        loaded = STMModel.load(out_dir)
        assert loaded.metadata["stm_hardening"] == {
            "reference_topic": True,
            "sigma_prior_scale": 2.0,
            "sigma_prior_count": 500.0,
        }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_mllib_stm.py::TestStreamingSTMHardeningThreading tests/test_mllib_stm_persistence.py::TestSTMModelPersistence::test_stm_hardening_metadata_roundtrips -v`
Expected: FAIL — `StreamingSTM.__init__` has no `reference_topic`/`sigma_prior_*` kwargs (TypeError), and `model.metadata` has no `"stm_hardening"` key (KeyError). (The persistence test fails on the missing key only after save/load; it will actually pass once the key is simply carried through metadata, so its red state is the KeyError before Step 3 adds nothing to persistence — persistence is automatic. If it passes already, that confirms the round-trip needs no STMModel change; keep it as a regression guard.)

- [ ] **Step 3: Add the three constructor params + storage**

In `StreamingSTM.__init__` signature ([stm.py:132-138](spark-vi/spark_vi/mllib/topic/stm.py#L132-L138)), add the Σ-prior pair after `sigma_ridge` and `reference_topic` after `random_seed`:

```python
        sigma_init: float = 1.0,
        sigma_ridge: float = 1e-6,
        sigma_prior_scale: float | None = None,
        sigma_prior_count: float = 0.0,
        lbfgs_max_iter: int = 50,
        lbfgs_tol: float = 1e-4,
        random_seed: int | None = None,
        reference_topic: bool = False,
        topic_blocks=None,
        doc_group_col: str | None = None,
    ) -> None:
```

Store them alongside the other hyperparameters (after `self.sigma_ridge = sigma_ridge` at [stm.py:172](spark-vi/spark_vi/mllib/topic/stm.py#L172) and after `self.random_seed = random_seed` at [stm.py:175](spark-vi/spark_vi/mllib/topic/stm.py#L175)):

```python
        self.sigma_init = sigma_init
        self.sigma_ridge = sigma_ridge
        self.sigma_prior_scale = sigma_prior_scale
        self.sigma_prior_count = sigma_prior_count
        self.lbfgs_max_iter = lbfgs_max_iter
        self.lbfgs_tol = lbfgs_tol
        self.random_seed = random_seed
        self.reference_topic = bool(reference_topic)
```

- [ ] **Step 4: Pass the three to OnlineSTM in fit()**

In `StreamingSTM.fit`, extend the `OnlineSTM(...)` construction ([stm.py:260-270](spark-vi/spark_vi/mllib/topic/stm.py#L260-L270)):

```python
        model = OnlineSTM(
            K=self.K,
            vocab_size=vocab_size,
            P=self.P,
            sigma_init=self.sigma_init,
            sigma_ridge=self.sigma_ridge,
            sigma_prior_scale=self.sigma_prior_scale,
            sigma_prior_count=self.sigma_prior_count,
            lbfgs_max_iter=self.lbfgs_max_iter,
            lbfgs_tol=self.lbfgs_tol,
            random_seed=self.random_seed,
            topic_blocks=self.topic_blocks,
            reference_topic=self.reference_topic,
        )
```

- [ ] **Step 5: Persist the knobs in metadata**

Extend the metadata block in `fit` ([stm.py:315-317](spark-vi/spark_vi/mllib/topic/stm.py#L315-L317)):

```python
        metadata = dict(result.metadata)
        if self.topic_blocks is not None:
            metadata.setdefault("topic_block_spec", self.topic_blocks.to_dict())
        # Provenance: record the opt-in hardening knobs that produced this fit.
        # Not load-bearing for the current export path (the dashboard prevalence
        # helpers use softmax(Gamma^T x), already correct under a reference fit
        # because Gamma[:, 0] = 0); persisted so a reloaded model's provenance is
        # complete and a future inference path can re-pin.
        metadata.setdefault("stm_hardening", {
            "reference_topic": self.reference_topic,
            "sigma_prior_scale": self.sigma_prior_scale,
            "sigma_prior_count": self.sigma_prior_count,
        })
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_mllib_stm.py::TestStreamingSTMHardeningThreading tests/test_mllib_stm_persistence.py -v`
Expected: PASS.

- [ ] **Step 7: Confirm the full mllib STM suite is green (default path unperturbed)**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_mllib_stm.py tests/test_mllib_stm_persistence.py tests/test_mllib_stm_formula.py -v`
Expected: PASS — the defaults match `OnlineSTM`'s, so existing fits are unchanged. If any test asserts exact metadata-dict equality on a `.fit()`-produced model and now sees the extra `stm_hardening` key, update that assertion to expect the new key (it is correct that fits now carry provenance).

- [ ] **Step 8: Commit**

```bash
git add spark-vi/spark_vi/mllib/topic/stm.py spark-vi/tests/test_mllib_stm.py spark-vi/tests/test_mllib_stm_persistence.py
git commit -m "feat(mllib): thread reference_topic + sigma-prior knobs through StreamingSTM"
```

---

## Task 2: Thread the knobs through the cloud driver

**Files:**
- Modify: `analysis/cloud/stm_bigquery_cloud.py` — `parse_args` ([stm_bigquery_cloud.py:147-234](analysis/cloud/stm_bigquery_cloud.py#L147-L234)) and the `StreamingSTM(...)` construction in `main` ([stm_bigquery_cloud.py:381-393](analysis/cloud/stm_bigquery_cloud.py#L381-L393)).
- Test: `analysis/cloud/tests/test_stm_driver_partition.py` (extend).

**Interfaces:**
- Consumes: `StreamingSTM(..., reference_topic, sigma_prior_scale, sigma_prior_count)` from Task 1.
- Produces: `parse_args(argv=None)` (now argv-injectable) exposing `--reference-topic` (store_true → `args.reference_topic`), `--sigma-prior-scale` (`float | None`, default None → `args.sigma_prior_scale`), `--sigma-prior-count` (`float`, default 0.0 → `args.sigma_prior_count`); `main` forwards all three to `StreamingSTM`.

- [ ] **Step 1: Write the failing test**

Append to `analysis/cloud/tests/test_stm_driver_partition.py`:

```python
def test_parse_args_hardening_flags_default_off():
    from stm_bigquery_cloud import parse_args
    args = parse_args(["--cdr", "p.d", "--billing", "b"])
    assert args.reference_topic is False
    assert args.sigma_prior_scale is None
    assert args.sigma_prior_count == 0.0


def test_parse_args_hardening_flags_set():
    from stm_bigquery_cloud import parse_args
    args = parse_args([
        "--cdr", "p.d", "--billing", "b",
        "--reference-topic",
        "--sigma-prior-scale", "2.0",
        "--sigma-prior-count", "500.0",
    ])
    assert args.reference_topic is True
    assert args.sigma_prior_scale == 2.0
    assert args.sigma_prior_count == 500.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && .venv/bin/python -m pytest analysis/cloud/tests/test_stm_driver_partition.py -v`
Expected: FAIL — `parse_args` takes no argv (TypeError: takes 0 positional arguments) and has no `--reference-topic` flag.

- [ ] **Step 3: Make parse_args argv-injectable**

Change the signature ([stm_bigquery_cloud.py:147](analysis/cloud/stm_bigquery_cloud.py#L147)) and the return ([stm_bigquery_cloud.py:234](analysis/cloud/stm_bigquery_cloud.py#L234)):

```python
def parse_args(argv=None) -> argparse.Namespace:
```

```python
    return p.parse_args(argv)
```

(`main` calls `parse_args()` with no argument → `argv=None` → reads `sys.argv`, unchanged behavior.)

- [ ] **Step 4: Add the three CLI flags**

In `parse_args`, after the existing `--sigma-ridge` flag ([stm_bigquery_cloud.py:197](analysis/cloud/stm_bigquery_cloud.py#L197)), add:

```python
    p.add_argument("--sigma-prior-scale", type=float, default=None,
                   help="Inverse-gamma Sigma-prior scale s0 (off when unset). "
                        "Shrinks the per-topic logistic-normal variance toward s0.")
    p.add_argument("--sigma-prior-count", type=float, default=0.0,
                   help="Inverse-gamma Sigma-prior pseudo-count c0 (default 0).")
    p.add_argument("--reference-topic", action="store_true",
                   help="Pin topic 0's eta to 0 (K-1 reference parameterization, "
                        "ADR 0031). Removes the softmax translation degeneracy.")
```

- [ ] **Step 5: Forward them to StreamingSTM**

Extend the `StreamingSTM(...)` construction in `main` ([stm_bigquery_cloud.py:381-393](analysis/cloud/stm_bigquery_cloud.py#L381-L393)):

```python
            est = StreamingSTM(
                K=args.K,
                features_col="features",
                covariates_col="covariates",
                covariate_names=covariate_names,
                sigma_init=args.sigma_init,
                sigma_ridge=args.sigma_ridge,
                sigma_prior_scale=args.sigma_prior_scale,
                sigma_prior_count=args.sigma_prior_count,
                lbfgs_max_iter=args.lbfgs_max_iter,
                lbfgs_tol=args.lbfgs_tol,
                random_seed=args.random_seed,
                reference_topic=args.reference_topic,
                topic_blocks=partition,
                doc_group_col=(args.group_var if partition is not None else None),
            )
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && .venv/bin/python -m pytest analysis/cloud/tests/test_stm_driver_partition.py -v`
Expected: PASS (existing partition tests + 2 new arg tests).

- [ ] **Step 7: Commit**

```bash
git add analysis/cloud/stm_bigquery_cloud.py analysis/cloud/tests/test_stm_driver_partition.py
git commit -m "feat(cloud): --reference-topic + --sigma-prior-* flags on STM driver"
```

---

## Task 3: Thread the knobs through the local driver

**Files:**
- Modify: `analysis/local/fit_stm_local.py` — parser ([fit_stm_local.py:113-130](analysis/local/fit_stm_local.py#L113-L130)) and the `StreamingSTM(...)` construction ([fit_stm_local.py:161-164](analysis/local/fit_stm_local.py#L161-L164)).
- Test: `tests/scripts/test_fit_stm_local.py` (extend).

**Interfaces:**
- Consumes: `StreamingSTM(..., reference_topic, sigma_prior_scale, sigma_prior_count)` from Task 1; the `main(argv)` end-to-end driver harness.
- Produces: `--reference-topic`, `--sigma-prior-scale`, `--sigma-prior-count` CLI flags forwarded to `StreamingSTM`; a reference run's saved `params/Gamma.npy` has column 0 all zero and `manifest.json` metadata carries `stm_hardening`.

- [ ] **Step 1: Write the failing test**

Append to `tests/scripts/test_fit_stm_local.py`:

```python
def test_fit_stm_local_reference_topic_end_to_end(tmp_path):
    """--reference-topic threads through to the engine: the saved Gamma has its
    reference column zeroed and the metadata records the hardening config."""
    import json
    import numpy as np
    from fit_stm_local import main as fit_main
    omop, person = _make_sim(tmp_path)
    out = tmp_path / "ckpt_ref"
    rc = fit_main([
        "--omop", str(omop), "--person", str(person),
        "--K", "5", "--background-k", "3", "--foreground", "rare_dx:2",
        "--covariate-formula", "~ C(sex) + age",
        "--reference-topic",
        "--sigma-prior-scale", "2.0", "--sigma-prior-count", "500.0",
        "--max-iter", "8", "--out-dir", str(out)])
    assert rc == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["metadata"]["stm_hardening"] == {
        "reference_topic": True,
        "sigma_prior_scale": 2.0,
        "sigma_prior_count": 500.0,
    }
    Gamma = np.load(out / "params" / "Gamma.npy")
    assert np.allclose(Gamma[:, 0], 0.0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && .venv/bin/python -m pytest tests/scripts/test_fit_stm_local.py::test_fit_stm_local_reference_topic_end_to_end -v`
Expected: FAIL — `fit_stm_local` has no `--reference-topic` flag (SystemExit from argparse on unrecognized argument).

- [ ] **Step 3: Add the three CLI flags**

In `main`'s parser, after `p.add_argument("--seed", ...)` ([fit_stm_local.py:128](analysis/local/fit_stm_local.py#L128)), add:

```python
    p.add_argument("--sigma-prior-scale", type=float, default=None,
                   help="Inverse-gamma Sigma-prior scale s0 (off when unset).")
    p.add_argument("--sigma-prior-count", type=float, default=0.0,
                   help="Inverse-gamma Sigma-prior pseudo-count c0 (default 0).")
    p.add_argument("--reference-topic", action="store_true",
                   help="Pin topic 0's eta to 0 (K-1 reference param, ADR 0031).")
```

- [ ] **Step 4: Forward them to StreamingSTM**

Extend the `StreamingSTM(...)` construction ([fit_stm_local.py:161-164](analysis/local/fit_stm_local.py#L161-L164)):

```python
        est = StreamingSTM(
            K=args.K, features_col="features", covariates_col="covariates",
            covariate_names=covariate_names, topic_blocks=partition,
            doc_group_col="source_cohort", random_seed=args.seed,
            sigma_prior_scale=args.sigma_prior_scale,
            sigma_prior_count=args.sigma_prior_count,
            reference_topic=args.reference_topic)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && .venv/bin/python -m pytest tests/scripts/test_fit_stm_local.py -v`
Expected: PASS (existing local-driver tests + the new reference end-to-end test).

- [ ] **Step 6: Commit**

```bash
git add analysis/local/fit_stm_local.py tests/scripts/test_fit_stm_local.py
git commit -m "feat(local): --reference-topic + --sigma-prior-* flags on STM driver"
```

---

## Task 4: Thread the knobs through the experiment runner (build_stm_args)

**Files:**
- Modify: `scripts/run_experiment.py` — `build_stm_args` ([run_experiment.py:433-478](scripts/run_experiment.py#L433-L478)).
- Test: `scripts/tests/test_run_experiment.py` (extend).

**Interfaces:**
- Consumes: the cloud driver's `--reference-topic` / `--sigma-prior-scale` / `--sigma-prior-count` flags from Task 2.
- Produces: `build_stm_args` appends `--reference-topic` when `effective.get("reference_topic")` is truthy, `--sigma-prior-scale <v>` when `effective.get("sigma_prior_scale") is not None`, and `--sigma-prior-count <v>` when `effective.get("sigma_prior_count")` is truthy. Absent frontmatter fields → flags omitted (driver defaults apply).

- [ ] **Step 1: Write the failing test**

Append to `scripts/tests/test_run_experiment.py`:

```python
def test_build_stm_args_threads_hardening_flags(monkeypatch):
    import run_experiment
    monkeypatch.setattr(run_experiment, "_require_workspace_env",
                        lambda: ("proj.ds", "billing"))
    effective = {
        "source_table": "condition_era", "doc_unit": "patient",
        "doc_min_length": 1, "K": 40, "max_iter": 300, "vocab_size": 3000,
        "min_df": 5, "min_patient_count": 20, "subsampling_rate": 1.0,
        "tau0": 64.0, "kappa": 0.7, "save_interval": 50, "person_mod": 4,
        "covariate_formula": "~ C(sex) + age", "categorical_cols": ["sex"],
        "continuous_cols": ["age"],
        "reference_topic": True,
        "sigma_prior_scale": 2.0, "sigma_prior_count": 500.0,
    }
    args = run_experiment.build_stm_args(effective, out_dir="/tmp/out")
    assert "--reference-topic" in args
    i = args.index("--sigma-prior-scale"); assert args[i + 1] == "2.0"
    j = args.index("--sigma-prior-count"); assert args[j + 1] == "500.0"


def test_build_stm_args_hardening_flags_omitted_by_default(monkeypatch):
    import run_experiment
    monkeypatch.setattr(run_experiment, "_require_workspace_env",
                        lambda: ("proj.ds", "billing"))
    effective = {
        "source_table": "condition_era", "doc_unit": "patient",
        "doc_min_length": 1, "K": 40, "max_iter": 300, "vocab_size": 3000,
        "min_df": 5, "min_patient_count": 20, "subsampling_rate": 1.0,
        "tau0": 64.0, "kappa": 0.7, "save_interval": 50, "person_mod": 4,
        "covariate_formula": "~ C(sex) + age", "categorical_cols": ["sex"],
        "continuous_cols": ["age"],
    }
    args = run_experiment.build_stm_args(effective, out_dir="/tmp/out")
    assert "--reference-topic" not in args
    assert "--sigma-prior-scale" not in args
    assert "--sigma-prior-count" not in args
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && .venv/bin/python -m pytest scripts/tests/test_run_experiment.py::test_build_stm_args_threads_hardening_flags -v`
Expected: FAIL — `build_stm_args` emits none of the three flags, so `"--reference-topic" in args` is False (AssertionError).

- [ ] **Step 3: Append the three flags in build_stm_args**

In `build_stm_args`, build a `hardening` list just before the final `return` ([run_experiment.py:470-478](scripts/run_experiment.py#L470-L478)) and append it:

```python
    hardening: list[str] = []
    if effective.get("reference_topic"):
        hardening.append("--reference-topic")
    if effective.get("sigma_prior_scale") is not None:
        hardening.extend(["--sigma-prior-scale", str(effective["sigma_prior_scale"])])
    if effective.get("sigma_prior_count"):
        hardening.extend(["--sigma-prior-count", str(effective["sigma_prior_count"])])
    return common + [
        "--covariate-formula", str(effective["covariate_formula"]),
        "--categorical-cols", ",".join(effective.get("categorical_cols", [])),
        "--continuous-cols", ",".join(effective.get("continuous_cols", [])),
        "--sigma-init", str(effective.get("sigma_init", 1.0)),
        "--sigma-ridge", str(effective.get("sigma_ridge", 1e-6)),
        "--lbfgs-max-iter", str(effective.get("lbfgs_max_iter", 50)),
        "--lbfgs-tol", str(effective.get("lbfgs_tol", 1e-4)),
    ] + hardening + gating + (["--resume-from", str(resume_from)] if resume_from is not None else [])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && .venv/bin/python -m pytest scripts/tests/test_run_experiment.py -v`
Expected: PASS (existing runner tests + the 2 new ones).

- [ ] **Step 5: Commit**

```bash
git add scripts/run_experiment.py scripts/tests/test_run_experiment.py
git commit -m "feat(run-exp): thread reference_topic + sigma-prior from STM frontmatter"
```

---

## Self-Review

**Spec coverage:**
- `reference_topic` through estimator + `OnlineSTM` + metadata → Task 1; cloud CLI → Task 2; local CLI → Task 3.
- `sigma_prior_scale`/`sigma_prior_count` (the unthreaded siblings) → same three tasks, same lines.
- Metadata provenance round-trip → Task 1 (persistence test).
- Default-off byte-identical → Task 1 `test_defaults_off_and_recorded` + the existing suites in Steps 7 (Task 1), 6 (Task 2), 5 (Task 3).
- Spectral init → explicitly OUT of scope, documented in the Scope section as a follow-on arc.

**Placeholder scan:** every step shows complete code (signatures, flag definitions, construction blocks, test bodies). No TBD/handle-edge-cases. The only conditional note (Task 1 Step 7) names the exact failure mode and the fix.

**Type consistency:** `reference_topic: bool` (default False), `sigma_prior_scale: float | None` (default None), `sigma_prior_count: float` (default 0.0) — identical names/types/defaults across `OnlineSTM`, `StreamingSTM`, both CLIs, and the `stm_hardening` metadata dict. CLI flags map `--reference-topic`→`reference_topic`, `--sigma-prior-scale`→`sigma_prior_scale`, `--sigma-prior-count`→`sigma_prior_count` (argparse's standard dash-to-underscore).

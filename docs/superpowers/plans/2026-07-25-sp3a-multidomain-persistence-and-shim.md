# SP3a — Multi-domain persistence + mllib shim — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a multi-domain gated LDA fit persistable and deployable — a per-domain dict λ can be written, loaded, and served through the MLlib shim with the fitted ω applied — so a cloud fit produces an artifact the export path can consume.

**Architecture:** Three seams, each independently testable. (1) `spark_vi/io/export.py` learns to write a dict-valued global param as one `.npy` per key with the keys recorded in the manifest. (2) `GatedOnlineLDA.get_metadata` reports `domains`/`omega`/`eta_m` so a saved result is reconstructable. (3) The shim accepts *separate per-domain feature columns*, derives the domain layout once from the first row, validates every row against it, and concatenates into the engine's existing single concatenated id space — the engine's representation does not change.

**Tech Stack:** Python, numpy, scipy, pyspark (MLlib `Estimator`/`Model`), pytest. Spark-local for the integration gates.

**Spec:** `docs/superpowers/specs/2026-07-25-sp3a-multidomain-persistence-and-shim-design.md`

## Global Constraints

- **Domain-neutral naming is binding in `spark_vi/**` and `spark-vi/tests/**`:** integer token ids and domain sizes only. NO clinical / OMOP / EHR / medical vocabulary in code, comments, docstrings or tests. Domains are `0`/`1`/`m` (or `a`/`b`), never real-world roles. (Clinical semantics belong to `charmpheno` — SP3b.)
- **Do NOT opportunistically fix** the known pre-existing naming violation in `dag_placement.py` (`disease_mass`, `auc_disease_mass`, `ap_disease_mass` are returned dict keys, so renaming breaks consumers). It needs its own scoped decision.
- **Backward compatibility is hard:** `domains=None` byte-identical; base `OnlineLDA`, vanilla LDA and HDP untouched in behavior; `omega=None` identity; `featuresCols` unset keeps the existing single-`featuresCol` path byte-identical; `load_result` still reads `format_version` 1 archives.
- **No LaTeX.** Unicode Greek (α β θ Σ η λ ω ρ) or the file's existing ASCII spellings — match the surrounding file.
- **Cite literature** in docstrings for any method, default or constant. MixEHR (Li, Nair, Lu et al. 2020, Nat. Commun.) for the per-modality model; Hoffman, Blei & Bach 2010 for online VB; Arora et al. 2013 for anchor recovery; ADR 0032 for the scalable sketch.
- **Any planted corpus passes `bg_frac` > 0 and `ancestor_signature_decay` < 1** (insights 0067, 0068). A background-starved or label-unidentifiable plant produces convincing false negatives — two were escalated as engine defects during SP2 before being traced to the plant.
- **If short of a gate, STRENGTHEN THE PLANT, NEVER loosen the assertion; a strong-plant failure is a genuine negative — STOP and report.** (Verbatim from the SP2 plan; it surfaced insights 0067 and 0068.)
- **Every task that adds a gate must name one mutation of the code under test and show the assertion fires.** SP2 spent four fix rounds on gates that passed for reasons unrelated to their deliverable. Put the mutation and its output in the task report; revert the mutation, never commit it.
- **The Bash tool's `timeout` is in MILLISECONDS, max 600000, default 120000.** A longer foreground command is auto-backgrounded and one SP2 agent lost its work waiting on one. Pass it explicitly; never wait on a backgrounded run. Everything foreground.
- Run tests from `/Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno/spark-vi` as a single `cd ... && python -m pytest ...` per bash call (shell cwd does not persist reliably). There is no `timeout` binary on this machine.
- Commit trailer EXACTLY as the last line of every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  ```

## File Structure

- `spark-vi/spark_vi/io/export.py` — MODIFY. Dict-valued global params: one `params/<name>_<k>.npy` per key, keys in the manifest, `format_version` 2. The existing `UnsupportedGlobalParamError` guard stays for values that are genuinely unwritable.
- `spark-vi/tests/test_export.py` — MODIFY. Dict round-trip, int-key preservation, v1 back-compat, guard still fires inside a dict.
- `spark-vi/spark_vi/models/topic/gated_lda.py` — MODIFY. `get_metadata` override; docstring linking the η-provenance invariant to the existing `optimize_eta` rejection.
- `spark-vi/spark_vi/mllib/topic/gated_lda.py` — MODIFY. `featuresCols` / `omega` / `etaPerDomain` / `domainBounds` Params; layout derivation + per-row validation; a shared per-domain concatenation helper used by both `_fit` and `_transform`; dict-λ `_transform` applying ω.
- `spark-vi/tests/test_gated_lda.py` — MODIFY. `get_metadata`, η-provenance pinning, the multi-seed scalable-vs-dense recovery gate.
- `spark-vi/tests/test_gated_lda_shim.py` — MODIFY. Multi-domain shim fit, row validation, save→load→identical `nodeAffinity`.

---

### Task 1: Dict-valued global params in the export format

**Files:**
- Modify: `spark-vi/spark_vi/io/export.py`
- Test: `spark-vi/tests/test_export.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `save_result` / `load_result` round-trip a `global_params` value that is a `dict[int, np.ndarray]`. Manifest gains `"dict_param_keys": {param_name: [int, ...]}`. `_FORMAT_VERSION` becomes `2`; `load_result` accepts 1 and 2. Later tasks rely on `load_result` returning `global_params["lambda"]` as a dict with **int** keys in domain order.

- [ ] **Step 1: Write the failing tests**

Add to `spark-vi/tests/test_export.py`:

```python
def test_dict_param_round_trips_per_key(tmp_path):
    """A dict-valued global param (the multi-domain per-domain lambda) writes one
    .npy per key and loads back as a dict with INT keys in domain order."""
    import numpy as np
    from spark_vi.core.result import VIResult
    from spark_vi.io.export import load_result, save_result
    lam = {0: np.arange(6.0).reshape(2, 3), 1: np.arange(4.0).reshape(2, 2)}
    res = VIResult(global_params={"lambda": lam, "alpha": np.array([0.1, 0.2])},
                   elbo_trace=[-1.0], n_iterations=1, converged=False,
                   metadata={"model_class": "GatedOnlineLDA"})
    save_result(res, tmp_path / "r")
    assert (tmp_path / "r" / "params" / "lambda_0.npy").exists()
    assert (tmp_path / "r" / "params" / "lambda_1.npy").exists()
    assert not (tmp_path / "r" / "params" / "lambda.npy").exists()
    back = load_result(tmp_path / "r")
    got = back.global_params["lambda"]
    assert isinstance(got, dict)
    assert list(got) == [0, 1]                     # int keys, domain order
    assert all(isinstance(k, int) for k in got)
    np.testing.assert_array_equal(got[0], lam[0])
    np.testing.assert_array_equal(got[1], lam[1])
    np.testing.assert_array_equal(back.global_params["alpha"], res.global_params["alpha"])


def test_dict_param_keys_recorded_in_manifest(tmp_path):
    """The manifest records the dict param's keys so load does not have to glob."""
    import json
    import numpy as np
    from spark_vi.core.result import VIResult
    from spark_vi.io.export import save_result
    res = VIResult(global_params={"lambda": {0: np.ones((2, 3)), 1: np.ones((2, 2))}},
                   elbo_trace=[], n_iterations=0, converged=False)
    save_result(res, tmp_path / "r")
    manifest = json.loads((tmp_path / "r" / "manifest.json").read_text())
    assert manifest["format_version"] == 2
    assert manifest["dict_param_keys"] == {"lambda": [0, 1]}
    assert manifest["param_names"] == ["lambda"]


def test_format_version_1_archive_still_loads(tmp_path):
    """A v1 archive has no dict_param_keys; its single-array lambda must still load."""
    import json
    import numpy as np
    out = tmp_path / "v1"
    (out / "params").mkdir(parents=True)
    np.save(out / "params" / "lambda.npy", np.ones((2, 5)))
    (out / "manifest.json").write_text(json.dumps({
        "format_version": 1, "elbo_trace": [-2.0], "n_iterations": 1,
        "converged": True, "metadata": {}, "param_names": ["lambda"],
        "diagnostic_traces": {},
    }))
    from spark_vi.io.export import load_result
    back = load_result(out)
    assert isinstance(back.global_params["lambda"], np.ndarray)
    assert back.global_params["lambda"].shape == (2, 5)


def test_unsupported_value_inside_a_dict_param_still_raises(tmp_path):
    """The guard must reach INSIDE a dict param: a non-numeric block is still
    unwritable, and the error must name the offending key, not just the param."""
    import numpy as np
    import pytest
    from spark_vi.core.result import VIResult
    from spark_vi.io.export import UnsupportedGlobalParamError, save_result
    res = VIResult(global_params={"lambda": {0: np.ones((2, 3)), 1: {"nested": 1}}},
                   elbo_trace=[], n_iterations=0, converged=False)
    with pytest.raises(UnsupportedGlobalParamError, match=r"lambda\[1\]"):
        save_result(res, tmp_path / "r")
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno/spark-vi && python -m pytest tests/test_export.py -k "dict_param or format_version_1" -v`
Expected: FAIL. `test_dict_param_round_trips_per_key` fails inside `save_result` with `UnsupportedGlobalParamError` (a dict currently converts to an object array and is rejected); the manifest test fails on `format_version == 2`.

- [ ] **Step 3: Implement**

In `spark_vi/io/export.py`, bump the version constant:

```python
# Manifest schema version. Bump when changing the on-disk shape; load_result
# rejects unknown versions with a clear error to provide a migration handle.
#   v2 (2026-07-25): global_params values may be a dict of arrays, written as
#   one params/<name>_<key>.npy per key with the keys listed in the manifest's
#   "dict_param_keys". Motivated by the multi-domain gated model's per-domain
#   lambda {m: (K, V_m)} (MixEHR-style storage; Li, Nair, Lu et al. 2020,
#   Nat. Commun.). v1 archives have no "dict_param_keys" and still load.
_FORMAT_VERSION = 2
_READABLE_FORMAT_VERSIONS = (1, 2)
```

Replace the `global_params` write loop in `save_result` with:

```python
    # A dict-valued param is stored per key, NOT as one array: the blocks have
    # different widths (V_m differs per domain), so there is no single array to
    # write, and np.asarray on the dict would yield a 0-d object array that
    # np.save can only pickle and load_result could never read back.
    dict_param_keys: dict[str, list[int]] = {}
    for name, arr in result.global_params.items():
        if isinstance(arr, dict):
            keys = sorted(arr)
            for k in keys:
                np.save(params_dir / f"{name}_{k}.npy",
                        _check_saveable_param(f"{name}[{k}]", arr[k]))
            dict_param_keys[name] = [int(k) for k in keys]
        else:
            np.save(params_dir / f"{name}.npy", _check_saveable_param(name, arr))
```

Add `"dict_param_keys": dict_param_keys,` to the `manifest` dict.

In `load_result`, replace the `global_params` comprehension with:

```python
    # JSON object keys are strings; the per-domain lambda is keyed by INT domain
    # index and every consumer indexes it with an int, so convert back.
    dict_param_keys = {n: [int(k) for k in ks]
                       for n, ks in manifest.get("dict_param_keys", {}).items()}
    global_params: dict[str, object] = {}
    for name in manifest["param_names"]:
        if name in dict_param_keys:
            global_params[name] = {
                k: np.load(params_dir / f"{name}_{k}.npy")
                for k in dict_param_keys[name]
            }
        else:
            global_params[name] = np.load(params_dir / f"{name}.npy")
```

Update the version check to accept `_READABLE_FORMAT_VERSIONS` instead of only the current version, and update `save_result`'s and `load_result`'s docstrings: `save_result` no longer raises for a per-domain dict λ of numeric blocks, and `_check_saveable_param`'s docstring paragraph claiming dict-λ export "is SP3's export task, not a silent write" should now say it IS implemented, with the guard retained for non-numeric blocks.

- [ ] **Step 4: Run to verify pass + regressions**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno/spark-vi && python -m pytest tests/test_export.py tests/test_checkpoint.py tests/test_runner.py tests/test_persist_check.py -q`
Expected: PASS, output pristine.

- [ ] **Step 5: Mutation check (required)**

Change `k: np.load(...)` to `str(k): np.load(...)` in the loader, re-run `-k dict_param_round_trips`, and confirm it FAILS on the int-key assertion. Revert. Put the command and output in your report; do not commit the mutation.

- [ ] **Step 6: Commit**

```bash
git add spark_vi/io/export.py tests/test_export.py
git commit -m "feat(export): write and load dict-valued global params per key

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `get_metadata` completion + η-provenance pinning

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/gated_lda.py`
- Test: `spark-vi/tests/test_gated_lda.py`

**Interfaces:**
- Consumes: Task 1's dict round-trip (the metadata test saves and loads).
- Produces: `GatedOnlineLDA.get_metadata()` returns the base `{"K", "V"}` plus, when `self.domains` is not None, `"domains": list[int]`, `"eta_m": list[float]`, `"omega": list[float]`. All JSON-serializable — `metadata` goes into `manifest.json` verbatim. Task 4 reads these keys to reconstruct a deployable model.

**Context the implementer needs:** the attributes are `self.domains` (list or None), `self._eta_domains` (per-domain η array), `self.omega` (per-domain ω array). `global_params["eta"]` in multi-domain mode is only a scalar-mean *placeholder* — do NOT read η from there.

**On the η-provenance invariant (arc blocker 4).** Multi-domain `update_global` / `compute_elbo` read η from instance state; the single-domain path reads `global_params["eta"]`, deliberately, so an η-optimization update mid-fit feeds back. These can only diverge if η changes during a fit — and `gated_lda.py:162-167` **already raises unconditionally** when `optimize_eta` is set, for single- and multi-domain alike. The spec called for adding that guard; it exists, so this task's deliverable is a **test pinning the rejection** (so a future relaxation cannot silently break the provenance invariant) plus a docstring that states the link. Do not add a second guard.

- [ ] **Step 1: Write the failing tests**

```python
def test_multidomain_get_metadata_carries_reconstruction_constants():
    """A saved multi-domain result must be reconstructable: domains, eta_m and
    omega have to travel in metadata, since global_params["eta"] is only a
    scalar-mean placeholder in multi-domain mode."""
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    lay = DagLayout({1: 0, 2: 1, 3: 1}, n_bg=2, tpn=1)
    m = GatedOnlineLDA(lay, vocab_size=56, domains=[40, 16],
                       eta=[0.02, 0.05], omega=[1.0, 0.4], random_seed=0)
    md = m.get_metadata()
    assert md["K"] == lay.K and md["V"] == 56
    assert md["domains"] == [40, 16]
    assert md["eta_m"] == [0.02, 0.05]
    assert md["omega"] == [1.0, 0.4]
    # JSON-serializable: metadata is written verbatim into manifest.json.
    import json
    json.dumps(md)


def test_single_domain_get_metadata_unchanged():
    """domains=None must return exactly the base contract, so existing archives
    and consumers see no new keys."""
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    lay = DagLayout({1: 0, 2: 1, 3: 1}, n_bg=2, tpn=1)
    m = GatedOnlineLDA(lay, vocab_size=12, random_seed=0)
    assert m.get_metadata() == {"K": lay.K, "V": 12}


def test_optimize_eta_rejected_pins_the_eta_provenance_invariant():
    """Multi-domain reads eta from instance state; single-domain reads it from
    global_params["eta"]. Those two sources are equivalent ONLY because eta
    cannot change during a fit -- optimize_eta is rejected outright. This test
    pins that rejection: if it is ever relaxed, multi-domain eta provenance
    silently diverges from single-domain and a resumed fit takes eta from the
    reconstructed model rather than the checkpoint."""
    import pytest
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    lay = DagLayout({1: 0, 2: 1, 3: 1}, n_bg=2, tpn=1)
    for kwargs in ({}, {"domains": [40, 16]}):
        vocab = 56 if kwargs else 12
        with pytest.raises(ValueError, match="optimize_eta"):
            GatedOnlineLDA(lay, vocab_size=vocab, optimize_eta=True, **kwargs)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno/spark-vi && python -m pytest tests/test_gated_lda.py -k "get_metadata or eta_provenance" -v`
Expected: the two `get_metadata` tests FAIL (multi-domain returns only `{"K", "V"}`); the `optimize_eta` test may already PASS — that is expected and correct, it is a pinning test for behavior that exists.

- [ ] **Step 3: Implement**

Add to `GatedOnlineLDA`:

```python
    def get_metadata(self) -> dict[str, Any]:
        """Shape constants plus, in multi-domain mode, the constants needed to
        RECONSTRUCT a fitted model from a saved VIResult.

        `domains` fixes the per-domain vocabulary widths that slice the
        concatenated id space; `eta_m` and `omega` are not recoverable from
        global_params -- in multi-domain mode global_params["eta"] is only a
        scalar-mean placeholder and omega never enters global_params at all
        (it weights theta during inference, not any stored parameter). Without
        these three a saved multi-domain result cannot be interpreted, let
        alone served. All values are plain Python types: `metadata` is written
        verbatim into manifest.json.

        eta provenance: multi-domain update_global/compute_elbo read eta from
        `self._eta_domains`, not from global_params, which is sound only because
        `optimize_eta` is rejected in __init__ so eta cannot change during a
        fit. `test_optimize_eta_rejected_pins_the_eta_provenance_invariant`
        pins that.
        """
        md = super().get_metadata()
        if self.domains is not None:
            md["domains"] = [int(v) for v in self.domains]
            md["eta_m"] = [float(x) for x in self._eta_domains]
            md["omega"] = [float(x) for x in self.omega]
        return md
```

- [ ] **Step 4: Run to verify pass + regressions**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno/spark-vi && python -m pytest tests/test_gated_lda.py tests/test_lda_contract.py -q`
Pass the Bash tool's `timeout` parameter as 600000 — `tests/test_gated_lda.py` alone takes ~10 minutes.
Expected: PASS, pristine.

- [ ] **Step 5: Mutation check (required)**

Drop the `if self.domains is not None:` guard so the three keys are always added, re-run `-k get_metadata`, and confirm `test_single_domain_get_metadata_unchanged` FAILS. Revert.

- [ ] **Step 6: Commit**

```bash
git add spark_vi/models/topic/gated_lda.py tests/test_gated_lda.py
git commit -m "feat(gated-lda): metadata carries domains/eta_m/omega; pin eta provenance

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Shim per-domain feature columns — layout derivation and per-row validation

**Files:**
- Modify: `spark-vi/spark_vi/mllib/topic/gated_lda.py`
- Test: `spark-vi/tests/test_gated_lda_shim.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-2.
- Produces:
  - Params `featuresCols` (list of str, default `[]` meaning unset → single-domain), `domainBounds` (list of int, optional explicit override).
  - `_domain_sizes(dataset, fcols, explicit_bounds) -> list[int]` — derives per-domain widths from the first row, or validates an explicit override against it.
  - `_concat_domain_features(vectors, sizes) -> (indices, counts)` — module-level helper concatenating per-domain sparse vectors into the engine's concatenated id space, raising on a size mismatch. **Task 4 reuses this in `_transform`**, so keep it module-level and free of estimator state.

**Design notes the implementer must honor:**
- Bounds are derived **once** from the fit dataset's first row (one Spark action, mirroring the existing `V = first[0][0].size` derivation at `_fit`), then **every row is validated**. Silently re-laying-out the vocabulary would corrupt a fit invisibly — this is the one failure mode the alternative concatenated-column design would not have had, so the validation is the point of the task, not an extra.
- Per-domain indices are already sorted within a domain, and offsetting domain *m* by `bounds[m]` keeps the concatenated indices globally sorted, which is what the engine's `expElogbeta[:, indices]` gather expects. Note that in a comment.
- `featuresCols` unset → the existing single-column path runs unchanged.

- [ ] **Step 1: Write the failing tests**

Add to `spark-vi/tests/test_gated_lda_shim.py`:

```python
def test_concat_domain_features_offsets_and_sorts():
    """Per-domain vectors concatenate into the engine's single id space: domain m
    ids shift by bounds[m], and the result stays globally sorted."""
    import numpy as np
    from pyspark.ml.linalg import SparseVector
    from spark_vi.mllib.topic.gated_lda import _concat_domain_features
    v0 = SparseVector(4, {0: 2.0, 3: 1.0})
    v1 = SparseVector(3, {1: 5.0})
    idx, cnt = _concat_domain_features([v0, v1], [4, 3])
    np.testing.assert_array_equal(idx, np.array([0, 3, 5], dtype=np.int32))
    np.testing.assert_array_equal(cnt, np.array([2.0, 1.0, 5.0]))
    assert np.all(np.diff(idx) > 0)


def test_concat_domain_features_rejects_a_mis_sized_vector():
    """A vector whose size disagrees with the established layout must raise, naming
    the domain and both sizes -- silently re-laying-out the vocabulary would
    corrupt the fit with no symptom."""
    import pytest
    from pyspark.ml.linalg import SparseVector
    from spark_vi.mllib.topic.gated_lda import _concat_domain_features
    with pytest.raises(ValueError, match=r"domain 1.*size 9.*expected 3"):
        _concat_domain_features([SparseVector(4, {0: 1.0}), SparseVector(9, {1: 1.0})],
                                [4, 3])


def test_multidomain_shim_fit_derives_domain_sizes(spark):
    """A fit with featuresCols derives the per-domain widths from the first row and
    produces a per-domain dict lambda of those widths."""
    from pyspark.ml.linalg import SparseVector, VectorUDT
    from pyspark.sql.types import ArrayType, IntegerType, StructField, StructType
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    schema = StructType([
        StructField("c", VectorUDT()), StructField("d", VectorUDT()),
        StructField("frontier", ArrayType(IntegerType())),
    ])
    rows = [
        (SparseVector(6, {0: 2.0, 1: 1.0}), SparseVector(4, {0: 3.0}), [2]),
        (SparseVector(6, {1: 1.0, 2: 2.0}), SparseVector(4, {1: 1.0}), [2]),
        (SparseVector(6, {3: 3.0, 4: 1.0}), SparseVector(4, {2: 2.0}), [3]),
        (SparseVector(6, {4: 1.0, 5: 1.0}), SparseVector(4, {3: 4.0}), [3]),
    ]
    df = spark.createDataFrame(rows, schema=schema).repartition(2)
    est = GatedLDAEstimator(featuresCols=["c", "d"], labelCol="frontier",
                            parent={1: 0, 2: 1, 3: 1}, nBg=2, tpn=1,
                            maxIter=1, seed=0)
    model = est.fit(df)
    lam = model.result.global_params["lambda"]
    assert isinstance(lam, dict) and sorted(lam) == [0, 1]
    assert lam[0].shape[1] == 6 and lam[1].shape[1] == 4
    assert model.result.metadata["domains"] == [6, 4]


def test_multidomain_shim_fit_rejects_a_mis_sized_row(spark):
    """A row disagreeing with the derived layout fails the fit rather than
    silently re-laying-out the vocabulary."""
    import pytest
    from pyspark.ml.linalg import SparseVector, VectorUDT
    from pyspark.sql.types import ArrayType, IntegerType, StructField, StructType
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    schema = StructType([
        StructField("c", VectorUDT()), StructField("d", VectorUDT()),
        StructField("frontier", ArrayType(IntegerType())),
    ])
    rows = [
        (SparseVector(6, {0: 1.0}), SparseVector(4, {0: 1.0}), [2]),
        (SparseVector(6, {1: 1.0}), SparseVector(9, {1: 1.0}), [3]),   # wrong width
    ]
    df = spark.createDataFrame(rows, schema=schema)
    est = GatedLDAEstimator(featuresCols=["c", "d"], labelCol="frontier",
                            parent={1: 0, 2: 1, 3: 1}, nBg=2, tpn=1,
                            maxIter=1, seed=0)
    with pytest.raises(Exception, match="domain 1"):
        est.fit(df)


def test_explicit_domain_bounds_override_an_unrepresentative_first_row(spark):
    """domainBounds is the escape hatch for a dataset whose first row does not
    carry the true per-domain widths (a narrower vector, e.g. a producer that
    sized it to the max nonzero id). When set it is AUTHORITATIVE: the fit uses
    those widths and every row -- the first included -- is validated against them,
    so a first row that disagrees fails instead of silently defining the layout."""
    import pytest
    from pyspark.ml.linalg import SparseVector, VectorUDT
    from pyspark.sql.types import ArrayType, IntegerType, StructField, StructType
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    schema = StructType([
        StructField("c", VectorUDT()), StructField("d", VectorUDT()),
        StructField("frontier", ArrayType(IntegerType())),
    ])
    # Every row is width 6/4; bounds declaring 6/4 must fit fine...
    ok = spark.createDataFrame(
        [(SparseVector(6, {0: 1.0}), SparseVector(4, {0: 1.0}), [2]),
         (SparseVector(6, {3: 1.0}), SparseVector(4, {2: 1.0}), [3])], schema=schema)
    est = GatedLDAEstimator(featuresCols=["c", "d"], labelCol="frontier",
                            domainBounds=[0, 6, 10], parent={1: 0, 2: 1, 3: 1},
                            nBg=2, tpn=1, maxIter=1, seed=0)
    model = est.fit(ok)
    assert model.result.metadata["domains"] == [6, 4]
    # ...and bounds that disagree with the actual vectors must fail per-row.
    bad = GatedLDAEstimator(featuresCols=["c", "d"], labelCol="frontier",
                            domainBounds=[0, 5, 9], parent={1: 0, 2: 1, 3: 1},
                            nBg=2, tpn=1, maxIter=1, seed=0)
    with pytest.raises(Exception, match="domain 0"):
        bad.fit(ok)
    # Malformed bounds are rejected on the driver, before any Spark work.
    with pytest.raises(ValueError, match="strictly increasing"):
        GatedLDAEstimator(featuresCols=["c", "d"], labelCol="frontier",
                          domainBounds=[0, 6, 6], parent={1: 0, 2: 1, 3: 1},
                          nBg=2, tpn=1, maxIter=1, seed=0).fit(ok)


def test_single_domain_shim_path_unchanged(spark):
    """featuresCols unset keeps the existing single-featuresCol behavior: a single
    (K, V) array lambda and no domains in metadata."""
    import numpy as np
    from pyspark.ml.linalg import SparseVector, VectorUDT
    from pyspark.sql.types import ArrayType, IntegerType, StructField, StructType
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    schema = StructType([
        StructField("features", VectorUDT()),
        StructField("frontier", ArrayType(IntegerType())),
    ])
    rows = [(SparseVector(8, {0: 1.0, 1: 2.0}), [2]),
            (SparseVector(8, {2: 1.0, 3: 1.0}), [3])]
    df = spark.createDataFrame(rows, schema=schema)
    est = GatedLDAEstimator(labelCol="frontier", parent={1: 0, 2: 1, 3: 1},
                            nBg=2, tpn=1, maxIter=1, seed=0)
    model = est.fit(df)
    assert isinstance(model.result.global_params["lambda"], np.ndarray)
    assert "domains" not in model.result.metadata
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno/spark-vi && python -m pytest tests/test_gated_lda_shim.py -k "concat_domain or multidomain_shim or single_domain_shim" -v`
Expected: FAIL — `_concat_domain_features` does not exist; `featuresCols` is an unknown kwarg.

- [ ] **Step 3: Implement**

Add the two Params to `_GatedLDAParams`:

```python
    featuresCols = Param(Params._dummy(), "featuresCols",
                         "ordered per-domain feature column names (multi-domain, "
                         "MixEHR-style per-modality vocabularies; Li, Nair, Lu et al. "
                         "2020, Nat. Commun.). Unset = single-domain via featuresCol.",
                         typeConverter=TypeConverters.toListString)
    domainBounds = Param(Params._dummy(), "domainBounds",
                         "optional explicit cumulative per-domain offsets "
                         "[0, V_0, V_0+V_1, ...]; normally DERIVED from the first "
                         "row's per-column vector sizes.",
                         typeConverter=TypeConverters.toListInt)
```

Add the module-level helper:

```python
def _concat_domain_features(vectors, sizes):
    """Concatenate per-domain sparse vectors into the engine's single id space.

    The engine stores one topic-word matrix per domain but consumes ONE
    concatenated token-id space, with a token's domain recovered by
    `searchsorted(domain_bounds, w)`. Domain m's local ids therefore shift by
    `sum(sizes[:m])`. Because each domain's ids are already ascending and the
    domains are laid out in order, the concatenated ids are GLOBALLY sorted,
    which is what the E-step's `expElogbeta[:, indices]` gather assumes.

    Raises ValueError naming the domain and both widths if a vector disagrees
    with the established layout: the layout is derived once per fit, and a row
    that silently re-lays-out the vocabulary would corrupt the fit with no
    symptom (SP3a design).
    """
    import numpy as np
    idx_parts, cnt_parts, offset = [], [], 0
    for m, (v, width) in enumerate(zip(vectors, sizes)):
        if int(v.size) != int(width):
            raise ValueError(
                f"featuresCols domain {m} vector size {int(v.size)} != expected "
                f"{int(width)} (layout derived from the first row); every row must "
                f"use the same per-domain vocabulary widths")
        bow = _vector_to_bow_document(v)
        idx_parts.append(bow.indices.astype(np.int64) + offset)
        cnt_parts.append(bow.counts)
        offset += int(width)
    indices = np.concatenate(idx_parts).astype(np.int32) if idx_parts else np.empty(0, np.int32)
    counts = np.concatenate(cnt_parts) if cnt_parts else np.empty(0, np.float64)
    return indices, counts
```

In `_fit`, replace the single-column `V` derivation with a branch. Keep the existing single-domain code path literally unchanged in the `else`:

```python
        fcols = list(self.getOrDefault("featuresCols") or [])
        label_col = self.getOrDefault("labelCol")
        if fcols:
            first = dataset.select(*fcols).head(1)
            if not first:
                raise ValueError("Cannot fit on an empty DataFrame.")
            if self.isSet("domainBounds"):
                # Explicit bounds are AUTHORITATIVE, not merely cross-checked: this
                # is the escape hatch for a dataset whose first row is
                # unrepresentative, so it must not be rejected for disagreeing with
                # that row. Every row -- the first included -- is then validated
                # against these widths by _concat_domain_features.
                bounds = [int(b) for b in self.getOrDefault("domainBounds")]
                if len(bounds) != len(fcols) + 1 or bounds[0] != 0 or \
                        any(b <= a for a, b in zip(bounds, bounds[1:])):
                    raise ValueError(
                        f"domainBounds {bounds} must be strictly increasing, start "
                        f"at 0, and have len(featuresCols)+1 = {len(fcols) + 1} entries")
                sizes = [b - a for a, b in zip(bounds, bounds[1:])]
            else:
                sizes = [int(first[0][i].size) for i in range(len(fcols))]
            V = sum(sizes)
            domains = sizes
        else:
            features_col = self.getOrDefault("featuresCol")
            first = dataset.select(features_col).head(1)
            if not first:
                raise ValueError("Cannot fit on an empty DataFrame.")
            V = first[0][0].size
            domains = None
```

Pass `domains=domains` into the `GatedOnlineLDA(...)` construction, and add the multi-domain row mapper, selecting the feature columns plus the label column:

```python
        if fcols:
            n_dom = len(fcols)

            def _to_gated(row):
                indices, counts = _concat_domain_features(
                    [row[i] for i in range(n_dom)], sizes)
                frontier = frozenset(int(x) for x in (row[n_dom] or []))
                return GatedBOWDocument(indices=indices, counts=counts,
                                        length=int(counts.sum()), frontier=frontier)

            rdd = (dataset.select(*fcols, label_col).rdd.map(_to_gated)
                   .persist(StorageLevel.MEMORY_AND_DISK))
        else:
            # ... existing single-column _to_gated and rdd construction, unchanged
```

`rdd.count()` after, as today. The dense-init `collected` branch must select `*fcols` and build its `train_docs` through the same helper when `fcols` is set, so the seed sees the same concatenated ids as the fit.

- [ ] **Step 4: Run to verify pass + regressions**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno/spark-vi && python -m pytest tests/test_gated_lda_shim.py tests/test_mllib_lda.py -q` (Bash `timeout` 600000)
Expected: PASS, pristine.

- [ ] **Step 5: Mutation check (required)**

Delete the size check inside `_concat_domain_features`, re-run `-k "concat_domain or mis_sized"`, and confirm both mis-size tests FAIL. Revert.

- [ ] **Step 6: Commit**

```bash
git add spark_vi/mllib/topic/gated_lda.py tests/test_gated_lda_shim.py
git commit -m "feat(gated-shim): per-domain featuresCols with derived, row-validated layout

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Shim ω / per-domain η Params + dict-λ `_transform`

**Files:**
- Modify: `spark-vi/spark_vi/mllib/topic/gated_lda.py`
- Test: `spark-vi/tests/test_gated_lda_shim.py`

**Interfaces:**
- Consumes: Task 3's `_concat_domain_features` and `featuresCols`; Task 2's metadata keys.
- Produces: Params `omega` (list of float) and `etaPerDomain` (list of float), both forwarded to `GatedOnlineLDA`. `GatedLDAModel._transform` handles a dict λ and applies ω.

**Why ω must be applied at transform.** ω weights the γ accumulation, so it changes θ — and θ *is* the deployment read-out (`nodeAffinity`). SP2 established there is no train/serve skew inside the engine because `infer_local` applies ω; the shim's `_transform` calls `_cavi_doc_inference` directly instead, so it must pass the same per-token weight or fitted and deployed θ diverge silently.

- [ ] **Step 1: Write the failing tests**

```python
def test_multidomain_shim_transform_produces_node_affinity(spark):
    """A multi-domain fitted model must transform: _transform currently assumes an
    ndarray lambda and raises on the per-domain dict."""
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    df = _two_domain_df(spark)                     # helper added in this task
    est = GatedLDAEstimator(featuresCols=["c", "d"], labelCol="frontier",
                            parent={1: 0, 2: 1, 3: 1}, nBg=2, tpn=1,
                            maxIter=2, seed=0, omega=[1.0, 0.4])
    model = est.fit(df)
    out = model.transform(df).select("nodeAffinity").collect()
    assert len(out) == df.count()
    for r in out:
        vals = list(r[0])
        assert len(vals) == 3                      # one per DAG node
        assert all(v >= 0.0 for v in vals)


def test_transform_applies_omega_so_deployed_theta_matches_fitted(spark):
    """omega weights theta, and theta IS the deployed read-out, so transform must
    apply the same per-token weight the fit used. Directional: down-weighting
    domain 1 must change nodeAffinity relative to omega=1 on the SAME fitted
    lambda."""
    import numpy as np
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    df = _two_domain_df(spark)
    est = GatedLDAEstimator(featuresCols=["c", "d"], labelCol="frontier",
                            parent={1: 0, 2: 1, 3: 1}, nBg=2, tpn=1,
                            maxIter=2, seed=0, omega=[1.0, 1.0])
    model = est.fit(df)
    base = np.array([list(r[0]) for r in model.transform(df).select("nodeAffinity").collect()])
    model._set(omega=[1.0, 0.05])                  # same lambda, different dial
    down = np.array([list(r[0]) for r in model.transform(df).select("nodeAffinity").collect()])
    assert not np.allclose(base, down), "transform ignored omega"
```

Add this module-level helper to `tests/test_gated_lda_shim.py` (Task 3's tests can be refactored onto it too, but do not block on that):

```python
def _two_domain_df(spark):
    """Six two-domain rows: domain 0 width 6, domain 1 width 4, frontiers over
    nodes 2 and 3 of the DAG {1:0, 2:1, 3:1}, two partitions so the reduce
    combines more than one. Deliberately tiny -- these are shim-contract tests,
    not recovery tests."""
    from pyspark.ml.linalg import SparseVector, VectorUDT
    from pyspark.sql.types import ArrayType, IntegerType, StructField, StructType
    schema = StructType([
        StructField("c", VectorUDT()), StructField("d", VectorUDT()),
        StructField("frontier", ArrayType(IntegerType())),
    ])
    rows = [
        (SparseVector(6, {0: 2.0, 1: 1.0}), SparseVector(4, {0: 3.0}), [2]),
        (SparseVector(6, {1: 1.0, 2: 2.0}), SparseVector(4, {1: 1.0}), [2]),
        (SparseVector(6, {0: 1.0, 2: 1.0}), SparseVector(4, {0: 2.0, 1: 1.0}), [2]),
        (SparseVector(6, {3: 3.0, 4: 1.0}), SparseVector(4, {2: 2.0}), [3]),
        (SparseVector(6, {4: 1.0, 5: 1.0}), SparseVector(4, {3: 4.0}), [3]),
        (SparseVector(6, {3: 1.0, 5: 2.0}), SparseVector(4, {2: 1.0, 3: 1.0}), [3]),
    ]
    return spark.createDataFrame(rows, schema=schema).repartition(2)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno/spark-vi && python -m pytest tests/test_gated_lda_shim.py -k "transform" -v`
Expected: FAIL — `AttributeError: 'dict' object has no attribute 'sum'` from `lam.sum(axis=1, keepdims=True)`.

- [ ] **Step 3: Implement**

Add the Params:

```python
    omega = Param(Params._dummy(), "omega",
                  "per-domain modality weight on the doc-topic accumulation "
                  "(theta only; lambda sstats and the data loglik use TRUE counts). "
                  "Default all 1.0 = faithful MixEHR, volume speaks (Li, Nair, Lu "
                  "et al. 2020). A tuned-vs-task tempering weight, NOT fitted.",
                  typeConverter=TypeConverters.toListFloat)
    etaPerDomain = Param(Params._dummy(), "etaPerDomain",
                         "per-domain Dirichlet prior on the topic-word blocks; "
                         "unset = the scalar 1/K used by the single-domain path.",
                         typeConverter=TypeConverters.toListFloat)
```

Forward both into the `GatedOnlineLDA(...)` construction in `_fit` (only when `fcols` is set; passing `omega` without `domains` raises by design):

```python
            omega=(list(self.getOrDefault("omega")) if self.isSet("omega") else None),
            eta=(list(self.getOrDefault("etaPerDomain")) if self.isSet("etaPerDomain")
                 else 1.0 / lay.K),
```

In `_transform`, replace the `expElogbeta` construction and add the per-token weight:

```python
        lam = self._result.global_params["lambda"]
        if isinstance(lam, dict):
            # Per-domain dict lambda: each block normalizes over its OWN vocabulary
            # (the MixEHR per-modality model, where a token's domain is exogenous),
            # then the blocks concatenate into the engine's single id space. This is
            # GatedOnlineLDA._assemble_expElogbeta's arithmetic; it is repeated here
            # because _transform holds a VIResult, not a model instance.
            blocks = [lam[m] for m in sorted(lam)]
            expElogbeta = np.concatenate(
                [np.exp(digamma(b) - digamma(b.sum(axis=1, keepdims=True)))
                 for b in blocks], axis=1)
            sizes = [int(b.shape[1]) for b in blocks]
            bounds = np.cumsum([0] + sizes)
            omega = (np.asarray(self.getOrDefault("omega"), dtype=np.float64)
                     if self.isSet("omega")
                     else np.ones(len(sizes), dtype=np.float64))
            if omega.shape != (len(sizes),):
                raise ValueError(f"omega has {omega.shape[0]} entries for "
                                 f"{len(sizes)} domains")
        else:
            expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
            bounds, omega = None, None
```

Broadcast `bounds` and `omega` alongside the rest, and inside `_affinity`:

```python
            # omega weights theta, and theta is what this function returns, so the
            # deployed read-out must use the SAME per-token weight the fit used or
            # fitted and served theta diverge silently.
            w_tok = None
            if p["bounds"] is not None:
                dom = np.searchsorted(p["bounds"], doc.indices, side="right") - 1
                w_tok = p["omega"][dom]
            gamma, _, _, _ = _cavi_doc_inference(
                indices=doc.indices, counts=doc.counts, expElogbeta=p["expElogbeta"],
                alpha=p["alpha"], gamma_init=gamma_init,
                max_iter=p["cavi_max_iter"], tol=p["cavi_tol"],
                gamma_count_weight=w_tok)
```

`_transform` must also accept the multi-domain input shape: when `featuresCols` is set, build the document with `_concat_domain_features` from Task 3 instead of a single-column `_vector_to_bow_document`.

- [ ] **Step 4: Run to verify pass + regressions**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno/spark-vi && python -m pytest tests/test_gated_lda_shim.py tests/test_mllib_lda.py tests/test_mllib_stm.py -q` (Bash `timeout` 600000)
Expected: PASS, pristine.

- [ ] **Step 5: Mutation check (required)**

Drop `gamma_count_weight=w_tok` from the `_cavi_doc_inference` call, re-run `-k applies_omega`, confirm FAIL. Revert.

- [ ] **Step 6: Commit**

```bash
git add spark_vi/mllib/topic/gated_lda.py tests/test_gated_lda_shim.py
git commit -m "feat(gated-shim): omega/etaPerDomain Params and dict-lambda transform

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: save → load → identical `nodeAffinity`

**Files:**
- Test only: `spark-vi/tests/test_gated_lda_shim.py`

**Interfaces:**
- Consumes: Tasks 1-4 (dict persistence, metadata, `featuresCols`, dict-λ transform).
- Produces: nothing new — this is the acceptance gate for the whole persistence story.

**Why this gate and not "it loaded without error".** The round-trip has to reproduce the *read-out*, because that is what a downstream consumer uses. Loading a λ whose domain keys came back as strings, or whose blocks were reordered, would load cleanly and score differently.

- [ ] **Step 1: Write the failing test**

```python
def test_multidomain_save_load_reproduces_node_affinity(spark, tmp_path):
    """The whole persistence story end to end: fit multi-domain through the shim,
    score, save, load, rebuild the Model from the loaded VIResult, score again --
    and get the SAME nodeAffinity. A dict lambda that round-trips with string keys
    or reordered blocks would load cleanly and score differently."""
    import numpy as np
    from spark_vi.io.export import load_result, save_result
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator, GatedLDAModel
    df = _two_domain_df(spark)
    est = GatedLDAEstimator(featuresCols=["c", "d"], labelCol="frontier",
                            parent={1: 0, 2: 1, 3: 1}, nBg=2, tpn=1,
                            maxIter=2, seed=0, omega=[1.0, 0.4])
    model = est.fit(df)
    before = np.array([list(r[0]) for r in
                       model.transform(df).select("nodeAffinity").collect()])

    save_result(model.result, tmp_path / "fit")
    loaded = load_result(tmp_path / "fit")
    assert isinstance(loaded.global_params["lambda"], dict)
    assert sorted(loaded.global_params["lambda"]) == [0, 1]
    assert loaded.metadata["domains"] == [6, 4]
    assert loaded.metadata["omega"] == [1.0, 0.4]

    rebuilt = GatedLDAModel(loaded, parent={1: 0, 2: 1, 3: 1}, nBg=2, tpn=1)
    rebuilt._set(featuresCols=["c", "d"], omega=loaded.metadata["omega"])
    after = np.array([list(r[0]) for r in
                      rebuilt.transform(df).select("nodeAffinity").collect()])
    np.testing.assert_allclose(before, after, rtol=1e-12, atol=0.0)
```

- [ ] **Step 2: Run to verify it fails before the stack is complete** — if Tasks 1-4 are already merged this test should PASS on first run; say so in the report rather than manufacturing a failure. If it fails, the failure is real and must be fixed, not accommodated.

- [ ] **Step 3: If it passes, no implementation needed.** If the ω or `featuresCols` state does not survive into the rebuilt model, that is a genuine gap in Task 2's metadata or Task 4's Params — fix it there and note it.

- [ ] **Step 4: Run to verify pass + regressions**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno/spark-vi && python -m pytest tests/test_gated_lda_shim.py tests/test_export.py -q` (Bash `timeout` 600000)

- [ ] **Step 5: Mutation check (required)**

In `load_result`, reverse the dict key order (`for k in reversed(dict_param_keys[name])`) so the blocks come back swapped, re-run this test, confirm FAIL. Revert. (This is the mutation the "it loaded fine" version of this test would have missed.)

- [ ] **Step 6: Commit**

```bash
git add tests/test_gated_lda_shim.py
git commit -m "test(gated-shim): multi-domain save/load reproduces nodeAffinity

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Scalable-vs-dense per-domain recovery (the arc's pre-registered gate)

**Files:**
- Test: `spark-vi/tests/test_gated_lda.py`
- Modify (docstring only): `spark-vi/spark_vi/models/topic/gated_init.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-5 — independent, and may be done in parallel with them if the controller prefers.
- Produces: the recorded equivalence that closes the arc's SP3 blocker 1.

**Background.** The arc design pre-registered this as "the thing to test first, not to assume": the per-domain candidate floor exists only on the dense init path, while the shim routes to the *scalable* path above `spectralMaxVocab`. A throwaway probe (2026-07-25) already answered it — the scalable seed matched or beat the dense+floor seed on five of six per-domain cells post-EM. This task **commits** that result across seeds.

The probe's numbers, one corpus, post-EM (50 full-batch iterations), 3-node DAG `{1:0,2:1,3:1}`, `b_only_node=3`, `V_a=40`, `V_b=16`, `bg_frac=0.2`, `ancestor_signature_decay=0.5`, `anchor_scope="frontier"`, 800 docs:

| seed | domain 0 (n1/n2/n3) | domain 1 (n1/n2/n3) |
|---|---|---|
| dense + floor | 0.895 / 0.701 / 0.492 | 0.826 / 0.699 / 0.995 |
| dense, no floor | 0.933 / 0.703 / 0.530 | 0.970 / 0.737 / 0.882 |
| scalable | 0.951 / 0.686 / 0.589 | 0.984 / 0.706 / 0.964 |

`scalable_block_aligned_lambda(rdd, lay, V, *, d=None, seed=0, min_doc_freq=5, anchor_scope="closure", topo_order="forward")` returns a single joint `(K, V)` array, so per-domain blocks come from `spectral_init.split_domains(beta, bounds)` (returns a **list** of `(K, V_m)` row-stochastic matrices in domain order — not a dict).

- [ ] **Step 1: Write the failing test**

```python
def test_scalable_init_matches_dense_per_domain_floor_seed(spark):
    """The arc's pre-registered SP3 gate. The production (scalable) spectral init
    never received the per-domain candidate floor that the dense path has, and the
    mllib shim routes to the scalable path above spectralMaxVocab -- so a
    production-size multi-domain fit runs on an init that the dense acceptance
    tests never covered.

    The floors are different RULES, which is why immunity was plausible: dense
    find_anchors uses a MEAN-RELATIVE marginal floor (a denser domain can dominate
    the pooled mean), while find_anchors_projected uses an ABSOLUTE
    document-frequency floor (df_w >= min_doc_freq, ADR 0032, adopted because the
    mean-relative rule over-excludes rare-but-pure words in the sketch setting).
    An absolute per-word threshold has no pooled mean to be swamped.

    Gate: post-EM per-domain recovery from the scalable seed is not materially
    worse than from the dense+floor seed, per node and per domain, across several
    fit seeds. TOL is set from the observed seed-to-seed spread, NOT from a single
    comparison (SP2 lost a review round to a single-seed gate that passed on a
    lucky draw)."""
    import numpy as np
    from tests._stm_synth import two_domain_dag_corpus
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_init import (
        multidomain_spectral_lambda, scalable_block_aligned_lambda,
    )
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    from spark_vi.models.topic.spectral_init import split_domains
    from spark_vi.models.topic.types import GatedBOWDocument
    parent, b_only = {1: 0, 2: 1, 3: 1}, 3
    V_A, V_B = 40, 16
    docs, labels, bounds, pa, pb, slot, _ = two_domain_dag_corpus(
        parent=parent, node_prev={1: 1.0, 2: 1.0, 3: 1.0}, V_a=V_A, V_b=V_B,
        doc_len=40, seed=5, b_only_node=b_only, bg_frac=0.2,
        ancestor_signature_decay=0.5)
    lay = DagLayout(parent, n_bg=2, tpn=1)
    V = bounds[-1]
    rng = np.random.default_rng(0)
    keep = [int(i) for i in rng.permutation(len(docs))[:800]]
    tr_docs = [np.asarray(docs[i]) for i in keep]
    tr_labels = [labels[i] for i in keep]
    gdocs = []
    for d, y in zip(tr_docs, tr_labels):
        idx, cnt = np.unique(np.asarray(d), return_counts=True)
        fr = frozenset(y) if hasattr(y, "__iter__") else frozenset({int(y)})
        gdocs.append(GatedBOWDocument(indices=idx.astype(np.int32),
                                      counts=cnt.astype(float),
                                      length=int(cnt.sum()), frontier=fr))
    ds = {"train_docs": tr_docs, "train_labels": tr_labels,
          "anchor_scope": "frontier"}
    planted, Vm = {0: pa, 1: pb}, {0: V_A, 1: V_B}

    def _post_em(lam_seed, fit_seed):
        m = GatedOnlineLDA(lay, vocab_size=V, domains=[V_A, V_B],
                           random_seed=fit_seed)
        gp = m.initialize_global(None)
        gp["lambda"] = {md: np.array(lam_seed[md], dtype=np.float64)
                        for md in (0, 1)}
        for _ in range(50):
            gp = m.update_global(gp, m.local_update(gdocs, gp), learning_rate=1.0)
        out = {}
        for md in (0, 1):
            beta = gp["lambda"][md] / gp["lambda"][md].sum(axis=1, keepdims=True)
            for u in lay.nodes:
                sup = np.where(planted[md][slot[u]] > 1e-3)[0]
                out[(md, u)] = float(beta[lay.block[u]][:, sup].sum(axis=1).max())
        return out

    dense = multidomain_spectral_lambda(ds, lay, [V_A, V_B], anchor_scope="frontier")
    rdd = spark.sparkContext.parallelize(gdocs, 4)
    scal_joint = scalable_block_aligned_lambda(rdd, lay, V, anchor_scope="frontier",
                                              min_doc_freq=5, seed=0)
    scal_blocks = split_domains(scal_joint, bounds)
    scal = {md: (scal_blocks[md] + 1e-9) * 200.0 for md in (0, 1)}

    TOL = 0.15                 # see spread assertion below
    spreads = []
    for fit_seed in (0, 1, 2):
        d_rec, s_rec = _post_em(dense, fit_seed), _post_em(scal, fit_seed)
        for key in d_rec:
            spreads.append(s_rec[key] - d_rec[key])
            assert s_rec[key] > d_rec[key] - TOL, (fit_seed, key, d_rec[key], s_rec[key])
        # neither seed may leave a node at the dead-topic floor
        for md in (0, 1):
            unif = None
            for u in lay.nodes:
                sup = np.where(planted[md][slot[u]] > 1e-3)[0]
                unif = len(sup) / Vm[md]
                assert s_rec[(md, u)] > 1.5 * unif, (fit_seed, md, u, s_rec[(md, u)], unif)
    # TOL must exceed the observed spread, or the gate is a coin flip.
    assert TOL > float(np.std(spreads)) * 2.0, (TOL, float(np.std(spreads)))
```

- [ ] **Step 2: Run**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno/spark-vi && python -m pytest tests/test_gated_lda.py -k scalable_init_matches -v` (Bash `timeout` 600000)
Expected: PASS (the probe says so). **If it fails: follow the Global Constraint — strengthen the plant, never loosen TOL, and if a strong plant still fails, STOP and report BLOCKED with the per-seed numbers.** A failure here is a genuine negative about the production init path and is exactly what this task exists to surface.

- [ ] **Step 3: Record the equivalence in the docstrings**

Update `scalable_block_aligned_lambda`'s docstring to state that it carries no per-domain candidate floor and does not need one, with the measured comparison and a pointer to this test; and amend `spectral_block_aligned_lambda`'s `domain_bounds` paragraph, which currently points at SP3 as an open acceptance item, to point at the settled result instead.

- [ ] **Step 4: Report whether the secondary finding reproduced**

While the harness is in hand, also record (in the task report, not a new test) whether the *dense* floor's own value reproduces as plant-dependent: on a degenerate corpus (`bg_frac=0.0`, `ancestor_signature_decay=1.0`, `anchor_scope="closure"`) the probe saw dense-without-floor leave three near-dead cells post-EM (0.015 / 0.022 / 0.004) while the floor rescued them, and saw no difference on the well-specified corpus. **If it reproduces across the three fit seeds, say so — the controller owns writing insight 0070 and will not write it on one seed.** Do not add this as a committed test; it is a finding, not a gate.

- [ ] **Step 5: Run regressions**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno/spark-vi && python -m pytest tests/test_gated_lda.py tests/test_gated_init.py tests/test_spectral_init.py tests/test_spectral_init_scalable.py -q` (Bash `timeout` 600000)

- [ ] **Step 6: Commit**

```bash
git add tests/test_gated_lda.py spark_vi/models/topic/gated_init.py
git commit -m "test(gated-init): scalable init needs no per-domain floor (arc SP3 blocker 1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Post-plan wrap-up (controller)

- [ ] Whole-branch review over the SP3a commit range on the most capable model.
- [ ] Write insight 0070 **only if** Task 6 Step 4 reproduced the plant-dependence of the dense floor across seeds. If it did, it also means insight 0065's framing of that floor as load-bearing should be flagged the way 0066's premise was.
- [ ] Update the arc design's SP3 stub: blocker 1 closed by a committed test, blockers 2 and 3 closed by Tasks 1-2, blocker 4 resolved by invariant + pinning test.
- [ ] Report to the user; do NOT merge or push. Next: the SP3b plan (`docs/superpowers/specs/2026-07-25-sp3b-drug-domain-and-cloud-driver-design.md`), just-in-time.

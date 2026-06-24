# Gated STM Local Harness (Plan 2a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A fully local pipeline that simulates a gated background/foreground STM corpus and fits it in-process, producing a real gated `STMModel` checkpoint the dashboard (Plan 2b) can render — plus the small engine fix that makes a foreground-less group background-only.

**Architecture:** A gated OMOP simulator (`scripts/simulate_gated_omop.py`, sibling to the LDA sim) plants background concepts shared by all patients plus per-group distinctive concepts, tags each patient with `source_cohort` + `sex`/`age`, and gates each patient's topics to background ∪ its group's foreground. A local fitter (`analysis/local/fit_stm_local.py`) follows `fit_lda_local.py`'s pattern (local Spark + the `StreamingSTM` shim) to fit it and save a checkpoint with full metadata.

**Tech Stack:** Python, NumPy, pandas, PySpark (local mode + MLlib shim), pytest.

## Global Constraints

- `fit_stm_local.py` matches `fit_lda_local.py`'s pattern: local Spark (`local[2]`) + the `StreamingSTM` shim (not a bespoke pure-Python loop).
- A document whose group has no foreground block gets **background-only** (engine fix), never a crash.
- `source_cohort` is the gating group label, NOT a covariate — it must not appear in `--covariate-formula`. Covariates key per-person (`age = static`, `sex = static`).
- Terminology: "group" / "background/foreground block"; `source_cohort` is the charmpheno domain label fed into the engine's generic group slot.
- The simulator's oracle columns (`true_topic_id`, `true_block`) are evaluation-only — the fitter must not read them.
- No LaTeX in docstrings (Unicode Greek OK: α β η θ Σ Γ); no emojis in committed files.
- TDD throughout. Commit messages end with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

**Reference spec:** [docs/superpowers/specs/2026-06-23-gated-stm-local-dashboard-design.md](../specs/2026-06-23-gated-stm-local-dashboard-design.md)

---

## File Structure

- Modify: `spark-vi/spark_vi/models/topic/partition.py` — `allowed_indices` skips foreground-less groups.
- Modify: `spark-vi/tests/test_topic_block_partition.py` — flip the unknown-group test.
- Create: `scripts/simulate_gated_omop.py` — gated simulator (β builder + `simulate_gated` core + CLI).
- Create: `tests/scripts/test_simulate_gated_omop.py` — simulator unit tests.
- Create: `analysis/local/fit_stm_local.py` — local gated-STM fitter.
- Create: `tests/scripts/test_fit_stm_local.py` — end-to-end harness recovery test.

---

## Task 1: Engine fix — foreground-less group is background-only

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/partition.py` (`allowed_indices`)
- Test: `spark-vi/tests/test_topic_block_partition.py`

**Interfaces:**
- Consumes: existing `TopicBlockPartition`.
- Produces: `allowed_indices(groups: frozenset[str]) -> np.ndarray` now returns `background ∪ ⋃ block_indices(g) for g in groups that ARE foreground groups`; groups absent from the foreground map contribute nothing (no `KeyError`).

- [ ] **Step 1: Replace the raising test with a background-only test**

In `spark-vi/tests/test_topic_block_partition.py`, DELETE `test_unknown_group_in_allowed_indices_raises` and add:

```python
def test_foregroundless_group_yields_background_only():
    p = _part()  # background_k=3, foreground cancer:2, dementia:2
    # a group with no foreground block contributes nothing -> background only
    np.testing.assert_array_equal(
        p.allowed_indices(frozenset({"nope"})), [0, 1, 2])
    # mixing a known and an unknown group -> background + the known block only
    np.testing.assert_array_equal(
        p.allowed_indices(frozenset({"cancer", "nope"})), [0, 1, 2, 3, 4])
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_topic_block_partition.py -k foregroundless -v`
Expected: FAIL with `KeyError: "unknown group 'nope'; ..."` (the current code raises).

- [ ] **Step 3: Implement the skip**

In `spark-vi/spark_vi/models/topic/partition.py`, replace `allowed_indices`:

```python
    def allowed_indices(self, groups: frozenset[str]) -> np.ndarray:
        # A group with no foreground block contributes nothing (background-only).
        # This is what lets a large "common" cohort inform the background while
        # only rare groups carry foreground topics.
        known = set(self.groups)
        parts = [self.background_indices()]
        for g in sorted(groups):
            if g in known:
                parts.append(self.block_indices(g))
        return np.unique(np.concatenate(parts)).astype(np.int64)
```

Update the method's behavior note in the class docstring if it mentions raising on unknown groups (it should now say foreground-less groups are background-only).

- [ ] **Step 4: Run to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_topic_block_partition.py -v`
Expected: PASS (the foregroundless test plus all pre-existing partition tests).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/partition.py spark-vi/tests/test_topic_block_partition.py
git commit -m "$(printf 'fix(stm): foreground-less group is background-only, not a crash\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 2: Gated simulator core (β builder + simulate_gated)

**Files:**
- Create: `scripts/simulate_gated_omop.py` (pure functions only this task)
- Test: `tests/scripts/test_simulate_gated_omop.py`

**Interfaces:**
- Produces:
  - `build_gated_beta(*, n_background_concepts, n_group_concepts, background_k, foreground, rng, bleed=0.1) -> GatedBeta` where `foreground` is `tuple[tuple[str, int], ...]` (group, K_fg) and `GatedBeta` is a dataclass with: `beta` (K_total × V row-stochastic), `concept_ids` (np.int array length V), `concept_names` (dict cid->str), `topic_blocks` (list[str] length K_total: "background" or group), `group_concepts` (dict group->list[int] concept_ids), `background_concepts` (list[int]).
  - `simulate_gated(gb: GatedBeta, *, n_patients, group_props, foreground, visits_per_patient_mean, codes_per_visit_mean, age_means, theta_alpha, seed) -> tuple[pd.DataFrame, pd.DataFrame, dict]` returning `(events_df, person_df, oracle)`. `group_props` is `dict[str, float]` (group -> share, includes any background-only group). `age_means` is `dict[str, float]`. events_df columns: `person_id, visit_occurrence_id, concept_id, concept_name, source_cohort, true_topic_id, true_block`. person_df columns: `person_id, source_cohort, sex, age`. oracle: `{"background_concepts": [...], "group_concepts": {...}, "topic_blocks": [...], "background_k": int, "foreground": [[g, k], ...]}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/scripts/test_simulate_gated_omop.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import numpy as np
import pandas as pd

from simulate_gated_omop import build_gated_beta, simulate_gated


def _gb():
    rng = np.random.default_rng(0)
    return build_gated_beta(
        n_background_concepts=10, n_group_concepts=5,
        background_k=3, foreground=(("rare_dx", 2),), rng=rng, bleed=0.05)


def test_beta_is_row_stochastic_and_blocks_labeled():
    gb = _gb()
    assert gb.beta.shape == (5, 15)  # 3 bg + 2 fg topics; 10 bg + 5 group concepts
    np.testing.assert_allclose(gb.beta.sum(axis=1), 1.0, atol=1e-9)
    assert gb.topic_blocks == ["background", "background", "background",
                               "rare_dx", "rare_dx"]


def test_foreground_topics_concentrate_on_group_concepts():
    gb = _gb()
    grp_cols = [list(gb.concept_ids).index(c) for c in gb.group_concepts["rare_dx"]]
    # each rare_dx foreground topic puts the majority of its mass on rare concepts
    fg = gb.beta[3:]
    assert (fg[:, grp_cols].sum(axis=1) > 0.8).all()


def test_simulate_emits_expected_columns_and_gating():
    gb = _gb()
    events, persons, oracle = simulate_gated(
        gb, n_patients=200, group_props={"common": 0.8, "rare_dx": 0.2},
        foreground=(("rare_dx", 2),), visits_per_patient_mean=3.0,
        codes_per_visit_mean=6.0, age_means={"common": 55.0, "rare_dx": 70.0},
        theta_alpha=0.3, seed=1)
    assert set(events.columns) == {
        "person_id", "visit_occurrence_id", "concept_id", "concept_name",
        "source_cohort", "true_topic_id", "true_block"}
    assert set(persons.columns) == {"person_id", "source_cohort", "sex", "age"}
    # common patients (no foreground block) NEVER emit a foreground topic
    common_pids = set(persons.loc[persons.source_cohort == "common", "person_id"])
    common_ev = events[events.person_id.isin(common_pids)]
    assert (common_ev.true_block == "background").all()
    # rare_dx patients DO emit some foreground
    rare_ev = events[~events.person_id.isin(common_pids)]
    assert (rare_ev.true_block == "rare_dx").any()
    assert oracle["background_k"] == 3
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && python -m pytest tests/scripts/test_simulate_gated_omop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simulate_gated_omop'`.

- [ ] **Step 3: Implement the core**

Create `scripts/simulate_gated_omop.py` (CLI added in Task 3; this task is the pure core):

```python
"""Generate synthetic gated background/foreground OMOP data for STM.

Plants a background concept block shared by all patients plus per-group
distinctive concept blocks. A patient's group selects which foreground topics
it may express (background ∪ its group's foreground); a group with no foreground
block (e.g. a large "common" cohort) emits background only. This is the local
analogue of the gated STM the dashboard renders.

Oracle columns true_topic_id / true_block are evaluation-only; the fitter must
not read them. See docs/superpowers/specs/2026-06-23-gated-stm-local-dashboard-design.md
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = Path("data/simulated")


@dataclass(frozen=True)
class GatedBeta:
    beta: np.ndarray            # (K_total, V) row-stochastic
    concept_ids: np.ndarray     # (V,) int
    concept_names: dict         # cid -> "C<cid>"
    topic_blocks: list          # length K_total: "background" or group label
    group_concepts: dict        # group -> list[int] concept_ids
    background_concepts: list    # list[int] concept_ids


def build_gated_beta(*, n_background_concepts, n_group_concepts, background_k,
                     foreground, rng, bleed=0.1) -> GatedBeta:
    """Build a block-structured beta. Background topics live on background
    concepts; each group's foreground topics live on that group's distinctive
    concepts plus a small `bleed` of background mass."""
    groups = [g for g, _ in foreground]
    V = n_background_concepts + n_group_concepts * len(groups)
    concept_ids = np.arange(V, dtype=np.int64)
    concept_names = {int(c): f"C{int(c)}" for c in concept_ids}

    bg_concepts = list(range(n_background_concepts))
    group_concepts = {}
    start = n_background_concepts
    for g in groups:
        group_concepts[g] = list(range(start, start + n_group_concepts))
        start += n_group_concepts

    rows = []
    topic_blocks = []
    # Background topics: Dirichlet over background concepts only.
    for _ in range(background_k):
        v = np.zeros(V)
        v[bg_concepts] = rng.dirichlet(np.full(len(bg_concepts), 0.5))
        rows.append(v)
        topic_blocks.append("background")
    # Foreground topics per group: mostly the group's concepts + `bleed` of bg.
    fg_sizes = dict(foreground)
    for g in groups:
        for _ in range(fg_sizes[g]):
            v = np.zeros(V)
            v[group_concepts[g]] = (1.0 - bleed) * rng.dirichlet(
                np.full(n_group_concepts, 0.5))
            v[bg_concepts] = bleed * rng.dirichlet(np.full(len(bg_concepts), 0.5))
            rows.append(v)
            topic_blocks.append(g)

    beta = np.vstack(rows)
    beta = beta / beta.sum(axis=1, keepdims=True)
    return GatedBeta(beta=beta, concept_ids=concept_ids,
                     concept_names=concept_names, topic_blocks=topic_blocks,
                     group_concepts=group_concepts, background_concepts=bg_concepts)


def _allowed_topic_indices(topic_blocks, group):
    """Background topics ∪ the group's foreground topics (background-only if the
    group has no foreground block)."""
    return [i for i, b in enumerate(topic_blocks)
            if b == "background" or b == group]


def simulate_gated(gb: GatedBeta, *, n_patients, group_props, foreground,
                   visits_per_patient_mean, codes_per_visit_mean, age_means,
                   theta_alpha, seed):
    rng = np.random.default_rng(seed)
    groups = list(group_props.keys())
    probs = np.array([group_props[g] for g in groups], dtype=np.float64)
    probs = probs / probs.sum()
    K_total, V = gb.beta.shape

    ev_rows = []
    person_rows = []
    visit_counter = 0
    for p in range(n_patients):
        g = groups[int(rng.choice(len(groups), p=probs))]
        sex = "M" if rng.random() < 0.5 else "F"
        age = int(max(18, min(95, rng.normal(age_means.get(g, 60.0), 8.0))))
        person_rows.append((p, g, sex, age))

        allowed = _allowed_topic_indices(gb.topic_blocks, g)
        alpha = np.full(len(allowed), theta_alpha)
        theta_allowed = rng.dirichlet(alpha)
        theta = np.zeros(K_total)
        theta[allowed] = theta_allowed

        n_visits = max(1, int(rng.poisson(visits_per_patient_mean)))
        for _ in range(n_visits):
            visit_counter += 1
            n_codes = max(1, int(rng.poisson(codes_per_visit_mean)))
            z = rng.choice(K_total, size=n_codes, p=theta)
            for zi in z:
                w_col = rng.choice(V, p=gb.beta[zi])
                cid = int(gb.concept_ids[w_col])
                ev_rows.append((p, visit_counter, cid, gb.concept_names[cid],
                                g, int(zi), gb.topic_blocks[zi]))

    events = pd.DataFrame(ev_rows, columns=[
        "person_id", "visit_occurrence_id", "concept_id", "concept_name",
        "source_cohort", "true_topic_id", "true_block"])
    persons = pd.DataFrame(person_rows, columns=[
        "person_id", "source_cohort", "sex", "age"])
    oracle = {
        "background_concepts": gb.background_concepts,
        "group_concepts": {g: gb.group_concepts[g] for g in gb.group_concepts},
        "topic_blocks": gb.topic_blocks,
        "background_k": sum(1 for b in gb.topic_blocks if b == "background"),
        "foreground": [[g, k] for g, k in foreground],
    }
    return events, persons, oracle
```

- [ ] **Step 4: Run to verify they pass**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && python -m pytest tests/scripts/test_simulate_gated_omop.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/simulate_gated_omop.py tests/scripts/test_simulate_gated_omop.py
git commit -m "$(printf 'feat(sim): gated background/foreground OMOP simulator core\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 3: Simulator CLI + file outputs

**Files:**
- Modify: `scripts/simulate_gated_omop.py` (add `main`/CLI)
- Test: `tests/scripts/test_simulate_gated_omop.py` (add a CLI output test)

**Interfaces:**
- Consumes: `build_gated_beta`, `simulate_gated` (Task 2).
- Produces: a `main(argv)` that writes three files to `--output-dir`:
  `gated_omop_N<n>_seed<s>.parquet` (events), `gated_person_N<n>_seed<s>.parquet`
  (persons), `gated_oracle_N<n>_seed<s>.json` (oracle). CLI args: `--n-patients`,
  `--seed`, `--background-k`, `--foreground` (`"g:K,g:K"`), `--group-props`
  (`"g:frac,g:frac"`), `--n-background-concepts`, `--n-group-concepts`,
  `--age-means` (`"g:mean,g:mean"`), `--bleed`, `--theta-alpha`,
  `--visits-per-patient-mean`, `--codes-per-visit-mean`, `--output-dir`.

- [ ] **Step 1: Write the failing CLI test**

```python
# append to tests/scripts/test_simulate_gated_omop.py
def test_cli_writes_three_files(tmp_path):
    import json
    from simulate_gated_omop import main
    rc = main([
        "--n-patients", "150", "--seed", "2",
        "--background-k", "3", "--foreground", "rare_dx:2",
        "--group-props", "common:0.8,rare_dx:0.2",
        "--age-means", "common:55,rare_dx:70",
        "--n-background-concepts", "10", "--n-group-concepts", "5",
        "--output-dir", str(tmp_path)])
    assert rc == 0
    ev = tmp_path / "gated_omop_N150_seed2.parquet"
    pe = tmp_path / "gated_person_N150_seed2.parquet"
    orc = tmp_path / "gated_oracle_N150_seed2.json"
    assert ev.exists() and pe.exists() and orc.exists()
    persons = pd.read_parquet(pe)
    assert set(persons.source_cohort.unique()) <= {"common", "rare_dx"}
    oracle = json.loads(orc.read_text())
    assert oracle["foreground"] == [["rare_dx", 2]]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && python -m pytest tests/scripts/test_simulate_gated_omop.py -k cli -v`
Expected: FAIL — `main` not defined / `ImportError`.

- [ ] **Step 3: Implement the CLI**

Append to `scripts/simulate_gated_omop.py`:

```python
def _parse_pairs(s, valtype):
    out = []
    for piece in s.split(","):
        k, _, v = piece.partition(":")
        out.append((k.strip(), valtype(v)))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-patients", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--background-k", type=int, default=3)
    p.add_argument("--foreground", default="rare_dx:2",
                   help="per-group foreground sizes 'g:K,g:K'")
    p.add_argument("--group-props", default="common:0.99,rare_dx:0.01",
                   help="group proportions 'g:frac,g:frac' (need not sum to 1)")
    p.add_argument("--age-means", default="common:55,rare_dx:70")
    p.add_argument("--n-background-concepts", type=int, default=40)
    p.add_argument("--n-group-concepts", type=int, default=12)
    p.add_argument("--bleed", type=float, default=0.1)
    p.add_argument("--theta-alpha", type=float, default=0.3)
    p.add_argument("--visits-per-patient-mean", type=float, default=3.0)
    p.add_argument("--codes-per-visit-mean", type=float, default=8.0)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    foreground = tuple((g, int(k)) for g, k in _parse_pairs(args.foreground, int))
    group_props = dict(_parse_pairs(args.group_props, float))
    age_means = dict(_parse_pairs(args.age_means, float))

    rng = np.random.default_rng(args.seed)
    gb = build_gated_beta(
        n_background_concepts=args.n_background_concepts,
        n_group_concepts=args.n_group_concepts,
        background_k=args.background_k, foreground=foreground,
        rng=rng, bleed=args.bleed)
    events, persons, oracle = simulate_gated(
        gb, n_patients=args.n_patients, group_props=group_props,
        foreground=foreground,
        visits_per_patient_mean=args.visits_per_patient_mean,
        codes_per_visit_mean=args.codes_per_visit_mean,
        age_means=age_means, theta_alpha=args.theta_alpha, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"N{args.n_patients}_seed{args.seed}"
    events.to_parquet(args.output_dir / f"gated_omop_{stem}.parquet", index=False)
    persons.to_parquet(args.output_dir / f"gated_person_{stem}.parquet", index=False)
    (args.output_dir / f"gated_oracle_{stem}.json").write_text(
        json.dumps(oracle, indent=2))
    log.info("wrote gated sim: %d events, %d patients, groups=%s",
             len(events), len(persons), list(group_props))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && python -m pytest tests/scripts/test_simulate_gated_omop.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/simulate_gated_omop.py tests/scripts/test_simulate_gated_omop.py
git commit -m "$(printf 'feat(sim): gated simulator CLI + parquet/oracle outputs\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 4: Local gated-STM fitter

**Files:**
- Create: `analysis/local/fit_stm_local.py`
- Test: `tests/scripts/test_fit_stm_local.py` (a small in-process fit asserting checkpoint shape + metadata; the recovery assertion is Task 5)

**Interfaces:**
- Consumes: the simulator outputs (Task 3), `TopicBlockPartition`, `StreamingSTM(topic_blocks=, doc_group_col=, covariate_formula=)`, `STMModel`, `build_patient_covariate_df`, `load_omop_parquet`, `to_bow_dataframe`, `doc_spec_from_cli`.
- Produces: `main(argv) -> int` writing an `STMModel` checkpoint to `--out-dir` (via `STMModel.save`) with metadata `corpus_manifest` (incl `topic_block_spec`, `vocab`, `name_by_id`, `min_patient_count`, `cohort`, `doc_spec`), `covariate_manifest`, `model_class="stm"`, `concept_names`/`concept_domains`; and a sibling `covariates.parquet` (the per-doc covariate vectors + group labels) for the dashboard's offline masked prevalence.

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_fit_stm_local.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis" / "local"))

import json
import numpy as np


def _make_sim(tmp_path):
    from simulate_gated_omop import main as sim_main
    sim_main([
        "--n-patients", "300", "--seed", "5",
        "--background-k", "3", "--foreground", "rare_dx:2",
        "--group-props", "common:0.7,rare_dx:0.3",
        "--age-means", "common:55,rare_dx:72",
        "--n-background-concepts", "12", "--n-group-concepts", "6",
        "--codes-per-visit-mean", "6", "--output-dir", str(tmp_path)])
    return (tmp_path / "gated_omop_N300_seed5.parquet",
            tmp_path / "gated_person_N300_seed5.parquet")


def test_fit_stm_local_writes_gated_checkpoint(tmp_path):
    from fit_stm_local import main as fit_main
    omop, person = _make_sim(tmp_path)
    out = tmp_path / "ckpt"
    rc = fit_main([
        "--omop", str(omop), "--person", str(person),
        "--K", "5", "--background-k", "3", "--foreground", "rare_dx:2",
        "--covariate-formula", "~ C(sex) + age",
        "--max-iter", "8", "--out-dir", str(out)])
    assert rc == 0
    manifest = json.loads((out / "manifest.json").read_text())
    cm = manifest["metadata"]["corpus_manifest"]
    assert cm["topic_block_spec"]["background_k"] == 3
    assert cm["topic_block_spec"]["foreground"] == [["rare_dx", 2]]
    assert manifest["metadata"]["model_class"] == "stm"
    assert (out / "covariates.parquet").exists()
    lam = np.load(out / "params" / "lambda.npy")
    assert lam.shape[0] == 5  # K rows
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && python -m pytest tests/scripts/test_fit_stm_local.py -v`
Expected: FAIL — `fit_stm_local` not defined.

- [ ] **Step 3: Implement the fitter**

Create `analysis/local/fit_stm_local.py`:

```python
"""End-to-end local: gated sim parquet -> OnlineSTM via the shim -> checkpoint.

Sibling of fit_lda_local.py for gated STM. Builds a local SparkSession, loads
the gated OMOP + person parquets, builds the patient_cohort BOW (doc_id =
"source_cohort:person_id") and the covariate DataFrame (~ C(sex) + age, keyed
per person), and fits StreamingSTM with a TopicBlockPartition + doc_group_col.
Saves the STMModel checkpoint with full metadata + a covariates.parquet so the
local dashboard can compute masked prevalence offline.

source_cohort is the gating group label, NOT a covariate (it must not be in the
formula); it is materialized from doc_id, exactly as the cloud driver does.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

from charmpheno.omop import doc_spec_from_cli, load_omop_parquet, to_bow_dataframe
from charmpheno.omop.covariates import build_patient_covariate_df
from spark_vi.models.topic.partition import TopicBlockPartition
from spark_vi.mllib.topic.stm import StreamingSTM

log = logging.getLogger(__name__)


def _build_spark() -> SparkSession:
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    return (
        SparkSession.builder.master("local[2]")
        .appName("fit_stm_local")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
        .getOrCreate()
    )


def _parse_foreground(s):
    out = []
    for piece in s.split(","):
        g, _, k = piece.partition(":")
        out.append((g.strip(), int(k)))
    return tuple(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--omop", type=Path, required=True)
    p.add_argument("--person", type=Path, required=True)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--background-k", type=int, required=True)
    p.add_argument("--foreground", required=True, help="'g:K,g:K'")
    p.add_argument("--group-var", default="source_cohort")
    p.add_argument("--covariate-formula", default="~ C(sex) + age")
    p.add_argument("--categorical-cols", default="sex")
    p.add_argument("--continuous-cols", default="age")
    p.add_argument("--max-iter", type=int, default=40)
    p.add_argument("--subsampling-rate", type=float, default=1.0)
    p.add_argument("--tau0", type=float, default=64.0)
    p.add_argument("--kappa", type=float, default=0.7)
    p.add_argument("--min-patient-count", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    foreground = _parse_foreground(args.foreground)
    partition = TopicBlockPartition(group_var=args.group_var,
                                    background_k=args.background_k,
                                    foreground=foreground)
    if partition.K != args.K:
        raise SystemExit(f"partition K {partition.K} != --K {args.K}")
    cat_cols = [c for c in args.categorical_cols.split(",") if c]
    cont_cols = [c for c in args.continuous_cols.split(",") if c]
    doc_spec = doc_spec_from_cli("patient_cohort", min_doc_length=None)

    spark = _build_spark()
    try:
        omop = load_omop_parquet(str(args.omop), spark=spark)
        bow_df, vocab_map = to_bow_dataframe(omop, doc_spec=doc_spec)
        # source_cohort from doc_id (gating label; not a covariate).
        bow_df = bow_df.withColumn(
            "source_cohort", F.split(F.col("doc_id"), ":").getItem(0))

        person_df = spark.read.parquet(str(args.person))
        cov_df, model_spec, covariate_names = build_patient_covariate_df(
            person_df, covariate_formula=args.covariate_formula,
            categorical_cols=cat_cols, continuous_cols=cont_cols,
            key_cols=("person_id",))
        joined = bow_df.join(F.broadcast(cov_df), on="person_id", how="inner")

        est = StreamingSTM(
            K=args.K, features_col="features", covariates_col="covariates",
            covariate_names=covariate_names, topic_blocks=partition,
            doc_group_col="source_cohort", random_seed=args.seed)
        model = est.fit(joined, max_iter=args.max_iter,
                        subsampling_rate=args.subsampling_rate,
                        tau0=args.tau0, kappa=args.kappa)

        # Concept names from the simulator's concept_name column.
        name_rows = (omop.select("concept_id", "concept_name")
                     .dropDuplicates(["concept_id"]).collect())
        name_by_id = {int(r["concept_id"]): r["concept_name"] for r in name_rows}
        vocab_list = [None] * len(vocab_map)
        for cid, idx in vocab_map.items():
            vocab_list[idx] = cid

        model.metadata["corpus_manifest"] = {
            "cdr": "local", "source_table": "condition_occurrence",
            "cohort": "gated_sim", "prior_obs_days": 0, "person_mod": 1,
            "doc_spec": doc_spec.manifest(), "vocab_size": len(vocab_map),
            "vocab": vocab_list, "name_by_id": name_by_id,
            "min_patient_count": args.min_patient_count,
            "topic_block_spec": partition.to_dict(),
        }
        model.metadata["covariate_manifest"] = {
            "covariate_formula": args.covariate_formula,
            "categorical_cols": cat_cols, "continuous_cols": cont_cols,
            "covariate_names": covariate_names,
        }
        model.metadata["model_class"] = "stm"
        model.metadata["concept_names"] = {str(k): v for k, v in name_by_id.items()}
        model.metadata["concept_domains"] = {str(k): "condition" for k in name_by_id}

        args.out_dir.mkdir(parents=True, exist_ok=True)
        model.save(args.out_dir)

        # Persist per-doc covariate vectors + group label for offline masked
        # prevalence (one row per joined doc).
        from pyspark.ml.functions import vector_to_array
        (joined.select("person_id", "source_cohort",
                       vector_to_array("covariates").alias("covariates"))
         .toPandas().to_parquet(args.out_dir / "covariates.parquet", index=False))

        log.info("wrote gated STM checkpoint to %s (K=%d, V=%d)",
                 args.out_dir, args.K, len(vocab_map))
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && poetry run python -m pytest tests/scripts/test_fit_stm_local.py -v`
Expected: PASS. (Use `poetry run` so the local-Spark venv with a working `distutils` is used — the same venv the other local fitters run under.)

- [ ] **Step 5: Commit**

```bash
git add analysis/local/fit_stm_local.py tests/scripts/test_fit_stm_local.py
git commit -m "$(printf 'feat(local): fit_stm_local — local gated STM fit via the shim\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 5: End-to-end harness recovery test

**Files:**
- Test: `tests/scripts/test_fit_stm_local.py` (add the recovery assertion)

**Interfaces:**
- Consumes: the simulator + `fit_stm_local` (Tasks 2-4) + the oracle JSON.

- [ ] **Step 1: Write the recovery test**

```python
# append to tests/scripts/test_fit_stm_local.py
def test_fit_recovers_planted_rare_foreground(tmp_path):
    from simulate_gated_omop import main as sim_main
    from fit_stm_local import main as fit_main
    from spark_vi.io import load_result

    sim_main([
        "--n-patients", "600", "--seed", "7",
        "--background-k", "3", "--foreground", "rare_dx:2",
        "--group-props", "common:0.7,rare_dx:0.3",
        "--age-means", "common:55,rare_dx:72",
        "--n-background-concepts", "12", "--n-group-concepts", "6",
        "--codes-per-visit-mean", "6", "--output-dir", str(tmp_path)])
    omop = tmp_path / "gated_omop_N600_seed7.parquet"
    person = tmp_path / "gated_person_N600_seed7.parquet"
    oracle = json.loads((tmp_path / "gated_oracle_N600_seed7.json").read_text())
    out = tmp_path / "ckpt"
    fit_main([
        "--omop", str(omop), "--person", str(person),
        "--K", "5", "--background-k", "3", "--foreground", "rare_dx:2",
        "--covariate-formula", "~ C(sex) + age",
        "--max-iter", "40", "--out-dir", str(out)])

    result = load_result(out)
    lam = result.global_params["lambda"]
    beta = lam / lam.sum(axis=1, keepdims=True)
    vocab = result.metadata["vocab"]               # index -> concept_id
    cid_to_idx = {int(c): i for i, c in enumerate(vocab)}
    rare_cids = oracle["group_concepts"]["rare_dx"]
    rare_cols = [cid_to_idx[c] for c in rare_cids if c in cid_to_idx]
    # foreground topics are indices 3,4 (after 3 background); at least one must
    # concentrate on the rare-dx distinctive concepts.
    fg_mass = beta[3:][:, rare_cols].sum(axis=1).max()
    bg_mass = beta[:3][:, rare_cols].sum(axis=1).max()
    assert fg_mass > 0.4, fg_mass
    assert bg_mass < 0.15, bg_mass
```

- [ ] **Step 2: Run to verify it passes**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && poetry run python -m pytest tests/scripts/test_fit_stm_local.py -k recovers -v`
Expected: PASS. If it FAILS (fg_mass low), the gated fit didn't recover the planted structure — debug the fitter wiring (partition K, doc_group_col threading, foreground sizes) before continuing; do not lower the thresholds.

- [ ] **Step 3: Commit**

```bash
git add tests/scripts/test_fit_stm_local.py
git commit -m "$(printf 'test(local): end-to-end gated sim->fit recovery of rare foreground\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Final verification

- [ ] Engine + harness tests pass:

```bash
cd spark-vi && python -m pytest tests/test_topic_block_partition.py -v
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && poetry run python -m pytest tests/scripts/test_simulate_gated_omop.py tests/scripts/test_fit_stm_local.py -v
```
Expected: all PASS.

- [ ] The checkpoint produced by Task 4/5 is the input for Plan 2b (dashboard). Confirm `<out-dir>/manifest.json` has `metadata.corpus_manifest.topic_block_spec` and `<out-dir>/covariates.parquet` exists — Plan 2b reads both.

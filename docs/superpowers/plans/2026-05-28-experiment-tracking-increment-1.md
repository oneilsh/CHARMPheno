# Experiment Tracking — Increment 1 (MVP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author experiments locally in markdown files with YAML frontmatter, run them on the cluster with `make next-exp` / `make exp ID=N`, get a sanitized `summary.md` per run for copy/paste back.

**Architecture:** A Python wrapper (`scripts/run_experiment.py`) reads experiment records from `docs/experiments/NNNN-slug.md`, merges defaults from `experiments/defaults/*.yaml`, dispatches `lda_bigquery_cloud.py` via spark-submit subprocess (with stdout teed + sanitized to `$RUNS_DIR/NNNN-slug/summary.md`), then dispatches `eval_coherence_cloud.py` and appends eval section. Three Makefile targets are thin shells around the wrapper. The wrapper does **plain stdout capture** in this increment — structured per-iter parsing and SIGTERM trapping are Increment 2.

**Tech Stack:** Python 3.11, PyYAML (for frontmatter), subprocess, pytest, GNU Make.

**Scope boundary for this plan:** Increment 1 of three. Out of scope here: streaming per-iter markdown structure, SIGTERM trap, `eval-exp` standalone target, `warm_start_from`, `build-dashboard-exp`, cross-link template enforcement. Those are Increments 2 and 3 — separate plans.

---

## File Structure

**Created:**
- `experiments/defaults/_base.yaml` — cross-cohort defaults
- `experiments/defaults/general.yaml` — general-population cohort overrides
- `experiments/defaults/cancer.yaml` — cancer cohort overrides
- `experiments/defaults/dementia.yaml` — dementia cohort overrides
- `scripts/run_experiment.py` — single-file Python wrapper; pure functions + `main()`
- `scripts/tests/test_run_experiment.py` — unit tests for pure functions
- `scripts/tests/fixtures/sample_experiment.md` — test fixture
- `scripts/tests/fixtures/sample_defaults/_base.yaml` — test fixture
- `scripts/tests/fixtures/sample_defaults/dementia.yaml` — test fixture
- `docs/experiments/0001-pilot.md` — skeleton record for end-to-end smoke

**Modified:**
- `analysis/cloud/Makefile` — three new targets: `next-exp`, `exp`, `summary`
- `analysis/cloud/lda_bigquery_cloud.py:420-429` — strip the `transform sample` block that prints hashed person rows

**Not touched:**
- `analysis/cloud/hdp_bigquery_cloud.py` — HDP support deferred to Increment 3 (MVP runs LDA only)
- `analysis/cloud/eval_coherence_cloud.py` — wrapper invokes it as-is via subprocess
- Existing per-cohort targets (`lda-bq-fit-eval-cancer`, etc.) — coexist unchanged

---

## Module decomposition within `scripts/run_experiment.py`

The wrapper is a single file with focused pure functions plus a thin `main()`:

| Symbol | Type | Responsibility |
|---|---|---|
| `REPO_ROOT` | constant | `Path(__file__).resolve().parent.parent` |
| `EXPERIMENTS_DIR` | constant | `REPO_ROOT / "docs" / "experiments"` |
| `DEFAULTS_DIR` | constant | `REPO_ROOT / "experiments" / "defaults"` |
| `PATIENT_PATTERNS` | constant | regex list for sanitization |
| `read_frontmatter(path)` | pure | parses YAML frontmatter from `.md`; returns `dict` |
| `load_defaults(cohort, defaults_dir)` | pure | reads `_base.yaml` + `<cohort>.yaml`; returns merged `dict` |
| `merge_config(base, override)` | pure | shallow dict merge; later wins |
| `find_next_pending(experiments_dir)` | pure | returns lowest-id experiment `.md` Path with `status: pending`, or None |
| `find_by_id(experiments_dir, id)` | pure | returns the `.md` Path for given id, or raises |
| `sanitize_line(line, patterns)` | pure | returns `line` or `None` (drop) |
| `build_lda_args(effective, save_dir, resume_from)` | pure | returns `list[str]` of CLI args for `lda_bigquery_cloud.py` |
| `build_eval_args(checkpoint_dir, effective)` | pure | returns `list[str]` of CLI args for `eval_coherence_cloud.py` |
| `build_spark_submit_cmd(script, args, repo_root)` | pure | returns full `list[str]` for `subprocess.Popen` |
| `run_subprocess_tee_sanitize(cmd, summary_path, patterns)` | I/O | runs subprocess, tees stdout to terminal + sanitized append to `summary_path`; returns exit code |
| `write_summary_header(summary_path, exp_id, slug, effective)` | I/O | writes initial header block |
| `append_eval_section(summary_path, eval_stdout)` | I/O | appends `## Eval (NPMI)` block |
| `main(argv)` | I/O | orchestrates |

---

## Task 1: Defaults YAML files

**Files:**
- Create: `experiments/defaults/_base.yaml`
- Create: `experiments/defaults/general.yaml`
- Create: `experiments/defaults/cancer.yaml`
- Create: `experiments/defaults/dementia.yaml`

- [ ] **Step 1: Create `_base.yaml`**

Write to `experiments/defaults/_base.yaml`:

```yaml
# Cross-cohort defaults for experiment-tracking system.
# Layered as: _base.yaml -> <cohort>.yaml -> per-experiment frontmatter overrides.
# The cluster writes the effective (merged) config into summary.md at fit time
# so changes here do not retroactively affect already-run experiments.
model_class: lda
source_table: condition_era
doc_unit: patient_year
doc_min_length: 20
K: 40
max_iter: 20
vocab_size: 10000
min_df: 20
min_patient_count: 20
subsampling_rate: 0.2
tau0: 64
kappa: 0.7
save_interval: 5
print_topics_every: 1
person_mod: 10
top_n_tokens: 6
seed: 42
optimize_doc_concentration: true
optimize_topic_concentration: false
```

- [ ] **Step 2: Create `general.yaml`**

Write to `experiments/defaults/general.yaml`:

```yaml
# General-population cohort. Inherits all of _base.yaml.
cohort: none
```

(`cohort: none` is what `lda_bigquery_cloud.py` accepts as "no cohort filter"; the driver maps it to `None` internally at [lda_bigquery_cloud.py:259-260](analysis/cloud/lda_bigquery_cloud.py#L259-L260).)

- [ ] **Step 3: Create `cancer.yaml`**

Write to `experiments/defaults/cancer.yaml`:

```yaml
# Cancer-cohort overrides on top of _base.yaml.
cohort: cancer
```

- [ ] **Step 4: Create `dementia.yaml`**

Write to `experiments/defaults/dementia.yaml`:

```yaml
# Dementia-cohort overrides on top of _base.yaml.
cohort: dementia
```

- [ ] **Step 5: Verify all four files parse**

Run:
```bash
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno
poetry run python -c "import yaml; from pathlib import Path; [print(p.name, yaml.safe_load(p.read_text())) for p in sorted(Path('experiments/defaults').glob('*.yaml'))]"
```
Expected: prints four dicts without errors.

- [ ] **Step 6: Commit**

```bash
git add experiments/defaults/
git commit -m "feat(experiments): add cross-cohort and per-cohort defaults files"
```

---

## Task 2: Frontmatter reader

**Files:**
- Create: `scripts/run_experiment.py` (start the file)
- Create: `scripts/tests/test_run_experiment.py`
- Create: `scripts/tests/fixtures/sample_experiment.md`

- [ ] **Step 1: Create the test fixture**

Write to `scripts/tests/fixtures/sample_experiment.md`:

```markdown
---
id: 42
slug: try-k60-dementia
status: pending
model_class: lda
cohort: dementia
created: 2026-05-28
K: 60
---

# Experiment 0042 — try-k60-dementia

## Intent
Test fixture body. Should be ignored by frontmatter reader.
```

- [ ] **Step 2: Write the failing test**

Write to `scripts/tests/test_run_experiment.py`:

```python
"""Unit tests for scripts/run_experiment.py."""
from __future__ import annotations

from pathlib import Path

import pytest

# scripts/tests/conftest.py already inserts scripts/ into sys.path.
import run_experiment as rx

FIXTURES = Path(__file__).parent / "fixtures"


def test_read_frontmatter_parses_yaml_block():
    path = FIXTURES / "sample_experiment.md"
    fm = rx.read_frontmatter(path)
    assert fm["id"] == 42
    assert fm["slug"] == "try-k60-dementia"
    assert fm["status"] == "pending"
    assert fm["model_class"] == "lda"
    assert fm["cohort"] == "dementia"
    assert fm["K"] == 60


def test_read_frontmatter_raises_on_missing_block(tmp_path):
    path = tmp_path / "no_frontmatter.md"
    path.write_text("# No frontmatter\nJust a body.\n")
    with pytest.raises(ValueError, match="frontmatter"):
        rx.read_frontmatter(path)


def test_read_frontmatter_raises_on_unterminated_block(tmp_path):
    path = tmp_path / "bad.md"
    path.write_text("---\nid: 1\n# never closed\n")
    with pytest.raises(ValueError, match="frontmatter"):
        rx.read_frontmatter(path)
```

- [ ] **Step 3: Run tests to verify failure**

```bash
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno
poetry run pytest scripts/tests/test_run_experiment.py -v
```
Expected: collection error or `ImportError: No module named 'run_experiment'`.

- [ ] **Step 4: Create the wrapper file**

Write to `scripts/run_experiment.py`:

```python
"""Experiment-tracking runner: reads a docs/experiments/NNNN-slug.md
record, merges defaults, dispatches lda_bigquery_cloud.py via spark-submit,
captures sanitized stdout to summary.md in the run dir, then runs eval.

See docs/superpowers/specs/2026-05-28-experiment-tracking-design.md.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "docs" / "experiments"
DEFAULTS_DIR = REPO_ROOT / "experiments" / "defaults"


def read_frontmatter(path: Path) -> dict:
    """Parse YAML frontmatter from a markdown file.

    Frontmatter is delimited by leading and trailing `---` lines on their own.
    Returns the parsed dict. Raises ValueError if absent or unterminated.
    """
    text = path.read_text()
    if not text.startswith("---\n"):
        raise ValueError(f"{path}: missing frontmatter block (expected leading '---')")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise ValueError(f"{path}: unterminated frontmatter block (no trailing '---')")
    yaml_text = text[4:end]
    return yaml.safe_load(yaml_text) or {}
```

- [ ] **Step 5: Run tests to verify pass**

```bash
poetry run pytest scripts/tests/test_run_experiment.py -v
```
Expected: 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_experiment.py scripts/tests/test_run_experiment.py scripts/tests/fixtures/sample_experiment.md
git commit -m "feat(scripts): run_experiment.py frontmatter reader + tests"
```

---

## Task 3: Defaults loader + config merger

**Files:**
- Modify: `scripts/run_experiment.py` — add `load_defaults` + `merge_config`
- Modify: `scripts/tests/test_run_experiment.py` — add tests
- Create: `scripts/tests/fixtures/sample_defaults/_base.yaml`
- Create: `scripts/tests/fixtures/sample_defaults/dementia.yaml`

- [ ] **Step 1: Create test fixtures**

Write to `scripts/tests/fixtures/sample_defaults/_base.yaml`:

```yaml
model_class: lda
K: 40
max_iter: 20
vocab_size: 10000
```

Write to `scripts/tests/fixtures/sample_defaults/dementia.yaml`:

```yaml
cohort: dementia
K: 50
```

- [ ] **Step 2: Write the failing tests**

Append to `scripts/tests/test_run_experiment.py`:

```python
def test_merge_config_later_wins():
    base = {"a": 1, "b": 2, "c": 3}
    override = {"b": 20, "d": 4}
    merged = rx.merge_config(base, override)
    assert merged == {"a": 1, "b": 20, "c": 3, "d": 4}
    # Inputs not mutated
    assert base == {"a": 1, "b": 2, "c": 3}
    assert override == {"b": 20, "d": 4}


def test_load_defaults_three_way_merge():
    fixtures = FIXTURES / "sample_defaults"
    effective = rx.load_defaults("dementia", fixtures)
    # base provides model_class, max_iter, vocab_size; dementia overrides K + adds cohort
    assert effective["model_class"] == "lda"
    assert effective["max_iter"] == 20
    assert effective["vocab_size"] == 10000
    assert effective["K"] == 50          # dementia override beats base
    assert effective["cohort"] == "dementia"


def test_load_defaults_missing_cohort_file_raises(tmp_path):
    (tmp_path / "_base.yaml").write_text("K: 40\n")
    with pytest.raises(FileNotFoundError, match="bogus"):
        rx.load_defaults("bogus", tmp_path)


def test_load_defaults_missing_base_raises(tmp_path):
    (tmp_path / "dementia.yaml").write_text("cohort: dementia\n")
    with pytest.raises(FileNotFoundError, match="_base.yaml"):
        rx.load_defaults("dementia", tmp_path)
```

- [ ] **Step 3: Run tests to verify failure**

```bash
poetry run pytest scripts/tests/test_run_experiment.py -v
```
Expected: 4 new tests FAIL (`AttributeError: module 'run_experiment' has no attribute 'merge_config'` or similar).

- [ ] **Step 4: Add `merge_config` + `load_defaults` to run_experiment.py**

Append to `scripts/run_experiment.py`:

```python
def merge_config(base: dict, override: dict) -> dict:
    """Shallow merge: returns a new dict with override taking precedence over base."""
    out = dict(base)
    out.update(override)
    return out


def load_defaults(cohort: str, defaults_dir: Path) -> dict:
    """Load _base.yaml then <cohort>.yaml and merge.

    Raises FileNotFoundError if either file is missing.
    """
    base_path = defaults_dir / "_base.yaml"
    cohort_path = defaults_dir / f"{cohort}.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"missing defaults file: {base_path}")
    if not cohort_path.exists():
        raise FileNotFoundError(f"missing defaults file: {cohort_path}")
    base = yaml.safe_load(base_path.read_text()) or {}
    cohort_overrides = yaml.safe_load(cohort_path.read_text()) or {}
    return merge_config(base, cohort_overrides)
```

- [ ] **Step 5: Run tests to verify pass**

```bash
poetry run pytest scripts/tests/test_run_experiment.py -v
```
Expected: all 7 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_experiment.py scripts/tests/test_run_experiment.py scripts/tests/fixtures/sample_defaults/
git commit -m "feat(scripts): defaults loader + config merger with three-way precedence"
```

---

## Task 4: Find-next-pending + find-by-id

**Files:**
- Modify: `scripts/run_experiment.py`
- Modify: `scripts/tests/test_run_experiment.py`

- [ ] **Step 1: Write the failing tests**

Append to `scripts/tests/test_run_experiment.py`:

```python
def _write_experiment(dir_path: Path, *, id: int, slug: str, status: str) -> Path:
    """Test helper: writes a minimal experiment record file."""
    path = dir_path / f"{id:04d}-{slug}.md"
    path.write_text(
        f"---\n"
        f"id: {id}\n"
        f"slug: {slug}\n"
        f"status: {status}\n"
        f"model_class: lda\n"
        f"cohort: dementia\n"
        f"---\n\n# {slug}\n"
    )
    return path


def test_find_next_pending_picks_lowest_id(tmp_path):
    _write_experiment(tmp_path, id=3, slug="c", status="pending")
    _write_experiment(tmp_path, id=1, slug="a", status="done")
    _write_experiment(tmp_path, id=2, slug="b", status="pending")
    result = rx.find_next_pending(tmp_path)
    assert result is not None
    assert result.name == "0002-b.md"


def test_find_next_pending_returns_none_when_no_pending(tmp_path):
    _write_experiment(tmp_path, id=1, slug="a", status="done")
    _write_experiment(tmp_path, id=2, slug="b", status="archived")
    assert rx.find_next_pending(tmp_path) is None


def test_find_next_pending_empty_dir_returns_none(tmp_path):
    assert rx.find_next_pending(tmp_path) is None


def test_find_next_pending_ignores_non_md_files(tmp_path):
    _write_experiment(tmp_path, id=1, slug="a", status="pending")
    (tmp_path / "notes.txt").write_text("ignore me")
    (tmp_path / "0099-draft.md.bak").write_text("ignore me too")
    result = rx.find_next_pending(tmp_path)
    assert result is not None
    assert result.name == "0001-a.md"


def test_find_by_id_returns_matching_path(tmp_path):
    _write_experiment(tmp_path, id=42, slug="try-k60", status="pending")
    _write_experiment(tmp_path, id=43, slug="other", status="pending")
    result = rx.find_by_id(tmp_path, 42)
    assert result.name == "0042-try-k60.md"


def test_find_by_id_raises_when_missing(tmp_path):
    _write_experiment(tmp_path, id=1, slug="a", status="pending")
    with pytest.raises(FileNotFoundError, match="0042"):
        rx.find_by_id(tmp_path, 42)
```

- [ ] **Step 2: Run tests to verify failure**

```bash
poetry run pytest scripts/tests/test_run_experiment.py -v
```
Expected: 6 new tests FAIL.

- [ ] **Step 3: Add functions to run_experiment.py**

Append to `scripts/run_experiment.py`:

```python
def _list_experiment_files(experiments_dir: Path) -> list[Path]:
    """All files matching NNNN-*.md in experiments_dir, sorted by id."""
    pattern = re.compile(r"^(\d{4})-.+\.md$")
    out = []
    for p in experiments_dir.iterdir():
        if pattern.match(p.name):
            out.append(p)
    out.sort(key=lambda p: p.name)
    return out


def find_next_pending(experiments_dir: Path) -> Path | None:
    """Return the lowest-id experiment file with status: pending, or None."""
    for p in _list_experiment_files(experiments_dir):
        try:
            fm = read_frontmatter(p)
        except ValueError:
            continue
        if fm.get("status") == "pending":
            return p
    return None


def find_by_id(experiments_dir: Path, exp_id: int) -> Path:
    """Return the experiment file with the given id. Raises FileNotFoundError if absent."""
    prefix = f"{exp_id:04d}-"
    for p in _list_experiment_files(experiments_dir):
        if p.name.startswith(prefix):
            return p
    raise FileNotFoundError(f"no experiment with id {exp_id:04d} in {experiments_dir}")
```

- [ ] **Step 4: Run tests to verify pass**

```bash
poetry run pytest scripts/tests/test_run_experiment.py -v
```
Expected: all 13 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_experiment.py scripts/tests/test_run_experiment.py
git commit -m "feat(scripts): find_next_pending + find_by_id with sequential-id matching"
```

---

## Task 5: Sanitization filter

**Files:**
- Modify: `scripts/run_experiment.py`
- Modify: `scripts/tests/test_run_experiment.py`

- [ ] **Step 1: Write the failing tests**

Append to `scripts/tests/test_run_experiment.py`:

```python
def test_sanitize_line_passes_normal_lines():
    line = "[iter 5] ELBO=-1.234e9  time=180s\n"
    assert rx.sanitize_line(line, rx.PATIENT_PATTERNS) == line


def test_sanitize_line_drops_person_hash_show_output():
    # PySpark .show() output format
    line = "|a1b2c3d4e5f6  |(60,[0,1],[0.5,0.5])  |\n"
    # The first column is 12 hex chars; without context this is fine, but the
    # surrounding .show()-header makes it identifiable as the transform-sample.
    # Hard to safely match by hash alone -- match by header context instead.
    assert rx.sanitize_line("|person_hash|topicDistribution|\n", rx.PATIENT_PATTERNS) is None


def test_sanitize_line_drops_lines_with_person_id_equals():
    line = "[debug] person_id=12345 has 17 tokens\n"
    assert rx.sanitize_line(line, rx.PATIENT_PATTERNS) is None


def test_sanitize_line_drops_lines_with_explicit_hash_prefix():
    line = "    hash:a1b2c3d4e5f6 -> topic 3\n"
    assert rx.sanitize_line(line, rx.PATIENT_PATTERNS) is None


def test_sanitize_line_drops_transform_sample_phase_marker():
    line = "[driver] >>> transform sample\n"
    assert rx.sanitize_line(line, rx.PATIENT_PATTERNS) is None


def test_sanitize_line_keeps_aggregate_person_count():
    line = "[driver]   OMOP: 1234567 rows, 89012 distinct persons\n"
    assert rx.sanitize_line(line, rx.PATIENT_PATTERNS) == line


def test_sanitize_line_keeps_iter_topic_prints():
    line = "  Topic 5:    192671  Type 2 diabetes mellitus  0.0234\n"
    assert rx.sanitize_line(line, rx.PATIENT_PATTERNS) == line
```

- [ ] **Step 2: Run tests to verify failure**

```bash
poetry run pytest scripts/tests/test_run_experiment.py -v
```
Expected: 7 new tests FAIL.

- [ ] **Step 3: Add sanitizer to run_experiment.py**

Append to `scripts/run_experiment.py`:

```python
# Patterns that indicate per-patient row-level info. Belt-and-suspenders for
# driver-side stripping; if a new driver path re-introduces patient prints,
# these catch them at the wrapper boundary before they reach summary.md.
PATIENT_PATTERNS: list[re.Pattern] = [
    re.compile(r"person_hash", re.IGNORECASE),
    re.compile(r"person_id\s*=\s*\S+"),
    re.compile(r"\bhash:[0-9a-f]{6,}", re.IGNORECASE),
    re.compile(r"transform sample", re.IGNORECASE),  # the phase marker bracketing it
]


def sanitize_line(line: str, patterns: list[re.Pattern]) -> str | None:
    """Return the line if safe to commit, or None to drop.

    Drops any line matching any patient-info pattern. Aggregate counts and
    topic-level prints (no per-patient identifiers) pass through.
    """
    for pat in patterns:
        if pat.search(line):
            return None
    return line
```

- [ ] **Step 4: Run tests to verify pass**

```bash
poetry run pytest scripts/tests/test_run_experiment.py -v
```
Expected: all 20 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_experiment.py scripts/tests/test_run_experiment.py
git commit -m "feat(scripts): patient-info sanitization filter for summary.md writes"
```

---

## Task 6: build_lda_args + build_eval_args

**Files:**
- Modify: `scripts/run_experiment.py`
- Modify: `scripts/tests/test_run_experiment.py`

- [ ] **Step 1: Write the failing tests**

Append to `scripts/tests/test_run_experiment.py`:

```python
def test_build_lda_args_required_fields(tmp_path):
    effective = {
        "model_class": "lda",
        "source_table": "condition_era",
        "doc_unit": "patient_year",
        "doc_min_length": 20,
        "K": 60,
        "max_iter": 20,
        "vocab_size": 10000,
        "min_df": 20,
        "min_patient_count": 20,
        "subsampling_rate": 0.2,
        "tau0": 64,
        "kappa": 0.7,
        "save_interval": 5,
        "print_topics_every": 1,
        "person_mod": 10,
        "top_n_tokens": 6,
        "seed": 42,
        "optimize_doc_concentration": True,
        "optimize_topic_concentration": False,
        "cohort": "dementia",
    }
    save_dir = tmp_path / "0042-try-k60-dementia"
    args = rx.build_lda_args(effective, save_dir, resume_from=None)
    # Required overrides
    assert "--save-dir" in args
    assert str(save_dir) in args
    assert "--K" in args
    assert "60" in args
    assert "--source-table" in args
    assert "condition_era" in args
    assert "--cohort" in args
    assert "dementia" in args
    # Resume not set when None
    assert "--resume-from" not in args


def test_build_lda_args_threads_resume_from(tmp_path):
    effective = {
        "model_class": "lda", "source_table": "condition_era",
        "doc_unit": "patient_year", "doc_min_length": 20,
        "K": 60, "max_iter": 20, "vocab_size": 10000,
        "min_df": 20, "min_patient_count": 20,
        "subsampling_rate": 0.2, "tau0": 64, "kappa": 0.7,
        "save_interval": 5, "print_topics_every": 1,
        "person_mod": 10, "top_n_tokens": 6, "seed": 42,
        "optimize_doc_concentration": True,
        "optimize_topic_concentration": False,
        "cohort": "dementia",
    }
    save_dir = tmp_path / "0042-try-k60"
    resume = tmp_path / "0042-try-k60"
    args = rx.build_lda_args(effective, save_dir, resume_from=resume)
    assert "--resume-from" in args
    idx = args.index("--resume-from")
    assert args[idx + 1] == str(resume)


def test_build_lda_args_omits_cohort_when_none():
    effective = {
        "model_class": "lda", "source_table": "condition_era",
        "doc_unit": "patient", "doc_min_length": 20,
        "K": 40, "max_iter": 20, "vocab_size": 10000,
        "min_df": 20, "min_patient_count": 20,
        "subsampling_rate": 0.2, "tau0": 64, "kappa": 0.7,
        "save_interval": 5, "print_topics_every": 1,
        "person_mod": 10, "top_n_tokens": 6, "seed": 42,
        "optimize_doc_concentration": True,
        "optimize_topic_concentration": False,
        "cohort": "none",
    }
    args = rx.build_lda_args(effective, Path("/tmp/foo"), resume_from=None)
    assert "--cohort" in args
    idx = args.index("--cohort")
    assert args[idx + 1] == "none"


def test_build_eval_args_minimum(tmp_path):
    checkpoint = tmp_path / "0042-try-k60"
    args = rx.build_eval_args(checkpoint, {"model_class": "lda"})
    assert "--checkpoint" in args
    assert str(checkpoint) in args
    assert "--model-class" in args
    assert "lda" in args
```

- [ ] **Step 2: Run tests to verify failure**

```bash
poetry run pytest scripts/tests/test_run_experiment.py -v
```
Expected: 4 new tests FAIL.

- [ ] **Step 3: Add builders to run_experiment.py**

Append to `scripts/run_experiment.py`:

```python
def build_lda_args(
    effective: dict, save_dir: Path, resume_from: Path | None,
) -> list[str]:
    """Build the CLI arg list for lda_bigquery_cloud.py from an effective config.

    The driver's argparse defaults handle anything not set here; we explicitly
    pass everything in `effective` to keep the cluster's behavior reproducible
    from the config alone.
    """
    args: list[str] = [
        "--save-dir", str(save_dir),
        "--save-interval", str(effective["save_interval"]),
        "--source-table", str(effective["source_table"]),
        "--doc-unit", str(effective["doc_unit"]),
        "--doc-min-length", str(effective["doc_min_length"]),
        "--K", str(effective["K"]),
        "--max-iter", str(effective["max_iter"]),
        "--vocab-size", str(effective["vocab_size"]),
        "--min-df", str(effective["min_df"]),
        "--min-patient-count", str(effective["min_patient_count"]),
        "--subsampling-rate", str(effective["subsampling_rate"]),
        "--tau0", str(effective["tau0"]),
        "--kappa", str(effective["kappa"]),
        "--print-topics-every", str(effective["print_topics_every"]),
        "--person-mod", str(effective["person_mod"]),
        "--top-n-tokens", str(effective["top_n_tokens"]),
        "--seed", str(effective["seed"]),
        "--cohort", str(effective["cohort"]),
    ]
    # BooleanOptionalAction in the driver: --optimize-doc-concentration / --no-...
    if effective.get("optimize_doc_concentration", True):
        args.append("--optimize-doc-concentration")
    else:
        args.append("--no-optimize-doc-concentration")
    if effective.get("optimize_topic_concentration", False):
        args.append("--optimize-topic-concentration")
    else:
        args.append("--no-optimize-topic-concentration")
    if resume_from is not None:
        args += ["--resume-from", str(resume_from)]
    return args


def build_eval_args(checkpoint_dir: Path, effective: dict) -> list[str]:
    """Build the CLI arg list for eval_coherence_cloud.py."""
    return [
        "--checkpoint", str(checkpoint_dir),
        "--model-class", str(effective.get("model_class", "lda")),
    ]
```

- [ ] **Step 4: Run tests to verify pass**

```bash
poetry run pytest scripts/tests/test_run_experiment.py -v
```
Expected: all 24 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_experiment.py scripts/tests/test_run_experiment.py
git commit -m "feat(scripts): build_lda_args + build_eval_args for driver dispatch"
```

---

## Task 7: Spark-submit command builder + summary header writer

**Files:**
- Modify: `scripts/run_experiment.py`
- Modify: `scripts/tests/test_run_experiment.py`

- [ ] **Step 1: Write the failing tests**

Append to `scripts/tests/test_run_experiment.py`:

```python
def test_build_spark_submit_cmd_structure(tmp_path):
    script = "/repo/analysis/cloud/lda_bigquery_cloud.py"
    script_args = ["--K", "60", "--save-dir", "/tmp/foo"]
    repo_root = tmp_path
    (repo_root / "spark-vi" / "dist").mkdir(parents=True)
    (repo_root / "spark-vi" / "dist" / "spark_vi.zip").touch()
    (repo_root / "charmpheno" / "dist").mkdir(parents=True)
    (repo_root / "charmpheno" / "dist" / "charmpheno.zip").touch()

    cmd = rx.build_spark_submit_cmd(script, script_args, repo_root)
    assert cmd[0] == "spark-submit"
    assert "--master" in cmd and "yarn" in cmd
    assert "--deploy-mode" in cmd and "client" in cmd
    assert "--py-files" in cmd
    # py-files value contains both zips comma-joined
    py_files_idx = cmd.index("--py-files")
    py_files_val = cmd[py_files_idx + 1]
    assert "spark_vi.zip" in py_files_val
    assert "charmpheno.zip" in py_files_val
    # Script + script args at the end, in order
    assert cmd[-len(script_args) - 1] == script
    assert cmd[-len(script_args):] == script_args


def test_write_summary_header_creates_file(tmp_path):
    summary_path = tmp_path / "summary.md"
    effective = {"model_class": "lda", "cohort": "dementia", "K": 60}
    rx.write_summary_header(summary_path, exp_id=42, slug="try-k60", effective=effective)
    text = summary_path.read_text()
    assert "# Experiment 0042" in text
    assert "try-k60" in text
    assert "## Effective config" in text
    assert "K: 60" in text
    assert "cohort: dementia" in text
    assert "model_class: lda" in text


def test_write_summary_header_appends_session_marker(tmp_path):
    summary_path = tmp_path / "summary.md"
    summary_path.write_text("# Existing summary\n\n## Fit session 1\n... (old) ...\n")
    effective = {"model_class": "lda", "cohort": "dementia", "K": 60}
    rx.write_summary_header(summary_path, exp_id=42, slug="try-k60", effective=effective)
    text = summary_path.read_text()
    # Old content preserved
    assert "(old)" in text
    # New session marker appended
    assert "## Fit session 2" in text
```

- [ ] **Step 2: Run tests to verify failure**

```bash
poetry run pytest scripts/tests/test_run_experiment.py -v
```
Expected: 3 new tests FAIL.

- [ ] **Step 3: Add builders to run_experiment.py**

Append to `scripts/run_experiment.py`:

```python
import datetime as _dt
import subprocess

# Spark configuration constants. Match the existing per-cohort Makefile targets.
# If these need to vary per-experiment, lift them into _base.yaml in a future
# increment.
SPARK_SUBMIT_FLAGS = [
    "--master", "yarn",
    "--deploy-mode", "client",
    "--driver-memory", "4g",
    "--conf", "spark.executor.cores=4",
    "--conf", "spark.executor.memory=6g",
    "--conf", "spark.executor.memoryOverhead=2g",
    "--conf", "spark.python.worker.memory=2g",
]


def build_spark_submit_cmd(
    script: str, script_args: list[str], repo_root: Path,
) -> list[str]:
    """Build the full spark-submit command line."""
    spark_vi_zip = repo_root / "spark-vi" / "dist" / "spark_vi.zip"
    charmpheno_zip = repo_root / "charmpheno" / "dist" / "charmpheno.zip"
    py_files_val = f"{spark_vi_zip},{charmpheno_zip}"
    return (
        ["spark-submit"]
        + SPARK_SUBMIT_FLAGS
        + ["--py-files", py_files_val, script]
        + script_args
    )


def _count_existing_fit_sessions(summary_text: str) -> int:
    """Count occurrences of '## Fit session N' headers in existing summary."""
    return len(re.findall(r"^## Fit session \d+", summary_text, re.MULTILINE))


def write_summary_header(
    summary_path: Path, *, exp_id: int, slug: str, effective: dict,
) -> None:
    """Write or append the per-session header to summary.md.

    If the file doesn't exist: writes the experiment-level header (title + effective
    config block) plus '## Fit session 1' header.
    If the file exists: appends '## Fit session N' header where N is (count+1).
    """
    started = _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    if not summary_path.exists():
        lines = [
            f"# Experiment {exp_id:04d} — {slug}",
            "",
            "## Effective config",
        ]
        for k in sorted(effective.keys()):
            lines.append(f"{k}: {effective[k]}")
        lines += ["", "## Fit session 1", f"Started: {started}", ""]
        summary_path.write_text("\n".join(lines) + "\n")
    else:
        existing = summary_path.read_text()
        n = _count_existing_fit_sessions(existing) + 1
        with summary_path.open("a") as f:
            f.write(f"\n## Fit session {n}\nStarted: {started}\n\n")
```

- [ ] **Step 4: Run tests to verify pass**

```bash
poetry run pytest scripts/tests/test_run_experiment.py -v
```
Expected: all 27 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_experiment.py scripts/tests/test_run_experiment.py
git commit -m "feat(scripts): build_spark_submit_cmd + write_summary_header"
```

---

## Task 8: Subprocess dispatch with stdout tee + sanitization

**Files:**
- Modify: `scripts/run_experiment.py`
- Modify: `scripts/tests/test_run_experiment.py`

- [ ] **Step 1: Write the failing test**

Append to `scripts/tests/test_run_experiment.py`:

```python
def test_run_subprocess_tee_sanitize_writes_filtered_output(tmp_path, capsys):
    summary_path = tmp_path / "summary.md"
    summary_path.write_text("# header\n")  # pre-existing
    # Use a tiny shell command that prints 3 lines, two of which should drop
    cmd = [
        "sh", "-c",
        "echo '[iter 1] ELBO=-1.23'; echo '|person_hash|topicDistribution|'; echo '[driver] done'",
    ]
    exit_code = rx.run_subprocess_tee_sanitize(cmd, summary_path, rx.PATIENT_PATTERNS)
    assert exit_code == 0
    summary_text = summary_path.read_text()
    # Sanitized lines NOT in summary
    assert "person_hash" not in summary_text
    # Clean lines in summary
    assert "[iter 1] ELBO=-1.23" in summary_text
    assert "[driver] done" in summary_text
    # Pre-existing header preserved
    assert "# header" in summary_text
    # All three lines (including dropped) appear on stdout for live debugging
    captured = capsys.readouterr()
    assert "person_hash" in captured.out
    assert "[iter 1]" in captured.out


def test_run_subprocess_tee_sanitize_propagates_exit_code(tmp_path):
    summary_path = tmp_path / "summary.md"
    cmd = ["sh", "-c", "echo 'starting'; exit 7"]
    exit_code = rx.run_subprocess_tee_sanitize(cmd, summary_path, rx.PATIENT_PATTERNS)
    assert exit_code == 7
    # Partial output captured
    assert "starting" in summary_path.read_text()
```

- [ ] **Step 2: Run tests to verify failure**

```bash
poetry run pytest scripts/tests/test_run_experiment.py -v
```
Expected: 2 new tests FAIL.

- [ ] **Step 3: Add `run_subprocess_tee_sanitize` to run_experiment.py**

Append to `scripts/run_experiment.py`:

```python
def run_subprocess_tee_sanitize(
    cmd: list[str], summary_path: Path, patterns: list[re.Pattern],
) -> int:
    """Run `cmd` as a subprocess; stream stdout line-by-line.

    Each line is printed to this process's stdout (live debugging) AND, if
    `sanitize_line` returns non-None, appended to `summary_path`.

    Returns the subprocess exit code.
    """
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True,
    )
    assert proc.stdout is not None
    with summary_path.open("a") as fout:
        for line in proc.stdout:
            # Live debugging: always print to terminal
            sys.stdout.write(line)
            sys.stdout.flush()
            # Committed record: sanitized only
            clean = sanitize_line(line, patterns)
            if clean is not None:
                fout.write(clean)
                fout.flush()
    return proc.wait()
```

- [ ] **Step 4: Run tests to verify pass**

```bash
poetry run pytest scripts/tests/test_run_experiment.py -v
```
Expected: all 29 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_experiment.py scripts/tests/test_run_experiment.py
git commit -m "feat(scripts): subprocess tee with sanitized append to summary.md"
```

---

## Task 9: Eval section appender

**Files:**
- Modify: `scripts/run_experiment.py`
- Modify: `scripts/tests/test_run_experiment.py`

- [ ] **Step 1: Write the failing test**

Append to `scripts/tests/test_run_experiment.py`:

```python
def test_append_eval_section_marker_and_body(tmp_path):
    summary_path = tmp_path / "summary.md"
    summary_path.write_text("# header\n\n## Fit session 1\n... fit output ...\n")
    eval_stdout = "mean NPMI: 0.224  median: 0.198\n  topic 0: 0.31\n"
    rx.append_eval_section(summary_path, eval_stdout)
    text = summary_path.read_text()
    assert "## Eval (NPMI)" in text
    assert "mean NPMI: 0.224" in text
    # Original content preserved
    assert "## Fit session 1" in text


def test_append_eval_section_sanitizes_body(tmp_path):
    summary_path = tmp_path / "summary.md"
    summary_path.write_text("# header\n")
    eval_stdout = (
        "mean NPMI: 0.2\n"
        "|person_hash|topicDistribution|\n"   # should be dropped
        "  topic 0: 0.31\n"
    )
    rx.append_eval_section(summary_path, eval_stdout)
    text = summary_path.read_text()
    assert "mean NPMI: 0.2" in text
    assert "topic 0: 0.31" in text
    assert "person_hash" not in text
```

- [ ] **Step 2: Run tests to verify failure**

```bash
poetry run pytest scripts/tests/test_run_experiment.py -v
```
Expected: 2 new tests FAIL.

- [ ] **Step 3: Add to run_experiment.py**

Append to `scripts/run_experiment.py`:

```python
def append_eval_section(summary_path: Path, eval_stdout: str) -> None:
    """Append a sanitized '## Eval (NPMI)' section to summary_path."""
    with summary_path.open("a") as f:
        f.write("\n## Eval (NPMI)\n")
        for line in eval_stdout.splitlines(keepends=True):
            clean = sanitize_line(line, PATIENT_PATTERNS)
            if clean is not None:
                f.write(clean)
        if not eval_stdout.endswith("\n"):
            f.write("\n")
```

- [ ] **Step 4: Run tests to verify pass**

```bash
poetry run pytest scripts/tests/test_run_experiment.py -v
```
Expected: all 31 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_experiment.py scripts/tests/test_run_experiment.py
git commit -m "feat(scripts): append_eval_section appends sanitized eval block to summary.md"
```

---

## Task 10: `main()` orchestration

**Files:**
- Modify: `scripts/run_experiment.py`

This task wires everything together. No new unit tests — the orchestration glue is exercised by the end-to-end smoke in Task 13.

- [ ] **Step 1: Add argparse + main() to run_experiment.py**

Append to `scripts/run_experiment.py`:

```python
import argparse

# RUNS_DIR mirrors the existing Makefile constant. Override via --runs-dir CLI
# for local testing. On the cluster, the GCS-mounted path is the canonical home.
DEFAULT_RUNS_DIR = "/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--next", action="store_true",
                          help="Pick the lowest-id experiment with status: pending.")
    selector.add_argument("--id", type=int, default=None,
                          help="Run the experiment with the given numeric id.")
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR,
                        help="Base directory for run output. Default: %(default)s")
    parser.add_argument("--experiments-dir", type=Path, default=EXPERIMENTS_DIR,
                        help="Where docs/experiments/NNNN-*.md files live.")
    parser.add_argument("--defaults-dir", type=Path, default=DEFAULTS_DIR,
                        help="Where experiments/defaults/*.yaml files live.")
    args = parser.parse_args(argv)

    # 1. Select experiment file
    if args.next:
        exp_path = find_next_pending(args.experiments_dir)
        if exp_path is None:
            print("[run-exp] no pending experiments found", flush=True)
            return 1
    else:
        exp_path = find_by_id(args.experiments_dir, args.id)
    print(f"[run-exp] experiment: {exp_path}", flush=True)

    # 2. Read frontmatter + merge defaults
    fm = read_frontmatter(exp_path)
    required = ["id", "slug", "cohort", "model_class"]
    for k in required:
        if k not in fm:
            print(f"[run-exp] ERROR: frontmatter missing required field {k!r}", flush=True)
            return 2
    if fm["model_class"] != "lda":
        print(f"[run-exp] ERROR: only model_class: lda supported in Increment 1 "
              f"(got {fm['model_class']!r})", flush=True)
        return 2
    defaults = load_defaults(fm["cohort"], args.defaults_dir)
    effective = merge_config(defaults, fm)

    # 3. Resolve save_dir, detect resume
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    save_dir = runs_dir / f"{fm['id']:04d}-{fm['slug']}"
    resume_from: Path | None = save_dir if save_dir.exists() else None
    print(f"[run-exp] save_dir: {save_dir}  resume: {resume_from is not None}",
          flush=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 4. Write summary header (new file or append session marker)
    summary_path = save_dir / "summary.md"
    write_summary_header(
        summary_path, exp_id=fm["id"], slug=fm["slug"], effective=effective,
    )

    # 5. Dispatch fit
    lda_script = REPO_ROOT / "analysis" / "cloud" / "lda_bigquery_cloud.py"
    lda_args = build_lda_args(effective, save_dir, resume_from)
    fit_cmd = build_spark_submit_cmd(str(lda_script), lda_args, REPO_ROOT)
    print(f"[run-exp] spark-submit: {' '.join(fit_cmd)}", flush=True)
    fit_rc = run_subprocess_tee_sanitize(fit_cmd, summary_path, PATIENT_PATTERNS)
    if fit_rc != 0:
        print(f"[run-exp] fit exited non-zero ({fit_rc}); skipping eval", flush=True)
        with summary_path.open("a") as f:
            f.write(f"\n### Session ended with exit code {fit_rc}\n")
        return fit_rc
    with summary_path.open("a") as f:
        f.write("\n### Session complete (exit 0)\n")

    # 6. Dispatch eval (capture stdout into a string for sanitized append)
    eval_script = REPO_ROOT / "analysis" / "cloud" / "eval_coherence_cloud.py"
    eval_args = build_eval_args(save_dir, effective)
    eval_cmd = build_spark_submit_cmd(str(eval_script), eval_args, REPO_ROOT)
    print(f"[run-exp] eval spark-submit: {' '.join(eval_cmd)}", flush=True)
    eval_proc = subprocess.run(
        eval_cmd, capture_output=True, text=True, check=False,
    )
    sys.stdout.write(eval_proc.stdout)
    sys.stdout.flush()
    if eval_proc.returncode != 0:
        print(f"[run-exp] eval exited non-zero ({eval_proc.returncode}); "
              "appending captured output anyway", flush=True)
    append_eval_section(summary_path, eval_proc.stdout)
    print(f"[run-exp] DONE. summary at: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify the unit tests still pass and the file imports cleanly**

```bash
poetry run pytest scripts/tests/test_run_experiment.py -v
poetry run python -c "import scripts.run_experiment as rx; print(rx.main.__doc__ or 'main defined')"
```
Expected: all 31 tests PASS; `python -c` runs without ImportError.

- [ ] **Step 3: Verify `--help` works**

```bash
poetry run python scripts/run_experiment.py --help
```
Expected: argparse help text printed; mentions `--next`, `--id`, `--runs-dir`, `--experiments-dir`, `--defaults-dir`.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_experiment.py
git commit -m "feat(scripts): run_experiment.py main() orchestrates fit+eval pipeline"
```

---

## Task 11: Driver-side strip of patient-hash transform sample

**Files:**
- Modify: `analysis/cloud/lda_bigquery_cloud.py:420-432`

The block at lines 420-432 prints hashed person rows after fit and emits a stale "SMOKE TEST PASSED" message that misleads in non-smoke runs. Strip both. Top-N tokens print above stays (vocab-level, safe).

- [ ] **Step 1: Read the surrounding context**

```bash
sed -n '405,435p' analysis/cloud/lda_bigquery_cloud.py
```

Expected: shows the `topicsMatrix` print loop ending around line 418, the `with _phase("transform sample"):` block at 420-429, the `bow_df.unpersist()` line 431, and the SMOKE TEST PASSED print at 432.

- [ ] **Step 2: Apply the edit**

In `analysis/cloud/lda_bigquery_cloud.py`, find this block (lines 420-432):

```python
    with _phase("transform sample"):
        # Transform against the un-split bow_df so the sample shows topic
        # distributions for some patients regardless of holdout assignment.
        (model.transform(bow_df)
              .withColumn("person_hash",
                          F.substring(
                              F.sha2(F.col("person_id").cast("string"), 256),
                              1, 12))
              .select("person_hash", "topicDistribution")
              .show(3, truncate=False))

    bow_df.unpersist()
    print("[driver] LDA BQ SMOKE TEST PASSED", flush=True)
```

Replace with:

```python
    bow_df.unpersist()
    print("[driver] fit complete", flush=True)
```

The `transform sample` block goes away entirely (not committable per the design's sanitization layer). The stale "SMOKE TEST PASSED" line gets replaced with the accurate "fit complete" marker.

- [ ] **Step 3: Verify the file still parses**

```bash
poetry run python -c "import ast; ast.parse(open('analysis/cloud/lda_bigquery_cloud.py').read()); print('OK')"
```
Expected: prints "OK".

- [ ] **Step 4: Commit**

```bash
git add analysis/cloud/lda_bigquery_cloud.py
git commit -m "fix(analysis): remove transform-sample person_hash print + stale smoke-test marker"
```

---

## Task 12: Makefile targets

**Files:**
- Modify: `analysis/cloud/Makefile`

- [ ] **Step 1: Locate the right insertion point**

```bash
grep -n "^lda-bq-fit-eval:" analysis/cloud/Makefile
```

Expected: returns a line number around 265. Insert the new targets above the `lda-bq-fit-eval:` block (between the `RUN_ID` declaration around line 263 and the existing target).

- [ ] **Step 2: Add the new targets**

Open `analysis/cloud/Makefile`. After the existing `RUN_ID ?=` line (around line 263) and before `lda-bq-fit-eval:`, insert:

```make
# ─── Experiment-tracking targets (Increment 1) ──────────────────────────────
# See docs/superpowers/specs/2026-05-28-experiment-tracking-design.md.
#
# Author experiments locally in docs/experiments/NNNN-slug.md with YAML
# frontmatter; run them on the cluster via these targets.

# Pick lowest-NNNN with status: pending; fit + eval; write summary.md
next-exp: zip $(WORKSPACE_ENV)
	. ./$(WORKSPACE_ENV) && \
	python $(REPO_ROOT)/scripts/run_experiment.py --next --runs-dir $(RUNS_DIR)

# Same pipeline pinned to specific id (auto-resumes if RUN_ID dir exists)
exp: zip $(WORKSPACE_ENV)
	@if [ -z "$(ID)" ]; then echo "ERROR: provide ID=N"; exit 1; fi
	. ./$(WORKSPACE_ENV) && \
	python $(REPO_ROOT)/scripts/run_experiment.py --id $(ID) --runs-dir $(RUNS_DIR)

# Print summary.md for an experiment (copy/paste back to chat)
summary:
	@if [ -z "$(ID)" ]; then echo "ERROR: provide ID=N"; exit 1; fi
	@cat $(RUNS_DIR)/$(shell printf '%04d' $(ID))-*/summary.md
```

- [ ] **Step 3: Add a `help` description for the new targets**

Locate the existing `help:` target (around line 62). Find the existing per-target help lines and add new ones for the experiment-tracking targets in the same style. The existing block looks like a series of `@echo` lines. Add (just below the `lda-bq-fit-eval` help line, wherever it sits):

```make
	@echo "  next-exp                Pick lowest-NNNN pending experiment; fit + eval."
	@echo "  exp ID=N                Run/resume experiment N; fit + eval."
	@echo "  summary ID=N            Print summary.md for experiment N."
```

- [ ] **Step 4: Verify Make can parse the file**

```bash
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno
make -C analysis/cloud -n next-exp 2>&1 | head -10
```

Expected: prints the dry-run commands the target would execute (or a "no rule to make target zip" / WORKSPACE_ENV-not-found message, which is acceptable here — what we want is *no syntax error from Make itself*).

- [ ] **Step 5: Commit**

```bash
git add analysis/cloud/Makefile
git commit -m "feat(cloud/make): next-exp, exp, summary targets for experiment-tracking"
```

---

## Task 13: Skeleton experiment record + end-to-end smoke

**Files:**
- Create: `docs/experiments/0001-pilot.md`

This task validates the end-to-end pipeline. It does NOT require actually running spark-submit (that needs the cluster); instead it verifies that the wrapper:
1. Finds the experiment by `--next`,
2. Merges defaults correctly,
3. Resolves the right save_dir,
4. Builds the spark-submit command without error,
5. Writes a summary.md header.

A real cluster run is a manual follow-on validation (noted in the Verification section below).

- [ ] **Step 1: Create the skeleton experiment record**

Write to `docs/experiments/0001-pilot.md`:

```markdown
---
id: 1
slug: pilot
status: pending
model_class: lda
cohort: dementia
created: 2026-05-28

# Tiny K to keep the smoke run cheap. Override max_iter on the cluster
# (e.g. via a one-off frontmatter bump) if you want a fuller run.
K: 5
max_iter: 2
vocab_size: 500
print_topics_every: 1
---

# Experiment 0001 — pilot

## Intent
First end-to-end validation of the experiment-tracking pipeline.
Tiny K (5) and max_iter (2) so the smoke is cheap. After this proves out,
mark `status: done` and the next real experiment starts at 0002.

## Fit history

## Results

## Interpretation

## Links
- Spec: docs/superpowers/specs/2026-05-28-experiment-tracking-design.md
- Plan: docs/superpowers/plans/2026-05-28-experiment-tracking-increment-1.md
```

- [ ] **Step 2: Smoke-test the wrapper's planning phase locally**

Replace `find_by_id` with `--next`, point at a temp runs-dir, and verify the wrapper builds the spark-submit command (but expect spark-submit itself to fail since this is local — we just want the planning to succeed up to the subprocess attempt):

```bash
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno
RUNS=$(mktemp -d)
poetry run python scripts/run_experiment.py --next --runs-dir "$RUNS" 2>&1 | head -30 || true
ls -la "$RUNS"/0001-pilot/
cat "$RUNS"/0001-pilot/summary.md
```

Expected:
- The `head -30` output shows `[run-exp] experiment: docs/experiments/0001-pilot.md`, then `[run-exp] save_dir: .../0001-pilot resume: False`, then `[run-exp] spark-submit: spark-submit ...` (with the full command), then likely a `FileNotFoundError` / `command not found` for `spark-submit` (since spark isn't installed locally).
- `ls -la "$RUNS"/0001-pilot/` shows `summary.md`.
- `cat "$RUNS"/0001-pilot/summary.md` shows the `# Experiment 0001 — pilot` header, the `## Effective config` block with merged defaults (`K: 5`, `cohort: dementia`, `max_iter: 2`, `model_class: lda`, etc.), and the `## Fit session 1` marker.

- [ ] **Step 3: Document expected cluster-side smoke**

The cluster validation (when you next have time on the cluster) is:

```bash
git pull
make -C analysis/cloud next-exp
make -C analysis/cloud summary ID=1
```

What to verify:
- spark-submit launches the LDA fit with the merged effective config visible in the driver banner.
- summary.md grows as the fit runs (because of `flush()` on every line).
- After eval completes, summary.md has both `## Fit session 1` and `## Eval (NPMI)` sections.
- No `person_hash` lines anywhere in summary.md.
- `make summary ID=1` cats a single bounded file.

This step is a documentation note, not an automated check. Add a short note in the experiment record file's `## Fit history` once it's been run on the cluster.

- [ ] **Step 4: Commit**

```bash
git add docs/experiments/0001-pilot.md
git commit -m "feat(experiments): 0001-pilot skeleton for end-to-end smoke validation"
```

---

## Verification

After all 13 tasks:

```bash
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno
poetry run pytest scripts/tests/test_run_experiment.py -v
```

Expected: all 31 tests PASS.

Local planning smoke (from Task 13 Step 2):

```bash
RUNS=$(mktemp -d) && poetry run python scripts/run_experiment.py --next --runs-dir "$RUNS" 2>&1 | head -30 ; cat "$RUNS"/0001-pilot/summary.md
```

Expected: planning phase prints the resolved experiment + save_dir + spark-submit command line; summary.md contains the merged effective config block.

Cluster-side smoke (manual, when next on cluster):

```bash
git pull
make -C analysis/cloud next-exp
make -C analysis/cloud summary ID=1
```

Expected outcomes documented in Task 13 Step 3.

---

## Out of Scope / Follow-Ups for Increment 2

These are captured here so the next plan has a clear starting point:

1. **Structured per-iter parsing in summary.md.** Parse `[iter N]` lines from stdout and structure them into a `### Per-iter trend` markdown block inside `## Fit session N`. Currently stdout is appended verbatim.
2. **SIGTERM trap with "killed at iter N" marker.** Currently a SIGTERM kills the wrapper outright; resume picks up but no "killed" record appears.
3. **Resume-aware session count.** `write_summary_header` already counts existing `## Fit session N` headers, but if the existing summary.md is missing because the dir was nuked-but-checkpoint-restored externally, the count restarts. Edge case; defer until needed.
4. **`eval-exp ID=N` standalone target.** Re-run eval against an existing checkpoint; append-only to summary.md. Currently eval only runs as part of the full pipeline.
5. **HDP support (`model_class: hdp`).** Wrapper rejects non-LDA in Increment 1. Adding HDP needs `build_hdp_args` and dispatch branching.

## Out of Scope / Follow-Ups for Increment 3

1. **`warm_start_from: NNNN`** field + checkpoint copy logic.
2. **`build-dashboard-exp ID=N OUT=X`** target.
3. **Cross-link convention enforcement** in record file template (manually maintained for now).
4. **First REVIEW_LOG entry citing experiments by id** — when there are real experiments to cite.
5. **Spark config promotion to YAML** (currently hardcoded in `SPARK_SUBMIT_FLAGS`). Defer until an experiment actually needs different executor settings.

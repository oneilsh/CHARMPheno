# Experiment Tracking — Increment 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `make build-dashboard-exp` with auto-discovery of the
most-recent-fit-needing-build, plus distinct per-experiment zip filenames,
plus a `## Dashboard build — <ts>` section appended to `summary.md`.

**Architecture:** Mirrors the Inc 2.5 `eval-exp` shape. Add `--zip-name` flag
to `build_dashboard_cloud.py` (single arg, falls back to existing default).
In `scripts/run_experiment.py`, add four new helpers
(`find_most_recent_fit_needing_build`, `build_dashboard_args`,
`write_build_section_header`, plus a `--build-only` branch in `main()`).
Defaults `vocab_top_n` and `top_n_codes_for_npmi` get added to `_base.yaml`
to make the build args YAML-overridable. Make target plumbs the new flag.

**Tech Stack:** Python 3, `argparse`, `subprocess`, `re`, PyYAML, pytest,
existing `spark-submit` constants.

**Scope boundary:** LDA-only (HDP wrapper rejection still in place);
`warm_start_from`, structured per-iter parsing, and the wider Inc 3 menu
items remain parked. See spec
[2026-05-29-experiment-tracking-increment-3-design.md](../specs/2026-05-29-experiment-tracking-increment-3-design.md).

**Pilot/exp 0002 dependency:** Plan ships before the user authors
`docs/experiments/0002-*.md` (the next real experiment will exercise this
target end-to-end against Pilot 0001's existing checkpoint first as a smoke).

---

## File Structure

**Modified:**
- `scripts/run_experiment.py` — four new helpers + `--build-only` branch +
  mutual-exclusion validation.
- `scripts/tests/test_run_experiment.py` — ~10 new unit tests.
- `analysis/cloud/build_dashboard_cloud.py` — `--zip-name` argparse + zip
  path resolution.
- `experiments/defaults/_base.yaml` — two new keys.
- `analysis/cloud/Makefile` — new `build-dashboard-exp` target.

**Not touched:**
- Experiment-record `.md` schema. Frontmatter required fields unchanged.
- Per-cohort YAML defaults (`general.yaml`, `cancer.yaml`, `dementia.yaml`).
- HDP driver / wrapper HDP-branch (still parked).
- Existing `build-dashboard-bundle CHECKPOINT=...` target.

---

## Module additions inside `scripts/run_experiment.py`

| Symbol | Type | Responsibility |
|---|---|---|
| `find_most_recent_fit_needing_build(runs_dir)` | pure | Returns int exp id (or None) — picks freshest fit whose `dashboard_bundle/corpus_stats.json` is missing or older than `manifest.json`. |
| `build_dashboard_args(effective, checkpoint_dir, zip_name)` | pure | Returns `list[str]` of CLI args for `build_dashboard_cloud.py`. |
| `write_build_section_header(summary_path)` | I/O | Appends `## Dashboard build — <UTC ts>` header to `summary.md`. |
| `--build-only` branch in `main()` | I/O | Skips fit + eval; runs only the build dispatch. Auto-discovers when `--id` omitted. |

---

## Task 1: `--zip-name` flag in `build_dashboard_cloud.py`

**Files:**
- Modify: `analysis/cloud/build_dashboard_cloud.py:47-58` (argparse block)
- Modify: `analysis/cloud/build_dashboard_cloud.py:268` (zip path resolution)

The script is hard to unit-test directly (depends on Spark + BigQuery), so the
test surface here is just argparse parsing — confirming the flag is accepted
and propagates to `args.zip_name`.

- [ ] **Step 1: Read the current argparse block to confirm context**

```bash
sed -n '44,58p' /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno/analysis/cloud/build_dashboard_cloud.py
```

Confirm the existing flags are `--checkpoint`, `--out-dir`, `--model-class`,
`--hdp-top-k`, `--vocab-top-n`, `--top-n-codes-for-npmi`.

- [ ] **Step 2: Add the new argparse entry**

After the `--top-n-codes-for-npmi` line in
[build_dashboard_cloud.py:57](analysis/cloud/build_dashboard_cloud.py#L57),
insert:

```python
    parser.add_argument("--zip-name", default=None,
                        help="basename for the zip artifact (written as sibling "
                             "of --out-dir). Default: <out_dir_name>.zip "
                             "(e.g. dashboard_bundle.zip).")
```

- [ ] **Step 3: Change the zip-path resolution**

At [build_dashboard_cloud.py:268](analysis/cloud/build_dashboard_cloud.py#L268), change:

```python
            zip_path = out_dir.with_suffix(".zip")
```

to:

```python
            zip_path = (
                out_dir.parent / args.zip_name if args.zip_name
                else out_dir.with_suffix(".zip")
            )
```

- [ ] **Step 4: Smoke-test argparse parses the new flag**

There's no existing test file for `build_dashboard_cloud.py`. Rather than
introduce one for a single argparse change, smoke from the shell:

```bash
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno
python -c "
import sys
sys.argv = ['build_dashboard_cloud.py', '--checkpoint', '/tmp/x',
            '--zip-name', '0001-pilot-dashboard.zip', '--help']
try:
    import analysis.cloud.build_dashboard_cloud as m
    m.main()
except SystemExit:
    pass
" 2>&1 | grep -A 2 zip-name
```

Expected: `--zip-name` shows up in the help output with the description.

- [ ] **Step 5: Commit**

```bash
git add analysis/cloud/build_dashboard_cloud.py
git commit -m "feat(cloud): --zip-name flag on build_dashboard_cloud.py

Allows callers to override the bundle zip basename; default behavior
(dashboard_bundle.zip) preserved when unset. The experiment-tracking wrapper
will pass NNNN-slug-dashboard.zip so downloaded bundles don't collide in
~/Downloads."
```

---

## Task 2: `find_most_recent_fit_needing_build` helper

**Files:**
- Modify: `scripts/run_experiment.py` — add helper after
  `find_most_recent_fit` (~line 123).
- Modify: `scripts/tests/test_run_experiment.py` — six new tests.

- [ ] **Step 1: Write the failing tests**

In `scripts/tests/test_run_experiment.py`, add (mirroring the
`find_most_recent_fit` test class shape):

```python
class TestFindMostRecentFitNeedingBuild:
    """find_most_recent_fit_needing_build returns the freshest fit whose
    dashboard bundle is missing or stale."""

    def test_returns_none_when_runs_dir_missing(self, tmp_path):
        from scripts.run_experiment import find_most_recent_fit_needing_build
        assert find_most_recent_fit_needing_build(tmp_path / "nope") is None

    def test_returns_none_when_no_fits_exist(self, tmp_path):
        from scripts.run_experiment import find_most_recent_fit_needing_build
        runs = tmp_path / "runs"
        runs.mkdir()
        assert find_most_recent_fit_needing_build(runs) is None

    def test_picks_never_built_fit(self, tmp_path):
        from scripts.run_experiment import find_most_recent_fit_needing_build
        runs = tmp_path / "runs"
        (runs / "0001-pilot").mkdir(parents=True)
        (runs / "0001-pilot" / "manifest.json").write_text("{}")
        assert find_most_recent_fit_needing_build(runs) == 1

    def test_skips_current_bundles(self, tmp_path):
        from scripts.run_experiment import find_most_recent_fit_needing_build
        import os, time
        runs = tmp_path / "runs"
        d = runs / "0001-pilot"
        d.mkdir(parents=True)
        manifest = d / "manifest.json"
        manifest.write_text("{}")
        bundle = d / "dashboard_bundle"
        bundle.mkdir()
        marker = bundle / "corpus_stats.json"
        marker.write_text("{}")
        # Force marker mtime strictly after manifest mtime
        later = manifest.stat().st_mtime + 100
        os.utime(marker, (later, later))
        assert find_most_recent_fit_needing_build(runs) is None

    def test_picks_stale_bundle(self, tmp_path):
        from scripts.run_experiment import find_most_recent_fit_needing_build
        import os
        runs = tmp_path / "runs"
        d = runs / "0001-pilot"
        d.mkdir(parents=True)
        manifest = d / "manifest.json"
        manifest.write_text("{}")
        bundle = d / "dashboard_bundle"
        bundle.mkdir()
        marker = bundle / "corpus_stats.json"
        marker.write_text("{}")
        # Force manifest mtime strictly after marker mtime
        later = marker.stat().st_mtime + 100
        os.utime(manifest, (later, later))
        assert find_most_recent_fit_needing_build(runs) == 1

    def test_multiple_candidates_picks_newest_fit(self, tmp_path):
        from scripts.run_experiment import find_most_recent_fit_needing_build
        import os
        runs = tmp_path / "runs"
        # 0001-pilot: stale bundle, older fit
        d1 = runs / "0001-pilot"
        d1.mkdir(parents=True)
        m1 = d1 / "manifest.json"
        m1.write_text("{}")
        # 0002-bigger: never-built, newer fit
        d2 = runs / "0002-bigger"
        d2.mkdir(parents=True)
        m2 = d2 / "manifest.json"
        m2.write_text("{}")
        os.utime(m2, (m1.stat().st_mtime + 100, m1.stat().st_mtime + 100))
        assert find_most_recent_fit_needing_build(runs) == 2

    def test_skips_dirs_without_manifest(self, tmp_path):
        from scripts.run_experiment import find_most_recent_fit_needing_build
        runs = tmp_path / "runs"
        (runs / "0001-empty").mkdir(parents=True)
        # No manifest.json — not yet fit, nothing to build
        assert find_most_recent_fit_needing_build(runs) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno
poetry run pytest scripts/tests/test_run_experiment.py::TestFindMostRecentFitNeedingBuild -v
```

Expected: All seven tests FAIL with `ImportError` or `AttributeError` — the
function doesn't exist yet.

- [ ] **Step 3: Implement the function**

In [scripts/run_experiment.py](scripts/run_experiment.py), after the
`find_most_recent_fit` function (~line 123), add:

```python
def find_most_recent_fit_needing_build(runs_dir: Path) -> int | None:
    """Return exp id of the most-recently-fit experiment whose dashboard
    bundle is missing or stale.

    Compares each `<NNNN-slug>/manifest.json` mtime against the same dir's
    `dashboard_bundle/corpus_stats.json` mtime (the last JSON written by
    build_dashboard_cloud before the zip step). If the marker is absent or
    older than the manifest, the fit is a build candidate. Among all
    candidates, returns the one with the latest manifest mtime — the
    freshest fit that still wants a build. Returns None if no candidates
    exist (all bundles current, or no fits at all).

    Caller invariant: dir-naming convention `NNNN-slug` is enforced by
    `main()` at save_dir creation; this function trusts it and skips any
    sibling whose name doesn't match.
    """
    if not runs_dir.is_dir():
        return None
    best_mtime = -1.0
    best_id: int | None = None
    for manifest in runs_dir.glob("*/manifest.json"):
        m = _RUN_DIR_PATTERN.match(manifest.parent.name)
        if m is None:
            continue
        marker = manifest.parent / "dashboard_bundle" / "corpus_stats.json"
        manifest_mtime = manifest.stat().st_mtime
        if marker.exists() and marker.stat().st_mtime >= manifest_mtime:
            continue
        if manifest_mtime > best_mtime:
            best_mtime = manifest_mtime
            best_id = int(m.group(1))
    return best_id
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
poetry run pytest scripts/tests/test_run_experiment.py::TestFindMostRecentFitNeedingBuild -v
```

Expected: All seven tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_experiment.py scripts/tests/test_run_experiment.py
git commit -m "feat(scripts): find_most_recent_fit_needing_build helper

Returns the freshest fit whose dashboard bundle is missing or older than
the fit checkpoint. Used by --build-only auto-discovery in the next task."
```

---

## Task 3: `build_dashboard_args` builder

**Files:**
- Modify: `scripts/run_experiment.py` — add helper after `build_eval_args` (~line 228).
- Modify: `scripts/tests/test_run_experiment.py` — three new tests.
- Modify: `experiments/defaults/_base.yaml` — add two keys.

- [ ] **Step 1: Add the two new defaults**

In [experiments/defaults/_base.yaml](experiments/defaults/_base.yaml), append
after `optimize_topic_concentration: false`:

```yaml
# Dashboard-build defaults (used by --build-only / make build-dashboard-exp).
# Match build_dashboard_cloud.py's argparse defaults; override per-experiment
# in frontmatter to dial bundle size or NPMI sample size.
vocab_top_n: 5000
top_n_codes_for_npmi: 20
```

- [ ] **Step 2: Write the failing tests**

In `scripts/tests/test_run_experiment.py`, add:

```python
class TestBuildDashboardArgs:
    """build_dashboard_args produces the argv for build_dashboard_cloud.py."""

    def test_minimal_lda(self, tmp_path):
        from scripts.run_experiment import build_dashboard_args
        effective = {
            "model_class": "lda",
            "vocab_top_n": 5000,
            "top_n_codes_for_npmi": 20,
        }
        args = build_dashboard_args(
            effective, tmp_path / "ck", "0001-pilot-dashboard.zip",
        )
        assert args[:2] == ["--checkpoint", str(tmp_path / "ck")]
        assert "--model-class" in args
        idx = args.index("--model-class")
        assert args[idx + 1] == "lda"
        assert "--zip-name" in args
        assert args[args.index("--zip-name") + 1] == "0001-pilot-dashboard.zip"
        assert "--vocab-top-n" in args
        assert args[args.index("--vocab-top-n") + 1] == "5000"
        assert "--top-n-codes-for-npmi" in args
        assert args[args.index("--top-n-codes-for-npmi") + 1] == "20"

    def test_vocab_top_n_override(self, tmp_path):
        from scripts.run_experiment import build_dashboard_args
        effective = {
            "model_class": "lda",
            "vocab_top_n": 1000,
            "top_n_codes_for_npmi": 10,
        }
        args = build_dashboard_args(effective, tmp_path / "ck", "z.zip")
        assert args[args.index("--vocab-top-n") + 1] == "1000"
        assert args[args.index("--top-n-codes-for-npmi") + 1] == "10"

    def test_missing_required_keys_raises(self, tmp_path):
        from scripts.run_experiment import build_dashboard_args
        # Missing vocab_top_n
        effective = {"model_class": "lda", "top_n_codes_for_npmi": 20}
        import pytest
        with pytest.raises(KeyError):
            build_dashboard_args(effective, tmp_path / "ck", "z.zip")
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
poetry run pytest scripts/tests/test_run_experiment.py::TestBuildDashboardArgs -v
```

Expected: All three FAIL with `ImportError`.

- [ ] **Step 4: Implement the function**

In [scripts/run_experiment.py](scripts/run_experiment.py), after the
`build_eval_args` function (~line 228), add:

```python
def build_dashboard_args(
    effective: dict, checkpoint_dir: Path, zip_name: str,
) -> list[str]:
    """Build the CLI arg list for build_dashboard_cloud.py.

    `zip_name` is the basename (not full path) written as a sibling of the
    bundle directory inside the checkpoint dir. Callers should pass
    `f"{exp_id:04d}-{slug}-dashboard.zip"`.
    """
    return [
        "--checkpoint", str(checkpoint_dir),
        "--model-class", str(effective.get("model_class", "lda")),
        "--zip-name", zip_name,
        "--vocab-top-n", str(effective["vocab_top_n"]),
        "--top-n-codes-for-npmi", str(effective["top_n_codes_for_npmi"]),
    ]
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
poetry run pytest scripts/tests/test_run_experiment.py::TestBuildDashboardArgs -v
```

Expected: All three PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_experiment.py scripts/tests/test_run_experiment.py experiments/defaults/_base.yaml
git commit -m "feat(scripts): build_dashboard_args helper + YAML defaults

vocab_top_n and top_n_codes_for_npmi join the YAML defaults chain so
build-time knobs are per-experiment overridable. build_dashboard_args
is the symmetric counterpart of build_lda_args / build_eval_args."
```

---

## Task 4: `write_build_section_header` helper

**Files:**
- Modify: `scripts/run_experiment.py` — add helper after `append_eval_section` (~line 381).
- Modify: `scripts/tests/test_run_experiment.py` — two new tests.

- [ ] **Step 1: Write the failing tests**

In `scripts/tests/test_run_experiment.py`, add:

```python
class TestWriteBuildSectionHeader:
    def test_appends_header_with_timestamp(self, tmp_path):
        from scripts.run_experiment import write_build_section_header
        import re
        summary = tmp_path / "summary.md"
        summary.write_text("# Existing content\n\n")
        write_build_section_header(summary)
        text = summary.read_text()
        # Existing content preserved
        assert "# Existing content" in text
        # Timestamped header appended
        assert re.search(
            r"^## Dashboard build — \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC$",
            text, re.MULTILINE,
        )

    def test_multiple_calls_append_distinct_sections(self, tmp_path):
        from scripts.run_experiment import write_build_section_header
        summary = tmp_path / "summary.md"
        summary.write_text("")
        write_build_section_header(summary)
        write_build_section_header(summary)
        text = summary.read_text()
        assert text.count("## Dashboard build —") == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
poetry run pytest scripts/tests/test_run_experiment.py::TestWriteBuildSectionHeader -v
```

Expected: Both FAIL with `ImportError`.

- [ ] **Step 3: Implement the function**

In [scripts/run_experiment.py](scripts/run_experiment.py), after the
`append_eval_section` function (~line 381), add:

```python
def write_build_section_header(summary_path: Path) -> None:
    """Append a timestamped '## Dashboard build — <UTC ts>' header.

    The build dispatch then streams its sanitized stdout via
    `run_subprocess_tee_sanitize` (same pattern as fit). After the dispatch
    completes, `main()` appends `### Build complete (exit N)` for symmetry
    with the eval pattern.
    """
    started = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with summary_path.open("a") as f:
        f.write(f"\n## Dashboard build — {started}\n")
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
poetry run pytest scripts/tests/test_run_experiment.py::TestWriteBuildSectionHeader -v
```

Expected: Both PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_experiment.py scripts/tests/test_run_experiment.py
git commit -m "feat(scripts): write_build_section_header for summary.md"
```

---

## Task 5: `--build-only` wiring in `main()`

**Files:**
- Modify: `scripts/run_experiment.py` — add argparse flag, mutual-exclusion
  check, auto-discovery branch, build dispatch block.
- Modify: `scripts/tests/test_run_experiment.py` — three new integration tests.

- [ ] **Step 1: Write the failing tests**

In `scripts/tests/test_run_experiment.py`, add (mirroring the
`--eval-only` test pattern; mocks subprocess calls):

```python
class TestBuildOnlyMain:
    """--build-only branch in main(): auto-discovery, force-rebuild,
    mutual-exclusion."""

    def test_build_only_and_eval_only_mutually_exclusive(self, tmp_path, capsys):
        from scripts.run_experiment import main
        rc = main(["--eval-only", "--build-only", "--runs-dir", str(tmp_path)])
        assert rc == 2
        captured = capsys.readouterr()
        assert "contradictory" in captured.out or "exclusive" in captured.out

    def test_build_only_and_no_eval_mutually_exclusive(self, tmp_path, capsys):
        from scripts.run_experiment import main
        rc = main(["--no-eval", "--build-only", "--runs-dir", str(tmp_path)])
        assert rc == 2

    def test_build_only_auto_discover_none_found_exits_zero(
        self, tmp_path, monkeypatch, capsys,
    ):
        """When no fits need building, --build-only without --id should
        print a friendly message and exit 0 (not error)."""
        from scripts.run_experiment import main
        runs = tmp_path / "runs"
        runs.mkdir()
        # No checkpoints under runs_dir -> nothing to build
        rc = main([
            "--build-only",
            "--runs-dir", str(runs),
            "--experiments-dir", str(tmp_path / "exp_dir_unused"),
            "--defaults-dir", str(tmp_path / "defaults_unused"),
        ])
        assert rc == 0
        captured = capsys.readouterr()
        assert "no fits need building" in captured.out.lower() \
               or "nothing to build" in captured.out.lower()
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
poetry run pytest scripts/tests/test_run_experiment.py::TestBuildOnlyMain -v
```

Expected: All three FAIL — `--build-only` flag doesn't exist yet (argparse
will error).

- [ ] **Step 3: Add the argparse flag and mutual-exclusion check**

In [scripts/run_experiment.py](scripts/run_experiment.py) `main()` around
line 400 (after `--no-eval`), add:

```python
    parser.add_argument("--build-only", action="store_true",
                        help="Skip fit and eval; only run the dashboard build "
                             "against the existing checkpoint at $RUNS_DIR/NNNN-slug/. "
                             "If --id is omitted, auto-selects the most recent "
                             "fit whose dashboard_bundle/corpus_stats.json is "
                             "missing or older than manifest.json.")
```

Then after the existing `if args.no_eval and args.eval_only:` check
(around line 410), add:

```python
    if args.build_only and (args.eval_only or args.no_eval):
        print("[run-exp] ERROR: --build-only is contradictory with --eval-only "
              "and --no-eval", flush=True)
        return 2
```

- [ ] **Step 4: Add the auto-discovery branch in experiment selection**

In the experiment-selection ladder (around line 414-444), add a new branch
mirroring the `elif args.eval_only:` shape. After the `elif args.eval_only:`
block, add:

```python
    elif args.build_only:
        # No --id with --build-only: auto-select the freshest fit whose
        # bundle is missing or stale.
        runs_dir_path = Path(args.runs_dir)
        auto_id = find_most_recent_fit_needing_build(runs_dir_path)
        if auto_id is None:
            print(f"[run-exp] no fits need building under {runs_dir_path} "
                  "(all bundles current or no checkpoints yet)", flush=True)
            return 0
        print(f"[run-exp] --build-only auto-selected id={auto_id} "
              "(fit newer than dashboard bundle)", flush=True)
        try:
            exp_path = find_by_id(args.experiments_dir, auto_id)
        except FileNotFoundError as e:
            print(f"[run-exp] ERROR: {e}", flush=True)
            return 2
```

And update the trailing `else:` error message to include `--build-only`:

```python
    else:
        print("[run-exp] ERROR: provide --next, --id N, --eval-only, "
              "or --build-only (the last two auto-discover when --id is omitted)",
              flush=True)
        return 2
```

- [ ] **Step 5: Add the build-dispatch block in main()**

After the eval-dispatch block (around line 537), but only reached when
`--build-only` is set (since otherwise the function returns after eval),
restructure so the build block is reachable. Cleanest shape: add the
build block after eval, and let `--eval-only` / `--no-eval` / regular runs
all fall through to it when `args.build_only` is set. But since `--build-only`
is mutually exclusive with both `--eval-only` and `--no-eval` (per Step 3),
the simpler shape is:

1. Skip fit dispatch when `--build-only` (same shape as `--eval-only`).
2. Skip eval dispatch when `--build-only` (don't run eval as part of build).
3. Run build dispatch unconditionally when `--build-only`.

Edit the fit-dispatch guard (around line 494):

```python
    # 5. Dispatch fit (unless --eval-only or --build-only)
    if args.eval_only or args.build_only:
        if not (save_dir / "manifest.json").exists():
            print(f"[run-exp] ERROR: --eval-only / --build-only requires "
                  f"checkpoint at {save_dir}/manifest.json (none found)", flush=True)
            return 2
        mode = "--eval-only" if args.eval_only else "--build-only"
        print(f"[run-exp] {mode}: skipping fit dispatch", flush=True)
    else:
        # ... existing fit dispatch unchanged
```

Edit the eval-dispatch guard (around line 516):

```python
    # Skip eval dispatch when --no-eval (fit-only mode) or --build-only.
    if args.no_eval or args.build_only:
        if args.no_eval:
            print("[run-exp] --no-eval: skipping eval dispatch", flush=True)
        if args.build_only:
            print("[run-exp] --build-only: skipping eval dispatch", flush=True)
        # Fall through to build dispatch if --build-only, else done.
        if not args.build_only:
            print(f"[run-exp] DONE. summary at: {summary_path}", flush=True)
            return 0
    else:
        # ... existing eval dispatch unchanged
```

Then after the eval block, add the build block:

```python
    # 7. Dispatch build (only when --build-only).
    if args.build_only:
        build_script = REPO_ROOT / "analysis" / "cloud" / "build_dashboard_cloud.py"
        zip_name = f"{fm['id']:04d}-{fm['slug']}-dashboard.zip"
        b_args = build_dashboard_args(effective, save_dir, zip_name)
        build_cmd = build_spark_submit_cmd(str(build_script), b_args, REPO_ROOT)
        # Display-only join; cmd is passed as list to Popen, not via shell.
        print(f"[run-exp] build spark-submit: {' '.join(build_cmd)}", flush=True)
        write_build_section_header(summary_path)
        build_rc = run_subprocess_tee_sanitize(build_cmd, summary_path, DROP_PATTERNS)
        with summary_path.open("a") as f:
            f.write(f"\n### Build complete (exit {build_rc})\n")
        if build_rc != 0:
            print(f"[run-exp] build exited non-zero ({build_rc})", flush=True)
            return build_rc

    print(f"[run-exp] DONE. summary at: {summary_path}", flush=True)
    return 0
```

- [ ] **Step 6: Run the new tests + existing test suite to verify everything passes**

```bash
poetry run pytest scripts/tests/test_run_experiment.py -v
```

Expected: All ~62 tests PASS (49 existing + 13 new across Tasks 2-5).

- [ ] **Step 7: Commit**

```bash
git add scripts/run_experiment.py scripts/tests/test_run_experiment.py
git commit -m "feat(scripts): --build-only flag + auto-discovery branch in main()

Mirrors the --eval-only pattern. With --id, force-rebuilds the named
experiment; without --id, auto-selects the freshest fit whose dashboard
bundle is missing or older than the fit checkpoint. Mutually exclusive
with --eval-only and --no-eval."
```

---

## Task 6: Make target + manual smoke against the pilot

**Files:**
- Modify: `analysis/cloud/Makefile` — add `build-dashboard-exp` target,
  add to `.PHONY` and `help`.

- [ ] **Step 1: Add the target**

In [analysis/cloud/Makefile](analysis/cloud/Makefile), after the existing
`eval-exp` target (~line 295), add:

```make
# Build dashboard bundle for an experiment. With no ID, picks the most
# recent fit whose bundle is missing or older than the fit checkpoint.
# With ID=N, force-rebuilds the named experiment.
build-dashboard-exp: zip $(WORKSPACE_ENV)
	. ./$(WORKSPACE_ENV) && \
	python $(REPO_ROOT)/scripts/run_experiment.py $(if $(ID),--id $(ID)) --build-only --runs-dir $(RUNS_DIR)
```

- [ ] **Step 2: Update `.PHONY` and `help`**

In [analysis/cloud/Makefile:1](analysis/cloud/Makefile#L1), add
`build-dashboard-exp` to the `.PHONY` list.

In the `help` target (~line 87), add a line:

```make
	@echo "  make build-dashboard-exp    - build dashboard bundle (auto-picks freshest fit needing build)"
	@echo "  make build-dashboard-exp ID=N - force rebuild for a specific experiment"
```

- [ ] **Step 3: Local sanity check the Make syntax**

```bash
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno/analysis/cloud
make help | grep -A 1 build-dashboard-exp
```

Expected: both new help lines appear.

- [ ] **Step 4: Commit**

```bash
git add analysis/cloud/Makefile
git commit -m "feat(cloud/make): build-dashboard-exp target with optional ID"
```

- [ ] **Step 5: Cluster smoke (manual; user-driven, not part of the plan)**

After pushing, the user will run on the cluster:

```bash
git pull
make build-dashboard-exp           # should auto-pick id=1 (pilot, never built)
# OR
make build-dashboard-exp ID=1      # explicit force
```

Expected output:
- `[run-exp] --build-only auto-selected id=1 (fit newer than dashboard bundle)` (first form)
- A streamed build with phase markers (load checkpoint, BQ load, vectorize,
  write bundle, zip bundle)
- New section appended to `runs/0001-pilot/summary.md`:
  `## Dashboard build — <ts>` ... `### Build complete (exit 0)`
- `runs/0001-pilot/0001-pilot-dashboard.zip` exists alongside
  `dashboard_bundle/`

Second invocation (immediately after) should print
`[run-exp] no fits need building` and exit 0, confirming the staleness
check works.

User pastes back `make summary ID=1` so the pilot record's `## Dashboard
build` section can be embedded.

---

## Verification

After all six tasks land:

```bash
poetry run pytest scripts/tests/test_run_experiment.py -v
```

Expected: all ~62 tests PASS, including:
- 7 in `TestFindMostRecentFitNeedingBuild`
- 3 in `TestBuildDashboardArgs`
- 2 in `TestWriteBuildSectionHeader`
- 3 in `TestBuildOnlyMain`

Cluster verification: pilot 0001's `summary.md` gains a third
section type (`## Dashboard build —`), pilot dir gains
`0001-pilot-dashboard.zip`, and the `make build-dashboard-exp` second
invocation correctly reports "no fits need building."

---

## Out of Scope / Follow-Ups

These remain parked for future increments (cited from the Inc 3 spec):

1. **HDP support** (`model_class: hdp`).
2. **`warm_start_from: NNNN`** field + checkpoint copy.
3. **Structured per-iter parsing** (`### Iter N` subsections).
4. **`make diff-exp A=N B=M`** cross-experiment comparison.
5. **Spark config promotion to YAML.**
6. **Per-deployment `runs-dir` override** via host-config file.
7. **`status: archived` workflow / `make list-pending`.**
8. **Cross-link convention enforcement.**

# Experiment Tracking — Increment 2.5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users split fit and eval cleanly when desired, without losing the integrated default. Add a `NO_EVAL=1` opt-out for the fit-then-eval Make targets, and let `make eval-exp` (no `ID=`) auto-discover the most-recently-fit experiment.

**Architecture:** Three small surface additions: (1) a `find_most_recent_fit` helper that walks `$RUNS_DIR/<NNNN-slug>/manifest.json` files and picks the one with the latest mtime; (2) `--no-eval` flag in `main()` that skips the eval block; (3) `--eval-only` without `--id` falls back to the auto-discovery helper. Makefile targets gain a `$(if $(NO_EVAL),--no-eval)` conditional and `eval-exp` makes `ID` optional.

**Tech Stack:** Python 3.11, GNU Make conditional functions, pytest.

**Scope boundary:** Strictly the integrated-vs-split ergonomic. No new model classes, no new file layouts, no checkpoint-format changes. Builds straight on Increment 2.

---

## File Structure

**Modified:**
- `scripts/run_experiment.py` — `find_most_recent_fit` helper; `--no-eval` flag in `main()`; auto-discovery branch when `--eval-only` and no `--id`/`--next`; contradictory-flag guard.
- `scripts/tests/test_run_experiment.py` — unit tests for `find_most_recent_fit`.
- `analysis/cloud/Makefile` — `next-exp` and `exp` forward `$(if $(NO_EVAL),--no-eval)`; `eval-exp` makes `ID` optional and passes `$(if $(ID),--id $(ID))`.

**Not touched:**
- `--eval-only` semantics when `--id` IS provided (unchanged).
- The fit dispatch path and any of the summary-writing logic.
- Experiment record schema, defaults files, driver scripts.

---

## Task 1: `find_most_recent_fit` helper

**Files:**
- Modify: `scripts/run_experiment.py`
- Modify: `scripts/tests/test_run_experiment.py`

**Goal:** pure-ish helper (filesystem-only side effect) that scans `<runs_dir>/<NNNN-slug>/manifest.json` and returns the integer experiment id of the most-recently-modified one, or `None` if no checkpoints exist.

- [ ] **Step 1: Write the failing tests**

Append to `scripts/tests/test_run_experiment.py`:

```python
def test_find_most_recent_fit_returns_none_for_empty_dir(tmp_path):
    assert rx.find_most_recent_fit(tmp_path) is None


def test_find_most_recent_fit_picks_latest_mtime(tmp_path):
    """Three experiment dirs, only the second has the most recent manifest mtime."""
    # 0010-old/manifest.json — oldest
    (tmp_path / "0010-old").mkdir()
    older = tmp_path / "0010-old" / "manifest.json"
    older.write_text("{}")
    import os
    os.utime(older, (1_700_000_000, 1_700_000_000))

    # 0042-target/manifest.json — newest
    (tmp_path / "0042-target").mkdir()
    target = tmp_path / "0042-target" / "manifest.json"
    target.write_text("{}")
    os.utime(target, (1_700_000_200, 1_700_000_200))

    # 0050-middle/manifest.json — between
    (tmp_path / "0050-middle").mkdir()
    middle = tmp_path / "0050-middle" / "manifest.json"
    middle.write_text("{}")
    os.utime(middle, (1_700_000_100, 1_700_000_100))

    assert rx.find_most_recent_fit(tmp_path) == 42


def test_find_most_recent_fit_ignores_dirs_without_manifest(tmp_path):
    """A save_dir without manifest.json is in-progress / failed, not eligible."""
    (tmp_path / "0001-pending").mkdir()  # no manifest.json — ignored
    (tmp_path / "0002-done").mkdir()
    (tmp_path / "0002-done" / "manifest.json").write_text("{}")
    assert rx.find_most_recent_fit(tmp_path) == 2


def test_find_most_recent_fit_skips_malformed_dir_names(tmp_path):
    """Dirs not matching NNNN-* are ignored even if they have a manifest."""
    (tmp_path / "scratch").mkdir()
    (tmp_path / "scratch" / "manifest.json").write_text("{}")
    (tmp_path / "0007-real").mkdir()
    (tmp_path / "0007-real" / "manifest.json").write_text("{}")
    assert rx.find_most_recent_fit(tmp_path) == 7


def test_find_most_recent_fit_missing_runs_dir_returns_none(tmp_path):
    """Nonexistent runs_dir is treated as empty."""
    assert rx.find_most_recent_fit(tmp_path / "does-not-exist") is None
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno
poetry run pytest scripts/tests/test_run_experiment.py -v
```
Expected: 5 new tests FAIL with `AttributeError: module 'run_experiment' has no attribute 'find_most_recent_fit'`.

- [ ] **Step 3: Add `find_most_recent_fit` to run_experiment.py**

Append after `find_by_id` in `scripts/run_experiment.py`:

```python
_RUN_DIR_PATTERN = re.compile(r"^(\d{4})-")


def find_most_recent_fit(runs_dir: Path) -> int | None:
    """Return the experiment id of the most-recently-checkpointed fit.

    Walks `runs_dir/<NNNN-slug>/manifest.json`, picks the one with the latest
    mtime, parses the leading NNNN from the directory name. Returns None if
    no checkpoint exists (empty / nonexistent runs_dir, or no manifest.json
    files in any subdirectory).

    The dir-naming convention `NNNN-slug` is enforced by `main()` when it
    creates save_dir; this function trusts that convention and skips any
    directory whose name doesn't start with four digits + hyphen.
    """
    if not runs_dir.is_dir():
        return None
    best_mtime = -1.0
    best_id: int | None = None
    for manifest in runs_dir.glob("*/manifest.json"):
        m = _RUN_DIR_PATTERN.match(manifest.parent.name)
        if m is None:
            continue
        mtime = manifest.stat().st_mtime
        if mtime > best_mtime:
            best_mtime = mtime
            best_id = int(m.group(1))
    return best_id
```

- [ ] **Step 4: Run tests to verify pass**

```bash
poetry run pytest scripts/tests/test_run_experiment.py -v
```
Expected: all 49 tests PASS (44 prior + 5 new).

- [ ] **Step 5: Commit**

```bash
git add scripts/run_experiment.py scripts/tests/test_run_experiment.py
git commit -m "feat(scripts): find_most_recent_fit helper for auto-discovery in eval-exp"
```

---

## Task 2: `--no-eval` flag + `--eval-only` auto-discovery in `main()`

**Files:**
- Modify: `scripts/run_experiment.py`

**Goal:** add a `--no-eval` flag that skips the eval block at the end of `main()`; allow `--eval-only` without `--id`/`--next` to fall back to `find_most_recent_fit`; guard the contradictory `--no-eval --eval-only` combination.

This task is orchestration plumbing — no new unit tests; the helpers it composes are already tested. We verify by `--help` and a planning-only smoke.

- [ ] **Step 1: Update the argparse selector to be conditionally required**

Currently in `main()`:

```python
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--next", action="store_true",
                          help="Pick the lowest-id experiment with status: pending.")
    selector.add_argument("--id", type=int, default=None,
                          help="Run the experiment with the given numeric id.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip fit dispatch; only run eval against the existing "
                             "checkpoint at $RUNS_DIR/NNNN-slug/. Requires checkpoint "
                             "manifest.json to exist.")
```

Change `required=True` to `required=False` and add `--no-eval`. The argparse layer accepts "neither --next nor --id"; main() validates that combination against `--eval-only`:

```python
    selector = parser.add_mutually_exclusive_group(required=False)
    selector.add_argument("--next", action="store_true",
                          help="Pick the lowest-id experiment with status: pending.")
    selector.add_argument("--id", type=int, default=None,
                          help="Run the experiment with the given numeric id.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip fit dispatch; only run eval against the existing "
                             "checkpoint at $RUNS_DIR/NNNN-slug/. If --id is omitted, "
                             "auto-selects the most-recently-fit experiment "
                             "(latest manifest.json mtime under $RUNS_DIR).")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip the eval dispatch at the end of the run. Fit only.")
```

- [ ] **Step 2: Add contradictory-flag guard at the top of main()**

Immediately after the `args = parser.parse_args(argv)` line, before the "Select experiment file" step, add:

```python
    if args.no_eval and args.eval_only:
        print("[run-exp] ERROR: --no-eval and --eval-only are contradictory", flush=True)
        return 2
```

- [ ] **Step 3: Rework Step 1 (experiment selection) to support eval-only auto-discovery**

Currently Step 1 looks like:

```python
    # 1. Select experiment file
    if args.next:
        exp_path = find_next_pending(args.experiments_dir)
        if exp_path is None:
            print("[run-exp] no pending experiments found", flush=True)
            return 1
    else:
        try:
            exp_path = find_by_id(args.experiments_dir, args.id)
        except FileNotFoundError as e:
            print(f"[run-exp] ERROR: {e}", flush=True)
            return 2
    print(f"[run-exp] experiment: {exp_path}", flush=True)
```

This works when `--next` or `--id` is set. But if neither is set, `args.id` is None and `find_by_id(..., None)` would crash. Replace with explicit branching:

```python
    # 1. Select experiment file
    if args.next:
        exp_path = find_next_pending(args.experiments_dir)
        if exp_path is None:
            print("[run-exp] no pending experiments found", flush=True)
            return 1
    elif args.id is not None:
        try:
            exp_path = find_by_id(args.experiments_dir, args.id)
        except FileNotFoundError as e:
            print(f"[run-exp] ERROR: {e}", flush=True)
            return 2
    elif args.eval_only:
        # No --id with --eval-only: auto-select most-recently-fit experiment.
        runs_dir_path = Path(args.runs_dir)
        auto_id = find_most_recent_fit(runs_dir_path)
        if auto_id is None:
            print(f"[run-exp] ERROR: --eval-only without --id requires at least one "
                  f"checkpoint under {runs_dir_path}; none found", flush=True)
            return 2
        print(f"[run-exp] --eval-only auto-selected most-recent-fit id={auto_id}",
              flush=True)
        try:
            exp_path = find_by_id(args.experiments_dir, auto_id)
        except FileNotFoundError as e:
            print(f"[run-exp] ERROR: {e}", flush=True)
            return 2
    else:
        print("[run-exp] ERROR: provide --next, --id N, or --eval-only "
              "(auto-discovers most-recent-fit)", flush=True)
        return 2
    print(f"[run-exp] experiment: {exp_path}", flush=True)
```

- [ ] **Step 4: Guard the eval dispatch with --no-eval**

The current Step 6 (eval dispatch) in main() starts with `# 6. Dispatch eval (capture stdout ...)`. Just before that block, add:

```python
    # Skip eval dispatch when --no-eval (fit-only mode).
    if args.no_eval:
        print("[run-exp] --no-eval: skipping eval dispatch", flush=True)
        print(f"[run-exp] DONE. summary at: {summary_path}", flush=True)
        return 0
```

The rest of Step 6 stays unchanged below.

- [ ] **Step 5: Verify `--help` shows both new flags**

```bash
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno
poetry run python scripts/run_experiment.py --help
```
Expected: `--no-eval` line present; `--eval-only` help text mentions auto-selection.

- [ ] **Step 6: Verify the unit tests still pass**

```bash
poetry run pytest scripts/tests/test_run_experiment.py -v
```
Expected: 49 tests PASS.

- [ ] **Step 7: Local smoke — `--eval-only` with no `--id` errors when no checkpoints exist**

```bash
RUNS=$(mktemp -d) && poetry run python scripts/run_experiment.py --eval-only --runs-dir "$RUNS" 2>&1 | head -5
echo "exit code: $?"
```

Expected:
```
[run-exp] ERROR: --eval-only without --id requires at least one checkpoint under /tmp/.../runs; none found
exit code: 2
```

- [ ] **Step 8: Local smoke — `--eval-only` with no `--id` finds the right one when a checkpoint exists**

```bash
RUNS=$(mktemp -d)
mkdir -p "$RUNS/0001-pilot"
echo "{}" > "$RUNS/0001-pilot/manifest.json"
poetry run python scripts/run_experiment.py --eval-only --runs-dir "$RUNS" 2>&1 | head -5
```

Expected: `[run-exp] --eval-only auto-selected most-recent-fit id=1` then `[run-exp] experiment: docs/experiments/0001-pilot.md`, then likely a spark-submit FileNotFoundError (since spark isn't installed locally). The point is the auto-selection fired and resolved the right experiment file.

- [ ] **Step 9: Local smoke — contradictory flags rejected**

```bash
poetry run python scripts/run_experiment.py --id 1 --no-eval --eval-only 2>&1 | head -3
echo "exit code: $?"
```

Expected: `[run-exp] ERROR: --no-eval and --eval-only are contradictory` and exit code 2.

- [ ] **Step 10: Commit**

```bash
git add scripts/run_experiment.py
git commit -m "feat(scripts): --no-eval flag + --eval-only auto-discovery in main()"
```

---

## Task 3: Makefile updates

**Files:**
- Modify: `analysis/cloud/Makefile`

**Goal:** forward `NO_EVAL=1` from `make next-exp` / `make exp` to the runner; make `ID` optional on `make eval-exp` so the runner can auto-discover.

- [ ] **Step 1: Add `$(if $(NO_EVAL),--no-eval)` to `next-exp` and `exp`**

Current `next-exp` target:
```make
next-exp: zip $(WORKSPACE_ENV)
	. ./$(WORKSPACE_ENV) && \
	python $(REPO_ROOT)/scripts/run_experiment.py --next --runs-dir $(RUNS_DIR)
```

Change to:
```make
next-exp: zip $(WORKSPACE_ENV)
	. ./$(WORKSPACE_ENV) && \
	python $(REPO_ROOT)/scripts/run_experiment.py --next --runs-dir $(RUNS_DIR) $(if $(NO_EVAL),--no-eval)
```

Current `exp` target:
```make
exp: zip $(WORKSPACE_ENV)
	@if [ -z "$(ID)" ]; then echo "ERROR: provide ID=N"; exit 1; fi
	. ./$(WORKSPACE_ENV) && \
	python $(REPO_ROOT)/scripts/run_experiment.py --id $(ID) --runs-dir $(RUNS_DIR)
```

Change to:
```make
exp: zip $(WORKSPACE_ENV)
	@if [ -z "$(ID)" ]; then echo "ERROR: provide ID=N"; exit 1; fi
	. ./$(WORKSPACE_ENV) && \
	python $(REPO_ROOT)/scripts/run_experiment.py --id $(ID) --runs-dir $(RUNS_DIR) $(if $(NO_EVAL),--no-eval)
```

- [ ] **Step 2: Make `ID` optional on `eval-exp` and forward conditionally**

Current `eval-exp` target:
```make
eval-exp: zip $(WORKSPACE_ENV)
	@if [ -z "$(ID)" ]; then echo "ERROR: provide ID=N"; exit 1; fi
	. ./$(WORKSPACE_ENV) && \
	python $(REPO_ROOT)/scripts/run_experiment.py --id $(ID) --eval-only --runs-dir $(RUNS_DIR)
```

Change to:
```make
eval-exp: zip $(WORKSPACE_ENV)
	. ./$(WORKSPACE_ENV) && \
	python $(REPO_ROOT)/scripts/run_experiment.py $(if $(ID),--id $(ID)) --eval-only --runs-dir $(RUNS_DIR)
```

Remove the `@if [ -z "$(ID)" ]; then ...` guard — the runner handles the no-ID case via auto-discovery (and errors out if no checkpoints exist).

- [ ] **Step 3: Update the help lines**

Find the existing help-block lines added in Inc 1+2 (look for `@echo "  next-exp ..."`, etc.). Update them to document the new flags:

Find:
```make
	@echo "  next-exp                Pick lowest-NNNN pending experiment; fit + eval."
	@echo "  exp ID=N                Run/resume experiment N; fit + eval."
	@echo "  summary ID=N            Print summary.md for experiment N."
	@echo "  eval-exp ID=N           Re-run eval against existing checkpoint; appends ## Eval section."
```

Replace with:
```make
	@echo "  next-exp [NO_EVAL=1]    Pick lowest-NNNN pending experiment; fit (+ eval unless NO_EVAL)."
	@echo "  exp ID=N [NO_EVAL=1]    Run/resume experiment N; fit (+ eval unless NO_EVAL)."
	@echo "  summary ID=N            Print summary.md for experiment N."
	@echo "  eval-exp [ID=N]         Re-run eval; if ID omitted, auto-selects most-recently-fit."
```

- [ ] **Step 4: Verify Make can parse all three targets**

```bash
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno
make -C analysis/cloud -n next-exp 2>&1 | tail -3
make -C analysis/cloud -n next-exp NO_EVAL=1 2>&1 | tail -3
make -C analysis/cloud -n eval-exp 2>&1 | tail -3
make -C analysis/cloud -n eval-exp ID=1 2>&1 | tail -3
```

Expected: all four commands dry-run without Make syntax errors. The `NO_EVAL=1` invocation's dry-run output should show `--no-eval` at the end of the spark-submit args. The no-ID `eval-exp` dry-run should NOT show `--id` in the args. The `ID=1` invocation should show `--id 1`.

- [ ] **Step 5: Commit**

```bash
git add analysis/cloud/Makefile
git commit -m "feat(cloud/make): NO_EVAL=1 opt-out + optional ID for eval-exp auto-discovery"
```

---

## Verification

After all 3 tasks:

```bash
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno
poetry run pytest scripts/tests/test_run_experiment.py -v
```
Expected: 49 tests PASS (44 prior + 5 new from Task 1).

Local planning smokes verified inline (Task 2 Steps 7-9, Task 3 Step 4).

Cluster-side smoke (manual, when next on cluster):

```bash
git pull
# Author 0002 (or flip 0001-pilot back to pending), then:
make -C analysis/cloud next-exp NO_EVAL=1
# Fit runs; no eval at the end.
make -C analysis/cloud eval-exp
# Auto-discovers most-recently-fit experiment (0002); runs eval; appends timestamped ## Eval (NPMI) section to summary.md.
```

To verify the auto-discovery picks the right one when multiple checkpoints exist: run two fits (in sequence, with NO_EVAL=1), then `make eval-exp` — should target the second one. If you want to eval the first, `make eval-exp ID=<first-id>` explicit form.

---

## Out of Scope / Follow-Ups for Increment 3

(Unchanged from the Inc 2 plan.)

1. Structured per-iter parsing into `### Iter N` markdown subsections.
2. HDP support (`model_class: hdp`).
3. `warm_start_from: NNNN` field + checkpoint copy logic.
4. `build-dashboard-exp ID=N OUT=X` target.
5. Spark config promotion to YAML.
6. Per-deployment runs-dir override.

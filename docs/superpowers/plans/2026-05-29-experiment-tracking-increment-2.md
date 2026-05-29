# Experiment Tracking — Increment 2 (Lean) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `summary.md` high-signal (drop Spark/Hadoop noise), cleanly record killed fits (SIGTERM trap), and let eval be re-run independently against an existing checkpoint.

**Architecture:** Additive on Increment 1. The streaming wrapper grows: (1) a second pattern list (`NOISE_PATTERNS`) sanitizes cluster-log noise; (2) a SIGTERM/SIGINT trap parses last-seen iter from the tee loop and writes a `### Killed at iter N` marker before exiting; (3) `main()` learns an `--eval-only` mode that skips fit dispatch and just runs eval; (4) the `## Eval (NPMI)` section is timestamped and gains a `### Eval complete (exit 0)` marker for consistency with the fit `### Session complete` pattern.

**Tech Stack:** Python 3.11, `signal`, `subprocess`, pytest, GNU Make.

**Scope boundary:** Lean Inc 2. **Deferred** to Increment 3 / follow-on plans: structured per-iter markdown parsing (`### Iter N` subsections), HDP support (`model_class: hdp`), `warm_start_from`, `build-dashboard-exp`.

**Pilot context that informed this plan:** Experiment 0001 produced a `summary.md` where the high-signal `[driver]` lines were buried among ~30 lines of Hadoop/Spark/YARN INFO logs per fit session. The information was correct; the readability was poor. Filtering noise is the highest-ROI change for Inc 2. Pilot also surfaced a small consistency gap: the fit section ends with `### Session complete (exit 0)`, but the eval section has no equivalent — easy to add alongside the timestamping.

---

## File Structure

**Modified:**
- `scripts/run_experiment.py` — `NOISE_PATTERNS` constant; `parse_iter_marker` function; signal-trap logic inside `run_subprocess_tee_sanitize`; timestamped `## Eval (NPMI)` header + `### Eval complete` marker in `append_eval_section`; `--eval-only` flag + branch in `main()`.
- `scripts/tests/test_run_experiment.py` — new tests for noise filter, iter parser, SIGTERM trap (integration), eval-section timestamping.
- `analysis/cloud/Makefile` — new `eval-exp` target + help line.

**Not touched:**
- The Inc 1 schema for the experiment record file (`docs/experiments/NNNN-slug.md`) and the defaults files.
- The driver-side `lda_bigquery_cloud.py` (Inc 1 stripped the patient-hash transform sample; nothing else needed here).
- HDP driver, dashboard build pipeline.

---

## Module additions inside `scripts/run_experiment.py`

| Symbol | Type | Responsibility |
|---|---|---|
| `NOISE_PATTERNS` | constant | regex list matching Spark/Hadoop/YARN INFO log lines |
| `DROP_PATTERNS` | constant | `PATIENT_PATTERNS + NOISE_PATTERNS` (composed once, used at call sites) |
| `parse_iter_marker(line)` | pure | returns iter number from a `[driver] iter N/M:` line, else None |
| Signal trap | logic added to `run_subprocess_tee_sanitize` | catches SIGTERM/SIGINT, writes killed marker, terminates subprocess, returns 130 |
| Eval section timestamping | logic added to `append_eval_section` | header becomes `## Eval (NPMI) — YYYY-MM-DD HH:MM:SS UTC`; appends `### Eval complete (exit N)` marker |
| `--eval-only` flag | argparse + branching in `main()` | when set, skips fit dispatch and runs only the eval block |

`run_subprocess_tee_sanitize` and `append_eval_section` keep their existing signatures — additions are internal. The constant change at call sites is `PATIENT_PATTERNS` → `DROP_PATTERNS` (or callers continue to pass `PATIENT_PATTERNS` and we update the canonical constant to alias the union).

---

## Task 1: `NOISE_PATTERNS` + `DROP_PATTERNS` composition

**Files:**
- Modify: `scripts/run_experiment.py`
- Modify: `scripts/tests/test_run_experiment.py`

**Goal:** filter Spark/Hadoop/YARN INFO log lines from `summary.md` while keeping all `[driver]` content. Continue dropping patient-info lines (existing behavior).

- [ ] **Step 1: Write the failing tests**

Append to `scripts/tests/test_run_experiment.py`:

```python
def test_noise_patterns_drops_hadoop_info_log():
    line = "26/05/28 20:46:09 INFO Configuration: resource-types.xml not found\n"
    assert rx.sanitize_line(line, rx.DROP_PATTERNS) is None


def test_noise_patterns_drops_yarn_log():
    line = "26/05/28 20:46:10 INFO YarnClientImpl: Submitted application application_1779908374279_0012\n"
    assert rx.sanitize_line(line, rx.DROP_PATTERNS) is None


def test_noise_patterns_drops_gcs_chatter():
    line = "26/05/28 20:46:13 INFO GoogleHadoopOutputStream: hflush(): No-op due to rate limit ...\n"
    assert rx.sanitize_line(line, rx.DROP_PATTERNS) is None


def test_noise_patterns_keep_driver_lines():
    lines = [
        "[driver] Spark 3.5.3, master=yarn, defaultParallelism=2\n",
        "[driver]   iter 1/2: ELBO=-97881.5970, batch=245, rho=0.0538, 86.4s\n",
        "[driver]   --- topics @ iter 1 ---\n",
        "[driver]    topic  3  α=0.1969  E[β]=0.2405  Σλ=1.58e+03  ...\n",
        "[driver] fit complete\n",
    ]
    for ln in lines:
        assert rx.sanitize_line(ln, rx.DROP_PATTERNS) == ln, f"unexpectedly dropped: {ln!r}"


def test_noise_patterns_keep_traceback_lines():
    """Python tracebacks are signal, not noise — they explain failures."""
    lines = [
        "Traceback (most recent call last):\n",
        '  File "/some/path.py", line 42, in main\n',
        "FileNotFoundError: No manifest.json at resumeFrom path: ...\n",
    ]
    for ln in lines:
        assert rx.sanitize_line(ln, rx.DROP_PATTERNS) == ln


def test_drop_patterns_still_drops_patient_info():
    """DROP_PATTERNS must be a superset of PATIENT_PATTERNS."""
    line = "|person_hash|topicDistribution|\n"
    assert rx.sanitize_line(line, rx.DROP_PATTERNS) is None
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno
poetry run pytest scripts/tests/test_run_experiment.py -v
```
Expected: 6 new tests FAIL with `AttributeError: module 'run_experiment' has no attribute 'DROP_PATTERNS'`.

- [ ] **Step 3: Add NOISE_PATTERNS and DROP_PATTERNS to run_experiment.py**

After the existing `PATIENT_PATTERNS` block in `scripts/run_experiment.py`, append:

```python
# Cluster log noise that has no signal value in the committed record.
# These match the standard log4j-style timestamp prefix that Spark, Hadoop,
# YARN, and the GCS connector emit. The wrapper drops them when writing to
# summary.md but still tees them to stdout so they're visible during live
# debugging.
NOISE_PATTERNS: list[re.Pattern] = [
    # log4j format: "YY/MM/DD HH:MM:SS LEVEL Logger: message"
    re.compile(r"^\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2} (INFO|WARN|DEBUG) "),
    # Bracketed CONTEXT ratelimit_period footers from GCS connector
    re.compile(r"\[CONTEXT ratelimit_period="),
]

# Composed: what summary.md drops. Callers pass DROP_PATTERNS at the sanitize
# boundary; the split between PATIENT_PATTERNS (privacy) and NOISE_PATTERNS
# (readability) is documentary.
DROP_PATTERNS: list[re.Pattern] = PATIENT_PATTERNS + NOISE_PATTERNS
```

- [ ] **Step 4: Update call sites to use DROP_PATTERNS**

Two call sites in `scripts/run_experiment.py` currently pass `PATIENT_PATTERNS`:

In `append_eval_section`:
```python
clean = sanitize_line(line, PATIENT_PATTERNS)
```
Change to:
```python
clean = sanitize_line(line, DROP_PATTERNS)
```

In `main()` (the fit dispatch call):
```python
fit_rc = run_subprocess_tee_sanitize(fit_cmd, summary_path, PATIENT_PATTERNS)
```
Change to:
```python
fit_rc = run_subprocess_tee_sanitize(fit_cmd, summary_path, DROP_PATTERNS)
```

(The `run_subprocess_tee_sanitize` function signature stays the same — it takes whatever pattern list the caller passes.)

- [ ] **Step 5: Run tests to verify pass**

```bash
poetry run pytest scripts/tests/test_run_experiment.py -v
```
Expected: all 37 tests PASS (31 prior + 6 new).

- [ ] **Step 6: Commit**

```bash
git add scripts/run_experiment.py scripts/tests/test_run_experiment.py
git commit -m "feat(scripts): NOISE_PATTERNS drops Spark/Hadoop INFO log noise from summary.md"
```

---

## Task 2: `parse_iter_marker` pure function

**Files:**
- Modify: `scripts/run_experiment.py`
- Modify: `scripts/tests/test_run_experiment.py`

**Goal:** parse the iter number from a `[driver]   iter N/M: ...` line. Returns `int | None`. Used by Task 3's SIGTERM trap to know which iter was last seen so the killed marker can be informative.

- [ ] **Step 1: Write the failing tests**

Append to `scripts/tests/test_run_experiment.py`:

```python
def test_parse_iter_marker_extracts_iter_number():
    line = "[driver]   iter 1/2: ELBO=-97881.5970, batch=245, rho=0.0538, 86.4s\n"
    assert rx.parse_iter_marker(line) == 1


def test_parse_iter_marker_extracts_higher_iter():
    line = "[driver]   iter 17/40: ELBO=-1.234e9, batch=512, time=183s\n"
    assert rx.parse_iter_marker(line) == 17


def test_parse_iter_marker_returns_none_for_non_iter_lines():
    lines = [
        "[driver]   --- topics @ iter 1 ---\n",   # not an iter-start line
        "[driver]    topic  3  α=0.1969  ...\n",
        "[driver] fit complete\n",
        "26/05/28 20:46:09 INFO Configuration: ...\n",
        "\n",
    ]
    for ln in lines:
        assert rx.parse_iter_marker(ln) is None, f"unexpectedly matched: {ln!r}"


def test_parse_iter_marker_tolerates_no_newline():
    """In case the wrapper passes a line without trailing newline."""
    assert rx.parse_iter_marker("[driver]   iter 5/10: ELBO=-1.0") == 5
```

- [ ] **Step 2: Run tests to verify failure**

```bash
poetry run pytest scripts/tests/test_run_experiment.py -v
```
Expected: 4 new tests FAIL.

- [ ] **Step 3: Add `parse_iter_marker` to run_experiment.py**

Append after `sanitize_line` in `scripts/run_experiment.py`:

```python
_ITER_MARKER = re.compile(r"^\[driver\]\s+iter\s+(\d+)/\d+:")


def parse_iter_marker(line: str) -> int | None:
    """Return the iter number from a `[driver]   iter N/M: ...` line, or None.

    Used by the SIGTERM trap to know which iter was last in flight so the
    killed marker can be informative.
    """
    m = _ITER_MARKER.match(line)
    return int(m.group(1)) if m else None
```

- [ ] **Step 4: Run tests to verify pass**

```bash
poetry run pytest scripts/tests/test_run_experiment.py -v
```
Expected: all 41 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_experiment.py scripts/tests/test_run_experiment.py
git commit -m "feat(scripts): parse_iter_marker extracts iter N from driver progress lines"
```

---

## Task 3: SIGTERM/SIGINT trap in `run_subprocess_tee_sanitize`

**Files:**
- Modify: `scripts/run_experiment.py`
- Modify: `scripts/tests/test_run_experiment.py`

**Goal:** when the wrapper receives SIGTERM or SIGINT during a fit, write `### Killed at iter N (signal: ...)` to summary.md, terminate the spark-submit child cleanly, and exit with code 130 (standard signal-exit).

**Design:** install signal handlers that raise a sentinel exception; catch it around the tee loop; handle killed-marker + child termination in the `except` block; restore previous handlers in `finally`.

- [ ] **Step 1: Write the failing test**

Append to `scripts/tests/test_run_experiment.py`:

```python
import os
import signal as _signal
import threading


def test_run_subprocess_writes_killed_marker_on_sigterm(tmp_path):
    """SIGTERM during the tee loop writes a killed marker referencing last seen iter."""
    summary_path = tmp_path / "summary.md"
    summary_path.write_text("# header\n")

    # Subprocess prints two iter lines fast, then hangs for 60s.
    cmd = [
        "sh", "-c",
        "echo '[driver]   iter 1/10: ELBO=-1.0 time=1s'; "
        "sleep 0.2; "
        "echo '[driver]   iter 2/10: ELBO=-2.0 time=1s'; "
        "sleep 60",
    ]

    # From a daemon thread, SIGTERM the current process ~0.5s in.
    def send_sigterm():
        import time
        time.sleep(0.5)
        os.kill(os.getpid(), _signal.SIGTERM)
    threading.Thread(target=send_sigterm, daemon=True).start()

    exit_code = rx.run_subprocess_tee_sanitize(cmd, summary_path, rx.DROP_PATTERNS)

    assert exit_code == 130, f"expected signal-exit 130, got {exit_code}"
    text = summary_path.read_text()
    assert "Killed at iter 2" in text, f"missing iter marker in: {text!r}"
    # Original header preserved
    assert "# header" in text
    # Both iter lines tee'd to summary before kill
    assert "iter 1/10" in text
    assert "iter 2/10" in text
```

- [ ] **Step 2: Run test to verify failure**

```bash
poetry run pytest scripts/tests/test_run_experiment.py::test_run_subprocess_writes_killed_marker_on_sigterm -v
```
Expected: FAIL (exit code is not 130; signal kills the test harness instead, or no killed marker is written).

- [ ] **Step 3: Add SIGTERM trap to `run_subprocess_tee_sanitize`**

Add `import signal` near the top of `scripts/run_experiment.py` if not already present (it is not).

Replace the existing `run_subprocess_tee_sanitize` function body with:

```python
class _SignalReceived(Exception):
    """Raised inside the tee loop when SIGTERM or SIGINT arrives."""
    def __init__(self, signum: int):
        super().__init__(f"signal {signum}")
        self.signum = signum


def run_subprocess_tee_sanitize(
    cmd: list[str], summary_path: Path, patterns: list[re.Pattern],
) -> int:
    """Run `cmd` as a subprocess; stream stdout line-by-line.

    Each line is printed to this process's stdout (live debugging) AND, if
    `sanitize_line` returns non-None, appended to `summary_path`.

    If SIGTERM or SIGINT arrives during streaming, writes a
    `### Killed at iter N (signal: ...)` marker to summary_path, terminates
    the child, and returns 130 (standard signal-exit code). Otherwise returns
    the child's exit code.

    Tracks the last `[driver]   iter N/M:` line seen so the killed marker can
    name the iter that was in flight.
    """
    def _handler(signum, frame):  # noqa: ARG001 — frame unused
        raise _SignalReceived(signum)

    prev_term = signal.signal(signal.SIGTERM, _handler)
    prev_int = signal.signal(signal.SIGINT, _handler)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True,
    )
    assert proc.stdout is not None
    last_iter: int | None = None
    try:
        with summary_path.open("a") as fout:
            for line in proc.stdout:
                # Live debugging: always print to terminal
                sys.stdout.write(line)
                sys.stdout.flush()
                # Track iter for the killed-marker, even on lines we don't
                # commit (so we know which iter was in flight when the signal
                # arrived).
                m = parse_iter_marker(line)
                if m is not None:
                    last_iter = m
                # Committed record: sanitized only
                clean = sanitize_line(line, patterns)
                if clean is not None:
                    fout.write(clean)
                    fout.flush()
        return proc.wait()
    except _SignalReceived as sig:
        with summary_path.open("a") as fout:
            iter_str = f"iter {last_iter}" if last_iter is not None else "unknown iter"
            fout.write(f"\n### Killed at {iter_str} (signal: {sig.signum})\n")
            fout.flush()
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return 130
    finally:
        signal.signal(signal.SIGTERM, prev_term)
        signal.signal(signal.SIGINT, prev_int)
```

- [ ] **Step 4: Run test to verify pass**

```bash
poetry run pytest scripts/tests/test_run_experiment.py -v
```
Expected: all 42 tests PASS (Inc 1's 31 + Task 1's 6 + Task 2's 4 + Task 3's 1).

- [ ] **Step 5: Verify the killed-marker test doesn't break sibling tests**

The SIGTERM test signals the test process. If the harness can't restore signal handlers cleanly, subsequent tests may misbehave. Run twice in a row to confirm:

```bash
poetry run pytest scripts/tests/test_run_experiment.py -v
poetry run pytest scripts/tests/test_run_experiment.py -v
```
Expected: both runs show 42 PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_experiment.py scripts/tests/test_run_experiment.py
git commit -m "feat(scripts): SIGTERM/SIGINT trap writes killed-at-iter marker, exits 130"
```

---

## Task 4: Timestamped eval sections + `### Eval complete` marker

**Files:**
- Modify: `scripts/run_experiment.py`
- Modify: `scripts/tests/test_run_experiment.py`

**Goal:** when `## Eval (NPMI)` is appended, include a timestamp in the header (so multiple eval runs are distinguishable in the same summary), and append a `### Eval complete (exit N)` marker for consistency with the fit `### Session complete` pattern.

- [ ] **Step 1: Update the failing tests**

The current `test_append_eval_section_marker_and_body` and `test_append_eval_section_sanitizes_body` assert against the bare `## Eval (NPMI)` header. We need to update them and add a new test for the timestamp + completion marker.

Find these two existing tests in `scripts/tests/test_run_experiment.py` and replace their body, plus append the new test:

```python
def test_append_eval_section_marker_and_body(tmp_path):
    summary_path = tmp_path / "summary.md"
    summary_path.write_text("# header\n\n## Fit session 1\n... fit output ...\n")
    eval_stdout = "mean NPMI: 0.224  median: 0.198\n  topic 0: 0.31\n"
    rx.append_eval_section(summary_path, eval_stdout, exit_code=0)
    text = summary_path.read_text()
    # Header includes timestamp prefix
    assert "## Eval (NPMI)" in text
    # Content present
    assert "mean NPMI: 0.224" in text
    # Original content preserved
    assert "## Fit session 1" in text
    # Completion marker
    assert "### Eval complete (exit 0)" in text


def test_append_eval_section_sanitizes_body(tmp_path):
    summary_path = tmp_path / "summary.md"
    summary_path.write_text("# header\n")
    eval_stdout = (
        "mean NPMI: 0.2\n"
        "|person_hash|topicDistribution|\n"   # patient info — dropped
        "26/05/28 20:46:09 INFO YarnClientImpl: ...\n"  # cluster noise — dropped
        "  topic 0: 0.31\n"
    )
    rx.append_eval_section(summary_path, eval_stdout, exit_code=0)
    text = summary_path.read_text()
    assert "mean NPMI: 0.2" in text
    assert "topic 0: 0.31" in text
    assert "person_hash" not in text
    assert "YarnClientImpl" not in text


def test_append_eval_section_records_nonzero_exit(tmp_path):
    summary_path = tmp_path / "summary.md"
    summary_path.write_text("# header\n")
    rx.append_eval_section(summary_path, "partial output\n", exit_code=1)
    text = summary_path.read_text()
    assert "### Eval complete (exit 1)" in text


def test_append_eval_section_timestamp_in_header(tmp_path):
    """Header includes a UTC timestamp so multiple evals are distinguishable."""
    summary_path = tmp_path / "summary.md"
    summary_path.write_text("# header\n")
    rx.append_eval_section(summary_path, "body\n", exit_code=0)
    text = summary_path.read_text()
    import re
    # Match `## Eval (NPMI) — YYYY-MM-DD HH:MM:SS UTC`
    assert re.search(r"## Eval \(NPMI\) — \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC", text), \
        f"missing timestamped eval header in: {text!r}"
```

- [ ] **Step 2: Run tests to verify failure**

```bash
poetry run pytest scripts/tests/test_run_experiment.py -v
```
Expected: 4 tests FAIL — the two updated ones now demand the marker; the new timestamp test and exit-code test demand the new signature/behavior.

- [ ] **Step 3: Update `append_eval_section`**

Replace the existing `append_eval_section` in `scripts/run_experiment.py`:

```python
def append_eval_section(
    summary_path: Path, eval_stdout: str, *, exit_code: int = 0,
) -> None:
    """Append a sanitized, timestamped '## Eval (NPMI)' section to summary_path.

    Header includes a UTC timestamp so multiple eval runs against the same
    experiment (e.g. `make eval-exp ID=N` re-runs) are distinguishable in
    the same summary file. A `### Eval complete (exit N)` marker is appended
    at the end for consistency with the fit `### Session complete` pattern.
    """
    started = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with summary_path.open("a") as f:
        f.write(f"\n## Eval (NPMI) — {started}\n")
        for line in eval_stdout.splitlines(keepends=True):
            clean = sanitize_line(line, DROP_PATTERNS)
            if clean is not None:
                f.write(clean)
        if not eval_stdout.endswith("\n"):
            f.write("\n")
        f.write(f"\n### Eval complete (exit {exit_code})\n")
```

- [ ] **Step 4: Update the call site in `main()`**

In `main()`, find:
```python
    append_eval_section(summary_path, eval_proc.stdout)
```

Change to:
```python
    append_eval_section(summary_path, eval_proc.stdout, exit_code=eval_proc.returncode)
```

- [ ] **Step 5: Run tests to verify pass**

```bash
poetry run pytest scripts/tests/test_run_experiment.py -v
```
Expected: 44 tests PASS (42 prior + 2 new in this task; the 2 updated tests still count as the same tests).

- [ ] **Step 6: Commit**

```bash
git add scripts/run_experiment.py scripts/tests/test_run_experiment.py
git commit -m "feat(scripts): timestamp eval sections + '### Eval complete' marker"
```

---

## Task 5: `--eval-only` flag in `main()`

**Files:**
- Modify: `scripts/run_experiment.py`

**Goal:** `python scripts/run_experiment.py --id N --eval-only` skips the fit dispatch entirely; runs only the eval block; appends a new `## Eval (NPMI) — <timestamp>` section to the existing `summary.md`. Errors out clearly if the checkpoint doesn't exist.

This task is mostly orchestration plumbing — no new unit tests; the eval dispatch + append behavior is already covered by Tasks 1, 3, 4. We verify by manual `--help` and a planning-only smoke.

- [ ] **Step 1: Update argparse + main() in run_experiment.py**

Replace the argparse selector block in `main()`. Currently:

```python
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--next", action="store_true",
                          help="Pick the lowest-id experiment with status: pending.")
    selector.add_argument("--id", type=int, default=None,
                          help="Run the experiment with the given numeric id.")
```

Add `--eval-only` as a separate (non-exclusive) flag:

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

Then update the dispatch logic. Before Step 5 ("Dispatch fit"), branch on `--eval-only`:

```python
    # 5. Dispatch fit (unless --eval-only)
    if args.eval_only:
        if not (save_dir / "manifest.json").exists():
            print(f"[run-exp] ERROR: --eval-only requires checkpoint at "
                  f"{save_dir}/manifest.json (none found)", flush=True)
            return 2
        print(f"[run-exp] --eval-only: skipping fit dispatch", flush=True)
    else:
        lda_script = REPO_ROOT / "analysis" / "cloud" / "lda_bigquery_cloud.py"
        lda_args = build_lda_args(effective, save_dir, resume_from)
        fit_cmd = build_spark_submit_cmd(str(lda_script), lda_args, REPO_ROOT)
        # Display-only join; cmd is passed as list to Popen/run, not via shell.
        print(f"[run-exp] spark-submit: {' '.join(fit_cmd)}", flush=True)
        fit_rc = run_subprocess_tee_sanitize(fit_cmd, summary_path, DROP_PATTERNS)
        if fit_rc != 0:
            print(f"[run-exp] fit exited non-zero ({fit_rc}); skipping eval", flush=True)
            with summary_path.open("a") as f:
                f.write(f"\n### Session ended with exit code {fit_rc}\n")
            return fit_rc
        with summary_path.open("a") as f:
            f.write("\n### Session complete (exit 0)\n")
```

The Step 6 (eval dispatch) block stays unchanged — it runs in both fit-and-eval and eval-only modes.

Also: when `--eval-only` is set, `write_summary_header` should NOT add a new `## Fit session N` header (since there's no fit session in this invocation). Branch around the existing call:

In the current Step 4:
```python
    # 4. Write summary header (new file or append session marker)
    summary_path = save_dir / "summary.md"
    write_summary_header(
        summary_path, exp_id=fm["id"], slug=fm["slug"], effective=effective,
    )
```

Change to:
```python
    # 4. Write summary header (new file or append session marker)
    #    Skip the fit-session header when --eval-only — we only append an
    #    eval section in that mode.
    summary_path = save_dir / "summary.md"
    if not args.eval_only:
        write_summary_header(
            summary_path, exp_id=fm["id"], slug=fm["slug"], effective=effective,
        )
```

- [ ] **Step 2: Verify `--help` shows the new flag**

```bash
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno
poetry run python scripts/run_experiment.py --help
```
Expected: argparse help shows `--eval-only` with the description.

- [ ] **Step 3: Verify the unit tests still pass**

```bash
poetry run pytest scripts/tests/test_run_experiment.py -v
```
Expected: 44 tests PASS (no new tests in this task; existing tests verify the helpers main() composes).

- [ ] **Step 4: Local smoke — `--eval-only` against a checkpoint-less save_dir errors cleanly**

```bash
RUNS=$(mktemp -d) && mkdir -p "$RUNS/0001-pilot" && poetry run python scripts/run_experiment.py --id 1 --eval-only --runs-dir "$RUNS" 2>&1 | head -10
```
Expected: error message `[run-exp] ERROR: --eval-only requires checkpoint at .../0001-pilot/manifest.json (none found)`; exit code 2.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_experiment.py
git commit -m "feat(scripts): --eval-only flag for re-running eval against existing checkpoint"
```

---

## Task 6: Makefile `eval-exp` target

**Files:**
- Modify: `analysis/cloud/Makefile`

- [ ] **Step 1: Add the target near the other experiment-tracking targets**

In `analysis/cloud/Makefile`, find the `summary:` target block added in Inc 1. After it (before any other unrelated targets), insert:

```make
# Re-run eval only against an existing experiment's checkpoint; appends a
# timestamped ## Eval (NPMI) section to summary.md.
eval-exp: zip $(WORKSPACE_ENV)
	@if [ -z "$(ID)" ]; then echo "ERROR: provide ID=N"; exit 1; fi
	. ./$(WORKSPACE_ENV) && \
	python $(REPO_ROOT)/scripts/run_experiment.py --id $(ID) --eval-only --runs-dir $(RUNS_DIR)
```

- [ ] **Step 2: Add a help line**

Locate the help-block lines added in Inc 1 (`@echo "  next-exp ..."`, `@echo "  exp ID=N ..."`, `@echo "  summary ID=N ..."`). Add a new line in the same style, immediately after the `summary` line:

```make
	@echo "  eval-exp ID=N           Re-run eval against existing checkpoint; appends ## Eval section."
```

- [ ] **Step 3: Verify Make can parse**

```bash
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno
make -C analysis/cloud -n eval-exp ID=1 2>&1 | head -5
```
Expected: prints dry-run commands (or a benign WORKSPACE_ENV-not-found warning) — no syntax error from Make itself.

- [ ] **Step 4: Commit**

```bash
git add analysis/cloud/Makefile
git commit -m "feat(cloud/make): eval-exp ID=N target for re-running eval against checkpoint"
```

---

## Verification

After all 6 tasks:

```bash
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno
poetry run pytest scripts/tests/test_run_experiment.py -v
```
Expected: 44 tests PASS.

Local planning smoke (sanity check that --next still works end-to-end through the planning phase):

```bash
RUNS=$(mktemp -d) && poetry run python scripts/run_experiment.py --next --runs-dir "$RUNS" 2>&1 | head -15
```

Expected: same shape as the Inc 1 verification — finds 0001-pilot (now `status: done` so this should print `[run-exp] no pending experiments found` and exit 1). To re-validate the planning flow, temporarily flip pilot status back to `pending`, run the smoke, then flip back. Or author a 0002 stub. **Don't include this temporary flip in the commit.**

Cluster-side smoke (manual, when next on cluster):

```bash
git pull
# author or set up a pending experiment (e.g. flip 0001-pilot back to pending temporarily,
# or commit a new 0002 record), then:
make -C analysis/cloud next-exp
# observe: summary.md should have NO 26/05/29 INFO lines, only [driver] content
# eval section should have timestamp in header + ### Eval complete marker
make -C analysis/cloud eval-exp ID=1
# (re-runs eval; appends another ## Eval (NPMI) — <ts> section)
```

To verify SIGTERM trap end-to-end: start `make next-exp` on a longer fit, then `Ctrl-C` mid-fit. summary.md should gain `### Killed at iter N (signal: 2)` (SIGINT = 2). `make next-exp` again resumes (since manifest.json doesn't exist yet, it's a fresh fit — note this is a known limitation: SIGTERM mid-fit before first save_interval means no resume; SIGTERM after first save_interval resumes correctly).

---

## Out of Scope / Follow-Ups for Increment 3

(Unchanged from Inc 1 plan, plus what Inc 2 punted.)

1. **Structured per-iter parsing** — turn `[driver] iter N/M:` lines + subsequent topic blocks into `### Iter N` markdown subsections. Modest visual win; defer until there's a concrete cross-experiment comparison use case.
2. **HDP support** (`model_class: hdp`).
3. **`warm_start_from: NNNN`** field + checkpoint copy logic.
4. **`build-dashboard-exp ID=N OUT=X`** target.
5. **Spark config promotion to YAML** (currently hardcoded `SPARK_SUBMIT_FLAGS`).
6. **Per-deployment runs-dir override** (currently hardcoded `DEFAULT_RUNS_DIR`; the `--runs-dir` CLI flag handles override, but a per-host config would be cleaner once a second deployment exists).

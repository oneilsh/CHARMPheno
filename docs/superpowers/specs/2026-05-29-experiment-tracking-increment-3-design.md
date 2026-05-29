# Experiment Tracking — Increment 3 (`build-dashboard-exp`) Design

**Date:** 2026-05-29
**Context:** Increments 1, 2, and 2.5 of the experiment-tracking system are
shipped on `dashboarding` (see
[2026-05-28-experiment-tracking-design.md](2026-05-28-experiment-tracking-design.md)
for the original spec and the Inc 1/2/2.5 plan trio for what landed). Pilot
0001 validated the full chain (fit → resume → fit → eval → re-eval) end-to-end
on the cluster.

Increment 3 adds the third spark-submit stage to the experiment-tracking
wrapper: dashboard-bundle build. Today, building a dashboard bundle from an
experiment's checkpoint requires manually passing `--checkpoint` to
`make build-dashboard-bundle`. This shipment lets the user run

```
make build-dashboard-exp           # auto-discover which exp needs a build
make build-dashboard-exp ID=2      # force rebuild for a specific exp
```

with the same auto-discovery ergonomics as `make eval-exp` (Inc 2.5).

Three other Inc 3 items were scoped out and parked: HDP support in the wrapper
(`model_class: hdp` rejection still in place), `warm_start_from`, and the
larger menu items (structured per-iter parsing, `diff-exp`, Spark config in
YAML, etc.). See "Out of Scope" below.

---

## Motivating use cases

1. **One command to build the bundle for whichever experiment needs it most.**
   After `make next-exp && make eval-exp`, the natural next step is "build the
   dashboard." The wrapper should know which experiment that means without
   the user having to remember the id.
2. **Distinct downloadable zip per experiment.** Today every bundle is
   `dashboard_bundle.zip` regardless of which experiment produced it. When
   the user downloads multiple to `~/Downloads`, they collide and need
   manual renaming. The zip basename should encode the experiment id and
   slug so multiple bundles coexist cleanly.
3. **Self-documenting build record.** The build's stdout (concept-name
   counts, vocab trim outcome, bundle file list) should land in the same
   `summary.md` that already records fit + eval sessions. One record, full
   experiment lifecycle.

---

## Design

### Auto-discovery: "most recent fit needing a build"

A new helper in `scripts/run_experiment.py`:

```python
def find_most_recent_fit_needing_build(runs_dir: Path) -> int | None:
    """Return exp id of the most recent fit whose dashboard bundle is missing
    or older than the fit checkpoint. None if all fits are up-to-date.

    Walks `runs_dir/<NNNN-slug>/manifest.json`. For each, compares its mtime
    against `<NNNN-slug>/dashboard_bundle/corpus_stats.json` mtime. If the
    bundle marker doesn't exist or is older than the manifest, the fit is a
    candidate. Returns the id of the candidate with the latest manifest
    mtime — the "freshest" fit that still wants a build.
    """
```

**Marker file:** `<checkpoint>/dashboard_bundle/corpus_stats.json`. This is
the last of the four JSON files written by `build_dashboard_cloud.py` before
the zip step. Using the JSON (not the zip) keeps the marker stable across
the upcoming zip-naming change. If it exists and is newer than the fit's
`manifest.json`, the bundle is considered current; otherwise the fit needs
a build.

**Why "needing a build" and not "never built":** the latter misses the
re-fit case. After Pilot 0001 was re-fit in Session 4, its existing
session-3 dashboard bundle (if one existed) would be stale. The `manifest
mtime > bundle mtime` test catches both "never built" (no bundle at all)
and "stale" (bundle exists but fit is newer) in one rule.

**Multiple candidates:** pick the one with the latest manifest mtime. The
"freshest fit" is the one the user is most likely to be focused on right
now. If they want a different one, `ID=N` is the escape hatch.

### CLI surface

In `scripts/run_experiment.py`:

- `--build-only` flag — mirrors `--eval-only`. Skips fit and eval; runs only
  the build dispatch against the experiment's existing checkpoint.
- When `--build-only` is set without `--id`, auto-selects via
  `find_most_recent_fit_needing_build`. If none need building, exits 0 with
  a "no fits need building" message (not an error — this is the
  steady-state when everything is built).
- When `--build-only --id N` is set, builds for that specific experiment
  regardless of whether the bundle is current (force-rebuild path).
- `--build-only` is mutually exclusive with `--eval-only` and `--no-eval`.

In `analysis/cloud/Makefile`:

```make
# Build dashboard bundle for an experiment. With no ID, picks the most
# recent fit whose bundle is missing or older than the fit checkpoint.
# With ID=N, force-rebuilds the named experiment.
build-dashboard-exp: zip $(WORKSPACE_ENV)
	. ./$(WORKSPACE_ENV) && \
	python $(REPO_ROOT)/scripts/run_experiment.py $(if $(ID),--id $(ID)) --build-only --runs-dir $(RUNS_DIR)
```

### Distinct zip filename per experiment

A new `--zip-name STR` flag on `build_dashboard_cloud.py`. Defaults to the
existing behavior (`dashboard_bundle.zip` derived from `out_dir.with_suffix(".zip")`)
when unset. When set, that string is the zip basename written as a sibling
of `out_dir`. The bundle directory name (`dashboard_bundle/`) is unchanged.

The wrapper passes `--zip-name {exp_id:04d}-{slug}-dashboard.zip`, so
artifacts in `<checkpoint>/` look like:

```
runs/0001-pilot/
  manifest.json
  ...other checkpoint files...
  dashboard_bundle/
    model.json
    vocab.json
    phenotypes.json
    corpus_stats.json
  0001-pilot-dashboard.zip        ← downloadable, distinct per experiment
  summary.md
```

Pattern: `NNNN-slug-dashboard.zip`. The `NNNN-` prefix sorts by id in
`~/Downloads`; the `-dashboard` suffix distinguishes from other artifacts
the user may download from the cluster.

### Build args plumbed via `build_dashboard_args`

New pure helper in `scripts/run_experiment.py`:

```python
def build_dashboard_args(
    effective: dict, checkpoint_dir: Path, zip_name: str,
) -> list[str]:
    """Build the CLI arg list for build_dashboard_cloud.py."""
```

Passes (always):
- `--checkpoint <save_dir>`
- `--model-class lda` (Inc 3 still LDA-only)
- `--zip-name <NNNN-slug-dashboard.zip>`

Passes (from effective config, with defaults in `_base.yaml`):
- `--vocab-top-n <N>` — added to `_base.yaml` as `vocab_top_n: 5000`
- `--top-n-codes-for-npmi <N>` — added to `_base.yaml` as `top_n_codes_for_npmi: 20`

Both defaults match `build_dashboard_cloud.py`'s argparse defaults; adding
them to the YAML makes them visible and per-experiment overridable.

### `summary.md` integration

A new helper `write_build_section_header(summary_path)` writes
`## Dashboard build — <UTC ts>` to the summary file. The build dispatch
uses the existing `run_subprocess_tee_sanitize` (same streaming-tee
behavior as fit) so the user sees live progress on the cluster terminal.
After successful build, the caller appends `### Build complete (exit N)`
to mirror the eval pattern.

The full session shape in `summary.md` becomes:

```
## Fit session N
...streamed sanitized stdout...
### Session complete (exit 0)

## Eval (NPMI) — 2026-05-29 14:15:42 UTC
...
### Eval complete (exit 0)

## Dashboard build — 2026-05-29 16:30:01 UTC
...streamed sanitized stdout...
### Build complete (exit 0)
```

If the same experiment is re-fit and re-built later, each new section is
timestamped, so the historical record stays intact in one file.

---

## What changes in `build_dashboard_cloud.py`

Two small touches in [analysis/cloud/build_dashboard_cloud.py](analysis/cloud/build_dashboard_cloud.py):

1. Add `--zip-name` argparse entry (string, default empty/None).
2. Around [build_dashboard_cloud.py:268](analysis/cloud/build_dashboard_cloud.py#L268), change

   ```python
   zip_path = out_dir.with_suffix(".zip")
   ```

   to

   ```python
   zip_path = (
       out_dir.parent / args.zip_name if args.zip_name
       else out_dir.with_suffix(".zip")
   )
   ```

That's it — no other script behavior changes. The `make build-dashboard-bundle CHECKPOINT=...` target keeps working with the old default zip name.

---

## File touches summary

**Modified:**
- `scripts/run_experiment.py` — `find_most_recent_fit_needing_build`,
  `build_dashboard_args`, `write_build_section_header`, `--build-only` flag
  + branch in `main()`, mutual-exclusion checks.
- `scripts/tests/test_run_experiment.py` — ~10 new tests across the four
  new functions plus the `--build-only` `main()` paths.
- `analysis/cloud/build_dashboard_cloud.py` — `--zip-name` flag + zip-path
  resolution.
- `experiments/defaults/_base.yaml` — `vocab_top_n: 5000`,
  `top_n_codes_for_npmi: 20` additions.
- `analysis/cloud/Makefile` — new `build-dashboard-exp` target.

**Not touched:**
- The experiment-record schema (`docs/experiments/NNNN-slug.md` layout).
- Per-cohort defaults files (`general.yaml`, `cancer.yaml`, `dementia.yaml`).
- HDP driver / wrapper HDP-branch (still parked).
- Existing `make build-dashboard-bundle CHECKPOINT=...` target (preserved
  for manual checkpoint-path-driven builds).

---

## Decisions made

| Decision | Choice | Rationale |
|---|---|---|
| Auto-discovery rule | "fit newer than dashboard bundle" | Catches both never-built and re-fit-after-build cases in one rule |
| Marker file | `<ck>/dashboard_bundle/corpus_stats.json` | Last JSON written before zip; decoupled from zip-naming change |
| Multiple candidates → tie-break | Newest fit mtime wins | Freshest fit is what the user is likely focused on |
| No candidates → exit code | 0 with message | "All bundles current" is steady-state, not an error |
| Zip filename pattern | `NNNN-slug-dashboard.zip` | Sorts by id, visually distinct from other artifacts |
| Bundle dir name | Unchanged (`dashboard_bundle/`) | Avoids touching the JSON-file consumers downstream |
| Build args in YAML defaults | Yes (`vocab_top_n`, `top_n_codes_for_npmi`) | Consistent with how every other arg is plumbed; enables per-exp override |
| Build progress display | Stream-tee (like fit) | Build is ~5–10 min on the cluster; capture-then-append would feel hung |

---

## Out of Scope (parked for future increments)

- **HDP support** (`model_class: hdp`). Wrapper still rejects non-LDA.
- **`warm_start_from: NNNN`** field + checkpoint copy.
- **Structured per-iter parsing** (`### Iter N` subsections inside fit
  sessions).
- **`make diff-exp A=2 B=3`** cross-experiment comparison.
- **Spark config promotion from `SPARK_SUBMIT_FLAGS` constants to YAML.**
  Build inherits the same hardcoded constants as fit + eval.
- **Per-deployment `runs-dir` override** via host-config file.
- **`status: archived` workflow** — same semantics as `done`; filtering
  tooling deferred.
- **Cross-link / list-pending Make targets.**

---

## Risks

- **`corpus_stats.json` mtime as marker is fragile under tampering.**
  A user could `touch` it and confuse auto-discovery. In practice this is
  the same trust model as `eval-exp`'s manifest-mtime auto-discovery and
  hasn't been a problem.
- **Build args drift between `_base.yaml` and `build_dashboard_cloud.py`
  argparse defaults.** Mitigation: defaults in `_base.yaml` match the
  script's argparse defaults at this shipment; if the script's defaults
  change in a future PR, the YAML must change too. Worth a brief note in
  the script's argparse `help=` strings.
- **`--zip-name` not validated.** A user passing
  `--zip-name "../escape.zip"` could write outside the checkpoint dir.
  Wrapper-generated names are safe by construction; manual invocation of
  `build_dashboard_cloud.py --zip-name` is a footgun. Acceptable risk for
  a local-trust script; can add a path-traversal guard if it bites.
- **Auto-discovery returns "all current" when the user expected a
  rebuild.** The escape hatch is `ID=N`. If users hit this regularly, a
  `FORCE=1` env-var flag for `make build-dashboard-exp` is easy to add.

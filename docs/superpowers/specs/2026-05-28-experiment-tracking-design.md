# Experiment Tracking System — Design

**Date:** 2026-05-28
**Status:** Spec — ready for implementation plan

## Problem

CHARMPheno fits run on a cloud cluster (AoU Researcher Workbench / Dataproc); dev work happens locally. The current workflow has:

- Long spark-submit commands invoked from cohort-specific Makefile targets (`lda-bq-fit-eval-cancer`, etc.) with many hyperparams hardcoded per target.
- Per-fit RUN_ID directories under `$RUNS_DIR` with auto-generated names like `lda_20260528_171134`.
- Checkpoint metadata that includes `corpus_manifest` (rich enough to reproduce a fit from checkpoint alone).
- A `docs/insights/` + `docs/decisions/` + `docs/REVIEW_LOG.md` system that captures empirical findings, architectural choices, and session retrospectives — but with **no bidirectional links to specific fits**.

The pain points:

1. Copy/pasting long argument lists between local chat and cluster terminal loses context (which run was that?) and is error-prone.
2. The auto-generated run names don't capture *why* a fit was run.
3. Per-iter training output is high-signal but lives only in JupyterLab terminal scrollback (bounded; lost on session end).
4. Per-patient sample lines in driver output prevent direct log archival.
5. Stop/restart mid-fit is frequent but the resumption history isn't recorded anywhere durable.
6. Questions like *"why did the fit three weeks ago converge so much more cleanly?"* and *"why can't I import this bundle anymore?"* have no system-level answer; they require git archaeology.
7. The user iterates with fresh fits (no migration tooling) — so each run is meaningful in isolation, but the *narrative* across runs is unrecorded.

## Goals

- Author experiments **locally** in plain text files with low ceremony (a "try K=60 on dementia" experiment is ≤10 lines of frontmatter).
- Run them on the cluster with **short commands** (`git pull && make next-exp`).
- Capture **per-iter training trend**, final eval, and exit status to a single bounded text file (`summary.md`) per experiment, **as the fit progresses** (so killed fits keep a partial record).
- Sanitize patient-derived info so the artifact is committable.
- Round-trip via **copy/paste through JupyterLab** (no git push from cluster, no GitHub auth on cluster).
- Cross-link experiments ↔ insights ↔ decisions ↔ review-log so future queries about "why was X like that?" have answers grounded in concrete runs.
- Compose with existing per-cohort Makefile targets (don't remove them; this flow is additive).

## Non-goals

- Migration tooling for past runs (the user iterates with fresh fits; backfill isn't worth it).
- Cluster-side git push (auth, egress, and data-hygiene complexity outweigh benefit).
- A web UI / dashboard for browsing experiments (the existing dashboard is for *fit outputs*, not experiment metadata; markdown files in `docs/experiments/` are the browse surface).
- Automated dashboard-bundle build/export tracking (dashboard is local-iterative; not principled).
- Experiment-tracking infrastructure as a reusable library (this is CHARMPheno-specific; spark-vi stays generic).

---

## Architecture

### Storage layout

```
docs/experiments/
  0001-slug.md                  # one file per experiment, sequential id
  0002-slug.md
  ...
experiments/
  defaults/
    _base.yaml                  # cross-cohort defaults
    general.yaml                # per-cohort overrides
    cancer.yaml
    dementia.yaml
scripts/
  run_experiment.py             # cluster-side runner
analysis/cloud/Makefile         # gains 4 new targets; per-cohort targets unchanged
```

### Conceptual model

| Concept | Physical form | Source of truth |
|---|---|---|
| Experiment | A `docs/experiments/NNNN-slug.md` file (intent + interpretation) | Git |
| Checkpoint directory | `$RUNS_DIR/NNNN-slug/` on the cluster (model + metadata) | Cluster filesystem |
| Effective config | `summary.md` + `corpus_manifest` (in checkpoint metadata) | Cluster filesystem; pasted into git record post-run |
| Fit history | `## Fit session N` blocks in `summary.md`, appended each session | Cluster filesystem |
| Eval results | `## Eval (NPMI)` block in `summary.md` | Cluster filesystem |
| Interpretation | `## Interpretation` section of the record file | Git |
| Cross-links | `## Links` section of the record file (manually maintained) | Git |

The cluster writes only to its own filesystem. The local repo (with me as authoring agent) writes the record file. The interface is `summary.md` text travelling via copy/paste.

### Record file schema

```yaml
---
id: 0042                        # required, sequential, matches filename + RUN_ID dir
slug: try-k60-dementia          # required, kebab-case
status: pending                 # pending | done | archived
model_class: lda                # lda | hdp | (future: stm)
cohort: dementia                # general | cancer | dementia
created: 2026-05-28

# Optional cross-refs
parent: 0038                    # documentary — previous experiment in lineage
warm_start_from: 0038           # operational — runner copies that checkpoint as starting state

# Overrides only — everything else from defaults/_base.yaml + defaults/<cohort>.yaml
K: 60
---

# Experiment 0042 — try-k60-dementia

## Intent
One paragraph (or less) — what question this fit is meant to answer.
Optional; sparse intent is acceptable.

## Fit history
- 2026-05-28: started, completed 20 iters. (Filled in by me from summary.md.)
- 2026-06-15: continued for 20 more iters (max_iter 20→40). (Continuation.)

## Results
Verbatim paste of summary.md goes here under fenced markdown.

## Interpretation
What I (the user, via me) understood from the results.

## Links
**Upstream (motivating):**
- experiments/0038 — predecessor run
- insights/0023 — what made me want to try this

**Downstream (generated):**
- insights/0026 — what this run produced
- decisions/0023 — what architectural choice it motivated

**Operational:**
- Dashboard bundle: dashboard/public/data/dementia/ (rebuilt 2026-06-15)
- Checkpoint: $RUNS_DIR/0042-try-k60-dementia/ on cluster
```

**Frontmatter rules:**
- `id` is sequential, padded to 4 digits, matches filename's `NNNN`.
- `slug` is kebab-case; together `NNNN-slug` is also the RUN_ID dir name on the cluster.
- `status` is curatorial (queue-filter for `next-exp`); not an operational state.
- Override fields are flat YAML scalars. The merge precedence is `_base.yaml` → `<cohort>.yaml` → frontmatter, with later values winning.

### Defaults files

Example `experiments/defaults/_base.yaml`:

```yaml
# Cross-cohort defaults. Override per cohort or per experiment.
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
```

Example `experiments/defaults/dementia.yaml`:

```yaml
cohort: dementia
# (Add only fields that differ from _base.yaml.)
```

**Drift safety:** the cluster writes the *effective* config (after merge) into `summary.md` and into the checkpoint's `corpus_manifest`. The record file embeds the summary verbatim under `## Results`. So defaults files can evolve over time without invalidating past experiments — each experiment's effective config is frozen at run time on disk and in git (once pasted back).

### `scripts/run_experiment.py`

A Python wrapper that:

1. **Reads the experiment record** — either `--next` (find lowest-NNNN with `status: pending` in `docs/experiments/`) or `--id NNNN` (specific file). Parses the YAML frontmatter.
2. **Merges defaults** — loads `experiments/defaults/_base.yaml` and `experiments/defaults/<cohort>.yaml`, applies frontmatter overrides. Writes the merged dict to `$RUNS_DIR/NNNN-slug/effective_config.yaml`.
3. **Handles warm-start** — if `warm_start_from: M` is set and `$RUNS_DIR/NNNN-slug/` doesn't yet exist, copies `$RUNS_DIR/MMMM-<slug-of-M>/` into the new dir as the starting state.
4. **Detects resume** — if `$RUNS_DIR/NNNN-slug/` exists with a checkpoint, threads `--resume-from $RUNS_DIR/NNNN-slug/` into the spark-submit. Otherwise fresh start.
5. **Writes summary.md header** — opens `$RUNS_DIR/NNNN-slug/summary.md` (append mode), writes `## Fit session N` header (N = count of prior sessions + 1) and the effective config block.
6. **Streams the fit** — `subprocess.Popen(spark-submit, stdout=PIPE)`; reads lines, tees them: stdout (live view) + filtered append to `summary.md` (per-iter blocks structured into markdown).
7. **Sanitization filter** — regex-drops any line matching the patient-info patterns before writing to `summary.md`. Original line still goes to stdout.
8. **Handles signals** — SIGTERM / Ctrl-C: writes `### Killed at iter N (signal: ...)` to summary.md before exiting non-zero. Clean exit: writes `### Session N complete; exit reason: ...`.
9. **Runs eval** — on successful fit, invokes `eval_coherence_cloud.py` against the checkpoint; appends `## Eval (NPMI)` section to `summary.md` with per-topic NPMI table.
10. **(Optional, `--build-dashboard`)** — invokes `build_dashboard_cloud.py` against the checkpoint with `--out-dir dashboard/public/data/<OUT>/`. Doesn't touch `summary.md` by default.

### Makefile targets

Added to `analysis/cloud/Makefile`:

```make
# Pick lowest-NNNN with status: pending; fit + eval; write summary.md
next-exp: zip $(WORKSPACE_ENV)
	@. ./$(WORKSPACE_ENV) && python $(REPO_ROOT)/scripts/run_experiment.py --next

# Same pipeline but pinned to a specific id (ignores status; auto-resumes if dir exists)
exp: zip $(WORKSPACE_ENV)
	@. ./$(WORKSPACE_ENV) && python $(REPO_ROOT)/scripts/run_experiment.py --id $(ID)

# Re-run only eval against existing fit checkpoint; appends eval section
eval-exp: zip $(WORKSPACE_ENV)
	@. ./$(WORKSPACE_ENV) && python $(REPO_ROOT)/scripts/run_experiment.py --id $(ID) --eval-only

# Build dashboard bundle from experiment's checkpoint; out-dir name explicit
build-dashboard-exp: zip $(WORKSPACE_ENV)
	@. ./$(WORKSPACE_ENV) && python $(REPO_ROOT)/scripts/run_experiment.py --id $(ID) --build-dashboard --out $(OUT)

# cat the summary.md for an experiment (for copy/paste back to chat)
summary:
	@cat $(RUNS_DIR)/$(shell printf '%04d' $(ID))-*/summary.md
```

The existing per-cohort targets (`lda-bq-fit-eval-cancer`, `hdp-bq-fit-eval`, etc.) remain unchanged. They are for ad-hoc / debugging fits where authoring an experiment file is overkill.

### `summary.md` schema

Cluster writes (one file per RUN_ID dir):

```markdown
# Experiment NNNN — slug

## Effective config
model_class: lda
cohort: dementia
K: 60
max_iter: 20
... (full effective config — every field that went into the driver call)
git_sha: <commit>          # HEAD on the cluster when the fit started

## Fit session 1
Started: 2026-05-28 17:14:03 UTC

### Per-iter trend
[iter 1] ELBO=-1.2345e9  time=180s
  topic 0: pneumonia, sepsis, copd, ...
  topic 1: dementia, alzheimer, memory, ...
  ...
[iter 2] ELBO=-1.2240e9  time=176s  ΔELBO=+0.84%
  ...
[iter 20] ELBO=-1.1987e9  time=174s  ΔELBO=+0.02%
  ...

### Session result
Completed iter 20 at 2026-05-28 18:32:17 UTC. Exit: max-iter reached.

## Fit session 2 (continuation, 2026-06-15)
...

## Eval (NPMI)
mean NPMI: 0.224  median: 0.198  pair_coverage_mean: 0.81

per-topic:
  topic 0: NPMI=0.31  coverage=0.95  top: pneumonia, sepsis, copd, ...
  ...
```

Size: a 20-iter K=60 fit produces ~400 lines (mostly per-iter topic prints). Comfortably select-all-copyable in JupyterLab terminal via `cat`.

### Sanitization

**Two layers, defense in depth:**

1. **Driver-side strip:** remove the per-iter "patient profile sample" printing from `lda_bigquery_cloud.py` (and any equivalent in hdp/eval drivers). It's not high-signal — spot-checking can be done by hand on the cluster terminal. Removing it at source is the cleanest fix.

2. **Wrapper-side regex filter:** in `run_experiment.py`'s streaming tee, drop any line matching `patient_profile|person_id=|hash:[a-f0-9]+` (and similar patterns) from the `summary.md` append. The original line still goes to stdout for live debugging. Defends against future driver changes that reintroduce per-patient diagnostics without the developer knowing about the sanitization layer.

The resulting `summary.md` is doubly-clean and safe to commit verbatim into `docs/experiments/NNNN-slug.md` under `## Results`.

### Cross-system linking conventions

Four systems, independent numbering, bidirectional links by `<system>/<id>`:

- `experiments/0042` ↔ `insights/0026` ↔ `decisions/0023` ↔ `docs/REVIEW_LOG.md` entries
- Each experiment record's `## Links` section is manually maintained (I update it when paste-back happens).
- Insights and decisions that cite an experiment back-link in their own body.
- REVIEW_LOG entries cite experiments by id when summarizing a session.

Numbering is independent per system: experiment 0042 ≠ insight 0042. Cross-refs use the full `<system>/<id>` form to avoid ambiguity.

### Operational state vs. curatorial state

| Question | Where the answer lives |
|---|---|
| "Has experiment N been run?" | Does `$RUNS_DIR/NNNN-slug/` exist on the cluster? |
| "Is fit N complete?" | Does `summary.md` end with `Session N complete`? |
| "Has the user interpreted N?" | `status: done` in the record's frontmatter |
| "Should `next-exp` pick this up?" | `status: pending` in the record's frontmatter |

The runner does **not** modify the record file's frontmatter. Status transitions are entirely manual (I propose `pending → done` when paste-back happens; user confirms). This keeps the cluster ↔ local interface text-only.

---

## Workflow examples

### Authoring + first run

1. **User** in chat: "let's try K=60 on dementia, see if Lewy body / vascular separate."
2. **I** create `docs/experiments/0042-try-k60-dementia.md` with frontmatter (K=60, cohort=dementia, status=pending) and a one-line intent block.
3. **I** commit; user pushes from local.
4. **User** on cluster: `git pull && make next-exp`. Runner reads 0042, merges defaults, dispatches spark-submit. Streams per-iter prints to terminal AND `summary.md`. Eval runs at end.
5. **User** on cluster: `make summary ID=42`. Cat output, select-all, copy.
6. **User** in chat: pastes the summary.
7. **I** update `docs/experiments/0042-try-k60-dementia.md` — embed the summary verbatim under `## Results`, write `## Interpretation`, propose `status: done`. Commit.

### Stop/restart mid-fit

1. User starts `make next-exp` for 0042. Reaches iter 12.
2. User sends Ctrl-C. Wrapper traps SIGTERM, appends `### Killed at iter 12 (signal: SIGTERM)` to `summary.md`, exits.
3. `summary.md` on disk has Session 1 header + iters 1-12 + killed marker.
4. User: `make next-exp` again. Runner sees `$RUNS_DIR/0042-try-k60-dementia/` exists with a checkpoint → auto-resume. Writes `## Fit session 2 (resumed, 2026-05-28 …)` header to summary.md. Continues from iter 12.
5. When eventually complete: `summary.md` has Session 1 (iters 1-12, killed), Session 2 (iters 13-20, complete), Eval. One bounded file, full history.

### Continuation of a "done" experiment

1. Experiment 0042 is `status: done`, K=60, max_iter=20. User: "I want 20 more iters."
2. **I** bump `max_iter: 40` in 0042's frontmatter, flip `status: done → pending`. Commit.
3. User: `git pull && make exp ID=42` (note: `exp` not `next-exp`, since the file's pending-but-also-has-a-checkpoint is the resume case explicitly).
4. Runner detects existing checkpoint, threads `--resume-from`, writes `## Fit session 3 (continuation, 2026-06-15)`.
5. Paste-back, interpretation updated, status flipped back to `done`.

### Warm-start with new hyperparams

1. User: "now try asymmetric alpha, but warm-start from 0042."
2. **I** create `docs/experiments/0043-asymalpha-from-0042.md` with `warm_start_from: 0042` + `alpha_init: asymmetric`. Commit.
3. User: `git pull && make next-exp`.
4. Runner sees `warm_start_from`, copies `$RUNS_DIR/0042-…/` into `$RUNS_DIR/0043-…/`, runs fit with new args against the copied checkpoint.
5. Two independent run dirs, two independent records, traceable lineage via `parent` + `warm_start_from`.

### Re-export dashboard

1. Experiment 0042 done; new prevalence semantics shipped in the exporter. Want a new bundle.
2. User: `make build-dashboard-exp ID=42 OUT=dementia`.
3. Runner invokes `build_dashboard_cloud.py` against 0042's checkpoint with `--out-dir dashboard/public/data/dementia/`.
4. No automatic record update. If notable, **I** add a one-liner to the record's `## Links → Operational` section manually.

---

## Implementation increments

Three independent increments. Each lands as its own PR / commit; stopping after any of them leaves a working system.

**Increment 1 (foundation):**
- `experiments/defaults/{_base, general, cancer, dementia}.yaml`
- `scripts/run_experiment.py` skeleton — frontmatter parser, defaults merge, subprocess dispatch to existing `lda_bigquery_cloud.py` and `eval_coherence_cloud.py`
- Three Makefile targets: `next-exp`, `exp ID=`, `summary ID=`
- Skeleton `docs/experiments/0001-pilot.md` for end-to-end validation on a fast run
- Driver-side strip of patient-profile sample
- Wrapper-side regex sanitization filter
- Basic `summary.md` writing: header + effective config + plain stdout capture (not yet structured into Fit Session sections)

**Increment 2 (streaming + structured summary):**
- Streaming wrapper with line-by-line tee
- Per-iter parser → `## Fit session N` markdown structure
- SIGTERM trap → "killed at iter N" marker
- Resume detection → new fit-session header
- `eval-exp ID=` target + eval section append
- `## Eval (NPMI)` block formatter

**Increment 3 (advanced workflows + integration):**
- `warm_start_from` field + checkpoint copy logic
- `build-dashboard-exp ID= OUT=` target
- Record file template with cross-link conventions documented
- First REVIEW_LOG entry citing experiments by id
- Optional: migrate `corpus_manifest`-using checkpoints' metadata to embed effective-config schema fully (likely already does, audit-only)

MVP cut: **Increment 1**. End-to-end workflow works; Increments 2 and 3 are quality-of-life.

---

## Open questions deferred to implementation

1. **Cohort-specific Makefile target deprecation:** the per-cohort `lda-bq-fit-eval-{cancer,dementia}` targets stay through this design's lifetime. Future deprecation is a separate decision once experiment-tracking has been used in practice for some weeks.
2. **`status: archived` semantics:** treated identically to `done` by the runner; the distinction is curatorial (hide from default browsing, mark as "completed and not part of active analysis"). Tooling for filtering by status (e.g. `make list-pending`) is a future addition if needed.
3. **Cross-experiment search/diff:** comparing fit configs or NPMI tables across experiments is grep-friendly given the markdown layout. A dedicated `make diff-exp A=42 B=43` target could be added later; not in MVP.
4. **What goes into `git_sha` field in summary.md:** simplest is `git rev-parse HEAD` on the cluster at fit start. Stops being meaningful if the cluster pulls between fit start and fit end of a long resume chain; resume sessions could each capture their own sha.

---

## Risks

- **Effective config divergence between summary.md and corpus_manifest:** the runner writes both. They should be byte-identical for the same fit. Tests should cover that. Drift here would invalidate the "checkpoint metadata is the truth" claim.
- **Sanitization regex incomplete:** new drivers may print patient-derived info in formats the regex misses. Mitigation: driver-side strip is the primary defense; wrapper filter is backup; periodic review of `summary.md` files committed to the repo.
- **Long copy/paste through JupyterLab failing for some terminals:** if `cat summary.md` produces output the user can't reliably select-all-copy, fallback is `summary.md` lives on the cluster filesystem and can be downloaded via the JupyterLab file browser. Worth verifying once Increment 1 ships.
- **Sequential id collisions when multiple experiments are authored offline in parallel:** since I author them in chat, single-threaded by construction. If the user manually creates one between sessions, a collision is possible — easy to resolve by renaming on next commit.
- **Increment 2's streaming wrapper interacting badly with spark-submit's stdout buffering:** Python's stdin/stdout buffering is the usual culprit. `-u` or `flush=True` on the driver side, plus `bufsize=1` on subprocess.Popen, should cover it. Worth validating on a fast smoke test before relying on it.

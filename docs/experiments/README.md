# Experiment Log

One dated, numbered Markdown file per experiment: `NNNN-<slug>.md`. An experiment
is a **run with a question** — a fit, a readout, a diagnostic sweep — recorded so
its motivation, configuration, and result survive the session that produced it.
Experiments complement the other `docs/` systems (see
[`../META_process.md`](../META_process.md)):

- **ADRs** (`../decisions/`) record *decisions*; **insights** (`../insights/`)
  record *observations*; **experiments** record *runs* — the concrete thing that
  was executed and what came back. A run that surfaces a general modeling
  phenomenon spins off an insight; a run that motivates an architectural choice
  spins off an ADR. The experiment doc is the primary record they cite.

## What an experiment doc contains

- **Front matter** — the identity and knobs: `id`, `slug`, `status`,
  `model_class`, cohort/disease, and the hyperparameters that differ from recent
  defaults (so a reader can judge whether a result is regime-specific). This is
  what makes two experiments comparable — or, when a knob like the document unit
  changes, explicitly *not* comparable (see insight 0010).
- **Rationale** — the question the run is asking and what a positive/negative
  result would mean.
- **Run log** — dated entries as the experiment progresses: launches, receipts,
  crashes and their forensics, intermediate numbers. Append, don't overwrite;
  a superseded number is struck through or noted, not deleted.
- **Results** — the headline numbers, respecting the egress floor (pooled
  figures and counts only; per-node/per-cell tables stay workspace-internal).

## Status lifecycle

`planned` → `running` → `done` (or `abandoned`, with the reason). A `done`
experiment's numbers are frozen; later work that builds on it cites them rather
than editing them. When an experiment is superseded as a control (e.g. a change
of document unit retires it), say so in its closing entry rather than deleting it
— it stays a historical record.

## Index

Experiments are numbered in creation order; browse `NNNN-*.md` newest-last, or
start from the newest entry to see the active line of work. The current
experiment, and the spec/plan driving it, are cross-linked from
[`../superpowers/`](../superpowers/) and [`../reports/`](../reports/).

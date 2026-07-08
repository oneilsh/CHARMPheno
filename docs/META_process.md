# Keeping an Agentically-Coded Repo Legible

A repo developed with heavy AI-agent assistance has a specific failure mode:
**the cost of agent-assisted velocity is that reasoning evaporates faster than
code accumulates.** An agent can produce a correct-looking change in seconds, but
the "why" behind it — the alternatives weighed, the empirical result that
motivated a default, the review that vetted it — is gone unless something durable
captures it. Six months later the code is still there and the reasoning is not,
and nobody (human or agent) can safely change it.

This document describes four lightweight systems that prevent that. Each makes
one *type* of reasoning durable. None is heavyweight: the whole apparatus is
numbered Markdown files plus, where a system involves actual runs, a small runner
script and a task wrapper. The unifying principle:

> **Every non-obvious choice or observation leaves a durable, numbered artifact
> in `docs/`, so the repo's reasoning is version-controlled alongside its code.**

These artifacts are edited when they need correcting, but never in a way that
erases a past decision or result — supersession is recorded, not overwritten. The
numbering and this history-preserving discipline are what keep them searchable and
citable long after the session that produced them. The systems are deliberately
generic; adopt the ones that fit and rename directories to taste.

---

## 1. Decision tracking — ADRs (`docs/decisions/`)

Architecture Decision Records capture *why* a structural or design choice was
made, in numbered, individually-owned files (`0001-…`, `0002-…`, …). Each follows
a fixed skeleton — **Status / Date / Context / Decision / Alternatives considered
/ Consequences** — so the record preserves not just what was chosen but what was
*rejected and why*. That "alternatives" section is the part that pays off later:
it's the difference between "this is weird, let me rewrite it" and "this is weird
*because* the obvious alternative was tried and broke X." ADRs are referenced by
number from code comments, commit messages, and the review log, so a puzzling bit
of code resolves to a one-click answer.

**Conventions that make it work**

- **No tooling** — the discipline *is* the convention. A new ADR is just the next
  number.
- Significant structural, architectural, or mathematical choices get an ADR
  **before or as they land**, not reconstructed afterward.
- ADRs **are** edited in place for corrections — citation fixes, tone passes,
  fixing a code link, flipping Status from Proposed to Accepted. What is *not*
  done is silently overwriting a past decision so the record no longer shows what
  was decided.
- When a decision is **materially superseded**, the original reasoning is
  preserved rather than deleted: add a dated **Amendment** at the top pointing to
  the superseding ADR, fence off the now-outdated text (e.g. `--- begin/end
  historical decision text ---`) so it stays readable as the historical record,
  and add the new ADR alongside. The old file still explains what past-you
  believed and why; the new file governs.

## 2. Insights tracking (`docs/insights/`)

Where ADRs record *engineering* decisions, insights record *empirical* findings —
the things learned by running the system, not by designing it. In a
modeling/ML project these are the observations from actual fits: a hyperparameter
that traps the optimizer in a bad basin, a metric that turns out to measure
something other than its name, a default that only holds in a particular data
regime. Numbered `0001-…` onward, each carries a **Date / Topic / Status**
(Confirmed, Tentative, Refuted…) header and a short body stating: the finding;
the *mechanism* (with the governing math where there is one — a self-consistency
argument, a collapsed update, whatever explains the "why"); the *implications*
for defaults or design; and — critically — a **setting context** block recording
the regime it was observed in (dataset, sample, hyperparameters). An empirical
claim quoted without its regime is a trap, and the setting block is what stops the
same lesson being re-learned or misapplied.

Because findings evolve, a later insight can revisit and overturn an earlier one
(`0039 revisits 0028`), and the numbering makes that dialogue traceable rather
than leaving two contradictory claims floating loose.

**Conventions that make it work**

- **No tooling** — add a numbered entry whenever a run surfaces something
  non-obvious.
- Always include the **setting context**; and prefer stating the *mechanism*
  over just the observation, so the finding is falsifiable and transferable.

## 3. Experiment tracking (`docs/experiments/` + a runner + task targets)

Experiments are *runs*, not just prose, so this is the one system worth
automating. Each experiment is a Markdown file with **structured frontmatter**
(an id, a slug, a status, and the run's configuration — model/method, dataset,
and key hyperparameters) plus a free-text **Intent** section and a running **Fit
history** log.

A small runner script reads that frontmatter, merges it over shared defaults
(a base config plus per-dataset or per-scenario overlays), dispatches the actual
job, captures sanitized output into the run directory, and triggers evaluation.
A thin task layer (Make, `just`, npm scripts — whatever the repo uses) drives it,
e.g.:

- `next-exp` — pick the lowest-numbered pending experiment and run it.
- `eval-exp [ID=N]` — (re-)run evaluation for a given or most-recent run.
- downstream stages (build artifacts, dashboards, reports) keyed by the same id.

Two properties make this pull its weight. First, the frontmatter is the **single
source of truth** for a run's configuration — there's no "what settings produced
this again?" Second, the **Fit history** section turns each experiment into its
own mini lab-notebook: every failed attempt, the error, the fix, and the commit,
in order. A file that records four failed sessions before a clean run is worth far
more than one that only shows the success.

For environments where experiments must be run by a human (e.g. a secure computing
enclave with git access but no agentic coding tools), this also helps reduce burden;
a simple `git pull && make exp ID=34` is all it takes, and the setup/tracking is 
agent-managed. (The human may need to report back results to the agent in these
settings however.)

**Conventions that make it work**

- Configuration lives in **frontmatter**, not in ad-hoc command lines that vanish.
- The runner **sanitizes captured output** (strip anything sensitive) before it
  lands in a committed run directory.
- Append to **Fit history** as you go; don't tidy away the failures.

## 4. Periodic walkthroughs + a review log (`docs/REVIEW_LOG.md`)

This is the human-in-the-loop system that ties the other three together, and the
one that most directly counteracts the evaporation problem. Periodically, a
maintainer is walked bottom-up through a slice of the codebase — a guided reading,
not a skim — often with an agent doing the exposition and the human interrogating
it. A stateful curriculum (tracked per branch or per subsystem) lets a review span
several sessions without losing its place.

These walkthroughs are where latent bugs actually surface: reading code aloud
with intent catches correctness issues, missing guards, and dead complexity that
no test flagged, and those fixes ship as in-line detours during the review. Each
session is then recorded as a dated section at the **top** of the review log
(newest first): what area was reviewed, what refactors shipped, which pre-existing
issues were caught, and which ADRs/insights changed as a result.

The review log is therefore the **audit trail of sign-off** — durable proof that
a given area of code was actually read and understood by a human — and it
cross-links straight back into the ADRs and insights the review touched, closing
the loop between the four systems.

**Conventions that make it work**

- Reviews are **guided readings with a human in the loop**, not automated lint
  passes — the point is human understanding, not just find-and-fix.
- Log entries are **dated, newest-first, and impersonal** (project-scoped, not
  per-contributor); pedagogical content and personal preferences live elsewhere.
- Fixes found mid-walkthrough are shipped and **noted in the same entry**, so the
  log records both what was reviewed and what it produced.

---

## The through-line

| System       | Type of reasoning | Artifact                     | Automation             |
| ------------ | ----------------- | ---------------------------- | ---------------------- |
| ADRs         | Structural        | `docs/decisions/NNNN-*.md`   | Convention only        |
| Insights     | Empirical         | `docs/insights/NNNN-*.md`    | Convention only        |
| Experiments  | Experimental      | `docs/experiments/NNNN-*.md` | Runner + task targets  |
| Walkthroughs | Review / sign-off | `docs/REVIEW_LOG.md`         | Stateful curriculum    |

Four kinds of reasoning, four durable homes. None of it is heavyweight; all of it
is numbered, history-preserving, and cross-linked. That is what makes an
agent-accelerated repo something a human can still keep up with — not slowing the
agents down, but insisting that every fast change leave a slow, legible trace
behind it.

# Specs & Plans

The design artifacts for a body of work, kept separate from the code they
describe so the *reasoning* is durable (see [`../META_process.md`](../META_process.md)).

- [`specs/`](specs/) — **normative design**. What a body of work IS: definitions,
  requirements (numbered, so they can be cited and their disposition tracked),
  the metrics protocol, and the decisions that are settled vs. open. A spec is
  the contract; it does not say how the code is organized.
- [`plans/`](plans/) — **build plan**. How a spec gets implemented: work packages
  with dependencies, gates, agent assignment where relevant, cache-key impact,
  and sequencing. A plan cites its spec's requirements by number.

## The flow

For a substantial body of work the artifacts chain, each citing the last:

```
audit / scouting (docs/reports/)  →  spec (specs/)  →  plan (plans/)
      →  experiment (docs/experiments/)  →  results report (docs/reports/)
```

Not every change needs the full chain — a one-file fix needs none of it. Reach
for a spec+plan when the work is large enough that the reasoning would otherwise
evaporate: a new experiment's design, a cross-cutting refactor, a change with
cache-key or comparability consequences. Requirements in a spec get **dispositions**
in its plan (built / done / deferred, with the commit or reason), so the two
read together as a live checklist.

Files are dated `YYYY-MM-DD-<slug>.md`. A spec and its plan share the slug where
practical, so they sort together.

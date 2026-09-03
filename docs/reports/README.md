# Reports

Dated analyses that inform decisions but are not themselves experiments:
adversarial audits of a design, scouting notes that size a question before a
fit is spent, results write-ups that interpret an experiment's numbers, and
notes handed between collaborators. Files are `YYYY-MM-DD-<slug>.md` — dated,
not numbered (they are not a citable series the way ADRs, insights, and
experiments are).

Where an experiment log records *what was run*, a report records *what a run or
a design means* — the interpretation, the audit, the recommendation. In the
work flow (see [`../superpowers/`](../superpowers/)), reports typically sit at
the ends: an **audit / scouting** report opens a body of work by mapping the
terrain, and a **results** report closes it by interpreting the numbers.

Respect the egress floor: pooled figures and counts-of-nodes only; per-node or
per-cell tables stay workspace-internal.

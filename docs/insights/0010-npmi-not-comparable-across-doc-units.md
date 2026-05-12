# 0010 — NPMI absolute values are not comparable across doc units
**Date:** 2026-05-12
**Topic:** npmi | doc-units | diagnostics
**Status:** Confirmed

The reference distribution used to estimate P(w_i), P(w_j), P(w_i, w_j)
in NPMI is the *document collection*. When the doc unit changes
(patient-lifetime → patient-year → patient-visit), the joint
co-occurrence rates change too — sometimes dramatically, because the
same set of events gets sliced into many more, smaller documents.

Concretely, two concepts that co-occur in every adult year of a chronic
patient's life have:
- High P(i,j) and high NPMI on patient-year docs (they appear in every
  one of that patient's year-bins).
- High P(i,j) but moderate NPMI on patient-lifetime docs (they appear
  in the same one doc per patient).

So "+0.14 NPMI on patient-year" and "+0.14 NPMI on patient-lifetime"
mean different things. The relative ranking of topics within a single
run is meaningful; the absolute number across runs with different doc
units is not.

**Implications.** Eval driver stamps `corpus_manifest['doc_spec']` and
aborts on doc-unit mismatch at fit→eval (see
[ADR 0018](../decisions/0018-document-unit-abstraction.md)). Human
reporting needs the same discipline: never compare NPMI numbers between
runs without naming the doc unit. The eval-output banner surfaces the
doc unit; respect it.

A safer cross-run metric to develop in the future: NPMI computed against
a **fixed reference corpus** (e.g. always evaluate NPMI on
patient-lifetime co-occurrences, regardless of which doc unit was fit
against). Not implemented; flagged as a follow-on.

**Setting context.** This is a property of the metric, not a finding
from a specific run, but became operationally relevant when comparing
patient-lifetime vs patient-year NPMI distributions during the
ADR 0018 work.

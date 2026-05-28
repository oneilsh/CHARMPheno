# 0021 — Producer-side vocab trim for dashboard bundles (egress-budget rationale)

**Status:** Accepted
**Date:** 2026-05-28
**Related:** ADR 0020 (dashboard static hosting and artifact contract);
insight 0011 (min-doc-length tradeoffs); insight 0023 (producer/consumer
unit mismatches invisible until small scale)

## Context

The dashboard bundle's `vocab.json` ships a top-N subset of the full
training vocabulary (default `--vocab-top-n 5000`), ranked by corpus
marginal frequency. The full vocabulary on cohort fits is typically
10k–15k unique concept_ids; the trimmed vocab is what reaches the
consumer.

The trim happens on the *producer* side — inside
`charmpheno/charmpheno/export/dashboard.py`'s
`write_model_and_vocab_bundles`, called from `analysis/cloud/build_dashboard_cloud.py`
and `analysis/local/build_dashboard.py`. The dashboard frontend receives
only the trimmed vocab; it has no opportunity to filter further.

A purely architectural read would prefer the trim downstream — let the
producer ship the full vocab and let the dashboard pick which to display.
That would keep the producer schema model-agnostic, and would let the UI
adjust the display vocab without re-fitting or re-exporting.

That architectural preference is overridden here by a hard operational
constraint: **bundle egress at the data-environment perimeter.** Producer-side
trim stays.

## Decision

The dashboard vocab is trimmed on the producer side before the bundle is
written. The trimmed vocab is the bundle's `vocab.json`; the full
vocabulary lives only in the checkpoint metadata
(`metadata["vocab"]`) and the corpus statistics
(`corpus_stats.json` carries `v_full`). The consumer sees the trim as
ground truth.

The trim parameter (`--vocab-top-n`, default 5000) is exposed on the
build drivers, so re-export to widen or narrow the trim is a one-line
change. Re-export is the path to a different vocab; the dashboard does
not adjust this at display time.

## Why egress is the load-bearing constraint

CHARMPheno fits run on the AllOfUs Researcher Workbench (and analogous
secure-compute environments at federated sites). Egress from these
environments is subject to automated detection and review of "data
movement" events; large outbound transfers can trigger compliance review,
add latency to release cycles, and in some configurations can be blocked.

Current bundle sizes at `--vocab-top-n 5000` sit at approximately 500 KB
zipped per cohort. Shipping the full untrimmed vocab would inflate this
3–5× (the model.json `beta` matrix and vocab metadata grow linearly with
displayed vocab size). The increase moves bundles into a size regime
where automated egress review is more likely to fire.

The trim is not just a display optimization. It is the mechanism by which
the bundle's egress size stays predictably under the operational
threshold of the producer environment. This is a deployment fact, not an
architecture fact.

## Alternatives considered

1. **Ship the full vocab, trim in the dashboard.** Rejected: 3–5× bundle
   size growth puts every re-export at risk of triggering an egress
   review. Even if the displayed vocab is identical, the wire bytes are
   not.

2. **Two-tier export — small bundle + on-demand "full" download.** Adds
   a backend the static-hosting design (ADR 0020) explicitly rejected.
   Doesn't actually solve the egress problem either — the on-demand
   download would still have to leave the secure environment.

3. **Compress more aggressively (better encoding, sparser model.json).**
   Real but small wins. Doesn't change the order-of-magnitude
   relationship between trimmed and full bundles. Worth doing on its
   own merits, not as a substitute for the trim.

4. **Move the trim ceiling onto a per-deployment config.** Possibly
   warranted if sites with looser egress regimes want fuller bundles.
   Not done here; deferred until a concrete second deployment needs it.

## Consequences

- The `--vocab-top-n` parameter is operationally load-bearing on the
  cloud drivers. Changing it requires re-fitting and re-exporting; the
  cost is on the producer, not the consumer.
- The dashboard contract (per ADR 0020) is "the bundle ships exactly
  what the dashboard displays." This ADR is the *reason* — not just a
  design preference.
- The trim is downstream of the small-cell suppression introduced in
  the `min_df` / `min_patient_count` arc (privacy thresholds at vocab
  *construction*) and the per-topic top-N display ranking in
  `select_top_n_by_marginal`. These three filters compose: privacy at
  fit-vocab-build, top-N-by-marginal at display-rank, then the
  cross-topic union becomes the bundle's `vocab.json`.
- Reviewers of the codebase reading the build driver should not
  refactor the trim downstream without checking against this ADR.
  Producer-side trim looks like a smell in isolation; it isn't.

## Open follow-ups (not in scope for this ADR)

- The actual egress thresholds for AllOfUs (and other deployment
  environments) should be pulled from authoritative documentation
  rather than left at empirical "haven't been flagged yet" intuition.
- A per-deployment trim ceiling (alternative 4 above) becomes a real
  question once a second deployment site has different constraints.

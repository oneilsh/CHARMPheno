# 0088 — On the native-Mondo path the vocabulary leakage strip is a no-op (Mondo node ids looked up in OMOP vocabularies), so every prevalent readout since 0110 has the disease's own pre-index codes in the features and blends tracking with prediction

**Date:** 2026-09-09
**Topic:** leakage, evaluation, native-mondo, strip_mode, case-finding, exp 0110–0120
**Status:** Established by code reading (mechanism below is unambiguous); the confirming count is the off-YARN `inspect_topics --strip-audit` on any 0110+ run (expected: ~0 dims per domain). Not yet run on the cluster.

## Observation

Every experiment from 0110 on sets `strip_mode: both` and `dag_source: mondo_native`. The
assembler's strip (`charmpheno/omop/multi_domain.py:296`) is

```
drop_idxs = {vm[c] for c in node_cids if c in vm}      # node_cids = before_dag.nodes()
```

— the label DAG's node ids looked up in each domain's `{concept_id: idx}` vocab map. On the
anchor Mondo path (`mondo_dag`, exps ≤ 0109) node ids ARE OMOP concept ids, so this drops the
DAG-node codes. On the native path, `mondo_native_dag` keys nodes by the curie's numeric part
(`MONDO:0004995 → 4995`; recorded as the "engine id space deviation" in that module and in
exp 0110's log, with the note "there is no collision with the OMOP concept ids … NO node is
keyed by an OMOP concept id"). That note is correct about identity and misses the consequence:
an OMOP-keyed vocab map contains a Mondo numeric only by coincidence, so the strip set is
(near-)empty and **`strip_mode: both` strips nothing**. No other strip exists on that path
(grep: `strip_test_features` / `drop_idxs` appear only in the two assemblers).

What stays in the features is therefore governed only by the window: `window_mode: lookback`
puts features strictly pre-index, but under `index_mode: population` (a random event-anchored
index with no disease semantics) a chronic patient's pre-index window routinely carries codes
for the very node labeled in the forward label window. The lookback spec already named this
caveat ("the strip is only a partial backstop"); the backstop is in fact absent.

## Why it matters

- **Prevalent AUC on 0110+ is tracking-contaminated with no vocabulary backstop.** Insight 0075
  measured the tracking share on 0110 at ~0.067 macro (prevalent 0.741 vs incident 0.674) —
  that measurement was already taken under a no-op strip, so it stands; but 0113–0120 report
  *prevalent-only* readouts (`preindex_closure: false`, no incident arm), and their 0.75–0.78
  macros carry the full tracking share. **The incident cohort (`c ∉ R_d`) is the only leakage
  control actually in force, and it was switched off for the whole CV-branch arc.**
- **It re-reads the "discriminability wall" (0082–0087).** The fed deep topics 0113 saw
  "dominated by own-label variant codes" are dominated by the node's own codes, exact and
  variant alike — nothing removed them. The nodes whose AUC is high are the chronic, repeatedly
  coded ones (own codes precede the index); the credited rare nodes (0087: median 0.734 <
  0.767) are those whose own codes rarely precede a random index. That split is a tracking
  gradient, not a discriminability gradient. Levers that reshape topic *words* (spectral init,
  profile-eta, the co-fit head) could not move a number that is mostly "did the chart already
  say it."
- **It does NOT rescue profile-eta.** The HPO profile concepts are not stripped either, so
  0084/0085's "the prior cannot move θ" is not a strip artefact: those concepts are simply
  rare in positives' pre-index windows (stage-2 probe: 53% of a node's positives carry none of
  the surviving profile tokens) and shared across profiles. The `--collinearity` report
  quantifies that reading (profile Jaccard, boosted-topic cosine, decoder own-share).
- **`strip_mode` in front matter has been a false statement of the design since 0110.** Treat
  it as such when reading any 0110+ experiment doc.

## What to do

1. Run the audit on 0120 (off-YARN, seconds) to record the count:
   `make -C analysis/cloud inspect-topics ID=120 CREDITED=1 INSPECT_ARGS="--strip-audit"`.
2. Read 0113–0120 as prevalent/tracking numbers; the honest case-finding read needs the
   incident arm (`preindex_closure: true` + the R_d witness), which is exactly what the
   episode program's incident evaluation provides. Any "ceiling classifier" test (LR on codes)
   must run on the incident cohort or it measures the same tracking.
3. Fixing the strip itself means an OMOP strip set on the native path — a driver-seam change
   (map node → its `coded_cids`/mapped standard concepts, hand the assembler an OMOP-keyed
   `node_cids` for the strip only), NOT an edit to the hashed assembler. Whether to fix it at
   all is a design question: with incident evaluation the strip is redundant for cases by
   construction, and stripping the whole branch's condition vocabulary from a gated topic
   model removes legitimate comorbidity signal (0079's "aggressive global strip" worry).

## Reproduce

```bash
make -C analysis/cloud inspect-topics ID=120 CREDITED=1 INSPECT_ARGS="--strip-audit"
make -C analysis/cloud inspect-topics ID=120 CREDITED=1 INSPECT_ARGS="--collinearity"
```

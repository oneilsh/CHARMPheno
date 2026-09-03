# Exp 0111 — episode-anchored index sampling · spec

**Date:** 2026-09-02 · **Status:** draft, awaiting probe numbers + review
**Lineage:** audit §E5 (`docs/reports/2026-09-01-temporal-eval-program-audit.md:137-191`,
verdict CONFLICT — highest risk, three walls) → incident-program spec §E5
(`specs/2026-09-01-incident-episode-eval-program.md:453-569`, the normative
R5.1–R5.15 catalogue this spec inherits) → 0111 scouting
(`docs/reports/2026-09-02-0111-scouting-window-depth.md`, which refuted the
window alternative and quantified the bottleneck).

This spec DEFINES the experiment and FIXES its decisions; the 0111 plan doc
(to be written after the probe numbers land) sequences the build.

---

## 1. Why this experiment, in one paragraph

The 0110 incident readout dropped **923 of the scoreable nodes as
incident-starved** (<20 incident positives): a population-random index catches
only the small slice of each disease's cases whose onset happens to fall in the
one sampled year. The scout established that the two cheap alternatives are a
negative and a characterization — widening the label window LOWERS shared-set
AUC (0.6003 → 0.5857), and depth only describes where the signal sits. Episode
anchoring attacks the quantified bottleneck directly: put a document's index
just before each moment something NEW enters a person's record, so every case
is captured at presentation and the thin incident-positive counts multiply.
This deliberately reverses the recorded stance against index fan-out
(`cohorts.py:723-728`): **the goal here is capture, not representativeness**
(R5.14) — representativeness is what the random control arm is for.

## 2. Definitions

Definitions D1–D7 of the incident-program spec (eligibility `c ∉ R_d`, the
incident label as first-attestation-of-closure in the window, P-strata never
gates, dual prevalent/incident naming) carry over unchanged and are not
restated. New definitions:

- **D8 — Episode.** A gap-and-islands cluster of a person's FIRST-ATTESTATION
  dates: the dates on which some label node enters their record for the first
  time ever (the E4 sidecar's frame). Two new-diagnosis dates ≤ `gap` days
  apart belong to one episode; a date more than `gap` days after the previous
  one starts the next (inclusive boundary: exactly `gap` apart = same episode).
  First attestations, not raw condition rows: a chronic patient refilling one
  diagnosis for a decade is one episode, not sixty. `gap` is a spec parameter
  fixed by the probe (§4); candidates 60 and 90 days.
- **D9 — Episode index.** `index = episode_start − 1 day`. The label window is
  half-open `[index, index + W)` and the lookback is `[index − L, index)`, so
  the episode's own first codes land inside the label window and outside the
  features — the model stands immediately before the presentation it must
  predict. `W` stays 365 (the scout closed the window question); `L` stays the
  corpus default.
- **D10 — Episode document.** One document per `(person, index)` surviving the
  assembler's own observation gates (`_window_observed_cohort`, both clauses:
  `index ≥ op_start + 365` and `index + W ≤ op_end`), subject to the per-person
  cap (D11). `doc_id = cohort:person:index` — the index APPENDED, because
  prefix parsers survive appends only (R5.1).
- **D11 — Per-person cap.** At most `cap` episode documents per person, chosen
  by a DETERMINISTIC salted-hash sample over the person's surviving episodes —
  the `min hash(person_id, event_date, salt)` idiom the population index
  already uses (resume-stable, never `F.rand()`). Uniform over surviving
  episodes, not "earliest" or "latest": either extreme would correlate the kept
  documents with record position and quietly re-bias what the gate already
  biases. `cap` is fixed by the probe (§4).
- **D12 — Arms.** 0111 is a TWO-ARM experiment on one corpus identity:
  the **episode arm** (D8–D11) and a **random arm** — the existing
  population-random index, refit under 0111's identical configuration. The
  random arm is the control (R5.13); 0104/0109/0110 numbers are never
  comparison targets for either arm (insight 0010: doc-unit-sensitive numbers
  do not compare across doc units). What makes the two arms comparable at all
  is that C and the label DAG are person-level and index-independent (audit
  seam 10) — both arms share the node set by construction.

## 3. Requirements

### 3.1 Inherited catalogue, with dispositions

| Req | What | Disposition |
|---|---|---|
| R5.3 | doc-spec cache-key hole | **DONE** (`25acc62`, WP7): `doc_spec_identity()` travels in `key_extra`. The stated precondition for any doc-unit work is met. |
| R5.9 | episode multiplier probe | **DONE** (`diag_episode_probe.py`, `4e9bb33`; results `2026-09-02-0111-episode-probe-results.md`). Gated ×8.55 (gap 90); cap 3 → ×2.66, cap 5 → ×3.98. Decisions in §4 resolved. |
| R5.10 | prior-obs-gate kill rate probe | **DONE** (same run). Overall kill 19.8%; **first-episode kill 66.2% vs later 14.1%** — the anti-correlation, measured. Forces the incidence-definition fork (§4). |
| R5.1 | `EpisodeDocSpec` | BUILD. Sanctioned extension point (ADR 0018); the "one class + manifest round-trip" claim is false under the ADR-0046 readout stack — R5.4–R5.7 are its real price. |
| R5.2 | `index_date` passthrough in `cohorts.py` | BUILD — **the one-blast commit** (§3.2). |
| R5.4 | int64 doc-key synthesis | BUILD. `_lean_eval_kernel` and its driver twin hard-code int64 ids; string doc_ids raise. Synthesize `doc_key = person_id * 2^k + episode_no` or equivalent, one definition shared by both sides. Hard break otherwise. |
| R5.5 | A/B alignment dict fix | BUILD, **before the smoke**: the A/B gate is the correctness instrument for everything after it, and under multi-doc its person-keyed dict silently overwrites — wrongness inside the gate itself. Key on the doc key; make `sample_frac` doc-keyed too (seam 6). |
| R5.6 | person-keyed calibration split, driver path | BUILD. The distributed twin is already person-keyed; make the driver path match (exp 0079 run-2 failure otherwise). |
| R5.7 | person-level detection dedup | BUILD. `detection_readout` silently becomes episode-weighted; detection claims dedup to persons. |
| R5.8 | wire the distributed eval path | BUILD, **conditional trigger in §4** (multiplier ≥ 3 ⇒ mandatory before any episode fit). Rider: the calibration fit-small / broadcast / apply-distributed change shares this wiring — `calibrate_per_node`'s driver-side float64 copy is the same wall. |
| R5.11 | insight-0009 catch-all risk | SMOKE CHECK. Overlapping per-person docs (~80% shared events at L=1825) are doc-multiplication for chronic patients; the smoke runs coherence + topic-usage diagnostics before any record run, and `n_bg` absorbing it is the open question, not an assumption. |
| R5.12 | ESS caveat | PROTOCOL. Every CI and every `min_count ≥ 20` threshold assumes independent rows; overlapping documents from one person violate that. Every reported interval carries the caveat; nothing publishes a nominal CI silently. |
| R5.13 | within-0111 control | PROTOCOL — D12. |
| R5.14 | reversed prose stance | RIDES the one-blast commit (§3.2): amend `cohorts.py:723-728` so the code stops reading as forbidding what the driver now does on purpose. |
| R5.15 | machinery reuse | DESIGN FACT. `_random_event_windows` cannot be reused (it IS the one-per-person sampler); `_window_observed_cohort` can (arbitrary `(person, index)` frame, N rows preserved; rejoin dropped columns as `_mdd_antidepressant_index` does). The probe already exercises exactly this reuse. |

### 3.2 New requirements

- **R7.1 — The one-blast cache-drop commit.** *(Scope WIDENED 2026-09-03 by
  the plan doc, after code verification.)* R5.2 edits `cohorts.py`, which
  moves `cohort_defs_version()` and invalidates EVERY cache key in the repo
  plus all four pinned tripwire hashes — and the same blast must also carry
  **`multi_domain.py`**, which hard-codes the index-builder selection
  (`:382-394`, no external-frame option) and the doc spec (`:408`): WP7/R5.3
  closed the cache-KEY hole, not the injection hole. Both files in ONE
  commit, containing exactly: (a) the `index_date` passthrough in the
  windowing helpers; (b) `index_df=` / `index_mode="external"` and
  `doc_spec=` injection params on the assembler — seams only, on the
  `attested_provider` precedent, so ALL episode logic stays driver-owned and
  this is the last planned edit to either file; (c) the WP0 residue — the
  stale strip-claim comment at `case_finding_assembly.py:394-396`; (d) the
  R5.14 prose amendment at `cohorts.py:723-728`; (e) re-pinning the four
  tripwire hashes in `tests/scripts/test_case_finding_cache_mondo.py`, with
  the commit message naming the deliberate drop per the tripwire's own
  instruction. The standing "never edit" rule on these files is suspended for
  this commit alone and reinstated immediately after. Nothing else rides:
  behavior changes beyond the seams are out of scope for the blast, and
  existing callers passing no new arguments get byte-equivalent behavior.
- **R7.2 — Probe-before-plan.** The 0111 plan doc is not written until
  `diag-episode-probe` numbers exist. The multiplier decides R5.8's path
  (build order changes materially either way) and the kill decomposition
  decides how the capture claim is worded; a plan written first would encode
  guesses the tool exists to remove.
- **R7.3 — Node-yield acceptance anchor.** The probe's frontier-level node
  yield (nodes with ≥ 20 gated episodes) is the pre-registered acceptance
  anchor for the whole experiment: 0111's episode arm succeeds in its stated
  purpose iff the incident census of the episode corpus scores materially more
  nodes than the 0110 incident readout's 1,798 (923 dropped). The probe's
  number is a lower bound (frontier grain; the closure fold only adds); the
  corpus census after the build is the real measurement, on the census tool's
  existing GO/NO-GO discipline.
- **R7.4 — Vocabulary discipline across arms.** `min_df` is a DOCUMENT count
  (`topic_prep.py:222-224`) and the arms have different document counts, so
  each arm derives its own vocabulary; `min_patient_count` (insight 0025) is
  the person-level guard both share. Arm comparisons happen on the shared
  scoreable node set (R2.2 discipline), macro over identical node lists, and
  claims name the arm and node set exactly as D7 naming already requires.
- **R7.5 — `min_doc_length` monitoring.** `min_doc_length: 10` drops episode
  docs non-uniformly toward the incident end (the shortest lookbacks are the
  earliest episodes — seam 9). The smoke reports the drop rate by episode
  ordinal; if the drop concentrates on first episodes, that is R5.10's bias
  wearing a second mask and gets recorded next to it.
- **R7.6 — Egress.** Unchanged floor: per-node tables stay workspace-internal;
  banners and committed docs carry pooled figures and counts-of-nodes only.

## 4. Pre-measurements and the decision rules they feed

**RESOLVED 2026-09-02** — probe run, `2026-09-02-0111-episode-probe-results.md`.
Locked: **gap = 90d**, **cap = 3** (cap 5 fallback if the corpus census erodes
node yield below ~1,800), **R5.8 distributed eval + distributed calibration
apply wired as a plan precondition** (×2.66 sits at the no-headroom boundary and
the calibration driver copy is the wall regardless of cap). Node yield PASSED
R7.3 by a wide margin (2,583 nodes ≥20 gated episodes vs 0110's ~1,791
scoreable / 923 starved). **The incidence fork is SET (Shawn, 2026-09-03):**
the 365-day prior-obs gate STAYS for the primary arm — the corpus is "incident
among the year-plus-observed," and the 66.2% first-episode kill is reported as
the measured conditional-capture caveat, not engineered away. A relaxed-gate
**sensitivity** (probe re-run at reduced prior-obs, no fit) is a named plan
item quantifying what the gate costs. All §4 inputs to the plan are resolved.

The rules that produced those decisions, `make diag-episode-probe ID=110`
(requires the 0110 sidecar witness; no fit, no cache impact), per gap ∈ {60, 90}:

| measurement | decides | rule |
|---|---|---|
| gated per-person multiplier (and capped variants) | R5.8 build order | capped multiplier ≥ 3 ⇒ distributed eval path wired and A/B-verified BEFORE any episode fit; < 3 ⇒ driver path may carry the smoke, R5.8 still lands before the record run (no headroom without it) |
| per-person count histogram (p50/p90/p99/max, bands) | D11's `cap` | smallest cap ∈ {3, 5} keeping ≥ ~90% of gated episodes; if p99 is small enough that no cap binds, cap is recorded as inert and kept anyway as a guard |
| raw vs gated multiplier at 60 vs 90 | D8's `gap` | if the two gaps agree within ~10% on gated episodes, take 90 (fewer split workups, less doc overlap); otherwise default 60 and record the sensitivity |
| kill decomposition (both / prior-only / follow-up-only) + first-vs-later kill | the capture claim's wording | recorded verbatim in the plan and the eventual run log; "100% incident capture" is already superseded (R5.10) — the probe replaces it with the measured conditional-capture rate |
| new-nodes-per-episode | expected incident-positive yield per doc | context for the census expectation; no gate |
| node yield ≥ 20 / ≥ 100 | R7.3's anchor | pre-registered before the corpus exists |

## 5. What 0111 is not

- **Not a window change.** W stays 365; the scout refuted widening (falling
  shared-set AUC; the AP rise is a prevalence artifact). Longer-horizon
  labeling remains an eval-side reporting axis via the sidecar, free.
- **Not survival modeling.** Time-to-event labeling stays the named long-term
  endgame, out of scope here.
- **Not comparable to 0104/0109/0110.** Insight 0010, again: any number that
  touches the doc unit compares only within 0111's own arms.
- **Not a representativeness design.** The reversal of `cohorts.py:723-728`'s
  stance is deliberate and one-directional; the random arm carries the
  representativeness reading.

## 6. Sequencing sketch (the plan doc owns the real one)

probes (§4) → 0111 plan → R5.4/R5.5/R5.6/R5.7 seam fixes (+ R5.8 per its
trigger, with the calibration rider) → R7.1 one-blast commit → R5.1
`EpisodeDocSpec` → smoke on both arms (A/B gate, R5.11 coherence + topic
usage, R7.5 drop-rate) → episode-corpus incident census (GO/NO-GO vs R7.3)
→ record runs → incident/conversion analyses on the shipped E1–E4 tooling.

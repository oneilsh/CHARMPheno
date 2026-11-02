# Mondo usage dashboard — mapping provenance (step colors, paths, named-vs-rolled-up split) — design

Date: 2026-11-02 · Status: approved; Part A (driver) built — exp 0120 · Branch: claude/mondo-usage-dashboard

## Intent

The dashboard ([`mondo-usage-dashboard/index.html`](../../../mondo-usage-dashboard/index.html),
payloads from [`analysis/cloud/mondo_usage_cloud.py`](../../../analysis/cloud/mondo_usage_cloud.py),
`source_climb` count space, `--with-hpo`) attributes each EHR condition to one best-home term
by a 4-step ladder (first hit wins):

1. **same-as** — the source code (e.g. ICD) is a Mondo term's `same_as` xref (driver tier `source_exact`)
2. **via SNOMED** — the source's `Maps to` standard SNOMED concept is a mapped Mondo term (`standard_exact`)
3. **HPO** — that standard concept is an HPO xref (`hpo_exact`; the HPO axis)
4. **climbed** — the standard concept is rolled up SNOMED `concept_ancestor` to the nearest mapped Mondo term (`climbed`)

The purpose of this work is **mapping-quality auditing**: for any term, see how much of its count
comes from codes that genuinely *name* it (steps 1–3) versus codes *rolled up* to it from
something more specific (step 4), and for the rolled-up codes see the SNOMED path they climbed —
which shows where a direct Mondo mapping is most needed.

## Prerequisite: Phase 0 — fix the existing upper-bound leak (user decision, 2026-11-02)

The builder check found, and a measurement on the committed payloads confirmed, that **today's
published payloads already let masked ≤20 codes be capped from above**. On a collision-free
term (every m = 1, and `collision == false` is published), `frac ≥ Σ_codes |persons(code)|`,
so `frac − Σ_{other codes} band_lo` caps a ≤20 code. Committed `mondo_usage.json`: 128 of
1,051 such terms cap a ≤20 code below 20 (tightest caps 1, 1, 2 — i.e. exact disclosure);
`hpo_usage.json`: 43 of 418. `_forced_lower` / `assert_diff_safe` check lower bounds only.
A weaker selection-effect leak also exists: whether a total is banded depends on its exact
value, so the banded display can prove a ≤20 code is ≥ 2.

The user chose **fix first, then feature**, plus a git-history scrub of the leaky payloads.
Phase 0 (its own plan, shipped and re-exported before the feature work):

- Disclosure safety becomes **two-sided**: any published combination that narrows a ≤20 cell
  to a strict sub-interval of [1, 20] — from above or below — is unsafe. Add an upper-bound
  analysis (`frac` and, where applicable, the total against per-code band lower bounds) and
  coarsen the offending figure (`frac` → standard band / withheld) when it bites.
- Remove the selection effect: the banding decision for a total must not be a function of the
  exact value in a way the display reveals (e.g. decide on the band containing the total, not
  the exact total).
- `assert_diff_safe` becomes a two-sided tripwire; tests include the measured patterns.
- This **does** change the "All" view for affected terms. The anti-goal below about "All"
  applies relative to the Phase-0-fixed baseline.
- The same cluster run can serve Phase 0 and this feature if both are built before running;
  but the public payload must not wait on the feature (history scrub happens independently).

## Goals

1. **Step coloring (side panel).** Every source code in a term's "Source codes" list is
   visually tagged with its step (1–4): four distinct colors, **plus** a glyph so it survives
   color-blindness (the panel already uses ≡ / = / ↑ — extend to four). Legend shown.
2. **Path on hover.** Hovering a source code shows its real mapping path with actual concepts:
   - step 1: `ICD10CM E27.40 → MONDO:0000004 <label>`
   - step 2: `ICD10CM E27.49 → SNOMED 386584007 → MONDO:… <label>`
   - step 3: `ICD10CM R51 → SNOMED 25064002 → HP:0002315 <label>`
   - step 4: `ICD10CM E27.3 → SNOMED a → SNOMED b → SNOMED c → MONDO:… <label>` — the actual
     intermediate SNOMED parents climbed through, ending at the SNOMED concept that maps to
     the Mondo term.
   The existing NIH Clinical Tables name lookup for ICD/HCPCS on hover is kept and merged into
   the same tooltip.
3. **Whole-graph named / rolled-up toggle.** A global three-way toggle — **All** (today) /
   **Exclude climbed** / **Climbed only** — that recounts every term and drives node sizes,
   colors, the hide-unused filter, the tree/list views and the rolled-up subtree sizes, on the
   Mondo axis. The side-panel code list and its band chips filter to match the mode.

## Non-goals

- A ranked "where mappings are most needed" view (terms by rolled-up share; most common climb
  chains). Explicitly deferred by the user — a natural follow-up once the data is in the payload.
- HPO-side climbing or a source-ICD→HPO same-as rung (the existing v2 items of the 0109 design).
- Shipping SNOMED (or any licensed vocabulary) concept **names**. See Constraints.

## Anti-goals (would be a failure even if it works)

- Any new number that, alone or by addition/subtraction with other published numbers for the
  same term, reveals a patient count in 1..20.
- Changing the "All" view beyond the Phase 0 fix (counts, banding, sizes). The new numbers are
  additive to the Phase-0-fixed baseline.
- Editing a source-hashed module (`analysis/cloud/{mondo_dag,mondo_to_omop_mapping,…}.py`,
  see AGENTS.md). `mondo_usage_cloud.py` is driver-owned and free to edit.

## Constraints

- **Egress floor (All of Us):** any patient cell in 1..20 is not disclosable. Per-code exact
  counts never leave the cluster (only bands). See the existing machinery,
  [`mondo_usage_cloud.py:490-592`](../../../analysis/cloud/mondo_usage_cloud.py#L490-L592)
  (`safe_total_band`, `complementary_suppress`, `assert_diff_safe`).
- **Identity only, no names:** the export carries vocab + code identities, never concept names
  (egress + SNOMED licensing; see the comment at
  [`mondo_usage_cloud.py:441-446`](../../../analysis/cloud/mondo_usage_cloud.py#L441-L446)).
  So the hover path shows SNOMED **codes** (linked to Athena), ICD/HCPCS names resolved
  client-side from NIH as today, and Mondo/HPO labels (already in the payload).
- Payload size: `mondo_usage.json` is ~12 MB; path data should be compact (shared concept
  table, integer ids), target well under +3 MB.
- Old payloads must still load: if the new fields are absent, the panel falls back to today's
  ≡ / = / ↑ inference and the toggle is hidden.

## Decisions (made with the user)

### D1. Step is recorded by the driver, not inferred

The driver already knows the tier per attribution row (`via` in `t1/t2/t_hpo/t3`) but
collapses it to `exact`/`climbed` at
[`mondo_usage_cloud.py:1199`](../../../analysis/cloud/mondo_usage_cloud.py#L1199). Keep the
full tier in the catalog entries (`"step": 1|2|3|4`). When one code reaches the same term by
more than one step, the lowest step wins (consistent with today's "exact wins"). The
client-side same-as inference is retained only as the legacy fallback.

**A code's step is decided per (code, term): lowest step wins, everywhere.** *(User decision,
2026-11-02.)* One ICD code with several `Maps to` targets can reach a term by step 2 on some
rows and step 4 on others (step 1 is decided by the source code alone, so it never mixes with
2/4 for one code). Such a code is listed **once**, as its lowest step, and the climb version is
not counted separately: patients reaching the term through that code's climb rows count as
**named** for the split. So the split is defined over (term, code) steps, not rows:
*named* = patients with ≥1 attribution through a code whose (code, term) step is 1–3;
*rolled-up only* = the rest (every attribution through a code whose step for this term is 4).
Each part is then exactly the union of its own codes' patients, so the per-code bands (computed
per (term, code), as today) cover each part exactly, and D4 rule 5 applies without slack. The
total, catalog and `source_bands` are unchanged from today. Such codes' hover shows their
direct path.

### D2. Paths use real concepts and come from the cluster

Each catalogued source code carries the standard concept it went through and, for step 4, one
representative shortest SNOMED climb chain from that standard concept to the mapped ancestor.
Where several equally short chains exist, one is chosen deterministically and the entry is
flagged so the hover can say "one of several equally short paths".

### D3. The split partitions the total ("named" + "rolled-up only")

For each term (Mondo axis) publish two parts that **add up to the existing total**:

- **named** = distinct patients with ≥1 step-1/2 code attributed to the term
- **rolled-up only** = distinct patients attributed to the term **only** via step-4 codes
  (= total − named; the patients who would vanish from the term if climbing were off)

"Exclude climbed" shows *named*; "Climbed only" shows *rolled-up only*. A patient with both
kinds of code counts as *named*. The user confirmed this framing.

**No frac at all** *(user decision after the second review, 2026-11-02)*. Any published
frac leaks: its upper end caps ≤20 codes, its lower end forces them up, and `frac − N`
bounds the patients holding 2+ of a term's codes. Making it safe withheld it on almost every
multi-code term, and the user prefers informative roll-ups anyway: the rolled-up view
becomes the **sum of the published per-term counts** (patient-term diagnoses — a patient
counts once per term they touch, so multi-mapping and comorbidity both count, by design),
computed client-side with zero new disclosure; uncertainty comes only from masked/banded
terms. To keep that informative, banded totals use a finer **1-2-5 ladder** (21–50, 51–100,
101–200, …) under the same selection-free rule. A true distinct-patients-per-subtree count
is a possible follow-up needing its own disclosure design (a subtree total beside exact child
totals is a new differencing channel).

**Governing rule for all disclosure decisions** *(planning revision)*: every exact / banded /
withheld choice is a function of already-published information (bands, identities, the displayed
total), never of a hidden exact value; for the split, eligibility is checked over every
hypothesis consistent with the public bands. The plan's Task A3 is normative for the split
procedure and supersedes rules 2–6 of D4 below where they differ.

HPO axis: every attribution is step 3, so *named* = total and *rolled-up only* = 0 by
construction; the toggle leaves HPO unchanged under "Exclude climbed" and empty under "Climbed
only".

### D4. Disclosure rule for the split (the user approved this direction)

Two parts that sum to a published total leave exactly one differencing channel
(part = total − other part). Rule, per term, applied after today's `complementary_suppress`:

1. **Total already withheld/banded** (`used_small`, or `total_banded`): both parts are
   withheld (displayed "—"/hidden in split modes, treated as the small-cell placeholder in
   sizing). Never publish a part more precisely than its total.
2. **Both parts 0 or > 20**: publish both exactly.
3. **Total ≤ 40 and both parts ≥ 1**: withhold both, regardless of which part is small.
   *(Builder-check fix; my call.)* The earlier draft branched on "one part small" vs "both
   small", and which branch fired was itself visible: e.g. total 22 shown as `≤20` + `[2,21]`
   proves named > 20, so named = 21 and rolled-up = 1 exactly. Withholding whenever
   total ≤ 40 reveals only that each part is in `[1, total − 1]`.
4. **Total ≥ 41 and one part in 1..20** (both small is impossible here): publish that part as
   `≤20` and the other as the range `[total − 20, total − 1]` — equal information to the
   suppressed cell, no narrowing.
5. **Per-code boxing, per part:** run the existing `_forced_lower` check on each part using
   only that part's catalog entries' bands (see D1: entries are per (code, part), so the
   part's entries' person sets are a superset cover of the part) and the part's published
   `[lo, hi]`. If a ≤20 entry would be boxed (≥ 2), widen that part using the same
   standard-band / ceiling fallback as `safe_total_band`, and widen the other part to the
   complementary range.
6. **Frac split follows the person split.** *(Builder-check fix; my call.)* Exact fracs can
   undo person suppression: on a collision-free term (`collision == false` is published, so
   every m = 1) `frac_named ≥ named`, so exact `frac_named = 97` against a person split of
   `[85,104]` / `≤20` boxes rolled-up into `[8,20]`. So `frac_named` / `frac_climbed` are
   published exactly **only** when the person split was published exactly under rule 2 **and**
   each frac part (rounded) is 0 or > 20; in every other case both frac parts are withheld and
   the client's roll-up treats that node's split as uncertain over `[0, frac]` (the existing
   "estimate ±" path). `_forced_lower` is **not** applied to fracs — a frac is not a union of
   person sets.
7. **Tripwire:** extend `assert_diff_safe` to also check the split (parts sum consistently
   with the total; no part published exactly in 1..20; rule 3 honored for totals ≤ 40;
   per-part boxing; frac parts exact only where the person split is exact) and fail the export
   if violated.

All new rules are pure functions with unit tests in
[`analysis/cloud/tests/test_mondo_usage.py`](../../../analysis/cloud/tests/test_mondo_usage.py),
including the adversarial cases above (total 22 / 21+1; total 38 / 19+19; total 105 /
97+8 with collision-free frac 97/48; total 105 / 100+5).

### D5. The toggle is global and drives everything the count drives

In "Exclude climbed" / "Climbed only", each node's displayed count, state (reported /
used-small / unused), size, color category, hide-unused visibility, list/tree sorting and the
rolled-up subtree estimate all derive from that mode's part. The rolled-up view's existing
"central estimate ± suppression" machinery handles parts published as ranges. "All" mode is
byte-for-byte today's behavior.

## How (decided parts)

- **Driver:** `mondo_usage_cloud.py` `source_climb` branch + `assemble_axis`. Keep tier in the
  catalog; carry `std_cid` through t2/t_hpo/t3 into the catalog; compute per-term named /
  rolled-up-only person counts and split fracs; reconstruct step-4 chains; apply D4; extend the
  safe summary (log banner + payload `meta.survey`). *(Builder-check fix; my call.)* The
  summary gets **only** counts of terms (how many terms have exact / ranged / withheld splits,
  how many have rolled-up-only > 0) and **cohort-wide distinct-person** counts in the style of
  today's `persons_climbed` (e.g. distinct persons with ≥1 rolled-up-only term attribution),
  suppressed with `suppress_count`. Never a sum of per-term parts — subtracting the exactly
  published terms from such a sum would recover a `≤20` term. Behavior is always on for
  `source_climb` (additive fields), no new flag.
- **Payload (additive fields, names are the builder's call):** per catalog entry `step`,
  `std` (standard concept id; absent for step 1), and for step 4 a `path` (intermediate SNOMED
  concept ids, ordered from the standard concept up to the mapped ancestor) + `path_ambiguous`;
  per node the split counts with display strings and ranges mirroring `count` / `display` /
  `count_range` / `frac` / `frac_display`; a top-level concept table `{id: [vocab, code]}` for
  every concept referenced by `std`/`path`, so entries stay integer-only.
- **Dashboard:** step colors + glyphs + legend in `sourceCatalogRow`; a path tooltip on the
  existing custom-tooltip mechanism (merged with the NIH name lookup); a three-way toggle next
  to the Per-term / Rolled-up control; a single per-mode count accessor that every
  count-consuming site goes through.
- **Run:** a new experiment entry (next free number across branches — **0120** at time of
  writing; verify) re-running 0109's config (`source_climb`, `--with-hpo`), then refresh both
  payloads in `mondo-usage-dashboard/`. Cluster commands carry the AGENTS.md branch preamble
  for this branch.

## Left to the builder

- Exact payload field names and JSON shapes; the step color palette (must work in light and
  dark, distinct from node category colors) and glyphs.
- How to reconstruct the shortest climb chain (e.g. walk SNOMED `Is a` edges / `concept_ancestor`
  rows with `min_levels_of_separation = 1`, constrained so each hop keeps the remaining distance
  to the ancestor consistent with `min_levels_of_separation`), and the deterministic tie-break
  (e.g. lowest concept id).
- A source code whose rows map to more than one standard concept for the same term: carry one
  representative `std`/path (prefer the one with the most patients) or a short list — whichever
  keeps the payload compact; flag it in the hover.
- Whether per-step band histograms ship from the driver (needed so the chips filter correctly
  when the list is capped at 400) — likely yes: `source_bands_by_step`.
- Toggle placement and wording, as long as "All" stays the default.
- Docs: experiment entry, README for the dashboard (`mondo-usage-dashboard/README.md`) and the
  Methods tooltip text updated for the split and its suppression.

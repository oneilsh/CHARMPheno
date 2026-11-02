# Mondo usage — Phase 0 disclosure fix + mapping provenance — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the usage export's small-cell protection two-sided and selection-free (Phase 0), and add per-code mapping step, real SNOMED paths, and a disclosure-safe named / rolled-up-only split of every term's count (feature), so one cluster run yields fixed, feature-complete payloads.

**Architecture:** All disclosure logic stays as pure, unit-tested functions in `analysis/cloud/mondo_usage_cloud.py` (the driver; not source-hashed, free to edit). New Spark work is factored into small DataFrame-in/DataFrame-out helpers with a local-Spark `@slow` test, then wired into `source_climb` / `assemble_axis`. Part A (driver) ships first and is run on the cluster; Part B (dashboard) is built against the downloaded export and previewed locally before anything is published.

**Tech Stack:** Python 3.12, PySpark (driver-side pandas for small collects), pytest; vanilla JS in a single static `index.html`.

**Spec:** [`docs/superpowers/specs/2026-11-02-mondo-usage-mapping-provenance-design.md`](../specs/2026-11-02-mondo-usage-mapping-provenance-design.md)

## Global Constraints

- Floor: any patient cell in 1..20 must not be published, derivable, or **narrowed to a strict sub-interval of [1, 20] from above or below** (`min_cell = 20`, floor inclusive).
- **Selection-free decisions (the governing rule):** every exact / banded / withheld choice is a function of information that is itself published (bands, code identities, the displayed total), never of a hidden exact value. Tests below pin this.
- Identity only: vocab + code, never concept names (egress + SNOMED licensing).
- No UDFs and no new executor-side modules (`--py-files` unchanged); new Spark work uses built-ins and driver-side pandas.
- No edits to source-hashed modules (AGENTS.md list). Only `mondo_usage_cloud.py`, its tests, the experiment doc, the dashboard, and docs change.
- Payload changes are additive except where Phase 0 deliberately coarsens (`count` banding, `frac`).
- Test command: `poetry run pytest analysis/cloud/tests/test_mondo_usage.py -q -p no:randomly` (33 green at start). The `@slow` local-Spark test runs with `-m ""`.

## Review Focus

1. A term whose per-code bands are inconsistent with a hypothesis (e.g. the rolled-up part has one `≤20` code, so it can never be the large part): the split must not over-withhold on impossible hypotheses, and must not leak via a possible one. → Task A3 tests `test_split_*_hypothesis_*`.
2. `frac` on a term that has `≤20` codes: the published range's upper end must always leave every `≤20` code able to reach 20. → Task A2 `test_safe_frac_upper_bound_never_narrows`.
3. A banded total adjacent to the exact/band threshold (e.g. N = 101 with `[21–100, ≤20]`): must band by band, not by exact value. → Task A1 `test_total_banding_is_band_decided`.
4. A code reaching a term via step 2 on some rows and step 4 on others: listed once as step 2, its climb-row patients count as named. → Task A5 `test_catalog_steps_lowest_wins_and_named_counts` (local Spark).
5. A climb with no reconstructable `Is a` chain (vocabulary oddity): payload carries `path: null`, hover falls back to "SNOMED, N levels". → Task A4 `test_chain_missing_returns_none` + Task B1.

---

## Part A — driver (ship, then cluster run)

### Task A1: Selection-free total banding (Phase 0)

**Files:**
- Modify: `analysis/cloud/mondo_usage_cloud.py` — `safe_total_band` (~L535-552), `assert_diff_safe` (~L575-592)
- Test: `analysis/cloud/tests/test_mondo_usage.py`

**Interfaces:**
- Produces: `safe_total_band(n_persons, source_bands, min_cell=20) -> (needs_banding: bool, display: str, (lo, hi))` — same signature; new semantics. `_max_forced_exact(lo: int, hi: int, source_bands, min_cell=20) -> int` (max of `_forced_lower(x, x, …)` over integer x in [lo, hi]; evaluated at `hi` and at every code-band upper edge inside [lo, hi], since the function is piecewise linear with those breakpoints).

- [ ] **Step 1: Write failing tests**

```python
def test_total_banding_is_band_decided():
    sb = [{"band": "21–100", "codes": 1}, {"band": "≤20", "codes": 1}]
    # old rule published 101 exactly (forced==1) but banded 102 -> the band leaked ">=102".
    need, disp, (lo, hi) = m.safe_total_band(101, sb)
    assert need and disp == "101–1k" and (lo, hi) == (101, 1000)
    assert m.safe_total_band(102, sb)[1] == "101–1k"          # same outcome across the band

def test_total_ceiling_fallback_respects_reported_floor():
    # inconsistent-on-purpose histogram so the containing band boxes the code:
    need, disp, (lo, hi) = m.safe_total_band(150, [{"band": "≤20", "codes": 5}])
    assert need and disp == "≤1k" and lo == 21 and hi == 1000  # 'reported' already implies >=21
    assert m._forced_lower(21, 1000, [{"band": "≤20", "codes": 5}]) <= 1

def test_max_forced_exact_finds_interior_breakpoint():
    sb = [{"band": "21–100", "codes": 1}, {"band": "≤20", "codes": 1}]
    assert m._max_forced_exact(21, 1000, sb) == m._forced_lower(1000, 1000, sb)
```
Existing `test_safe_total_band_picks_finest_boring_band` and `test_complementary_suppress_*` must stay green unchanged.

- [ ] **Step 2: Run, confirm the three new tests fail.**
- [ ] **Step 3: Implement.** Exact iff `_max_forced_exact(band_lo, band_hi_or_BIG, bands) < 2` for the total's containing band (`BIG = 10**12` for `>100k`). Else display the containing band if `_forced_lower(max(lo, min_cell+1), hi) <= 1`; else escalate through ceilings `≤100, ≤1k, ≤10k, ≤100k` with range `[min_cell+1, hi]`, taking the first with `_forced_lower(min_cell+1, hi) <= 1`. `assert_diff_safe`: for reported nodes use `lo = max(lo, min_cell+1)`; for exact nodes additionally assert `_max_forced_exact` over the containing band `< 2`.
- [ ] **Step 4: Run the suite; all green.**
- [ ] **Step 5: Commit** `mondo-usage: selection-free total banding + reported-floor-aware fallback (Phase 0)`.

### Task A2: Two-sided `frac` publication (Phase 0)

**Files:** Modify `mondo_usage_cloud.py` — new `safe_frac`, `assemble_payload` frac block (~L837-841, L865-866, root ~L940), `assert_diff_safe`. Test file as above.

**Interfaces:**
- Produces: `safe_frac(frac: float, source_bands: list[dict], min_cell=20, grid=10) -> (value: int | None, display: str, rng: list[int] | None)`. Node fields: `frac` (int, only when exact), `frac_display` (str), `frac_range` (`[lo, hi]` or None). Root: `frac: None, frac_display: "", frac_range: None`.

Rule (decided only from public `source_bands`): `F = round(frac)`. `F == 0` → `(0, "0", [0, 0])`. If the term has **no** `≤20` code: exact when `F > min_cell` → `(F, str(F), [F, F])`, else `(None, "≤20", [1, 20])`. If it has a `≤20` code: `T = Σ band_lo·codes + (min_cell − 1)` (band_lo of `≤20` is 1; this is the smallest frac value at which no `≤20` code can be capped below 20 under m = 1, the worst case), `Tg = ceil(T / grid)·grid`; bucket `[((F−1)//grid)·grid + 1, ((F−1)//grid + 1)·grid]`; if bucket_hi ≤ Tg → `(None, f"≤{Tg}", [1, Tg])`, else `(None, f"{lo}–{hi}", [lo, hi])`.

- [ ] **Step 1: Failing tests**

```python
def test_safe_frac_exact_without_small_codes():
    assert m.safe_frac(2500.6, [{"band": "1k–10k", "codes": 2}]) == (2501, "2501", [2501, 2501])
    assert m.safe_frac(12.4, [])[1:] == ("≤20", [1, 20])

def test_safe_frac_upper_bound_never_narrows():
    sb = [{"band": "101–1k", "codes": 1}, {"band": "≤20", "codes": 1}]   # T = 101+1+19 = 121
    assert m.safe_frac(108, sb) == (None, "≤130", [1, 130])             # the measured live leak shape
    v, d, (lo, hi) = m.safe_frac(1234, sb)
    assert (v, d, lo, hi) == (None, "1231–1240", 1231, 1240)
    for F in range(1, 3000):                                             # property: cap >= 20 always
        _, _, (_, hi) = m.safe_frac(F, sb)
        assert hi - (101 + 1 - 1) >= 20
```
Update `test_fractional_count_suppressed_independently`: A → `frac None, frac_display "≤20", frac_range [1, 20]`; B → `2501`; C → `90`; root `frac_range None`. Add an `assert_diff_safe` negative test: a node with a `≤20` code and exact `frac` must raise; one with `frac_range` hi below `T` must raise.

- [ ] **Step 2: Run, confirm failures.**
- [ ] **Step 3: Implement** `safe_frac`; use it in `assemble_payload` for every non-root node (frac input as today: `n_frac` falling back to `n_persons`); extend `assert_diff_safe` with the two frac checks.
- [ ] **Step 4: Suite green.**
- [ ] **Step 5: Commit** `mondo-usage: two-sided frac publication (Phase 0 upper-bound leak fix)`.

### Task A3: Named / rolled-up-only split disclosure

**Files:** Modify `mondo_usage_cloud.py` — new `split_parts`, `assemble_payload` (accept + emit), `assert_diff_safe`. Tests as above.

**Interfaces:**
- Consumes: `_forced_lower`, `_max_forced_exact`, `_band_bounds` (A1).
- Produces: `split_parts(node: dict, *, n_total: int, n_named: int, named_bands: list[dict], rolled_bands: list[dict], min_cell=20) -> dict` where `node` is the already-suppressed node (`state`, `count`, `count_range`, `total_banded`, `display`) and `n_total` is the true (pre-suppression) total. Tests use a helper `_rep(N) -> {"state": "reported", "count": N, "display": str(N)}`. Returns `{"status": "exact"|"ranged"|"mirror"|"withheld", "named": {"count": int|None, "display": str, "range": [lo, hi]|None}, "rolled": {…}}`. `term_rows` gain optional `n_named` (int; absent → no `split` key on the node), `named_bands`, `rolled_bands`. Node gains `split`.

Decision procedure (spec D4 as amended; every branch condition except the zero-test is public):
1. `n_total == 0` → exact `{0, 0}`.
2. A part is 0 → `mirror`: the non-zero part copies the node's total display/count/range; the other is `{0, "0", [0, 0]}`.
3. Total not exact (`used_small` or `total_banded`) → `withheld` (both `{None, "—", None}`).
4. `N = count`. `N ≤ 40` → `withheld`.
5. Eligibility `E` over every hypothesis **consistent with the public bands** (`caps_p = Σ min(band_hi, N)·codes` for part p):
   - (i) both > 20, consistent iff `caps_n ≥ 21 and caps_r ≥ 21`: safe iff `_max_forced_exact(21, min(N−21, caps_n), named_bands) < 2` and same for rolled.
   - (ii) named ≤ 20, rolled large, consistent iff `caps_r ≥ N − 20`: safe iff `N − caps_r ≤ 1` and `_forced_lower(N − 20, min(N − 1, caps_r), rolled_bands) < 2`.
   - (iii) symmetric with roles swapped.
   `E` = all consistent hypotheses safe. Not `E` → `withheld`.
6. `E` and both > 20 → `exact`. `E` and one part ≤ 20 → `ranged`: small `{None, "≤20", [1, 20]}`, other `{None, f"{N−20}–{N−1}", [N−20, N−1]}`.

- [ ] **Step 1: Failing tests** (`node` built with `state="reported", count=N` unless stated)

```python
def test_split_withholds_small_totals():            # total 22 = 21+1 and 38 = 19+19
    for nn, nr in [(21, 1), (19, 19)]:
        s = m.split_parts(_rep(nn + nr), n_total=nn + nr, n_named=nn,
                          named_bands=[{"band": "≤20", "codes": 2}], rolled_bands=[{"band": "≤20", "codes": 1}])
        assert s["status"] == "withheld"

def test_split_hypothesis_consistent_leak_is_withheld():   # 105 = 100 named + 5 rolled
    s = m.split_parts(_rep(105), n_total=105, n_named=100,
                      named_bands=[{"band": "21–100", "codes": 1}], rolled_bands=[{"band": "≤20", "codes": 1}])
    assert s["status"] == "withheld"                    # publishing would force rolled >= 5

def test_split_hypothesis_impossible_does_not_overwithhold():
    s = m.split_parts(_rep(5000), n_total=5000, n_named=4990,
                      named_bands=[{"band": "1k–10k", "codes": 1}, {"band": "≤20", "codes": 1}],
                      rolled_bands=[{"band": "≤20", "codes": 1}])
    assert s["status"] == "ranged"
    assert s["rolled"]["display"] == "≤20" and s["named"]["range"] == [4980, 4999]

def test_split_exact_and_mirror():
    s = m.split_parts(_rep(5000), n_total=5000, n_named=4000,
                      named_bands=[{"band": "1k–10k", "codes": 1}], rolled_bands=[{"band": "101–1k", "codes": 1}])
    assert s["status"] == "exact" and s["named"]["count"] == 4000 and s["rolled"]["count"] == 1000
    banded = {"state": "reported", "count": None, "display": "21–100", "count_range": [21, 100], "total_banded": True}
    s = m.split_parts(banded, n_total=60, n_named=60, named_bands=[{"band": "≤20", "codes": 4}], rolled_bands=[])
    assert s["status"] == "mirror" and s["named"]["display"] == "21–100" and s["rolled"]["count"] == 0
    small = {"state": "used_small", "count": None, "display": "≤20"}
    assert m.split_parts(small, n_total=9, n_named=4, named_bands=[{"band": "≤20", "codes": 1}],
                         rolled_bands=[{"band": "≤20", "codes": 1}])["status"] == "withheld"
```
Plus `assert_diff_safe` negatives: an exact split part in 1..20; a `ranged` split whose large part's code bands are boxed; parts not summing to an exact total.

- [ ] **Step 2: Run, confirm failures.**
- [ ] **Step 3: Implement** `split_parts`; in `assemble_payload`, after `complementary_suppress`, set `nd["split"]` for rows carrying `n_named`; extend `assert_diff_safe` to re-derive and check each split (status-specific: exact parts sum to count and each > 20 or 0; ranged small part's range is `[1,20]`, other's is `[N−20, N−1]`, and `_forced_lower` on it < 2).
- [ ] **Step 4: Suite green.**
- [ ] **Step 5: Commit** `mondo-usage: named / rolled-up-only split with selection-free disclosure`.

### Task A4: Shortest SNOMED climb chain (pure)

**Files:** Modify `mondo_usage_cloud.py` — new `shortest_climb_chain`. Tests as above.

**Interfaces:**
- Produces: `shortest_climb_chain(start: int, target: int, dist: int, parents_of: dict[int, list[int]], lev_to_target: dict[int, int]) -> tuple[list[int] | None, bool]` → `([start, …, target], ambiguous)`. Walks `Is a` parents; at each hop keeps parents `p` with `lev_to_target[p] == remaining − 1` (`target` itself has level 0); picks the lowest concept id; `ambiguous` is True if any hop had > 1 candidate. Returns `(None, False)` when a hop has no candidate.

- [ ] **Step 1: Failing tests** `test_chain_linear` (1→2→3, dist 2 → `[1,2,3], False`), `test_chain_diamond_is_ambiguous_and_deterministic` (1→{3,2}→4 → `[1,2,4], True`), `test_chain_missing_returns_none`, `test_chain_ignores_off_shortest_parent` (a parent at the wrong level is skipped).
- [ ] **Step 2: Fail. Step 3: Implement. Step 4: Green.**
- [ ] **Step 5: Commit** `mondo-usage: shortest SNOMED climb-chain reconstruction`.

### Task A5: Spark wiring — steps, named counts, paths, concepts, survey

**Files:**
- Modify: `mondo_usage_cloud.py` — `source_climb` branch (~L1403-1640) and `assemble_axis` (~L1127-1381); `assemble_payload(…, concepts: dict | None = None)` adds top-level `"concepts"`; `build_safe_summary` (split lines).
- Test: `analysis/cloud/tests/test_mondo_usage.py` (`@pytest.mark.slow` local-Spark test).

**Interfaces:**
- Produces (module-level, DataFrame helpers so they can be tested locally):
  - `STEP_OF_VIA = {"source_exact": 1, "standard_exact": 2, "hpo_exact": 3, "climbed": 4}`
  - `catalog_steps_df(attribution) -> DataFrame[mondo_id, origin_cid, step]` — min step per (term, code).
  - `named_counts_df(attribution, steps) -> DataFrame[mondo_id, n_named]` — distinct persons with ≥1 attribution through a code whose (term, code) step ≤ 3 (so step-2 codes' climb rows count as named).
  - `code_std_df(attribution, steps) -> DataFrame[mondo_id, origin_cid, std_cid, multi_std]` — among the pair's rows at the winning step, the `k_std` with the most distinct persons (tie → lowest id); `multi_std` if > 1 distinct `k_std`.
- Catalog entry fields (additive): `step` (1-4), `std` (int|None; None for step 1), `path` (list[int] SNOMED chain from `std` to the mapped ancestor inclusive; `[std]` for steps 2-3; None when unreconstructable), `path_ambiguous` (bool), `multi_std` (bool). Keep `via` = `"exact"`/`"climbed"` for old clients. The per-(term, code) step replaces today's `vrank` collapse; band computation unchanged.
- `term_rows` gain `n_named`, `named_bands` (histogram over entries with step ≤ 3, full list before the cap), `rolled_bands` (step 4).
- Payload top-level `concepts: {str(id): [vocab, code]}` for every id in any `std`/`path`.
- Survey (Mondo axis): `split_terms_by_status` (counts of terms per split status) and `persons_rolled_up_only` (cohort-wide distinct persons having ≥1 term whose attributions for them are all step-4 codes; `suppress_count`-style ≤20 handling). Never sums of per-term parts.
- Chain inputs (driver-side): for climbed (std, anc) pairs reduced to the term's anc (`nearest` ⋈ `anc_map` restricted to the reduced (std, mondo) pairs; anc tie → lowest id) with distance `L` from `concept_ancestor.min_levels_of_separation`; collect `concept_ancestor` rows (X, anc, lev) for X in the ancestor closure of the climbed std set, and `concept_relationship` `Is a` edges among that closure; call `shortest_climb_chain`. Everything collected is identity/structure (no persons).
- HPO axis: entries are all step 3 (`std` = `k_std`, `path = [std]`); `n_named = n_persons`, `rolled_bands = []` → splits mirror.
- TSV: `source_codes` items become `vocab:code:step:band`; add `split_status`, `named`, `rolled` (display strings).

- [ ] **Step 1: Failing `@slow` test** `test_catalog_steps_lowest_wins_and_named_counts`: tiny local frames where code C reaches T via standard_exact for person p1 and climbed for p2, code D reaches T only climbed for p3 → steps `{C: 2, D: 4}`, `n_named(T) == 2`, rolled-up only = 1; `code_std_df` picks the majority std.
- [ ] **Step 2: Run with `-m ""`, confirm failure.**
- [ ] **Step 3: Implement helpers + wiring** per the Interfaces block (no UDFs; driver-side pandas for the chain step).
- [ ] **Step 4: Full suite incl. slow green; `python -m py_compile analysis/cloud/mondo_usage_cloud.py`.**
- [ ] **Step 5: Commit** `mondo-usage: per-code step, SNOMED paths, named split + survey in source_climb export`.

### Task A6: Experiment 0120 + ship

**Files:** Create `docs/experiments/0120-mondo-usage-provenance-split.md` (front matter copied from 0109, status `pending`, `with_hpo: true`); update `docs/experiments/README.md` index if it lists entries; spec status line → "approved; Part A built".

- [ ] **Step 1:** Write the experiment doc (why: Phase 0 + provenance; outputs; run command `make -C analysis/cloud exp ID=120`).
- [ ] **Step 2:** `poetry run pytest analysis/cloud/tests/test_mondo_usage.py -q -p no:randomly -m ""` green; `poetry run pytest scripts/tests/test_run_experiment_mondo_usage.py -q` green.
- [ ] **Step 3:** Commit (driver, tests, spec, plan, experiment doc only — **not** the interim payloads/sanitizer/index.html, which the user commits separately) and push `claude/mondo-usage-dashboard`.
- [ ] **Step 4:** Hand the user the cluster command with the branch preamble for `claude/mondo-usage-dashboard`.

---

## Part B — dashboard (after the export is downloaded; preview locally before publishing)

### Task B1: Step colors, glyphs, legend, path tooltip
**Files:** `mondo-usage-dashboard/index.html` — `sourceCatalogRow` (~L1561-1630) and the custom tooltip handler (~L1530-1557).
- Four step colors as CSS tokens (light + dark), glyphs `≡` (1) `→` (2) `◆` (3) `↑` (4); legend lists only steps present. Legacy payload (no `step`) keeps today's inference.
- Tooltip path: `VOCAB code [NIH name] → SNOMED code → … → MONDO:id label` (or `HP:id label`), SNOMED codes from `DATA.concepts`; `path_ambiguous` → "one of several equally short paths"; `path: null` → "SNOMED (n levels)"; `multi_std` → "maps to several SNOMED concepts; showing the most common".
- Verify: headless render with the downloaded payload; spot-check one entry of each step.

### Task B2: Global mode toggle + per-mode counts + frac ranges
**Files:** `index.html` — state (~L426), `ownNum` (~L397), `computeRollup` (~L571-601), size accessor (~L784), mode buttons (~L348, L1734).
- Three-way toggle **All / Exclude climbed / Climbed only** (default All; hidden when no node has `split`). One accessor `partOf(n)` returns `{count, display, range, state}` for the active mode; `withheld` → displayed "—", sized as a used-small placeholder, state derived from its range; `mirror`/`exact`/`ranged` per payload.
- Roll-up: All mode uses `frac`/`frac_range` (range → floor += lo, ceil += hi, est += midpoint) — fixes the midpoint-as-exact bracket bug for banded totals too; split modes use the person-count bracket over `partOf`.
- Side panel code list and band chips filter to the mode (step ≤ 3 vs step 4; chips from `source_bands` recomputed client-side over the listed entries' bands only when the list is complete, else hidden with a note).

### Task B3: Copy + docs
- Methods tooltip: the 4-step ladder now color-keyed; the split and its suppression ("parts of 1–20, totals ≤ 40, or combinations that would narrow a masked code are withheld"); frac shown as a range near masked codes. Remove the interim notice once the new payload has no `meta.sanitized`.
- `mondo-usage-dashboard/README.md`: new fields; refresh instructions name exp 0120.

### Task B4: Local preview, then publish
- Copy the downloaded `mondo_usage.json` / `hpo_usage.json` into `mondo-usage-dashboard/`; set `meta.cdr` to the release-only string if the export carries a workspace name (user confirms); run `assert_diff_safe` over both files via a one-off script; serve with `python3 -m http.server` and preview; user decides on commit/push/merge.

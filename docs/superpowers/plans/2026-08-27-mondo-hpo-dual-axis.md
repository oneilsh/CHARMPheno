# Mondo + HPO Dual-Axis Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a phenotype (HPO) axis beside the Mondo disease axis — each EHR condition routed to its single best home, shown as two side-by-side DAG browsers.

**Architecture:** Extend the `source_climb` attribution ladder with one HPO-exact rung between Mondo standard-exact and Mondo climb. Refactor the per-term assembly so it runs once per axis (Mondo, HPO) over the same engine (bands, fractional 1/m, catalog). Emit `mondo_usage.json` + `hpo_usage.json`; the dashboard loads both and mirrors the HPO DAG leftward. All gated behind `--with-hpo`, fully reversible.

**Tech Stack:** Python 3 / PySpark (driver, `analysis/cloud/`), pytest (pure-helper tests), vanilla JS + inline SVG (`mondo-usage-dashboard/index.html`).

**Spec:** `docs/superpowers/specs/2026-08-27-mondo-hpo-dual-axis-design.md`

## Global Constraints

- **Disclosure:** per-axis small-cell suppression; counts 1–20 masked as `≤20`, never emitted raw (display/count/frac/frac_display/bands). No per-code patient counts. Bands are ranges. `meta.cdr` scrubbed to `All of Us v8 (R2024Q3R8)`; `meta.survey` dropped from published payloads. (These are applied at scrub/gate time as today — the driver may emit raw; the publish gate enforces.)
- **Reversible:** without `--with-hpo`, `source_climb` output and behavior are byte-identical to today. The dashboard with no `hpo_usage.json` renders exactly today's single Mondo view.
- **Pure vs Spark:** pure helpers are TDD'd in `analysis/cloud/tests/test_mondo_usage.py`; the BQ/Spark ladder + the browser UI are validated by cluster run / in-browser, not unit tests (existing convention).
- **HPO source:** `hp.obo` via `--hpo-obo-url` (default `http://purl.obolibrary.org/obo/hp.obo`); best-effort (a load failure disables the HPO axis, never fails the export). SNOMED-first matching (UMLS is v2).
- **No LaTeX / plain-text + Unicode Greek** in any user-facing copy.

## File Structure

- `analysis/cloud/mondo_usage_cloud.py` — add `parse_hpo_dag` + `dag_structures` (pure); HPO ingest, extended ladder, per-axis assembly, `--with-hpo`, safe-summary HPO section.
- `analysis/cloud/tests/test_mondo_usage.py` — tests for the new pure helpers.
- `mondo-usage-dashboard/index.html` — load both payloads, graph growth-direction param, side-by-side render, cross-axis search/drawer, single-axis fallback.
- `docs/experiments/0107-mondo-ehr-usage-source-climb.md` — document `--with-hpo`.

---

## Increment A — HPO ingest (pure parsing + driver wiring)

### Task A1: HPO DAG parser (pure)

**Files:**
- Modify: `analysis/cloud/mondo_usage_cloud.py` (add `parse_hpo_dag`, near `parse_hpo_xrefs`)
- Test: `analysis/cloud/tests/test_mondo_usage.py`

**Interfaces:**
- Consumes: nothing (pure over OBO text).
- Produces: `parse_hpo_dag(obo_text: str) -> tuple[dict, dict]` returning `(labels, parents)` where `labels: {hp_id: name}` and `parents: {hp_id: [parent_hp_id, ...]}` from `is_a:` lines. Obsolete terms (`is_obsolete: true`) excluded. Only `HP:` ids.

- [ ] **Step 1: Write the failing test**

```python
def test_parse_hpo_dag():
    obo = """format-version: 1.2

[Term]
id: HP:0000118
name: Phenotypic abnormality

[Term]
id: HP:0002917
name: Hypomagnesemia
is_a: HP:0004363 ! Abnormal circulating metabolite concentration
is_a: HP:0012418 ! Hypomagnesemia parent two

[Term]
id: HP:0000001
name: All

[Term]
id: HP:9999999
name: obsolete thing
is_a: HP:0000118 ! Phenotypic abnormality
is_obsolete: true

[Typedef]
id: part_of
"""
    labels, parents = m.parse_hpo_dag(obo)
    assert labels["HP:0002917"] == "Hypomagnesemia"
    assert labels["HP:0000118"] == "Phenotypic abnormality"
    assert sorted(parents["HP:0002917"]) == ["HP:0004363", "HP:0012418"]
    assert parents.get("HP:0000118", []) == []          # a root: no is_a
    assert "HP:9999999" not in labels                    # obsolete excluded
    assert "HP:9999999" not in parents
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd analysis/cloud && python3 -m pytest tests/test_mondo_usage.py::test_parse_hpo_dag -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'parse_hpo_dag'`

- [ ] **Step 3: Write minimal implementation**

Add near `parse_hpo_xrefs`:

```python
def parse_hpo_dag(obo_text: str) -> "tuple[dict, dict]":
    """Parse ``hp.obo`` into ``(labels, parents)``: ``labels`` maps ``HP:id`` -> name,
    ``parents`` maps ``HP:id`` -> list of ``is_a`` parent ids. Obsolete terms are dropped.
    Pure — no I/O. Mirrors Mondo's (nodes, edges) shape so the DAG machinery is reused."""
    labels, parents = {}, {}
    hp_id = hp_name = None
    par: list = []
    obsolete = False
    in_term = False

    def _flush():
        if in_term and hp_id and hp_id.startswith("HP:") and not obsolete:
            labels[hp_id] = hp_name or hp_id
            parents[hp_id] = list(par)

    for raw in obo_text.splitlines():
        line = raw.rstrip()
        if line == "[Term]":
            _flush()
            in_term, hp_id, hp_name, par, obsolete = True, None, None, [], False
            continue
        if line.startswith("[") and line.endswith("]"):
            _flush()
            in_term = False
            continue
        if not in_term:
            continue
        if line.startswith("id:"):
            hp_id = line[3:].strip()
        elif line.startswith("name:"):
            hp_name = line[5:].strip()
        elif line.startswith("is_obsolete:") and line.split(":", 1)[1].strip() == "true":
            obsolete = True
        elif line.startswith("is_a:"):
            p = line[5:].strip().split("{")[0].split(" ! ")[0].strip()
            if p.startswith("HP:"):
                par.append(p)
    _flush()
    return labels, parents
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd analysis/cloud && python3 -m pytest tests/test_mondo_usage.py::test_parse_hpo_dag -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analysis/cloud/mondo_usage_cloud.py analysis/cloud/tests/test_mondo_usage.py
git commit -m "feat(hpo): pure parse_hpo_dag (labels + is_a parents from hp.obo)"
```

### Task A2: Generic DAG structures (pure)

**Files:**
- Modify: `analysis/cloud/mondo_usage_cloud.py` (add `dag_structures`)
- Test: `analysis/cloud/tests/test_mondo_usage.py`

**Interfaces:**
- Consumes: `parents` dict from A1 (child -> [parents]).
- Produces: `dag_structures(parents: dict) -> tuple[dict, set]` returning `(parent_adj, has_child)` where `parent_adj` is the same child->parents restricted to known ids, and `has_child` is the set of ids that are a parent of ≥1 known id. Mirrors the Mondo `parent_adj` / `has_child` used by `run_space`.

- [ ] **Step 1: Write the failing test**

```python
def test_dag_structures():
    parents = {"A": [], "B": ["A"], "C": ["A", "B"], "D": ["Z"]}   # Z unknown
    parent_adj, has_child = m.dag_structures(parents)
    assert parent_adj["C"] == ["A", "B"]
    assert parent_adj["D"] == []            # unknown parent Z dropped
    assert has_child == {"A", "B"}          # A parents B,C; B parents C; D has no children
    assert "C" not in has_child
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd analysis/cloud && python3 -m pytest tests/test_mondo_usage.py::test_dag_structures -v`
Expected: FAIL (`no attribute 'dag_structures'`)

- [ ] **Step 3: Write minimal implementation**

```python
def dag_structures(parents: dict) -> "tuple[dict, set]":
    """From a child->[parents] map, return ``(parent_adj, has_child)`` restricted to known
    ids: ``parent_adj`` drops parents not in the map; ``has_child`` is every id that is a
    parent of a known id. Pure. Shared by the Mondo and HPO axes for the DAG browse/roll-up."""
    known = set(parents)
    parent_adj = {c: [p for p in ps if p in known] for c, ps in parents.items()}
    has_child = {p for ps in parent_adj.values() for p in ps}
    return parent_adj, has_child
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd analysis/cloud && python3 -m pytest tests/test_mondo_usage.py::test_dag_structures -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analysis/cloud/mondo_usage_cloud.py analysis/cloud/tests/test_mondo_usage.py
git commit -m "feat(hpo): pure dag_structures (parent_adj + has_child) shared by both axes"
```

### Task A3: Driver HPO ingest wiring (cluster-only)

**Files:**
- Modify: `analysis/cloud/mondo_usage_cloud.py` (`main`, near the existing HPO-probe block)

**Interfaces:**
- Consumes: `parse_hpo_dag`, `dag_structures`, existing `hpo_by_snomed` / `hpo_by_icd` (already built by the probe block).
- Produces: in-scope-of-`main` HPO structures reused by the ladder + assembly: `hpo_labels` (dict), `hpo_parent_adj` (dict), `hpo_has_child` (set), and a Spark broadcast frame `hpo_map_sdf` = rows `(map_cid, hp_id)` where `map_cid` is an OMOP concept_id (SNOMED std, joined from `concept_pd` on `concept_code`; + ICD source concepts from `hpo_by_icd`). Empty/absent when the obo failed to load.

- [ ] **Step 1: Extend the HPO load block** (in `main`, where `hpo_by_snomed` is built) to also build the DAG and the OMOP-concept map. This is Spark/pandas glue — validated by cluster run, not a unit test.

```python
# after parse_hpo_xrefs(...) populates hpo_by_snomed / hpo_by_icd:
hpo_labels, hpo_parents = parse_hpo_dag(dest.read_text())
hpo_parent_adj, hpo_has_child = dag_structures(hpo_parents)
# OMOP concept_id -> hp_id, via SNOMED concept_code (primary) and ICD source code:
snomed_cp = concept_pd[concept_pd["vocabulary_id"] == "SNOMED"]
hpo_cid_rows = [(int(r.concept_id), hpo_by_snomed[str(r.concept_code)][0])
                for r in snomed_cp.itertuples() if str(r.concept_code) in hpo_by_snomed]
icd_cp = concept_pd[concept_pd["vocabulary_id"].isin(["ICD10CM", "ICD9CM"])]
hpo_cid_rows += [(int(r.concept_id), hpo_by_icd[(str(r.vocabulary_id), str(r.concept_code))][0])
                 for r in icd_cp.itertuples()
                 if (str(r.vocabulary_id), str(r.concept_code)) in hpo_by_icd]
```

Guard everything with the existing `try/except` so a load failure leaves `hpo_labels = {}` (HPO axis disabled). Only build these when `args.with_hpo` (added in Task B3) — until then behind the same `if args.hpo_obo_url ...` guard is fine; wire the flag in B3.

- [ ] **Step 2: Cluster smoke** (deferred; run in Task B validation) — confirm stderr logs `[hpo] N DAG terms, M concept ids mapped`. Add that stderr line.

- [ ] **Step 3: Commit**

```bash
git add analysis/cloud/mondo_usage_cloud.py
git commit -m "feat(hpo): build HPO DAG + OMOP-concept->HP map alongside the xref probe"
```

---

## Increment B — Dual-axis attribution + payloads (driver)

### Task B1: Extended ladder — the HPO-exact rung (cluster-only)

**Files:**
- Modify: `analysis/cloud/mondo_usage_cloud.py` (`run_space`, source_climb branch, the t1/t2/t3 construction)

**Interfaces:**
- Consumes: `hpo_map_sdf` (A3), existing `rem1` (conditions not source-exact) and the std/climb frames.
- Produces: `t_hpo` (attribution rows to HP ids) and a Mondo climb tier `t3` computed on the remainder **after** HPO. Adds a boolean `args.with_hpo` gate: when false, no `t_hpo`, `t3` unchanged (today's behavior).

- [ ] **Step 1: Insert the HPO rung** between standard-exact (`t2`/`rem1`) and the climb. After computing `rem2` (rem1 minus standard-exact — conditions with no direct Mondo term):

```python
if args.with_hpo and hpo_labels:
    hpo_map = broadcast(spark.createDataFrame(
        pd.DataFrame(hpo_cid_rows, columns=["map_cid", "hp_id"]).drop_duplicates()))
    # HPO-exact: a rem2 condition whose STANDARD concept (or source) is an HPO xref.
    t_hpo = (rem2.join(hpo_map, rem2["std_cid"] == hpo_map["map_cid"], "inner")
             .select("person_id", F.col("hp_id").alias("mondo_id"),   # reuse 'mondo_id' col name
                     origin.alias("origin_cid"), F.lit("hpo_exact").alias("via"),
                     rem2["src_cid"].alias("k_src"), rem2["std_cid"].alias("k_std")))
    # climb only what HPO did NOT claim:
    rem2_climb = rem2.join(t_hpo.select("k_src", "k_std").distinct(),
                           (rem2["src_cid"].eqNullSafe(F.col("k_src")) &
                            rem2["std_cid"].eqNullSafe(F.col("k_std"))), "left_anti")
else:
    t_hpo = None
    rem2_climb = rem2
```

Then build the existing climb `t3` from `rem2_climb` instead of `rem2`. Keep `t_hpo` separate from the Mondo `attribution` (do NOT union it into Mondo).

- [ ] **Step 2: Cluster validation** (deferred to B-end): `t_hpo` non-empty; Mondo `persons_climbed` drops vs the non-HPO run; headache SNOMED `25064002` now appears under HP, not `vertebral artery occlusion`.

- [ ] **Step 3: Commit**

```bash
git add analysis/cloud/mondo_usage_cloud.py
git commit -m "feat(hpo): HPO-exact ladder rung; Mondo climbs only the un-HPO'd remainder"
```

### Task B2: Per-axis assembly + `hpo_usage.json` (cluster-only)

**Files:**
- Modify: `analysis/cloud/mondo_usage_cloud.py` (`run_space`)

**Interfaces:**
- Consumes: `t_hpo` (B1), `hpo_parent_adj` / `hpo_has_child` / `hpo_labels` (A3), the existing per-term pipeline (bands / fractional / catalog).
- Produces: a second payload written to `<out>/hpo_usage.json` with the same schema as the Mondo payload, built from `t_hpo` over the HPO DAG. Returns an HPO summary row for the safe summary.

- [ ] **Step 1: Factor the assembly.** Extract the block that turns an attribution DataFrame + `(parent_adj, has_child, label_of, rare_of)` into `term_rows` → `assemble_payload` → written JSON, into a local closure `assemble_axis(attribution, parent_adj, has_child, label_of, rare_of, out_name, axis_label)`. Call it for Mondo (existing frames, `attribution`) and, when `t_hpo` is not None, for HPO (`hpo_parent_adj`, `hpo_has_child`, `hpo_labels`, `rare_of={}`, `out_name="hpo_usage.json"`). Reuse bands/fractional/catalog verbatim per axis (they key off the attribution + concept table only).

This is a mechanical extraction (branch bodies unchanged, just parameterized) + one extra call — validate by `python3 -m py_compile` and cluster run (matches the earlier `run_space` refactor pattern).

- [ ] **Step 2: `py_compile` + full pure test suite still green.**

Run: `cd analysis/cloud && python3 -m py_compile mondo_usage_cloud.py && python3 -m pytest tests/test_mondo_usage.py -q`
Expected: compile OK; all tests pass (no pure logic changed).

- [ ] **Step 3: Commit**

```bash
git add analysis/cloud/mondo_usage_cloud.py
git commit -m "feat(hpo): assemble_axis — emit hpo_usage.json from the HPO attribution"
```

### Task B3: `--with-hpo` flag + safe-summary HPO section

**Files:**
- Modify: `analysis/cloud/mondo_usage_cloud.py` (argparse; `build_safe_summary`)
- Test: `analysis/cloud/tests/test_mondo_usage.py`
- Modify: `docs/experiments/0107-mondo-ehr-usage-source-climb.md`

**Interfaces:**
- Consumes: HPO summary row (B2).
- Produces: `--with-hpo` (store_true, default False) — the master switch for the whole HPO axis. `build_safe_summary` renders an "HPO axis" block (mapped HP terms used, ≤-suppressed persons on HPO) when an HPO row is present.

- [ ] **Step 1: Write the failing test** (summary renders + suppresses the HPO-axis row):

```python
def test_build_safe_summary_hpo_axis_row():
    results = [{
        "space": "source_climb", "min_cell": 20, "mondo_version": "2026-06-02",
        "generated_utc": "2026-08-27T00:00:00Z",
        "stats": {"mapped_terms": 8895, "used_terms": 4800, "used_fraction": 0.54,
                  "reported_terms": 3000, "used_small_terms": 1800,
                  "collision_terms": 2000, "rare_used_terms": 2100},
        "n_total": 626396, "n_coded": 349815, "n_on_mondo": 342394, "survey": {},
        "hpo_axis": {"used_terms": 512, "reported_terms": 300, "persons": 7},  # persons -> ≤20
    }]
    out = m.build_safe_summary(results)
    assert "HPO axis" in out and "512" in out
    assert "persons ≤20" in out and " 7 " not in out
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd analysis/cloud && python3 -m pytest tests/test_mondo_usage.py::test_build_safe_summary_hpo_axis_row -v`
Expected: FAIL (no "HPO axis" text)

- [ ] **Step 3: Implement** — add `--with-hpo` to argparse; in `build_safe_summary`, after the source_climb survey section, if any result has `hpo_axis`, append:

```python
hx = next((r["hpo_axis"] for r in results if r.get("hpo_axis")), None)
if hx:
    L += ["", "## HPO axis (phenotypes)", "",
          f"- HPO terms used in the EHR: {hx.get('used_terms', 0)} "
          f"(reported >{min_cell}: {hx.get('reported_terms', 0)}) · persons "
          f"{suppress_count(hx.get('persons'), min_cell)}"]
```

Wire `run_space`/`main` to attach `hpo_axis` to the source_climb result dict (used/reported term counts from the HPO payload stats; `persons` = distinct persons in `t_hpo`).

- [ ] **Step 4: Run to verify it passes**

Run: `cd analysis/cloud && python3 -m pytest tests/test_mondo_usage.py -q`
Expected: PASS (all)

- [ ] **Step 5: Document + commit** — add a `--with-hpo` note to `docs/experiments/0107-*.md` (one paragraph: routing rung, two payloads, reversible).

```bash
git add analysis/cloud/mondo_usage_cloud.py analysis/cloud/tests/test_mondo_usage.py docs/experiments/0107-mondo-ehr-usage-source-climb.md
git commit -m "feat(hpo): --with-hpo flag + HPO-axis block in the safe summary"
```

- [ ] **Step 6: CLUSTER VALIDATION GATE** (whole driver). Run `make -C analysis/cloud exp ID=108` with `--with-hpo` wired into the frontmatter/args. Confirm: `hpo_usage.json` written; Mondo `persons_climbed` dropped vs the prior run; headache/neck-pain no longer under vertebral-artery-occlusion; summary shows the HPO axis block. Paste the summary back for review before publishing.

---

## Increment C — Dual-axis dashboard UI

> All Task-C validation is in-browser (open `mondo-usage-dashboard/index.html` over `python3 -m http.server`). No unit tests. Parse-check every JS edit with:
> `end=$(grep -n "</script>" index.html | tail -1 | cut -d: -f1); sed -n "378,$((end-1))p" index.html > /tmp/m.js && node --check /tmp/m.js`

### Task C1: Load both payloads + single-axis fallback

**Files:**
- Modify: `mondo-usage-dashboard/index.html` (`boot()` / data load)

**Interfaces:**
- Produces: globals `MONDO` (today's `DATA`) and `HPO` (second payload or `null`). `HAS_HPO = !!HPO`. Everything downstream that says `DATA` keeps working on the Mondo axis; HPO is additive.

- [ ] **Step 1:** In `boot()`, after loading `mondo_usage.json` into `MONDO`, attempt `fetch("hpo_usage.json")`; on success set `HPO`, else `HPO=null`. Log which loaded. Keep `DATA` aliased to `MONDO` so no existing code breaks.
- [ ] **Step 2:** Parse-check (command above). Open in browser: with no `hpo_usage.json` present, the page renders exactly as today.
- [ ] **Step 3: Commit** — `git commit -m "feat(hpo-ui): load optional hpo_usage.json alongside mondo (fallback safe)"`

### Task C2: Parameterize graph growth direction

**Files:**
- Modify: `mondo-usage-dashboard/index.html` (graph layout — the x-position / layering code)

**Interfaces:**
- Produces: the layout accepts a `dir` (+1 rightward, -1 leftward); node x = `dir * depth * COL_W` (+ origin offset). Edges/labels/hit-testing follow. Default +1 (today's Mondo view unchanged).

- [ ] **Step 1:** Find where node x-position is computed from depth/layer; multiply the horizontal advance by a `dir` param threaded from the render call. Flip label anchor (`text-anchor`) and the child-expansion side when `dir < 0`.
- [ ] **Step 2:** Parse-check + browser: Mondo view with `dir=+1` looks identical to today.
- [ ] **Step 3: Commit** — `git commit -m "feat(hpo-ui): graph layout growth-direction param (default rightward)"`

### Task C3: Side-by-side render (Mondo right, HPO left-mirrored)

**Files:**
- Modify: `mondo-usage-dashboard/index.html` (stage layout / render dispatch, CSS)

**Interfaces:**
- Consumes: `MONDO`, `HPO`, the `dir` param (C2).
- Produces: when `HAS_HPO`, the stage shows two graph panels — Mondo (`dir=+1`, right) and HPO (`dir=-1`, left) — each rendering its own payload with the existing engine (bands/fractional/roll-up all per-axis). When `!HAS_HPO`, the single Mondo panel as today.

- [ ] **Step 1:** Split the graph stage into two containers (flex row; HPO left, Mondo right) shown only when `HAS_HPO`. Render each with its payload + `dir`. Share theme/zoom where trivial; independent pan is fine for v1.
- [ ] **Step 2:** Parse-check + browser with a real `hpo_usage.json`: two DAGs, HPO mirrored leftward, each browsable/expandable. Iterate on spacing/mirroring.
- [ ] **Step 3: Commit** — `git commit -m "feat(hpo-ui): side-by-side Mondo (right) + HPO (left, mirrored) DAGs"`

### Task C4: Cross-axis search + drawer

**Files:**
- Modify: `mondo-usage-dashboard/index.html` (search, `select`, drawer)

**Interfaces:**
- Consumes: both axes' node maps.
- Produces: search matches across both ontologies (results tagged by axis); selecting a node opens the shared drawer for that axis's node (the drawer already reads a node object — pass the node + its axis's roll-up map). Header/methods copy updated to describe the two axes + the routing (Mondo-exact > HPO-exact > Mondo-climb).

- [ ] **Step 1:** Generalize the node lookup / roll-up access the drawer + search use so they resolve against whichever axis owns the id (namespace by `MONDO:`/`HP:` prefix). Update `computeRollup` to run per axis (store `MONDO._roll` / `HPO._roll`).
- [ ] **Step 2:** Update the methods tooltip: two axes, the routing ladder, "phenotypes (HPO) shown left, diseases (Mondo) right; each code routed to its single best home, Mondo preferred."
- [ ] **Step 3:** Parse-check + browser: search finds terms in both; clicking either opens the right drawer; single-axis fallback still clean.
- [ ] **Step 4: Commit** — `git commit -m "feat(hpo-ui): cross-axis search + drawer + methods copy for the two axes"`
- [ ] **Step 5: DEPLOY GATE** — publish only after the driver CLUSTER VALIDATION GATE (B3 Step 6) passed and the refreshed `hpo_usage.json` cleared the safety gate. Push; watch `dashboard.yml`; verify both payloads live under `mondo-aou-usage/`.

---

## Self-Review

**Spec coverage:** routing ladder → B1; HPO exact-only → B1 (no HPO climb built); SNOMED-first + ICD → A3/B1 (UMLS explicitly deferred); full HPO DAG → A1/A3 (no anchor filter); reversible `--with-hpo` → B3 + C1 fallback; two payloads → B2; side-by-side mirrored UI → C2/C3; disclosure per-axis → reuses existing suppression (assemble_payload/scrub gate) + B3 summary test. Covered.

**Placeholder scan:** pure-helper tasks carry real test + impl code; Spark/UI tasks carry concrete code sketches + explicit cluster/browser validation (the codebase convention — Spark/UI are not unit-testable locally), not vague "add handling."

**Type consistency:** `parse_hpo_dag -> (labels, parents)`; `dag_structures(parents) -> (parent_adj, has_child)`; `t_hpo` reuses the `mondo_id` column (holding HP ids) so `assemble_axis` is identical across axes; `hpo_axis` summary dict keys (`used_terms`/`reported_terms`/`persons`) match B3 test + render.

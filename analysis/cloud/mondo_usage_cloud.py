"""Whole-Mondo EHR-USAGE export (exact map, NO roll-up) — the "how much of Mondo
does an EHR actually touch" report (BQ-only).

Sibling to exp 0087/0088 but answers a different question and, deliberately, does
NOT roll patient counts up the SNOMED/Mondo hierarchy. Where `mondo_hierarchy_cloud`
(0088) power-counts each anchor by climbing `concept_ancestor` (every in-subtree
descendant code counts toward the ancestor), THIS driver counts a Mondo term ONLY
by the patients whose condition code maps EXACTLY to that term's own standard
concept(s). Rationale (see docs/insights/0075):

  * Real EHR diagnoses are recorded at different granularities. A roll-up hides
    that: a mid-level Mondo term that itself carries a SNOMED code may legitimately
    be used by ≤20 patients even when its descendants are common, and a very
    abstract term may be used by 0. Those are findings, not artifacts — so we
    report each term's OWN exact-code usage and let any roll-up happen downstream,
    where it can carry the double-counting caveat explicitly.
  * A patient with comorbidities is counted once PER Mondo term (distinct persons),
    never within a term. Across terms they may appear more than once — that is the
    honest per-term number; summing across terms double-counts and is the
    consumer's responsibility.

COUNT SPACE (``--count-space``). Three ways to count a term's patients:
  * ``standard`` (default): distinct persons with a ``condition_concept_id`` that the
    term maps to via ``same_as -> Maps to`` (standard SNOMED Condition). OMOP's
    ICD->SNOMED ``Maps to`` is one-to-many, so a single ICD source code can DECOMPOSE
    into several standard concepts, including generic context ones (e.g. O90.3
    "peripartum cardiomyopathy" pulls in "pregnancy finding"), which then inflates
    the term and creates cross-term collisions (insight 0076).
  * ``source``: distinct persons whose ``condition_source_concept_id`` exactly equals
    one of the term's OWN Mondo ``same_as`` source codes. No ``Maps to`` -> no
    decomposition, no inflation, and NO cross-term collisions (``same_as`` is
    source-injective). Trade-off: coverage is limited to the vocabularies Mondo lists
    (a patient coded only in ICD9 is missed). Source mode requires
    ``condition_occurrence``. This space is also the route to SNOMED-license-free
    deployment (structure from Mondo, tokens from ICD).
  * ``source_climb``: a 3-tier PARTIAL ROLL-UP that credits each condition to the most
    SPECIFIC mapped Mondo term reachable, cataloguing the originating source code per
    term (tagged exact/climbed). Precedence, first hit wins, never climb past an exact:
    (1) source-exact (``condition_source_concept_id`` is a same_as code), else
    (2) standard-exact (``condition_concept_id`` via same_as -> Maps to), else
    (3) climb the nearest mapped SNOMED ancestor of the standard concept via
    ``concept_ancestor`` (ties -> counted in each, flagged as a collision). Because
    ``concept_ancestor`` carries subclass edges only for STANDARD concepts, only the
    SNOMED concept is climbed (ICD is non-standard); the ICD source rides up through
    whatever standard concept OMOP assigned. Recovers coverage ``source`` drops without
    the blanket ``Maps to`` decomposition of ``standard``. Emits a per-vocabulary
    coverage survey (persons exact/climbed/unmatched). Requires ``condition_occurrence``.

MULTI-MAPPING / COLLISIONS. In ``standard`` space the only cross-term collision is
the ``Maps to`` convergence above (measurable from the mapping frame,
`standard_concept_id -> {mondo_id}`); each affected term still reports its full count
and is FLAGGED with its co-mapped siblings. In ``source`` space collisions vanish.

CODE MULTIPLICITY. Per term we also publish how many distinct source/target ids roll
into it, by vocabulary (`n_codes`, `codes_by_vocab`, `codes`). These are exact CODE
counts, never patient counts, so they are unsuppressed and cannot be differenced
against the (suppressed) patient totals — we never publish per-code patient counts.

AoU SMALL-CELL SUPPRESSION: every reported patient count is three-state —
`unused` (0), `used ≤20` (0<n<=20, kept & flagged as used but never given an exact
number or conflated with 0), or the exact count (>20). We publish only per-term
floored cardinalities, never additive decompositions or parent-minus-child deltas,
so nothing can be differenced back to a suppressed cell. Term/node COUNTS in the
headline stats are counts of Mondo terms, not patients, so they are exact.

Artifacts (written to --out): `mondo_usage.json` (the dashboard payload: nodes +
nearest-mapped-ancestor edges + three-state counts + collision flags + headline
stats) and `mondo_usage_nodes.tsv` (the same, spreadsheet-friendly, suppressed).

Run:  make -C analysis/cloud exp ID=105   (model_class=mondo_usage)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_MIN_CELL = 20
_ROOT_ID = "root"


# --------------------------------------------------------------------------- #
# Pure, Spark-free core (unit-tested in tests/test_mondo_usage.py)            #
# --------------------------------------------------------------------------- #
def usage_state(n: int, min_cell: int = _MIN_CELL) -> tuple[str, str, int | None]:
    """Three-state AoU small-cell rule for a per-term distinct-person count.

    Returns ``(state, display, public_count)``:
      * n <= 0         -> ("unused",     "0",            0)    — term not used
      * 0 < n <= floor -> ("used_small", f"≤{floor}", None) — USED, count withheld
      * n >  floor     -> ("reported",   str(n),         n)    — exact

    The floor is inclusive (mask counts of 1..floor, publish only n > floor),
    matching the All of Us Data Browser's "≤ 20" display — nothing at or
    below the floor is ever emitted exactly. The middle state is the crux: a term
    used by 1..floor patients is a first-class USED term (kept, counted toward
    "fraction of Mondo used", never dropped and never shown as 0), but its exact
    count is never emitted.
    """
    n = int(n)
    if n <= 0:
        return "unused", "0", 0
    if n <= min_cell:
        return "used_small", f"≤{min_cell}", None
    return "reported", str(n), n


def collision_map(pairs) -> dict[int, list[str]]:
    """``{standard_concept_id: sorted distinct mondo_ids}`` from (std_cid, mondo_id)
    pairs. A std concept mapped from >1 Mondo term is a cross-term collision (OMOP
    `Maps to` coarsening): patients on that concept are attributed to every such
    term."""
    out: dict[int, set] = {}
    for cid, mid in pairs:
        out.setdefault(int(cid), set()).add(str(mid))
    return {cid: sorted(mids) for cid, mids in out.items()}


def term_collision_siblings(
    term_std: dict[str, list[int]], std_to_mondos: dict[int, list[str]]
) -> dict[str, list[str]]:
    """For each term, the OTHER Mondo terms it shares any standard concept with
    (its collision siblings). ``term_std`` = {mondo_id: [std_cid, ...]}."""
    out: dict[str, list[str]] = {}
    for mid, cids in term_std.items():
        sib: set = set()
        for cid in cids:
            sib.update(std_to_mondos.get(int(cid), ()))
        sib.discard(mid)
        out[mid] = sorted(sib)
    return out


_RARE_SRC = {"gard_rare": "GARD", "orphanet_rare": "Orphanet", "nord_rare": "NORD",
             "doid_rare": "DOID", "ncit_rare": "NCIt", "inferred_rare": "inferred",
             "mondo_curated_rare": "Mondo"}

# Only the dedicated rare-disease registries are trusted to designate a term "rare".
# DOID's `rare_slim` subset (and NCIt's, and Mondo's own `inferred_rare`) is broadly
# over-inclusive — it sweeps in common cancers like prostate cancer — so a term is
# rare here iff it is listed by GARD, Orphanet, or NORD. The umbrella `rare` token is
# NOT trusted on its own for the same reason (it is the union of all subsets). See the
# dashboard README's rare-disease note.
_TRUSTED_RARE = {"GARD", "Orphanet", "NORD"}


def rare_from_nodes(nodes_df):
    """Per Mondo term: (is_rare, [source registries]) parsed directly from the Mondo
    `subsets` field (works regardless of count space, since it doesn't need the OMOP
    mapping). A term is rare iff a *trusted* rare-disease registry (`_TRUSTED_RARE`:
    GARD/Orphanet/NORD) lists it; the untrusted DOID/NCIt/inferred subsets and the
    umbrella `rare` token do not designate rarity on their own.
    Returns two ``{mondo_id: ...}`` dicts (only rare terms are populated)."""
    rare_of, src_of = {}, {}
    subs_col = nodes_df["subsets"] if "subsets" in nodes_df.columns else None
    if subs_col is None:
        return rare_of, src_of
    for mid, subs in zip(nodes_df["id"], subs_col.fillna("")):
        toks = set(str(subs).split("|"))
        srcs = sorted({_RARE_SRC[t] for t in _RARE_SRC if t in toks} & _TRUSTED_RARE)
        if srcs:
            rare_of[str(mid)] = True
            src_of[str(mid)] = srcs
    return rare_of, src_of


def term_rare_flags(mapping):
    """Per Mondo term: (is_rare, [source registries]) from the mapping's rare-subset
    columns (the mondo2omop port exposes `rare` + gard/nord/orphanet/inferred_rare).
    A term is rare iff a *trusted* rare-disease registry (`_TRUSTED_RARE`:
    GARD/Orphanet/NORD) lists it; the umbrella `rare` flag and the untrusted
    doid/ncit/inferred columns do not designate rarity on their own (they over-include
    common diseases — e.g. prostate cancer via DOID). Pure over a pandas-like frame
    (columns accessed by name); returns two ``{mondo_id: ...}`` dicts."""
    cols = [c for c in _RARE_SRC if c in getattr(mapping, "columns", [])]
    rare_of, src_of = {}, {}
    if not cols:
        return rare_of, src_of
    for mid, sub in mapping.groupby("mondo_id"):
        srcs = sorted({_RARE_SRC[c] for c in cols
                       if int(sub[c].max() or 0) == 1} & _TRUSTED_RARE)
        if srcs:
            rare_of[str(mid)] = True
            src_of[str(mid)] = srcs
    return rare_of, src_of


def nearest_mapped_parents(
    mapped_ids: set, parent_adj: dict[str, list[str]]
) -> dict[str, list[str]]:
    """For each mapped term, its nearest ANCESTOR terms that are themselves mapped
    (collapsing unmapped Mondo intermediates), over a child->parents adjacency. A
    term with no mapped ancestor gets ``[]`` (attaches to the synthetic root).

    "Nearest" = the first mapped node on every upward path; the search stops
    climbing a branch as soon as it hits a mapped ancestor, so we keep the closest
    mapped parents on each branch (the induced Hasse edges over mapped terms)."""
    out: dict[str, list[str]] = {}
    for node in mapped_ids:
        found: set = set()
        seen: set = set()
        stack = list(parent_adj.get(node, ()))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if cur in mapped_ids:
                found.add(cur)          # nearest on this branch; don't climb past it
            else:
                stack.extend(parent_adj.get(cur, ()))
        out[node] = sorted(found)
    return out


def nearest_mapped_standard_ancestors(edges) -> dict[int, list[int]]:
    """The ``source_climb`` count-space climb, as pure logic.

    ``edges`` is an iterable of ``(descendant_concept_id, ancestor_concept_id,
    levels_of_separation)`` triples from OMOP ``concept_ancestor``. The caller
    pre-filters so that every ``ancestor_concept_id`` is a *mapped* standard
    concept (a concept some Mondo term carries via ``same_as -> Maps to``) and
    that ``descendant_concept_id`` is a standard concept with NO exact Mondo match
    (an exact, distance-0 match wins before any climb and is resolved earlier).

    For each descendant we keep only the ancestor(s) at the SMALLEST
    ``levels_of_separation >= 1`` — the nearest mapped Mondo term(s) up the SNOMED
    hierarchy. Ties (>=2 mapped ancestors equally near) are all kept, so the term
    rows can flag the shared attribution as a collision (a patient with that one
    code counts under each tied term). A descendant with no mapped ancestor is
    simply absent (it stays unmatched and is surveyed).

    Returns ``{descendant_concept_id: sorted[ancestor_concept_id, ...]}``.
    Concept-ancestor is only populated for STANDARD concepts (SNOMED for the
    condition domain); non-standard source codes (ICD) never appear as
    descendants here, which is why the ladder climbs the standard concept."""
    best: dict[int, tuple[int, set]] = {}
    for d, a, lv in edges:
        d, a, lv = int(d), int(a), int(lv)
        if lv < 1:                       # level 0 = self/exact, never a climb
            continue
        cur = best.get(d)
        if cur is None or lv < cur[0]:
            best[d] = (lv, {a})
        elif lv == cur[0]:
            cur[1].add(a)
    return {d: sorted(anc) for d, (lv, anc) in best.items()}


def _depths(parents_of: dict[str, list[str]], root: str = _ROOT_ID) -> dict[str, int]:
    """Longest-path depth from ``root`` for each node (root=0). Memoized; robust to
    the Mondo DAG's multi-parenthood and (defensively) to cycles."""
    depth: dict[str, int] = {}

    def d(n: str, on_path: frozenset = frozenset()) -> int:
        if n in depth:
            return depth[n]
        ps = parents_of.get(n, [])
        if not ps or n in on_path:
            depth[n] = 0
            return 0
        depth[n] = 1 + max(d(p, on_path | {n}) for p in ps)
        return depth[n]

    for n in parents_of:
        d(n)
    return depth


def _used_path_set(parents_of: dict[str, list[str]], used: set) -> set:
    """Every node that is used OR a (transitive) ancestor of a used node, over the
    child->parents DAG. These are the "used skeleton": the used terms plus the
    branch points above them — the structural nodes that stay interesting even at a
    0 direct count (no roll-up, so an abstract ancestor may itself be coded 0 times
    while still sitting above real usage)."""
    on_path = set(used)
    stack = list(used)
    while stack:
        n = stack.pop()
        for parent in parents_of.get(n, ()):
            if parent not in on_path:
                on_path.add(parent)
                stack.append(parent)
    return on_path


def assemble_payload(*, meta: dict, term_rows: list[dict],
                     min_cell: int = _MIN_CELL) -> dict:
    """Build the dashboard JSON. ``term_rows`` is one dict per mapped Mondo term:
    ``mondo_id, label, is_internal(bool), parents(list[mondo_id]), std_concepts
    (list[int]), n_persons(int raw), collision_siblings(list[mondo_id])``.

    Pure and time-free (the driver stamps ``meta['generated_utc']``). Each node
    carries a three-state ``state`` (unused/used_small/reported) AND a four-way
    display ``category`` for the dashboard's show/hide layers:
      * ``reported``     — exact count (>= floor)
      * ``used_small``   — used, 0<count<floor (count withheld)
      * ``used_branch``  — 0 direct count but an ANCESTOR of a used node (a branch
                           point on the used skeleton — kept, structurally relevant)
      * ``other``        — 0 count and NOT above any used node (the rest of Mondo)
    Headline stats are exact term counts (terms are not patients)."""
    nodes: list[dict] = []
    parents_of: dict[str, list[str]] = {_ROOT_ID: []}
    state_of: dict[str, str] = {}
    counts = {"unused": 0, "used_small": 0, "reported": 0}
    n_internal_used = n_collision = 0

    for r in term_rows:
        state, display, public = usage_state(r["n_persons"], min_cell)
        counts[state] += 1
        state_of[r["mondo_id"]] = state
        parents = r["parents"] or [_ROOT_ID]
        parents_of[r["mondo_id"]] = parents
        siblings = r.get("collision_siblings") or []
        if siblings:
            n_collision += 1
        if state != "unused" and r["is_internal"]:
            n_internal_used += 1
        nodes.append({
            "id": r["mondo_id"],
            "label": r["label"],
            "kind": "internal" if r["is_internal"] else "leaf",
            "parents": parents,
            "std_concepts": [int(c) for c in r["std_concepts"]],
            "state": state,
            "display": display,
            "count": public,                 # None when withheld
            "collision": bool(siblings),
            "collision_siblings": siblings,
            "rare": bool(r.get("rare")),
            "rare_src": list(r.get("rare_src") or []),
            # code multiplicity (exact COUNTS of source/target ids — not patient
            # counts, so unsuppressed and un-differenceable): how many distinct ids
            # roll into this term, by vocabulary, plus the ids themselves.
            "codes": list(r.get("codes") or []),
            "n_codes": int(r.get("n_codes") or len(r.get("codes") or [])),
            "codes_by_vocab": dict(r.get("codes_by_vocab") or {}),
            # source_climb catalog: the ORIGINATING source codes (ICD etc.) whose
            # conditions were attributed to this term, by any tier (source-exact /
            # standard-exact / climbed) — identity only (vocab+code+name+via), never
            # per-code patient counts. Empty for the exact-map count spaces.
            "source_codes": list(r.get("source_codes") or []),
            "n_source_codes": int(r.get("n_source_codes") or len(r.get("source_codes") or [])),
        })

    depth = _depths(parents_of)
    used = {mid for mid, st in state_of.items() if st != "unused"}
    on_used_path = _used_path_set(parents_of, used)
    cat_counts = {"reported": 0, "used_small": 0, "used_branch": 0, "other": 0}
    for nd in nodes:
        nd["depth"] = depth.get(nd["id"], 1)
        if nd["state"] == "reported":
            cat = "reported"
        elif nd["state"] == "used_small":
            cat = "used_small"
        elif nd["id"] in on_used_path:
            cat = "used_branch"
        else:
            cat = "other"
        nd["category"] = cat
        cat_counts[cat] += 1

    n_terms = len(term_rows)
    n_used = counts["used_small"] + counts["reported"]
    stats = {
        "mapped_terms": n_terms,
        "used_terms": n_used,
        "used_small_terms": counts["used_small"],
        "reported_terms": counts["reported"],
        "unused_terms": counts["unused"],
        "used_branch_terms": cat_counts["used_branch"],
        "other_terms": cat_counts["other"],
        "used_fraction": (n_used / n_terms) if n_terms else 0.0,
        "internal_terms": sum(1 for r in term_rows if r["is_internal"]),
        "internal_used_terms": n_internal_used,
        "collision_terms": n_collision,
        "rare_terms": sum(1 for r in term_rows if r.get("rare")),
        "rare_used_terms": sum(1 for r, nd in zip(term_rows, nodes)
                               if r.get("rare") and nd["state"] != "unused"),
        "total_codes": sum(nd["n_codes"] for nd in nodes),
        "multi_code_terms": sum(1 for nd in nodes if nd["n_codes"] > 1),
        "max_depth": max(depth.values()) if depth else 0,
    }
    root = {"id": _ROOT_ID, "label": "Mondo disease (mapped-term view)",
            "kind": "root", "parents": [], "std_concepts": [], "state": "root",
            "category": "root", "display": "", "count": None, "collision": False,
            "collision_siblings": [], "rare": False, "rare_src": [],
            "codes": [], "n_codes": 0, "codes_by_vocab": {},
            "source_codes": [], "n_source_codes": 0, "depth": 0}
    return {"meta": meta, "stats": stats, "nodes": [root] + nodes}


def format_summary(stats: dict, *, min_cell: int = _MIN_CELL) -> str:
    """Human-readable headline block (stderr). Term counts are exact; the one
    patient-derived line (persons-on-Mondo) is suppressed by the caller."""
    s = stats
    pct = 100.0 * s["used_fraction"]
    return "\n".join([
        "=" * 74,
        "WHOLE-MONDO EHR USAGE (exact map, no roll-up)",
        f"  mapped Mondo disease terms:        {s['mapped_terms']:>8}",
        f"  used (>=1 patient):                {s['used_terms']:>8}   {pct:5.1f}%",
        f"    of which used-small (≤{min_cell}):        {s['used_small_terms']:>8}   "
        f"(kept & flagged, exact count withheld)",
        f"    of which reported (>{min_cell}):          {s['reported_terms']:>8}",
        f"  unused mapped terms (0 patients):  {s['unused_terms']:>8}",
        f"    of which used-branch (0 count, above a used node): {s['used_branch_terms']:>8}",
        f"    of which other (rest of Mondo):  {s['other_terms']:>8}",
        f"  internal (non-leaf) terms:         {s['internal_terms']:>8}   "
        f"({s['internal_used_terms']} used) <- mid-level, un-rolled",
        f"  collision-flagged terms (shared concept): {s['collision_terms']:>8}",
        f"  rare-disease terms (GARD/Orphanet/NORD): {s.get('rare_terms', 0):>8}   "
        f"({s.get('rare_used_terms', 0)} used in the EHR)",
        f"  source/target codes mapped (total): {s.get('total_codes', 0):>8}   "
        f"({s.get('multi_code_terms', 0)} terms map >1 code)",
        f"  max mapped-term tree depth:        {s['max_depth']:>8}",
        "=" * 74,
    ])


# --------------------------------------------------------------------------- #
# Spark driver                                                                #
# --------------------------------------------------------------------------- #
def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cdr", required=True)
    p.add_argument("--billing", required=True)
    p.add_argument("--mondo-version", default="2026-06-02")
    p.add_argument("--mondo-cache-dir", default="data/mondo")
    p.add_argument("--out", required=True, help="output dir for the artifacts")
    p.add_argument("--source-table", default="condition_occurrence",
                   help="OMOP condition source (condition_occurrence or condition_era)")
    p.add_argument("--count-space", choices=("standard", "source", "source_climb"),
                   default="standard",
                   help="'standard': count condition_concept_id (via same_as->Maps to); "
                        "'source': count condition_source_concept_id against the term's own "
                        "same_as codes (no Maps to -> no decomposition/collisions, but "
                        "coverage limited to the vocabularies Mondo lists); "
                        "'source_climb': 3-tier partial roll-up — source-exact (ICD same_as), "
                        "then standard-exact (condition_concept_id), then climb SNOMED "
                        "concept_ancestor to the nearest mapped Mondo term; catalogs the "
                        "originating source code per term. All source modes require "
                        "condition_occurrence.")
    p.add_argument("--min-cell", type=int, default=_MIN_CELL)
    args = p.parse_args(argv)

    from datetime import datetime, timezone

    import pandas as pd
    from pyspark.sql import SparkSession, functions as F
    from pyspark.sql.functions import broadcast

    from charmpheno.omop.bigquery import load_omop_bigquery
    from anchor_selection_cloud import _download_cached, _read_bq
    from mondo_to_omop_mapping import (
        build_mondo_to_omop, seed_source_xrefs, _disease_child_adjacency)

    spark = SparkSession.builder.appName("mondo-usage").getOrCreate()
    min_cell = int(args.min_cell)

    # --- 1. Mondo frames + whole-Mondo -> OMOP standard-Condition mapping --------
    cache = Path(args.mondo_cache_dir)
    edges_df = pd.read_csv(_download_cached(args.mondo_version, "mondo_edges.tsv", cache),
                           sep="\t", low_memory=False)
    nodes_df = pd.read_csv(_download_cached(args.mondo_version, "mondo_nodes.tsv", cache),
                           sep="\t", low_memory=False)
    all_ids = set(nodes_df["id"])

    concept_pd = (_read_bq(spark, args.cdr, args.billing, "concept")
                  .select("concept_id", "concept_name", "vocabulary_id", "domain_id",
                          "concept_code", "standard_concept")
                  .where(F.col("vocabulary_id").isin("SNOMED", "ICD10CM", "MeSH"))
                  .toPandas())
    same_as = seed_source_xrefs(mondo_edges_df=edges_df, mondo_nodes_df=nodes_df,
                                restrict_mondo_ids=all_ids)
    src = same_as.merge(concept_pd, on=["concept_code", "vocabulary_id"], how="inner")
    source_ids = sorted({int(x) for x in src["concept_id"]})

    # code multiplicity per term (# distinct SOURCE ids by vocabulary) — from the
    # Mondo same_as xrefs; an exact code COUNT, never a patient count.
    srcu = src[["mondo_id", "concept_id", "vocabulary_id", "concept_code"]].drop_duplicates()
    codes_of, codesbyvocab_of = {}, {}
    for mid, sub in srcu.groupby("mondo_id"):
        codes_of[str(mid)] = [{"id": int(r.concept_id), "vocab": str(r.vocabulary_id),
                               "code": str(r.concept_code)} for r in sub.itertuples()]
        codesbyvocab_of[str(mid)] = {v: int(c) for v, c
                                     in sub["vocabulary_id"].value_counts().items()}

    source_codes_of: dict[str, list[dict]] = {}   # source_climb catalog (empty otherwise)
    survey: dict = {}                             # source_climb tier coverage (empty otherwise)

    if args.count_space == "source":
        # --- SOURCE space: match condition_source_concept_id to the term's OWN
        #     same_as source concepts. No Maps to => no decomposition, no cross-term
        #     collisions (same_as is source-injective). Coverage limited to the
        #     vocabularies Mondo lists (patients coded only in ICD9 etc. are missed).
        term_match = {m: sorted({c["id"] for c in cs}) for m, cs in codes_of.items()}
        pairs = [(cid, m) for m, cs in term_match.items() for cid in cs]
        id_to_mondos = collision_map(pairs)
        siblings = term_collision_siblings(term_match, id_to_mondos)
        match_pd = srcu[["mondo_id", "concept_id"]].drop_duplicates().rename(
            columns={"concept_id": "match_cid"})
        match_pd["match_cid"] = match_pd["match_cid"].astype(int)
        m_sdf = broadcast(spark.createDataFrame(match_pd))
        cond = (_read_bq(spark, args.cdr, args.billing, "condition_occurrence")
                .select("person_id",
                        F.col("condition_source_concept_id").alias("match_cid2"))
                .where(F.col("match_cid2").isNotNull() & (F.col("match_cid2") != 0))
                ).cache()
        hit = (cond.join(m_sdf, cond["match_cid2"] == m_sdf["match_cid"], "inner")
               .select("person_id", "mondo_id"))
    elif args.count_space == "source_climb":
        # --- SOURCE_CLIMB: 3-tier partial roll-up, preferring the most SPECIFIC
        #     mapped Mondo term reachable, and always cataloguing the ORIGINATING
        #     source code. Precedence (first hit wins; never climb past an exact):
        #       (1) source-exact  : condition_source_concept_id is a term same_as code
        #       (2) standard-exact: condition_concept_id (same_as -> Maps to) hits a term
        #       (3) climb         : nearest mapped ancestor of condition_concept_id in
        #                           SNOMED concept_ancestor (ties -> all, flagged)
        #     concept_ancestor is SNOMED-only (ICD is non-standard), so only the
        #     standard concept can be climbed; the ICD source rides up through it.
        from pyspark.sql import Window
        CATALOG_CAP = 60

        # (a) standard mapping: term -> standard concept(s) (drives std_concepts + climb targets)
        src_sdf = spark.createDataFrame(pd.DataFrame({"concept_id_1": source_ids}))
        cr_pd = (_read_bq(spark, args.cdr, args.billing, "concept_relationship")
                 .select("concept_id_1", "concept_id_2", "relationship_id")
                 .where(F.col("relationship_id") == "Maps to")
                 .join(broadcast(src_sdf), "concept_id_1", "inner").toPandas())
        mapping = build_mondo_to_omop(
            mondo_edges_df=edges_df, mondo_nodes_df=nodes_df,
            concept_df=concept_pd, concept_relationship_df=cr_pd, restrict_mondo_ids=None)
        ts = mapping[["mondo_id", "standard_concept_id"]].drop_duplicates()
        ts["standard_concept_id"] = ts["standard_concept_id"].astype(int)
        term_match = ts.groupby("mondo_id")["standard_concept_id"].apply(
            lambda s: sorted(set(int(x) for x in s))).to_dict()
        std_map_pd = ts.rename(columns={"standard_concept_id": "map_cid"})   # map_cid -> mondo_id
        mapped_std_ids = sorted({int(x) for x in ts["standard_concept_id"]})
        std_sdf = broadcast(spark.createDataFrame(std_map_pd))
        std_ids_sdf = broadcast(spark.createDataFrame(
            pd.DataFrame({"map_cid": mapped_std_ids})))

        # (b) source-exact mapping: term same_as source concept -> mondo_id
        srcmap_pd = srcu[["mondo_id", "concept_id"]].drop_duplicates().rename(
            columns={"concept_id": "map_cid"})
        srcmap_pd["map_cid"] = srcmap_pd["map_cid"].astype(int)
        srcmap_sdf = broadcast(spark.createDataFrame(srcmap_pd))
        src_ids_sdf = broadcast(spark.createDataFrame(
            pd.DataFrame({"src_cid": sorted({int(x) for x in srcmap_pd["map_cid"]})})))

        # condition rows: originating source (src_cid) + its standard concept (std_cid)
        cond = (_read_bq(spark, args.cdr, args.billing, "condition_occurrence")
                .select("person_id",
                        F.col("condition_source_concept_id").alias("src_cid"),
                        F.col("condition_concept_id").alias("std_cid"))
                .where(F.col("src_cid").isNotNull() | F.col("std_cid").isNotNull())
                ).cache()

        # originating identity: the ICD source code, falling back to the standard
        # concept when no usable source concept was recorded (src null or 0).
        origin = F.when(F.col("src_cid").isNotNull() & (F.col("src_cid") != 0),
                        F.col("src_cid")).otherwise(F.col("std_cid"))

        # TIER 1 — source-exact
        t1 = (cond.join(srcmap_sdf, cond["src_cid"] == srcmap_sdf["map_cid"], "inner")
              .select("person_id", "mondo_id",
                      F.col("src_cid").alias("origin_cid"), F.lit("exact").alias("via")))
        # rows whose source code is NOT a same_as (fall through to tier 2/3)
        rem1 = cond.join(src_ids_sdf, cond["src_cid"] == src_ids_sdf["src_cid"], "left_anti")

        # TIER 2 — standard-exact (origin = the ICD source code, or std if none)
        t2 = (rem1.join(std_sdf, rem1["std_cid"] == std_sdf["map_cid"], "inner")
              .select("person_id", "mondo_id",
                      origin.alias("origin_cid"), F.lit("exact").alias("via")))
        # rows whose standard concept is ALSO not a mapped term -> candidates to climb
        rem2 = rem1.join(std_ids_sdf, rem1["std_cid"] == std_ids_sdf["map_cid"], "left_anti")

        # TIER 3 — climb SNOMED concept_ancestor to the nearest mapped term(s)
        ca = (_read_bq(spark, args.cdr, args.billing, "concept_ancestor")
              .select("ancestor_concept_id", "descendant_concept_id",
                      "min_levels_of_separation")
              .where(F.col("min_levels_of_separation") >= 1))
        unmatched_std = rem2.select("std_cid").distinct()
        ca_f = (ca.join(std_ids_sdf, ca["ancestor_concept_id"] == std_ids_sdf["map_cid"], "inner")
                  .join(unmatched_std, ca["descendant_concept_id"] == unmatched_std["std_cid"], "inner")
                  .select(ca["descendant_concept_id"].alias("std_cid"),
                          ca["ancestor_concept_id"].alias("anc_cid"),
                          ca["min_levels_of_separation"].alias("lev")))
        nearest = (ca_f.withColumn(
                       "mlev", F.min("lev").over(Window.partitionBy("std_cid")))
                   .where(F.col("lev") == F.col("mlev"))
                   .select("std_cid", "anc_cid"))
        anc_map = std_sdf.select(std_sdf["map_cid"].alias("anc_cid"),
                                 std_sdf["mondo_id"].alias("anc_mondo"))
        t3 = (rem2.join(nearest, "std_cid", "inner")
              .join(broadcast(anc_map), "anc_cid", "inner")
              .select("person_id", F.col("anc_mondo").alias("mondo_id"),
                      origin.alias("origin_cid"), F.lit("climbed").alias("via")))

        attribution = (t1.unionByName(t2).unionByName(t3)).cache()
        hit = attribution.select("person_id", "mondo_id")

        # per-term catalog of ORIGINATING source codes (identity only, via = exact/climbed;
        # exact wins if a code reaches a term both ways). Named via the concept table.
        cat = (attribution.select("mondo_id", "origin_cid", "via").distinct()
               .withColumn("vrank", F.when(F.col("via") == "exact", 0).otherwise(1)))
        cat = (cat.withColumn("best", F.min("vrank").over(
                   Window.partitionBy("mondo_id", "origin_cid")))
               .where(F.col("vrank") == F.col("best"))
               .select("mondo_id", "origin_cid", "via").distinct())
        concept_all = (_read_bq(spark, args.cdr, args.billing, "concept")
                       .select(F.col("concept_id").alias("origin_cid"),
                               "vocabulary_id", "concept_code"))
        cat_named = (cat.join(concept_all, "origin_cid", "left")
                     .select("mondo_id", "origin_cid", "via", "vocabulary_id", "concept_code")
                     .toPandas())
        for mid, sub in cat_named.groupby("mondo_id"):
            recs = [{"id": int(r.origin_cid),
                     "vocab": (str(r.vocabulary_id) if pd.notna(r.vocabulary_id) else "?"),
                     "code": (str(r.concept_code) if pd.notna(r.concept_code) else str(int(r.origin_cid))),
                     "via": str(r.via)}
                    for r in sub.itertuples()]
            recs.sort(key=lambda c: (c["via"] != "exact", c["vocab"], c["code"]))
            source_codes_of[str(mid)] = {"list": recs[:CATALOG_CAP], "n": len(recs)}

        # collisions: a single source code attributed to >1 Mondo term (standard-exact
        # coarsening or a climb tie) -> a patient with that code counts under each.
        pairs_pd = cat_named[["origin_cid", "mondo_id"]].drop_duplicates()
        id_to_mondos = collision_map(
            [(int(c), str(m)) for c, m in pairs_pd.itertuples(index=False)])
        term_origins = (cat_named.groupby("mondo_id")["origin_cid"]
                        .apply(lambda s: [int(x) for x in s]).to_dict())
        siblings = term_collision_siblings(
            {str(k): v for k, v in term_origins.items()}, id_to_mondos)

        # survey: distinct persons resolved at each tier (overlapping across tiers) and
        # the unmatched remainder by originating-source vocabulary.
        unmatched = rem2.join(nearest.select("std_cid").distinct(), "std_cid", "left_anti")
        src_vocab = concept_all.select(F.col("origin_cid").alias("src_cid"), "vocabulary_id")
        unm_by_vocab = (unmatched.join(src_vocab, "src_cid", "left")
                        .groupBy("vocabulary_id")
                        .agg(F.countDistinct("person_id").alias("persons"))
                        .toPandas())
        survey = {
            "persons_source_exact": int(t1.select("person_id").distinct().count()),
            "persons_standard_exact": int(t2.select("person_id").distinct().count()),
            "persons_climbed": int(t3.select("person_id").distinct().count()),
            "persons_unmatched_by_vocab": {
                (str(r.vocabulary_id) if pd.notna(r.vocabulary_id) else "?"): int(r.persons)
                for r in unm_by_vocab.itertuples()},
        }
    else:
        # --- STANDARD space (default): condition_concept_id via same_as -> Maps to.
        src_sdf = spark.createDataFrame(pd.DataFrame({"concept_id_1": source_ids}))
        cr_pd = (_read_bq(spark, args.cdr, args.billing, "concept_relationship")
                 .select("concept_id_1", "concept_id_2", "relationship_id")
                 .where(F.col("relationship_id") == "Maps to")
                 .join(broadcast(src_sdf), "concept_id_1", "inner").toPandas())
        mapping = build_mondo_to_omop(
            mondo_edges_df=edges_df, mondo_nodes_df=nodes_df,
            concept_df=concept_pd, concept_relationship_df=cr_pd, restrict_mondo_ids=None)
        ts = mapping[["mondo_id", "standard_concept_id"]].drop_duplicates()
        ts["standard_concept_id"] = ts["standard_concept_id"].astype(int)
        term_match = ts.groupby("mondo_id")["standard_concept_id"].apply(
            lambda s: sorted(set(int(x) for x in s))).to_dict()
        std_to_mondos = collision_map(zip(ts["standard_concept_id"], ts["mondo_id"]))
        siblings = term_collision_siblings(term_match, std_to_mondos)
        ts_sdf = broadcast(spark.createDataFrame(
            ts.rename(columns={"standard_concept_id": "match_cid"})))
        cond = load_omop_bigquery(
            spark=spark, cdr_dataset=args.cdr, billing_project=args.billing,
            source_table=args.source_table).select("person_id", "concept_id").cache()
        hit = (cond.join(ts_sdf, cond["concept_id"] == ts_sdf["match_cid"], "inner")
               .select("person_id", "mondo_id"))

    # --- 2. EXACT-match person counts per term (NO concept_ancestor climb) --------
    term_counts = (hit.groupBy("mondo_id")
                   .agg(F.countDistinct("person_id").alias("n")).toPandas())
    count_of = {str(r["mondo_id"]): int(r["n"]) for _, r in term_counts.iterrows()}

    # placement ladder (persons-on-Mondo is the one suppressed patient figure).
    n_total = (_read_bq(spark, args.cdr, args.billing, "person")
               .select("person_id").distinct().count())
    n_coded = cond.select("person_id").distinct().count()
    n_on_mondo = hit.select("person_id").distinct().count()

    # --- 3. hierarchy structure over mapped terms (nearest mapped ancestor) -------
    child_adj = _disease_child_adjacency(edges_df, nodes_df)      # parent -> [children]
    disease_set = set(child_adj) | {c for ch in child_adj.values() for c in ch}
    has_child = {p for p, ch in child_adj.items()
                 if any(c in disease_set for c in ch)}
    parent_adj: dict[str, list[str]] = {}
    for parent, children in child_adj.items():
        for c in children:
            parent_adj.setdefault(c, []).append(parent)

    # term universe: the mapped terms, plus (source_climb) any term that received a
    # count or catalogued source code even if its standard mapping was empty.
    mapped_ids = set(term_match) | set(count_of) | set(source_codes_of)
    parents = nearest_mapped_parents(mapped_ids, parent_adj)
    label_of = {str(i): str(n) for i, n in zip(nodes_df["id"], nodes_df["name"])}

    # Mondo rare-disease designations, parsed straight from the Mondo `subsets`
    # field (count-space independent): per term, rare? + which source registries.
    rare_of, rare_src_of = rare_from_nodes(nodes_df)

    term_rows = []
    for mid in sorted(mapped_ids):
        cids = sorted({int(c) for c in term_match.get(mid, [])})
        codes = codes_of.get(mid, [])
        cat = source_codes_of.get(mid) or {"list": [], "n": 0}
        term_rows.append({
            "mondo_id": mid,
            "label": label_of.get(mid, mid),
            "is_internal": mid in has_child,
            "parents": parents.get(mid, []),
            "std_concepts": cids,
            "codes": codes,
            "n_codes": len(codes),
            "codes_by_vocab": codesbyvocab_of.get(mid, {}),
            "source_codes": cat["list"],
            "n_source_codes": cat["n"],
            "n_persons": count_of.get(mid, 0),
            "collision_siblings": siblings.get(mid, []),
            "rare": rare_of.get(mid, False),
            "rare_src": rare_src_of.get(mid, []),
        })

    # --- 4. assemble + persist ----------------------------------------------------
    meta = {
        "mondo_version": args.mondo_version,
        "cdr": args.cdr,
        "source_table": args.source_table,
        "min_cell": min_cell,
        "rollup": False,
        "count_space": args.count_space,
        "count_rule": {
            "source": ("distinct persons whose condition_source_concept_id exactly "
                       "matches one of the term's Mondo same_as source codes (no Maps to)"),
            "source_climb": ("distinct persons attributed to the most specific mapped "
                             "Mondo term reachable: source-exact (same_as), else "
                             "standard-exact (condition_concept_id), else nearest mapped "
                             "SNOMED ancestor via concept_ancestor (ties counted in each)"),
        }.get(args.count_space,
              "distinct persons with an EXACT-match standard condition concept "
              "(same_as -> Maps to)"),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if survey:
        meta["survey"] = survey
    payload = assemble_payload(meta=meta, term_rows=term_rows, min_cell=min_cell)

    # persons-on-Mondo ladder line (suppressed) for the log.
    def _sup(n):
        return f"≤{min_cell}" if 0 < n <= min_cell else str(int(n))
    sys.stderr.write(format_summary(payload["stats"], min_cell=min_cell) + "\n")
    _attr = "attributed" if args.count_space == "source_climb" else "exact"
    sys.stderr.write(
        f"[ladder] persons total {n_total} | coded {_sup(n_coded)} | on any mapped "
        f"Mondo term ({_attr}) {_sup(n_on_mondo)} "
        f"({100.0 * n_on_mondo / max(n_total, 1):.1f}% of all persons)\n")
    if survey:
        sv = survey
        sys.stderr.write(
            f"[source_climb survey] persons by tier (overlapping): "
            f"source-exact {_sup(sv['persons_source_exact'])} | "
            f"standard-exact {_sup(sv['persons_standard_exact'])} | "
            f"climbed {_sup(sv['persons_climbed'])}\n"
            f"[source_climb survey] unmatched persons by source vocabulary: " +
            ", ".join(f"{v}={_sup(n)}" for v, n
                      in sorted(sv["persons_unmatched_by_vocab"].items(),
                                key=lambda kv: -kv[1])) + "\n")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "mondo_usage.json").write_text(json.dumps(payload))
    rows = [{
        "mondo_id": nd["id"], "label": nd["label"], "kind": nd["kind"],
        "depth": nd["depth"], "state": nd["state"], "category": nd["category"],
        "n_patients": nd["display"],
        "collision": int(nd["collision"]),
        "collision_siblings": "|".join(nd["collision_siblings"]),
        "n_codes": nd["n_codes"],
        "codes_by_vocab": ";".join(f"{v}:{c}" for v, c in nd["codes_by_vocab"].items()),
        "codes": "|".join(f"{c['vocab']}:{c['code']}" for c in nd["codes"]),
        "n_source_codes": nd.get("n_source_codes", 0),
        "source_codes": "|".join(f"{c['vocab']}:{c['code']}:{c['via']}"
                                 for c in nd.get("source_codes", [])),
        "rare": int(nd["rare"]), "rare_src": "|".join(nd["rare_src"]),
        "parents": "|".join(nd["parents"]),
    } for nd in payload["nodes"] if nd["kind"] != "root"]
    pd.DataFrame(rows).to_csv(out / "mondo_usage_nodes.tsv", sep="\t", index=False)
    sys.stderr.write(
        f"[done] wrote mondo_usage.json ({len(payload['nodes'])} nodes) + "
        f"mondo_usage_nodes.tsv to {out}\n")
    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

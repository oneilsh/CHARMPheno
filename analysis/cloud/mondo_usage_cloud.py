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
    ``concept_ancestor`` (a SNOMED-distance tie is reduced to its MOST-SPECIFIC Mondo
    term(s) — nested ancestors dropped; a genuine orthogonal tie is counted in each and
    flagged as a collision). Because
    ``concept_ancestor`` carries subclass edges only for STANDARD concepts, only the
    SNOMED concept is climbed (ICD is non-standard); the ICD source rides up through
    whatever standard concept OMOP assigned. Recovers coverage ``source`` drops without
    the blanket ``Maps to`` decomposition of ``standard``. Emits a per-vocabulary
    coverage survey (persons exact/climbed/unmatched). Requires ``condition_occurrence``.
  * ``all``: run all three spaces in ONE Spark session (shared Mondo/DAG/rare structure);
    write ``mondo_usage_<space>.json`` for each, a primary ``mondo_usage.json`` (=
    source_climb, the dashboard default), per-space ``_nodes.tsv``, and a disclosure-SAFE
    ``mondo_usage_summary.md`` (term counts + ≤-suppressed person figures, no CDR id) for
    a side-by-side comparison. See exp 0108.

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
import shutil
import sys
import urllib.request
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


def classify_collision_kinds(pairs):
    """Split source_climb cross-term collisions by MECHANISM (code-centric).

    ``pairs`` are ``(origin_code, mondo_id, via)`` attribution triples with ``via`` in
    {``source_exact``, ``standard_exact``, ``climbed``}. A term collides when it shares an
    originating source code with another term; the mechanism is a property of the CODE:

      * ``shared_concept`` — the code reaches its several terms only through EXACT maps:
        one standard (SNOMED) concept is the mapping of >=2 Mondo terms (the OMOP
        ``Maps to`` coarsening) — the "these are genuinely the same concept" case, and the
        same phenomenon as standard-space collisions.
      * ``climb_tie`` — the code reaches its terms only through the CLIMB: it rolled up to
        >=2 equally-near mapped ancestors in Mondo's poly-hierarchy. Milder — a specific
        code sitting under two sibling branches (expected in a DAG).
      * ``mixed`` — both mechanisms contribute (e.g. exact to one term, climbed to another).

    Returns ``(siblings, term_kind, code_kind)``: ``siblings[mondo]`` = sorted colliding
    siblings (any mechanism — the "don't sum" set); ``code_kind[code]`` for each code that
    hits >=2 terms; ``term_kind[mondo]`` in {shared_concept, climb_tie, mixed} for each
    colliding term (mixed if its colliding codes disagree)."""
    from collections import defaultdict
    code_terms: dict[int, set] = defaultdict(set)
    code_vias: dict[int, set] = defaultdict(set)
    for c, m, v in pairs:
        code_terms[int(c)].add(str(m))
        code_vias[int(c)].add(str(v))
    code_kind: dict[int, str] = {}
    for c, terms in code_terms.items():
        if len(terms) < 2:
            continue
        vs = code_vias[c]
        code_kind[c] = ("climb_tie" if vs == {"climbed"}
                        else "shared_concept" if not (vs & {"climbed"})
                        else "mixed")
    sib: dict[str, set] = defaultdict(set)
    tk: dict[str, set] = defaultdict(set)
    for c, k in code_kind.items():
        terms = code_terms[c]
        for m in terms:
            sib[m] |= (terms - {m})
            tk[m].add(k)
    siblings = {m: sorted(s) for m, s in sib.items()}
    term_kind = {m: (ks.pop() if len(ks) == 1 else "mixed") for m, ks in
                 ({m: set(v) for m, v in tk.items()}).items()}
    return siblings, term_kind, code_kind


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


def meaningful_skeleton(seed_ids, parent_adj):
    """Terms to KEEP so the ontology reads with its real branch structure: every attributed
    ('seed') term, plus every un-attributed ancestor that is a genuine BRANCH POINT (>= 2 of
    its children lead to a seed term). Linear pass-through ancestors (0 or 1 seed-bearing
    child) are dropped, so their children reattach to the nearest kept ancestor. ``parent_adj``
    is child -> [parents] over the full DAG. Pure."""
    seed = set(seed_ids)
    closure = set(seed); stack = list(seed)            # ancestor closure of the seed
    while stack:
        for p in parent_adj.get(stack.pop(), []):
            if p not in closure:
                closure.add(p); stack.append(p)
    child_adj = {}                                      # child edges within the closure
    for c in closure:
        for p in parent_adj.get(c, []):
            if p in closure:
                child_adj.setdefault(p, []).append(c)
    reaches = {}                                        # subtree contains a seed term? (memoized)
    def reaches_seed(x):
        if x in reaches: return reaches[x]
        reaches[x] = False                              # cycle guard (ontologies are acyclic)
        r = x in seed or any(reaches_seed(c) for c in child_adj.get(x, []))
        reaches[x] = r; return r
    for x in closure: reaches_seed(x)
    keep = set(seed)
    for x in closure:
        if x not in seed and sum(1 for c in child_adj.get(x, []) if reaches[c]) >= 2:
            keep.add(x)                                 # a real branch point
    return keep


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


def reduce_tie_map(pairs, parents_of):
    """Reduce each climb tie-set to its MOST-SPECIFIC members in the Mondo DAG.

    The climb attributes a code to the nearest mapped ancestor(s) by SNOMED distance, but
    ties can include a specific term AND its more-general Mondo ancestors (e.g. a code that
    ties to {carbuncle, pyoderma, skin disorder}). The ladder's promise is the MOST SPECIFIC
    mapped term, so among tied terms we drop any that is a Mondo-ancestor of another tied
    term — keeping only the "leaves" of the set. Genuine orthogonal ties (neither term an
    ancestor of the other, e.g. {hereditary disease, endocrine system disorder}) are kept.

    ``pairs`` = iterable of ``(key, mondo_id)`` (key = the climbed descendant concept);
    ``parents_of`` = Mondo child->parents adjacency. Returns ``{key: [most-specific ids]}``.
    A shared ancestor-closure memo makes this cheap across many overlapping tie-sets."""
    from collections import defaultdict
    groups: dict = defaultdict(set)
    for k, t in pairs:
        groups[k].add(str(t))
    memo: dict[str, set] = {}

    def anc(t):
        if t in memo:
            return memo[t]
        out: set = set()
        stack = list(parents_of.get(t, ()))
        while stack:
            p = stack.pop()
            if p in out:
                continue
            out.add(p)
            stack.extend(parents_of.get(p, ()))
        memo[t] = out
        return out

    out: dict = {}
    for k, terms in groups.items():
        if len(terms) <= 1:
            out[k] = sorted(terms)
            continue
        out[k] = sorted(t for t in terms
                        if not any(t in anc(s) for s in terms if s != t))
    return out


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


def suppress_count(n, min_cell: int = _MIN_CELL) -> str:
    """AoU small-cell display for a patient count: ``≤{floor}`` when 1..floor,
    else the exact integer. ``None``/missing -> ``"n/a"``. Used for every
    patient-derived figure that leaves the cluster (the safe summary), so nothing
    at or below the floor is ever emitted."""
    if n is None:
        return "n/a"
    n = int(n)
    if n <= 0:
        return "0"
    return f"≤{min_cell}" if n <= min_cell else str(n)


# Log-decade volume bands for per-source-code magnitude. A code's exact person
# count NEVER leaves the cluster (differencing risk vs the per-term union total);
# instead each code is tagged with the RANGE it falls in. Ranges are
# differencing-proof by construction — the difference of two ranges is a range,
# never a person — so bands compose safely with the exact per-term totals, across
# count-spaces, and up/down the roll-up. The bottom band is the small-cell floor
# itself, so a rare code reads "≤{floor}" exactly as a suppressed cell does.
# Order is heavy -> light (canonical display order).
_BAND_UPPERS = [100, 1_000, 10_000, 100_000]     # decade edges above the floor

# The export carries code IDENTITIES only (vocab + code), never concept names: name text
# would be egress from the workbench, and SNOMED/CPT4/etc. names are licensed. The
# dashboard resolves human-readable names client-side for the public-domain vocabularies
# (ICD-9-CM, ICD-10-CM, HCPCS) from a public terminology service (NIH Clinical Tables);
# licensed vocabularies keep code-only display + an Athena click-through, covered by a
# SNOMED CT sub-license disclaimer in the UI.


def _band_label(min_cell: int) -> "list[str]":
    """Canonical band labels, heaviest first, for a given small-cell floor."""
    lo = [f"≤{min_cell}", f"{min_cell + 1}–100", "101–1k", "1k–10k", "10k–100k", ">100k"]
    return list(reversed(lo))


def volume_band(n, min_cell: int = _MIN_CELL) -> str:
    """Map a per-code patient count to a log-decade band label (a RANGE, never the
    number). ``None`` -> ``"n/a"``, ``<=0`` -> ``"0"``, ``1..floor`` -> ``"≤{floor}"``
    (identical to a suppressed cell), then ``21–100 / 101–1k / 1k–10k / 10k–100k /
    >100k`` for ``floor == 20``. Pure; disclosure-safe (emits no exact patient count)."""
    if n is None:
        return "n/a"
    n = int(n)
    if n <= 0:
        return "0"
    if n <= min_cell:
        return f"≤{min_cell}"
    labels = [f"{min_cell + 1}–100", "101–1k", "1k–10k", "10k–100k"]
    for up, lab in zip(_BAND_UPPERS, labels):
        if n <= up:
            return lab
    return ">100k"


def band_histogram(bands, min_cell: int = _MIN_CELL) -> "list[dict]":
    """Given the band label of every source code on a term (the FULL set, before any
    display cap), return the distribution as ``[{"band": label, "codes": k}, ...]`` in
    canonical heavy->light order, omitting empty bands. The ``codes`` figure counts
    CODES (public non-personal identities), never persons — so it is safe at any value
    (no small-cell suppression applies to a code count). This is the "where's the
    weight" shape: e.g. ``>100k:2 · ≤20:184`` reads as Mondo-thin-but-heavy. Pure."""
    from collections import Counter
    c = Counter(bands)
    out = []
    for lab in _band_label(min_cell):
        if c.get(lab):
            out.append({"band": lab, "codes": int(c[lab])})
    return out


# HPO (Human Phenotype Ontology) cross-reference vocab prefixes -> the OMOP vocabulary_id
# they correspond to. HPO's `xref:` values look like ``SNOMEDCT_US:190855004`` /
# ``UMLS:C0151723`` / ``ICD-10:E83.42``. SNOMED is the bridge that matters most (it is
# OMOP's standard Condition vocabulary); ICD lets a source code match directly. UMLS has no
# native OMOP key, so it is carried but not matched here.
_XREF_VOCAB = {
    "SNOMEDCT_US": "SNOMED", "SNOMEDCT": "SNOMED", "SNOMED_CT": "SNOMED", "SCTID": "SNOMED",
    "ICD10": "ICD10CM", "ICD-10": "ICD10CM", "ICD10CM": "ICD10CM", "ICD-10-CM": "ICD10CM",
    "ICD9": "ICD9CM", "ICD-9": "ICD9CM", "ICD9CM": "ICD9CM", "ICD-9-CM": "ICD9CM",
    "MSH": "MeSH", "MESH": "MeSH", "UMLS": "UMLS",
}


def normalize_xref_vocab(prefix: str):
    """Map an ontology xref prefix (e.g. ``SNOMEDCT_US``) to the OMOP ``vocabulary_id``
    (``SNOMED``), or ``None`` when it is not one we match on. Pure."""
    return _XREF_VOCAB.get(str(prefix).strip().upper())


def parse_hpo_xrefs(obo_text: str) -> "list[tuple]":
    """Parse an HPO ``hp.obo`` into ``(hp_id, hp_label, vocab, code)`` rows for the xref
    vocabularies we recognise (see ``_XREF_VOCAB``). One row per (term, mapped xref); a
    term with several xrefs yields several rows. Trailing OBO qualifiers/comments on an
    xref line (``{source=...}`` or `` ! label``) are stripped. Pure — no I/O."""
    rows = []
    hp_id = hp_name = None
    in_term = False
    for raw in obo_text.splitlines():
        line = raw.rstrip()
        if line == "[Term]":
            in_term, hp_id, hp_name = True, None, None
            continue
        if line.startswith("[") and line.endswith("]"):   # a different stanza (Typedef, ...)
            in_term = False
            continue
        if not in_term:
            continue
        if line.startswith("id:"):
            hp_id = line[3:].strip()
        elif line.startswith("name:"):
            hp_name = line[5:].strip()
        elif line.startswith("xref:") and hp_id and hp_id.startswith("HP:"):
            x = line[5:].strip().split("{")[0].split(" ! ")[0].strip()
            if ":" not in x:
                continue
            prefix, code = x.split(":", 1)
            vocab = normalize_xref_vocab(prefix)
            if vocab:
                rows.append((hp_id, hp_name, vocab, code.strip()))
    return rows


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


def dag_structures(parents: dict) -> "tuple[dict, set]":
    """From a child->[parents] map, return ``(parent_adj, has_child)`` restricted to known
    ids: ``parent_adj`` drops parents not in the map; ``has_child`` is every id that is a
    parent of a known id. Pure. Shared by the Mondo and HPO axes for the DAG browse/roll-up."""
    known = set(parents)
    parent_adj = {c: [p for p in ps if p in known] for c, ps in parents.items()}
    has_child = {p for ps in parent_adj.values() for p in ps}
    return parent_adj, has_child


def build_safe_summary(results: list[dict]) -> str:
    """A copy-pasteable, disclosure-SAFE summary of one or more count-space runs.

    ``results`` is one dict per space: ``space``, ``stats`` (the payload stats block),
    ``survey`` (source_climb tier coverage, or empty), ``n_total/n_coded/n_on_mondo``
    (raw person counts), ``min_cell``, ``mondo_version``, ``generated_utc``, and
    optionally ``hpo_axis`` (used/reported HPO term counts + distinct persons on the
    HPO attribution, attached to the source_climb result when ``--with-hpo`` is on).

    SAFE by construction: it prints only term COUNTS (terms are not patients) and
    aggregate fractions, and every patient-derived figure is run through
    ``suppress_count`` (≤floor). It deliberately never includes the workbench/CDR id or
    any per-term patient number — so it can be pasted anywhere. Pure and time-free."""
    if not results:
        return "# Mondo EHR usage — no results\n"
    min_cell = results[0].get("min_cell", _MIN_CELL)
    mv = results[0].get("mondo_version", "?")
    gen = results[0].get("generated_utc", "?")
    L = ["# Mondo EHR usage — count-space comparison",
         "",
         f"- Mondo version: **{mv}**  ·  generated: {gen}",
         f"- Source: All of Us EHR (aggregated; counts of ≤{min_cell} suppressed as "
         f"`≤{min_cell}`, never shown exactly)",
         "- All figures below are term COUNTS (not patients) or ≤-suppressed person "
         "counts; nothing here is a per-term patient number or a CDR identifier.",
         "",
         "| count space | mapped terms | used | used % | reported (>{c}) | used-small (≤{c}) | collision-flagged | rare used |"
         .format(c=min_cell),
         "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for r in results:
        s = r["stats"]
        L.append("| `{sp}` | {mapped} | {used} | {pct:.1f}% | {rep} | {sm} | {col} | {rare} |".format(
            sp=r["space"], mapped=s["mapped_terms"], used=s["used_terms"],
            pct=100.0 * s.get("used_fraction", 0.0), rep=s["reported_terms"],
            sm=s["used_small_terms"], col=s["collision_terms"], rare=s.get("rare_used_terms", 0)))
    L += ["", "## Person coverage (≤-suppressed)", ""]
    for r in results:
        nc, nm = r.get("n_coded"), r.get("n_on_mondo")
        uncov = (int(nc) - int(nm)) if (nc is not None and nm is not None) else None
        upct = (f" ({100.0 * uncov / nc:.1f}%)" if uncov is not None and nc else "")
        L.append(f"- `{r['space']}` — total persons {suppress_count(r.get('n_total'), min_cell)}"
                 f" · coded {suppress_count(nc, min_cell)}"
                 f" · on any mapped Mondo term {suppress_count(nm, min_cell)}"
                 f" · **no mapped term {suppress_count(uncov, min_cell)}{upct}**")
    climb = next((r for r in results if r.get("survey")), None)
    if climb:
        sv = climb["survey"]
        L += ["", "## source_climb attribution survey", "",
              f"- persons resolved by tier (≤-suppressed; overlapping — a person can be "
              f"counted in several tiers via different conditions): source-exact "
              f"{suppress_count(sv.get('persons_source_exact'), min_cell)} · standard-exact "
              f"{suppress_count(sv.get('persons_standard_exact'), min_cell)} · climbed "
              f"{suppress_count(sv.get('persons_climbed'), min_cell)}",
              "- unmatched **source codes** by vocabulary (distinct code counts, not "
              "patients — the vocabularies the SNOMED climb can't reach): " +
              (", ".join(f"{v}={n}" for v, n in sorted(
                  sv.get("unmatched_codes_by_vocab", {}).items(),
                  key=lambda kv: -(kv[1] or 0))) or "(none)")]
        tk = sv.get("collision_terms_by_kind", {})
        ck = sv.get("collision_codes_by_kind", {})
        if tk or ck:
            L += ["", "### collision split (of the flagged terms)", "",
                  f"- flagged **terms** by mechanism: shared-concept {tk.get('shared_concept', 0)} "
                  f"(genuine Maps-to coarsening, = standard space) · climb-tie "
                  f"{tk.get('climb_tie', 0)} (rolled up to ≥2 nearest ancestors) · mixed "
                  f"{tk.get('mixed', 0)}",
                  f"- colliding **codes** by mechanism: shared-concept {ck.get('shared_concept', 0)} "
                  f"· climb-tie {ck.get('climb_tie', 0)} · mixed {ck.get('mixed', 0)}"]
            for title, key in [("climb-tie", "collision_examples_climb_tie"),
                               ("shared-concept", "collision_examples_shared_concept")]:
                exs = sv.get(key, [])
                if exs:
                    L += ["", f"Example {title} multi-maps (one source code → the Mondo "
                          "terms it lands on — judge whether the overlap makes sense):"]
                    L += [f"- `{e['code']}` → {', '.join(e['terms'])}" for e in exs]
        hp = sv.get("hpo") or {}
        if hp:
            def _cov(d):
                c, k = d.get("concepts", 0), d.get("with_hpo", 0)
                pct = f" ({100.0 * k / c:.0f}%)" if c else ""
                return (f"{k} of {c} concepts{pct} · person-mass "
                        f"{suppress_count(d.get('mass_hpo'), min_cell)} of "
                        f"{suppress_count(d.get('mass'), min_cell)}")
            L += ["", "### HPO phenotype-gap probe", "",
                  "How much EHR signal that Mondo can only CLIMB to a general term, or can't "
                  "map at all (DROP), has an EXACT HPO term instead. Concept counts are "
                  "identities; person-mass = sum over concepts of distinct persons "
                  "(≤-suppressed; not distinct persons), sizing the phenotype axis Mondo "
                  f"doesn't cover. HPO SNOMED-xref'd terms loaded: {hp.get('hpo_snomed_terms', 0)}.",
                  f"- **climbed** standard concepts recoverable by HPO: {_cov(hp.get('climb', {}))}",
                  f"- **dropped** standard concepts recoverable by HPO: {_cov(hp.get('drop', {}))}"]
            exs = hp.get("examples", [])
            if exs:
                L += ["", "Example climbed concepts HPO would place exactly (SNOMED code → HP "
                      "term → what it currently climbs to in Mondo):"]
                L += [f"- `SNOMED {e['snomed']}` → {e['hp_id']} {e['hp_label']}"
                      + (f"  (climbs to: {', '.join(e['climbs_to'])})" if e.get("climbs_to") else "")
                      for e in exs]
    hx = next((r["hpo_axis"] for r in results if r.get("hpo_axis")), None)
    if hx:
        L += ["", "## HPO axis (phenotypes)", "",
              f"- HPO terms used in the EHR: {hx.get('used_terms', 0)} "
              f"(reported >{min_cell}: {hx.get('reported_terms', 0)}) · persons "
              f"{suppress_count(hx.get('persons'), min_cell)}"]
    L.append("")
    return "\n".join(L)


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
    collision_kind_counts = {"shared_concept": 0, "climb_tie": 0, "mixed": 0}

    for r in term_rows:
        state, display, public = usage_state(r["n_persons"], min_cell)
        # fractional (1/m) count — suppressed the same way (it is patient-derived and can
        # dip below the floor even when the exact count clears it, so it needs its own gate).
        frac_val = r.get("n_frac")
        frac_val = r["n_persons"] if frac_val is None else frac_val
        _fst, frac_display, frac_public = usage_state(int(round(frac_val)), min_cell)
        counts[state] += 1
        state_of[r["mondo_id"]] = state
        parents = r["parents"] or [_ROOT_ID]
        parents_of[r["mondo_id"]] = parents
        siblings = r.get("collision_siblings") or []
        # collision mechanism: source_climb sets it explicitly; other spaces' collisions
        # are all shared-concept (Maps-to coarsening), so default to that when flagged.
        ckind = r.get("collision_kind") or ("shared_concept" if siblings else "")
        if siblings:
            n_collision += 1
            if ckind in collision_kind_counts:
                collision_kind_counts[ckind] += 1
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
            "frac": frac_public,             # de-double-counted addable count (None if withheld)
            "frac_display": frac_display,
            "collision": bool(siblings),
            "collision_siblings": siblings,
            "collision_kind": ckind,
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
            # per-term magnitude histogram over the source codes: how many CODES fall in
            # each volume band (a count of codes, never patients — safe at any value).
            # The "where's the weight" shape; empty for the exact-map count spaces.
            "source_bands": list(r.get("source_bands") or []),
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
        "collision_terms_by_kind": collision_kind_counts,
        "rare_terms": sum(1 for r in term_rows if r.get("rare")),
        "rare_used_terms": sum(1 for r, nd in zip(term_rows, nodes)
                               if r.get("rare") and nd["state"] != "unused"),
        "total_codes": sum(nd["n_codes"] for nd in nodes),
        "multi_code_terms": sum(1 for nd in nodes if nd["n_codes"] > 1),
        "max_depth": max(depth.values()) if depth else 0,
    }
    root = {"id": _ROOT_ID, "label": "Mondo disease (mapped-term view)",
            "kind": "root", "parents": [], "std_concepts": [], "state": "root",
            "category": "root", "display": "", "count": None,
            "frac": None, "frac_display": "", "collision": False,
            "collision_siblings": [], "collision_kind": "", "rare": False, "rare_src": [],
            "codes": [], "n_codes": 0, "codes_by_vocab": {},
            "source_codes": [], "n_source_codes": 0, "source_bands": [], "depth": 0}
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
    p.add_argument("--count-space",
                   choices=("standard", "source", "source_climb", "all"),
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
    p.add_argument("--hpo-obo-url",
                   default="http://purl.obolibrary.org/obo/hp.obo",
                   help="HPO hp.obo URL for the phenotype-gap probe (source_climb only): "
                        "how many EHR codes that CLIMB or DROP in Mondo have an exact HPO "
                        "term. Set to '' to skip the probe.")
    p.add_argument("--with-hpo", action="store_true",
                   help="build the HPO phenotype axis (routing rung + hpo_usage.json); "
                        "default off keeps source_climb Mondo-only")
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

    # ---- space-independent structure (computed once, reused for every count space) ----
    n_total = (_read_bq(spark, args.cdr, args.billing, "person")
               .select("person_id").distinct().count())
    child_adj = _disease_child_adjacency(edges_df, nodes_df)      # parent -> [children]
    disease_set = set(child_adj) | {c for ch in child_adj.values() for c in ch}
    has_child = {p for p, ch in child_adj.items()
                 if any(c in disease_set for c in ch)}
    parent_adj: dict[str, list[str]] = {}
    for parent, children in child_adj.items():
        for c in children:
            parent_adj.setdefault(c, []).append(parent)
    label_of = {str(i): str(n) for i, n in zip(nodes_df["id"], nodes_df["name"])}
    rare_of, rare_src_of = rare_from_nodes(nodes_df)
    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # HPO phenotype-gap probe inputs (source_climb only): parse hp.obo xrefs -> the SNOMED
    # (and ICD) codes HPO gives an exact term to, so we can measure how much EHR signal that
    # currently CLIMBS or DROPS in Mondo would instead land precisely in HPO. Best-effort:
    # a download/parse failure skips the probe, never fails the export.
    hpo_by_snomed: dict[str, tuple] = {}   # SNOMED concept_code -> (hp_id, hp_label)
    hpo_by_icd: dict[tuple, tuple] = {}    # (vocab, code)       -> (hp_id, hp_label)
    # HPO axis structures (built only when --with-hpo): DAG labels/parents/has_child, plus
    # the OMOP concept_id -> hp_id map used to route persons onto the HPO ladder. Initialized
    # empty so the axis is cleanly disabled by default or on any load/parse failure.
    hpo_labels: dict[str, str] = {}
    hpo_parent_adj: dict[str, list[str]] = {}
    hpo_has_child: set = set()
    hpo_cid_rows: "list[tuple]" = []
    if args.hpo_obo_url and args.count_space in ("source_climb", "all"):
        try:
            dest = cache / "hp.obo"
            if not (dest.exists() and dest.stat().st_size > 0):
                sys.stderr.write(f"[hpo] downloading {args.hpo_obo_url}\n")
                urllib.request.urlretrieve(args.hpo_obo_url, dest)  # noqa: S310
            for hp_id, hp_label, vocab, code in parse_hpo_xrefs(dest.read_text()):
                if vocab == "SNOMED":
                    hpo_by_snomed.setdefault(code, (hp_id, hp_label))
                elif vocab in ("ICD10CM", "ICD9CM"):
                    hpo_by_icd.setdefault((vocab, code), (hp_id, hp_label))
            sys.stderr.write(f"[hpo] {len(hpo_by_snomed)} SNOMED + {len(hpo_by_icd)} ICD "
                             f"xref'd HPO terms loaded\n")

            if args.with_hpo:
                # HPO DAG (labels + is_a parents), and the OMOP concept_id -> hp_id map used
                # to place persons on the HPO ladder: SNOMED std concepts joined on
                # concept_code (primary), plus ICD source concepts from hpo_by_icd.
                hpo_labels, hpo_parents = parse_hpo_dag(dest.read_text())
                hpo_parent_adj, hpo_has_child = dag_structures(hpo_parents)
                snomed_cp = concept_pd[concept_pd["vocabulary_id"] == "SNOMED"]
                hpo_cid_rows = [
                    (int(r.concept_id), hpo_by_snomed[str(r.concept_code)][0])
                    for r in snomed_cp.itertuples() if str(r.concept_code) in hpo_by_snomed]
                icd_cp = concept_pd[concept_pd["vocabulary_id"].isin(["ICD10CM", "ICD9CM"])]
                hpo_cid_rows += [
                    (int(r.concept_id), hpo_by_icd[(str(r.vocabulary_id), str(r.concept_code))][0])
                    for r in icd_cp.itertuples()
                    if (str(r.vocabulary_id), str(r.concept_code)) in hpo_by_icd]
                sys.stderr.write(f"[hpo] {len(hpo_labels)} DAG terms, "
                                 f"{len(hpo_cid_rows)} concept ids mapped\n")
        except Exception as e:                          # noqa: BLE001 — probe is best-effort
            sys.stderr.write(f"[hpo] probe skipped: {e}\n")

    def run_space(space):
        """Count + assemble ONE count space; write its payload + TSV; return a summary
        row for the safe cross-space summary. All the space-specific attribution lives
        here; the Mondo/DAG/rare structure above is shared across spaces."""
        from pyspark.sql import Window
        source_codes_of: dict[str, dict] = {}   # source_climb catalog (empty otherwise)
        survey: dict = {}                       # source_climb tier coverage (empty otherwise)
        collision_kind_of: dict[str, str] = {}  # source_climb collision mechanism per term
        frac_of: dict[str, float] = {}          # source_climb fractional (1/m) term count

        def assemble_axis(attribution, parent_adj, has_child, label_of, rare_of,
                          out_name, axis_label):
            """CORE per-term assembly for ONE axis, run once per axis (Mondo, then HPO).

            Turns an ``attribution`` frame (person_id, mondo_id, origin_cid, via, k_src,
            k_std — the source_climb attribution for Mondo, ``t_hpo`` for HPO) plus the
            axis's OWN DAG structures (``parent_adj``/``has_child``/``label_of``/``rare_of``,
            passed in — NOT the Mondo ones) into the written payload ``<out>/<out_name>``
            and its ``_nodes.tsv``. Computes, per axis: the fractional (1/m) count, the
            source-code catalog + volume bands, the collision split, the term universe and
            term_rows, then ``assemble_payload`` + write.

            The source_climb SURVEY (tier persons, unmatched-vocab, HPO phenotype-gap probe)
            is Mondo-axis-only; its branch inputs (t1/t2/t3, unm_codes_by_vocab, hpo_probe)
            are computed once in the branch and closed over here, and the collision-split
            fields (which need this axis's own catalog) are folded in for the Mondo axis
            only. ``concept_pd``/``concept_all``/``spark`` are shared (closed over). Returns
            the same summary-row dict shape as the space runner. Byte-identical to the prior
            single-axis path when called for Mondo with the source_climb attribution."""
            from collections import Counter
            is_mondo = axis_label == "mondo"
            hit = attribution.select("person_id", "mondo_id")

            # FRACTIONAL (1/m) per-term count — the addable, de-double-counted number.
            # A condition keyed by (person, source concept, standard concept) maps to m
            # terms via its resolved tier (the collision); credit 1/m to each so the shares
            # add back to 1 and summed / rolled-up counts never double-count the map
            # ambiguity. m is a property of the condition (person-independent).
            attr_k = attribution.select(
                "person_id", "mondo_id",
                F.coalesce(F.col("k_src"), F.lit(-1)).alias("k_src"),
                F.coalesce(F.col("k_std"), F.lit(-1)).alias("k_std"))
            m_of = (attr_k.select("k_src", "k_std", "mondo_id").distinct()
                    .groupBy("k_src", "k_std").agg(F.count(F.lit(1)).alias("m")))
            frac_pd = (attr_k.distinct().join(m_of, ["k_src", "k_std"])
                       .groupBy("mondo_id")
                       .agg(F.sum(F.lit(1.0) / F.col("m")).alias("frac")).toPandas())
            frac_of = {str(r["mondo_id"]): float(r["frac"]) for _, r in frac_pd.iterrows()}

            # per-term catalog of ORIGINATING source codes. Keep the FULL tier via for the
            # collision split; the catalog display collapses to exact/climbed (exact wins if
            # a code reaches a term both ways). Named via the concept table.
            cat = (attribution.select("mondo_id", "origin_cid", "via").distinct()
                   .withColumn("vrank", F.when(F.col("via") == "climbed", 1).otherwise(0)))
            cat = (cat.withColumn("best", F.min("vrank").over(
                       Window.partitionBy("mondo_id", "origin_cid")))
                   .where(F.col("vrank") == F.col("best"))
                   .select("mondo_id", "origin_cid", "via").distinct())
            cat_named = (cat.join(concept_all, "origin_cid", "left")
                         .select("mondo_id", "origin_cid", "via", "vocabulary_id", "concept_code")
                         .toPandas())
            # per-(term, code) distinct-person volume -> a differencing-safe magnitude BAND
            # per code (never the exact count) + a per-term band histogram.
            code_counts = (attribution.groupBy("mondo_id", "origin_cid")
                           .agg(F.countDistinct("person_id").alias("np")).toPandas())
            np_of = {(str(r.mondo_id), int(r.origin_cid)): int(r.np)
                     for r in code_counts.itertuples()}
            band_rank = {lab: i for i, lab in enumerate(_band_label(min_cell))}  # heavy=0
            source_codes_of: dict[str, dict] = {}
            code_disp = {}       # origin_cid -> "VOCAB code" for examples
            for mid, sub in cat_named.groupby("mondo_id"):
                recs = []
                seen = set()   # one row per concept id: a code reaching a term via BOTH
                               # source_exact and standard_exact must not display (or count
                               # in the histogram) twice — both collapse to display "exact".
                for r in sub.itertuples():
                    cid = int(r.origin_cid)
                    if cid in seen:
                        continue
                    seen.add(cid)
                    vocab = str(r.vocabulary_id) if pd.notna(r.vocabulary_id) else "?"
                    code = str(r.concept_code) if pd.notna(r.concept_code) else str(cid)
                    disp = "climbed" if str(r.via) == "climbed" else "exact"
                    band = volume_band(np_of.get((str(mid), cid)), min_cell)
                    recs.append({"id": cid, "vocab": vocab, "code": code,
                                 "via": disp, "band": band})
                    code_disp[cid] = f"{vocab} {code}"
                # histogram over the FULL set (before the display cap); sort heavy-first so
                # the cap keeps the heavy hitters and the light tail collapses in the UI.
                hist = band_histogram([c["band"] for c in recs], min_cell)
                recs.sort(key=lambda c: (band_rank.get(c["band"], 99),
                                         c["via"] != "exact", c["vocab"], c["code"]))
                source_codes_of[str(mid)] = {"list": recs[:CATALOG_CAP], "n": len(recs),
                                             "bands": hist}

            # collisions: a single source code attributed to >1 term -> a patient with that
            # code counts under each. Split by MECHANISM (shared_concept / climb_tie / mixed).
            full_pairs = [(int(c), str(m), str(v)) for c, m, v
                          in cat_named[["origin_cid", "mondo_id", "via"]].itertuples(index=False)]
            siblings, collision_kind_of, code_kind = classify_collision_kinds(full_pairs)
            label = lambda mm: label_of.get(mm, mm)

            def _examples(kind, n=8):
                out_ex = []
                for c, k in code_kind.items():
                    if k != kind:
                        continue
                    terms = sorted({m for cc, m, v in full_pairs if cc == c})
                    out_ex.append({"code": code_disp.get(c, str(c)),
                                   "terms": [label(t) for t in terms][:6]})
                    if len(out_ex) >= n:
                        break
                return out_ex

            # --- per-term EXACT-match person counts (NO concept_ancestor climb) -----------
            term_counts = (hit.groupBy("mondo_id")
                           .agg(F.countDistinct("person_id").alias("n")).toPandas())
            count_of = {str(r["mondo_id"]): int(r["n"]) for _, r in term_counts.iterrows()}
            n_coded = cond.select("person_id").distinct().count()
            n_on_mondo = hit.select("person_id").distinct().count()

            # axis-specific mapping seed: for Mondo, the full same_as->Maps to mapping (so
            # zero-usage mapped terms still appear + carry their std_concepts/codes); for
            # HPO, the standard concept(s) each HP term was attributed, derived from t_hpo.
            if is_mondo:
                tm = term_match
                codes_map, codesbyvocab_map, rare_src_map = codes_of, codesbyvocab_of, rare_src_of
            else:
                tm_pd = (attribution.select("mondo_id", "k_std")
                         .where(F.col("k_std").isNotNull()).distinct().toPandas())
                tm = {}
                for hid, sub in tm_pd.groupby("mondo_id"):
                    tm[str(hid)] = sorted({int(x) for x in sub["k_std"]})
                codes_map, codesbyvocab_map, rare_src_map = {}, {}, {}

            # term universe: the SEED = mapped terms, plus any term that received a count or
            # a catalogued source code even if its standard mapping was empty. The KEPT
            # universe adds back un-attributed ancestors that are genuine branch points (>= 2
            # seed-bearing children), collapsing purely-linear pass-through ancestors so the
            # DAG reads with its real branch structure on both axes (topology-preserving
            # skeleton reduction; see ``meaningful_skeleton``). Newly-added branch-point ids
            # are absent from tm/count_of/source_codes_of, so they fall through the .get()s
            # below to empty std_concepts/codes/source_codes and n_persons=0 — structural,
            # zero-count nodes — while still getting a real label via ``label_of`` (populated
            # for the full DAG on both axes, not just attributed terms).
            seed_ids = set(tm) | set(count_of) | set(source_codes_of)
            keep_ids = meaningful_skeleton(seed_ids, parent_adj)
            parents = nearest_mapped_parents(keep_ids, parent_adj)

            term_rows = []
            for mid in sorted(keep_ids):
                cids = sorted({int(c) for c in tm.get(mid, [])})
                codes = codes_map.get(mid, [])
                catrec = source_codes_of.get(mid) or {"list": [], "n": 0, "bands": []}
                term_rows.append({
                    "mondo_id": mid,
                    "label": label_of.get(mid, mid),
                    "is_internal": mid in has_child,
                    "parents": parents.get(mid, []),
                    "std_concepts": cids,
                    "codes": codes,
                    "n_codes": len(codes),
                    "codes_by_vocab": codesbyvocab_map.get(mid, {}),
                    "source_codes": catrec["list"],
                    "n_source_codes": catrec["n"],
                    "source_bands": catrec.get("bands", []),
                    "n_persons": count_of.get(mid, 0),
                    "n_frac": frac_of.get(mid, float(count_of.get(mid, 0))),
                    "collision_siblings": siblings.get(mid, []),
                    "collision_kind": collision_kind_of.get(mid, ""),
                    "rare": rare_of.get(mid, False),
                    "rare_src": rare_src_map.get(mid, []),
                })

            # --- survey (Mondo axis only): tier persons + unmatched vocab + collision split
            #     + HPO probe. Branch inputs (t1/t2/t3, unm_codes_by_vocab, hpo_probe) are
            #     closed over; the collision-split fields use this axis's own catalog. ------
            survey = {}
            if is_mondo:
                term_kind_counts = Counter(collision_kind_of.values())
                code_kind_counts = Counter(code_kind.values())
                survey = {
                    "persons_source_exact": int(t1.select("person_id").distinct().count()),
                    "persons_standard_exact": int(t2.select("person_id").distinct().count()),
                    "persons_climbed": int(t3.select("person_id").distinct().count()),
                    "unmatched_codes_by_vocab": {
                        (str(r.vocabulary_id) if pd.notna(r.vocabulary_id) else "?"): int(r.codes)
                        for r in unm_codes_by_vocab.itertuples()},
                    "collision_terms_by_kind": {k: int(v) for k, v in term_kind_counts.items()},
                    "collision_codes_by_kind": {k: int(v) for k, v in code_kind_counts.items()},
                    "collision_examples_climb_tie": _examples("climb_tie"),
                    "collision_examples_shared_concept": _examples("shared_concept"),
                    "hpo": hpo_probe,
                }

            # --- assemble this axis + write its payload/TSV -------------------------------
            if is_mondo:
                meta_space = space
                meta_rule = {
                    "source": ("distinct persons whose condition_source_concept_id exactly "
                               "matches one of the term's Mondo same_as source codes (no Maps to)"),
                    "source_climb": ("distinct persons attributed to the most specific mapped "
                                     "Mondo term reachable: source-exact (same_as), else "
                                     "standard-exact (condition_concept_id), else nearest mapped "
                                     "SNOMED ancestor via concept_ancestor (ties reduced to the "
                                     "most-specific Mondo term; orthogonal ties counted in each)"),
                }.get(space,
                      "distinct persons with an EXACT-match standard condition concept "
                      "(same_as -> Maps to)")
            else:
                meta_space = "hpo_exact"
                meta_rule = ("distinct persons whose condition's standard/source concept has an "
                             "EXACT HPO xref term (the phenotype axis), attributed to that HP "
                             "term over the HPO is_a DAG — disjoint from the Mondo climb")
            meta = {
                "mondo_version": args.mondo_version,
                "cdr": args.cdr,
                "source_table": args.source_table,
                "min_cell": min_cell,
                "rollup": False,
                "count_space": meta_space,
                "count_rule": meta_rule,
                "generated_utc": generated_utc,
            }
            if survey:
                meta["survey"] = survey
            payload = assemble_payload(meta=meta, term_rows=term_rows, min_cell=min_cell)

            (out / out_name).write_text(json.dumps(payload))
            tsv_name = (out_name[:-5] if out_name.endswith(".json") else out_name) + "_nodes.tsv"
            rows = [{
                "mondo_id": nd["id"], "label": nd["label"], "kind": nd["kind"],
                "depth": nd["depth"], "state": nd["state"], "category": nd["category"],
                "n_patients": nd["display"],
                "n_frac": nd.get("frac_display", ""),
                "collision": int(nd["collision"]),
                "collision_siblings": "|".join(nd["collision_siblings"]),
                "n_codes": nd["n_codes"],
                "codes_by_vocab": ";".join(f"{v}:{c}" for v, c in nd["codes_by_vocab"].items()),
                "codes": "|".join(f"{c['vocab']}:{c['code']}" for c in nd["codes"]),
                "n_source_codes": nd.get("n_source_codes", 0),
                "source_codes": "|".join(f"{c['vocab']}:{c['code']}:{c['via']}:{c.get('band','')}"
                                         for c in nd.get("source_codes", [])),
                "source_bands": ";".join(f"{b['band']}:{b['codes']}"
                                         for b in nd.get("source_bands", [])),
                "rare": int(nd["rare"]), "rare_src": "|".join(nd["rare_src"]),
                "parents": "|".join(nd["parents"]),
            } for nd in payload["nodes"] if nd["kind"] != "root"]
            pd.DataFrame(rows).to_csv(out / tsv_name, sep="\t", index=False)

            # per-axis stderr (suppressed) for the run log
            sys.stderr.write(f"\n=== count space: {meta_space} ===\n" if is_mondo
                             else f"\n=== axis: {axis_label} ===\n")
            sys.stderr.write(format_summary(payload["stats"], min_cell=min_cell) + "\n")
            _attr = ("attributed" if space == "source_climb" else "exact") if is_mondo else "hpo-exact"
            _ont = "Mondo" if is_mondo else "HPO"
            sys.stderr.write(
                f"[ladder] persons total {suppress_count(n_total, min_cell)} | coded "
                f"{suppress_count(n_coded, min_cell)} | on any mapped {_ont} term ({_attr}) "
                f"{suppress_count(n_on_mondo, min_cell)} "
                f"({100.0 * n_on_mondo / max(n_total, 1):.1f}% of all persons)\n")
            return {"space": (space if is_mondo else "hpo"), "payload": payload,
                    "stats": payload["stats"], "survey": survey, "n_total": n_total,
                    "n_coded": n_coded, "n_on_mondo": n_on_mondo, "min_cell": min_cell,
                    "mondo_version": args.mondo_version, "generated_utc": generated_utc}

        if space == "source":
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
        elif space == "source_climb":
            # --- SOURCE_CLIMB: 3-tier partial roll-up, preferring the most SPECIFIC
            #     mapped Mondo term reachable, and always cataloguing the ORIGINATING
            #     source code. Precedence (first hit wins; never climb past an exact):
            #       (1) source-exact  : condition_source_concept_id is a term same_as code
            #       (2) standard-exact: condition_concept_id (same_as -> Maps to) hits a term
            #       (3) climb         : nearest mapped ancestor of condition_concept_id in
            #                           SNOMED concept_ancestor; a tie is reduced to its
            #                           most-specific Mondo term(s), orthogonal ties flagged
            #     concept_ancestor is SNOMED-only (ICD is non-standard), so only the
            #     standard concept can be climbed; the ICD source rides up through it.
            from pyspark.sql import Window
            # per-term display cap on the catalogued source codes (heavy-first, so the cap
            # keeps the high-volume codes). 400 fully covers all but a couple dozen very
            # broad terms; the dashboard's expandable drawer reveals the full in-payload
            # list (incl. the ≤20 tail), and any residual beyond the cap is noted as
            # "+N more". Codes are public identities, so this is a size/UX bound, not a
            # disclosure one.
            CATALOG_CAP = 400

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
                          F.col("src_cid").alias("origin_cid"),
                          F.lit("source_exact").alias("via"),
                          cond["src_cid"].alias("k_src"), cond["std_cid"].alias("k_std")))
            # rows whose source code is NOT a same_as (fall through to tier 2/3)
            rem1 = cond.join(src_ids_sdf, cond["src_cid"] == src_ids_sdf["src_cid"], "left_anti")

            # TIER 2 — standard-exact (origin = the ICD source code, or std if none)
            t2 = (rem1.join(std_sdf, rem1["std_cid"] == std_sdf["map_cid"], "inner")
                  .select("person_id", "mondo_id",
                          origin.alias("origin_cid"),
                          F.lit("standard_exact").alias("via"),
                          rem1["src_cid"].alias("k_src"), rem1["std_cid"].alias("k_std")))
            # rows whose standard concept is ALSO not a mapped term -> candidates to climb
            rem2 = rem1.join(std_ids_sdf, rem1["std_cid"] == std_ids_sdf["map_cid"], "left_anti")

            # HPO-EXACT RUNG — a rem2 condition whose STANDARD concept is an HPO xref.
            # Kept SEPARATE from the Mondo attribution (t_hpo is not unioned in); the
            # Mondo climb below only climbs what HPO did NOT claim.
            if args.with_hpo and hpo_labels:
                hpo_map = broadcast(spark.createDataFrame(
                    pd.DataFrame(hpo_cid_rows, columns=["map_cid", "hp_id"]).drop_duplicates()))
                # HPO-exact: a rem2 condition whose STANDARD concept is an HPO xref. v1 matches
                # on std_cid only; source-ICD HPO xrefs (few) are staged in hpo_cid_rows but not
                # yet joined here (spec decision #3 left std-vs-source open) — a v2 rung.
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

            # TIER 3 — climb SNOMED concept_ancestor to the nearest mapped term(s)
            ca = (_read_bq(spark, args.cdr, args.billing, "concept_ancestor")
                  .select("ancestor_concept_id", "descendant_concept_id",
                          "min_levels_of_separation")
                  .where(F.col("min_levels_of_separation") >= 1))
            unmatched_std = rem2_climb.select("std_cid").distinct()
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
            # map nearest SNOMED ancestors -> Mondo terms, then REDUCE each code's tie-set
            # to its MOST-SPECIFIC Mondo term(s): a SNOMED-distance tie can include a
            # specific term and its more-general Mondo ancestors (nested chains like
            # {carbuncle, pyoderma, skin disorder}); the ladder promises the most specific,
            # so we drop tied ancestors of tied descendants (genuine orthogonal ties stay).
            tie_pd = (nearest.join(broadcast(anc_map), "anc_cid", "inner")
                      .select("std_cid", "anc_mondo").distinct().toPandas())
            reduced = reduce_tie_map(
                [(int(r.std_cid), str(r.anc_mondo)) for r in tie_pd.itertuples(index=False)],
                parent_adj)
            reduced_pd = pd.DataFrame(
                [(k, mm) for k, ms in reduced.items() for mm in ms],
                columns=["std_cid", "mondo_id"])
            if len(reduced_pd):
                reduced_sdf = broadcast(spark.createDataFrame(reduced_pd))
                t3 = (rem2_climb.join(reduced_sdf, "std_cid", "inner")
                      .select("person_id", "mondo_id",
                              origin.alias("origin_cid"), F.lit("climbed").alias("via"),
                              rem2_climb["src_cid"].alias("k_src"), F.col("std_cid").alias("k_std")))
            else:
                t3 = t1.limit(0)          # no climbs (empty, keeps t1's schema for the union)

            attribution = (t1.unionByName(t2).unionByName(t3)).cache()
            # concept identities for the origin-code catalog (built per axis inside
            # assemble_axis) AND the unmatched-vocab survey below. Read once, shared.
            concept_all = (_read_bq(spark, args.cdr, args.billing, "concept")
                           .select(F.col("concept_id").alias("origin_cid"),
                                   "vocabulary_id", "concept_code"))

            # The per-term assembly (fractional 1/m count, source-code catalog + bands,
            # collision split, term_rows, assemble_payload, write) is factored into
            # assemble_axis and run once per axis (Mondo below, then HPO). Only the
            # source_climb SURVEY inputs are computed here (Mondo-axis-only): tier person
            # counts (from t1/t2/t3, inside assemble_axis), the unmatched-vocab tally, and
            # the HPO phenotype-gap probe.

            # survey: distinct persons resolved at each tier (overlapping across tiers),
            # plus the unmatched remainder as distinct source CODES by vocabulary. Counting
            # unmatched CODES (not persons) is the honest "which vocabularies can't the
            # climb reach" measure — persons-with-any-unmatched-condition overlaps almost
            # everyone and reads as a false gap. Code counts are identity (unsuppressed,
            # never per-code patient counts). The real uncovered-persons figure is the
            # ladder's coded - on_mondo, surfaced in the summary.
            # NOTE: rem2_climb (not rem2) — codes claimed by the HPO-exact rung must not be
            # mis-counted here as unmatched. When --with-hpo is off, rem2_climb IS rem2, so
            # this is byte-identical to the prior behavior.
            unmatched = rem2_climb.join(nearest.select("std_cid").distinct(), "std_cid", "left_anti")
            src_vocab = concept_all.select(F.col("origin_cid").alias("src_cid"), "vocabulary_id")
            unm_codes_by_vocab = (unmatched.select("src_cid").distinct()
                                  .join(src_vocab, "src_cid", "left")
                                  .groupBy("vocabulary_id")
                                  .agg(F.countDistinct("src_cid").alias("codes"))
                                  .toPandas())

            # ---- HPO phenotype-gap probe -------------------------------------------------
            # Of the STANDARD (SNOMED) concepts coded in the EHR, how many that Mondo can
            # only reach by CLIMBING to a more-general term, or can't map at all (DROP),
            # does HPO give an EXACT term to? Sizes the phenotype axis Mondo doesn't cover
            # (hypomagnesemia, lab abnormalities, ...). Emits concept COUNTS (safe) + a
            # ≤-suppressed person MASS (sum over concepts of distinct-persons; not distinct
            # persons) + identity-only examples (SNOMED code -> HP term -> what it climbs to).
            hpo_probe = {}
            if hpo_by_snomed:
                snomed_cp = concept_pd[concept_pd["vocabulary_id"] == "SNOMED"]
                hp_of_cid = {}   # SNOMED concept_id -> (hp_id, hp_label, snomed_code)
                for r in snomed_cp.itertuples():
                    code = str(r.concept_code)
                    if code in hpo_by_snomed:
                        hp_id, hp_label = hpo_by_snomed[code]
                        hp_of_cid[int(r.concept_id)] = (hp_id, hp_label, code)
                hpo_cids = set(hp_of_cid)
                direct_set = {int(x) for x in mapped_std_ids}
                climb_set = ({int(x) for x in reduced_pd["std_cid"].unique()}
                             if len(reduced_pd) else set())
                climbs_to = {}   # std_cid -> [Mondo labels] it currently climbs to
                if len(reduced_pd):
                    for r in reduced_pd.itertuples(index=False):
                        climbs_to.setdefault(int(r.std_cid), []).append(
                            label_of.get(str(r.mondo_id), str(r.mondo_id)))
                std_np = (cond.where(F.col("std_cid").isNotNull() & (F.col("std_cid") != 0))
                          .groupBy("std_cid").agg(F.countDistinct("person_id").alias("np"))
                          .toPandas())
                agg = {"climb": {"concepts": 0, "with_hpo": 0, "mass": 0, "mass_hpo": 0},
                       "drop":  {"concepts": 0, "with_hpo": 0, "mass": 0, "mass_hpo": 0}}
                ex = []
                for r in std_np.itertuples(index=False):
                    cid, npv = int(r.std_cid), int(r.np)
                    if cid in direct_set:
                        continue                        # Mondo already has a direct term
                    status = "climb" if cid in climb_set else "drop"
                    a = agg[status]
                    a["concepts"] += 1; a["mass"] += npv
                    if cid in hpo_cids:
                        a["with_hpo"] += 1; a["mass_hpo"] += npv
                        if status == "climb":           # npv used ONLY to rank; never emitted
                            hp_id, hp_label, code = hp_of_cid[cid]
                            ex.append((npv, {"snomed": code, "hp_id": hp_id,
                                             "hp_label": hp_label,
                                             "climbs_to": climbs_to.get(cid, [])[:3]}))
                ex.sort(key=lambda t: -t[0])
                hpo_probe = {"hpo_snomed_terms": len(hpo_by_snomed),
                             "climb": agg["climb"], "drop": agg["drop"],
                             "examples": [e for _, e in ex[:8]]}

            # --- assemble both axes from their attributions -------------------------------
            # Mondo: the source_climb attribution over the Mondo DAG (byte-identical to the
            # prior single-axis path); the survey is attached to this axis only.
            mondo_result = assemble_axis(
                attribution, parent_adj, has_child, label_of, rare_of,
                f"mondo_usage_{space}.json", "mondo")
            # HPO: the HPO-exact rung (t_hpo) over the HPO DAG -> hpo_usage.json. Only when
            # --with-hpo built the axis (t_hpo is None otherwise -> no HPO payload).
            if t_hpo is not None:
                hpo_result = assemble_axis(
                    t_hpo, hpo_parent_adj, hpo_has_child, hpo_labels, {},
                    "hpo_usage.json", "hpo")
                mondo_result["hpo_axis"] = {
                    "used_terms": hpo_result["stats"]["used_terms"],
                    "reported_terms": hpo_result["stats"]["reported_terms"],
                    "persons": hpo_result["n_on_mondo"],
                }
            return mondo_result
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
        n_coded = cond.select("person_id").distinct().count()
        n_on_mondo = hit.select("person_id").distinct().count()

        # term universe: the mapped terms, plus (source_climb) any term that received a
        # count or catalogued source code even if its standard mapping was empty.
        mapped_ids = set(term_match) | set(count_of) | set(source_codes_of)
        parents = nearest_mapped_parents(mapped_ids, parent_adj)   # parent_adj: shared, above

        term_rows = []
        for mid in sorted(mapped_ids):
            cids = sorted({int(c) for c in term_match.get(mid, [])})
            codes = codes_of.get(mid, [])
            cat = source_codes_of.get(mid) or {"list": [], "n": 0, "bands": []}
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
                "source_bands": cat.get("bands", []),
                "n_persons": count_of.get(mid, 0),
                # fractional (1/m) de-double-counted count; falls back to the exact count
                # for spaces without collisions (source) or that don't compute it.
                "n_frac": frac_of.get(mid, float(count_of.get(mid, 0))),
                "collision_siblings": siblings.get(mid, []),
                "collision_kind": collision_kind_of.get(mid, ""),
                "rare": rare_of.get(mid, False),
                "rare_src": rare_src_of.get(mid, []),
            })

        # --- assemble this space + write its per-space payload/TSV --------------------
        meta = {
            "mondo_version": args.mondo_version,
            "cdr": args.cdr,
            "source_table": args.source_table,
            "min_cell": min_cell,
            "rollup": False,
            "count_space": space,
            "count_rule": {
                "source": ("distinct persons whose condition_source_concept_id exactly "
                           "matches one of the term's Mondo same_as source codes (no Maps to)"),
                "source_climb": ("distinct persons attributed to the most specific mapped "
                                 "Mondo term reachable: source-exact (same_as), else "
                                 "standard-exact (condition_concept_id), else nearest mapped "
                                 "SNOMED ancestor via concept_ancestor (ties reduced to the "
                                 "most-specific Mondo term; orthogonal ties counted in each)"),
            }.get(space,
                  "distinct persons with an EXACT-match standard condition concept "
                  "(same_as -> Maps to)"),
            "generated_utc": generated_utc,
        }
        if survey:
            meta["survey"] = survey
        payload = assemble_payload(meta=meta, term_rows=term_rows, min_cell=min_cell)

        (out / f"mondo_usage_{space}.json").write_text(json.dumps(payload))
        rows = [{
            "mondo_id": nd["id"], "label": nd["label"], "kind": nd["kind"],
            "depth": nd["depth"], "state": nd["state"], "category": nd["category"],
            "n_patients": nd["display"],
            "n_frac": nd.get("frac_display", ""),
            "collision": int(nd["collision"]),
            "collision_siblings": "|".join(nd["collision_siblings"]),
            "n_codes": nd["n_codes"],
            "codes_by_vocab": ";".join(f"{v}:{c}" for v, c in nd["codes_by_vocab"].items()),
            "codes": "|".join(f"{c['vocab']}:{c['code']}" for c in nd["codes"]),
            "n_source_codes": nd.get("n_source_codes", 0),
            "source_codes": "|".join(f"{c['vocab']}:{c['code']}:{c['via']}:{c.get('band','')}"
                                     for c in nd.get("source_codes", [])),
            "source_bands": ";".join(f"{b['band']}:{b['codes']}"
                                     for b in nd.get("source_bands", [])),
            "rare": int(nd["rare"]), "rare_src": "|".join(nd["rare_src"]),
            "parents": "|".join(nd["parents"]),
        } for nd in payload["nodes"] if nd["kind"] != "root"]
        pd.DataFrame(rows).to_csv(out / f"mondo_usage_{space}_nodes.tsv",
                                  sep="\t", index=False)

        # per-space stderr (suppressed) for the run log
        sys.stderr.write(f"\n=== count space: {space} ===\n")
        sys.stderr.write(format_summary(payload["stats"], min_cell=min_cell) + "\n")
        _attr = "attributed" if space == "source_climb" else "exact"
        sys.stderr.write(
            f"[ladder] persons total {suppress_count(n_total, min_cell)} | coded "
            f"{suppress_count(n_coded, min_cell)} | on any mapped Mondo term ({_attr}) "
            f"{suppress_count(n_on_mondo, min_cell)} "
            f"({100.0 * n_on_mondo / max(n_total, 1):.1f}% of all persons)\n")
        return {"space": space, "payload": payload, "stats": payload["stats"],
                "survey": survey, "n_total": n_total, "n_coded": n_coded,
                "n_on_mondo": n_on_mondo, "min_cell": min_cell,
                "mondo_version": args.mondo_version, "generated_utc": generated_utc}

    # --- run the requested count space(s), write per-space payloads + a safe summary --
    spaces = (["standard", "source", "source_climb"]
              if args.count_space == "all" else [args.count_space])
    results = [run_space(sp) for sp in spaces]

    # primary copy for the dashboard's default fetch (mondo_usage.json / _nodes.tsv):
    # source_climb when present (richest), else the single requested space.
    primary = next((r for r in results if r["space"] == "source_climb"), results[-1])
    ps = primary["space"]
    (out / "mondo_usage.json").write_text(json.dumps(primary["payload"]))
    shutil.copyfile(out / f"mondo_usage_{ps}_nodes.tsv", out / "mondo_usage_nodes.tsv")

    # disclosure-safe, copy-pasteable cross-space summary (suppressed; no CDR id)
    summary_md = build_safe_summary(results)
    (out / "mondo_usage_summary.md").write_text(summary_md)
    sys.stderr.write("\n" + summary_md + "\n")
    sys.stderr.write(
        f"[done] wrote {len(results)} payload(s) + mondo_usage_summary.md "
        f"(primary mondo_usage.json = {ps}) to {out}\n")
    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

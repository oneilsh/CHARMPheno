"""Pure, Spark-free core of the whole-Mondo map-and-roll machinery.

PROVENANCE: ported VERBATIM from main's `analysis/cloud/mondo_usage_cloud.py`
@ c3fba5813299badae2eef24b962fd06ee08ba401 (the exp 0105/0108 EHR-usage export driver). `git merge-base`
between that branch and this one FAILS — the two histories are unrelated (1,410
vs 97 commits) — so "reconciliation" is a PORT, not a merge, and this module is
the minimal one needed to unblock exp 0110's native-Mondo label space
(docs/superpowers/plans/2026-08-31-native-mondo-label-space-plan.md §4 step 1).

WHAT WAS TAKEN: every function in that driver's "Pure, Spark-free core" section,
byte-for-byte, together with its module constants. Nothing was edited: no
renames, no signature changes, no "while I'm here" fixes. Its 33 unit tests came
across with it (`tests/test_mondo_usage_core.py`, adapted only for the module
name), so a later unification can diff the two files and find them identical
modulo this docstring.

WHAT WAS LEFT BEHIND: the Spark driver (`main`), which reads BigQuery and writes
the dashboard payload. The label front-end (`mondo_native_dag.py`) re-derives the
attribution frame it needs from these same pure functions; the export driver
itself is not part of this branch.

WHAT THIS BRANCH ACTUALLY USES (the rest rides along so the port stays verbatim
and the 33 tests stay meaningful):

  `nearest_mapped_standard_ancestors` + `reduce_tie_map`
      the 3-tier source_climb ladder's third rung — nearest mapped SNOMED
      ancestor by `concept_ancestor` distance, tie-reduced to the MOST SPECIFIC
      Mondo term(s). This is the frontier attribution exp 0110 replaces
      `powered_anchor_climb` with (plan §3): the old climb attested EVERY powered
      ancestor, which is the co-attestation half of exp 0104's 619 subsumed
      category-anchors.
  `nearest_mapped_parents`
      the induced multi-parent Hasse relation over a kept term set — Mondo's own
      hierarchy restricted to the label nodes, which is what makes a kept node's
      sibling set unable to contain its own descendant.
  `meaningful_skeleton`
      keep attributed terms plus genuine branch points; drop linear pass-through
      ancestors. (Exp 0110 does thin-chain removal with `mondo_collapse`'s
      splice-to-fixpoint instead, on the DAG rather than the term set; this is
      here because the port is verbatim and its tests came with it.)

EGRESS NOTE (plan §6): the suppression / banding / complementary-suppression
helpers below are PUBLISHING rules (min_cell=20, volume bands, differencing
safety). Model-internal label powering (min_positives=100 on CLOSURE support) is
a different dial living in `mondo_native_dag.py`. The two must never be
conflated — nothing in the label path calls anything in this file's egress half.

CACHE-KEY NOTE: this module is NOT source-hashed into any existing bundle key.
It is folded into the Mondo key only via `mondo_native_dag`, and only when
`dag_source=mondo_native` (see `_case_finding_cache.compute_bundle_cache_key`),
so every SNOMED / legacy-Mondo key stays byte-identical.
"""
from __future__ import annotations

_MIN_CELL = 20
_ROOT_ID = "root"


# --------------------------------------------------------------------------- #
# Pure, Spark-free core (unit-tested in tests/test_mondo_usage_core.py)      #
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


# --- complementary suppression: the source-code -> term roll-up marginal ------------
# A term's total N_T is the union of its source codes' person sets, so it is a
# roll-up MARGINAL over its per-code cells. That within-term roll-up (NOT any
# term-to-term one — those don't exist here) lets inclusion-exclusion narrow a
# suppressed cell: with two codes and N_T=115 where one code caps at 100,
# |other| >= 115 - 100 = 15, boxing a <=floor code into [15,20] even though no exact
# small count is published. So per-code bands do NOT compose safely with an EXACT
# total for near-floor terms (correcting the note above): we complementary-suppress
# the marginal — publish such a term's total as a STANDARD band instead of the exact
# number. Because the only differencing channel is within a single term (no global
# per-code count and no rolled-up term-to-term total are published), per-term safety
# implies whole-file safety, and banding is information-removing, so ONE pass with no
# recheck suffices. `assert_diff_safe` re-checks only as a redundant tripwire that
# fires if those two structural preconditions are ever violated.
def _band_bounds(label: str, min_cell: int = _MIN_CELL):
    """(lo, hi) integer bounds for a standard band label. Pure."""
    for lab, lo, hi in [
        (f"≤{min_cell}", 1, min_cell), (f"{min_cell + 1}–100", min_cell + 1, 100),
        ("101–1k", 101, 1000), ("1k–10k", 1001, 10000),
        ("10k–100k", 10001, 100000), (">100k", 100001, float("inf")),
    ]:
        if lab == label:
            return lo, hi
    return 1, float("inf")


def _forced_lower(total_lo, total_hi, source_bands, min_cell: int = _MIN_CELL):
    """Tightest lower bound sound differencing can place on a ≤floor source code of a
    term whose published total lies in ``[total_lo, total_hi]`` and whose per-code band
    histogram is ``source_bands`` (``[{"band", "codes"}]``, the FULL set).

    From |c| >= N - Σ_{k≠c} min(band_hi(k), N): the attacker uses the smallest total
    (total_lo) and each OTHER code's largest possible count (min(band_hi(k), total_hi)).
    Returns the forced lower bound; >= 2 means a ≤floor cell is boxed into the
    suppressed interior [2, floor]. 0 when the term carries no ≤floor code. Pure."""
    floor = f"≤{min_cell}"
    if not any(b["band"] == floor for b in source_bands):
        return 0
    others_max = 0
    for b in source_bands:
        _, hi = _band_bounds(b["band"], min_cell)
        others_max += min(hi, total_hi) * int(b["codes"])
    return total_lo - (others_max - min_cell)       # drop one ≤floor code (its cap = min_cell)


def safe_total_band(n_persons, source_bands, min_cell: int = _MIN_CELL):
    """``(needs_banding, display, (lo, hi))`` for a reported term's total.

    If publishing the EXACT total would let differencing box a ≤floor code >= 2 into
    the suppressed interior, return the finest STANDARD band that removes it: the
    two-sided band CONTAINING the total when that already suffices (small, boring,
    e.g. ``21–100``), else its ceiling (``≤100``), whose published lower bound is 1
    and so is always safe. Standard buckets only — never a bespoke tight range.
    Deterministic and one-shot (no retry loop). Pure."""
    N = int(n_persons)
    if _forced_lower(N, N, source_bands, min_cell) < 2:
        return False, str(N), (N, N)
    lbl = volume_band(N, min_cell)                   # containing two-sided band
    lo, hi = _band_bounds(lbl, min_cell)
    if _forced_lower(lo, hi, source_bands, min_cell) <= 1:
        return True, lbl, (int(lo), int(hi))         # small two-sided band suffices
    up = lbl.split("–")[1] if "–" in lbl else lbl.lstrip("≤")
    return True, f"≤{up}", (1, int(hi) if hi != float("inf") else hi)  # ceiling fallback


def complementary_suppress(nodes, min_cell: int = _MIN_CELL) -> int:
    """Band the total of any reported node whose exact total + per-code bands would
    box a ≤floor code into the suppressed interior. Mutates nodes in place (count ->
    None, display -> band label, adds ``count_range`` + ``total_banded``); keeps
    state/category/per-code bands untouched. Returns how many nodes were banded."""
    banded = 0
    for nd in nodes:
        if nd.get("state") != "reported" or nd.get("count") is None:
            continue
        need, disp, (lo, hi) = safe_total_band(
            nd["count"], nd.get("source_bands") or [], min_cell)
        if need:
            nd["count"] = None
            nd["display"] = disp
            nd["count_range"] = [lo, hi if hi != float("inf") else None]
            nd["total_banded"] = True
            banded += 1
    return banded


def assert_diff_safe(nodes, min_cell: int = _MIN_CELL) -> None:
    """Post-condition tripwire: no published node lets sound differencing box a ≤floor
    source code into a strict sub-interval of [1, floor]. Redundant given
    ``complementary_suppress`` + within-term independence; it fires only if those
    structural preconditions are broken later (e.g. a rolled-up total is added)."""
    for nd in nodes:
        if nd.get("state") != "reported":
            continue
        if nd.get("count") is not None:
            lo = hi = int(nd["count"])
        else:
            rng = nd.get("count_range") or [1, None]
            lo, hi = rng[0], (rng[1] if rng[1] is not None else float("inf"))
        f = _forced_lower(lo, hi, nd.get("source_bands") or [], min_cell)
        if f >= 2:
            raise AssertionError(
                f"diff-unsafe term {nd.get('id')}: a ≤{min_cell} source code is boxed "
                f"to >= {f} (within-term roll-up narrowing not suppressed)")



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
    Headline stats are exact term counts (terms are not patients).

    A final `complementary_suppress` pass bands the total (``count`` -> None,
    ``display`` -> a standard band, ``count_range`` + ``total_banded`` added) for any
    reported term whose exact total + per-code bands would let differencing box a
    ≤floor source code into the suppressed interior; `assert_diff_safe` then re-checks
    as a redundant post-condition. Such terms stay ``reported``/used in the stats."""
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

    # Complementary suppression of the source-code -> term roll-up marginal: band the
    # total of any term whose exact total + per-code bands would box a ≤floor code into
    # the suppressed interior. One pass suffices (within-term independence; see the note
    # by `complementary_suppress`); category/state/stats are unaffected (banded terms are
    # still reported & used — only their exact number becomes a standard band).
    n_total_banded = complementary_suppress(nodes, min_cell)

    n_terms = len(term_rows)
    n_used = counts["used_small"] + counts["reported"]
    stats = {
        "mapped_terms": n_terms,
        "used_terms": n_used,
        "used_small_terms": counts["used_small"],
        "reported_terms": counts["reported"],
        "total_banded_terms": n_total_banded,
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
    assert_diff_safe(nodes, min_cell)     # redundant post-condition; see the note above
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

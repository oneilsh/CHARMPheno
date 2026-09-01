"""Native-Mondo label DAG (exp 0110): map-and-roll labels, keep what's supported.

WHY THIS EXISTS — the limit of a two-patch sequence
---------------------------------------------------
Exp 0104's whole-Mondo readout carried 763 degenerate (constant-fallback) label
heads at C=3,820. Exp 0109 decomposed them exactly: 1 root, 143 STRUCTURAL
only-child class nodes (removed by the splice-to-fixpoint reduction), and 619
SUBSUMED CATEGORY-ANCHORS. The 619 are the deep one. Three facts compose into
them:

  * `powered_anchor_climb` attests EVERY powered ancestor of a coded concept;
  * `anchor_hierarchy.reduce_to_anchor_hierarchy` nests anchors only under CLASS
    covers, so a category-anchor sits as a SIBLING of its own descendants;
  * closure masking hands out negatives only via siblings.

So a category is co-attested on every doc that fires a "sibling" and is never
observed as a negative: an all-positive cell, a constant head, on 619 common
clinical categories. The proposed patch (nest anchor-under-anchor) is a
hand-rolled local approximation of what Mondo's own graph does globally — in a
TRANSITIVE REDUCTION of the real ontology restricted to kept nodes a subsumed
sibling cannot exist, because the redundant sibling edge is exactly what
reduction deletes. This module stops reconstructing Mondo's hierarchy from
covers and uses Mondo's hierarchy.

THE DESIGN (plan §3, docs/superpowers/plans/2026-08-31-native-mondo-label-space-plan.md)
----------------------------------------------------------------------------
  1. **Frontier / attestation.** A document's attested nodes are the MOST
     SPECIFIC mapped Mondo terms its in-window condition codes resolve to, via
     main's `source_climb` ladder (`mondo_usage_core`): standard-exact first,
     then a `concept_ancestor` climb to the nearest mapped standard ancestor,
     tie-reduced by `reduce_tie_map` to the most-specific Mondo term(s). This
     replaces `powered_anchor_climb` and the whole OMOP-cid attestation
     provider. The semantic improvement is inherited, not added: the old climb
     attested every powered ancestor; this one attests the frontier.
  2. **Powering.** Producer-side CLOSURE support: distinct persons per node
     rolled through the is-a closure over Mondo's own graph. Keep nodes with
     closure support >= `min_positives` (100). This replaces terminal powering
     + class covers with ONE uniform rule — a kept node is "a Mondo disease with
     enough patients under it", whether or not any code maps to it directly.
     "Directly coded" becomes a node PROPERTY (`coded_cids`), not a node type.
  3. **The label DAG.** `induced_hasse_parents(kept, parent_adj)` — the induced
     multi-parent Hasse relation over the kept set — then `mondo_collapse`'s
     splice-to-fixpoint as a generic thin-chain post-pass, with `is_terminal`
     read as the directly-coded PROPERTY instead of exp 0109's positive-concept-id
     sign test. By construction no kept node's sibling set can contain its own
     descendant, so the 619-class trap is structurally impossible; the residual
     degeneracy should be `1 (root) + a small thin-chain residue`.

A CORRECTION TO THE PLAN, FOUND BY A TEST (recorded)
----------------------------------------------------
Plan §4 step 2 names `nearest_mapped_parents` as the label DAG's parent map, on
main's docstring calling its output "the induced Hasse edges over mapped terms".
It is not, and the gap is the plan's own bug in miniature: nearest-per-branch
lets a DISTANT ancestor in as a parent whenever an intermediate is unpowered,
which re-manufactures the subsumed-sibling shape (`induced_hasse_parents` has
the three-node worked example and the arithmetic). The reduction step is added
here, in this branch's new module; main's dashboard is unaffected, since it
never turns that map into label cells. The acceptance property — no kept node is
a sibling of its own descendant — is a unit test, not an aspiration.

ENGINE ID SPACE (a deviation from the plan, recorded)
-----------------------------------------------------
Plan §3 says "`int2cid` maps engine ints to Mondo curies (strings)". That is NOT
implementable without editing source-hashed modules, and the code says so in
three places:

  * `case_finding_assembly.attach_frontiers` hard-casts every attested id with
    `int(c)` before `roll_up_to_survivors` looks it up in the DAG's node set
    (`:269-271`); string node ids would miss on every lookup and every document's
    frontier would collapse to the root;
  * `doc_attested_nodes` / the attestation contract type the column as
    `array<bigint>` (`:218,:232-240`), and `multi_domain._attest` feeds it
    straight into that same cast;
  * `gated_pc_readout.bundle_drift_report` re-reads the map as
    `{int(i): int(c) ...}` (`:416`), so the drift gate itself is int-typed.

All three live in modules whose source hashes are folded into every bundle cache
key (`_case_finding_cache.py:112-113,133`), and editing any of them orphans exp
0104's and 0109's cached bundles and moves four pinned tripwire hashes. So the
engine id space here is **stable positive integers = the Mondo curie's numeric
part** (`MONDO:0004995` -> `4995`), with the forest root at `-1` (the
`_FOREST_ROOT_CID` / `MONDO_ROOT_CID` convention). The encoding is:

  * INJECTIVE — Mondo ids are `MONDO:%07d`, so the numeric part identifies the
    term;
  * STABLE — it depends only on the curie, not on the kept set or an enumeration
    order, unlike `mondo_dag`'s sorted synthetic negatives, so two builds of
    different kept sets agree on what node `4995` is;
  * REVERSIBLE — `mondo_curie(cid)` reconstructs the curie for reports, and
    `name_by_id` carries the human label the readout actually renders.

There is no collision with the OMOP concept ids the legacy Mondo path uses,
because on this path NO node is keyed by an OMOP concept id: the label space is
Mondo end to end. `ConditionDag` / `build_condition_dag` / `DagLayout` are
id-agnostic (sorting and dict lookups only) and are used unmodified.

THE SOURCE-EXACT RUNG IS OFF, AND NOT BY CHOICE (deviation, recorded)
---------------------------------------------------------------------
Main's ladder has three rungs; this one has two. Rung 1 (source-exact:
`condition_source_concept_id` is one of the term's own Mondo `same_as` codes)
discriminates two rows that share a standard concept but differ in their source
code, so it can only be applied to a frame that CARRIES the source concept id.
The corpus's label frame does not and structurally cannot: it is built from
`condition_era` (`multi_domain.assemble_multidomain_case_finding_corpus:365-368`),
and OMOP's `condition_era` has no source-concept column at all — which is why
main's own driver refuses the source spaces unless `--source-table
condition_occurrence`. Adding the column would mean editing `cohorts.py`
(`lookback_feature_label_events`), the maximum-blast-radius hashed module.

So both rungs that a `condition_era` frame can express are implemented, and the
SAME two-rung resolution powers AND attests. Keeping powering richer than
attestation would be worse than the missing rung: nodes powered only by
source-exact rows would be kept and then never attested, manufacturing exactly
the `no_pos` degenerates the acceptance gate is trying to drive to zero. The
HPO rung stays off by scope (plan §6).

CACHE-KEY DISCIPLINE
--------------------
This module and `mondo_usage_core` are NEW, so their source hashes are free to
move: `_case_finding_cache.compute_bundle_cache_key` folds them ONLY when
`dag_source=mondo_native`, exactly as `dag_collapse` folds `mondo_collapse` only
when it is on. `mondo_dag.py`, `condition_dag.py`, `case_finding_assembly.py`,
`multi_domain.py` and `cohorts.py` are untouched, so every SNOMED key and every
legacy-Mondo key (including exp 0104's record bundle and exp 0109's) is
byte-identical. `mondo_collapse` is folded here too, because this build calls
its splice unconditionally.

BROADCAST DISCIPLINE (ADR 0047)
-------------------------------
Nothing here creates an `sc.broadcast`, so there is nothing to `destroy()`;
every join hint is `F.broadcast(df)`, which is a planner hint, not a broadcast
object (ADR 0047 §2 puts those explicitly out of scope). Nothing array-shaped
rides a task closure: the per-document attestation is a JOIN against a small
code-map DataFrame, not a UDF over a captured dict, and the closure roll-up is a
join against a small `(term, ancestor)` frame built on the driver.
"""
from __future__ import annotations

# The forest root's node id, matching case_finding_assembly._FOREST_ROOT_CID and
# mondo_dag.MONDO_ROOT_CID so this DAG roots exactly like the others (engine-id 0).
# Mondo numeric ids are positive, so -1 cannot collide with a real term.
MONDO_NATIVE_ROOT_CID = -1

# Bumped when this build's OUTPUT would change for the same inputs. Folded into
# the bundle cache key alongside this module's source hash: the hash is the
# automatic guard (nobody has to remember), the version string is the citable
# record of WHICH construction a cached bundle was built under.
MONDO_NATIVE_VERSION = "native-mondo-v1"

_MONDO_PREFIX = "MONDO:"


# --------------------------------------------------------------------------- #
# Id space                                                                     #
# --------------------------------------------------------------------------- #
def mondo_cid(mondo_id) -> int:
    """Engine concept-id for a Mondo curie: its numeric part (`MONDO:0004995` ->
    4995). Injective (Mondo ids are `MONDO:%07d`) and independent of the kept set,
    so node identity is stable across builds. See the module docstring for why the
    engine id space is int and not the curie itself."""
    s = str(mondo_id)
    if not s.startswith(_MONDO_PREFIX):
        raise ValueError(f"not a Mondo curie: {mondo_id!r}")
    body = s[len(_MONDO_PREFIX):]
    if not body.isdigit():
        raise ValueError(f"not a Mondo curie: {mondo_id!r}")
    return int(body)


def mondo_curie(cid) -> str:
    """The Mondo curie for an engine concept-id (4995 -> 'MONDO:0004995'). The
    inverse of `mondo_cid`; the root (-1) has no curie and raises."""
    c = int(cid)
    if c < 0:
        raise ValueError(f"{cid!r} is not a Mondo term id (the root has no curie)")
    return f"{_MONDO_PREFIX}{c:07d}"


# --------------------------------------------------------------------------- #
# Pure graph core (no Spark, no pandas — unit-tested in                        #
# tests/test_mondo_native_dag.py)                                              #
# --------------------------------------------------------------------------- #
def parent_adjacency(child_adj) -> dict:
    """`{child: [parents]}` from Mondo's `{parent: [children]}` adjacency (what
    `mondo_to_omop_mapping._disease_child_adjacency` returns). Parents are deduped
    and sorted so every downstream walk is deterministic."""
    out: dict = {}
    for parent, children in child_adj.items():
        for c in children:
            out.setdefault(c, []).append(parent)
    return {c: sorted(set(ps)) for c, ps in out.items()}


def ancestor_closure(term, parent_adj) -> set:
    """The REFLEXIVE is-a closure of `term`: itself plus every transitive parent.
    Set-based, so a diamond is visited once and a (malformed) cycle terminates."""
    out = {term}
    stack = list(parent_adj.get(term, ()))
    while stack:
        p = stack.pop()
        if p in out:
            continue
        out.add(p)
        stack.extend(parent_adj.get(p, ()))
    return out


def closure_rows(terms, parent_adj) -> list:
    """Sorted `(term, ancestor)` pairs covering the reflexive closure of every term
    in `terms` — the small driver-built frame the producer-side support aggregation
    joins the attribution frame against (plan §3: "one aggregation over the
    attribution frame joined to the closure"). Ships as data, never as a task
    closure (ADR 0047's closure clause)."""
    return sorted({(t, a) for t in terms for a in ancestor_closure(t, parent_adj)})


def closure_support(person_terms, parent_adj) -> dict:
    """`{term: distinct persons in its is-a closure}` — the PURE twin of the Spark
    aggregation, for tests and small frames.

    `person_terms` is an iterable of `(person_id, mondo_id)` attribution pairs. A
    person counts toward a node once, no matter how many of its descendants they
    attest, and counts toward EVERY ancestor of every term they attest (that is
    what "closure support" means, and why closure support >= direct support). This
    is the number `min_positives` thresholds — deliberately NOT main's fractional
    1/m per-term count, which is an addability/egress device (plan §2's
    "egress vs internal" split)."""
    memo: dict = {}
    persons: dict = {}
    for person, term in person_terms:
        anc = memo.get(term)
        if anc is None:
            anc = memo[term] = ancestor_closure(term, parent_adj)
        for a in anc:
            persons.setdefault(a, set()).add(person)
    return {a: len(ps) for a, ps in persons.items()}


def nearest_kept_ancestors(term, kept, parent_adj, memo=None) -> set:
    """Where an attestation of `term` LANDS in a DAG built over `kept`: `{term}`
    when it is kept, else the nearest kept ancestor on each upward branch (the walk
    stops climbing a branch at its first kept node), else `set()`.

    The reflexive twin of `nearest_mapped_parents`'s walk and of
    `condition_dag._nearest_surviving_ancestors`'s rewire, so an attestation lands
    exactly where the label DAG reattaches its node — the single-source-of-truth
    property the pruning ledger relies on. Composing it over the Hasse and then the
    splice is why resolving codes against the FINAL node set is correct: nearest
    kept ancestor of nearest kept ancestor is nearest kept ancestor."""
    if term in kept:
        return {term}
    if memo is not None and term in memo:
        return memo[term]
    found, seen = set(), set()
    stack = list(parent_adj.get(term, ()))
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if cur in kept:
            found.add(cur)
        else:
            stack.extend(parent_adj.get(cur, ()))
    if memo is not None:
        memo[term] = found
    return found


def roll_terms_to_kept(terms, kept, parent_adj) -> dict:
    """`{term: sorted landing nodes}` over `terms` via `nearest_kept_ancestors`.

    This is the native analogue of `mondo_dag.powered_anchor_climb`'s restriction
    to the powered terminals, and it ROLLS UP rather than dropping: a code whose
    most-specific Mondo term missed `min_positives` still attests that term's
    nearest kept ancestors instead of vanishing. Because every attested id is then
    a node of the DAG handed to the assembler, `attach_frontiers`'s own roll-up
    never has to fire — an attestation can never fall through to the root by
    accident (which it WOULD, silently, if a spliced-out or unpowered id reached
    it: `_nearest_surviving_ancestors` returns `{root}` for an id the DAG has never
    heard of)."""
    memo: dict = {}
    return {t: sorted(nearest_kept_ancestors(t, kept, parent_adj, memo))
            for t in terms}


def induced_hasse_parents(kept_ids, parent_adj) -> dict:
    """The TRANSITIVE REDUCTION of Mondo's is-a order restricted to `kept_ids`.

    `mondo_usage_core.nearest_mapped_parents` is the first half: for each kept
    term, the first kept node on every upward branch. Its docstring calls that
    "the induced Hasse edges", and on main's dashboard it is — but it is NOT a
    transitive reduction on a multi-path DAG, and the difference is precisely exp
    0104's bug. Worked example, from this module's own unit tests:

        A                A, C, D kept; B unpowered and dropped.
       / \\               D's parents are B and C. Climbing D's LEFT branch stops
      B   C              at A (past the dropped B); the RIGHT branch stops at C.
       \\ /               nearest_mapped_parents(D) = {A, C}.
        D

    A is then a parent of both C and D, so C and D are SIBLINGS — and C is an
    ANCESTOR of D. Under closure masking a doc that fires D observes its siblings
    under every parent, C among them, and C is positive on every such doc: an
    all-positive cell and a constant head. That is the 619-category-anchor
    mechanism reconstructed inside the very construction meant to make it
    impossible, and it appears exactly when an intermediate is dropped — i.e. on
    every unpowered rung of a real ontology.

    So the second half: a parent `p` is REDUNDANT when some other parent `q` of
    the same node has `p` among its own ancestors. Redundant parents are dropped,
    which is what "the transitive reduction deletes the redundant sibling edge"
    means operationally. Ancestry is read from the FULL Mondo graph, which is
    sound because the induced order on kept nodes is the restriction of the full
    one: if `p` is an ancestor of `q` in Mondo and both are kept, the edge `p->c`
    carries no information the path `p ... q -> c` does not.

    Multi-parenthood survives untouched wherever the parents are genuinely
    incomparable (the orthogonal-axis case, ~50% of Mondo terms): only comparable
    pairs are reduced. Returns `{kept term: [kept parents]}`; a term with no kept
    ancestor gets `[]` and attaches to the synthetic root."""
    from mondo_usage_core import nearest_mapped_parents

    kept = {str(k) for k in kept_ids}
    nearest = nearest_mapped_parents(kept, parent_adj)
    memo: dict = {}

    def strict_ancestors(term):
        if term not in memo:
            memo[term] = ancestor_closure(term, parent_adj) - {term}
        return memo[term]

    out: dict = {}
    for child, parents in nearest.items():
        if len(parents) <= 1:
            out[child] = list(parents)
            continue
        out[child] = sorted(
            p for p in parents
            if not any(p in strict_ancestors(q) for q in parents if q != p))
    return out


def resolve_code_terms(standard_pairs, climb_pairs, parent_adj) -> dict:
    """The `source_climb` ladder as a pure `{standard_concept_id: [mondo_id]}` map.

    `standard_pairs` are `(standard_concept_id, mondo_id)` exact mappings (rung 2:
    the term's own `same_as -> Maps to` standard concept). `climb_pairs` are
    `(standard_concept_id, mondo_id)` candidates from rung 3 — the nearest mapped
    standard ANCESTORS produced by
    `mondo_usage_core.nearest_mapped_standard_ancestors`, mapped through the same
    standard->term table. Precedence is main's, verbatim in spirit: an exact hit
    wins outright and is NEVER climbed past; only codes with no exact hit climb,
    and their tie-set is reduced by `mondo_usage_core.reduce_tie_map` to its
    MOST-SPECIFIC Mondo term(s) (nested ancestors dropped, genuine orthogonal ties
    kept).

    Returns `{std_cid: sorted[mondo_id]}`. Rung 1 (source-exact) is absent — see
    the module docstring; a `condition_era` frame cannot express it."""
    from mondo_usage_core import reduce_tie_map

    exact: dict = {}
    for cid, mid in standard_pairs:
        exact.setdefault(int(cid), set()).add(str(mid))
    climb_candidates = [(int(c), str(m)) for c, m in climb_pairs
                        if int(c) not in exact]
    climbed = reduce_tie_map(climb_candidates, parent_adj)
    out = {c: sorted(ms) for c, ms in exact.items()}
    for c, ms in climbed.items():
        if ms:
            out[int(c)] = sorted(ms)
    return out


def build_native_label_dag(kept_ids, parent_adj, coded_ids, *, names=None,
                           root=MONDO_NATIVE_ROOT_CID):
    """The label DAG over a POWERED Mondo term set: induced Hasse, then splice.

    `kept_ids` are the Mondo curies clearing `min_positives` on closure support;
    `parent_adj` is Mondo's own `{child: [parents]}` disease adjacency; `coded_ids`
    are the curies some condition code resolves to DIRECTLY (the node property that
    replaces exp 0109's positive-concept-id terminal test). `names` is
    `{mondo_id: label}`.

    Two steps, both reused rather than rewritten:

      1. `induced_hasse_parents(kept_ids, parent_adj)` — `nearest_mapped_parents`
         followed by the transitive reduction that makes it an actual Hasse
         diagram (read that function's docstring: the reduction is not optional,
         it is what closes the 619-category-anchor mechanism). Multi-parenthood is
         PRESERVED wherever parents are genuinely incomparable (~50% of Mondo
         terms have more than one parent); only comparable pairs are reduced, and
         a transitive reduction cannot leave a node as a sibling of its own
         descendant.
      2. `mondo_collapse.collapse_engine_dag` — exp 0109's splice-to-fixpoint,
         applied as a generic thin-chain post-pass with `is_terminal` = "directly
         coded". A non-coded node with exactly one kept child is a rung in a
         ladder, not a label, and its would-be attestations simply roll one level
         further up (`roll_terms_to_kept` is computed against the POST-splice node
         set, so nothing is stranded).

    Returns `(dag, stats)`; `stats` carries both stages' counts plus the splice's
    own `predicted_degenerate` (the number the readout banner then confirms or
    refutes)."""
    from charmpheno.omop.condition_dag import build_condition_dag
    from mondo_collapse import collapse_engine_dag

    names = {str(k): str(v) for k, v in (names or {}).items()}
    kept = {str(k) for k in kept_ids}
    coded = {str(c) for c in coded_ids}

    hasse = induced_hasse_parents(kept, parent_adj)
    edges, node_ids = set(), set()
    for child, parents in hasse.items():
        ci = mondo_cid(child)
        node_ids.add(ci)
        if parents:
            for p in parents:
                pi = mondo_cid(p)
                edges.add((pi, ci))
                node_ids.add(pi)
        else:
            edges.add((root, ci))
    node_names = {root: "mondo disease root"}
    for term in kept:
        node_names[mondo_cid(term)] = names.get(term, term)
    dag = build_condition_dag(sorted(edges), root, sorted(node_ids), node_names)

    coded_cids = {mondo_cid(c) for c in coded if c in kept}
    n_hasse_nodes = len(dag.nodes())
    n_multi_parent = sum(1 for ps in dag.parents.values() if len(ps) > 1)
    collapsed, collapse_stats = collapse_engine_dag(
        dag, is_terminal=lambda cid: cid in coded_cids)
    stats = {
        "version": MONDO_NATIVE_VERSION,
        "n_kept": len(kept),
        "n_coded_kept": len(coded_cids),
        "n_hasse_nodes": n_hasse_nodes,
        "n_hasse_multi_parent": n_multi_parent,
        "n_final_nodes": len(collapsed.nodes()),
        "n_final_multi_parent": sum(1 for ps in collapsed.parents.values()
                                    if len(ps) > 1),
        "collapse": collapse_stats,
    }
    return collapsed, stats


def format_native_build_report(stats) -> str:
    """The one-line DAG-build diagnostic, printed BEFORE any fit so the structural
    claim is on the record ahead of the readout's own banner (the same discipline
    `mondo_collapse.format_collapse_report` follows)."""
    col = stats["collapse"]
    return (
        f"[mondo-native] label DAG ({stats['version']}): {stats['n_kept']} powered "
        f"term(s) ({stats['n_coded_kept']} directly coded); induced Hasse "
        f"{stats['n_hasse_nodes']} node(s), {stats['n_hasse_multi_parent']} with "
        f">1 parent; splice removed {col['spliced']} thin-chain + "
        f"{col['dropped_childless']} childless in {col['passes']} pass(es) -> "
        f"{stats['n_final_nodes']} node(s) "
        f"({stats['n_final_multi_parent']} multi-parent); "
        f"predicted residual degenerate = {col['predicted_degenerate']}")


def format_native_powering_report(stats) -> str:
    """The powering half of the build receipt: how many Mondo terms carry ANY
    closure support, how many clear `min_positives`, and the smallest kept
    support. C and K are expected to GROW versus exp 0104 (closure support >=
    direct support), so these numbers are MEASURED here and cited by the
    experiment doc rather than predicted."""
    return (
        f"[mondo-native] powering: {stats['n_codes_resolved']} standard code(s) "
        f"resolve to {stats['n_coded_terms']} Mondo term(s); "
        f"{stats['n_terms_with_any_support']} term(s) carry closure support, "
        f"{stats['n_powered']} clear min_positives={stats['min_positives']} "
        f"(smallest kept support {stats['min_support_kept']}); "
        f"{stats['n_codes_attesting']} code(s) attest the final DAG"
        + (f"; branch={stats['branch']}" if stats.get("branch") else ""))


# --------------------------------------------------------------------------- #
# Spark / BigQuery build                                                       #
# --------------------------------------------------------------------------- #
def build_mondo_native_fit_inputs(spark, *, cdr, billing,
                                  mondo_version="2026-06-02",
                                  mondo_cache_dir="data/mondo",
                                  min_positives=100, branch_root=None,
                                  condition_source_table="condition_occurrence"):
    """Build the native-Mondo `before_dag` + the per-code attestation map (BQ).

    The native sibling of `mondo_dag.build_mondo_fit_inputs`, same call shape so
    `gated_pc_cloud.mondo_assemble_fn` can dispatch to either:

      1. whole-Mondo -> OMOP mapping (`build_mondo_to_omop`, restrict=None), which
         gives the standard-exact rung and the climb's targets;
      2. the climb rung — `concept_ancestor` restricted to mapped ancestors and to
         the standard concepts that have NO exact term, reduced to nearest by
         `nearest_mapped_standard_ancestors` and tie-reduced by `reduce_tie_map`;
      3. CLOSURE support: distinct persons per node, rolled through Mondo's own
         is-a closure, thresholded at `min_positives`;
      4. the label DAG (`build_native_label_dag`: induced Hasse + splice);
      5. the code map re-resolved against the FINAL node set, so every attestation
         lands on a node the DAG actually has.

    `branch_root` (optional) restricts the kept set to one body-system subtree
    (the template-branch knob `mondo_dag.branch_mondo_id_set` provides); `None` =
    whole Mondo, which is what exp 0110 runs.

    Returns `(before_dag, code_map_sdf, kept_cids, support_of, stats)`:
      before_dag    integer-id `ConditionDag` over Mondo term ids (feed to the
                    multi-domain assembler with `min_n=0`);
      code_map_sdf  `(std_cid, node_cid)` — wrap with
                    `make_mondo_native_attested_provider`;
      kept_cids     the powered node ids surviving the splice;
      support_of    `{node_cid: closure support}` for logging / per-node reports;
      stats         the build receipt (`format_native_build_report`).
    """
    from pathlib import Path

    import pandas as pd
    from pyspark.sql import functions as F
    from pyspark.sql.functions import broadcast

    from charmpheno.omop.bigquery import load_omop_bigquery
    from anchor_selection_cloud import _download_cached, _read_bq
    from mondo_to_omop_mapping import (
        build_mondo_to_omop, seed_source_xrefs, _disease_child_adjacency)
    from mondo_usage_core import nearest_mapped_standard_ancestors

    cache = Path(mondo_cache_dir)
    edges_df = pd.read_csv(_download_cached(mondo_version, "mondo_edges.tsv", cache),
                           sep="\t", low_memory=False)
    nodes_df = pd.read_csv(_download_cached(mondo_version, "mondo_nodes.tsv", cache),
                           sep="\t", low_memory=False)

    # --- 1. whole-Mondo -> OMOP standard-Condition mapping (as exp 0088/0104) ---
    all_ids = set(nodes_df["id"])
    concept_pd = (_read_bq(spark, cdr, billing, "concept")
                  .select("concept_id", "concept_name", "vocabulary_id", "domain_id",
                          "concept_code", "standard_concept")
                  .where(F.col("vocabulary_id").isin("SNOMED", "ICD10CM", "MeSH"))
                  .toPandas())
    same_as = seed_source_xrefs(mondo_edges_df=edges_df, mondo_nodes_df=nodes_df,
                                restrict_mondo_ids=all_ids)
    src = same_as.merge(concept_pd, on=["concept_code", "vocabulary_id"], how="inner")
    source_ids = sorted({int(x) for x in src["concept_id"]})
    src_sdf = spark.createDataFrame(pd.DataFrame({"concept_id_1": source_ids}))
    cr_pd = (_read_bq(spark, cdr, billing, "concept_relationship")
             .select("concept_id_1", "concept_id_2", "relationship_id")
             .where(F.col("relationship_id") == "Maps to")
             .join(broadcast(src_sdf), "concept_id_1", "inner").toPandas())
    mapping = build_mondo_to_omop(
        mondo_edges_df=edges_df, mondo_nodes_df=nodes_df,
        concept_df=concept_pd, concept_relationship_df=cr_pd, restrict_mondo_ids=None)

    child_adj = _disease_child_adjacency(edges_df, nodes_df)
    parent_adj = parent_adjacency(child_adj)
    label_of = {str(i): str(n) for i, n in zip(nodes_df["id"], nodes_df["name"])}

    std_pairs = [(int(c), str(m)) for c, m in
                 zip(mapping["standard_concept_id"], mapping["mondo_id"])]
    mapped_std_ids = sorted({c for c, _ in std_pairs})
    term_of_std: dict = {}
    for c, m in std_pairs:
        term_of_std.setdefault(c, set()).add(m)

    # --- 2. the condition frame + the climb rung -------------------------------
    # Powering reads the WHOLE population (no person_mod): a node's support is a
    # corpus-independent property, the same convention mondo_dag's power-count uses.
    cond = load_omop_bigquery(
        spark=spark, cdr_dataset=cdr, billing_project=billing,
        source_table=condition_source_table).select("person_id", "concept_id").cache()
    try:
        std_ids_sdf = broadcast(spark.createDataFrame(
            pd.DataFrame({"map_cid": mapped_std_ids})))
        # Only codes with NO exact term climb (main's precedence: never climb past
        # an exact hit), and only codes patients actually carry are worth climbing.
        used_std = cond.select(F.col("concept_id").alias("std_cid")).distinct()
        unmatched = used_std.join(std_ids_sdf,
                                  used_std["std_cid"] == std_ids_sdf["map_cid"],
                                  "left_anti")
        ca = (_read_bq(spark, cdr, billing, "concept_ancestor")
              .select("ancestor_concept_id", "descendant_concept_id",
                      "min_levels_of_separation")
              .where(F.col("min_levels_of_separation") >= 1))
        ca_f = (ca.join(std_ids_sdf,
                        ca["ancestor_concept_id"] == std_ids_sdf["map_cid"], "inner")
                  .join(unmatched,
                        ca["descendant_concept_id"] == unmatched["std_cid"], "inner")
                  .select(ca["descendant_concept_id"].alias("d"),
                          ca["ancestor_concept_id"].alias("a"),
                          ca["min_levels_of_separation"].alias("lev"))
                  .distinct().toPandas())
        nearest = nearest_mapped_standard_ancestors(
            (int(r.d), int(r.a), int(r.lev)) for r in ca_f.itertuples(index=False))
        climb_pairs = [(d, m) for d, ancs in nearest.items()
                       for a in ancs for m in term_of_std.get(a, ())]
        code_terms = resolve_code_terms(std_pairs, climb_pairs, parent_adj)

        # --- 3. closure support over Mondo's own graph -------------------------
        # The attribution frame, reduced to distinct (person, term) BEFORE the
        # closure join: persons x terms-per-person is small (~1e7), the raw
        # condition table is not, and the join multiplies rows by closure depth.
        attr_pd = pd.DataFrame(
            [(c, mondo_cid(m)) for c, ms in code_terms.items() for m in ms],
            columns=["std_cid", "node_cid"]).astype({"std_cid": int, "node_cid": int})
        attr_sdf = broadcast(spark.createDataFrame(attr_pd))
        attributed = (cond.join(attr_sdf, cond["concept_id"] == attr_sdf["std_cid"],
                                "inner")
                      .select("person_id", "node_cid").distinct())
        coded_terms = sorted({m for ms in code_terms.values() for m in ms})
        clos_pd = pd.DataFrame(
            [(mondo_cid(t), mondo_cid(a))
             for t, a in closure_rows(coded_terms, parent_adj)],
            columns=["node_cid", "anc_cid"])
        clos_sdf = broadcast(spark.createDataFrame(clos_pd))
        # `.distinct()` then `count()` rather than `countDistinct`: the pair frame
        # is already deduped per (person, node), so this is one shuffle instead of
        # a per-group hash set. The column is renamed off `count` because
        # `DataFrame.itertuples` shadows that name with the namedtuple method.
        support_pd = (attributed.join(clos_sdf, "node_cid", "inner")
                      .select("person_id", "anc_cid").distinct()
                      .groupBy("anc_cid").count()
                      .withColumnRenamed("count", "n_persons").toPandas())
    finally:
        cond.unpersist()

    support_of = {int(r.anc_cid): int(r.n_persons)
                  for r in support_pd.itertuples(index=False)}
    kept_terms = {mondo_curie(c) for c, n in support_of.items()
                  if n >= int(min_positives)}
    if branch_root:
        kept_terms &= branch_mondo_id_set(branch_root, edges_df=edges_df,
                                          nodes_df=nodes_df)

    # --- 4. the label DAG, and 5. the code map against its FINAL node set ------
    before_dag, stats = build_native_label_dag(
        kept_terms, parent_adj, coded_terms, names=label_of)
    final_terms = {mondo_curie(c) for c in before_dag.nodes()
                   if c != MONDO_NATIVE_ROOT_CID}
    landing = roll_terms_to_kept(coded_terms, final_terms, parent_adj)
    rows = sorted({(int(c), mondo_cid(land))
                   for c, ms in code_terms.items() for m in ms
                   for land in landing.get(m, ())})
    code_map_sdf = spark.createDataFrame(
        pd.DataFrame(rows, columns=["std_cid", "node_cid"])
        .astype({"std_cid": int, "node_cid": int}))
    # Measured at BUILD time, not guessed (plan §3 "scale expectations"): C and K
    # are expected to GROW relative to exp 0104's 3,677, because closure support
    # >= direct support and mid-level terms with no code of their own now qualify.
    stats.update(min_positives=int(min_positives),
                 n_codes_resolved=len(code_terms),
                 n_coded_terms=len(coded_terms),
                 n_terms_with_any_support=len(support_of),
                 n_powered=len(kept_terms),
                 min_support_kept=min((support_of[mondo_cid(t)]
                                       for t in kept_terms), default=0),
                 n_codes_attesting=len({c for c, _ in rows}),
                 branch=str(branch_root or ""))
    kept_cids = {c for c in before_dag.nodes() if c != MONDO_NATIVE_ROOT_CID}
    return before_dag, code_map_sdf, kept_cids, support_of, stats


def branch_mondo_id_set(branch_root, *, edges_df, nodes_df) -> set:
    """The Mondo ids of `branch_root` and its is-a descendants — the
    template-branch node set, same helper `mondo_dag.branch_mondo_id_set`
    provides, re-exposed here so this module does not import that hashed one."""
    from mondo_to_omop_mapping import _disease_child_adjacency, _descendants
    child_adj = _disease_child_adjacency(edges_df, nodes_df)
    return ({str(branch_root)}
            | {str(x) for x in _descendants(child_adj, str(branch_root))})


def make_mondo_native_attested_provider(code_map_sdf, *, doc_spec):
    """A `provider(events_df) -> attested_df` for the `attested_provider` seam —
    the native analogue of `mondo_dag.make_mondo_attested_provider`.

    Per document, the attested nodes are the Mondo label nodes the patient's
    in-window condition codes resolve to through the ladder
    (`condition ⋈ code_map on concept_id = std_cid -> node_cid`). Every `node_cid`
    is already a node of the DAG handed to the assembler (the map was re-resolved
    against the final node set), so `attach_frontiers`'s roll-up is a no-op and
    `frontier_from_coded` reduces the set to its most-specific members — the
    frontier, not every powered ancestor.

    A full doc roster is LEFT-joined so background docs (no resolvable condition
    code) survive with an empty `attested_cids` and a `[]` frontier, exactly like
    the SNOMED and legacy-Mondo providers.

    Returns `[doc_id, person_id, source_cohort, attested_cids: array<bigint>]`.
    A join, not a UDF: nothing array-shaped rides the task closure (ADR 0047)."""
    from pyspark.sql import functions as F
    from pyspark.sql.functions import broadcast

    def provider(events_df):
        ev = doc_spec.derive_docs(events_df)
        roster = ev.groupBy("doc_id").agg(
            F.first("person_id").alias("person_id"),
            F.first("source_cohort").alias("source_cohort"),
        )
        attested = (
            ev.join(broadcast(code_map_sdf),
                    ev["concept_id"] == code_map_sdf["std_cid"], "inner")
              .groupBy("doc_id")
              .agg(F.collect_set(F.col("node_cid").cast("long"))
                   .alias("attested_cids"))
        )
        return (
            roster.join(attested, on="doc_id", how="left")
            .withColumn("attested_cids",
                        F.coalesce(F.col("attested_cids"),
                                   F.array().cast("array<bigint>")))
        )

    return provider

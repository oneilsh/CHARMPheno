"""Native-Mondo label space (exp 0110; redesigned by spec 2026-10-30): the Mondo graph,
the code ladder, and the SUBGRAPH a DAG-gated fit is given.

WHAT THIS MODULE HANDS OVER
---------------------------
Two things, both in Mondo's own ids (CURIE strings, `MONDO:%07d`):

  1. **The code ladder** (codes -> Mondo terms). A standard condition concept
     resolves to the MOST SPECIFIC Mondo term(s) it maps to, via main's
     `source_climb` ladder (ported from main's `mondo_usage_cloud.py`; the two
     functions used are at the end of this module): standard-exact first, then a
     `concept_ancestor` climb to the nearest mapped standard ancestor,
     tie-reduced by `reduce_tie_map` to the most-specific Mondo term(s). The
     mapping is guarded before the climb: curated wrong cross-references
     (`EXCLUDED_MONDO_XREFS`), the hierarchy-consistency test
     (`overbroad_exact_rows`) and the multi-map guard
     (`drop_ancestor_multimap_targets`).
  2. **The relevant subgraph** (`subgraph_parents`): every Mondo term some code
     patients carry resolves to, plus all its Mondo ancestors (within
     `branch_root` when set), with Mondo's own is-a edges among them. A
     document's labels are the terms its in-window codes resolve to
     (`make_mondo_native_attested_provider`), i.e. labels at the most-specific
     attested terms.

WHAT IT NO LONGER DOES
----------------------
Powering (closure support >= `min_positives`), the induced Hasse over the kept
set, the landing of unpowered terms on kept ancestors (`roll_terms_to_kept`) and
the thin-chain splice (`collapse`) were the build's until spec 2026-10-30. They are
generic DAG work that depends only on the TRAINING documents, so they moved into
`spark_vi`'s `DagGatedLDAEstimator` (R3: support over the fit DataFrame,
`minDocsPerNode` prune, landing, transitive reduction, splice to a fixpoint). The
bundle therefore no longer depends on a threshold: the fit driver sets
`minDocsPerNode` and the node set is the fitted model's (`dag()`, `prunedNodes()`,
`splicedNodes()`). `induced_hasse_parents` stays for the readout DAG
(`analysis/cloud/readout_dag.py`).

ID SPACE
--------
The handed-over DAG and the label columns use the CURIE strings (`array<string>`
labels). `mondo_cid` (`MONDO:0004995` -> 4995) and `mondo_curie` remain the
integer encoding the readout tables, the code map file and the conversion sidecar
use (injective, stable across builds, reversible); `MONDO_NATIVE_ROOT_CID` (-1) is
the synthetic root in those integer tables. The code map frame carries both
(`node_id` string, `node_cid` long).

THE SOURCE-EXACT RUNG IS OFF, AND NOT BY CHOICE (deviation, recorded)
---------------------------------------------------------------------
Main's ladder has three rungs; this one has two. Rung 1 (source-exact:
`condition_source_concept_id` is one of the term's own Mondo `same_as` codes)
discriminates two rows that share a standard concept but differ in their source
code, so it can only be applied to a frame that CARRIES the source concept id.
The corpus's label frame does not: it is built from `condition_era`, which has no
source-concept column. Both rungs that a `condition_era` frame can express are
implemented. The HPO rung stays off by scope.

CACHE-KEY DISCIPLINE
--------------------
`_case_finding_cache.compute_bundle_cache_key` folds this module's source hash, and
those of `omop_mapping`, `ontology` and `spark_vi.dag.ops`, into every native bundle
key: any edit here is a cache drop (AGENTS.md). `MONDO_NATIVE_VERSION` is the
citable record of which construction a bundle was built under.

BROADCAST DISCIPLINE (ADR 0047)
-------------------------------
Nothing here creates an `sc.broadcast`; every join hint is `F.broadcast(df)`, a
planner hint, not a broadcast object. Nothing array-shaped rides a task closure:
the per-document attestation is a JOIN against the small code-map DataFrame.
"""
from __future__ import annotations

from typing import NamedTuple

# The generic graph primitives (spark_vi.dag.ops) stay importable from here under
# the names this module's callers have always used.
from spark_vi.dag.ops import (ancestor_closure, parent_adjacency,  # noqa: F401
                              transitive_reduction_parents)

# The synthetic root's id in the INTEGER tables (readout DAG, int2cid bridges, the
# conversion sidecar's node ids). Mondo numeric ids are positive, so -1 cannot
# collide with a real term. The handed-over DAG has no such node: the estimator adds
# its own synthetic root above every parentless term.
MONDO_NATIVE_ROOT_CID = -1

# Bumped when this build's OUTPUT would change for the same inputs; folded into the
# bundle cache key beside this module's source hash.
MONDO_NATIVE_VERSION = "native-mondo-v7"   # v2: multi-map guard; v4/v5: hierarchy-consistency test, threshold 3 -> 10; v6: curated xref exclusions (exp 0137); v7: the subgraph handover (spec 2026-10-30: powering and splice moved to the estimator)

_MONDO_PREFIX = "MONDO:"


# --------------------------------------------------------------------------- #
# Id space                                                                     #
# --------------------------------------------------------------------------- #
def mondo_cid(mondo_id) -> int:
    """The integer id of a Mondo curie: its numeric part (`MONDO:0004995` -> 4995).
    Injective (Mondo ids are `MONDO:%07d`) and independent of any node set, so it is
    stable across builds: the id the integer tables (readout DAG, bundle_meta bridge,
    code map file, conversion sidecar) key Mondo terms by. The DAG and labels handed
    to the fit use the curie itself."""
    s = str(mondo_id)
    if not s.startswith(_MONDO_PREFIX):
        raise ValueError(f"not a Mondo curie: {mondo_id!r}")
    body = s[len(_MONDO_PREFIX):]
    if not body.isdigit():
        raise ValueError(f"not a Mondo curie: {mondo_id!r}")
    return int(body)


def mondo_curie(cid) -> str:
    """The Mondo curie for an integer id (4995 -> 'MONDO:0004995'). The inverse of
    `mondo_cid`; the integer tables' root (-1) has no curie and raises."""
    c = int(cid)
    if c < 0:
        raise ValueError(f"{cid!r} is not a Mondo term id (the root has no curie)")
    return f"{_MONDO_PREFIX}{c:07d}"


# --------------------------------------------------------------------------- #
# Pure graph core (no Spark, no pandas — unit-tested in                        #
# tests/test_mondo_native_dag.py)                                              #
# --------------------------------------------------------------------------- #
def subgraph_parents(terms, parent_adj, *, within=None) -> dict:
    """The relevant subgraph handed to the fit: `{term: [parents]}` over every term in
    `terms` and all its Mondo ancestors, with Mondo's own is-a edges among them.

    Why the closure and not just `terms`: a document labelled at a term is evidence
    for every ancestor (the estimator's support is counted over each document's
    label closure), and an ancestor with no code of its own can still collect enough
    documents to earn a topic block. Restricting to `terms` would drop exactly the
    mid-level groupings the closure-support powering used to keep.

    Every parent of a node in the closure is itself in the closure, so the edges are
    complete. `within` (a set of curies: the `branch_root` subtree) restricts both
    the terms and their ancestors; a term outside it is dropped, and the branch root
    becomes parentless. Parentless terms map to `[]` (the estimator hangs them from
    its synthetic root). Parent lists are sorted; ids are kept as given."""
    nodes = set()
    for t in terms:
        t = str(t)
        if within is not None and t not in within:
            continue
        for a in ancestor_closure(t, parent_adj):
            if within is None or a in within:
                nodes.add(a)
    return {n: sorted(p for p in parent_adj.get(n, ()) if p in nodes)
            for n in sorted(nodes)}


def induced_hasse_parents(kept_ids, parent_adj) -> dict:
    """The transitive reduction of Mondo's is-a order restricted to the kept curies
    (`spark_vi.dag.transitive_reduction_parents`, whose docstring explains why the
    nearest-kept-parents walk alone is not enough: exp 0104's 619 category anchors).
    Returns `{kept term: [kept parents]}`; `[]` attaches to the synthetic root."""
    return transitive_reduction_parents({str(k) for k in kept_ids}, parent_adj)


# Mondo `same_as` cross-references found WRONG on inspection (exp 0137 audit), as
# (mondo_id, vocabulary_id, concept_code). Dropped before mapping; each one is a
# Mondo curation issue to report upstream. Kept short and explicit on purpose: the
# hierarchy test REPORTS suspect SNOMED xrefs (`snomed_suspects`) but never drops
# them, so an entry here is a reviewed decision, not an automatic one.
EXCLUDED_MONDO_XREFS = frozenset({
    # 'thrombophilia due to thrombin defect' -> SCTID 111293003 'Venous thrombosis':
    # every DVT code climbed to it and rolled up to thrombophilia (162 codes, 106 DVT).
    ("MONDO:0008559", "SNOMED", "111293003"),
})


def drop_excluded_xrefs(same_as_df, excluded=EXCLUDED_MONDO_XREFS):
    """`same_as_df` (`id`/`mondo_id`, `concept_code`, `vocabulary_id`) minus the
    curated wrong cross-references. Returns `(kept_df, n_dropped)`. Pure pandas."""
    idc = "mondo_id" if "mondo_id" in same_as_df.columns else "id"
    bad = [(str(m), str(v), str(c)) in excluded for m, v, c in
           zip(same_as_df[idc], same_as_df["vocabulary_id"], same_as_df["concept_code"])]
    n = sum(bad)
    return same_as_df[[not b for b in bad]], n


def overbroad_exact_rows(rows, subsumed_terms, parent_adj, *, min_unrelated=10,
                         near_floor=3):
    """HIERARCHY-CONSISTENCY test on exact-map rows (exp 0137, native-mondo-v4). Pure.

    `rows` are `(mondo_id, source_vocabulary_id, source_concept_id,
    standard_concept_id)`. `subsumed_terms` is `{standard_concept_id t: {mondo_id}}`
    — the Mondo terms whose OWN exact concepts are strict SNOMED descendants of t.
    A non-SNOMED row (ICD10CM, MeSH) of term m with target t is dropped when t
    subsumes the exact concepts of at least `min_unrelated` Mondo terms that are
    NOT m's Mondo descendants: SNOMED says t is above them, Mondo says m is not,
    so the mapping put m too high. Rows from Mondo's own SNOMED `same_as` are never
    dropped (they are Mondo's declared equivalent). Returns `(kept, dropped)`.

    Why this and not the anchor test it replaces: 0137 launch 2 (v3) required every
    ICD target to sit at or under the term's SNOMED concept, and dropped dilated
    cardiomyopathy, Down syndrome, Lyme disease, giardiasis and seven more ribbon
    members whose main ICD code maps to a concept that is a sibling or a synonym of
    their SNOMED xref, while leaving tularemia (GI-tract disorders) and
    thrombophilia (every DVT code) untouched. The consistency test asks the
    question those captures actually answer wrongly — is this concept broader than
    the disease, by the disease hierarchy's own account? — and leaves a target
    alone when everything it subsumes is Mondo-below the term (DCM's concept
    subsumes peripartum and familial DCM, which Mondo places under DCM).

    THRESHOLD (launch 3, v4 at 3): the true captures subsume the concepts of tens to
    hundreds of unrelated terms (pregnancy findings, GI-tract disorders, vertebral-
    column disorders), but a disease concept that SNOMED subdivides more finely than
    Mondo also clears 3 — Down syndrome, acute kidney injury, autoimmune
    thrombocytopenic purpura and acne left the label set. Default 10 (v5). Returns
    `(kept, dropped, unrelated_count)`, where `unrelated_count[(m, t)]` is reported
    for every row at or above `near_floor`, so the receipt shows the borderline band
    and the threshold is calibrated on data, not guessed."""
    rows = [(str(m), str(v), int(sc), int(tc)) for m, v, sc, tc in rows]
    memo: dict = {}

    def _is_desc(child, anc):
        if child not in memo:
            memo[child] = ancestor_closure(child, parent_adj)
        return anc in memo[child]

    kept, dropped, counts = [], [], {}
    snomed_suspects = {}
    for r in rows:
        m, v, _sc, tc = r
        n_un = sum(1 for u in subsumed_terms.get(tc, ()) if u != m and not _is_desc(u, m))
        if v == "SNOMED":
            # Mondo's declared equivalent: never dropped, but a failing one is
            # REPORTED (thrombophilia due to thrombin defect -> Venous thrombosis was
            # exactly this) for review into EXCLUDED_MONDO_XREFS.
            if n_un >= int(min_unrelated):
                snomed_suspects[(m, tc)] = n_un
            kept.append(r)
            continue
        if n_un >= int(near_floor):
            counts[(m, tc)] = n_un
        (dropped if n_un >= int(min_unrelated) else kept).append(r)
    overbroad_exact_rows.snomed_suspects = snomed_suspects
    return kept, dropped, counts


def drop_ancestor_multimap_targets(source_targets, ancestor_pairs, *,
                                   terms_of_target=None, parent_adj=None):
    """Drop, per SOURCE code, every 'Maps to' target that is an is-a ancestor of
    another target of the same source code. Pure.

    `source_targets` is `{source_concept_id: {standard_concept_id, ...}}`;
    `ancestor_pairs` the strict `(ancestor, descendant)` concept_ancestor pairs
    among those targets. Returns `(kept {source: set}, dropped [(source, target)])`.

    Why (exp 0136, 2026-10-09): OMOP maps many ICD-10-CM codes to TWO standard
    concepts, the disorder and a broad context finding — `O90.3` "Peripartum
    cardiomyopathy" -> {Dilated peripartum cardiomyopathy, Finding related to
    pregnancy (444094)}. Mondo's `same_as O90.3` then made 444094 an EXACT standard
    concept of peripartum cardiomyopathy, and the climb rung sent every pregnancy
    code without a term of its own (gestation weeks, trimesters, primigravida,
    high-risk pregnancy: 106 codes) up to 444094 and so to the disorder. Every
    pregnant woman in All of Us was labelled peripartum cardiomyopathy — the
    "young-women stratum" of exps 0127-0134's DCM block. The broad target is the
    one an is-a ancestor of its sibling target; the specific one is the
    disorder the source code names, so keeping only the most specific targets of
    each source code removes the context finding and keeps the disorder. A source
    code whose targets are incomparable keeps them all.

    FAN-IN (main's insight 0076, 2026-08-24, which found this same O90.3 case in the
    usage export): `terms_of_target` (`{standard_concept_id: {mondo_id}}` over ALL
    exact pairs) plus Mondo's `parent_adj` add a second test that does not depend on
    SNOMED placing the context finding above the disorder. A multi-map target that is
    the exact concept of two or more Mondo terms neither of which is an is-a ancestor
    of the other (444094 <- peripartum CM, preeclampsia, severe pre-eclampsia) is a
    shared context concept, not any one disease's, and is dropped from that source.
    Applied ONLY to targets of multi-target source codes: a 1:1 map is never touched."""
    anc = {}
    for a, d in ancestor_pairs:
        anc.setdefault(int(d), set()).add(int(a))
    shared = set()
    if terms_of_target is not None and parent_adj is not None:
        memo: dict = {}

        def _anc(t):
            if t not in memo:
                memo[t] = ancestor_closure(t, parent_adj)
            return memo[t]

        for t, terms in terms_of_target.items():
            ts = sorted(terms)
            if any(b not in _anc(a) and a not in _anc(b)
                   for i, a in enumerate(ts) for b in ts[i + 1:]):
                shared.add(int(t))
    kept, dropped = {}, []
    for src, tg in source_targets.items():
        tg = {int(t) for t in tg}
        broad = {t for t in tg if any(t in anc.get(u, ()) for u in tg if u != t)}
        if len(tg) > 1:
            broad |= (tg & shared)
        kept[int(src)] = tg - broad
        dropped.extend((int(src), t) for t in sorted(broad))
    return kept, dropped


def code_map_audit(rows, exact_codes, *, names=None, top=12):
    """The attestation audit printed at build time (exp 0136 / insight 0097): the
    per-label-node attesting-code counts and the codes shared across label nodes,
    split by mechanism as main's usage dashboard classifies collisions
    (`classify_collision_kinds`): `exact` = the code is a Mondo term's own exact
    standard concept, `climb` = it reached its nodes only through the climb. Counts
    of codes and nodes only; no patient figure. Pure.

    `rows` are the final `(std_cid, node_cid)` pairs; `exact_codes` the standard
    concepts some term carries exactly; `names` `{node_cid: label}`."""
    names = names or {}
    nodes_of: dict = {}
    codes_of: dict = {}
    for c, n in rows:
        nodes_of.setdefault(int(c), set()).add(int(n))
        codes_of.setdefault(int(n), set()).add(int(c))
    shared = {c for c, ns in nodes_of.items() if len(ns) > 1}
    exact = {int(c) for c in exact_codes}
    top_nodes = sorted(codes_of.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:int(top)]
    return {
        "n_codes": len(nodes_of), "n_nodes": len(codes_of),
        "n_shared_codes": len(shared),
        "n_shared_exact": len(shared & exact),
        "n_shared_climb": len(shared - exact),
        "n_nodes_sharing": len({n for c in shared for n in nodes_of[c]}),
        "top_nodes": [(names.get(n, str(n)), len(cs)) for n, cs in top_nodes],
    }


def format_code_map_audit(audit) -> str:
    return (f"[mondo-native] attestation audit: {audit['n_codes']} code(s) -> "
            f"{audit['n_nodes']} label node(s); {audit['n_shared_codes']} code(s) attest "
            f">= 2 nodes ({audit['n_shared_exact']} exact-shared, "
            f"{audit['n_shared_climb']} climb-tie) across {audit['n_nodes_sharing']} "
            f"node(s); most-attested: "
            + " · ".join(f"{nm} {k}" for nm, k in audit["top_nodes"]))


def resolve_code_terms(standard_pairs, climb_pairs, parent_adj) -> dict:
    """The `source_climb` ladder as a pure `{standard_concept_id: [mondo_id]}` map.

    `standard_pairs` are `(standard_concept_id, mondo_id)` exact mappings (rung 2:
    the term's own `same_as -> Maps to` standard concept). `climb_pairs` are
    `(standard_concept_id, mondo_id)` candidates from rung 3 — the nearest mapped
    standard ANCESTORS produced by
    `nearest_mapped_standard_ancestors` (below), mapped through the same
    standard->term table. Precedence is main's, verbatim in spirit: an exact hit
    wins outright and is NEVER climbed past; only codes with no exact hit climb,
    and their tie-set is reduced by `reduce_tie_map` (below) to its
    MOST-SPECIFIC Mondo term(s) (nested ancestors dropped, genuine orthogonal ties
    kept).

    Returns `{std_cid: sorted[mondo_id]}`. Rung 1 (source-exact) is absent — see
    the module docstring; a `condition_era` frame cannot express it."""

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



def _dropped_examples(dropped, concept_pd, top=10):
    """The most-dropped multi-map targets by number of source codes, named, for
    the receipt: [(concept_name, n_source_codes)]."""
    from collections import Counter
    if not dropped:
        return []
    name = dict(zip(concept_pd["concept_id"].astype(int), concept_pd["concept_name"]))
    cnt = Counter(t for _, t in dropped)
    return [(str(name.get(t, t)), int(k)) for t, k in cnt.most_common(int(top))]




def format_native_build_report(stats) -> str:
    """The one-line subgraph diagnostic, printed BEFORE any fit so the shape of what
    the estimator is given is on the record (the node set itself is the fit's: the
    estimator's own log line reports kept / pruned / spliced)."""
    return (
        f"[mondo-native] label subgraph ({stats['version']}): {stats['n_coded_terms']} "
        f"coded term(s) + ancestors -> {stats['n_nodes']} node(s), "
        f"{stats['n_multi_parent']} with >1 parent, {stats['n_top']} top-level"
        + (f"; branch={stats['branch']}" if stats.get("branch") else "")
        + "; powering and splice are the estimator's (minDocsPerNode)")


def format_native_mapping_report(stats) -> str:
    """The code-ladder half of the build receipt: how many standard codes resolve to
    how many Mondo terms, and what the mapping guards dropped."""
    return (
        f"[mondo-native] code ladder: {stats['n_codes_resolved']} standard code(s) "
        f"patients carry resolve to {stats['n_coded_terms']} Mondo term(s)"
        + (f"; branch={stats['branch']} ({stats['n_codes_outside_branch']} code(s) "
           f"outside it dropped)" if stats.get("branch") else "")
        + (f"; multi-map guard: {stats['n_multimap_sources']} source code(s) map to "
           f">1 standard concept, {stats['n_multimap_dropped']} ancestor target(s) "
           f"dropped ({len(stats.get('multimap_dropped_targets') or [])} distinct)"
           + (": " + " · ".join(f"{nm} x{k}" for nm, k in
                                stats.get("multimap_dropped_examples") or [])
              if stats.get("multimap_dropped_examples") else "")
           if "n_multimap_sources" in stats else "")
        + (f"; hierarchy test: {stats['n_overbroad_dropped']} over-broad non-SNOMED "
           f"target(s) dropped over {stats['n_overbroad_terms']} term(s)"
           + (": " + " · ".join(f"{nm} (unrelated {k})" for nm, k in
                                stats.get("overbroad_dropped_examples") or [])
              if stats.get("overbroad_dropped_examples") else "")
           + ("; KEPT below threshold: " + " · ".join(
                f"{nm} ({k})" for nm, k in stats.get("overbroad_near_kept") or [])
              if stats.get("overbroad_near_kept") else "")
           if "n_overbroad_dropped" in stats else "")
        + (f"; curated xref exclusions applied: {stats['n_xref_excluded']} row(s)"
           if "n_xref_excluded" in stats else "")
        + ("\n[mondo-native] SUSPECT Mondo SNOMED xrefs (kept; review for "
           "EXCLUDED_MONDO_XREFS): " + " · ".join(
               f"{m} -> {c} (unrelated {n})" for m, c, n in stats["snomed_xref_suspects"])
           if stats.get("snomed_xref_suspects") else "")
        + ("\n" + format_code_map_audit(stats["code_map_audit"])
           if stats.get("code_map_audit") else ""))




# --------------------------------------------------------------------------- #
# Spark / BigQuery build                                                       #
# --------------------------------------------------------------------------- #
class MondoLabelInputs(NamedTuple):
    """What `build_mondo_native_fit_inputs` hands over (a tuple, so it unpacks).

    dag           `{curie: [parent curies]}`: the relevant subgraph
                  (`subgraph_parents`); `[]` = top-level (hangs from the
                  estimator's synthetic root).
    code_map_sdf  `(std_cid long, node_id string, node_cid long)`: each standard
                  code patients carry -> the most-specific Mondo term(s) it resolves
                  to (`node_id` the curie, `node_cid` its `mondo_cid`). Wrap with
                  `make_mondo_native_attested_provider`.
    node_ids      the subgraph's curies (sorted).
    names         `{curie: Mondo label}` over the subgraph.
    stats         the build receipt (`format_native_mapping_report`,
                  `format_native_build_report`); counts of codes and terms only.
    """
    dag: dict
    code_map_sdf: object
    node_ids: list
    names: dict
    stats: dict


def build_mondo_native_fit_inputs(spark, *, cdr, billing,
                                  mondo_version="2026-06-02",
                                  mondo_cache_dir="data/mondo",
                                  branch_root=None,
                                  condition_source_table="condition_occurrence"
                                  ) -> MondoLabelInputs:
    """Build the code ladder and the label subgraph (BigQuery + Spark).

      1. whole-Mondo -> OMOP mapping (`build_mondo_to_omop`, restrict=None) with the
         curated exclusions, the hierarchy-consistency test and the multi-map guard:
         the standard-exact rung and the climb's targets;
      2. the climb rung — `concept_ancestor` restricted to mapped ancestors and to
         the standard concepts that have NO exact term, reduced to nearest by
         `nearest_mapped_standard_ancestors` and tie-reduced by `reduce_tie_map`;
      3. the codes patients actually carry (the WHOLE population's condition table,
         no person_mod: the ladder is a corpus-independent property), restricted to
         `branch_root`'s subtree when set;
      4. the subgraph: those codes' terms and all their Mondo ancestors
         (`subgraph_parents`).

    No threshold: which nodes get a topic block is learned by the estimator from
    the training documents (`minDocsPerNode`), so the same build serves every fit.
    `branch_root` (optional) restricts to one body-system subtree; `None` = whole
    Mondo.
    """
    from pathlib import Path

    import pandas as pd
    from pyspark.sql import functions as F
    from pyspark.sql.functions import broadcast

    from charmpheno.omop.bigquery import load_omop_bigquery
    from charmpheno.mondo.ontology import download_cached as _download_cached
    from charmpheno.mondo.ontology import read_bq as _read_bq
    from charmpheno.mondo.omop_mapping import (
        build_mondo_to_omop, seed_source_xrefs, _disease_child_adjacency)

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
    same_as, n_xref_excluded = drop_excluded_xrefs(same_as)
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
    # `build_mondo_to_omop` re-derives same_as from the nodes, so the curated
    # exclusions are applied to its output rows too (source vocabulary + code).
    mapping, n_map_excluded = drop_excluded_xrefs(mapping)

    child_adj = _disease_child_adjacency(edges_df, nodes_df)
    parent_adj = parent_adjacency(child_adj)
    label_of = {str(i): str(n) for i, n in zip(nodes_df["id"], nodes_df["name"])}

    # --- 1a. hierarchy-consistency test (v4; `overbroad_exact_rows`): drop a
    # non-SNOMED exact target that SNOMED places above the exact concepts of >= 3
    # Mondo terms that Mondo does not place under the term.
    all_tg = sorted({int(x) for x in mapping["standard_concept_id"]})
    cand_tg = sorted({int(x) for x in
                      mapping.loc[mapping["vocabulary_id"] != "SNOMED",
                                  "standard_concept_id"]})
    overbroad_dropped: list = []
    overbroad_top: list = []
    overbroad_near: list = []
    snomed_suspects: list = []
    if cand_tg:
        t2m = spark.createDataFrame(pd.DataFrame(
            {"d_cid": [int(x) for x in mapping["standard_concept_id"]],
             "d_term": [str(x) for x in mapping["mondo_id"]]}).drop_duplicates())
        c_sdf = broadcast(spark.createDataFrame(pd.DataFrame({"a_cid": cand_tg})))
        t_sdf = broadcast(spark.createDataFrame(pd.DataFrame({"t_cid": all_tg})))
        sub_pd = (_read_bq(spark, cdr, billing, "concept_ancestor")
                  .where(F.col("min_levels_of_separation") >= 1)
                  .join(c_sdf, F.col("ancestor_concept_id") == F.col("a_cid"), "inner")
                  .join(t_sdf, F.col("descendant_concept_id") == F.col("t_cid"), "inner")
                  .join(broadcast(t2m), F.col("descendant_concept_id") == F.col("d_cid"),
                        "inner")
                  .groupBy("a_cid").agg(F.collect_set("d_term").alias("terms"))
                  .where(F.size("terms") >= 3)
                  .toPandas())
        subsumed = {int(a): set(ts) for a, ts in zip(sub_pd["a_cid"], sub_pd["terms"])}
        _rows = list(zip(mapping["mondo_id"], mapping["vocabulary_id"],
                         mapping["concept_id"], mapping["standard_concept_id"]))
        _kept, overbroad_dropped, _ob_counts = overbroad_exact_rows(
            _rows, subsumed, parent_adj)
        _sus = getattr(overbroad_exact_rows, "snomed_suspects", {}) or {}
        snomed_suspects = sorted(
            ((label_of.get(m, m), str(dict(zip(concept_pd["concept_id"].astype(int),
                                               concept_pd["concept_name"])).get(t, t)), n)
             for (m, t), n in _sus.items()), key=lambda x: -x[2])[:12]
        _cname = dict(zip(concept_pd["concept_id"].astype(int), concept_pd["concept_name"]))
        _by_t: dict = {}
        for (m, t), n in _ob_counts.items():
            _by_t[t] = max(n, _by_t.get(t, 0))
        overbroad_top = [(str(_cname.get(t, t)), n) for t, n in
                         sorted(_by_t.items(), key=lambda kv: -kv[1]) if n >= 10][:15]
        overbroad_near = [(str(_cname.get(t, t)), n) for t, n in
                          sorted(_by_t.items(), key=lambda kv: -kv[1]) if n < 10][:20]
        if overbroad_dropped:
            _bad = {(m, int(sc), int(tc)) for m, _v, sc, tc in overbroad_dropped}
            mapping = mapping[[(str(m), int(sc), int(tc)) not in _bad for m, sc, tc in
                               zip(mapping["mondo_id"], mapping["concept_id"],
                                   mapping["standard_concept_id"])]]

    # --- 1b. multi-map guard (v2): per source code, drop 'Maps to' targets that
    # are is-a ancestors of another target (`drop_ancestor_multimap_targets`).
    src_tg: dict = {}
    for sc, tc in zip(mapping["concept_id"], mapping["standard_concept_id"]):
        src_tg.setdefault(int(sc), set()).add(int(tc))
    multi_std = sorted({t for tg in src_tg.values() if len(tg) > 1 for t in tg})
    multimap_dropped: list = []
    if multi_std:
        ms_sdf = broadcast(spark.createDataFrame(pd.DataFrame({"t": multi_std})))
        ms2 = ms_sdf.withColumnRenamed("t", "t2")
        anc_pd = (_read_bq(spark, cdr, billing, "concept_ancestor")
                  .where(F.col("min_levels_of_separation") >= 1)
                  .join(ms_sdf, F.col("ancestor_concept_id") == F.col("t"), "inner")
                  .join(ms2, F.col("descendant_concept_id") == F.col("t2"), "inner")
                  .select("ancestor_concept_id", "descendant_concept_id")
                  .distinct().toPandas())
        terms_of_tg: dict = {}
        for tc, mid in zip(mapping["standard_concept_id"], mapping["mondo_id"]):
            terms_of_tg.setdefault(int(tc), set()).add(str(mid))
        kept_tg, multimap_dropped = drop_ancestor_multimap_targets(
            src_tg, zip(anc_pd["ancestor_concept_id"], anc_pd["descendant_concept_id"]),
            terms_of_target=terms_of_tg, parent_adj=parent_adj)
        if multimap_dropped:
            bad = set(multimap_dropped)
            keep_row = [(int(sc), int(tc)) not in bad for sc, tc in
                        zip(mapping["concept_id"], mapping["standard_concept_id"])]
            mapping = mapping[keep_row]
    std_pairs = [(int(c), str(m)) for c, m in
                 zip(mapping["standard_concept_id"], mapping["mondo_id"])]
    mapped_std_ids = sorted({c for c, _ in std_pairs})
    term_of_std: dict = {}
    for c, m in std_pairs:
        term_of_std.setdefault(c, set()).add(m)

    # --- 2. the condition frame + the climb rung -------------------------------
    # The WHOLE population (no person_mod): which codes are in use is a
    # corpus-independent property, so every corpus of this CDR gets the same ladder.
    cond = load_omop_bigquery(
        spark=spark, cdr_dataset=cdr, billing_project=billing,
        source_table=condition_source_table).select("person_id", "concept_id")
    used_std = cond.select(F.col("concept_id").alias("std_cid")).distinct().cache()
    try:
        std_ids_sdf = broadcast(spark.createDataFrame(
            pd.DataFrame({"map_cid": mapped_std_ids})))
        # Only codes with NO exact term climb (main's precedence: never climb past
        # an exact hit), and only codes patients actually carry are worth climbing.
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

        # --- 3. the codes patients carry ----------------------------------------
        # The exact rung resolves every mapped code, carried or not; the subgraph
        # is built from carried codes only, so no term enters it that no patient
        # can attest (it would be pruned at fit time anyway, and it would bloat the
        # DAG every fit is given).
        keys_sdf = broadcast(spark.createDataFrame(
            pd.DataFrame({"k": sorted(code_terms)}).astype({"k": int})))
        carried = {int(r[0]) for r in
                   used_std.join(keys_sdf, used_std["std_cid"] == keys_sdf["k"],
                                 "inner").select("std_cid").collect()}
    finally:
        used_std.unpersist()
    code_terms = {c: ts for c, ts in code_terms.items() if c in carried}

    within = None
    n_outside = 0
    if branch_root:
        within = branch_mondo_id_set(branch_root, edges_df=edges_df, nodes_df=nodes_df)
        restricted = {c: [t for t in ts if t in within] for c, ts in code_terms.items()}
        n_outside = sum(1 for ts in restricted.values() if not ts)
        code_terms = {c: ts for c, ts in restricted.items() if ts}

    # --- 4. the subgraph ---------------------------------------------------------
    coded_terms = sorted({t for ts in code_terms.values() for t in ts})
    dag = subgraph_parents(coded_terms, parent_adj, within=within)
    names = {n: label_of.get(n, n) for n in dag}
    rows = sorted({(int(c), str(t)) for c, ts in code_terms.items() for t in ts})
    code_map_sdf = spark.createDataFrame(
        pd.DataFrame({"std_cid": [c for c, _ in rows],
                      "node_id": [t for _, t in rows],
                      "node_cid": [mondo_cid(t) for _, t in rows]},
                     columns=["std_cid", "node_id", "node_cid"])
        .astype({"std_cid": "int64", "node_id": str, "node_cid": "int64"}))
    stats = {
        "version": MONDO_NATIVE_VERSION,
        "n_codes_resolved": len(code_terms),
        "n_coded_terms": len(coded_terms),
        "n_nodes": len(dag),
        "n_multi_parent": sum(1 for ps in dag.values() if len(ps) > 1),
        "n_top": sum(1 for ps in dag.values() if not ps),
        "branch": str(branch_root or ""),
        "n_codes_outside_branch": int(n_outside),
        "n_multimap_sources": sum(1 for tg in src_tg.values() if len(tg) > 1),
        "n_multimap_dropped": len(multimap_dropped),
        "multimap_dropped_targets": sorted({t for _, t in multimap_dropped}),
        "multimap_dropped_examples": _dropped_examples(multimap_dropped, concept_pd),
        "n_overbroad_dropped": len(overbroad_dropped),
        "n_overbroad_terms": len({m for m, *_ in overbroad_dropped}),
        "overbroad_dropped_examples": overbroad_top,
        "overbroad_near_kept": overbroad_near,
        "n_xref_excluded": int(n_map_excluded),
        "snomed_xref_suspects": snomed_suspects,
        "code_map_audit": code_map_audit(
            [(c, mondo_cid(t)) for c, t in rows], {c for c, _ in std_pairs},
            names={mondo_cid(t): label_of.get(t, t) for t in coded_terms}),
    }
    return MondoLabelInputs(dag=dag, code_map_sdf=code_map_sdf, node_ids=sorted(dag),
                            names=names, stats=stats)


def branch_mondo_id_set(branch_root, *, edges_df, nodes_df) -> set:
    """The Mondo ids of `branch_root` and its is-a descendants — the
    template-branch node set (the removed anchor build had the same helper)."""
    from charmpheno.mondo.omop_mapping import _disease_child_adjacency
    from spark_vi.dag.ops import descendants as _descendants
    child_adj = _disease_child_adjacency(edges_df, nodes_df)
    return ({str(branch_root)}
            | {str(x) for x in _descendants(child_adj, str(branch_root))})




def make_mondo_native_attested_provider(code_map_sdf, *, doc_spec):
    """A `provider(events_df) -> attested_df` for the assembler's `attested_provider`
    seam.

    Per document, the labels are the Mondo terms the patient's in-window condition
    codes resolve to through the ladder (`condition ⋈ code_map on concept_id =
    std_cid -> node_id`): the most-specific term per code, in Mondo curies. Not
    reduced across codes (a document may carry a term and, through another code, its
    ancestor): the estimator's label contract allows ancestors, and its frontier
    reduction is internal.

    A full doc roster is LEFT-joined so background docs (no resolvable condition
    code) survive with an empty `labels` array.

    Returns `[doc_id, person_id, source_cohort, labels: array<string>]`. Joins, not
    UDFs: nothing array-shaped rides the task closure (ADR 0047)."""
    from pyspark.sql import functions as F
    from pyspark.sql.functions import broadcast

    def provider(events_df):
        cm = code_map_sdf.select("std_cid",
                                 F.col("node_id").cast("string").alias("node_id"))
        ev = doc_spec.derive_docs(events_df)
        roster = ev.groupBy("doc_id").agg(
            F.first("person_id").alias("person_id"),
            F.first("source_cohort").alias("source_cohort"),
        )
        pair_df = (ev.join(broadcast(cm), ev["concept_id"] == cm["std_cid"], "inner")
                     .select("doc_id", "node_id").distinct())
        attested = (pair_df.groupBy("doc_id")
                    .agg(F.array_sort(F.collect_set("node_id")).alias("labels")))
        return (
            roster.join(attested, on="doc_id", how="left")
            .withColumn("labels",
                        F.coalesce(F.col("labels"), F.array().cast("array<string>")))
        )

    return provider


# --------------------------------------------------------------------------- #
# The code ladder's climb (from analysis/cloud/mondo_usage_core.py, ADR 0051:   #
# the two functions of that port the native build uses).                       #
# --------------------------------------------------------------------------- #
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

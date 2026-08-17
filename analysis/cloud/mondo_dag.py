"""Whole-Mondo label DAG for a gated-PC fit (the engine DAG + per-patient frontier).

exp 0088 (`mondo_hierarchy_cloud`) proved the whole-Mondo powered hierarchy is a
clean K≈3,800 tree; exp 0089 proved the localized head fits it. This module turns
that hierarchy into the two SNOMED-specific seams a fit needs, so the multi-domain
assembler (`charmpheno.omop.multi_domain`) can swap them in via its `before_dag` /
`attested_provider` overrides and reuse EVERYTHING else (split, prune, ledger,
frontier roll-up, N-domain BOW, strip, labels) verbatim:

  1. **the engine DAG** (`build_mondo_engine_dag`): the Mondo powered hierarchy
     as an integer-id `ConditionDag`, so `to_engine()` / `DagLayout` / the
     localized head all just work.

  2. **the per-patient frontier** (`make_mondo_attested_provider`): a patient's
     attested Mondo nodes = their SNOMED condition codes rolled up to mapped Mondo
     anchors (the SNOMED-climb), replacing the SNOMED path's exact-node
     attestation.

Id space (the crux): a terminal Mondo anchor is keyed by its **OMOP standard-
condition concept id** (positive), so the climb attestation — which lands on OMOP
anchor cids — matches terminal node ids directly with no remap. Class nodes
(abstract Mondo branch points; no OMOP code, never directly attested) get distinct
**synthetic negative ids**. The forest root is `-1` (the `_FOREST_ROOT_CID`
convention). Positive-vs-negative cleanly separates the two node kinds and cannot
collide (OMOP concept ids are positive).

The Mondo mapping/reduction reuse the anchor-selection suite verbatim
(`mondo_to_omop_mapping`, `anchor_hierarchy.reduce_to_anchor_hierarchy`) — the
same code exp 0088 ran.
"""
from __future__ import annotations

from charmpheno.omop.condition_dag import build_condition_dag

# The forest root's concept-id, matching case_finding_assembly._FOREST_ROOT_CID so
# the Mondo DAG roots exactly like the SNOMED multi-anchor forest (engine-id 0).
MONDO_ROOT_CID = -1

_ANCHOR_PREFIX = "anchor:"


def _anchor_cid(node: str) -> int:
    """OMOP concept id of a terminal node id 'anchor:{cid}'."""
    return int(node[len(_ANCHOR_PREFIX):])


def build_mondo_engine_dag(parent_of, *, anchor_names=None, class_names=None,
                           root=MONDO_ROOT_CID):
    """Integer-id `ConditionDag` from a reduced Mondo hierarchy.

    `parent_of` is the `reduce_to_anchor_hierarchy` result's `parent_of`: a
    ``{node: [parent nodes]}`` map whose keys are terminal ids ``'anchor:{cid}'``
    (cid = OMOP standard-condition concept id) and class ids ``'MONDO:xxxxxxx'``.
    A node with no kept parent maps to ``[]`` (it attaches to the synthetic root).

    Returns a `ConditionDag` in integer id space:
      - terminal ``'anchor:{cid}'`` -> ``cid`` (positive OMOP concept id), so a
        patient's SNOMED-climb attestation matches terminal node ids directly;
      - class ``'MONDO:xxxxxxx'`` -> a distinct SYNTHETIC NEGATIVE id, assigned in
        sorted-Mondo-id order so the engine remap (`to_engine`) is reproducible;
      - the forest root -> ``root`` (-1).

    `anchor_names` (``{cid: name}``) and `class_names` (``{mondo_id: name}``) label
    the nodes for interpretation (defaults to the id string).
    """
    anchor_names = {int(k): str(v) for k, v in (anchor_names or {}).items()}
    class_names = {str(k): str(v) for k, v in (class_names or {}).items()}

    # Deterministic synthetic ids for the class (Mondo) nodes: sorted, then
    # -2, -3, ... ( -1 is the root; positive ids are the anchor terminals).
    class_ids = sorted({n for n in _all_nodes(parent_of)
                        if not n.startswith(_ANCHOR_PREFIX)})
    class2int = {m: -(k + 2) for k, m in enumerate(class_ids)}

    def _to_int(node: str) -> int:
        return _anchor_cid(node) if node.startswith(_ANCHOR_PREFIX) else class2int[node]

    edges = set()
    node_ids = set()
    for node, parents in parent_of.items():
        child = _to_int(node)
        node_ids.add(child)
        if parents:
            for p in parents:
                edges.add((_to_int(p), child))
                node_ids.add(_to_int(p))
        else:
            edges.add((root, child))

    names = {root: "mondo disease root"}
    for node in _all_nodes(parent_of):
        i = _to_int(node)
        if node.startswith(_ANCHOR_PREFIX):
            names[i] = anchor_names.get(_anchor_cid(node), node)
        else:
            names[i] = class_names.get(node, node)

    return build_condition_dag(sorted(edges), root, sorted(node_ids), names)


def _all_nodes(parent_of) -> set:
    """Every node id referenced in a parent_of map (keys + parent values)."""
    nodes = set(parent_of)
    for parents in parent_of.values():
        nodes.update(parents)
    return nodes


def mondo_reduced_hierarchy(mapping, powered_cids, *, edges_df, nodes_df,
                            branch_mondo_ids=None, min_class_size=2,
                            max_class_fraction=1.0):
    """Reduce the Mondo is-a DAG over a set of POWERED OMOP anchors to the compact
    branch-point hierarchy — the same reduction exp 0088 ran, extracted here so a
    fit can build the DAG (and optionally restrict to one body-system BRANCH).

    `mapping` is the `build_mondo_to_omop` frame (columns standard_concept_id,
    standard_concept_name, mondo_id); `powered_cids` is the set of OMOP anchor cids
    kept by the power-count. `edges_df`/`nodes_df` are the Mondo OBO frames.

    `branch_mondo_ids` (optional): restrict terminals to anchors whose Mondo id is
    in this set (a body-system subtree ∪ its root) — the TEMPLATE-BRANCH knob
    (Step A). ``None`` = whole Mondo.

    Returns ``(reduced, anchor_names, class_names)`` where `reduced` is the
    `reduce_to_anchor_hierarchy` dict (feed `reduced["parent_of"]` to
    `build_mondo_engine_dag`), `anchor_names` is ``{cid: name}`` and `class_names`
    is ``{mondo_id: name}``.
    """
    from anchor_hierarchy import reduce_to_anchor_hierarchy
    from mondo_to_omop_mapping import (
        _disease_child_adjacency, _HUMAN_DISEASE, _DISEASE_SUSCEPTIBILITY,
        _DISEASE_CHARACTERISTIC, _INJURY)

    powered = {int(c) for c in powered_cids}
    child_adj = _disease_child_adjacency(edges_df, nodes_df)  # parent -> [children]
    parent_adj: dict = {}
    for parent, children in child_adj.items():
        for c in children:
            parent_adj.setdefault(c, []).append(parent)

    anchor_names = dict(zip(mapping["standard_concept_id"].astype(int),
                            mapping["standard_concept_name"]))
    class_names = {str(i): str(n) for i, n in zip(nodes_df["id"], nodes_df["name"])}

    # anchor:{cid} terminals wired to their mondo ids as parents (a powered anchor
    # may map from several mondo ids; keep them all, deduped).
    anchor_mondos: dict = {}
    for cid, mid in zip(mapping["standard_concept_id"].astype(int), mapping["mondo_id"]):
        cid = int(cid)
        if cid not in powered:
            continue
        if branch_mondo_ids is not None and str(mid) not in branch_mondo_ids:
            continue
        anchor_mondos.setdefault(cid, []).append(str(mid))
    for cid, mids in anchor_mondos.items():
        parent_adj[f"{_ANCHOR_PREFIX}{cid}"] = list(dict.fromkeys(mids))
    terminals = [f"{_ANCHOR_PREFIX}{cid}" for cid in anchor_mondos]

    stop = {_HUMAN_DISEASE, _DISEASE_SUSCEPTIBILITY, _DISEASE_CHARACTERISTIC,
            _INJURY, "MONDO:0000001"}
    reduced = reduce_to_anchor_hierarchy(
        terminals, parent_adj, stop=stop,
        min_class_size=min_class_size, max_class_fraction=max_class_fraction)
    return reduced, anchor_names, class_names


def branch_mondo_id_set(branch_root, *, edges_df, nodes_df):
    """The Mondo ids of `branch_root` ∪ its is-a descendants (the template-branch
    node set for `mondo_reduced_hierarchy`). `branch_root` e.g. 'MONDO:0004995'
    (cardiovascular disorder)."""
    from mondo_to_omop_mapping import _disease_child_adjacency, _descendants
    child_adj = _disease_child_adjacency(edges_df, nodes_df)
    return {str(branch_root)} | {str(x) for x in _descendants(child_adj, str(branch_root))}


def powered_anchor_climb(ca_df, powered_cids, *, spark):
    """The SNOMED-climb frame: `(ancestor_concept_id, descendant_concept_id)` from
    `concept_ancestor` restricted (broadcast join) to the powered anchor set. A
    patient's condition code that appears as a `descendant_concept_id` here rolls
    up to the powered anchor `ancestor_concept_id` — the same join exp 0088's
    power-count used, kept per-descendant for per-patient attestation."""
    import pandas as pd
    from pyspark.sql import functions as F
    from pyspark.sql.functions import broadcast

    anchors_sdf = spark.createDataFrame(
        pd.DataFrame({"ancestor_concept_id": sorted({int(c) for c in powered_cids})}))
    return (ca_df.select("ancestor_concept_id", "descendant_concept_id")
            .join(broadcast(anchors_sdf), "ancestor_concept_id", "inner"))


def build_mondo_fit_inputs(spark, *, cdr, billing, mondo_version="2026-06-02",
                           mondo_cache_dir="data/mondo", min_positives=100,
                           branch_root=None, min_class_size=2, max_class_fraction=1.0,
                           condition_source_table="condition_occurrence"):
    """Build the Mondo `before_dag` + SNOMED-climb frame for a gated-PC fit (BQ).

    The fit-side of exp 0088's `mondo_hierarchy_cloud`: whole-Mondo -> OMOP mapping,
    power-count each anchor (distinct persons with an in-subtree condition), keep
    those clearing `min_positives`, reduce the Mondo is-a DAG over the powered
    anchors to the compact branch-point hierarchy (optionally restricted to a
    `branch_root` body system — the Step-A template), and assemble the integer-id
    engine DAG. Also returns the climb frame restricted to the DAG's terminals.

    Returns ``(before_dag, climb_sdf, terminal_cids, count_of, reduced)``:
      before_dag    integer-id `ConditionDag` (feed to the multi-domain assembler);
      climb_sdf     `(ancestor_concept_id, descendant_concept_id)` for the terminals
                    (wrap with `make_mondo_attested_provider`);
      terminal_cids the powered (branch-filtered) OMOP anchor cids in the DAG;
      count_of      ``{anchor_cid: whole-pop patient count}`` (for logging/K sizing);
      reduced       the `reduce_to_anchor_hierarchy` dict.
    """
    from pathlib import Path

    import pandas as pd
    from pyspark.sql import functions as F
    from pyspark.sql.functions import broadcast

    from charmpheno.omop.bigquery import load_omop_bigquery
    from anchor_selection_cloud import _download_cached, _read_bq
    from mondo_to_omop_mapping import build_mondo_to_omop, seed_source_xrefs

    cache = Path(mondo_cache_dir)
    edges_df = pd.read_csv(_download_cached(mondo_version, "mondo_edges.tsv", cache),
                           sep="\t", low_memory=False)
    nodes_df = pd.read_csv(_download_cached(mondo_version, "mondo_nodes.tsv", cache),
                           sep="\t", low_memory=False)

    # 1) whole-Mondo -> OMOP mapping (scale-fixed, restrict=None), as exp 0088.
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
    anchors = sorted({int(x) for x in mapping["standard_concept_id"]})

    # 2) power-count each anchor (broadcast-join; distinct persons per subtree).
    anchors_sdf = spark.createDataFrame(pd.DataFrame({"ancestor_concept_id": anchors}))
    ca = (_read_bq(spark, cdr, billing, "concept_ancestor")
          .select("ancestor_concept_id", "descendant_concept_id")
          .join(broadcast(anchors_sdf), "ancestor_concept_id", "inner"))
    cond = load_omop_bigquery(
        spark=spark, cdr_dataset=cdr, billing_project=billing,
        source_table=condition_source_table).select("person_id", "concept_id")
    counts = (cond.join(ca, cond["concept_id"] == ca["descendant_concept_id"], "inner")
              .groupBy("ancestor_concept_id")
              .agg(F.countDistinct("person_id").alias("n")).toPandas())
    count_of = {int(r["ancestor_concept_id"]): int(r["n"]) for _, r in counts.iterrows()}
    powered = {c for c in anchors if count_of.get(c, 0) >= min_positives}

    # 3) reduce over powered anchors (optionally within one body-system branch).
    branch_mondo_ids = (None if not branch_root else
                        branch_mondo_id_set(branch_root, edges_df=edges_df, nodes_df=nodes_df))
    reduced, anchor_names, class_names = mondo_reduced_hierarchy(
        mapping, powered, edges_df=edges_df, nodes_df=nodes_df,
        branch_mondo_ids=branch_mondo_ids, min_class_size=min_class_size,
        max_class_fraction=max_class_fraction)
    before_dag = build_mondo_engine_dag(
        reduced["parent_of"], anchor_names=anchor_names, class_names=class_names)

    # 4) climb frame restricted to the DAG's terminal anchors (branch-filtered).
    terminal_cids = {int(n.split(":", 1)[1]) for n in reduced["parent_of"]
                     if n.startswith(_ANCHOR_PREFIX)}
    climb_sdf = powered_anchor_climb(ca, terminal_cids, spark=spark)
    return before_dag, climb_sdf, terminal_cids, count_of, reduced


def make_mondo_attested_provider(climb_sdf, *, doc_spec):
    """A `provider(events_df) -> attested_df` for `assemble_multidomain_from_events`'s
    `attested_provider` seam — the Mondo analogue of `doc_attested_nodes`.

    Per document, the attested nodes are the POWERED ANCHOR cids the patient's
    in-window condition codes climb to (`condition ⋈ climb_sdf on
    concept_id = descendant_concept_id -> ancestor_concept_id`). A full doc roster
    is LEFT-joined so background docs (no anchor-mapped code) survive with an empty
    `attested_cids` (they get a `[]` frontier downstream), exactly like the SNOMED
    provider. `attested_cids` are positive OMOP anchor cids = terminal node ids in
    the Mondo engine DAG, so the rest of the assembly is unchanged.

    Returns ``[doc_id, person_id, source_cohort, attested_cids: array<bigint>]``.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.functions import broadcast

    def provider(events_df):
        ev = doc_spec.derive_docs(events_df)
        roster = ev.groupBy("doc_id").agg(
            F.first("person_id").alias("person_id"),
            F.first("source_cohort").alias("source_cohort"),
        )
        attested = (
            ev.join(broadcast(climb_sdf),
                    ev["concept_id"] == climb_sdf["descendant_concept_id"], "inner")
              .groupBy("doc_id")
              .agg(F.collect_set(
                  F.col("ancestor_concept_id").cast("long")).alias("attested_cids"))
        )
        return (
            roster.join(attested, on="doc_id", how="left")
            .withColumn("attested_cids",
                        F.coalesce(F.col("attested_cids"),
                                   F.array().cast("array<bigint>")))
        )

    return provider

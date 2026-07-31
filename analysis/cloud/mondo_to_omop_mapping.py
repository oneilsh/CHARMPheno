"""MONDO -> OMOP standard-condition mapping (Stage 2 of anchor selection).

A faithful port of Monarch's `mondo_to_omop.py`
(https://github.com/monarch-initiative/mondo2omop) into a pure, injectable
function so it runs on the cluster against BigQuery vocab slices instead of local
CSVs. The mapping *semantics* are unchanged from the original — same disease
filters, same `same_as` -> (SNOMED|MeSH|ICD10CM) parse, same
`concept` (concept_code, vocabulary_id) join, same 'Maps to' -> standard
`Condition` concept resolution, same rare-subset flags. The only changes:

- file reads / `input()` become function arguments (MONDO frames + OMOP frames);
- `networkx.descendants` is replaced by an identical breadth-first transitive
  closure over the subclass_of edges (drops the networkx dependency);
- an optional `restrict_mondo_ids` prefilters the node set for speed.

Original authorship credit: monarch-initiative/mondo2omop. See
docs/superpowers/specs/2026-07-31-expanded-snomed-anchor-selection-design.md.
"""
from __future__ import annotations

import pandas as pd

# MONDO roots used by the original's include/exclude filters.
_HUMAN_DISEASE = "MONDO:0700096"
_DISEASE_SUSCEPTIBILITY = "MONDO:0042489"
_DISEASE_CHARACTERISTIC = "MONDO:0021125"
_INJURY = "MONDO:0021178"

# same_as URI prefixes -> OMOP vocabulary_id (the original's mapping).
_XREF_PREFIXES = {
    "http://identifiers.org/snomedct/": "SNOMED",
    "http://identifiers.org/mesh/": "MeSH",
    "http://purl.bioontology.org/ontology/ICD10CM/": "ICD10CM",
}
_RARE_SUBSETS = (
    "rare", "gard_rare", "nord_rare", "orphanet_rare", "inferred_rare", "mondo_rare",
)


def _descendants(child_adj: dict[str, list[str]], root: str) -> set[str]:
    """Transitive descendants of ``root`` (breadth-first), matching
    ``networkx.descendants`` over the same parent->child adjacency."""
    seen: set[str] = set()
    stack = list(child_adj.get(root, ()))
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(child_adj.get(node, ()))
    return seen


def _disease_child_adjacency(edges_df: pd.DataFrame, nodes_df: pd.DataFrame) -> dict:
    """parent(object) -> [children(subject)] over Disease--subclass_of-->Disease
    edges with obsolete endpoints dropped (mirrors the original graph build)."""
    cat = dict(zip(nodes_df["id"], nodes_df["category"]))
    name = {k: str(v) for k, v in zip(nodes_df["id"], nodes_df["name"])}
    adj: dict[str, list[str]] = {}
    for subj, obj, pred in zip(edges_df["subject"], edges_df["object"], edges_df["predicate"]):
        if pred != "biolink:subclass_of":
            continue
        if cat.get(subj) != "biolink:Disease" or cat.get(obj) != "biolink:Disease":
            continue
        if "obsolete" in name.get(subj, "") or "obsolete" in name.get(obj, ""):
            continue
        adj.setdefault(obj, []).append(subj)  # object is the parent of subject
    return adj


def _disease_nodes(nodes_df: pd.DataFrame, child_adj: dict) -> pd.DataFrame:
    """Disease nodes under human-disease, excluding susceptibility / characteristic
    / injury / obsolete — the original's node filter."""
    nodes = nodes_df[nodes_df["category"] == "biolink:Disease"].copy()
    nodes["name"] = nodes["name"].astype(str)
    nodes = nodes[~nodes["name"].str.contains(r"obsolete(?!$)", regex=True)]
    keep = _descendants(child_adj, _HUMAN_DISEASE)
    drop = (
        _descendants(child_adj, _DISEASE_SUSCEPTIBILITY)
        | _descendants(child_adj, _DISEASE_CHARACTERISTIC)
        | _descendants(child_adj, _INJURY)
    )
    nodes = nodes[nodes["id"].isin(keep) & ~nodes["id"].isin(drop)]
    return nodes


def _same_as_table(nodes: pd.DataFrame) -> pd.DataFrame:
    """Explode `same_as` to (id, concept_code, vocabulary_id) rows for the three
    external vocabularies of interest."""
    df = nodes[["id", "same_as"]].copy()
    df["xref"] = df["same_as"].str.split("|")
    df = df.explode("xref").dropna(subset=["xref"])
    frames = []
    for prefix, vocab in _XREF_PREFIXES.items():
        hit = df[df["xref"].str.startswith(prefix)].copy()
        hit["concept_code"] = hit["xref"].str.replace(prefix, "", regex=False)
        hit["vocabulary_id"] = vocab
        frames.append(hit[["id", "concept_code", "vocabulary_id"]])
    if not frames:
        return pd.DataFrame(columns=["id", "concept_code", "vocabulary_id"])
    return pd.concat(frames, ignore_index=True)


def _subsets_table(nodes: pd.DataFrame) -> pd.DataFrame:
    """Binary rare-designation columns per MONDO id from the `subsets` field."""
    df = nodes[["id", "subsets"]].copy()
    df["subset"] = df["subsets"].str.split("|")
    df = df.explode("subset")
    out = pd.DataFrame({"id": nodes["id"].unique()})
    for flag in _RARE_SUBSETS:
        ids = set(df.loc[df["subset"] == flag, "id"])
        out[flag] = out["id"].isin(ids).astype(int)
    return out


def seed_source_xrefs(
    *,
    mondo_edges_df: pd.DataFrame,
    mondo_nodes_df: pd.DataFrame,
    restrict_mondo_ids: set[str],
) -> pd.DataFrame:
    """(mondo_id, concept_code, vocabulary_id) source xrefs for the seed diseases.

    Lets the cluster driver bound the (large) `concept` / `concept_relationship`
    reads to just the codes the seed can hit, before the full mapping join.
    """
    child_adj = _disease_child_adjacency(mondo_edges_df, mondo_nodes_df)
    nodes = _disease_nodes(mondo_nodes_df, child_adj)
    nodes = nodes[nodes["id"].isin(restrict_mondo_ids)]
    return _same_as_table(nodes).rename(columns={"id": "mondo_id"})


def build_mondo_to_omop(
    *,
    mondo_edges_df: pd.DataFrame,
    mondo_nodes_df: pd.DataFrame,
    concept_df: pd.DataFrame,
    concept_relationship_df: pd.DataFrame,
    restrict_mondo_ids: set[str] | None = None,
) -> pd.DataFrame:
    """MONDO -> OMOP standard Condition concept, one row per (xref, mapping).

    Args mirror the original's four inputs. ``concept_df`` needs columns
    concept_id, concept_name, vocabulary_id, domain_id, concept_code,
    standard_concept; ``concept_relationship_df`` needs concept_id_1,
    concept_id_2, relationship_id. ``restrict_mondo_ids`` (our seed) prefilters
    the disease nodes so only relevant xrefs hit the OMOP joins.
    """
    child_adj = _disease_child_adjacency(mondo_edges_df, mondo_nodes_df)
    nodes = _disease_nodes(mondo_nodes_df, child_adj)
    if restrict_mondo_ids is not None:
        nodes = nodes[nodes["id"].isin(restrict_mondo_ids)]

    same_as = _same_as_table(nodes)
    subsets = _subsets_table(nodes)

    # same_as -> OMOP concept by (concept_code, vocabulary_id).
    m = same_as.merge(
        concept_df, on=["concept_code", "vocabulary_id"], how="inner"
    )
    # concept -> standard via 'Maps to'.
    maps_to = concept_relationship_df[
        concept_relationship_df["relationship_id"] == "Maps to"
    ][["concept_id_1", "concept_id_2"]]
    m = m.merge(maps_to, left_on="concept_id", right_on="concept_id_1", how="inner")

    std = concept_df[concept_df["standard_concept"] == "S"].rename(
        columns={
            "concept_id": "standard_concept_id",
            "concept_name": "standard_concept_name",
            "vocabulary_id": "standard_vocabulary_id",
            "domain_id": "standard_domain_id",
            "concept_code": "standard_concept_code",
        }
    )
    std = std[std["standard_domain_id"] == "Condition"][
        ["standard_concept_id", "standard_concept_name",
         "standard_vocabulary_id", "standard_domain_id", "standard_concept_code"]
    ]
    m = m.merge(std, left_on="concept_id_2", right_on="standard_concept_id", how="inner")
    m = m.merge(subsets, on="id", how="left")
    return m.rename(columns={"id": "mondo_id"})

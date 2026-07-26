"""Two-domain (MixEHR-style) corpus assembly for the multi-domain gated model.

A document carries TWO bag-of-words feature columns -- one per domain (e.g.
conditions and drugs) -- over two INDEPENDENT vocabularies. Per-modality
generative separation (each domain gets its own vocabulary and its own
observation model downstream) follows MixEHR (Li, Nair, Lu et al. 2020,
Nat. Commun. 11:2536): distinct EHR data types are modeled as distinct
"modalities" sharing patient-level topic structure while keeping separate
per-modality vocabularies. Here the shared structure is the gated topic
model's per-document theta; the gate itself is condition-only and orthogonal
to domain (arc design) -- this module only assembles the two feature
columns, it has no opinion about the gate.

This module is a thin two-domain layer over the domain-agnostic
`topic_prep.to_bow_dataframe`: it fits each domain's vocabulary and BOW
separately and joins the two feature columns per document. It does NOT
reimplement BOW/vocab fitting, and it does NOT touch the single-domain
`case_finding_assembly` path.
"""
from __future__ import annotations

from dataclasses import dataclass

from pyspark.ml.linalg import SparseVector, VectorUDT
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


@dataclass(frozen=True)
class DomainVocabSpec:
    """Per-domain vocabulary controls. Independent per domain because the two
    domains have very different natural sizes (SP3b design). `vocab` pins a
    pre-built vocabulary (concept-ids in assignment order) for eval/reproduce;
    None fits from the data."""
    vocab_size: int | None
    min_df: int | float = 1
    min_patient_count: int = 1
    vocab: list[int] | None = None


def _empty_vec_udf(size: int):
    return F.udf(lambda: SparseVector(size, [], []), VectorUDT())


def two_domain_bow(events_a: DataFrame, events_b: DataFrame, *, doc_spec,
                    vocab_a: DomainVocabSpec, vocab_b: DomainVocabSpec):
    """Two aligned per-domain BOW columns joined per doc. Returns
    (df[doc_id, person_id, features_a, features_b], vocab_map_a, vocab_map_b).

    A doc present in only one domain gets an EMPTY vector (of that domain's
    vocab size) on the absent side, never a dropped row -- so every doc
    carries both columns and each per-domain vector size is CONSTANT across
    the corpus (SP3a's shim derives domainBounds from the first row and
    validates every row against it; a variable size would raise).
    """
    from charmpheno.omop.topic_prep import to_bow_dataframe

    bow_a, vm_a = to_bow_dataframe(
        events_a, doc_spec=doc_spec, token_col="concept_id",
        vocab_size=vocab_a.vocab_size, min_df=vocab_a.min_df,
        min_patient_count=vocab_a.min_patient_count, vocab=vocab_a.vocab)
    bow_b, vm_b = to_bow_dataframe(
        events_b, doc_spec=doc_spec, token_col="concept_id",
        vocab_size=vocab_b.vocab_size, min_df=vocab_b.min_df,
        min_patient_count=vocab_b.min_patient_count, vocab=vocab_b.vocab)

    va, vb = len(vm_a), len(vm_b)

    a = bow_a.select("doc_id", "person_id", F.col("features").alias("features_a"))
    b = bow_b.select("doc_id", F.col("features").alias("features_b"),
                      F.col("person_id").alias("person_id_b"))

    joined = a.join(b, on="doc_id", how="full_outer")

    # Coalesce person_id across the outer join (present on whichever side
    # actually has the doc); fill absent per-domain vectors with an empty
    # SparseVector of that domain's fixed vocab size so every row's per-domain
    # vector size is constant across the corpus.
    joined = (
        joined
        .withColumn("person_id", F.coalesce(F.col("person_id"), F.col("person_id_b")))
        .drop("person_id_b")
        .withColumn("features_a", F.coalesce(F.col("features_a"), _empty_vec_udf(va)()))
        .withColumn("features_b", F.coalesce(F.col("features_b"), _empty_vec_udf(vb)()))
    )

    return (
        joined.select("doc_id", "person_id", "features_a", "features_b"),
        vm_a,
        vm_b,
    )

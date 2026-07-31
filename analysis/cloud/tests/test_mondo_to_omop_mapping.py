"""Tests for the MONDO->OMOP mapping port (synthetic frames, no cluster)."""
import pandas as pd


def _nodes():
    return pd.DataFrame(
        [
            # id, category, name, same_as, subsets
            ("MONDO:0700096", "biolink:Disease", "human disease", None, None),
            ("MONDO:0042489", "biolink:Disease", "disease susceptibility", None, None),
            ("MONDO:AAA", "biolink:Disease", "test disease A",
             "http://identifiers.org/snomedct/12345", "rare|orphanet_rare"),
            ("MONDO:OBS", "biolink:Disease", "obsolete test disease",
             "http://identifiers.org/snomedct/12345", None),
            ("MONDO:SUS", "biolink:Disease", "susceptibility child",
             "http://identifiers.org/snomedct/12345", None),
            ("MONDO:NOTDIS", "biolink:PhenotypicFeature", "not a disease",
             "http://identifiers.org/snomedct/12345", None),
        ],
        columns=["id", "category", "name", "same_as", "subsets"],
    )


def _edges():
    sc = "biolink:subclass_of"
    return pd.DataFrame(
        [
            ("MONDO:AAA", "MONDO:0700096", sc),
            ("MONDO:0042489", "MONDO:0700096", sc),
            ("MONDO:SUS", "MONDO:0042489", sc),
            ("MONDO:OBS", "MONDO:0700096", sc),
        ],
        columns=["subject", "object", "predicate"],
    )


def _concept():
    return pd.DataFrame(
        [
            # source SNOMED concept (non-standard)
            (111, "snomed source A", "SNOMED", "Condition", "12345", None),
            # standard Condition concept it Maps to
            (999, "Standard Condition A", "SNOMED", "Condition", "99", "S"),
        ],
        columns=["concept_id", "concept_name", "vocabulary_id", "domain_id",
                 "concept_code", "standard_concept"],
    )


def _concept_relationship():
    return pd.DataFrame(
        [(111, 999, "Maps to"), (999, 111, "Mapped from")],
        columns=["concept_id_1", "concept_id_2", "relationship_id"],
    )


def test_maps_disease_to_standard_condition_with_rare_flags():
    from mondo_to_omop_mapping import build_mondo_to_omop

    out = build_mondo_to_omop(
        mondo_edges_df=_edges(),
        mondo_nodes_df=_nodes(),
        concept_df=_concept(),
        concept_relationship_df=_concept_relationship(),
    )
    # Only disease A survives the disease/human/susceptibility/obsolete filters
    # AND has a code that Maps to a standard Condition concept.
    assert list(out["mondo_id"]) == ["MONDO:AAA"]
    row = out.iloc[0]
    assert row["standard_concept_id"] == 999
    assert row["standard_domain_id"] == "Condition"
    assert row["rare"] == 1 and row["orphanet_rare"] == 1
    assert row["nord_rare"] == 0


def test_obsolete_susceptibility_and_nondisease_are_excluded():
    from mondo_to_omop_mapping import build_mondo_to_omop

    out = build_mondo_to_omop(
        mondo_edges_df=_edges(),
        mondo_nodes_df=_nodes(),
        concept_df=_concept(),
        concept_relationship_df=_concept_relationship(),
    )
    got = set(out["mondo_id"])
    assert "MONDO:OBS" not in got      # obsolete name
    assert "MONDO:SUS" not in got      # susceptibility descendant
    assert "MONDO:NOTDIS" not in got   # not a Disease node


def test_restrict_mondo_ids_prefilters():
    from mondo_to_omop_mapping import build_mondo_to_omop

    out = build_mondo_to_omop(
        mondo_edges_df=_edges(),
        mondo_nodes_df=_nodes(),
        concept_df=_concept(),
        concept_relationship_df=_concept_relationship(),
        restrict_mondo_ids={"MONDO:ZZZ"},  # excludes A
    )
    assert len(out) == 0

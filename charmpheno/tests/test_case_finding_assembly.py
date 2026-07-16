"""Tests for charmpheno.omop.case_finding_assembly (piece 2 of the case-finding
cluster driver) + the diabetes disease-registry entry it depends on."""


def test_disease_registry_has_diabetes_anchor_201820():
    from charmpheno.omop.cohorts import _DISEASE_REGISTRY
    assert _DISEASE_REGISTRY["diabetes"] == {
        "inclusion_ancestors": (201820,),
        "exclusion_ancestors": (),
    }

"""Tests for charmpheno.omop.case_finding_assembly (piece 2 of the case-finding
cluster driver) + the diabetes disease-registry entry it depends on."""


def test_disease_registry_has_diabetes_anchor_201820():
    from charmpheno.omop.cohorts import _DISEASE_REGISTRY
    assert _DISEASE_REGISTRY["diabetes"] == {
        "inclusion_ancestors": (201820,),
        "exclusion_ancestors": (),
    }


from charmpheno.omop.condition_dag import build_condition_dag
from spark_vi.models.topic.dag_placement import DagLayout


def _diamond_dag():
    # concept-id DAG rooted at anchor 100:
    #   100 -> 200, 300 ; 200 -> 400 ; 300 -> 400 (diamond) ; 200 -> 500
    edges = [(100, 200), (100, 300), (200, 400), (300, 400), (200, 500)]
    node_ids = [200, 300, 400, 500]
    return build_condition_dag(edges, anchor=100, node_ids=node_ids)


def test_descendants_walks_transitively():
    from charmpheno.omop.case_finding_assembly import _descendants
    dag = _diamond_dag()
    ch = dag.children()
    assert _descendants(ch, 200) == {400, 500}
    assert _descendants(ch, 400) == set()


def test_most_specific_cids_drops_attested_ancestors_keeps_incomparable():
    from charmpheno.omop.case_finding_assembly import most_specific_cids
    dag = _diamond_dag()
    # attest 200 and its descendant 400 -> only 400 is most-specific.
    assert most_specific_cids({200, 400}, dag) == {400}
    # attest incomparable 400 and 500 -> both kept.
    assert most_specific_cids({400, 500}, dag) == {400, 500}
    # single node -> itself.
    assert most_specific_cids({300}, dag) == {300}


def test_roll_up_to_survivors_reattaches_dropped_to_nearest_ancestor():
    from charmpheno.omop.case_finding_assembly import roll_up_to_survivors
    dag = _diamond_dag()
    keep = {100, 200, 300}          # 400 and 500 pruned
    # 400 (pruned) rolls up to BOTH surviving parents 200 and 300.
    assert roll_up_to_survivors({400}, dag, keep) == {200, 300}
    # a kept node stays itself; a dropped one rolls up, in one call.
    assert roll_up_to_survivors({200, 500}, dag, keep) == {200}


def test_doc_frontier_engine_ids_maps_and_reduces_to_most_specific():
    from charmpheno.omop.case_finding_assembly import doc_frontier_engine_ids
    dag = _diamond_dag()
    keep = dag.nodes()                        # nothing pruned
    parent_int, int2cid, cid2int = dag.to_engine()
    lay = DagLayout(parent_int, n_bg=2, tpn=1)
    # attest 200 + 400: 400 is a descendant of 200 -> frontier = {engine(400)}.
    fr = doc_frontier_engine_ids({200, 400}, dag, keep, cid2int, lay)
    assert fr == [cid2int[400]]
    # empty attestation (background doc) -> [].
    assert doc_frontier_engine_ids(set(), dag, keep, cid2int, lay) == []
    # incomparable 400 + 500 -> both, in engine space, sorted.
    fr2 = doc_frontier_engine_ids({400, 500}, dag, keep, cid2int, lay)
    assert fr2 == sorted([cid2int[400], cid2int[500]])


def test_doc_frontier_engine_ids_rolls_pruned_attestation_up():
    from charmpheno.omop.case_finding_assembly import doc_frontier_engine_ids
    dag = _diamond_dag()
    # prune 400: a patient attesting only 400 rolls up to 200 and 300, which are
    # incomparable survivors -> frontier = {engine(200), engine(300)}.
    keep = {100, 200, 300, 500}
    from charmpheno.omop.condition_dag import prune_by_attestation
    counts = {200: 99, 300: 99, 500: 99, 400: 0}
    after = prune_by_attestation(dag, counts, min_n=1)
    parent_int, int2cid, cid2int = after.to_engine()
    lay = DagLayout(parent_int, n_bg=2, tpn=1)
    fr = doc_frontier_engine_ids({400}, dag, after.nodes(), cid2int, lay)
    assert fr == sorted([cid2int[200], cid2int[300]])

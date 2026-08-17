"""Pure-logic tests for the whole-Mondo completeness readout (small-cell rule +
report formatting). The BQ/Spark path is cluster-covered, not unit-tested."""
import mondo_completeness_cloud as m


def test_small_cell_suppression():
    assert m.suppress(0) == "0"            # zero is not a small cell
    assert m.suppress(1) == "<20"
    assert m.suppress(19) == "<20"
    assert m.suppress(20) == "20"          # floor is inclusive-exact
    assert m.suppress(6543) == "6543"


def test_ladder_suppresses_and_computes_residuals():
    out = m.format_ladder(1000, 600, 590)
    # coded-but-unplaced = 10 -> suppressed; no-code = 400 -> exact
    assert "coded but UNPLACED" in out and "<20" in out
    assert "400" in out                    # 1000 - 600 no-code residual shown exactly
    assert "100.00%" in out


def test_unplaced_table_suppresses_counts():
    rows = [
        {"concept_id": 320128, "concept_name": "Essential hypertension",
         "domain_id": "Condition", "standard_concept": "S", "n_patients": 40000},
        {"concept_id": 77, "concept_name": "rare code", "domain_id": "Condition",
         "standard_concept": "S", "n_patients": 5},
    ]
    out = m.format_unplaced(rows, n_unplaced_persons=49815, n_suppressed_concepts=812)
    assert "40000" in out                  # >= floor: exact
    assert "<20" in out                    # the n=5 row is suppressed
    assert "Essential hypertension" in out
    assert "812 more distinct concepts" in out

"""covariate_readout: per-node adj-vs-score_cv table + corroboration summary."""


def _manifest():
    return {
        "corpus_manifest": {
            "int2cid": {"1": 100, "2": 200, "3": 300},
            "name_by_id": {"100": "SLE", "200": "Scleroderma", "300": "Amyloidosis"},
        },
        "covariates": {"names": ["Intercept", "C(sex)[T.M]", "age"]},
        "metrics": {"covariate_adjusted": {
            "n_covariates": 3,
            "detection_auc_score_cv": 0.646, "detection_auc_adj": 0.655,
            "auc_score_cv_macro": 0.60, "auc_adj_macro": 0.70,
            "ap_score_cv_macro": 0.012, "ap_adj_macro": 0.013,
            # node 1: AUC up AND AP up -> corroborated (real)
            # node 2: AUC up but AP flat -> ranking-only (marginal)
            # node 3: CV could not run -> nan
            "node_auc_score_cv": {"1": 0.60, "2": 0.55, "3": float("nan")},
            "node_auc_adj":      {"1": 0.75, "2": 0.70, "3": float("nan")},
            "node_ap_score_cv":  {"1": 0.10, "2": 0.05, "3": float("nan")},
            "node_ap_adj":       {"1": 0.20, "2": 0.05, "3": float("nan")},
            "node_npos": {"1": 200, "2": 12, "3": 1},
        }},
    }


def test_build_rows_sorts_by_auc_lift_and_carries_nan():
    from covariate_readout import build_rows, node_names
    m = _manifest()
    rows = build_rows(m["metrics"]["covariate_adjusted"], node_names(m))
    # node 1 and 2 both have +0.15 AUC lift; node 3 (nan) sinks last.
    assert rows[-1]["node"] == 3
    assert rows[0]["name"] in {"SLE", "Scleroderma"}
    r1 = next(r for r in rows if r["node"] == 1)
    assert abs(r1["auc_delta"] - 0.15) < 1e-9
    assert abs(r1["ap_delta"] - 0.10) < 1e-9
    assert r1["npos"] == 200


def test_render_corroboration_summary_distinguishes_real_from_marginal():
    import math
    from covariate_readout import build_rows, node_names, render
    m = _manifest()
    ca = m["metrics"]["covariate_adjusted"]
    rows = build_rows(ca, node_names(m))
    out = render(ca, rows, m)
    assert "SLE" in out and "Scleroderma" in out
    assert "Intercept,C(sex)[T.M],age" in out
    # node 1's AUC lift is corroborated by AP; node 2's is not.
    assert "AUC lift > 0.05: 2 nodes" in out
    assert "corroborated, real): 1" in out
    # node 2 (npos=12) flagged small-node
    assert "small-node, suspect): 1" in out
    # nan node rendered, not crashed
    assert "nan" in out


def test_render_without_npos_omits_that_column():
    from covariate_readout import build_rows, node_names, render
    m = _manifest()
    ca = m["metrics"]["covariate_adjusted"]
    del ca["node_npos"]                       # simulate a pre-npos run
    rows = build_rows(ca, node_names(m))
    out = render(ca, rows, m)
    assert "npos" not in out.split("\n")[6]   # header row has no npos column
    assert "small-node" not in out            # summary drops the small-node line

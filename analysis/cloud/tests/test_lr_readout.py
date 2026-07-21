"""Tests for the lr_readout standalone driver (fork-settler): the pure arg-surface
helpers only. The Spark/bundle/npz body is cluster-covered, not unit-tested."""


def test_lr_readout_arg_surface():
    import importlib
    mod = importlib.import_module("lr_readout")
    p = mod.build_parser()
    ns = p.parse_args(["--run-dir", "/runs/0061", "--alpha-grid", "0,1,10"])
    assert ns.run_dir == "/runs/0061"
    assert mod.parse_alpha_grid(ns.alpha_grid) == [0.0, 1.0, 10.0]
    # 0 is always included even if omitted
    assert 0.0 in mod.parse_alpha_grid("1,10")


def test_render_decompose_rows_uses_concept_names():
    import importlib
    mod = importlib.import_module("lr_readout")
    # (w, count, contribution) engine tuples + a vocab-index -> concept-id -> name chain
    rows = [(0, 2.0, 5.9), (1, 3.0, -3.9)]
    idx_to_cid = {0: 111, 1: 222}
    name_by_id = {111: "Lupus erythematosus", 222: "Essential hypertension"}
    lines = mod.render_decompose_rows(rows, idx_to_cid, name_by_id)
    assert "Lupus erythematosus" in lines[0] and "+5.9" in lines[0]
    assert "Essential hypertension" in lines[1] and "-3.9" in lines[1]

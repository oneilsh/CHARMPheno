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

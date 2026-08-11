"""run_experiment wiring for model_class=pc (Phase-C PC replication)."""
import importlib
from pathlib import Path


def _run_exp(monkeypatch):
    mod = importlib.import_module("run_experiment")
    monkeypatch.setattr(mod, "_require_workspace_env", lambda: ("proj.ds", "bill"))
    return mod


def _eff(**over):
    base = {
        "model_class": "pc", "cohort": "mdd_antidepressant",
        "K": 25, "weight_y": 100.0, "alpha": 1.1, "tau": 1.1, "pi_iters": 100,
        "max_iter": 500, "lookback_days": 365, "window_days": 365,
        "stability_days": 90, "grace_gap_days": 30, "vocab_size": 2000,
        "min_df": 20, "min_patient_count": 20, "person_mod": 1,
        "test_frac": 0.25, "seed": 0,
    }
    base.update(over)
    return base


def test_validate_frontmatter_accepts_pc(monkeypatch):
    mod = _run_exp(monkeypatch)
    # minimal valid frontmatter; must not sys.exit (no stm-style required block)
    mod.validate_frontmatter({
        "id": 70, "slug": "x", "cohort": "mdd_antidepressant",
        "model_class": "pc"})


def test_driver_path_for_pc(monkeypatch):
    mod = _run_exp(monkeypatch)
    assert mod.build_fit_driver_path({"model_class": "pc"}) \
        == "analysis/cloud/pc_antidepressant_cloud.py"


def test_build_pc_args_key_to_flag_mapping(monkeypatch):
    mod = _run_exp(monkeypatch)
    args = mod.build_pc_args(_eff(), "/runs/0070-x")
    d = dict(zip(args[::2], args[1::2]))
    assert d["--K"] == "25"
    assert d["--weight-y"] == "100.0"
    assert d["--vocab-size"] == "2000"
    assert d["--stability-days"] == "90"
    assert d["--grace-gap-days"] == "30"
    assert d["--lookback-days"] == "365"
    assert d["--test-frac"] == "0.25"
    assert d["--cdr"] == "proj.ds" and d["--billing"] == "bill"
    # PC writes ONE results JSON into the run dir (no --out-dir like the others)
    assert d["--out"] == str(Path("/runs/0070-x") / "pc_results.json")
    # cohort is hard-wired in the driver; no --cohort / --prior-obs-days flags
    assert "--cohort" not in args
    assert "--prior-obs-days" not in args
    # resume unsupported; K IS a real flag (unlike dag_placement)
    assert "--resume-from" not in args


def test_build_pc_args_threads_min_label_count_default_20(monkeypatch):
    # --min-label-count defaults to 20 (the AoU small-cell floor) for BOTH
    # backends/cohorts, so the next run masks sub-20-count drug labels.
    mod = _run_exp(monkeypatch)
    d = dict(zip(*[iter(mod.build_pc_args(_eff(), "/out"))] * 2))
    assert d["--min-label-count"] == "20"
    d_vi = dict(zip(*[iter(mod.build_pc_args(_eff(backend="vi"), "/out"))] * 2))
    assert d_vi["--min-label-count"] == "20"


def test_build_pc_args_min_label_count_from_config(monkeypatch):
    mod = _run_exp(monkeypatch)
    d = dict(zip(*[iter(mod.build_pc_args(_eff(min_label_count=0), "/out"))] * 2))
    assert d["--min-label-count"] == "0"       # 0 disables masking


def test_build_pc_args_baseline_max_iter_default_and_no_skip(monkeypatch):
    mod = _run_exp(monkeypatch)
    args = mod.build_pc_args(_eff(), "/out")
    d = dict(zip(*[iter(args)] * 2))
    assert d["--baseline-max-iter"] == "-1"       # default: reuse --max-iter
    assert "--skip-two-stage" not in args          # store_true off by default


def test_build_pc_args_skip_two_stage_and_baseline_cap(monkeypatch):
    mod = _run_exp(monkeypatch)
    args = mod.build_pc_args(
        _eff(skip_two_stage=True, baseline_max_iter=100), "/out")
    assert "--skip-two-stage" in args
    d = dict(zip(*[iter(a for a in args if a != "--skip-two-stage")] * 2))
    assert d["--baseline-max-iter"] == "100"


def test_build_pc_args_vi_threads_head_lr_knobs(monkeypatch):
    mod = _run_exp(monkeypatch)
    d = dict(zip(*[iter(mod.build_pc_args(
        _eff(backend="vi", head_lr_scale=3.0, weight_y_warmup_iters=20), "/out"))] * 2))
    assert d["--head-lr-scale"] == "3.0"
    assert d["--weight-y-warmup-iters"] == "20"


def test_build_pc_args_vi_head_lr_defaults(monkeypatch):
    mod = _run_exp(monkeypatch)
    d = dict(zip(*[iter(mod.build_pc_args(_eff(backend="vi"), "/out"))] * 2))
    assert d["--head-lr-scale"] == "1.0"        # default: no head boost
    assert d["--weight-y-warmup-iters"] == "0"
    assert d["--head-optimizer"] == "sgd"       # default optimizer unchanged
    assert d["--head-lr"] == "0.05"


def test_build_pc_args_vi_threads_head_optimizer(monkeypatch):
    mod = _run_exp(monkeypatch)
    d = dict(zip(*[iter(mod.build_pc_args(
        _eff(backend="vi", head_optimizer="adam", head_lr=0.02), "/out"))] * 2))
    assert d["--head-optimizer"] == "adam"
    assert d["--head-lr"] == "0.02"


def test_build_pc_args_vi_topic_trust(monkeypatch):
    mod = _run_exp(monkeypatch)
    d0 = dict(zip(*[iter(mod.build_pc_args(_eff(backend="vi"), "/out"))] * 2))
    assert d0["--topic-trust"] == "0.1"          # default
    d1 = dict(zip(*[iter(mod.build_pc_args(
        _eff(backend="vi", topic_trust=0.02), "/out"))] * 2))
    assert d1["--topic-trust"] == "0.02"


def test_build_pc_args_vi_grad_cavi_iters(monkeypatch):
    mod = _run_exp(monkeypatch)
    d0 = dict(zip(*[iter(mod.build_pc_args(_eff(backend="vi"), "/out"))] * 2))
    assert d0["--grad-cavi-iters"] == "20"       # default
    d1 = dict(zip(*[iter(mod.build_pc_args(
        _eff(backend="vi", grad_cavi_iters=50), "/out"))] * 2))
    assert d1["--grad-cavi-iters"] == "50"


def test_build_pc_args_inmem_omits_head_lr_knobs(monkeypatch):
    # VI-only: inmem argv stays byte-for-byte, even if the config carries them.
    mod = _run_exp(monkeypatch)
    args = mod.build_pc_args(
        _eff(head_lr_scale=3.0, weight_y_warmup_iters=20, head_optimizer="adam"), "/out")
    assert "--head-lr-scale" not in args
    assert "--weight-y-warmup-iters" not in args
    assert "--head-optimizer" not in args
    assert "--head-lr" not in args


def test_build_pc_args_backend_defaults_inmem_no_svi_knobs(monkeypatch):
    mod = _run_exp(monkeypatch)
    # No backend key -> default inmem: --backend inmem passed, SVI knobs omitted
    # so the inmem argv is byte-for-byte the prior command line (plus --backend).
    args = mod.build_pc_args(_eff(), "/runs/0070-x")
    d = dict(zip(args[::2], args[1::2]))
    assert d["--backend"] == "inmem"
    for knob in ("--subsampling-rate", "--tau0", "--kappa"):
        assert knob not in args


def test_build_pc_args_backend_vi_threads_svi_knobs(monkeypatch):
    mod = _run_exp(monkeypatch)
    args = mod.build_pc_args(
        _eff(backend="vi", subsampling_rate=0.1, tau0=64.0, kappa=0.6),
        "/runs/0071-x",
    )
    d = dict(zip(args[::2], args[1::2]))
    assert d["--backend"] == "vi"
    assert d["--subsampling-rate"] == "0.1"
    assert d["--tau0"] == "64.0"
    assert d["--kappa"] == "0.6"


def test_build_pc_args_backend_vi_svi_knob_defaults(monkeypatch):
    mod = _run_exp(monkeypatch)
    # backend vi with no explicit knobs -> the driver-matching defaults appear.
    args = mod.build_pc_args(_eff(backend="vi"), "/out")
    d = dict(zip(args[::2], args[1::2]))
    assert d["--subsampling-rate"] == "0.05"
    assert d["--tau0"] == "1024.0"
    assert d["--kappa"] == "0.51"


# --- unsupervised warm-start knob threading (VI backend only) ----------------

def test_build_pc_args_vi_threads_warm_start_default_zero(monkeypatch):
    # backend vi, no warm_start_unsup_iters -> the flag is present with 0 (cold).
    mod = _run_exp(monkeypatch)
    args = mod.build_pc_args(_eff(backend="vi"), "/out")
    d = dict(zip(args[::2], args[1::2]))
    assert d["--warm-start-unsup-iters"] == "0"


def test_build_pc_args_vi_threads_warm_start_from_config(monkeypatch):
    mod = _run_exp(monkeypatch)
    args = mod.build_pc_args(
        _eff(backend="vi", warm_start_unsup_iters=50), "/out",
    )
    d = dict(zip(args[::2], args[1::2]))
    assert d["--warm-start-unsup-iters"] == "50"


def test_build_pc_args_inmem_omits_warm_start(monkeypatch):
    # inmem argv stays byte-for-byte unchanged: no --warm-start-unsup-iters even
    # if the config carries the key (it is VI-only).
    mod = _run_exp(monkeypatch)
    args = mod.build_pc_args(_eff(warm_start_unsup_iters=50), "/out")
    assert "--warm-start-unsup-iters" not in args


# --- checkpoint / resume threading (VI backend only) -------------------------

def test_build_pc_args_inmem_threads_no_save_flags(monkeypatch):
    # inmem is byte-for-byte unchanged: NO --save-dir / --save-interval /
    # --resume-from, even when a resume_from path is passed (L-BFGS can't resume).
    mod = _run_exp(monkeypatch)
    args = mod.build_pc_args(_eff(), "/runs/0070-x", resume_from=Path("/runs/0070-x"))
    for flag in ("--save-dir", "--save-interval", "--resume-from"):
        assert flag not in args, f"{flag} must not appear on the inmem argv"


def test_build_pc_args_vi_threads_save_flags_without_resume(monkeypatch):
    # vi (no prior checkpoint): --save-dir + --save-interval present, --resume-from absent.
    mod = _run_exp(monkeypatch)
    args = mod.build_pc_args(
        _eff(backend="vi", save_interval=5), "/runs/0071-x", resume_from=None,
    )
    d = dict(zip(args[::2], args[1::2]))
    assert d["--save-dir"] == "/runs/0071-x"
    assert d["--save-interval"] == "5"
    assert "--resume-from" not in args


def test_build_pc_args_vi_threads_resume_from_when_set(monkeypatch):
    # vi (checkpoint present): --resume-from points at the run dir; --save-dir too.
    mod = _run_exp(monkeypatch)
    args = mod.build_pc_args(
        _eff(backend="vi"), "/runs/0071-x", resume_from=Path("/runs/0071-x"),
    )
    d = dict(zip(args[::2], args[1::2]))
    assert d["--save-dir"] == "/runs/0071-x"
    assert d["--resume-from"] == "/runs/0071-x"
    # save_interval falls back to -1 when the config doesn't set it.
    assert d["--save-interval"] == "-1"


def test_build_pc_args_vi_save_interval_from_config(monkeypatch):
    mod = _run_exp(monkeypatch)
    args = mod.build_pc_args(_eff(backend="vi", save_interval=25), "/out")
    d = dict(zip(args[::2], args[1::2]))
    assert d["--save-interval"] == "25"


def test_build_pc_args_cache_uri_optional(monkeypatch):
    mod = _run_exp(monkeypatch)
    assert "--cache-uri" not in mod.build_pc_args(_eff(), "/out")
    on = mod.build_pc_args(_eff(cache_uri="gs://c"), "/out")
    assert on[on.index("--cache-uri") + 1] == "gs://c"


# --- cohort selector + stable-treatment knob threading -----------------------

def test_build_pc_args_antidepressant_omits_cohort_and_stable_knobs(monkeypatch):
    # mdd_antidepressant is the driver default: its argv stays byte-for-byte the
    # prior command line — NO --cohort and NO stable-treatment knobs.
    mod = _run_exp(monkeypatch)
    args = mod.build_pc_args(_eff(), "/runs/0070-x")
    assert "--cohort" not in args
    for knob in ("--min-days", "--max-gap-days", "--min-history-events",
                 "--age-min", "--age-max"):
        assert knob not in args, f"{knob} must not appear on the antidepressant argv"


def test_build_pc_args_stable_treatment_threads_cohort_and_knobs(monkeypatch):
    mod = _run_exp(monkeypatch)
    args = mod.build_pc_args(
        _eff(cohort="mdd_stable_treatment", backend="vi",
             min_days=120, max_gap_days=400, min_history_events=3,
             age_min=21, age_max=75),
        "/runs/0072-x",
    )
    d = dict(zip(args[::2], args[1::2]))
    assert d["--cohort"] == "mdd_stable_treatment"
    assert d["--min-days"] == "120"
    assert d["--max-gap-days"] == "400"
    assert d["--min-history-events"] == "3"
    assert d["--age-min"] == "21"
    assert d["--age-max"] == "75"


def test_build_pc_args_stable_treatment_knob_defaults(monkeypatch):
    # cohort set but the stable knobs unset -> the committed cohort defaults.
    mod = _run_exp(monkeypatch)
    args = mod.build_pc_args(_eff(cohort="mdd_stable_treatment"), "/out")
    d = dict(zip(args[::2], args[1::2]))
    assert d["--cohort"] == "mdd_stable_treatment"
    assert d["--min-days"] == "90" and d["--max-gap-days"] == "395"
    assert d["--min-history-events"] == "2"
    assert d["--age-min"] == "18" and d["--age-max"] == "80"


def test_load_defaults_mdd_stable_treatment_present_and_shaped(monkeypatch):
    # load_defaults REQUIRES experiments/defaults/mdd_stable_treatment.yaml for a
    # stable-treatment experiment; it must merge _base + carry the PC/stable shape.
    mod = _run_exp(monkeypatch)
    eff = mod.load_defaults("mdd_stable_treatment", mod.DEFAULTS_DIR)
    assert eff["cohort"] == "mdd_stable_treatment"
    assert eff["model_class"] == "pc"
    assert eff["backend"] == "vi"
    assert eff["vocab_size"] == 5000
    # all-history features: mdd_stable_treatment.yaml itself sets NO lookback_days
    # (the driver uses all-history for this cohort). Any lookback_days in the
    # merged config is the inert _base.yaml default, which the stable-treatment
    # driver path ignores.
    import yaml as _yaml
    cohort_yaml = _yaml.safe_load(
        (mod.DEFAULTS_DIR / "mdd_stable_treatment.yaml").read_text())
    assert "lookback_days" not in cohort_yaml
    # stable knobs present at their committed defaults.
    assert eff["min_days"] == 90 and eff["max_gap_days"] == 395
    assert eff["min_history_events"] == 2
    assert eff["age_min"] == 18 and eff["age_max"] == 80


def test_build_pc_args_from_stable_treatment_defaults(monkeypatch):
    # End-to-end: the shipped defaults produce a well-formed stable-treatment argv.
    mod = _run_exp(monkeypatch)
    eff = mod.load_defaults("mdd_stable_treatment", mod.DEFAULTS_DIR)
    args = mod.build_pc_args(eff, "/runs/0072-x")
    d = dict(zip(args[::2], args[1::2]))
    assert d["--cohort"] == "mdd_stable_treatment"
    assert d["--backend"] == "vi"
    assert d["--vocab-size"] == "5000"
    assert d["--min-days"] == "90"
    # VI backend also threads the SVI schedule + save flags. subsampling 0.2 is
    # the cohort default (partition-wise sample fills executors on the small corpus).
    assert d["--subsampling-rate"] == "0.2"
    assert d["--save-dir"] == "/runs/0072-x"


def test_build_fit_args_routes_pc(monkeypatch):
    mod = _run_exp(monkeypatch)
    args = mod.build_fit_args(_eff(), "/out")
    assert "--weight-y" in args  # routed to build_pc_args


def test_pc_gets_larger_driver_memory(monkeypatch):
    mod = _run_exp(monkeypatch)
    monkeypatch.delenv("CHARM_DRIVER_MEMORY", raising=False)
    assert mod._driver_memory_for("pc") == "8g"
    assert mod._driver_memory_for("lda") == "4g"
    # env override wins for either
    monkeypatch.setenv("CHARM_DRIVER_MEMORY", "16g")
    assert mod._driver_memory_for("pc") == "16g"
    assert mod._driver_memory_for("lda") == "16g"


def test_spark_cmd_carries_driver_memory(monkeypatch):
    mod = _run_exp(monkeypatch)
    cmd = mod.build_spark_submit_cmd("x.py", ["--K", "25"], mod.REPO_ROOT,
                                     driver_memory="8g")
    assert cmd[cmd.index("--driver-memory") + 1] == "8g"
    # default stays 4g for callers that don't pass it (lda/stm/dag/eval)
    cmd2 = mod.build_spark_submit_cmd("x.py", [], mod.REPO_ROOT)
    assert cmd2[cmd2.index("--driver-memory") + 1] == "4g"

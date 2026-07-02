"""Unit + integration tests for scripts/label_phenotypes.py.

Focus: the labeler must work on STM bundles, which have NO per-topic
asymmetric-Dirichlet alpha. The rubric's only load-bearing use of alpha is
disambiguating `background` (absorbs real corpus mass) from `dead` (no mass)
among low-KL topics; for STM that disambiguation falls back on the topic's
corpus mass share (the `usage` stat), which is model-agnostic.

scripts/tests/conftest.py inserts scripts/ into sys.path.
"""
from __future__ import annotations

import json
from pathlib import Path

import label_phenotypes as lp


# --- helpers ---------------------------------------------------------------

def _system_prompt_kwargs(*, has_alpha: bool) -> dict:
    """Minimal kwargs for _build_system_prompt; alpha fields only when used."""
    common = dict(
        max_words=6,
        kl_histogram_str="(kl hist)",
        kl_min=0.01, kl_median=0.5, kl_max=2.0,
        kl_dead_threshold=0.1,
        kl_dead_threshold_explanation="(kl explanation)",
        has_alpha=has_alpha,
    )
    if has_alpha:
        common.update(
            alpha_histogram_str="(alpha hist)",
            alpha_min=0.01, alpha_median=0.05, alpha_max=0.5,
            alpha_separates_well=True,
        )
    return common


def _write_bundle(tmp: Path, *, with_alpha: bool) -> Path:
    """Write a minimal valid dashboard bundle (model/vocab/phenotypes)."""
    bundle = tmp / "data"
    bundle.mkdir(parents=True, exist_ok=True)
    vocab = {"codes": [
        {"code": f"C{i}", "description": f"Condition {i}", "corpus_freq": 0.2}
        for i in range(5)
    ]}
    beta = [
        [0.5, 0.2, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.2, 0.5],
    ]
    model = {"beta": beta}
    if with_alpha:
        model["alpha"] = [0.05, 0.30]
    phenotypes = {"phenotypes": [
        {"npmi": 0.10, "pair_coverage": 0.5, "corpus_prevalence": 0.05},
        {"npmi": 0.20, "pair_coverage": 0.8, "corpus_prevalence": 0.40},
    ]}
    (bundle / "vocab.json").write_text(json.dumps(vocab))
    (bundle / "model.json").write_text(json.dumps(model))
    (bundle / "phenotypes.json").write_text(json.dumps(phenotypes))
    return bundle


# --- per-group block context (gated STM) -----------------------------------

def test_topic_block_line_foreground_names_subgroup_and_forbids_background():
    """A foreground topic is group-specific by construction; the line must name
    the subgroup and steer AWAY from a corpus-wide 'background' reading (the
    spot-check failure mode: cancer/dementia foreground symptom clusters were
    labeled 'background')."""
    line = lp._topic_block_line(
        block="cancer", groups=["cancer", "dementia"], group_var="source_cohort")
    assert "FOREGROUND" in line
    assert "cancer" in line
    assert "background" in line.lower()   # steers away from corpus-wide background
    # ...but must NOT push a coherent foreground topic into 'dead' (t44 regression):
    # it explicitly names the subgroup-burden framing and guards the dead reading.
    assert "burden" in line.lower()
    assert "dead" in line.lower()


def test_topic_block_line_background_says_shared():
    line = lp._topic_block_line(
        block="background", groups=["cancer", "dementia"], group_var="source_cohort")
    assert "BACKGROUND" in line
    assert "shared" in line.lower()


def test_topic_block_line_none_is_empty():
    """Non-gated bundles (no gating.json) pass block=None → no block line."""
    assert lp._topic_block_line(block=None, groups=None, group_var=None) == ""


def test_user_message_includes_block_line_when_foreground():
    msg = lp._build_user_message(
        phenotype_id=3,
        top_by_freq=[{"description": "X", "weight_pct": 50.0, "lift": 2.0}],
        top_by_lift=[{"description": "X", "weight_pct": 50.0, "lift": 2.0}],
        alpha=None, kl=0.7, npmi=0.1, pair_coverage=0.5, usage_frac=0.05,
        max_words=6,
        block="dementia", groups=["cancer", "dementia"], group_var="source_cohort",
    )
    assert "FOREGROUND" in msg and "dementia" in msg


def test_user_message_omits_block_line_when_none():
    msg = lp._build_user_message(
        phenotype_id=0,
        top_by_freq=[{"description": "X", "weight_pct": 50.0, "lift": 2.0}],
        top_by_lift=[{"description": "X", "weight_pct": 50.0, "lift": 2.0}],
        alpha=None, kl=0.7, npmi=0.1, pair_coverage=0.5, usage_frac=0.05,
        max_words=6,
    )
    assert "Topic block" not in msg


# --- _build_user_message ---------------------------------------------------

def test_user_message_omits_alpha_line_when_none():
    """STM topics have no alpha; the per-topic message must drop the line
    rather than crash, while still showing the corpus-mass `usage` stat."""
    msg = lp._build_user_message(
        phenotype_id=0,
        top_by_freq=[{"description": "X", "weight_pct": 50.0, "lift": 2.0}],
        top_by_lift=[{"description": "X", "weight_pct": 50.0, "lift": 2.0}],
        alpha=None,
        kl=0.7, npmi=0.1, pair_coverage=0.5, usage_frac=0.05,
        max_words=6,
    )
    assert "alpha:" not in msg
    assert "usage:" in msg


def test_user_message_includes_alpha_line_when_present():
    """LDA/HDP path unchanged: alpha line is shown when alpha is a number."""
    msg = lp._build_user_message(
        phenotype_id=0,
        top_by_freq=[{"description": "X", "weight_pct": 50.0, "lift": 2.0}],
        top_by_lift=[{"description": "X", "weight_pct": 50.0, "lift": 2.0}],
        alpha=0.4200,
        kl=0.7, npmi=0.1, pair_coverage=0.5, usage_frac=0.05,
        max_words=6,
    )
    assert "alpha:" in msg
    assert "0.4200" in msg


# --- _build_system_prompt --------------------------------------------------

def test_system_prompt_includes_alpha_section_when_present():
    """LDA/HDP path unchanged: the alpha-distribution block is present."""
    prompt = lp._build_system_prompt(**_system_prompt_kwargs(has_alpha=True))
    assert "α distribution across this fit" in prompt


def test_system_prompt_dead_criterion_has_coherence_override():
    """A low-KL / flat topic whose top-N is thematically coherent (a recognizable
    clinical syndrome) must not be condemned to `dead` on flatness alone — flatness
    depresses KL and NPMI. Both model regimes (alpha / corpus-mass) carry the
    override so t48-style diffuse-but-real phenotypes survive."""
    for has_alpha in (True, False):
        prompt = lp._build_system_prompt(**_system_prompt_kwargs(has_alpha=has_alpha))
        assert "coherence override" in prompt.lower()
        assert "diffuse" in prompt.lower()


def test_system_prompt_omits_alpha_section_when_absent():
    """STM path: no alpha-distribution block, and the background-vs-dead
    disambiguation is phrased in terms of corpus mass share instead."""
    prompt = lp._build_system_prompt(**_system_prompt_kwargs(has_alpha=False))
    assert "α distribution across this fit" not in prompt
    assert "corpus mass" in prompt


# --- main() integration ----------------------------------------------------

def test_main_dry_run_succeeds_on_bundle_without_alpha(tmp_path, capsys):
    """The hard-fail on a missing 'alpha' array must be gone: an STM-style
    bundle (no alpha) should dry-run to completion (exit 0)."""
    bundle = _write_bundle(tmp_path, with_alpha=False)
    rc = lp.main(["--dry-run", "--bundle-dir", str(bundle)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run only" in out


def test_main_dry_run_succeeds_on_bundle_with_alpha(tmp_path, capsys):
    """Regression: the LDA path (alpha present) still dry-runs to exit 0."""
    bundle = _write_bundle(tmp_path, with_alpha=True)
    rc = lp.main(["--dry-run", "--bundle-dir", str(bundle)])
    assert rc == 0
    assert "dry-run only" in capsys.readouterr().out


def test_main_topic_ids_selects_only_those(tmp_path, capsys):
    """--topic-ids restricts the labeling set to exactly the given ids
    (used for spot-checking specific topics across blocks/kinds)."""
    bundle = _write_bundle(tmp_path, with_alpha=False)
    rc = lp.main(["--dry-run", "--bundle-dir", str(bundle), "--topic-ids", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "phenotype 1" in out
    assert "phenotype 0" not in out


def test_main_topic_ids_out_of_range_errors(tmp_path):
    """An out-of-range id is a hard error, not a silent skip."""
    import pytest
    bundle = _write_bundle(tmp_path, with_alpha=False)
    with pytest.raises(SystemExit):
        lp.main(["--dry-run", "--bundle-dir", str(bundle), "--topic-ids", "99"])


def test_stm_bundle_ignores_alpha_equivalent(tmp_path, capsys):
    """STM's exported `alpha` is softmax(Gamma[intercept]) — a baseline-
    proportion alpha-EQUIVALENT, not a Dirichlet prior. When the bundle is
    STM (detected via the STM-only `sigma` array, or an explicit model_class),
    the labeler must DROP alpha and use the corpus-mass `usage` branch, so the
    per-topic message shows no `alpha:` line and the run reports the no-alpha
    path — even though an alpha array is physically present."""
    bundle = _write_bundle(tmp_path, with_alpha=True)
    # Make it an STM bundle: add the STM-only sigma array to model.json.
    model_p = bundle / "model.json"
    model = json.loads(model_p.read_text())
    model["sigma"] = [1.0, 1.0]              # STM logistic-normal prior variance
    model_p.write_text(json.dumps(model))
    rc = lp.main(["--dry-run", "--bundle-dir", str(bundle)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "no per-topic alpha" in out       # took the STM (no-alpha) path
    assert "alpha:" not in out               # per-topic message dropped the line


# --- covariate-effect context (prevalence insights for the annotator) -------

_COV_EFFECTS = [
    {"covariate": "Intercept", "per_topic": [0.0, -1.5, 0.3, 0.1]},
    {"covariate": "C(sex)[T.M]", "per_topic": [0.0, 2.0, 0.0, -2.0]},
    {"covariate": "age", "per_topic": [0.0, 0.01, 0.02, 0.03]},
]
_COV_SCHEMA = {
    "controls": [
        {"name": "age", "type": "continuous", "range": [37, 85], "default": 70},
        {"name": "sex", "type": "categorical", "reference": "F", "levels": ["F", "M"]},
    ],
}


def test_build_covariate_context_excludes_intercept_and_reference():
    ctx = lp.build_covariate_context(_COV_EFFECTS, _COV_SCHEMA, reference_topic=0)
    cols = [c["column"] for c in ctx["covariates"]]
    assert "Intercept" not in cols          # baseline, not a covariate relationship
    assert cols == ["C(sex)[T.M]", "age"]
    sex = ctx["covariates"][0]
    # Reference topic (0) excluded from the distribution and never annotated.
    assert sex["percentiles"][0] is None
    # Distribution is over the non-reference topics {2.0, 0.0, -2.0}.
    assert sex["distribution"]["median"] == 0.0
    assert sex["distribution"]["min"] == -2.0
    assert sex["distribution"]["max"] == 2.0


def test_build_covariate_context_percentiles_signed_by_position():
    ctx = lp.build_covariate_context(_COV_EFFECTS, _COV_SCHEMA, reference_topic=0)
    pct = ctx["covariates"][0]["percentiles"]
    # topic 1 has the max male effect -> high percentile; topic 3 the min -> low.
    assert pct[1] > pct[2] > pct[3]
    assert pct[1] > 80 and pct[3] < 20


def test_build_covariate_context_none_when_intercept_only():
    intercept_only = [{"covariate": "Intercept", "per_topic": [0.0, 0.3, 0.1, 0.2]}]
    assert lp.build_covariate_context(intercept_only, _COV_SCHEMA, reference_topic=0) is None


def test_build_covariate_context_display_parsing():
    ctx = lp.build_covariate_context(_COV_EFFECTS, _COV_SCHEMA, reference_topic=0)
    sex, age = ctx["covariates"]
    assert sex["kind"] == "categorical"
    assert "sex" in sex["display"].lower() and "M" in sex["display"] and "F" in sex["display"]
    assert age["kind"] == "continuous"
    assert "age" in age["display"].lower()


def test_covariate_lines_for_topic_reference_empty_nonref_populated():
    ctx = lp.build_covariate_context(_COV_EFFECTS, _COV_SCHEMA, reference_topic=0)
    assert lp.covariate_lines_for_topic(ctx, 0) == []      # reference topic: no lines
    lines = lp.covariate_lines_for_topic(ctx, 1)
    blob = "\n".join(lines)
    assert "sex" in blob.lower()
    assert "percentile" in blob.lower()                    # position-in-distribution shown
    assert "median" in blob.lower()                        # distribution context shown
    assert lp.covariate_lines_for_topic(None, 1) == []     # no context -> no lines


def test_covariate_guidance_block_states_caveats():
    ctx = lp.build_covariate_context(_COV_EFFECTS, _COV_SCHEMA, reference_topic=0)
    g = lp.build_covariate_guidance_block(ctx).lower()
    assert "logit" in g or "log-odds" in g                 # scale is named
    assert "reference" in g                                 # reference-relative
    assert "percentile" in g or "distribution" in g        # judge relative to spread

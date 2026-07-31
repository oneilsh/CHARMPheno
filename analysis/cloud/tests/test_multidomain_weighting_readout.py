import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy import sparse as sp


RARE6_CIDS = (79145, 438688, 257628, 40352976, 76685, 432595)


def _write_synthetic_artifact(run_dir: Path) -> Path:
    """Write the real driver-local artifact contract with no clinical data."""
    from multidomain_lr_readout import save_test_set

    run_dir.mkdir()
    (run_dir / "params").mkdir()
    parent_int = {engine_id: [0] for engine_id in range(1, 7)}
    parent_int[7] = [1]
    int2cid = {
        0: -1,
        **dict(enumerate(RARE6_CIDS, start=1)),
        7: 900001,
    }
    manifest = {
        "model_class": "multidomain_gated",
        "disease": "rare6",
        "domains": ["condition", "observation", "drug"],
        "n_bg": 1,
        "tpn": 1,
        "corpus_manifest": {
            "parent_int": {
                str(node): parents for node, parents in parent_int.items()
            },
            "int2cid": {
                str(engine_id): concept_id
                for engine_id, concept_id in int2cid.items()
            },
            "name_by_id": {
                **{
                    str(concept_id): f"disease-{index}"
                    for index, concept_id in enumerate(RARE6_CIDS, start=1)
                },
                "900001": "disease-1-descendant",
            },
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    # Reordered three-domain layout: one background plus seven node topics.
    for domain in range(3):
        lam = np.full((8, 5), 1.0)
        for node in range(1, 8):
            lam[node, (node + domain) % 5] += 4.0 + domain
        np.save(run_dir / "params" / f"lambda_{domain}.npy", lam)

    # Each disease has a different four-row positive set. Row 0 reaches anchor 1
    # only through descendant node 7, exercising descendant-frontier truth.
    frontiers = [
        [7],
        [1, 2],
        [1, 3],
        [1, 4],
        [2, 3],
        [2, 4, 6],
        [2, 5, 6],
        [3, 4],
        [3, 5],
        [4, 5],
        [5, 6],
        [6],
    ]
    rows = np.arange(12)[:, None]
    columns = np.arange(5)[None, :]
    bows = {}
    for domain in range(3):
        dense = 1.0 + ((rows + 1) * (columns + domain + 1)) % 5
        dense[:, domain] += np.array(
            [sum(frontier) % (domain + 3) for frontier in frontiers],
            dtype=float,
        )
        bows[domain] = sp.csr_matrix(dense)
    save_test_set(
        run_dir,
        bows,
        np.zeros((12, 7), dtype=float),
        frontiers,
        frontiers,
        person_ids=list(range(1001, 1013)),
    )
    return run_dir


def _run_readout_cli(run_dir: Path, output_prefix: Path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "analysis" / "cloud" / "multidomain_weighting_readout.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            [
                str(repo_root / "spark-vi"),
                env.get("PYTHONPATH"),
            ],
        )
    )
    return subprocess.run(
        [
            sys.executable,
            str(script),
            "--run-dir",
            str(run_dir),
            "--outer-folds",
            "2",
            "--inner-folds",
            "2",
            "--repeats",
            "1",
            "--grid-step",
            "0.5",
            "--seed",
            "17",
            "--output-prefix",
            str(output_prefix),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_parser_defaults_are_the_preregistered_design():
    """Catches accidental drift from the preregistered nested-CV defaults."""
    from multidomain_weighting_readout import build_parser

    args = build_parser().parse_args(["--run-dir", "/runs/0072"])

    assert args.run_dir == Path("/runs/0072")
    assert args.outer_folds == 5
    assert args.inner_folds == 4
    assert args.repeats == 5
    assert args.grid_step == 0.05
    assert args.seed == 0
    assert args.output_prefix == Path("/runs/0072/multidomain_weighting")


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--outer-folds", "1"),
        ("--inner-folds", "1"),
        ("--repeats", "0"),
        ("--grid-step", "0"),
        ("--grid-step", "-0.1"),
        ("--grid-step", "0.3"),
        ("--grid-step", "nan"),
        ("--grid-step", "inf"),
        ("--grid-step", "1e-309"),
        ("--grid-step", "5e-324"),
    ],
)
def test_parser_rejects_invalid_cv_values(option, value):
    """Catches invalid CV/grid settings crossing the CLI boundary."""
    from multidomain_weighting_readout import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--run-dir", "/runs/0072", option, value])


def test_parser_preserves_an_explicit_output_prefix():
    """Catches the dynamic default overwriting an explicit report location."""
    from multidomain_weighting_readout import build_parser

    args = build_parser().parse_args(
        [
            "--run-dir",
            "/runs/0072",
            "--output-prefix",
            "/reports/reliability",
        ]
    )

    assert args.output_prefix == Path("/reports/reliability")


def test_load_artifact_reconstructs_the_real_manifest_and_dag(tmp_path):
    """Catches loading a substitute schema instead of the persisted run contract."""
    from multidomain_weighting_readout import load_artifact

    run_dir = _write_synthetic_artifact(tmp_path / "0072-synthetic")
    artifact = load_artifact(run_dir, outer_folds=2)

    assert artifact["domains"] == ["condition", "observation", "drug"]
    assert artifact["domain_labels"] == {
        0: "condition",
        1: "observation",
        2: "drug",
    }
    assert artifact["lay"].nodes == [1, 2, 3, 4, 5, 6, 7]
    assert [target["concept_id"] for target in artifact["targets"]] == list(
        RARE6_CIDS
    )
    assert [target["anchor"] for target in artifact["targets"]] == list(
        range(1, 7)
    )


def test_load_artifact_rejects_noncontiguous_lambda_keys(tmp_path):
    """Catches silently renumbering or dropping a fitted domain."""
    from multidomain_weighting_readout import load_artifact

    run_dir = _write_synthetic_artifact(tmp_path / "0072-synthetic")
    (run_dir / "params" / "lambda_2.npy").rename(
        run_dir / "params" / "lambda_3.npy"
    )

    with pytest.raises(SystemExit, match="lambda keys.*contiguous"):
        load_artifact(run_dir, outer_folds=2)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("root-list", "manifest root must be a mapping"),
        ("domains-scalar", "manifest domains must be a list"),
        ("domains-unhashable", "manifest domain entries must be strings"),
        ("domains-nonstring", "manifest domain entries must be strings"),
        ("domains-duplicate", "manifest domain names must be unique"),
        ("domains-missing-fixed", "manifest domains must include condition and drug"),
    ],
)
def test_load_artifact_rejects_malformed_manifest_domains(
    tmp_path,
    mutation,
    message,
):
    """Catches raw Python errors or collapsed report keys from malformed metadata."""
    from multidomain_weighting_readout import load_artifact

    run_dir = _write_synthetic_artifact(tmp_path / "0072-synthetic")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "root-list":
        manifest = []
    elif mutation == "domains-scalar":
        manifest["domains"] = "condition,drug"
    elif mutation == "domains-unhashable":
        manifest["domains"] = [["condition"], "observation", "drug"]
    elif mutation == "domains-nonstring":
        manifest["domains"] = [1, "observation", "1"]
    elif mutation == "domains-duplicate":
        manifest["domains"] = ["condition", "observation", "condition"]
    elif mutation == "domains-missing-fixed":
        manifest["domains"] = ["condition", "observation", "measurement"]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(SystemExit, match=message):
        load_artifact(run_dir, outer_folds=2)


def test_load_artifact_rejects_bow_row_count_mismatch(tmp_path):
    """Catches evaluating domains against misaligned document rows."""
    from multidomain_weighting_readout import load_artifact

    run_dir = _write_synthetic_artifact(tmp_path / "0072-synthetic")
    sp.save_npz(run_dir / "test_bow_1.npz", sp.csr_matrix(np.ones((11, 5))))

    with pytest.raises(SystemExit, match="BOW domain 1.*n_docs=12"):
        load_artifact(run_dir, outer_folds=2)


def test_load_artifact_rejects_lambda_bow_vocabulary_mismatch(tmp_path):
    """Catches scoring a lambda matrix against a different vocabulary."""
    from multidomain_weighting_readout import load_artifact

    run_dir = _write_synthetic_artifact(tmp_path / "0072-synthetic")
    np.save(run_dir / "params" / "lambda_0.npy", np.ones((8, 4)))

    with pytest.raises(SystemExit, match="vocabulary width.*domain 0"):
        load_artifact(run_dir, outer_folds=2)


def test_load_artifact_rejects_frontier_count_mismatch(tmp_path):
    """Catches labels that are not aligned one-for-one with BOW rows."""
    from multidomain_weighting_readout import load_artifact

    run_dir = _write_synthetic_artifact(tmp_path / "0072-synthetic")
    meta_path = run_dir / "test_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["frontiers"] = meta["frontiers"][:-1]
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(SystemExit, match="frontiers length 11.*n_docs=12"):
        load_artifact(run_dir, outer_folds=2)


@pytest.mark.parametrize(
    ("attestation", "message"),
    [
        (None, "missing person-row uniqueness attestation"),
        (
            {
                "row_count": 12,
                "unique_person_count": 11,
                "one_row_per_person": False,
            },
            "one_row_per_person.*true",
        ),
        (
            {
                "row_count": 11,
                "unique_person_count": 11,
                "one_row_per_person": True,
            },
            "row_count=11.*n_docs=12",
        ),
        (
            {
                "row_count": 12,
                "unique_person_count": 11,
                "one_row_per_person": True,
            },
            "unique_person_count=11.*n_docs=12",
        ),
    ],
)
def test_load_artifact_requires_consistent_person_uniqueness_attestation(
    tmp_path,
    attestation,
    message,
):
    """Catches row-level CV on legacy, duplicate-person, or inconsistent artifacts."""
    from multidomain_weighting_readout import load_artifact

    run_dir = _write_synthetic_artifact(tmp_path / "0072-synthetic")
    meta_path = run_dir / "test_meta.json"
    meta = json.loads(meta_path.read_text())
    if attestation is None:
        meta.pop("person_row_attestation")
    else:
        meta["person_row_attestation"] = attestation
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(SystemExit, match=message):
        load_artifact(run_dir, outer_folds=2)


def test_load_artifact_skips_an_unscoreable_anchor(tmp_path):
    """At anchor scale an unscoreable anchor is skipped + reported, not fatal;
    the remaining anchors are still evaluated."""
    from multidomain_weighting_readout import load_artifact

    run_dir = _write_synthetic_artifact(tmp_path / "0072-synthetic")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["corpus_manifest"]["int2cid"].pop("6")  # anchor 432595 -> unresolvable
    manifest_path.write_text(json.dumps(manifest))

    artifact = load_artifact(run_dir, outer_folds=2)
    skipped_ids = {int(entry["concept_id"]) for entry in artifact["skipped"]}
    assert 432595 in skipped_ids
    assert all(target["concept_id"] != 432595 for target in artifact["targets"])
    assert len(artifact["targets"]) == 5  # the other five rare6 anchors survive


def test_load_artifact_rejects_unregistered_disease(tmp_path):
    """The readout now accepts any registered disease, but rejects unknown ones."""
    from multidomain_weighting_readout import load_artifact

    run_dir = _write_synthetic_artifact(tmp_path / "0072-synthetic")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["disease"] = "not_a_disease"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(SystemExit, match="not in registry"):
        load_artifact(run_dir, outer_folds=2)


def test_load_artifact_aborts_when_no_anchor_has_enough_cases(tmp_path):
    """Thin anchors are skipped; if that leaves nothing evaluable, abort clearly."""
    from multidomain_weighting_readout import load_artifact

    run_dir = _write_synthetic_artifact(tmp_path / "0072-synthetic")
    meta_path = run_dir / "test_meta.json"
    meta = json.loads(meta_path.read_text())
    # Only one doc has any frontier, so every anchor is below outer_folds=2.
    meta["frontiers"] = [[1]] + [[] for _ in range(11)]
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(SystemExit, match="no anchor has enough held-out cases"):
        load_artifact(run_dir, outer_folds=2)


STRATEGIES = (
    "fixed:condition_drug",
    "discrete",
    "continuous",
    "model:distinctiveness",
    "model:ownership",
    "model:product",
)


def _fake_evaluation(anchor: int) -> dict:
    repeat_aps = {
        "fixed:condition_drug": (0.20, 0.40),
        "discrete": (0.30, 0.50),
        "continuous": (0.40, 0.60),
        "model:distinctiveness": (0.35, 0.55),
        "model:ownership": (0.25, 0.45),
        "model:product": (0.38, 0.58),
    }
    repeats = []
    for repeat in range(2):
        agreements = {
            model: {
                "weights": [0.7, 0.3],
                "spearman_with_continuous": (0.6, 0.8)[repeat],
                "top_set_jaccard_with_continuous": (0.25, 0.75)[repeat],
                "same_domain_order_as_median_supervised": repeat == 0,
            }
            for model in STRATEGIES
            if model.startswith("model:")
        }
        folds = [
            {
                "fold": 0,
                "test_rows": [0, 4],
                "discrete_policy": ("all|none", "drop:1|std")[repeat],
                "continuous_weights": ([0.25, 0.75], [0.75, 0.25])[repeat],
            },
            {
                "fold": 1,
                "test_rows": [1, 5],
                "discrete_policy": ("drop:1|std", "all|none")[repeat],
                "continuous_weights": [0.5, 0.5],
            },
        ]
        repeats.append(
            {
                "repeat": repeat,
                "strategies": {
                    strategy: {
                        "ap": repeat_aps[strategy][repeat],
                        "precision_at_recall": {
                            "0.1": 0.8 - repeat * 0.1,
                            "0.25": 0.7 - repeat * 0.1,
                            "0.5": 0.6 - repeat * 0.1,
                            "0.8": 0.4 - repeat * 0.1,
                        },
                    }
                    for strategy in STRATEGIES
                },
                "agreements": agreements,
                "folds": folds,
            }
        )
    return {
        "anchor": anchor,
        "n_docs": 8,
        "n_positive": 2,
        "prevalence": 0.25,
        "repeats": repeats,
    }


def _fake_report_result() -> dict:
    from multidomain_weighting_readout import build_result

    targets = [
        {
            "anchor": anchor,
            "concept_id": concept_id,
            "name": f"disease-{anchor}",
        }
        for anchor, concept_id in enumerate(RARE6_CIDS, start=1)
    ]
    artifact = {
        "run_dir": Path("/runs/0072"),
        "disease": "rare6",
        "domains": ["condition", "drug"],
        "targets": targets,
    }
    cv_config = {
        "outer_folds": 2,
        "inner_folds": 2,
        "repeats": 2,
        "grid_step": 0.5,
        "seed": 11,
    }
    evaluations = [_fake_evaluation(anchor) for anchor in range(1, 7)]
    return build_result(artifact, evaluations, cv_config=cv_config)


def test_build_result_exposes_repeat_level_and_derived_report_schema():
    """Catches dropping detail needed to reproduce summaries and diagnostics."""
    result = _fake_report_result()

    assert set(result) == {
        "run_dir",
        "disease",
        "domains",
        "cv_config",
        "anchors",
        "macro_summary",
    }
    assert len(result["anchors"]) == 6
    anchor = result["anchors"][0]
    assert set(anchor["strategies"]) == set(STRATEGIES)
    assert anchor["strategies"]["continuous"]["median_ap"] == pytest.approx(0.5)
    assert anchor["strategies"]["continuous"][
        "median_lift_over_prevalence"
    ] == pytest.approx(2.0)
    assert anchor["strategies"]["continuous"]["median_precision_at_recall"] == {
        "0.1": pytest.approx(0.75),
        "0.25": pytest.approx(0.65),
        "0.5": pytest.approx(0.55),
        "0.8": pytest.approx(0.35),
    }
    assert anchor["continuous_median_weights"] == {
        "condition": pytest.approx(0.5),
        "drug": pytest.approx(0.5),
    }
    assert anchor["discrete_policy_frequencies"] == [
        {"policy": "all|none", "count": 2, "frequency": 0.5},
        {"policy": "drop:1|std", "count": 2, "frequency": 0.5},
    ]
    assert anchor["model_vs_ceiling"]["model:product"] == {
        "median_spearman": pytest.approx(0.7),
        "median_top_set_jaccard": pytest.approx(0.5),
        "same_domain_order_frequency": pytest.approx(0.5),
    }
    assert anchor["repeats"] == _fake_evaluation(1)["repeats"]

    macro = result["macro_summary"]
    assert macro["repeats"][0]["strategies"]["continuous"]["ap"] == pytest.approx(
        0.4
    )
    assert macro["repeats"][1]["strategies"]["continuous"]["ap"] == pytest.approx(
        0.6
    )
    assert macro["strategies"]["continuous"]["median_ap"] == pytest.approx(0.5)


def test_render_markdown_prints_all_preregistered_readouts():
    """Catches a human report omitting operational or reliability diagnostics."""
    from multidomain_weighting_readout import render_markdown

    markdown = render_markdown(_fake_report_result())

    for expected in (
        "# Hybrid domain-weight readout",
        "Prevalence",
        "Lift vs prevalence",
        "P@10%",
        "P@25%",
        "P@50%",
        "P@80%",
        "Continuous median weights",
        "Selected discrete policy frequencies",
        "Model versus ceiling agreement",
        "disease-1",
        "model:distinctiveness",
        "model:ownership",
        "model:product",
    ):
        assert expected in markdown


def test_write_reports_emits_strict_deterministic_json_and_markdown(tmp_path):
    """Catches non-JSON NumPy values, NaN output, or unstable serialization."""
    from multidomain_weighting_readout import write_reports

    result = _fake_report_result()
    prefix = tmp_path / "reports" / "weighting"
    json_path, markdown_path = write_reports(result, prefix)

    assert json_path == prefix.with_suffix(".json")
    assert markdown_path == prefix.with_suffix(".md")
    assert json.loads(json_path.read_text()) == result
    assert markdown_path.read_text().startswith("# Hybrid domain-weight readout\n")
    first_json = json_path.read_text()
    write_reports(result, prefix)
    assert json_path.read_text() == first_json

    invalid = dict(result)
    invalid["macro_summary"] = {"bad": float("nan")}
    with pytest.raises(ValueError, match="Out of range float values"):
        write_reports(invalid, tmp_path / "invalid")


def test_end_to_end_synthetic_artifact_cli_writes_parseable_reports(tmp_path):
    """Catches wiring that works only with monkeypatches, Spark, or clinical data."""
    run_dir = _write_synthetic_artifact(tmp_path / "0072-synthetic")
    output_prefix = tmp_path / "readout" / "hybrid"
    completed = _run_readout_cli(run_dir, output_prefix)

    assert completed.returncode == 0, completed.stderr
    result = json.loads(Path(f"{output_prefix}.json").read_text())
    markdown = Path(f"{output_prefix}.md").read_text()
    meta = json.loads((run_dir / "test_meta.json").read_text())
    assert len(result["anchors"]) == 6
    assert result["domains"] == ["condition", "observation", "drug"]
    assert result["cv_config"]["seed"] == 17
    assert result["macro_summary"]["repeats"][0]["repeat"] == 0
    assert result["anchors"][0]["n_positive"] == 4
    assert meta["frontiers"][0] == [7]
    assert meta["person_row_attestation"] == {
        "row_count": 12,
        "unique_person_count": 12,
        "one_row_per_person": True,
    }
    allowed_policy_prefixes = {
        "all",
        "only:condition",
        "only:observation",
        "only:drug",
        "drop:condition",
        "drop:observation",
        "drop:drug",
    }
    observed_policy_prefixes = set()
    for anchor in result["anchors"]:
        assert set(anchor["strategies"]) == set(STRATEGIES)
        assert set(anchor["continuous_median_weights"]) == {
            "condition",
            "observation",
            "drug",
        }
        for repeat in anchor["repeats"]:
            for fold in repeat["folds"]:
                prefix = fold["discrete_policy"].split("|", 1)[0]
                assert prefix in allowed_policy_prefixes
                observed_policy_prefixes.add(prefix)
    assert observed_policy_prefixes != {"all"}
    independently_computed_macro = sum(
        anchor["repeats"][0]["strategies"]["continuous"]["ap"]
        for anchor in result["anchors"]
    ) / 6
    assert result["macro_summary"]["repeats"][0]["strategies"]["continuous"][
        "ap"
    ] == pytest.approx(independently_computed_macro)
    assert markdown.startswith("# Hybrid domain-weight readout\n")
    assert "[weighting] macro/per-disease median AP" in completed.stdout
    assert str(Path(f"{output_prefix}.json")) in completed.stdout
    assert str(Path(f"{output_prefix}.md")) in completed.stdout


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "missing person-row uniqueness attestation"),
        ("duplicate", "one_row_per_person must be true"),
    ],
)
def test_subprocess_rejects_unattested_or_duplicate_person_rows(
    tmp_path,
    mutation,
    message,
):
    """Catches the real CLI bypassing person-row checks performed by the loader."""
    run_dir = _write_synthetic_artifact(tmp_path / "0072-synthetic")
    meta_path = run_dir / "test_meta.json"
    meta = json.loads(meta_path.read_text())
    if mutation == "missing":
        meta.pop("person_row_attestation")
    else:
        meta["person_row_attestation"] = {
            "row_count": 12,
            "unique_person_count": 11,
            "one_row_per_person": False,
        }
    meta_path.write_text(json.dumps(meta))
    output_prefix = tmp_path / "readout" / "invalid"

    completed = _run_readout_cli(run_dir, output_prefix)

    assert completed.returncode != 0
    assert message in completed.stderr
    assert not Path(f"{output_prefix}.json").exists()
    assert not Path(f"{output_prefix}.md").exists()


def test_make_target_resolves_the_run_and_uses_python_without_spark():
    """Catches accidental Spark submission or drift from cluster artifact paths."""
    cloud_dir = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        ["make", "--dry-run", "multidomain-weighting-readout", "ID=72"],
        cwd=cloud_dir,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "analysis/cloud/multidomain_weighting_readout.py" in completed.stdout
    assert "--run-dir " in completed.stdout
    assert "/0072-*" in completed.stdout
    assert "formulaic_overlay.zip" in completed.stdout
    assert "PYTHONPATH=" in completed.stdout
    assert "spark-submit" not in completed.stdout

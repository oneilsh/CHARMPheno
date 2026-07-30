"""Artifact-only report CLI for multi-domain reliability evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import json
import math
from pathlib import Path
from statistics import median


STRATEGY_ORDER = (
    "fixed:condition_drug",
    "discrete",
    "continuous",
    "model:distinctiveness",
    "model:ownership",
    "model:product",
)
MODEL_STRATEGIES = tuple(
    strategy for strategy in STRATEGY_ORDER if strategy.startswith("model:")
)
RECALL_KEYS = ("0.1", "0.25", "0.5", "0.8")


def _integer_at_least(minimum: int):
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError("must be an integer") from None
        if parsed < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}")
        return parsed

    return parse


def _grid_step(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a finite positive number") from None
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    reciprocal = 1.0 / parsed
    if not math.isfinite(reciprocal):
        raise argparse.ArgumentTypeError("reciprocal grid size must be finite")
    units = round(reciprocal)
    if units < 1 or not math.isclose(
        parsed * units,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise argparse.ArgumentTypeError("must divide 1 exactly")
    return parsed


class _ReadoutParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        if parsed.output_prefix is None:
            parsed.output_prefix = parsed.run_dir / "multidomain_weighting"
        return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = _ReadoutParser(
        description=(
            "Nested-CV supervised ceiling and model-derived domain reliability "
            "readout from an existing multidomain run."
        )
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--outer-folds", type=_integer_at_least(2), default=5)
    parser.add_argument("--inner-folds", type=_integer_at_least(2), default=4)
    parser.add_argument("--repeats", type=_integer_at_least(1), default=5)
    parser.add_argument("--grid-step", type=_grid_step, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-prefix", type=Path)
    return parser


def _abort(message: str) -> None:
    raise SystemExit(f"[weighting] {message}")


def load_artifact(
    run_dir: Path,
    *,
    outer_folds: int,
) -> dict:
    """Load and validate one persisted rare6 multidomain run.

    This mirrors ``multidomain_lr_readout.main`` for sidecar loading and DAG
    reconstruction, then establishes every alignment invariant needed before
    nested CV starts.
    """
    from charmpheno.omop.cohorts import disease_anchors
    from multidomain_lr_readout import (
        load_lambda_dict,
        load_person_row_attestation,
        load_test_set,
        scoreable_targets,
        subtree_nodes,
    )
    from spark_vi.models.topic.dag_placement import DagLayout

    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except FileNotFoundError:
        _abort(f"missing manifest.json under {run_dir}")
    except json.JSONDecodeError as error:
        _abort(f"invalid manifest.json under {run_dir}: {error}")
    if not isinstance(manifest, Mapping):
        _abort("manifest root must be a mapping")

    disease = manifest.get("disease")
    if disease != "rare6":
        _abort(
            f"expected a rare6 artifact, found disease={disease!r} in "
            f"{manifest_path}"
        )

    lam_dict = load_lambda_dict(run_dir)
    expected_keys = list(range(len(lam_dict)))
    actual_keys = sorted(lam_dict)
    if actual_keys != expected_keys:
        _abort(
            "lambda keys must be contiguous 0..n_domains-1; "
            f"found {actual_keys}"
        )
    n_domains = len(lam_dict)
    domains = manifest.get("domains")
    if not isinstance(domains, list):
        _abort("manifest domains must be a list")
    if any(not isinstance(domain, str) for domain in domains):
        _abort("manifest domain entries must be strings")
    if len(domains) != n_domains:
        _abort(
            "manifest domains must provide one name per lambda domain; "
            f"found {len(domains)} names for {n_domains} domains"
        )
    normalized_domains = [str(domain) for domain in domains]
    if len(set(normalized_domains)) != n_domains:
        _abort("manifest domain names must be unique after output normalization")
    if not {"condition", "drug"}.issubset(normalized_domains):
        _abort("manifest domains must include condition and drug")
    domain_labels = dict(zip(expected_keys, normalized_domains))

    try:
        corpus_manifest = manifest["corpus_manifest"]
        parent_int = {
            int(node): [int(parent) for parent in parents]
            for node, parents in corpus_manifest["parent_int"].items()
        }
        lay = DagLayout(
            parent_int,
            n_bg=int(manifest["n_bg"]),
            tpn=int(manifest["tpn"]),
        )
        int2cid = {
            int(engine_id): int(concept_id)
            for engine_id, concept_id in corpus_manifest["int2cid"].items()
        }
        name_by_id = {
            int(concept_id): str(name)
            for concept_id, name in corpus_manifest["name_by_id"].items()
        }
    except (KeyError, TypeError, ValueError) as error:
        _abort(f"manifest DAG metadata is invalid: {error}")
    cid2int = {concept_id: engine_id for engine_id, concept_id in int2cid.items()}

    for domain, lam in sorted(lam_dict.items()):
        if lam.ndim != 2 or lam.shape[0] != lay.K:
            _abort(
                f"lambda domain {domain} must have shape "
                f"({lay.K}, vocabulary); found {lam.shape}"
            )

    try:
        bows, frontiers, _, _, n_docs = load_test_set(run_dir, n_domains)
    except SystemExit:
        raise
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        _abort(f"test-set artifact is incomplete or invalid: {error}")

    for domain in expected_keys:
        bow = bows[domain]
        if bow.shape[0] != n_docs:
            _abort(
                f"BOW domain {domain} has {bow.shape[0]} rows; "
                f"expected n_docs={n_docs}"
            )
        if lam_dict[domain].shape[1] != bow.shape[1]:
            _abort(
                f"lambda/BOW vocabulary width mismatch for domain {domain}: "
                f"lambda={lam_dict[domain].shape[1]}, BOW={bow.shape[1]}"
            )
    if len(frontiers) != n_docs:
        _abort(
            f"frontiers length {len(frontiers)} does not match n_docs={n_docs}"
        )
    try:
        person_row_attestation = load_person_row_attestation(run_dir, n_docs)
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError) as error:
        _abort(str(error))

    targets = []
    for concept_id in disease_anchors(disease):
        resolved = scoreable_targets([concept_id], cid2int, lay, parent_int)
        if len(resolved) != 1 or resolved[0][1]:
            _abort(f"rare6 anchor {concept_id} does not resolve to a scoreable node")
        anchor = int(resolved[0][0])
        subtree = subtree_nodes(parent_int, anchor) & set(lay.nodes)
        n_positive = sum(bool(set(frontier) & subtree) for frontier in frontiers)
        n_negative = n_docs - n_positive
        if min(n_positive, n_negative) < outer_folds:
            _abort(
                f"anchor {anchor} ({concept_id}) has {n_positive} positives and "
                f"{n_negative} negatives; both must be at least "
                f"outer_folds={outer_folds}"
            )
        targets.append(
            {
                "anchor": anchor,
                "concept_id": int(concept_id),
                "name": name_by_id.get(int(concept_id), str(concept_id)),
            }
        )

    return {
        "run_dir": run_dir,
        "manifest": manifest,
        "disease": disease,
        "domains": normalized_domains,
        "domain_labels": domain_labels,
        "lam_dict": lam_dict,
        "bows": bows,
        "frontiers": frontiers,
        "lay": lay,
        "parent_int": parent_int,
        "targets": targets,
        "n_docs": int(n_docs),
        "person_row_attestation": person_row_attestation,
    }


def _median(values):
    values = [float(value) for value in values if value is not None]
    return float(median(values)) if values else None


def _validate_evaluations(artifact, evaluations, cv_config):
    targets = artifact["targets"]
    expected_repeats = int(cv_config["repeats"])
    if len(targets) != 6 or len(evaluations) != 6:
        _abort(
            "macro reporting requires exactly six rare6 targets and evaluations; "
            f"found {len(targets)} targets and {len(evaluations)} evaluations"
        )
    for target, evaluation in zip(targets, evaluations):
        if int(evaluation.get("anchor", -1)) != int(target["anchor"]):
            _abort(
                "evaluation order must match rare6 target order; "
                f"expected anchor {target['anchor']}, "
                f"found {evaluation.get('anchor')}"
            )
        repeats = evaluation.get("repeats", [])
        if len(repeats) != expected_repeats:
            _abort(
                f"anchor {target['anchor']} has {len(repeats)} repeat results; "
                f"expected {expected_repeats}"
            )
        for repeat_index, repeat in enumerate(repeats):
            if repeat.get("repeat") != repeat_index:
                _abort(
                    f"anchor {target['anchor']} repeat ids must be contiguous "
                    f"from 0; found {repeat.get('repeat')} at position "
                    f"{repeat_index}"
                )
            strategies = repeat.get("strategies", {})
            if tuple(strategies) != STRATEGY_ORDER:
                _abort(
                    f"anchor {target['anchor']} repeat {repeat_index} strategies "
                    f"must be {STRATEGY_ORDER}; found {tuple(strategies)}"
                )
            for strategy in STRATEGY_ORDER:
                metric = strategies[strategy]
                if tuple(metric.get("precision_at_recall", {})) != RECALL_KEYS:
                    _abort(
                        f"anchor {target['anchor']} strategy {strategy} must "
                        f"report recalls {RECALL_KEYS}"
                    )


def _anchor_summary(target, evaluation, domains):
    prevalence = float(evaluation["prevalence"])
    strategy_summaries = {}
    for strategy in STRATEGY_ORDER:
        metrics = [
            repeat["strategies"][strategy] for repeat in evaluation["repeats"]
        ]
        aps = [float(metric["ap"]) for metric in metrics]
        strategy_summaries[strategy] = {
            "repeat_ap": aps,
            "median_ap": _median(aps),
            "median_lift_over_prevalence": _median(
                [ap / prevalence for ap in aps]
            ),
            "median_precision_at_recall": {
                recall: _median(
                    [
                        metric["precision_at_recall"][recall]
                        for metric in metrics
                    ]
                )
                for recall in RECALL_KEYS
            },
        }

    folds = [
        fold
        for repeat in evaluation["repeats"]
        for fold in repeat["folds"]
    ]
    weight_rows = [fold["continuous_weights"] for fold in folds]
    if not weight_rows or any(len(weights) != len(domains) for weights in weight_rows):
        _abort(
            f"anchor {target['anchor']} continuous weights must have "
            f"{len(domains)} entries in every fold"
        )
    continuous_median_weights = {
        domain: _median(weights[index] for weights in weight_rows)
        for index, domain in enumerate(domains)
    }

    policy_counts = Counter(str(fold["discrete_policy"]) for fold in folds)
    total_policies = sum(policy_counts.values())
    policy_frequencies = [
        {
            "policy": policy,
            "count": int(count),
            "frequency": float(count / total_policies),
        }
        for policy, count in sorted(
            policy_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]

    model_vs_ceiling = {}
    for strategy in MODEL_STRATEGIES:
        agreements = [
            repeat["agreements"][strategy] for repeat in evaluation["repeats"]
        ]
        model_vs_ceiling[strategy] = {
            "median_spearman": _median(
                agreement["spearman_with_continuous"]
                for agreement in agreements
            ),
            "median_top_set_jaccard": _median(
                agreement["top_set_jaccard_with_continuous"]
                for agreement in agreements
            ),
            "same_domain_order_frequency": float(
                sum(
                    bool(agreement["same_domain_order_as_median_supervised"])
                    for agreement in agreements
                )
                / len(agreements)
            ),
        }

    return {
        **target,
        "n_docs": int(evaluation["n_docs"]),
        "n_positive": int(evaluation["n_positive"]),
        "prevalence": prevalence,
        "strategies": strategy_summaries,
        "continuous_median_weights": continuous_median_weights,
        "discrete_policy_frequencies": policy_frequencies,
        "model_vs_ceiling": model_vs_ceiling,
        "repeats": evaluation["repeats"],
    }


def _macro_summary(anchors, *, repeats):
    mean_prevalence = float(
        sum(anchor["prevalence"] for anchor in anchors) / len(anchors)
    )
    repeat_rows = []
    for repeat in range(repeats):
        strategies = {}
        for strategy in STRATEGY_ORDER:
            ap = float(
                sum(
                    anchor["repeats"][repeat]["strategies"][strategy]["ap"]
                    for anchor in anchors
                )
                / len(anchors)
            )
            strategies[strategy] = {
                "ap": ap,
                "lift_over_prevalence": float(ap / mean_prevalence),
            }
        repeat_rows.append({"repeat": int(repeat), "strategies": strategies})
    return {
        "mean_prevalence": mean_prevalence,
        "repeats": repeat_rows,
        "strategies": {
            strategy: {
                "repeat_ap": [
                    repeat["strategies"][strategy]["ap"]
                    for repeat in repeat_rows
                ],
                "median_ap": _median(
                    repeat["strategies"][strategy]["ap"]
                    for repeat in repeat_rows
                ),
                "median_lift_over_prevalence": _median(
                    repeat["strategies"][strategy]["lift_over_prevalence"]
                    for repeat in repeat_rows
                ),
            }
            for strategy in STRATEGY_ORDER
        },
    }


def build_result(artifact: dict, evaluations: list[dict], *, cv_config: dict) -> dict:
    """Build the deterministic, JSON-safe report schema from anchor evaluations."""
    _validate_evaluations(artifact, evaluations, cv_config)
    domains = list(artifact["domains"])
    anchors = [
        _anchor_summary(target, evaluation, domains)
        for target, evaluation in zip(artifact["targets"], evaluations)
    ]
    return {
        "run_dir": str(artifact["run_dir"]),
        "disease": str(artifact["disease"]),
        "domains": domains,
        "cv_config": {
            "outer_folds": int(cv_config["outer_folds"]),
            "inner_folds": int(cv_config["inner_folds"]),
            "repeats": int(cv_config["repeats"]),
            "grid_step": float(cv_config["grid_step"]),
            "seed": int(cv_config["seed"]),
        },
        "anchors": anchors,
        "macro_summary": _macro_summary(
            anchors,
            repeats=int(cv_config["repeats"]),
        ),
    }


def _format_number(value, digits=3):
    return "NA" if value is None else f"{float(value):.{digits}f}"


def _markdown_cell(value) -> str:
    return str(value).replace("|", r"\|")


def render_markdown(result: dict) -> str:
    """Render a concise human readout from the machine-readable result."""
    lines = [
        "# Hybrid domain-weight readout",
        "",
        f"- Run: `{result['run_dir']}`",
        f"- Disease cohort: `{result['disease']}`",
        f"- Domains: {', '.join(result['domains'])}",
        "",
        "## Macro summary",
        "",
        f"Mean prevalence: {_format_number(result['macro_summary']['mean_prevalence'], 4)}",
        "",
        "| Strategy | Median AP | Lift vs prevalence |",
        "|---|---:|---:|",
    ]
    for strategy in STRATEGY_ORDER:
        summary = result["macro_summary"]["strategies"][strategy]
        lines.append(
            f"| {_markdown_cell(strategy)} | "
            f"{_format_number(summary['median_ap'])} | "
            f"{_format_number(summary['median_lift_over_prevalence'])} |"
        )
    lines.extend(
        [
            "",
            "### Macro AP by repeat",
            "",
            "| Repeat | "
            + " | ".join(_markdown_cell(strategy) for strategy in STRATEGY_ORDER)
            + " |",
            "|---:|" + "---:|" * len(STRATEGY_ORDER),
        ]
    )
    for repeat in result["macro_summary"]["repeats"]:
        lines.append(
            f"| {repeat['repeat']} | "
            + " | ".join(
                _format_number(repeat["strategies"][strategy]["ap"])
                for strategy in STRATEGY_ORDER
            )
            + " |"
        )

    for anchor in result["anchors"]:
        lines.extend(
            [
                "",
                f"## {_markdown_cell(anchor['name'])}",
                "",
                f"- Anchor: `{anchor['anchor']}` (concept `{anchor['concept_id']}`)",
                f"- Prevalence: {_format_number(anchor['prevalence'], 4)} "
                f"({anchor['n_positive']}/{anchor['n_docs']})",
                "",
                "| Strategy | Median AP | Lift vs prevalence | P@10% | P@25% | "
                "P@50% | P@80% |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for strategy in STRATEGY_ORDER:
            summary = anchor["strategies"][strategy]
            precision = summary["median_precision_at_recall"]
            lines.append(
                f"| {_markdown_cell(strategy)} | "
                f"{_format_number(summary['median_ap'])} | "
                f"{_format_number(summary['median_lift_over_prevalence'])} | "
                + " | ".join(
                    _format_number(precision[recall]) for recall in RECALL_KEYS
                )
                + " |"
            )

        lines.extend(["", "### Continuous median weights", ""])
        lines.append(
            ", ".join(
                f"`{domain}`={_format_number(weight)}"
                for domain, weight in anchor["continuous_median_weights"].items()
            )
        )
        lines.extend(
            [
                "",
                "### Selected discrete policy frequencies",
                "",
                "| Policy | Count | Frequency |",
                "|---|---:|---:|",
            ]
        )
        for policy in anchor["discrete_policy_frequencies"]:
            lines.append(
                f"| {_markdown_cell(policy['policy'])} | {policy['count']} | "
                f"{_format_number(policy['frequency'])} |"
            )
        lines.extend(
            [
                "",
                "### Model versus ceiling agreement",
                "",
                "| Model | Median Spearman | Median top-set Jaccard | "
                "Same domain order frequency |",
                "|---|---:|---:|---:|",
            ]
        )
        for strategy in MODEL_STRATEGIES:
            agreement = anchor["model_vs_ceiling"][strategy]
            lines.append(
                f"| {_markdown_cell(strategy)} | "
                f"{_format_number(agreement['median_spearman'])} | "
                f"{_format_number(agreement['median_top_set_jaccard'])} | "
                f"{_format_number(agreement['same_domain_order_frequency'])} |"
            )
    return "\n".join(lines) + "\n"


def write_reports(result: dict, output_prefix: Path) -> tuple[Path, Path]:
    """Write strict deterministic JSON and the corresponding Markdown report."""
    json_text = json.dumps(
        result,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    markdown_text = render_markdown(result)
    output_prefix = Path(output_prefix)
    json_path = Path(f"{output_prefix}.json")
    markdown_path = Path(f"{output_prefix}.md")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json_text)
    markdown_path.write_text(markdown_text)
    return json_path, markdown_path


def _print_ap_table(result: dict) -> None:
    strategies = STRATEGY_ORDER
    print("[weighting] macro/per-disease median AP", flush=True)
    print(
        "[weighting] "
        + "disease".ljust(24)
        + "".join(strategy[:18].rjust(20) for strategy in strategies),
        flush=True,
    )

    def print_row(label, summaries):
        print(
            "[weighting] "
            + str(label)[:24].ljust(24)
            + "".join(
                _format_number(summaries[strategy]["median_ap"]).rjust(20)
                for strategy in strategies
            ),
            flush=True,
        )

    print_row("MACRO", result["macro_summary"]["strategies"])
    for anchor in result["anchors"]:
        print_row(anchor["name"], anchor["strategies"])


def main(argv=None) -> int:
    """Run the Spark-free artifact evaluation and write both report formats."""
    from multidomain_weighting import evaluate_anchor_nested

    args = build_parser().parse_args(argv)
    cv_config = {
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "repeats": args.repeats,
        "grid_step": args.grid_step,
        "seed": args.seed,
    }
    artifact = load_artifact(
        args.run_dir,
        outer_folds=args.outer_folds,
    )
    evaluations = []
    for target in artifact["targets"]:
        try:
            evaluations.append(
                evaluate_anchor_nested(
                    artifact["bows"],
                    artifact["lam_dict"],
                    artifact["lay"],
                    artifact["frontiers"],
                    anchor=target["anchor"],
                    parent_int=artifact["parent_int"],
                    domain_labels=artifact["domain_labels"],
                    **cv_config,
                )
            )
        except ValueError as error:
            _abort(
                f"anchor {target['anchor']} ({target['concept_id']}) "
                f"evaluation failed: {error}"
            )

    result = build_result(artifact, evaluations, cv_config=cv_config)
    json_path, markdown_path = write_reports(result, args.output_prefix)
    _print_ap_table(result)
    print(f"[weighting] JSON: {json_path}", flush=True)
    print(f"[weighting] Markdown: {markdown_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

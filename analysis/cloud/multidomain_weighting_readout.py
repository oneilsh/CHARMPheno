"""Artifact-only report CLI for multi-domain reliability evaluation."""
from __future__ import annotations

import argparse
import concurrent.futures
from collections import Counter
from collections.abc import Mapping
import json
import math
import multiprocessing
import os
import sys
import time
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
    parser.add_argument(
        "--fixed-only", action="store_true",
        help="FAST parameter-free readout: per-domain and fixed-inclusive "
             "(equal-weight all-domain) case-finding AP only. Skips the "
             "supervised nested-CV weight search (the slow part, and the part "
             "insight 0076 closed), so --inner-folds and --grid-step are ignored. "
             "Runs ~1000x faster; use to compare a fixed domain combination "
             "against a prior fit's fixed AP.",
    )
    parser.add_argument(
        "--jobs", type=_integer_at_least(1), default=1,
        help="parallel worker processes over anchors (anchors are independent; "
             "results are identical and in target order regardless of --jobs)",
    )
    return parser


def _abort(message: str) -> None:
    raise SystemExit(f"[weighting] {message}")


def load_artifact(
    run_dir: Path,
    *,
    outer_folds: int,
) -> dict:
    """Load and validate one persisted multidomain run (any registered disease).

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
    if not isinstance(disease, str) or not disease:
        _abort(f"manifest disease must be a non-empty string; found {disease!r}")
    try:
        disease_anchors(disease)
    except ValueError as error:
        _abort(str(error))

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

    # Build the scoreable target list. At anchor scale (N diseases, not just the
    # six rare6) some anchors will not resolve to a DAG node, or will have too few
    # held-out cases for nested CV; skip and report those rather than aborting the
    # whole readout, so the evaluable anchors still get scored.
    targets = []
    skipped = []
    for concept_id in disease_anchors(disease):
        name = name_by_id.get(int(concept_id), str(concept_id))
        resolved = scoreable_targets([concept_id], cid2int, lay, parent_int)
        if len(resolved) != 1 or resolved[0][1]:
            skipped.append({
                "concept_id": int(concept_id), "name": name,
                "reason": "anchor does not resolve to a single scoreable DAG node",
            })
            continue
        anchor = int(resolved[0][0])
        subtree = subtree_nodes(parent_int, anchor) & set(lay.nodes)
        n_positive = sum(bool(set(frontier) & subtree) for frontier in frontiers)
        n_negative = n_docs - n_positive
        if min(n_positive, n_negative) < outer_folds:
            skipped.append({
                "anchor": anchor, "concept_id": int(concept_id), "name": name,
                "reason": (f"too few held-out cases ({n_positive} positive / "
                           f"{n_negative} negative; need >= outer_folds={outer_folds})"),
            })
            continue
        targets.append({
            "anchor": anchor,
            "concept_id": int(concept_id),
            "name": name,
            "n_positive": int(n_positive),
        })

    if not targets:
        _abort(
            f"no anchor has enough held-out cases to evaluate at "
            f"outer_folds={outer_folds} (all {len(skipped)} anchors skipped)"
        )
    if skipped:
        sys.stderr.write(
            f"[weighting] skipping {len(skipped)}/{len(targets) + len(skipped)} "
            "anchors (unresolved or too few held-out cases): "
            + ", ".join(str(s["name"]) for s in skipped[:10])
            + (" ..." if len(skipped) > 10 else "") + "\n"
        )

    return {
        "skipped": skipped,
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
    if not targets or len(targets) != len(evaluations):
        _abort(
            "macro reporting requires at least one target with a matching "
            f"evaluation; found {len(targets)} targets and {len(evaluations)} "
            "evaluations"
        )
    for target, evaluation in zip(targets, evaluations):
        if int(evaluation.get("anchor", -1)) != int(target["anchor"]):
            _abort(
                "evaluation order must match target order; "
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

    skipped = result.get("skipped") or []
    if skipped:
        lines.extend([
            "",
            f"## Skipped anchors ({len(skipped)})",
            "",
            "| Anchor | Concept | Reason |",
            "|---|---|---|",
        ])
        for entry in skipped:
            lines.append(
                f"| {_markdown_cell(entry.get('name', '?'))} | "
                f"{_markdown_cell(entry.get('concept_id', '—'))} | "
                f"{_markdown_cell(entry.get('reason', ''))} |"
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


# --- per-anchor parallelism (anchors are independent) ---

# Read-only artifact data, stashed as process globals by the pool initializer so
# each per-anchor task carries only the (small) target dict, not the corpus.
_WORKER: dict = {}


def _worker_init(bows, lam_dict, lay, frontiers, parent_int, domain_labels, cv_config):
    """Pool initializer: pin BLAS to one thread per worker (avoid oversubscription
    on a small master) and stash the artifact data as process globals."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    _WORKER.update(
        bows=bows, lam_dict=lam_dict, lay=lay, frontiers=frontiers,
        parent_int=parent_int, domain_labels=domain_labels, cv_config=cv_config,
    )


def _worker_eval(target):
    """Evaluate one anchor from the process-global artifact data. Returns
    ``(target, evaluation, error_message)`` and never raises, so one bad anchor
    cannot sink the pool."""
    import multidomain_weighting as mw

    cv_config = dict(_WORKER["cv_config"])
    fixed = cv_config.pop("mode", "nested") == "fixed"
    # Attribute access (not a from-import of both names) so the nested path never
    # touches evaluate_anchor_fixed — keeps nested-only test stubs valid.
    evaluate = mw.evaluate_anchor_fixed if fixed else mw.evaluate_anchor_nested
    try:
        evaluation = evaluate(
            _WORKER["bows"], _WORKER["lam_dict"], _WORKER["lay"],
            _WORKER["frontiers"], anchor=target["anchor"],
            parent_int=_WORKER["parent_int"],
            domain_labels=_WORKER["domain_labels"], **cv_config,
        )
    except ValueError as error:
        return (target, None, str(error))
    return (target, evaluation, None)


def _evaluate_targets(artifact, cv_config, *, jobs):
    """Evaluate every target, serially (jobs<=1) or across a fork-based process
    pool, in target order. Identical ``(target, evaluation, error)`` triples
    either way — each anchor's nested CV is independent and seeded identically."""
    init_args = (
        artifact["bows"], artifact["lam_dict"], artifact["lay"],
        artifact["frontiers"], artifact["parent_int"], artifact["domain_labels"],
        cv_config,
    )
    targets = artifact["targets"]
    total = len(targets)

    def _tick(done, name):
        sys.stderr.write(f"[weighting] {done}/{total} anchors done (last: {name})\n")
        sys.stderr.flush()

    print(f"[weighting] evaluating {total} anchors (jobs={jobs}) ...",
          file=sys.stderr, flush=True)
    start = time.perf_counter()

    if jobs <= 1:
        _worker_init(*init_args)  # populate globals for the in-process path
        results = []
        for done, target in enumerate(targets, 1):
            results.append(_worker_eval(target))
            _tick(done, target["name"])
    else:
        ctx = multiprocessing.get_context("fork")
        results = [None] * total
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=jobs, mp_context=ctx,
            initializer=_worker_init, initargs=init_args,
        ) as executor:
            futures = {executor.submit(_worker_eval, t): i
                       for i, t in enumerate(targets)}
            for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
                index = futures[future]
                results[index] = future.result()  # completes out of order...
                _tick(done, targets[index]["name"])
        # ... but results stay in target order for a deterministic report.

    print(f"[weighting] evaluated {total} anchors in "
          f"{time.perf_counter() - start:.0f}s", file=sys.stderr, flush=True)
    return results


def _partition_results(results):
    """Split ``(target, evaluation, error)`` triples into aligned evaluated
    targets + evaluations, and eval-failure skip records."""
    evaluations, evaluated_targets, failures = [], [], []
    for target, evaluation, error in results:
        if error is not None:
            failures.append({
                "anchor": target["anchor"], "concept_id": target["concept_id"],
                "name": target["name"], "reason": f"evaluation failed: {error}",
            })
        else:
            evaluations.append(evaluation)
            evaluated_targets.append(target)
    return evaluations, evaluated_targets, failures


def build_fixed_result(artifact, evaluations) -> dict:
    """Compact summary for the --fixed-only path: per-anchor and macro median AP
    (+ median precision-at-recall) for each domain-alone strategy and the
    fixed-inclusive combination. Median-across-repeats per anchor, then
    median-across-anchors for the macro row (mirrors _macro_summary)."""
    order = list(evaluations[0]["strategy_order"])
    anchors = []
    for target, ev in zip(artifact["targets"], evaluations):
        strategies = {}
        for name in order:
            reps = ev["repeats"]
            strategies[name] = {
                "ap": median([r["strategies"][name]["ap"] for r in reps]),
                "median_precision_at_recall": {
                    rk: median([r["strategies"][name]["precision_at_recall"][rk]
                                for r in reps])
                    for rk in RECALL_KEYS},
            }
        anchors.append({
            "anchor": ev["anchor"], "concept_id": target["concept_id"],
            "name": target["name"], "n_positive": ev["n_positive"],
            "prevalence": ev["prevalence"], "strategies": strategies,
        })
    macro = {
        name: {
            "ap": median([a["strategies"][name]["ap"] for a in anchors]),
            "median_precision_at_recall": {
                rk: median([a["strategies"][name]["median_precision_at_recall"][rk]
                            for a in anchors])
                for rk in RECALL_KEYS},
        } for name in order}
    return {"mode": "fixed", "strategy_order": order, "macro": macro,
            "anchors": anchors,
            "mean_prevalence": median([a["prevalence"] for a in anchors])}


def render_fixed_markdown(result: dict) -> str:
    order = result["strategy_order"]
    lines = ["# Multidomain fixed-combination case-finding readout", "",
             f"Anchors scored: {len(result['anchors'])}  |  "
             f"mean prevalence: {_format_number(result['mean_prevalence'], 4)}", "",
             "Parameter-free: per-domain and fixed-inclusive AP only (no supervised "
             "weighting). Median across repeats, then across anchors.", "",
             "## Macro median AP", "",
             "| strategy | AP | " + " | ".join(f"P@{r}" for r in RECALL_KEYS) + " |",
             "| --- | --- |" + " --- |" * len(RECALL_KEYS)]
    for name in order:
        m = result["macro"][name]
        prec = " | ".join(_format_number(m["median_precision_at_recall"][r])
                          for r in RECALL_KEYS)
        lines.append(f"| {name} | {_format_number(m['ap'])} | {prec} |")
    lines += ["", "## Per-anchor AP", "",
              "| anchor | n+ | prev | " + " | ".join(order) + " |",
              "| --- | --- | --- |" + " --- |" * len(order)]
    for a in sorted(result["anchors"], key=lambda x: x["strategies"][order[-1]]["ap"],
                    reverse=True):
        aps = " | ".join(_format_number(a["strategies"][n]["ap"]) for n in order)
        lines.append(f"| {a['name']} | {a['n_positive']} | "
                     f"{_format_number(a['prevalence'], 4)} | {aps} |")
    return "\n".join(lines) + "\n"


def _print_fixed_table(result: dict) -> None:
    order = result["strategy_order"]
    print("[weighting] fixed-combination macro/per-anchor median AP", flush=True)
    header = f"{'anchor':32s} {'n+':>5s} " + " ".join(f"{n[:12]:>12s}" for n in order)
    print(header, flush=True)
    m = result["macro"]
    print(f"{'MACRO':32s} {'':>5s} "
          + " ".join(f"{m[n]['ap']:>12.3f}" for n in order), flush=True)
    for a in sorted(result["anchors"],
                    key=lambda x: x["strategies"][order[-1]]["ap"], reverse=True):
        print(f"{a['name'][:32]:32s} {a['n_positive']:>5d} "
              + " ".join(f"{a['strategies'][n]['ap']:>12.3f}" for n in order),
              flush=True)


def main(argv=None) -> int:
    """Run the Spark-free artifact evaluation and write both report formats."""
    args = build_parser().parse_args(argv)
    if args.fixed_only:
        cv_config = {"outer_folds": args.outer_folds, "repeats": args.repeats,
                     "seed": args.seed, "mode": "fixed"}
    else:
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
    # Evaluate anchors (independent), serially or across --jobs workers.
    results = _evaluate_targets(artifact, cv_config, jobs=args.jobs)
    evaluations, evaluated_targets, eval_failures = _partition_results(results)
    for failure in eval_failures:
        # A single anchor's nested-CV failure is recorded, not fatal, at scale.
        sys.stderr.write(
            f"[weighting] skipping {failure['name']} ({failure['concept_id']}): "
            f"{failure['reason']}\n"
        )

    # Only successfully-evaluated targets go into the report (kept aligned with
    # evaluations); everything dropped is reported under "skipped".
    artifact["targets"] = evaluated_targets
    if not evaluations:
        _abort("every anchor evaluation failed; nothing to report")

    if args.fixed_only:
        result = build_fixed_result(artifact, evaluations)
        result["skipped"] = list(artifact.get("skipped", [])) + eval_failures
        # Distinct prefix so a fixed report never clobbers a nested one.
        prefix = Path(f"{args.output_prefix}_fixed")
        json_text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        prefix.parent.mkdir(parents=True, exist_ok=True)
        json_path = Path(f"{prefix}.json")
        markdown_path = Path(f"{prefix}.md")
        json_path.write_text(json_text)
        markdown_path.write_text(render_fixed_markdown(result))
        _print_fixed_table(result)
        if result["skipped"]:
            print(f"[weighting] scored {len(evaluations)} anchors; skipped "
                  f"{len(result['skipped'])} — see report", flush=True)
        print(f"[weighting] JSON: {json_path}", flush=True)
        print(f"[weighting] Markdown: {markdown_path}", flush=True)
        return 0

    result = build_result(artifact, evaluations, cv_config=cv_config)
    result["skipped"] = list(artifact.get("skipped", [])) + eval_failures
    json_path, markdown_path = write_reports(result, args.output_prefix)
    _print_ap_table(result)
    if result["skipped"]:
        print(
            f"[weighting] scored {len(evaluations)} anchors; "
            f"skipped {len(result['skipped'])} (unresolved / too few held-out "
            "cases / eval failure) — see report",
            flush=True,
        )
    print(f"[weighting] JSON: {json_path}", flush=True)
    print(f"[weighting] Markdown: {markdown_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Standalone post-hoc case-finding readout on a FINISHED gated_pc run (no re-fit).

Loads a completed run's saved globals (gated_pc_result.npz: lambda/alpha/w_CK) +
manifest.json, reconstructs the OnlinePCLDAModel, reloads the cached
CaseFindingBundle the run was fit against, re-transforms train+test, and prints
the FULL case-finding readout for the gated_pc arm:

  - pc_topics_lr:   ranking AUC/AP + per-node precision@recall / recall@FDR +
                    case-vs-background detection (AUC/AP + precision@recall).
  - co-fit head:    the same readout on the head's own P(node)=sigmoid(w_CK·θ).

The other arms' summary AUC/AP are echoed from manifest["results"] for context
(only the gated_pc arm's globals are saved, so only it can be re-scored here; a
run fit with the current driver already computes+stores every arm's full readout
inline — this script is for re-reading, richer operating points, or a run whose
inline readout predates this eval).

The bundle is located by RECOMPUTING its assembly cache key from the manifest
(same fragility as lr_readout): if the assembly/DAG source changed since the fit,
or a key input is missing from an older manifest (notably --doc-min-length, which
older runs did not record — pass --doc-min-length), the key MISSES and you must
pass --bundle-path at the exact cached bundle dir.

Both readout paths of ADR 0046 are available here, with the same
`--readout-mode {driver,distributed,auto}` / `--readout-ab-check` semantics as the
fit driver: this tool IS the recovery path when a finished fit's readout output was
lost (exp 0103), and at whole-Mondo C the driver-collect readout is exactly what
does not fit — so re-reading has to be able to run distributed without re-fitting.

SUBMIT NOTE: the distributed path's mapPartitions kernels pickle by MODULE
REFERENCE, so `analysis/cloud/distributed_readout.py` must be importable on the
EXECUTORS as top-level `distributed_readout` — i.e. it must ride in --py-files
(the `gated-pc-readout` Makefile target does this; a hand-rolled spark-submit must
add `--py-files ...,<repo>/analysis/cloud/distributed_readout.py` or the
distributed mode dies unpickling on the executors).

Cluster-covered (Spark + live cache); the pure helpers (build_parser,
bundle_key_from_manifest, reconstruct_model) are unit-tested and the scoring body
(run_readout) is covered against the driver path on a local-Spark fixture in
tests/scripts/test_readout_integration.py.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from _driver_common import _phase, configure_logging, make_spark_session
from gated_pc_cloud import (
    _DRIVER_READOUT_MAX_C, _collect_head_proba, _collect_lean_proba,
    _collect_theta_labels, _dump_partial_results, distributed_score_arm,
    format_arm_readout, readout_ab_report, readout_from_proba, resolve_readout_mode,
    score_arm,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Post-hoc case-finding readout on a finished gated_pc run "
                    "(no re-fit): pc_topics_lr + precision@recall / recall@FDR / "
                    "detection for the gated_pc arm.",
        epilog="Submit via `make gated-pc-readout ID=N` (analysis/cloud/Makefile). A "
               "hand-rolled spark-submit MUST pass --py-files "
               "<repo>/analysis/cloud/distributed_readout.py alongside the source "
               "zips: --readout-mode distributed ships its partition kernels by "
               "module reference and they must import on the executors.")
    p.add_argument("--run-dir", required=True,
                   help="Run dir with gated_pc_result.npz + manifest.json.")
    p.add_argument("--cache-uri", default=None,
                   help="Bundle cache root the fit used (its --cache-uri). Required "
                        "unless --bundle-path is given.")
    p.add_argument("--bundle-path", default=None,
                   help="Exact cached bundle dir ({cache_uri}/{key}); bypasses the "
                        "key recompute if the manifest is incomplete.")
    p.add_argument("--doc-min-length", type=int, default=None,
                   help="Override doc_min_length for the cache key (older manifests "
                        "did not record it; the current driver does).")
    p.add_argument("--recall-targets", default="0.5,0.8,0.9")
    p.add_argument("--fdr-targets", default="0.1,0.25,0.5")
    p.add_argument("--min-label-count", type=int, default=None,
                   help="Small-cell mask floor; default = the run's own value.")
    p.add_argument("--readout-mode", choices=["driver", "distributed", "auto"],
                   default="auto",
                   help="where the pc_topics_lr readout is fit, same semantics as "
                        "gated_pc_cloud's flag (ADR 0046): 'driver' collects per-doc "
                        "theta + dense label/mask for BOTH splits and fits C sklearn "
                        "LRs on the driver; 'distributed' fits all C heads in ONE "
                        "batched L-BFGS on the executors and collects only the lean "
                        "float32/uint8 test-split eval bundle; 'auto' (default) = "
                        f"driver at C<={_DRIVER_READOUT_MAX_C}, else distributed. "
                        "Note the run's OWN --readout-mode is not consulted: the "
                        "collect that has to fit is this driver's, not the fit's.")
    p.add_argument("--readout-ab-check", action="store_true",
                   help="run BOTH readout paths on the same re-transformed frames and "
                        "print the deltas (macro AUC/AP, per-node |dAUC|, sampled max "
                        "|dproba|). Report only — never asserts. Ignored unless the "
                        f"mode resolves to distributed AND C<={_DRIVER_READOUT_MAX_C} "
                        "(the driver path must still be affordable to compare to).")
    p.add_argument("--readout-theta-topm", type=int, default=None,
                   help="OVERRIDE the manifest's readout_theta_topm for this re-readout "
                        "(0 forces full-K). Default: the manifest's value, which "
                        "reproduces the fit's own design matrix. The override exists "
                        "for exactly one job: PRICING truncation — re-read a full-K "
                        "run (e.g. the 0103 cardiovascular record) with top-m forced "
                        "on and compare per-node AUC against its recorded numbers, so "
                        "the whole-Mondo enablement decision rests on a measured "
                        "delta, not the mass-coverage heuristic alone.")
    return p


def resolve_run_dir(pattern):
    """Resolve --run-dir to a single gated_pc run directory.

    An exact dir that holds gated_pc_result.npz is used as-is. Otherwise the value
    is treated as a glob (the Makefile passes `.../<id>-*` QUOTED so the shell does
    not pre-expand it) and filtered to matched dirs that contain gated_pc_result.npz
    — so a runs/ numbering COLLISION with a non-gated_pc experiment of the same id
    (e.g. a stale `0076-multidomain-...` from another branch) is discarded
    automatically. Requires exactly one gated_pc match; a clear error otherwise."""
    p = Path(pattern)
    if p.is_dir() and (p / "gated_pc_result.npz").exists():
        return p
    matches = [Path(m) for m in glob.glob(str(pattern))]
    hits = [m for m in matches if m.is_dir() and (m / "gated_pc_result.npz").exists()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SystemExit(
            f"[readout] no run dir with gated_pc_result.npz matches {pattern!r} "
            f"(matched: {[m.name for m in matches]}). Has the fit finished + saved? "
            f"Pass an exact --run-dir / GPR_RUN_DIR.")
    raise SystemExit(
        f"[readout] {len(hits)} gated_pc run dirs match {pattern!r}: "
        f"{[m.name for m in hits]}. Pass an exact --run-dir / GPR_RUN_DIR.")


def bundle_key_from_manifest(manifest: dict, *, doc_min_length=None):
    """Recompute the CaseFindingBundle cache key from a gated_pc manifest.

    Pulls the corpus-key fields the fit stored (top-level + corpus_manifest) and
    the gated_pc invariants (emit_labels=True, label_mask_mode). `doc_min_length`
    overrides the manifest value (older manifests omit it). Raises KeyError with a
    clear message if a required field is absent and no override is given."""
    from _case_finding_cache import compute_bundle_cache_key

    cm = manifest.get("corpus_manifest", {})
    dml = doc_min_length if doc_min_length is not None else cm.get("doc_min_length")
    if dml is None:
        raise KeyError(
            "doc_min_length is not in the manifest and no --doc-min-length was "
            "given; it is a cache-key input, so the bundle cannot be located. "
            "Pass --doc-min-length (the fit's value) or --bundle-path.")
    return compute_bundle_cache_key(
        source_table=cm["source_table"], person_mod=cm["person_mod"],
        vocab_size=cm["vocab_size"], min_df=cm["min_df"],
        min_patient_count=cm["min_patient_count"], doc_min_length=int(dml),
        prior_obs_days=cm["prior_obs_days"], window_days=cm["window_days"],
        disease=manifest["disease"], min_n=manifest["min_n"],
        holdout_frac=cm["holdout_frac"], n_bg=manifest["n_bg"], tpn=manifest["tpn"],
        cdr=cm.get("cdr"), strip_mode=manifest["strip_mode"],
        window_mode=manifest["window_mode"], lookback_days=manifest["lookback_days"],
        label_window_days=manifest["label_window_days"],
        emit_labels=True, label_mask_mode=manifest.get("label_mask_mode", "full"))


def reconstruct_model(run_dir: Path, manifest: dict):
    """Rebuild a scoreable OnlinePCLDAModel from the saved gated_pc_result.npz.

    The driver saves raw globals (lambda/alpha/w_CK), not a persisted model, so we
    wrap them in a VIResult and an OnlinePCLDAModel and set weightY>0 + numLabels so
    transform appends both topicDistribution (θ) and the head probability. The CAVI
    read-out knobs (gammaShape/caviMaxIter/caviTol) come from the model defaults,
    which match the fit's defaults for this experiment family."""
    from spark_vi.core.result import VIResult
    from spark_vi.mllib.topic.pc import OnlinePCLDAModel

    npz = np.load(run_dir / "gated_pc_result.npz")
    gp = {"lambda": npz["lambda"], "alpha": npz["alpha"], "w_CK": npz["w_CK"]}
    result = VIResult(global_params=gp, elbo_trace=[],
                      n_iterations=int(manifest.get("max_iter", 0)), converged=True)
    model = OnlinePCLDAModel(result)
    model._set(weightY=float(manifest.get("weight_y", 1.0)),
               numLabels=int(manifest["C"]), closureParents="")
    return model


def run_readout(train_scored, test_scored, manifest, *, recall_targets, fdr_targets,
                min_count, readout_mode="auto", ab_check=False, out_dir=None,
                theta_topm=None):
    """Score both gated_pc arms off two already-TRANSFORMED splits. No argparse.

    The whole body of this tool that is worth testing: given the frames a finished
    fit's model produced and that fit's manifest, it routes the pc_topics_lr arm
    through the driver or the distributed readout (ADR 0046), optionally runs the
    A/B equality report, adds the co-fit head arm, and returns
    `{"gated_pc": ..., "gated_pc_head": ...}`. Taking DataFrames rather than a
    SparkSession + run dir is what makes it callable against `score_arm` on a local
    Spark fixture (same reason `distributed_score_arm` is shaped that way);
    argparse, run-dir resolution and the bundle-cache reload stay cluster-covered.

    K comes from the manifest (`lay.K` at fit time) and is only needed by the
    distributed solver; runs old enough to lack it fall back to the width of the
    transform's own theta, which is the same number by construction.

    `out_dir` gets a `results_readout.json` after EACH arm lands — a re-readout is
    a recovery action (exp 0103 lost a 4h fit's readout to an empty summary), so
    its output has to survive the terminal it was printed to. Deliberately NOT
    results_partial.json: that file belongs to the fit's own record."""
    C = int(manifest["C"])
    mode = resolve_readout_mode(readout_mode, C)
    K = int(manifest.get("K") or 0)
    if mode == "distributed" and not K:
        # Only the batched solver needs K, so only it pays for the peek.
        K = int(train_scored.select("topicDistribution").head()[0].size)
    print(f"[readout]   readout_mode={readout_mode} -> {mode} (C={C}, "
          f"K={K or 'n/a'}, driver-collect ceiling C<={_DRIVER_READOUT_MAX_C})",
          flush=True)
    ab = bool(ab_check) and mode == "distributed"
    if bool(ab_check) and not ab:
        print("[readout]   readout_ab_check ignored (readout_mode resolved to driver "
              "— there is nothing to compare against)", flush=True)
    elif ab and C > _DRIVER_READOUT_MAX_C:
        print(f"[readout]   readout_ab_check SKIPPED: C={C} exceeds the driver path's "
              f"own ceiling ({_DRIVER_READOUT_MAX_C}); the gate is meant to run at "
              "cardiovascular scale (C=444)", flush=True)
        ab = False

    def _dump(results):
        if out_dir is not None:
            _dump_partial_results(Path(out_dir), results,
                                  name="results_readout.json")

    results = {}
    dist = None
    if mode == "distributed":
        # No theta collect: the per-node LRs are fit on the executors and only the
        # lean test-split eval bundle comes back. theta_topm defaults to the
        # MANIFEST's value: a re-readout reproduces the fit's own design matrix —
        # silently refitting a top-m run at full K would change the estimator
        # (and crawl at whole-Mondo, the scale where top-m is on). An explicit
        # override (--readout-theta-topm) exists for the truncation-PRICING
        # experiment: force top-m onto a recorded full-K run and read the delta.
        if theta_topm is None:
            theta_topm = int(manifest.get("readout_theta_topm", 0) or 0)
            if theta_topm:
                print(f"[readout]   theta top-m={theta_topm} (from manifest)",
                      flush=True)
        else:
            theta_topm = int(theta_topm)
            print(f"[readout]   theta top-m={theta_topm} (CLI OVERRIDE — deltas vs "
                  "the recorded run price the truncation)", flush=True)
        dist = distributed_score_arm(
            train_scored, test_scored, C, K, recall_targets=recall_targets,
            fdr_targets=fdr_targets, min_count=min_count, label="gated_pc",
            theta_topm=theta_topm)
        results["gated_pc"] = dist[0]
    else:
        Pi_tr, y_tr, m_tr, _ = _collect_theta_labels(train_scored, C)
        Pi_te, y_te, m_te, _ = _collect_theta_labels(test_scored, C)
        results["gated_pc"] = score_arm(
            Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te, C, recall_targets=recall_targets,
            fdr_targets=fdr_targets, min_count=min_count)
    _dump(results)
    print(format_arm_readout("gated_pc (pc_topics_lr)", results["gated_pc"]),
          flush=True)
    if ab:
        # Reuses the distributed result just computed, so the gate costs one extra
        # (driver-path) readout, not two.
        readout_ab_report(train_scored, test_scored, C, K,
                          recall_targets=recall_targets, fdr_targets=fdr_targets,
                          min_count=min_count, label="gated_pc", distributed=dist)

    # Co-fit head arm. The scaled-back mainline (unsup gate + post-hoc readout) fits
    # at weightY=0, whose transform appends NO `probability` column — the head does
    # not exist, and asking for it used to fail the whole re-readout with an
    # AnalysisException AFTER the expensive arm above had already been printed. Skip
    # it on either witness (the manifest's weight_y, or the column itself, which also
    # covers a manifest too old to record weight_y).
    weight_y = float(manifest.get("weight_y", 1.0))
    if weight_y == 0.0 or "probability" not in test_scored.columns:
        print(f"[readout]   co-fit head arm SKIPPED (weight_y={weight_y:g}, "
              f"probability column "
              f"{'present' if 'probability' in test_scored.columns else 'absent'}): "
              "an unsupervised gate has no co-fit head to read out.", flush=True)
        return results
    if mode == "distributed":
        # No LR involved — `probability` IS the per-doc (C,) P(node) — so the
        # distributed variant is just the lean collector over that column.
        hp, hy, hm, _ = _collect_lean_proba(test_scored, C, score_col="probability")
    else:
        hp, hy, hm = _collect_head_proba(test_scored, C)
    results["gated_pc_head"] = readout_from_proba(
        hp, hy, hm, C, recall_targets=recall_targets, fdr_targets=fdr_targets,
        min_count=min_count)
    _dump(results)
    print(format_arm_readout("gated_pc (co-fit head)", results["gated_pc_head"]),
          flush=True)
    return results


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    run_dir = resolve_run_dir(args.run_dir)
    print(f"[readout]   run dir: {run_dir}", flush=True)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    C = int(manifest["C"])
    rt = [float(x) for x in args.recall_targets.split(",") if x]
    ft = [float(x) for x in args.fdr_targets.split(",") if x]
    min_count = (args.min_label_count if args.min_label_count is not None
                 else int(manifest.get("min_label_count", 20)))

    with make_spark_session(app_name="gated-pc-readout") as spark:
        from _case_finding_cache import try_load

        with _phase("reload cached bundle"):
            if args.bundle_path:
                base = args.bundle_path.rstrip("/")
                cache_uri, key = base.rsplit("/", 1)
            else:
                if not args.cache_uri:
                    print("[readout] ERROR: pass --cache-uri or --bundle-path.",
                          flush=True)
                    return 1
                cache_uri = args.cache_uri
                key = bundle_key_from_manifest(
                    manifest, doc_min_length=args.doc_min_length)
            bundle = try_load(spark, cache_uri, key)
            if bundle is None:
                print(f"[readout] ERROR: bundle cache MISS at {cache_uri}/{key}. "
                      "The assembly source may have changed since the fit, or a "
                      "key field differs. Pass --bundle-path at the exact cached "
                      "dir, or --doc-min-length if it was omitted.", flush=True)
                return 2
            print(f"[readout]   bundle loaded ({cache_uri}/{key}); C={C}", flush=True)

        with _phase("reconstruct model + transform"):
            # The theta collect (driver mode) now happens inside run_readout, so the
            # distributed mode can skip it entirely rather than paying for it here.
            model = reconstruct_model(run_dir, manifest)
            train_scored = model.transform(bundle.train_df).cache()
            test_scored = model.transform(bundle.test_df).cache()

        with _phase("score"):
            run_readout(train_scored, test_scored, manifest, recall_targets=rt,
                        fdr_targets=ft, min_count=min_count,
                        readout_mode=args.readout_mode,
                        ab_check=args.readout_ab_check, out_dir=run_dir,
                        theta_topm=args.readout_theta_topm)
            print(f"[readout]   arm results written to "
                  f"{run_dir / 'results_readout.json'}", flush=True)
            train_scored.unpersist(); test_scored.unpersist()

            # Echo the other arms' stored summary (only gated_pc can be re-scored).
            # `results` is legitimately absent-or-null on a FIT-ONLY manifest:
            # `gated_pc_cloud` writes the npz + manifest as soon as the fit lands
            # (`partial="fit-only"`) so a readout death cannot cost the fit, which
            # is precisely the run this tool exists to rescue — so say what is
            # missing rather than echoing nothing, and never index into it.
            if manifest.get("partial"):
                print(f"[readout]   manifest marked partial="
                      f"{manifest['partial']!r}: the fit landed but its own "
                      "readout did not, so there are no stored arm results to "
                      "echo (the numbers above are this re-readout's).",
                      flush=True)
            for name, res in (manifest.get("results") or {}).items():
                if name in ("gated_pc", "gated_pc_head"):
                    continue
                rk = res.get("ranking", res)   # tolerate old flat-macro manifests
                auc = rk.get("auc"); ap = rk.get("ap")
                print(f"[readout]   (manifest) {name}: AUC="
                      f"{'n/a' if auc is None else f'{auc:.4f}'} "
                      f"AP={'n/a' if ap is None else f'{ap:.4f}'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

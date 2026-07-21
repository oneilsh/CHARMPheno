"""Standalone post-hoc likelihood-ratio readout on an EXISTING dag_placement run
(no re-fit). The "fork-settler": loads a completed run's saved lambda (the
learned node-block word counts) + manifest.json, reloads the cached
CaseFindingBundle the run was fit against, and sweeps `lr_auc_sweep` (Task 1) over
an alpha grid -- printing LR-AUC(alpha) beside the run's own theta-mass
detection AUC (manifest["metrics"]["detection"]["auc"]).

LR-AUC >> theta-AUC => the disease signal is present in the corpus but was
buried by theta-mass competing on the simplex (theta-mass was the wrong lens).
LR-AUC ~= theta-AUC => the signal is genuinely weak/absent for this run; no
readout trick recovers it. See the module-level real-data note in the task
brief (docs/superpowers/sdd/task-2-brief.md) for two curve-shape caveats:
(1) alpha=0 is EXPECTED to score low (the gate under-represents common codes,
so an unshrunk log-ratio over-penalises them) and rise with alpha -- that shape
is informative, not a bug; (2) a near-flat/junk-topic node can win the
max-over-nodes at score ~0 for everyone, masking real signal at low LR-AUC.

Fragility (no re-fit protection): the bundle is located by RECOMPUTING the
assembly cache key from the run's manifest.json, not by a stored key -- if the
assembly/DAG source has changed since the fit, or a manifest field needed by
the key (e.g. --doc-min-length, which the fit driver does not currently record)
doesn't match the original run, the recomputed key MISSES the cache and this
driver cannot proceed without --bundle-path pointing at the exact cached
bundle directory.

Cluster-covered, not unit-tested: the Spark session, bundle load, and BOW-matrix
construction (they need a live corpus + cache to exercise meaningfully). The
pure helpers (`build_parser`, `parse_alpha_grid`, `resolve_alpha_grid`) are
unit-tested in analysis/cloud/tests/test_lr_readout.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse as sp

from _driver_common import make_spark_session


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Post-hoc LR-AUC(alpha) sweep on an existing dag_placement "
                    "run (no re-fit): the fork-settler vs the run's theta-mass "
                    "detection AUC.")
    p.add_argument("--run-dir", required=True,
                   help="Run directory containing dag_placement_result.npz + "
                        "manifest.json (e.g. .../runs/0061-diabetes-...).")
    p.add_argument("--alpha-grid", default="0,0.1,1,10",
                   help="Comma-separated alpha MULTIPLIERS of the run's median "
                        "node Sigma-lambda (see resolve_alpha_grid); 0.0 is "
                        "always included even if omitted (parse_alpha_grid), "
                        "and is kept absolute rather than scaled (0x median "
                        "would still be 0). Default '0,0.1,1,10' reproduces "
                        "the design-doc default grid [0, 0.1, 1, 10] x "
                        "median(node Sigma-lambda).")
    p.add_argument("--bundle-path", default=None,
                   help="Override: load the CaseFindingBundle cache directly "
                        "from this {cache_uri}/{key} directory instead of "
                        "recomputing the key from the manifest (use this when "
                        "the recomputed key misses -- see the module "
                        "docstring's fragility note).")
    p.add_argument("--cache-uri", default=None,
                   help="Bundle cache root used to recompute the cache key "
                        "when --bundle-path is not given. NOT recorded in "
                        "manifest.json (the fit driver's --cache-uri is not "
                        "persisted), so it must be supplied here matching the "
                        "original fit's --cache-uri.")
    p.add_argument("--doc-min-length", type=int, default=0,
                   help="The original fit's --doc-min-length. NOT recorded in "
                        "manifest.json (a genuine gap in dag_placement_cloud.py's "
                        "saved manifest); defaults to the fit driver's own CLI "
                        "default (0). Override if the original run used a "
                        "non-default value, or the recomputed cache key will "
                        "miss.")
    p.add_argument("--count-mode", choices=["raw", "log1p"], default="raw",
                   help="Passed through to lr_placement_scores/lr_auc_sweep.")
    p.add_argument("--length-normalize", action="store_true",
                   help="Passed through to lr_placement_scores/lr_auc_sweep.")
    p.add_argument("--sample-cases", type=int, default=0,
                   help="(Task 3) itemize an lr_decompose breakdown for this "
                        "many random held-out foreground cases. Not wired in "
                        "this driver yet.")
    p.add_argument("--person", type=int, default=None,
                   help="(Task 3) itemize an lr_decompose breakdown for this "
                        "single person_id instead of a random sample. Not "
                        "wired in this driver yet.")
    return p


def parse_alpha_grid(s: str) -> list[float]:
    """Comma-separated alpha-grid spec -> sorted list of float multipliers,
    ALWAYS containing 0.0 (the unshrunk alpha=0 baseline is always evaluated,
    even if the caller's grid omits it -- see the real-data note on why alpha=0
    is diagnostic, not skippable). Pure string->floats parsing; the multiplier
    -> actual-alpha scaling (x median node Sigma-lambda) is `resolve_alpha_grid`,
    which needs the loaded lambda and so cannot run at arg-parse time."""
    toks = [t.strip() for t in s.split(",") if t.strip() != ""]
    vals = {float(t) for t in toks}
    vals.add(0.0)
    return sorted(vals)


def resolve_alpha_grid(multipliers, lam, lay) -> list[float]:
    """Multiplier list -> actual alpha values: each nonzero multiplier scales
    median(node Sigma-lambda) (the Empirical-Bayes shrinkage target's natural
    unit -- see `_lr_logratio_rows`); 0.0 is kept absolute (0x median is still
    just 0, and 0 is the meaningful "no shrinkage" baseline, not a scaled
    quantity)."""
    lam = np.asarray(lam, dtype=float)
    node_sums = np.array([lam[lay.block[u]].sum() for u in lay.nodes], dtype=float)
    med = float(np.median(node_sums)) if len(node_sums) else 0.0
    return sorted({0.0 if m == 0.0 else m * med for m in multipliers})


def load_run(run_dir):
    """(lam, alpha_dirichlet, manifest) from a saved dag_placement run
    directory. `lam` = the learned topic-word counts (gp["lambda"], [K x V]);
    `alpha_dirichlet` = the fit's variational Dirichlet alpha (gp["alpha"],
    unrelated to the LR-shrinkage `alpha` this driver sweeps -- loaded for
    completeness/future use, not consumed by the LR readout itself)."""
    run_dir = Path(run_dir)
    npz = np.load(run_dir / "dag_placement_result.npz")
    lam = npz["lambda"]
    alpha_dirichlet = npz["alpha"]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return lam, alpha_dirichlet, manifest


def _corpus_cfg_from_manifest(manifest, *, doc_min_length=0):
    """The compute_bundle_cache_key kwargs recoverable from a saved
    manifest.json (see dag_placement_cloud.py's manifest dict). `doc_min_length`
    is NOT recorded there -- see the module docstring's fragility note -- so it
    is threaded through from the CLI (default = the fit driver's own default)."""
    cm = manifest["corpus_manifest"]
    return dict(
        source_table=cm["source_table"], person_mod=cm["person_mod"],
        vocab_size=cm["vocab_size"], min_df=cm["min_df"],
        min_patient_count=cm["min_patient_count"], doc_min_length=doc_min_length,
        prior_obs_days=cm["prior_obs_days"], window_days=cm["window_days"],
        disease=manifest["disease"], min_n=manifest["min_n"],
        holdout_frac=cm["holdout_frac"], n_bg=manifest["n_bg"], tpn=manifest["tpn"],
        cdr=cm["cdr"], strip_mode=manifest["strip_mode"],
        window_mode=manifest["window_mode"], lookback_days=manifest["lookback_days"],
        label_window_days=manifest["label_window_days"])


def locate_bundle(spark, manifest, *, bundle_path=None, cache_uri=None,
                  doc_min_length=0):
    """The CaseFindingBundle the run was fit against, or None (with a printed
    WARNING) if it cannot be located. `bundle_path` (if given) is the exact
    {cache_uri}/{key} directory, loaded directly via try_load(parent, name) --
    the same contract try_load/save use internally (base =
    f"{cache_uri}/{key}"). Otherwise the key is RECOMPUTED from the manifest
    (compute_bundle_cache_key) and looked up under `cache_uri`; a miss there
    means either the assembly/DAG source has drifted since the fit (no re-fit
    protection -- this driver cannot tell drift apart from a stale
    --doc-min-length) or `cache_uri`/`doc_min_length` don't match the original
    fit, and the caller is told to pass --bundle-path directly."""
    from _case_finding_cache import compute_bundle_cache_key, try_load

    if bundle_path:
        p = Path(bundle_path)
        bundle = try_load(spark, str(p.parent), p.name)
        if bundle is None:
            print(f"[lr_readout] WARNING: --bundle-path {bundle_path} did not "
                  "load (missing train.parquet/test.parquet/meta under that "
                  "directory).", flush=True)
        return bundle

    if not cache_uri:
        print("[lr_readout] WARNING: no --cache-uri or --bundle-path given; "
              "cannot locate the cached CaseFindingBundle. Pass --bundle-path "
              "(exact cache dir) or --cache-uri (matching the original fit's "
              "--cache-uri) to proceed.", flush=True)
        return None

    cfg = _corpus_cfg_from_manifest(manifest, doc_min_length=doc_min_length)
    key = compute_bundle_cache_key(**cfg)
    bundle = try_load(spark, cache_uri, key)
    if bundle is None:
        print(f"[lr_readout] WARNING: bundle cache MISS for recomputed key "
              f"{key!r} under {cache_uri!r}. This can mean (a) the assembly/DAG "
              "source changed since the fit (no re-fit protection: the "
              "recomputed key folds a content-hash of condition_dag + "
              "case_finding_assembly, so ANY edit since the fit invalidates "
              "it), or (b) --doc-min-length does not match the original run "
              "(not recorded in manifest.json). Pass --bundle-path to load the "
              "exact cached bundle directly.", flush=True)
    return bundle


def build_test_bow(bundle, vocab_size):
    """[n_docs x vocab_size] scipy.sparse CSR bag-of-words + boolean is_fg
    (True iff the doc's frontier is nonempty), read from
    bundle.test_df["features","frontier"] (features = a SparseVector per
    CaseFindingBundle's documented schema). Cluster-only: collects the held-out
    split to the driver (the same held-out scale dag_placement_cloud.py already
    collects for its inline eval), not unit-tested."""
    rows = bundle.test_df.select("features", "frontier").collect()
    n = len(rows)
    indptr = np.zeros(n + 1, dtype=np.int64)
    idx_chunks, data_chunks = [], []
    is_fg = np.zeros(n, dtype=bool)
    for i, r in enumerate(rows):
        sv = r["features"]
        idx_chunks.append(np.asarray(sv.indices, dtype=np.int64))
        data_chunks.append(np.asarray(sv.values, dtype=np.float64))
        indptr[i + 1] = indptr[i] + len(sv.indices)
        is_fg[i] = len(r["frontier"]) > 0
    indices = np.concatenate(idx_chunks) if idx_chunks else np.array([], dtype=np.int64)
    data = np.concatenate(data_chunks) if data_chunks else np.array([], dtype=np.float64)
    bow = sp.csr_matrix((data, indices, indptr), shape=(n, vocab_size))
    return bow, is_fg


def print_readout(multipliers, alpha_values, lr_aucs, theta_auc, *, gap_tol=0.05):
    """Table: alpha (+ its multiplier), LR-AUC, the gap vs theta_auc, and a
    verdict label. gap_tol brackets "approximately equal" (calibrated to the
    ROC-AUC's own noise floor at typical held-out sizes, not a tuned
    threshold -- move it if a run's CI is visibly wider)."""
    print(f"[lr_readout] theta-mass detection AUC (from the run's manifest): "
          f"{theta_auc:.4f}", flush=True)
    print("[lr_readout] LR-AUC(alpha) sweep:", flush=True)
    for m, a in zip(multipliers, alpha_values):
        auc = lr_aucs[a]
        gap = auc - theta_auc if np.isfinite(theta_auc) else float("nan")
        if not np.isfinite(gap):
            verdict = "n/a (theta_auc is nan)"
        elif gap > gap_tol:
            verdict = "LR >> theta: signal present but buried by theta-mass"
        elif gap < -gap_tol:
            verdict = "LR << theta: unexpected, theta-mass wins here"
        else:
            verdict = "LR ~= theta: signal likely genuinely weak/absent"
        print(f"[lr_readout]   alpha={a:12.4g}  (x{m:g} median Sigma-lambda)  "
              f"LR-AUC={auc:.4f}  gap={gap:+.4f}   {verdict}", flush=True)


def main() -> int:
    from spark_vi.models.topic.dag_placement import DagLayout, lr_auc_sweep

    args = build_parser().parse_args()
    lam, _alpha_dirichlet, manifest = load_run(args.run_dir)
    multipliers = parse_alpha_grid(args.alpha_grid)
    theta_auc = manifest["metrics"]["detection"]["auc"]

    with make_spark_session(app_name="lr-readout") as spark:
        bundle = locate_bundle(
            spark, manifest, bundle_path=args.bundle_path, cache_uri=args.cache_uri,
            doc_min_length=args.doc_min_length)
        if bundle is None:
            print("[lr_readout] ERROR: could not load the cached "
                  "CaseFindingBundle; aborting.", flush=True)
            return 1

        lay = DagLayout(bundle.parent_int, n_bg=manifest["n_bg"], tpn=manifest["tpn"])
        vocab_size = len(bundle.vocab_map)
        bow, is_fg = build_test_bow(bundle, vocab_size)
        print(f"[lr_readout]   held-out test set: {bow.shape[0]} docs, "
              f"{int(is_fg.sum())} foreground, V={vocab_size}, "
              f"K={lay.K} ({lay.n_bg} bg + {len(lay.nodes)} nodes x {lay.tpn} tpn)",
              flush=True)

        alpha_values = resolve_alpha_grid(multipliers, lam, lay)
        lr_aucs = lr_auc_sweep(
            bow, lam, lay, is_fg, alpha_grid=alpha_values,
            count_mode=args.count_mode, length_normalize=args.length_normalize)
        print_readout(multipliers, alpha_values, lr_aucs, theta_auc)

        if args.sample_cases or args.person is not None:
            print("[lr_readout]   --sample-cases/--person itemized breakdown: "
                  "not wired in this driver (Task 3).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

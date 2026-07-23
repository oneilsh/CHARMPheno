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
import hashlib
import json
import math
import os
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
                   help="Per-code weighting in the LR score, applied CONSISTENTLY "
                        "across the AUC sweep, the detection report, the error-class "
                        "classification, and the decompose viewer. 'raw' = weight by "
                        "code count (a code seen 4x contributes 4x); 'log1p' = weight "
                        "by log1p(count) (dampens repeated codes -- both repeated "
                        "positive matches and repeated generic-comorbidity negatives).")
    p.add_argument("--length-normalize", action="store_true",
                   help="Divide each node's LR score by the doc's token count "
                        "(reduces the penalty a comorbidity-heavy patient pays for "
                        "simply having more codes). Applied to the sweep + detection "
                        "+ classification scores.")
    p.add_argument("--sample-cases", type=int, default=0,
                   help="Itemize an lr_decompose breakdown (per true frontier "
                        "node, rendered with concept names) for this many "
                        "random held-out foreground cases, written to "
                        "{run_dir}/lr_readout/decompose.txt (in-enclave; "
                        "never printed to stdout).")
    p.add_argument("--sample-background", type=int, default=0,
                   help="Also itemize this many random BACKGROUND (empty-"
                        "frontier) held-out docs, each decomposed against the "
                        "node it scores HIGHEST on (the contrast: why did a "
                        "non-case most resemble a disease?). Written to the "
                        "same decompose.txt.")
    p.add_argument("--person", type=int, default=None,
                   help="Itemize an lr_decompose breakdown for this single "
                        "person_id instead of a random sample (also written "
                        "to {run_dir}/lr_readout/decompose.txt).")
    p.add_argument("--viewer-top-nodes", type=int, default=8,
                   help="How many top-ranked DAG nodes to list per case in the "
                        "decompose viewer (the 'what else did this patient match, "
                        "and was the call right' ranking). Default 8.")
    p.add_argument("--viewer-per-class", type=int, default=0,
                   help="Error-class viewer: sample up to this many held-out "
                        "patients from EACH background-vs-rare confusion class "
                        "(false-positive: background called rare; false-negative: "
                        "rare called background; node-confusion; correct; true-"
                        "negative), grouped in decompose.txt. 0 = off (use "
                        "--sample-cases/--sample-background instead).")
    p.add_argument("--viewer-call-sensitivity", type=float, default=0.80,
                   help="The detection operating point (target foreground "
                        "sensitivity) whose max-node-LR score threshold defines "
                        "'called rare' vs 'called background' for --viewer-per-class. "
                        "Default 0.80.")
    p.add_argument("--viewer-score-mode", choices=["lr", "explain_away"], default="lr",
                   help="Score the case viewer's per-doc ranking/classification with "
                        "either plain 'lr' (lr_placement_scores/lr_decompose, default, "
                        "unchanged behavior) or 'explain_away' (the responsibility-"
                        "weighted explain_away_placement_scores/explain_away_decompose, "
                        "at the same viewer alpha -- comorbid codes routed away from a "
                        "node are suppressed toward 0 instead of docking its score; the "
                        "plain-LR max score is also printed on each case's header line "
                        "for contrast). Independent of the unconditional plain-LR + "
                        "explain-away @alpha=inf detection block printed by "
                        "detection_report, which always prints both.")
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
    quantity), and inf is kept absolute (the parameter-free α->∞ limit -- a
    multiplier of median Sigma-lambda would be meaningless, and inf*0 is nan)."""
    lam = np.asarray(lam, dtype=float)
    node_sums = np.array([lam[lay.block[u]].sum() for u in lay.nodes], dtype=float)
    med = float(np.median(node_sums)) if len(node_sums) else 0.0

    def _scale(m):
        if m == 0.0 or math.isinf(m):
            return m                       # 0 and inf are absolute, not scaled
        return m * med
    return sorted({_scale(m) for m in multipliers})


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
        # Split with string ops, NOT pathlib: Path("gs://bucket/.../key").parent
        # collapses the scheme's "//" to "/" (gs:/bucket/...), losing the URI
        # authority and misdirecting try_load. rpartition keeps gs://... intact.
        parent, _, key = bundle_path.rstrip("/").rpartition("/")
        bundle = try_load(spark, parent, key)
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


def build_test_bow(bundle, vocab_size, lay):
    """([n_docs x vocab_size] scipy.sparse CSR bag-of-words, boolean is_fg, meta),
    read from bundle.test_df["person_id","features","frontier"] (features = a
    SparseVector per CaseFindingBundle's documented schema). `meta` is a list of
    (person_id, features SparseVector, frontier engine-id list) ALIGNED with the
    bow rows (one collect, one order) so the case viewer can classify/decompose
    the same docs the AUC is scored over. Cluster-only: collects the held-out
    split to the driver (the same held-out scale dag_placement_cloud.py already
    collects for its inline eval), not unit-tested.

    is_fg MUST match `evaluate`'s detection definition exactly: foreground iff the
    frontier intersects the SCOREABLE nodes `lay.nodes` (which exclude root 0), not
    merely "frontier nonempty". A root-only / out-of-layout frontier (e.g. {0} for
    a patient coded only with a single-anchor disease's anchor) is BACKGROUND in
    the manifest's theta-mass AUC; counting it foreground here would make LR-AUC and
    theta-AUC use different positive sets and confound the fork-settler."""
    scoreable = set(lay.nodes)
    rows = bundle.test_df.select("person_id", "features", "frontier").collect()
    n = len(rows)
    indptr = np.zeros(n + 1, dtype=np.int64)
    idx_chunks, data_chunks = [], []
    is_fg = np.zeros(n, dtype=bool)
    meta = []
    for i, r in enumerate(rows):
        sv = r["features"]
        idx_chunks.append(np.asarray(sv.indices, dtype=np.int64))
        data_chunks.append(np.asarray(sv.values, dtype=np.float64))
        indptr[i + 1] = indptr[i] + len(sv.indices)
        frontier = [int(x) for x in r["frontier"]]
        is_fg[i] = bool(set(frontier) & scoreable)
        meta.append((r["person_id"], sv, frontier))
    indices = np.concatenate(idx_chunks) if idx_chunks else np.array([], dtype=np.int64)
    data = np.concatenate(data_chunks) if data_chunks else np.array([], dtype=np.float64)
    bow = sp.csr_matrix((data, indices, indptr), shape=(n, vocab_size))
    return bow, is_fg, meta


def print_readout(multipliers, alpha_values, lr_aucs, theta_auc, *, gap_tol=0.05):
    """Table: alpha (+ its multiplier), LR-AUC, the gap vs theta_auc, and a
    verdict label. gap_tol brackets "approximately equal" -- a coarse readout
    default (0.05 AUC), NOT a calibrated threshold; widen it if a run's AUC
    bootstrap CI is visibly wider than that."""
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


def _hash_person(person_id) -> str:
    """SHA-256-truncated person id for row-level stdout/log lines (this repo
    hashes ids in row-level log output; aggregate outputs print freely). The
    written decompose.txt itself keeps the raw id -- it stays in the run dir,
    in-enclave, and is never printed."""
    return hashlib.sha256(str(person_id).encode("utf-8")).hexdigest()[:12]


def render_decompose_rows(rows, idx_to_cid, name_by_cid) -> list[str]:
    """[(w, count, contribution), ...] (Task 1's lr_decompose output, already
    sorted by |contribution| desc) OR [(w, count, r_u_w, contribution), ...]
    (explain_away_decompose's form, with the routing weight r(u|w)) -> rendered
    lines with concept NAMES in place of raw vocab indices. `idx_to_cid` =
    vocab-idx -> concept-id (the inverse of CaseFindingBundle.vocab_map, which
    is {concept_id: vocab_idx}); `name_by_cid` = concept-id -> concept name for
    the FULL vocabulary (from _vocab_concept_names' BigQuery lookup -- NOT
    bundle.name_by_id, which only covers the ~DAG-node concepts). Falls back to
    the concept id, then the raw vocab index, if a lookup misses. Pure string
    formatting; order preserved."""
    lines = []
    for r in rows:
        if len(r) == 4:
            w, count, r_uw, contribution = r
            cid = idx_to_cid.get(w, w)
            name = name_by_cid.get(cid, cid)
            lines.append(f"{contribution:+.1f}  x{count:g}  r={r_uw:.2f}  {name}")
        else:
            w, count, contribution = r
            cid = idx_to_cid.get(w, w)
            name = name_by_cid.get(cid, cid)
            lines.append(f"{contribution:+.1f}  x{count:g}  {name}")
    return lines


def select_cases(bundle, *, sample_cases=0, sample_background=0, person=None, seed=0):
    """Held-out test docs to itemize, as `(person_id, SparseVector features,
    frontier engine-ids)` tuples: the single `person` doc if given, else up to
    `sample_cases` random FOREGROUND docs (nonempty frontier) plus up to
    `sample_background` random BACKGROUND docs (empty frontier). A background
    doc's empty frontier signals the viewer to decompose it against its
    top-scoring node. Cluster-only: collects a small sample, not unit-tested."""
    from pyspark.sql import functions as F

    df = bundle.test_df
    if person is not None:
        rows = (df.filter(F.col("person_id") == person)
                  .select("person_id", "features", "frontier").collect())
        return [(r["person_id"], r["features"], list(r["frontier"])) for r in rows]

    def _take(sub, n, salt):
        rows = (sub.orderBy(F.rand(seed + salt)).limit(n)
                   .select("person_id", "features", "frontier").collect())
        return [(r["person_id"], r["features"], list(r["frontier"])) for r in rows]

    out = []
    if sample_cases:
        out += _take(df.filter(F.size(F.col("frontier")) > 0), sample_cases, 0)
    if sample_background:
        out += _take(df.filter(F.size(F.col("frontier")) == 0), sample_background, 1)
    return out


def _ranking_summary_lines(scores, nodes, true_set, node_name, top_nodes):
    """Pure: given per-node LR `scores` (aligned with `nodes`), the patient's TRUE
    frontier node set, and `node_name`, return (lines, top_node) for the viewer.
    The highest-scoring node is the model's CALL; HIT iff it is in `true_set`. For
    a foreground case also reports the true node(s) and their rank so a MISS is
    debuggable ('what did the patient match instead, and where did the truth land').
    Background cases (empty true_set) get the ranking only."""
    n = len(nodes)
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)   # score desc
    score_of = {int(nodes[i]): float(scores[i]) for i in range(n)}
    rank_of = {int(nodes[order[r]]): r + 1 for r in range(n)}
    top_node = int(nodes[order[0]])
    lines = []
    if true_set:
        hit = top_node in true_set
        best_rank = min(rank_of[u] for u in true_set)
        tn = ", ".join(f"{node_name.get(u, u)} (id {u}, rank {rank_of[u]}/{n}, "
                       f"score {score_of[u]:+.2f})" for u in true_set)
        lines.append(f"  TRUE frontier: {tn}")
        lines.append(f"  CALL: {'HIT' if hit else 'MISS'} "
                     f"(top = {node_name.get(top_node, top_node)}, "
                     f"true best rank = {best_rank}/{n})")
    k = min(top_nodes, n)
    lines.append(f"  LR ranking (top {k} of {n}):")
    for r in range(k):
        u = int(nodes[order[r]])
        mark = "  <- TRUE" if u in true_set else ("  <- TOP" if r == 0 else "")
        lines.append(f"    {r + 1:>2}. {str(node_name.get(u, u))[:36]:<36} "
                     f"score={score_of[u]:+.3f}{mark}")
    return lines, top_node


def _classify_error_class(is_fg, called_rare, hit):
    """The background-vs-rare confusion class for one held-out patient. `called_rare`
    = the max-node LR case score cleared the detection threshold; `hit` = the
    top-ranked node is (one of) the patient's true frontier node(s)."""
    if is_fg and called_rare:
        return "rare_called_rare_correct" if hit else "rare_called_rare_wrong_disease"
    if is_fg and not called_rare:
        return "rare_called_background"           # false negative
    if (not is_fg) and called_rare:
        return "background_called_rare"           # false positive
    return "background_called_background"          # true negative


# Display order + labels; false-positive (background->rare) first since it is the
# error class that most concerns deployment, then false-negative, then the
# less-bad node-confusion, then the correct calls.
_CLASS_ORDER = [
    ("background_called_rare",
     "FALSE POSITIVE  -  background patient called RARE"),
    ("rare_called_background",
     "FALSE NEGATIVE  -  rare-disease patient called BACKGROUND"),
    ("rare_called_rare_wrong_disease",
     "rare patient, called RARE but WRONG disease (node confusion)"),
    ("rare_called_rare_correct",
     "rare patient, called RARE and CORRECT disease"),
    ("background_called_background",
     "true negative  -  background called background"),
]


def _render_case(person_id, sv, frontier, *, lam, lay, alpha, background,
                 node_name, idx_to_cid, vocab_names, nodes, vocab_size, top_nodes,
                 count_mode="raw", length_normalize=False, score_mode="lr"):
    """One patient's viewer block (string): token/code counts, the LR ranking +
    HIT/MISS summary, and the per-code decompose for the top node (+ each true
    node not the top). Pure per-doc; scores this doc against all nodes under the
    same `count_mode`/`length_normalize` as the AUC sweep (so a log1p run's viewer
    matches its score). `score_mode` 'lr' (default, unchanged) uses
    lr_placement_scores/lr_decompose; 'explain_away' uses the responsibility-weighted
    explain_away_placement_scores/explain_away_decompose instead (whose decompose rows
    carry the routing weight r(u|w), rendered via render_decompose_rows' 4-tuple form)
    and additionally prints the plain-LR max score on the header line for contrast."""
    from spark_vi.models.topic.dag_placement import lr_decompose, lr_placement_scores
    bow_row = np.zeros(vocab_size, dtype=float)
    bow_row[np.asarray(sv.indices, dtype=np.int64)] = np.asarray(sv.values, dtype=float)
    lr_scores = lr_placement_scores(bow_row[None], lam, lay, alpha=alpha,
                                    background=background, count_mode=count_mode,
                                    length_normalize=length_normalize)[0]
    contrast = ""
    if score_mode == "explain_away":
        from spark_vi.models.topic.dag_placement import (
            explain_away_decompose, explain_away_placement_scores)
        scores = explain_away_placement_scores(
            bow_row[None], lam, lay, alpha=alpha, background=background,
            count_mode=count_mode, length_normalize=length_normalize)[0]
        contrast = f"  [plain-LR max score={lr_scores.max():+.3f} for contrast]"
    else:
        scores = lr_scores
    true_set = [u for u in frontier if u in lay.block]
    kind = "foreground" if true_set else "background"
    rank_lines, top_node = _ranking_summary_lines(scores, nodes, true_set,
                                                  node_name, top_nodes)

    def _decomp(u, why):
        if score_mode == "explain_away":
            rows = explain_away_decompose(bow_row, lam, lay, u, alpha=alpha,
                                          background=background, count_mode=count_mode)
        else:
            rows = lr_decompose(bow_row, lam, lay, u, alpha=alpha, background=background,
                                count_mode=count_mode)
        return [f"  WHY {why} ({node_name.get(u, u)}):"] + \
               ["    " + ln for ln in render_decompose_rows(rows, idx_to_cid, vocab_names)]

    out = [f"person_id={person_id}  [{kind}]  "
           f"({int(bow_row.sum())} coded tokens, {int((bow_row > 0).sum())} distinct codes)"
           f"{contrast}"]
    out += rank_lines
    out += _decomp(top_node, "top")                       # why the model called it
    for u in true_set:                                    # why the truth scored as it did
        if u != top_node:
            out += _decomp(u, "true")
    return "\n".join(out)


def _viewer_context(bundle, lay, vocab_names):
    """Shared lookups the case renderer needs (node names, vocab-idx->cid, sizes)."""
    idx_to_cid = {i: c for c, i in bundle.vocab_map.items()}
    node_name = {u: vocab_names.get(bundle.int2cid.get(u),
                                    bundle.name_by_id.get(bundle.int2cid.get(u), u))
                 for u in lay.nodes}
    return dict(node_name=node_name, idx_to_cid=idx_to_cid, nodes=list(lay.nodes),
                vocab_size=len(bundle.vocab_map))


def write_case_viewer(run_dir, bundle, lam, lay, cases, *, alpha, background,
                      vocab_names, top_nodes=8, count_mode="raw",
                      length_normalize=False, score_mode="lr"):
    """Write a flat (ungrouped) LR breakdown for each selected doc to
    {run_dir}/lr_readout/decompose.txt. Per patient: token/code counts; the LR
    RANKING over DAG nodes (top `top_nodes`); (foreground) the TRUE node(s) + rank
    + a HIT/MISS verdict; and the per-code lr_decompose for the top node (+ each
    true node not the top). Row-level content stays in this file (in-enclave, never
    printed); stdout references only SHA-256-hashed ids. `score_mode` 'lr'|'explain_away'
    -- see `_render_case`. Returns the written Path."""
    ctx = _viewer_context(bundle, lay, vocab_names)
    out_dir = Path(run_dir) / "lr_readout"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "decompose.txt"
    blocks = [_render_case(pid, sv, fr, lam=lam, lay=lay, alpha=alpha,
                           background=background, vocab_names=vocab_names,
                           top_nodes=top_nodes, count_mode=count_mode,
                           length_normalize=length_normalize, score_mode=score_mode,
                           **ctx)
              for pid, sv, fr in cases]
    out_path.write_text("\n\n".join(blocks) + "\n" if blocks else "")
    print(f"[lr_readout]   wrote {len(cases)} case decomposition(s) "
          f"(persons: {', '.join(_hash_person(c[0]) for c in cases)}) to "
          f"{out_path}", flush=True)
    return out_path


def write_case_viewer_by_class(run_dir, bundle, lam, lay, meta, scores, is_fg, *,
                               alpha, background, vocab_names, call_threshold,
                               per_class, top_nodes=8, seed=0, count_mode="raw",
                               length_normalize=False, score_mode="lr"):
    """Classify EVERY held-out doc by the background-vs-rare confusion (max-node
    case score vs `call_threshold`), sample up to `per_class` per class, and write
    them GROUPED by class (false-positive first) to decompose.txt. `scores` is the
    [n_docs x n_nodes] matrix (LR or explain-away, matching `score_mode`) at the
    viewer alpha (aligned with `meta`/`is_fg`). `score_mode` 'lr'|'explain_away' --
    see `_render_case`. Returns (written Path, {class: total_count})."""
    ctx = _viewer_context(bundle, lay, vocab_names)
    nodes = ctx["nodes"]
    block_set = set(lay.block)
    case_score = scores.max(axis=1)
    top_idx = scores.argmax(axis=1)

    buckets = {k: [] for k, _ in _CLASS_ORDER}
    for i, (_pid, _sv, frontier) in enumerate(meta):
        true_set = [u for u in frontier if u in block_set]
        top_node = int(nodes[int(top_idx[i])])
        cls = _classify_error_class(bool(is_fg[i]), bool(case_score[i] >= call_threshold),
                                    top_node in true_set)
        buckets[cls].append(i)

    rng = np.random.default_rng(seed)
    out_dir = Path(run_dir) / "lr_readout"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "decompose.txt"
    counts = {k: len(v) for k, v in buckets.items()}

    sections = [f"call threshold = {call_threshold:+.3f} (max-node {score_mode.upper()} case "
                "score); class totals: " +
                ", ".join(f"{k}={counts[k]}" for k, _ in _CLASS_ORDER)]
    for cls, label in _CLASS_ORDER:
        idxs = buckets[cls]
        if not per_class or not idxs:
            continue
        take = (idxs if len(idxs) <= per_class
                else sorted(rng.choice(idxs, size=per_class, replace=False).tolist()))
        bar = "=" * 72
        block = [bar, f"=== {label}   (showing {len(take)} of {len(idxs)})", bar]
        for i in take:
            pid, sv, frontier = meta[i]
            block.append(_render_case(pid, sv, frontier, lam=lam, lay=lay, alpha=alpha,
                                      background=background, vocab_names=vocab_names,
                                      top_nodes=top_nodes, count_mode=count_mode,
                                      length_normalize=length_normalize,
                                      score_mode=score_mode, **ctx))
        sections.append("\n\n".join(block))

    out_path.write_text("\n\n\n".join(sections) + "\n")
    shown = sum(min(per_class, counts[k]) for k, _ in _CLASS_ORDER if buckets[k])
    print(f"[lr_readout]   wrote error-class case viewer ({shown} cases across "
          f"{sum(1 for k, _ in _CLASS_ORDER if buckets[k])} classes) to {out_path}; "
          f"class totals: {counts}", flush=True)
    return out_path, counts


def build_topic_labels(lay, bundle):
    """{topic index in [0, K): label} for every topic in the fit's layout --
    background topics labeled 'bg{i}'; node topics labeled with the node's
    concept name (+ engine id), and a `[j]` tpn sub-index suffix only when
    tpn>1 (so single-topic-per-node layouts, the common case, stay terse)."""
    labels = {}
    for i in range(lay.n_bg):
        labels[i] = f"bg{i}"
    for u in lay.nodes:
        name = bundle.name_by_id.get(bundle.int2cid.get(u), str(u))
        for j, t in enumerate(lay.block[u]):
            suffix = f"[{j}]" if lay.tpn > 1 else ""
            labels[t] = f"{name} (node {u}){suffix}"
    return labels


def npmi_table(spark, lam, bundle, topic_labels, *, top_n=20):
    """NPMI coherence for every topic (beta = row-normalized lambda), against
    the held-out test corpus as the co-occurrence reference. Returns
    `(topic, label, npmi)` rows sorted by npmi desc (NaN/unrated topics last),
    and prints the table. `topic_labels` from `build_topic_labels`."""
    from spark_vi.eval.topic import compute_npmi_coherence
    from spark_vi.models.topic.types import BOWDocument

    lam = np.asarray(lam, dtype=float)
    beta = lam / np.maximum(lam.sum(axis=1, keepdims=True), 1e-12)
    ref = bundle.test_df.select("features").rdd.map(BOWDocument.from_spark_row)
    ref.cache()
    report = compute_npmi_coherence(beta, ref, top_n=top_n)

    rows = [(int(t), topic_labels.get(int(t), str(t)), float(s))
            for t, s in zip(report.topic_indices, report.per_topic_npmi)]
    rows.sort(key=lambda r: (math.isnan(r[2]), -r[2] if not math.isnan(r[2]) else 0.0))

    print(f"[lr_readout] NPMI coherence (top_n={top_n}, K={len(rows)}, "
          f"{report.n_topics_unrated} unrated, reference_size="
          f"{report.reference_size}): mean={report.mean:.4f} "
          f"median={report.median:.4f} min={report.min:.4f} "
          f"max={report.max:.4f}", flush=True)
    for t, label, npmi in rows:
        val = "nan" if math.isnan(npmi) else f"{npmi:+.4f}"
        print(f"[lr_readout]   topic {t:3d}  {label:40s}  npmi={val}", flush=True)
    return rows


def _viewer_alpha(alpha_values, lr_aucs):
    """The single alpha the per-case viewer decomposes at: the OPERATING point
    -- the swept alpha with the highest case-vs-background LR-AUC. The low-alpha
    log-LR over-penalises common codes (the gate under-represents them; see the
    real-data note), so at low alpha cases do not even separate -- decomposing
    there shows the wrong regime. Picking the best-AUC alpha means the per-code
    breakdown reflects where cases actually separate, so it is interpretable.

    inf is excluded from the choice: at inf the per-code contributions are the
    unbounded (nc/bg - Σλ) limit (huge numbers, dominated by the -Σλ offset), so
    the best-AUC FINITE alpha gives a readable log-ratio breakdown that still
    sums to its score -- and near the peak it approximates the inf limit anyway."""
    finite = [a for a in alpha_values if math.isfinite(a)]
    pool = finite or list(alpha_values)
    if not pool:
        return 0.0
    return max(pool, key=lambda a: lr_aucs.get(a, float("-inf")))


def _background_from_bow(bow, epsilon=1e-9):
    """Corpus code base rate from `bow` (mirrors dag_placement._lr_base_rate's
    background=None branch): the same base rate lr_auc_sweep computes
    internally for its own background=None calls, computed once here so the
    viewer's lr_decompose calls are directly comparable to the AUC sweep's
    scores."""
    col = np.asarray(bow.sum(axis=0)).ravel().astype(float)
    bg = col / max(col.sum(), 1.0)
    return np.maximum(bg, epsilon)


def detection_report(bow, is_fg, lam, lay, *, alpha, background, theta_det,
                     count_mode="raw", length_normalize=False):
    """Deployment metrics for the LR score at `alpha` (max-over-nodes), printed
    beside the run's theta-mass detection block from the manifest. On this
    heavily imbalanced data ROC-AUC hides low precision at low prevalence, so
    the honest numbers are PR-AUC (average precision; random baseline = the
    prevalence) and, at each target foreground sensitivity, the precision (PPV)
    and background false-positive rate you would operate at. Reuses the engine's
    _detection_metrics so LR and theta-mass are scored identically. `count_mode`
    /`length_normalize` match the AUC sweep so the reported operating points are
    for the same score. Also always prints an unconditional explain-away
    (responsibility-weighted LR) block at the alpha->inf lift limit, beside the
    plain-LR and theta-mass blocks, so the three detection lenses are directly
    comparable in every run."""
    from spark_vi.models.topic.dag_placement import (lr_placement_scores,
                                                      _detection_metrics,
                                                      explain_away_placement_scores)
    s = lr_placement_scores(bow, lam, lay, alpha=alpha, background=background,
                            count_mode=count_mode, length_normalize=length_normalize)
    d = _detection_metrics(s.max(axis=1), np.asarray(is_fg, dtype=bool))
    alab = "inf" if math.isinf(alpha) else f"{alpha:.4g}"

    def _ops(det, label):
        print(f"[lr_readout]   {label}: ROC-AUC={det.get('auc', float('nan')):.4f}  "
              f"PR-AUC(AP)={det.get('ap', float('nan')):.4f}", flush=True)
        for t in ("0.80", "0.90", "0.95"):
            op = det.get("operating_points", {}).get(t, {})
            if op:
                print(f"[lr_readout]       @{t} sens: precision={op.get('precision', float('nan')):.4f}"
                      f"  bg_fpr={op.get('bg_fpr', float('nan')):.4f}"
                      f"  specificity={op.get('specificity', float('nan')):.4f}", flush=True)

    print(f"[lr_readout] detection metrics (prevalence={d['prevalence']:.4f} = the "
          f"random PR-AUC baseline; n_fg={d['n_foreground']}/n_bg={d['n_background']}):",
          flush=True)
    _ops(d, f"LR   @alpha={alab}")
    if theta_det:
        _ops(theta_det, "theta-mass (from manifest)")

    # Explain-away (responsibility-weighted) LR, at the alpha->inf lift limit, beside
    # plain LR: does routing comorbid codes to background lift detection?
    ea = explain_away_placement_scores(
        bow, lam, lay, alpha=float("inf"), background=background,
        count_mode=count_mode, length_normalize=length_normalize)
    ea_det = _detection_metrics(ea.max(axis=1), np.asarray(is_fg, dtype=bool))
    _ops(ea_det, "explain-away @alpha=inf")
    return d


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
        bow, is_fg, meta = build_test_bow(bundle, vocab_size, lay)
        print(f"[lr_readout]   held-out test set: {bow.shape[0]} docs, "
              f"{int(is_fg.sum())} foreground, V={vocab_size}, "
              f"K={lay.K} ({lay.n_bg} bg + {len(lay.nodes)} nodes x {lay.tpn} tpn)",
              flush=True)
        # Self-check: our foreground/background counts (foreground = frontier ∩
        # lay.nodes, matching evaluate) must equal the manifest's detection block,
        # else LR-AUC and theta-AUC are scored over different truth sets. A
        # mismatch means the wrong bundle/test set was loaded -- warn loudly.
        det = manifest.get("metrics", {}).get("detection", {})
        m_fg, m_bg = det.get("n_foreground"), det.get("n_background")
        if m_fg is not None and (int(is_fg.sum()) != m_fg
                                 or int((~is_fg).sum()) != (m_bg or 0)):
            print(f"[lr_readout] WARNING: foreground/background counts "
                  f"({int(is_fg.sum())}/{int((~is_fg).sum())}) differ from the "
                  f"manifest's detection block ({m_fg}/{m_bg}) -- the loaded test "
                  f"set may not match the fit; LR-AUC vs theta-AUC is then not "
                  f"strictly comparable. Check --bundle-path / the corpus config.",
                  flush=True)

        alpha_values = resolve_alpha_grid(multipliers, lam, lay)
        lr_aucs = lr_auc_sweep(
            bow, lam, lay, is_fg, alpha_grid=alpha_values,
            count_mode=args.count_mode, length_normalize=args.length_normalize)
        print_readout(multipliers, alpha_values, lr_aucs, theta_auc)

        # Deployment metrics (PR-AUC + precision@sensitivity) at the best-AUC
        # alpha -- the honest read on imbalanced data, beside theta-mass.
        best_alpha = max(alpha_values, key=lambda a: lr_aucs.get(a, float("-inf")))
        detection_report(bow, is_fg, lam, lay, alpha=best_alpha,
                         background=_background_from_bow(bow),
                         theta_det=manifest["metrics"].get("detection", {}),
                         count_mode=args.count_mode,
                         length_normalize=args.length_normalize)

        # NPMI coherence: always printed (aggregate output, no egress concern).
        topic_labels = build_topic_labels(lay, bundle)
        npmi_table(spark, lam, bundle, topic_labels)

        # Per-case decomposition viewer: row-level, written in-enclave only.
        if (args.viewer_per_class or args.sample_cases or args.sample_background
                or args.person is not None):
            # Concept names for the FULL vocabulary via BigQuery (bundle.name_by_id
            # only covers DAG-node concepts, so vocab codes would print as raw ids).
            # Best-effort: on any failure fall back to numeric ids.
            cdr = manifest.get("corpus_manifest", {}).get("cdr") or manifest.get("cdr")
            billing = os.environ.get("GOOGLE_CLOUD_PROJECT")
            vocab_names = {}
            try:
                from dag_placement_cloud import _vocab_concept_names
                vocab_names = _vocab_concept_names(spark, cdr, billing, bundle.vocab_map)
            except Exception as exc:  # noqa: BLE001 - names are best-effort
                print(f"[lr_readout]   WARNING: could not fetch concept names "
                      f"({exc}); decompose.txt will show numeric concept ids. "
                      f"(needs manifest cdr + GOOGLE_CLOUD_PROJECT.)", flush=True)
            background = _background_from_bow(bow)
            viewer_alpha = _viewer_alpha(alpha_values, lr_aucs)

            if args.viewer_per_class:
                # Error-class grouped viewer: classify every held-out doc by the
                # background-vs-rare confusion at the target-sensitivity operating
                # point, sample per class, group by class (false-positive first).
                # `--viewer-score-mode` picks which score classifies/ranks the docs
                # (independent of the unconditional plain+explain-away detection block
                # in detection_report, which always prints both).
                from spark_vi.models.topic.dag_placement import (
                    lr_placement_scores, explain_away_placement_scores,
                    _detection_metrics)
                score_fn = (explain_away_placement_scores
                           if args.viewer_score_mode == "explain_away"
                           else lr_placement_scores)
                scores = score_fn(bow, lam, lay, alpha=viewer_alpha,
                                  background=background,
                                  count_mode=args.count_mode,
                                  length_normalize=args.length_normalize)
                sens = args.viewer_call_sensitivity
                op = _detection_metrics(
                    scores.max(axis=1), is_fg, sens_targets=(sens,)
                )["operating_points"].get(f"{sens:.2f}", {})
                thr = op.get("threshold")
                if thr is None:
                    print("[lr_readout]   WARNING: could not derive a call threshold "
                          f"at sensitivity {sens} (no foreground?); error-class "
                          "viewer not written.", flush=True)
                else:
                    print(f"[lr_readout]   error-class viewer: call threshold at "
                          f"sensitivity {sens:.2f} = {thr:+.3f} (bg_fpr="
                          f"{op.get('bg_fpr', float('nan')):.3f}, precision="
                          f"{op.get('precision', float('nan')):.3f})", flush=True)
                    write_case_viewer_by_class(
                        args.run_dir, bundle, lam, lay, meta, scores, is_fg,
                        alpha=viewer_alpha, background=background,
                        vocab_names=vocab_names, call_threshold=thr,
                        per_class=args.viewer_per_class, top_nodes=args.viewer_top_nodes,
                        count_mode=args.count_mode,
                        length_normalize=args.length_normalize,
                        score_mode=args.viewer_score_mode)
            else:
                cases = select_cases(bundle, sample_cases=args.sample_cases,
                                     sample_background=args.sample_background,
                                     person=args.person)
                if not cases:
                    print("[lr_readout]   WARNING: --sample-cases/--sample-background/"
                          "--person matched no held-out docs; viewer not written.",
                          flush=True)
                else:
                    write_case_viewer(args.run_dir, bundle, lam, lay, cases,
                                      alpha=viewer_alpha, background=background,
                                      vocab_names=vocab_names,
                                      top_nodes=args.viewer_top_nodes,
                                      count_mode=args.count_mode,
                                      length_normalize=args.length_normalize,
                                      score_mode=args.viewer_score_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

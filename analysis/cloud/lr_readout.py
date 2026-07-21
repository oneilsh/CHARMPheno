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
                   help="Itemize an lr_decompose breakdown (per true frontier "
                        "node, rendered with concept names) for this many "
                        "random held-out foreground cases, written to "
                        "{run_dir}/lr_readout/decompose.txt (in-enclave; "
                        "never printed to stdout).")
    p.add_argument("--person", type=int, default=None,
                   help="Itemize an lr_decompose breakdown for this single "
                        "person_id instead of a random sample (also written "
                        "to {run_dir}/lr_readout/decompose.txt).")
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
    """[n_docs x vocab_size] scipy.sparse CSR bag-of-words + boolean is_fg,
    read from bundle.test_df["features","frontier"] (features = a SparseVector
    per CaseFindingBundle's documented schema). Cluster-only: collects the
    held-out split to the driver (the same held-out scale dag_placement_cloud.py
    already collects for its inline eval), not unit-tested.

    is_fg MUST match `evaluate`'s detection definition exactly: foreground iff the
    frontier intersects the SCOREABLE nodes `lay.nodes` (which exclude root 0), not
    merely "frontier nonempty". A root-only / out-of-layout frontier (e.g. {0} for
    a patient coded only with a single-anchor disease's anchor) is BACKGROUND in
    the manifest's theta-mass AUC; counting it foreground here would make LR-AUC and
    theta-AUC use different positive sets and confound the fork-settler."""
    scoreable = set(lay.nodes)
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
        is_fg[i] = bool({int(x) for x in r["frontier"]} & scoreable)
    indices = np.concatenate(idx_chunks) if idx_chunks else np.array([], dtype=np.int64)
    data = np.concatenate(data_chunks) if data_chunks else np.array([], dtype=np.float64)
    bow = sp.csr_matrix((data, indices, indptr), shape=(n, vocab_size))
    return bow, is_fg


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


def render_decompose_rows(rows, idx_to_cid, name_by_id) -> list[str]:
    """[(w, count, contribution), ...] (Task 1's lr_decompose output, already
    sorted by |contribution| desc) -> rendered lines with concept NAMES in
    place of raw vocab indices. `idx_to_cid` = vocab-idx -> concept-id (the
    inverse of CaseFindingBundle.vocab_map, which is {concept_id: vocab_idx});
    `name_by_id` = concept-id -> concept name (CaseFindingBundle.name_by_id).
    Falls back to the concept id, then the raw vocab index, if a lookup
    misses. Pure string formatting; input order is preserved."""
    lines = []
    for w, count, contribution in rows:
        cid = idx_to_cid.get(w, w)
        name = name_by_id.get(cid, cid)
        lines.append(f"{contribution:+.1f}  x{count:g}  {name}")
    return lines


def select_cases(bundle, *, sample_cases=0, person=None, seed=0):
    """Up to `sample_cases` random foreground held-out test docs (nonempty
    frontier), or the single doc matching `person` (a person_id), as
    `(person_id, SparseVector features, frontier engine-ids)` tuples.
    Cluster-only: collects a small sample to the driver, not unit-tested."""
    from pyspark.sql import functions as F

    df = bundle.test_df
    if person is not None:
        df = df.filter(F.col("person_id") == person)
    else:
        df = df.filter(F.size(F.col("frontier")) > 0)
        if sample_cases:
            df = df.orderBy(F.rand(seed)).limit(sample_cases)
    rows = df.select("person_id", "features", "frontier").collect()
    return [(r["person_id"], r["features"], list(r["frontier"])) for r in rows]


def write_case_viewer(run_dir, bundle, lam, lay, cases, *, alpha, background):
    """Write an lr_decompose breakdown for each selected case's true frontier
    node(s), rendered with concept names, to
    {run_dir}/lr_readout/decompose.txt. Row-level (per-patient) content stays
    in that file only -- in-enclave, never printed to stdout; any stdout/log
    line here references only the SHA-256-truncated person id (this repo's
    row-level log hygiene rule). Returns the written Path.

    `idx_to_cid` inverts CaseFindingBundle.vocab_map ({concept_id: vocab_idx}
    -> {vocab_idx: concept_id}); node names come from `int2cid` (engine-id ->
    concept-id) composed with `name_by_id` (concept-id -> name) -- see the
    id-space note on CaseFindingBundle: name_by_id must NOT be indexed by
    engine-id directly."""
    from spark_vi.models.topic.dag_placement import lr_decompose

    idx_to_cid = {i: c for c, i in bundle.vocab_map.items()}
    node_name = {u: bundle.name_by_id.get(bundle.int2cid.get(u), u) for u in lay.nodes}
    vocab_size = len(bundle.vocab_map)

    out_dir = Path(run_dir) / "lr_readout"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "decompose.txt"

    blocks = []
    for person_id, sv, frontier in cases:
        bow_row = np.zeros(vocab_size, dtype=float)
        bow_row[np.asarray(sv.indices, dtype=np.int64)] = np.asarray(sv.values, dtype=float)
        header = [f"person_id={person_id}"]
        for u in frontier:
            if u not in lay.block:
                continue  # frontier node pruned out of this layout; skip
            rows = lr_decompose(bow_row, lam, lay, u, alpha=alpha,
                                background=background)
            header.append(f"  true node: {node_name.get(u, u)} (engine id {u})")
            header.extend("    " + ln for ln in render_decompose_rows(
                rows, idx_to_cid, bundle.name_by_id))
        blocks.append("\n".join(header))

    out_path.write_text("\n\n".join(blocks) + "\n" if blocks else "")
    print(f"[lr_readout]   wrote {len(cases)} case decomposition(s) "
          f"(persons: {', '.join(_hash_person(c[0]) for c in cases)}) to "
          f"{out_path}", flush=True)
    return out_path


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


def _viewer_alpha(multipliers, alpha_values):
    """The single alpha value the per-case viewer's lr_decompose calls use:
    the design-doc's canonical x1-median-Sigma-lambda shrinkage (multiplier
    == 1.0) if it's in the swept grid, else the middle of the swept alpha
    values (both lists are aligned + sorted the same way -- see
    resolve_alpha_grid)."""
    for m, a in zip(multipliers, alpha_values):
        if m == 1.0:
            return a
    return alpha_values[len(alpha_values) // 2] if alpha_values else 0.0


def _background_from_bow(bow, epsilon=1e-9):
    """Corpus code base rate from `bow` (mirrors dag_placement._lr_base_rate's
    background=None branch): the same base rate lr_auc_sweep computes
    internally for its own background=None calls, computed once here so the
    viewer's lr_decompose calls are directly comparable to the AUC sweep's
    scores."""
    col = np.asarray(bow.sum(axis=0)).ravel().astype(float)
    bg = col / max(col.sum(), 1.0)
    return np.maximum(bg, epsilon)


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
        bow, is_fg = build_test_bow(bundle, vocab_size, lay)
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

        # NPMI coherence: always printed (aggregate output, no egress concern).
        topic_labels = build_topic_labels(lay, bundle)
        npmi_table(spark, lam, bundle, topic_labels)

        # Per-case decomposition viewer: row-level, written in-enclave only.
        if args.sample_cases or args.person is not None:
            cases = select_cases(bundle, sample_cases=args.sample_cases,
                                 person=args.person)
            if not cases:
                print("[lr_readout]   WARNING: --sample-cases/--person "
                      "matched no held-out foreground cases; viewer not "
                      "written.", flush=True)
            else:
                background = _background_from_bow(bow)
                viewer_alpha = _viewer_alpha(multipliers, alpha_values)
                write_case_viewer(args.run_dir, bundle, lam, lay, cases,
                                  alpha=viewer_alpha, background=background)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

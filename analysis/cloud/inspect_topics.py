#!/usr/bin/env python
"""inspect_topics.py -- off-cluster inspector for a saved gated-PC fit.

Reads the two artifacts `_save_fit` (gated_pc_cloud.py) writes at the end of a
run -- `<run-dir>/gated_pc_result.npz` and `<run-dir>/manifest.json` -- and
renders a human-readable *topics view* of the model: per-node topic sharpness,
the readout loadings (w_CK: which topics decode each node), and, when a vocab
map is supplied, the top concept words per topic.

Pure numpy + stdlib. NO Spark, NO charmpheno/spark_vi import, NO bundle, NO
cluster. Point it at a run dir you have pulled down locally (the npz is a few
hundred MB at whole-Mondo K; it is mmap'd, not slurped). The npz and manifest
carry only MODEL PARAMETERS and label-node names -- no patient-level data and no
per-cell counts -- so the report is egress-safe by construction.

------------------------------------------------------------------------------
What the artifacts give us (and the one thing they do not)
------------------------------------------------------------------------------
`gated_pc_result.npz`:
  lambda / lambda_0,lambda_1,...   per-domain topic-word Dirichlet params, (K, V_m)
  alpha                            (K,) topic concentration
  w_CK                             (C, K) weightY CO-FIT head (untrained at
                                   weight_y=0 -- NOT the decoder; fallback only)
  b_CK                             (C,)  co-fit head intercept

`readout_heads_gated_pc.npz` (the REAL decoder -- preferred when present):
  V                                (C, K) raw-theta L-BFGS ridge-logistic coeffs
  b_raw                            (C,)  per-node intercept
  degenerate                       (C,)  bool: nodes with no fittable head

`manifest.json`:
  K, C, n_bg, tpn, domain_names, domain_vocab_sizes, per_node_domain_mass,
  corpus_manifest.int2cid  {engine-id -> concept_id}
  corpus_manifest.name_by_id {concept_id -> concept_name}

The topic layout is `DagLayout` (spark-vi dag_placement.py): topics [0, n_bg)
are shared BACKGROUND; then one block of `tpn` topics per non-root node, in
`sorted(engine-id)` order. So foreground topic t (t >= n_bg) belongs to
node = sorted(non-root engine ids)[(t - n_bg) // tpn], which int2cid/name_by_id
name. With tpn=1 this is one dedicated topic per Mondo node -- "inspect the
topics" is "inspect each node's topic."

NOT in the artifacts: the vocab map {concept_id -> vocab index} that would name a
topic's WORDS, and the `parent_int` map that would give each node's DEPTH. Both
live in the bundle, but NOT in a Spark parquet -- in the bundle's `meta` text
file (`_case_finding_cache.save`: `{cache_uri}/{key}/meta/part-*`, one line of
JSON with `vocab_maps`, `parent_int`, `int2cid`, `name_by_id`). That file is
fetchable with a plain `hdfs dfs -cat` (or `gsutil cat`) -- an HDFS/GCS CLIENT
read that requests NO YARN containers, so it never contends with a running fit
and needs no second Spark job. Pass it here as `--bundle-meta meta.json` and the
report gains topic WORDS and a DEPTH column (the deep-node starvation view). For
readable names of measurement/drug vocab features not on the label DAG, add
`--concept-names cid,name.csv`.

  hdfs dfs -cat <cache_uri>/<key>/meta/part-* > meta.json   # off-YARN, safe mid-fit

------------------------------------------------------------------------------
Why the sharpness metrics are what they are
------------------------------------------------------------------------------
A topic's word posterior in domain d is Dirichlet(lambda_dk); its mean is
E[beta_dk] = lambda_dk / lambda_dk.sum(). Two orthogonal readings:

  effective_support = exp(H(E[beta]))   (H = Shannon entropy, nats), in [1, V_m].
      =1  -> all mass on one concept (maximally sharp).
      =V_m -> uniform (flat = the prior; the topic learned nothing to separate
              words). This is the STARVATION signal: a node whose block never
              accrued discriminative tokens sits near the flat prior.
  evidence = lambda_dk.sum()            posterior pseudo-count mass on the topic.
      lambda = eta_prior + E[token counts], so a large sum means the topic
      absorbed real data; a sum near V_m * eta means it saw almost none.

`effective_support` is prior-scale invariant (it reads the *shape* of E[beta]),
so it is the robust flatness ranking; `evidence` is the raw how-much-data read
that explains WHY a topic is flat. A rare/deep node that underperforms and whose
topic is both flat (support ~ V_m) and low-evidence is starved, not mis-decoded;
one that is sharp but still underperforms is a decode/label-space problem. That
is the discrimination this tool exists to make.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #
def resolve_run_dir(pattern):
    """Resolve a run dir from an exact path or a glob (e.g. `.../0111-*`).

    Mirrors gated_pc_readout's resolution: an exact dir holding
    gated_pc_result.npz is used as-is; otherwise the pattern is globbed and
    filtered to matched dirs that contain the npz, so a shared runs/ dir with a
    same-numbered run from another experiment does not collide.
    """
    import glob as _glob
    p = Path(pattern)
    if p.is_dir() and (p / "gated_pc_result.npz").exists():
        return p
    hits = [Path(m) for m in _glob.glob(str(pattern))
            if Path(m).is_dir() and (Path(m) / "gated_pc_result.npz").exists()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SystemExit(
            f"[inspect_topics] no run dir with gated_pc_result.npz matches "
            f"{pattern!r} (has the fit written its result yet?)")
    raise SystemExit(f"[inspect_topics] {pattern!r} is ambiguous: "
                     f"{[str(h) for h in hits]}; pass an exact dir")


def load_run(run_dir: Path):
    """Return (npz, manifest). npz is mmap'd; manifest is the parsed JSON dict."""
    run_dir = Path(run_dir)
    npz = np.load(run_dir / "gated_pc_result.npz", mmap_mode="r")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return npz, manifest


def load_readout_heads(run_dir, label="gated_pc"):
    """Return the REAL readout decoder {'W','b','degenerate','src','standardized'}
    from a run dir, trying two sources in order, or None if neither exists.

    gated_pc_result.npz's `w_CK` is only the weightY co-fit head and is untrained
    on a weight_y=0 run, so it is NOT the decoder; the decoder is the L-BFGS
    ridge-logistic heads. Two on-disk forms of them:

    1. `readout_heads_{label}.npz` -- the COMPLETED fit's raw-theta coeffs
       (`_write_readout_heads`, written right after the solve returns): V (C,K),
       b_raw (C,), degenerate (C,) bool mask of no-fittable-head nodes (the
       "detection-skipped" ones). Preferred.
    2. `readout_ckpt_{label}.npz` -- the solver CHECKPOINT (W_std/b_std, every 10
       iters). STANDARDIZED-theta weights, not raw, but the ranking of which
       topics load on a node is unchanged by the per-topic scaling, so it is a
       fine loadings source. It survives the calibration sub-fit (which runs with
       no checkpoint_dir), so it is on disk even when the heads sidecar write was
       missed. Used only when (1) is absent.

    `W` is (C,K) whichever it found; `standardized` says which space it is in.
    """
    run_dir = Path(run_dir)
    heads_p = run_dir / f"readout_heads_{label}.npz"
    ckpt_p = run_dir / f"readout_ckpt_{label}.npz"

    # The checkpoint's W_std is the STANDARDIZED weight (per-SD-of-theta), the
    # honest loadings scale. It survives the calibration sub-fit, so it is on disk
    # alongside the heads. Load it whenever present, for the loadings column.
    ckpt_Wstd = ckpt_iter = None
    if ckpt_p.exists():
        z = np.load(ckpt_p)
        ckpt_Wstd = np.asarray(z["W_std"], dtype=np.float64)
        ckpt_iter = int(z["iter"]) if "iter" in z.files else -1

    if heads_p.exists():
        z = np.load(heads_p)
        V = np.asarray(z["V"], dtype=np.float64)
        # V is raw-theta (V = W_std / sigma_k): coefficients EXPLODE for
        # low-variance (starved) topics, so V is the scoring decoder but NOT an
        # honest importance ranking. Prefer the checkpoint's standardized W_std
        # for the loadings display; fall back to V with a caveat if absent.
        if ckpt_Wstd is not None and ckpt_Wstd.shape == V.shape:
            W_load, load_std = ckpt_Wstd, True
            load_note = f"standardized W_std from ckpt iter {ckpt_iter}"
        else:
            W_load, load_std = V, False
            load_note = ("raw-θ V — INFLATED for low-variance/starved topics; "
                         "no ckpt W_std to standardize against")
        return {"W_load": W_load, "b": np.asarray(z["b_raw"], dtype=np.float64),
                "degenerate": (np.asarray(z["degenerate"], dtype=bool)
                               if "degenerate" in z.files else None),
                "src": f"readout_heads_{label}.npz (V raw-θ decoder); "
                       f"loadings = {load_note}",
                "standardized": load_std}
    if ckpt_Wstd is not None:
        z = np.load(ckpt_p)
        return {"W_load": ckpt_Wstd,
                "b": np.asarray(z["b_std"], dtype=np.float64),
                "degenerate": None,
                "src": f"readout_ckpt_{label}.npz (W_std standardized-θ, "
                       f"checkpoint iter {ckpt_iter})",
                "standardized": True}
    return None


def domain_lambdas(npz):
    """The per-domain lambda arrays in domain order, as a list of (K, V_m).

    Mirrors `_reconstruct` in gated_pc_readout.py: a single-domain run stores
    `lambda`; the multi-domain (Mondo) path stores `lambda_0, lambda_1, ...`.
    """
    files = set(npz.files)
    if "lambda" in files:
        return [npz["lambda"]]
    doms = sorted(int(f.split("_", 1)[1]) for f in files if f.startswith("lambda_"))
    if not doms:
        raise KeyError(f"no lambda in npz: found {sorted(files)}")
    return [npz[f"lambda_{m}"] for m in doms]


# --------------------------------------------------------------------------- #
# The topic <-> node map (DagLayout ordering, reconstructed from the manifest)  #
# --------------------------------------------------------------------------- #
def node_order(manifest):
    """Sorted non-root engine ids -- DagLayout's `self.nodes` (sorted(parents)).

    `int2cid` keys every engine node including root (id 0). DagLayout blocks are
    laid over the non-root nodes in sorted-id order, so this list, indexed by the
    per-node block index i, is exactly what `block[i]` was built from.
    """
    cm = manifest.get("corpus_manifest", {})
    int2cid = {int(k): int(v) for k, v in cm.get("int2cid", {}).items()}
    return sorted(e for e in int2cid if e != 0), int2cid


def topic_labels(manifest):
    """A length-K list labelling each topic: 'BG{t}' for background, else the
    owning node's name (falling back to concept id / engine id when unnamed).

    Returns (labels, topic2engine) where topic2engine[t] is the node engine id a
    foreground topic decodes into (None for background) -- used to cross-index
    w_CK rows (which are indexed by node engine id 0..C-1) against topics.
    """
    n_bg = int(manifest["n_bg"])
    tpn = int(manifest["tpn"])
    K = int(manifest["K"])
    nodes, int2cid = node_order(manifest)
    cm = manifest.get("corpus_manifest", {})
    name_by_id = {int(k): v for k, v in cm.get("name_by_id", {}).items()}

    labels = [None] * K
    topic2engine = [None] * K
    for t in range(K):
        if t < n_bg:
            labels[t] = f"BG{t}"
            continue
        i = (t - n_bg) // tpn
        if i >= len(nodes):                      # defensive: K wider than nodes
            labels[t] = f"topic{t}(unmapped)"
            continue
        eng = nodes[i]
        cid = int2cid.get(eng)
        nm = name_by_id.get(cid)
        labels[t] = nm or (f"cid:{cid}" if cid is not None else f"eng:{eng}")
        topic2engine[t] = eng
    return labels, topic2engine


def node_names(manifest):
    """engine id -> display name for w_CK rows (indexed by node engine id)."""
    _, int2cid = node_order(manifest)
    cm = manifest.get("corpus_manifest", {})
    name_by_id = {int(k): v for k, v in cm.get("name_by_id", {}).items()}
    out = {}
    for eng, cid in int2cid.items():
        out[eng] = name_by_id.get(cid) or (f"cid:{cid}")
    return out


def node_depths(parent_int):
    """engine id -> longest-path depth from root (id 0). Replicates
    DagLayout.depth from the bundle meta's `parent_int` {child: [parents]} so the
    report can rank nodes by depth WITHOUT importing spark-vi. Memoized,
    cycle-guarded (parent maps are acyclic by construction, but this is defensive).
    """
    parents = {int(c): [int(p) for p in ps] for c, ps in parent_int.items()}
    memo = {}

    def d(v, stack=()):
        if v in memo:
            return memo[v]
        ps = [p for p in parents.get(v, []) if p != v and p not in stack]
        val = 0 if not ps else 1 + max(d(p, stack + (v,)) for p in ps)
        memo[v] = val
        return val

    return {v: d(v) for v in parents}


def load_bundle_meta(path):
    """Parse a bundle `meta` file (the `hdfs dfs -cat .../meta/part-*` output).

    Accepts the raw one-line JSON, or a file with that line among others (takes
    the first line that parses to a dict with `int2cid`). Returns the dict with
    `vocab_maps`, `parent_int`, `name_by_id`, `int2cid` (as written by
    _case_finding_cache._meta_dict), or None on any failure.
    """
    text = Path(path).read_text()
    for line in [text] + text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "int2cid" in obj:
            return obj
    return None


# --------------------------------------------------------------------------- #
# Sharpness                                                                    #
# --------------------------------------------------------------------------- #
def _entropy_support(beta_row):
    """exp(Shannon entropy) of a normalized nonneg row; in [1, len(row)]."""
    p = beta_row[beta_row > 0]
    if p.size == 0:
        return float("nan")
    h = -np.sum(p * np.log(p))
    return float(math.exp(h))


def topic_sharpness(lams):
    """Per-topic sharpness across domains.

    Returns arrays indexed by topic t (length K):
      evidence[t]        sum over domains of lambda_dk.sum()  (posterior mass)
      dom_mass[t]        (n_domains,) lambda mass by domain (which domain the
                         topic emits into -- matches per_node_domain_mass)
      support[t]         effective_support in the topic's DOMINANT domain
      top1[t]            max E[beta] in the dominant domain
      support_frac[t]    support / V_dominant  (0->sharp, 1->flat/prior)
    """
    K = lams[0].shape[0]
    n_dom = len(lams)
    evidence = np.zeros(K)
    dom_mass = np.zeros((K, n_dom))
    support = np.zeros(K)
    top1 = np.zeros(K)
    support_frac = np.zeros(K)
    for t in range(K):
        sums = np.array([float(np.asarray(l[t]).sum()) for l in lams])
        dom_mass[t] = sums
        evidence[t] = sums.sum()
        d = int(np.argmax(sums))               # dominant domain
        row = np.asarray(lams[d][t], dtype=np.float64)
        s = row.sum()
        beta = row / s if s > 0 else row
        support[t] = _entropy_support(beta)
        top1[t] = float(beta.max()) if s > 0 else float("nan")
        V = row.shape[0]
        support_frac[t] = support[t] / V if V else float("nan")
    return dict(evidence=evidence, dom_mass=dom_mass, support=support,
                top1=top1, support_frac=support_frac)


# --------------------------------------------------------------------------- #
# Optional vocab-word rendering                                               #
# --------------------------------------------------------------------------- #
def load_vocab_maps(path, n_domains):
    """Load vocab maps as a list of {vocab_idx -> concept_id} per domain.

    Accepts either a JSON list of {concept_id: idx} dicts (one per domain, the
    shape of `bundle.vocab_maps`), a single JSON {concept_id: idx} dict (one
    domain), or an .npz of arrays `vocab_0, vocab_1, ...` of concept ids in
    index order. Inverts to idx->cid for rendering.
    """
    path = Path(path)
    inv = []
    if path.suffix == ".npz":
        z = np.load(path)
        doms = sorted(int(f.split("_", 1)[1]) for f in z.files if f.startswith("vocab_"))
        for m in doms:
            arr = z[f"vocab_{m}"]
            inv.append({i: int(c) for i, c in enumerate(arr)})
    else:
        obj = json.loads(path.read_text())
        maps = obj if isinstance(obj, list) else [obj]
        for m in maps:
            inv.append({int(idx): int(cid) for cid, idx in m.items()})
    if len(inv) != n_domains:
        print(f"[warn] vocab map has {len(inv)} domains, model has {n_domains}; "
              "rendering the overlap only")
    return inv


def load_concept_names(path):
    """concept_id -> name from a CSV with columns (concept_id, concept_name)."""
    names = {}
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        header = next(r, None)
        # tolerate a header row or none
        if header and not header[0].strip().isdigit():
            pass
        elif header:
            names[int(header[0])] = header[1] if len(header) > 1 else ""
        for row in r:
            if len(row) >= 2 and row[0].strip():
                try:
                    names[int(row[0])] = row[1]
                except ValueError:
                    continue
    return names


def top_words(lam_row, inv_map, names, name_by_id, t_words):
    """Top-t (concept, E[beta]) for one topic's domain row, named if possible."""
    row = np.asarray(lam_row, dtype=np.float64)
    s = row.sum()
    if s <= 0:
        return []
    beta = row / s
    idx = np.argsort(beta)[::-1][:t_words]
    out = []
    for i in idx:
        cid = inv_map.get(int(i)) if inv_map else None
        nm = None
        if cid is not None:
            nm = (names.get(cid) if names else None) or name_by_id.get(cid)
        label = nm or (f"cid:{cid}" if cid is not None else f"idx:{int(i)}")
        out.append((label, float(beta[i])))
    return out


def topic_word_lines(t, lams, inv_maps, names, name_by_id, dom_names, t_words,
                     *, indent=""):
    """Markdown lines for topic t's top words in EVERY domain (one line each).

    Each domain is its own sub-line so the block reads as a list rather than a
    `·`-run-on, and all domains show (not just the dominant one) so a topic that
    splits its mass across labs/conditions/drugs is legible. `indent` prefixes
    every line (for the depth-indented tree tour).
    """
    out = []
    for d, lam in enumerate(lams):
        inv = inv_maps[d] if (inv_maps and d < len(inv_maps)) else None
        tw = top_words(np.asarray(lam[t]), inv, names, name_by_id, t_words)
        body = (" · ".join(f"{nm} ({p:.3f})" for nm, p in tw) if tw
                else "_(no vocab map — supply --bundle-meta/--vocab-map)_")
        dom = dom_names[d] if d < len(dom_names) else f"dom{d}"
        out.append(f"{indent}  - **{dom}:** {body}")
    return out


# --------------------------------------------------------------------------- #
# Report                                                                       #
# --------------------------------------------------------------------------- #
def build_report(run_dir, *, top_topics, top_loadings, t_words,
                 vocab_path=None, names_path=None, sort_by="sharpness",
                 bundle_meta_path=None, readout_label="gated_pc",
                 tour_per_depth=0):
    npz, manifest = load_run(run_dir)
    lams = domain_lambdas(npz)
    K = int(manifest["K"]); C = int(manifest["C"])
    n_bg = int(manifest["n_bg"]); tpn = int(manifest["tpn"])
    dom_names = manifest.get("domain_names") or [f"dom{i}" for i in range(len(lams))]
    alpha = np.asarray(npz["alpha"], dtype=np.float64)
    # The DECODER: prefer the persisted L-BFGS readout heads (V, raw-theta), the
    # model that actually scored the run. gated_pc_result.npz's w_CK is only the
    # weightY co-fit head -- untrained at weight_y=0 -- and is the fallback for
    # runs that predate the heads sidecar.
    heads = load_readout_heads(run_dir, readout_label)
    if heads is not None:
        w_CK = heads["W_load"]
        b_CK = heads["b"]
        degenerate = heads["degenerate"]
        decoder_src = heads["src"]
    else:
        w_CK = np.asarray(npz["w_CK"], dtype=np.float64)
        b_CK = (np.asarray(npz["b_CK"], dtype=np.float64)
                if "b_CK" in npz.files else np.zeros(C))
        degenerate = None
        decoder_src = ("gated_pc_result.npz w_CK (weightY co-fit head, NOT the "
                       "readout decoder — heads sidecar absent)")

    labels, topic2engine = topic_labels(manifest)
    nnames = node_names(manifest)
    sh = topic_sharpness(lams)

    inv_maps = names = None
    depths = None
    name_by_id = {int(k): v for k, v in
                  manifest.get("corpus_manifest", {}).get("name_by_id", {}).items()}
    # The bundle meta (off-YARN `hdfs dfs -cat .../meta/part-*`) supplies BOTH the
    # vocab maps (topic words) and parent_int (node depth). An explicit --vocab-map
    # overrides the meta's vocab; --concept-names always supplements the names.
    meta = load_bundle_meta(bundle_meta_path) if bundle_meta_path else None
    if meta:
        # Guard against a wrong bundle key: the meta must describe THIS run's
        # bundle, or its vocab_maps would mislabel every topic word and its
        # parent_int would give wrong depths. Check the two things the run's own
        # manifest also records -- the label DAG (int2cid) and the per-domain
        # vocab sizes -- and warn loudly on a mismatch instead of rendering lies.
        run_int2cid = {int(k): int(v) for k, v in
                       manifest.get("corpus_manifest", {}).get("int2cid", {}).items()}
        meta_int2cid = {int(k): int(v) for k, v in meta.get("int2cid", {}).items()}
        run_vsz = manifest.get("domain_vocab_sizes") or \
            manifest.get("corpus_manifest", {}).get("domain_vocab_sizes")
        meta_vsz = [len(vm) for vm in meta.get("vocab_maps", [])]
        mismatch = []
        if meta_int2cid and run_int2cid and meta_int2cid != run_int2cid:
            mismatch.append(f"label DAG differs ({len(meta_int2cid)} vs "
                            f"{len(run_int2cid)} nodes)")
        if run_vsz and meta_vsz and list(run_vsz) != meta_vsz:
            mismatch.append(f"vocab sizes {meta_vsz} != run's {list(run_vsz)}")
        if mismatch:
            warn = ("**WARNING: --bundle-meta looks like a DIFFERENT bundle than "
                    "this run** (" + "; ".join(mismatch) + "). Topic words / depth "
                    "may be MISLABELLED -- pick the run's own bundle key.")
            print("[inspect_topics] " + warn, file=sys.stderr)
            meta_mismatch_warn = warn
        else:
            meta_mismatch_warn = None
        if "parent_int" in meta:
            depths = node_depths(meta["parent_int"])
        if "name_by_id" in meta:
            name_by_id = {int(k): v for k, v in meta["name_by_id"].items()} or name_by_id
        if not vocab_path and "vocab_maps" in meta:
            inv_maps = [{int(idx): int(cid) for cid, idx in vm.items()}
                        for vm in meta["vocab_maps"]]
    else:
        meta_mismatch_warn = None
    if vocab_path:
        inv_maps = load_vocab_maps(vocab_path, len(lams))
    if names_path:
        names = load_concept_names(names_path)

    L = []
    w = L.append
    w(f"# Topics view -- {Path(run_dir).name}")
    w("")
    if meta_mismatch_warn:
        w("> " + meta_mismatch_warn)
        w("")
    w(f"- K = {K} topics ({n_bg} background + {len(range(n_bg, K))} node-tied, "
      f"tpn={tpn}); C = {C} label nodes")
    w(f"- domains: {', '.join(f'{n}(V={l.shape[1]})' for n, l in zip(dom_names, lams))}")
    w(f"- alpha: min {alpha.min():.4g} / median {np.median(alpha):.4g} / "
      f"max {alpha.max():.4g} / sum {alpha.sum():.4g}")
    fg_frac = sh["support_frac"][n_bg:K]
    fg_ev = sh["evidence"][n_bg:K]
    n_fg = fg_frac.size
    sharp = int(np.sum(fg_frac < 0.2))
    mid = int(np.sum((fg_frac >= 0.2) & (fg_frac <= 0.5)))
    flat = int(np.sum(fg_frac > 0.5))
    w(f"- node-topic sharpness ({n_fg} topics): {sharp} sharp (frac<0.2) / "
      f"{mid} mid / {flat} flat/starved (frac>0.5) "
      f"[{100*flat/max(n_fg,1):.0f}% starved]")
    # Evidence (pseudo-count mass) quantiles say how many topics saw ~no data;
    # the prior floor is ~min(evidence), so topics near it are starved.
    q = np.percentile(fg_ev, [10, 50, 90])
    w(f"- node-topic evidence: min {fg_ev.min():.3g} / p10 {q[0]:.3g} / "
      f"median {q[1]:.3g} / p90 {q[2]:.3g} / max {fg_ev.max():.3g} "
      f"(near-min = starved)")
    w(f"- decoder: {decoder_src}")
    if degenerate is not None:
        w(f"- degenerate heads: {int(degenerate.sum())} / {C} nodes had no "
          f"fittable head (const fallback; the 'detection-skipped' nodes)")
    w("")

    # ---- background topics ----
    w("## Background topics (shared)")
    w("")
    w("| topic | evidence | eff.support | top1 | dominant domain |")
    w("|---|--:|--:|--:|---|")
    for t in range(n_bg):
        d = int(np.argmax(sh["dom_mass"][t]))
        w(f"| BG{t} | {sh['evidence'][t]:.4g} | {sh['support'][t]:.1f} | "
          f"{sh['top1'][t]:.3f} | {dom_names[d]} |")
    w("")
    # Background carries most of the corpus mass; its words say what "everyone"
    # looks like (the backbone the node topics are deflated against).
    if inv_maps:
        w(f"Background words (top {t_words} per domain):")
        w("")
        for t in range(n_bg):
            w(f"- **BG{t}** — ev {sh['evidence'][t]:.3g}, support {sh['support'][t]:.0f}")
            for line in topic_word_lines(t, lams, inv_maps, names, name_by_id,
                                         dom_names, t_words):
                w(line)
        w("")

    # ---- per-node topic sharpness (the headline) ----
    def depth_of(t):
        eng = topic2engine[t]
        return depths.get(eng, -1) if (depths and eng is not None) else -1

    order = np.arange(n_bg, K)
    if sort_by == "sharpness":            # flattest (most starved) first
        order = order[np.argsort(-sh["support_frac"][order])]
    elif sort_by == "evidence":           # lowest evidence first
        order = order[np.argsort(sh["evidence"][order])]
    elif sort_by == "alpha":
        order = order[np.argsort(-alpha[order])]
    elif sort_by == "depth":              # deepest first (needs --bundle-meta)
        if not depths:
            w("_(sort=depth requested but no --bundle-meta parent_int; "
              "falling back to sharpness order)_")
            w("")
            order = order[np.argsort(-sh["support_frac"][order])]
        else:
            order = sorted(order, key=lambda t: (-depth_of(t), sh["support_frac"][t]))
            order = np.array(order)
    order = order[:top_topics]

    have_depth = depths is not None
    dh = "depth | " if have_depth else ""
    dsep = "--:|" if have_depth else ""

    def emit_table(topics):
        w(f"| node topic | {dh}evidence | eff.support | frac | dom | self-w | "
          "intercept | top borrowed topics |")
        w(f"|---|{dsep}--:|--:|--:|---|--:|--:|---|")
        for t in topics:
            eng = topic2engine[t]
            d = int(np.argmax(sh["dom_mass"][t]))
            self_w = borrow = intc = ""
            deg = (degenerate is not None and eng is not None
                   and eng < C and degenerate[eng])
            if deg:
                self_w = borrow = "(degenerate head)"
            elif eng is not None and eng < C:
                row = w_CK[eng]
                self_w = f"{row[t]:+.3f}"
                oth = np.argsort(-np.abs(row))
                picks = [j for j in oth if j != t][:top_loadings]
                borrow = ", ".join(f"{labels[j]}({row[j]:+.2f})" for j in picks
                                   if abs(row[j]) > 1e-6)
                intc = f"{b_CK[eng]:+.3f}"
            dcol = (f"{depth_of(t)} | " if have_depth else "")
            w(f"| {labels[t]} | {dcol}{sh['evidence'][t]:.4g} | "
              f"{sh['support'][t]:.1f} | {sh['support_frac'][t]:.2f} | "
              f"{dom_names[d]} | {self_w} | {intc} | {borrow} |")
        w("")

    w(f"## Node topics -- top {len(order)} by {sort_by}")
    w("")
    w("evidence = posterior pseudo-count mass; eff.support = exp(entropy) of the "
      "word dist in the dominant domain (low=sharp, ->V=flat/prior); "
      "self-w = readout weight the node puts on its OWN topic; "
      "borrows = its largest-|w| OTHER topics (ancestor/background/cousin decode)."
      + (" depth = longest path from root." if have_depth else ""))
    w("")
    emit_table(order)

    # The BEST-FED end -- the topics that actually learned something. The main
    # table's default (flattest-first) buries these; showing them explicitly is
    # what lets a reader check the SHARP topics are clinically coherent.
    n_best = min(len(order), 25)
    best_order = np.array(sorted(range(n_bg, K),
                                 key=lambda t: -sh["evidence"][t])[:n_best])
    w(f"## Best-fed node topics -- top {n_best} by evidence")
    w("")
    emit_table(best_order)

    # words render for both ends, de-duplicated, main order first
    seen = set()
    word_order = [t for t in list(order) + list(best_order)
                  if not (t in seen or seen.add(t))]

    # ---- sharpness-by-depth rollup (the deep-node question, when depth known) ----
    if have_depth:
        w("## Sharpness by depth")
        w("")
        w("_Median over each depth's node topics. If eff.support climbs and "
          "evidence falls with depth, deep nodes are STARVED (flat topics), which "
          "is a data problem, not a decode problem._")
        w("")
        w("| depth | n nodes | median evidence | median eff.support | median frac |")
        w("|--:|--:|--:|--:|--:|")
        fg_t = np.arange(n_bg, K)
        by_d = {}
        for t in fg_t:
            by_d.setdefault(depth_of(t), []).append(t)
        for dep in sorted(k for k in by_d if k >= 0):
            ts = by_d[dep]
            w(f"| {dep} | {len(ts)} | {np.median(sh['evidence'][ts]):.3g} | "
              f"{np.median(sh['support'][ts]):.1f} | "
              f"{np.median(sh['support_frac'][ts]):.2f} |")
        w("")

    # ---- topic -> words, ALL domains, one line each ----
    w(f"## Top {t_words} concepts per topic (all domains)")
    w("")
    if not inv_maps:
        w("_(no --bundle-meta/--vocab-map: topic word distributions are stored by "
          "vocab INDEX; supply one to name the words.)_")
        w("")
    else:
        for t in word_order:
            w(f"- **{labels[t]}** — ev {sh['evidence'][t]:.3g}, "
              f"support {sh['support'][t]:.0f}, frac {sh['support_frac'][t]:.2f}")
            for line in topic_word_lines(t, lams, inv_maps, names, name_by_id,
                                         dom_names, t_words):
                w(line)
        w("")

    # ---- tree tour: topics sampled across depths, indented by level ----
    if tour_per_depth and have_depth:
        w(f"## Tree tour -- up to {tour_per_depth} best-fed node(s) per depth")
        w("")
        w("_Indented by depth; the highest-evidence nodes at each level (what a "
          "topic at that depth looks like when it is fed), all domains shown._")
        w("")
        by_d = {}
        for t in range(n_bg, K):
            by_d.setdefault(depth_of(t), []).append(t)
        for dep in sorted(k for k in by_d if k >= 0):
            picks = sorted(by_d[dep], key=lambda t: -sh["evidence"][t])[:tour_per_depth]
            ind = "  " * min(max(dep - 1, 0), 10)
            for t in picks:
                eng = topic2engine[t]
                deg = (degenerate is not None and eng is not None
                       and eng < C and degenerate[eng])
                flag = " · DEGENERATE head" if deg else ""
                starved = " · STARVED (flat prior)" if sh["support_frac"][t] > 0.5 else ""
                w(f"{ind}- `d{dep}` **{labels[t]}** — ev {sh['evidence'][t]:.3g}, "
                  f"support {sh['support'][t]:.0f}/{lams[int(np.argmax(sh['dom_mass'][t]))].shape[1]}"
                  f"{starved}{flag}")
                for line in topic_word_lines(t, lams, inv_maps, names, name_by_id,
                                             dom_names, t_words, indent=ind):
                    w(line)
        w("")

    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="Run dir with gated_pc_result.npz + manifest.json")
    ap.add_argument("--out", default=None,
                    help="Write the markdown report here (default: "
                         "<run-dir>/topics_report.md). '-' for stdout only.")
    ap.add_argument("--top-topics", type=int, default=60,
                    help="How many node topics to detail (default 60).")
    ap.add_argument("--top-loadings", type=int, default=6,
                    help="Borrowed topics to show per node row (default 6).")
    ap.add_argument("--top-words", type=int, default=15,
                    help="Concepts per topic when --vocab-map is given (default 15).")
    ap.add_argument("--sort", choices=["sharpness", "evidence", "alpha", "depth"],
                    default="sharpness",
                    help="Order the node-topic table (default sharpness: "
                         "flattest/most-starved first; depth needs --bundle-meta).")
    ap.add_argument("--bundle-meta", default=None,
                    help="Bundle meta JSON (the `hdfs dfs -cat "
                         "<cache_uri>/<key>/meta/part-*` output). Off-YARN to "
                         "fetch, safe mid-fit. Supplies topic WORDS (vocab_maps) "
                         "and node DEPTH (parent_int).")
    ap.add_argument("--vocab-map", default=None,
                    help="Optional standalone vocab map (JSON list of {cid:idx} per "
                         "domain, or .npz of vocab_0.. arrays); overrides --bundle-meta's.")
    ap.add_argument("--concept-names", default=None,
                    help="Optional CSV (concept_id,concept_name) for vocab feature "
                         "names not covered by the label DAG's name_by_id.")
    ap.add_argument("--readout-label", default="gated_pc",
                    help="Arm label of the readout heads sidecar to read "
                         "(readout_heads_<label>.npz; default gated_pc).")
    ap.add_argument("--tour", type=int, default=0, metavar="N",
                    help="Add a TREE TOUR: the N best-fed node topics at EACH "
                         "depth, indented by level, all domains shown (needs "
                         "--bundle-meta for depth). Try --tour 2.")
    args = ap.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    report = build_report(
        run_dir, top_topics=args.top_topics, top_loadings=args.top_loadings,
        t_words=args.top_words, vocab_path=args.vocab_map,
        names_path=args.concept_names, sort_by=args.sort,
        bundle_meta_path=args.bundle_meta, readout_label=args.readout_label,
        tour_per_depth=args.tour)

    print(report)
    if args.out != "-":
        out = Path(args.out) if args.out else Path(run_dir) / "topics_report.md"
        out.write_text(report + "\n")
        print(f"\n[inspect_topics] wrote {out}", flush=True)


if __name__ == "__main__":
    main()

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
import re
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
        if "W_std" in z.files and np.asarray(z["W_std"]).shape == V.shape:
            # The sidecar carries the solve's own standardized weights (written
            # since the W_std-in-heads change): the honest scale, no ckpt needed.
            W_load, load_std = np.asarray(z["W_std"], dtype=np.float64), True
            load_note = "standardized W_std from the heads sidecar"
        elif ckpt_Wstd is not None and ckpt_Wstd.shape == V.shape:
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


def _topic_unit_vec(t, lams):
    """L2-normalized concat of a topic's E[beta] across domains (for cosine).

    Concatenating the per-domain E[beta] (mass-weighted by each domain's share of
    the topic, since a domain the topic barely emits into contributes a tiny
    sub-vector) gives one comparable vector per topic. Two topics with the same
    multi-domain content have cosine ~1; different content ~0. NOTE: two STARVED
    topics are both ~uniform and so trivially cosine ~1 -- redundancy is only
    meaningful among FED (sharp) topics, which the caller enforces.
    """
    parts = []
    for lam in lams:
        row = np.asarray(lam[t], dtype=np.float64)
        s = row.sum()
        parts.append(row / s if s > 0 else row)   # E[beta] per domain (sums to 1)
    v = np.concatenate(parts)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def sibling_redundancy(parent_int, topic2engine, lams, sh, n_bg, K, *,
                       fed_frac=0.5):
    """Per-parent differentiation among its children's topics.

    Returns a list of dicts (one per parent with >=2 children), each:
      parent (engine id), fanout, n_fed, med_cos_all, med_cos_fed, max_cos_fed,
      fed_children (engine ids). `med_cos_fed` over FED children (support_frac <
      fed_frac) is the signal: high = the parent's fed children collapse to one
      topic (capacity starvation at that parent); a spread = healthy heterogeneity.
    """
    eng2topic = {e: t for t, e in enumerate(topic2engine) if e is not None}
    children = {}
    for c, ps in parent_int.items():
        for p in ps:
            children.setdefault(int(p), []).append(int(c))
    out = []
    for p, kids in children.items():
        ktopics = [(e, eng2topic[e]) for e in kids if e in eng2topic]
        if len(ktopics) < 2:
            continue
        fed = [(e, t) for e, t in ktopics if sh["support_frac"][t] < fed_frac]
        vecs = {t: _topic_unit_vec(t, lams) for _, t in ktopics}

        def _pairwise(items):
            ts = [t for _, t in items]
            cs = [float(np.dot(vecs[a], vecs[b]))
                  for i, a in enumerate(ts) for b in ts[i + 1:]]
            return cs

        cos_all = _pairwise(ktopics)
        cos_fed = _pairwise(fed)
        out.append({
            "parent": p, "fanout": len(ktopics), "n_fed": len(fed),
            "med_cos_all": float(np.median(cos_all)) if cos_all else float("nan"),
            "med_cos_fed": float(np.median(cos_fed)) if cos_fed else float("nan"),
            "max_cos_fed": float(np.max(cos_fed)) if cos_fed else float("nan"),
            "fed_children": [e for e, _ in fed],
        })
    return out


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
    """Top-t (concept, E[beta], vocab_idx) for one topic's domain row, named if
    possible. The vocab index rides along so callers can cross-reference the
    word against index sets (e.g. the HPO-profile marker in `_digest_words`)."""
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
        out.append((label, float(beta[i]), int(i)))
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
        body = (" · ".join(f"{nm} ({p:.3f})" for nm, p, _ in tw) if tw
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
                 tour_per_depth=0, redundancy=0, grep_pattern=None):
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
    depths = parent_int = None
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
            parent_int = {int(c): [int(p) for p in ps]
                          for c, ps in meta["parent_int"].items()}
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

    # ---- --grep: look up specific nodes by name (evidence vs. depth probe) ----
    matched = []
    if grep_pattern:
        rx = re.compile(grep_pattern, re.I)
        matched = [t for t in range(n_bg, K) if rx.search(labels[t] or "")]
        matched = sorted(matched, key=lambda t: -sh["evidence"][t])
        w(f"## Matched nodes -- `--grep {grep_pattern}` ({len(matched)} match)")
        w("")
        w("_Look up specific nodes: is a HIGH-patient-count deep node still starved "
          "(evidence≈prior floor, frac≈1.0)? If so its topic is a flat-start/deflation "
          "casualty, not a doc-count one._")
        w("")
        if matched:
            emit_table(matched)
        else:
            w("_(no node label matched)_")
            w("")

    # words render for both ends + grep matches, de-duplicated, main order first
    seen = set()
    word_order = [t for t in list(order) + list(matched) + list(best_order)
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

    # ---- per-parent sibling redundancy (needs parent_int) ----
    if redundancy and parent_int is not None:
        rows = sibling_redundancy(parent_int, topic2engine, lams, sh, n_bg, K)
        scored = [r for r in rows if r["n_fed"] >= 2]
        w("## Sibling redundancy -- per-parent child differentiation")
        w("")
        w("For each parent, cosine of its children's multi-domain topic vectors. "
          "Restricted to FED children (support_frac<0.5) -- starved children are "
          "trivially uniform. **High median = the parent's fed children collapse to "
          "ONE topic (capacity starvation at that parent, the tpn=1 lever); a spread "
          "= healthy heterogeneity.** Partial redundancy is fine; watch UNIFORM "
          "collapse, especially at wide parents.")
        w("")
        if not scored:
            w("_(no parent has ≥2 fed children — nothing to compare; the fed nodes "
              "are too shallow/sparse under this fit.)_")
            w("")
        else:
            n_collapse = sum(1 for r in scored if r["med_cos_fed"] > 0.8)
            fanouts = np.array([r["fanout"] for r in scored], dtype=float)
            meds = np.array([r["med_cos_fed"] for r in scored], dtype=float)
            corr = (float(np.corrcoef(fanouts, meds)[0, 1])
                    if len(scored) > 2 and fanouts.std() > 0 else float("nan"))
            w(f"- {len(scored)} parents with ≥2 fed children; "
              f"**{n_collapse} show uniform collapse (median fed-cosine > 0.8)**; "
              f"corr(fan-out, median fed-cosine) = {corr:.2f}")
            w("")
            w("| parent | fanout | n fed | median fed-cos | max fed-cos |")
            w("|---|--:|--:|--:|--:|")
            for r in sorted(scored, key=lambda r: -r["med_cos_fed"])[:redundancy]:
                w(f"| {nnames.get(r['parent'], 'cid:'+str(r['parent']))} | "
                  f"{r['fanout']} | {r['n_fed']} | {r['med_cos_fed']:.2f} | "
                  f"{r['max_cos_fed']:.2f} |")
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


def _trunc(s, n):
    """Truncate to n chars without leaving a dangling '[unit' fragment (lab
    names like 'Diastolic blood pressure [measured]' otherwise cut mid-bracket)."""
    if len(s) <= n:
        return s
    s = s[:n].rstrip()
    if "[" in s and "]" not in s.rsplit("[", 1)[-1]:
        s = s[:s.rfind("[")].rstrip()
    return s + "…"


def _digest_words(t, lams, sh, inv_maps, names, name_by_id, dom_names, *,
                  k=6, maxlen=40, profile_idx=None):
    """One compact line summarising a topic as a PHENOTYPE SIGNATURE: the
    condition-domain top-k names, then the top-3 drug names after `//`. Leads
    with conditions (the disease identity) rather than the dominant domain,
    because on these disease nodes the highest-MASS domain is often measurement
    -- generic lab panels that read as noise in a compact view -- while the
    interpretable signal a reader wants is diagnoses + drugs. Falls back to the
    dominant domain only when there is no condition domain. Probabilities are
    dropped and names truncated; the verbose report keeps the full per-domain
    mass breakdown. Empty when no vocab map is supplied."""
    if not inv_maps:
        return ""

    cond_d = next((i for i, n in enumerate(dom_names) if "condition" in n.lower()),
                  None)

    def render(d, kk, mark=False):
        inv = inv_maps[d] if d < len(inv_maps) else None
        tw = top_words(np.asarray(lams[d][t]), inv, names, name_by_id, kk)
        # `*` marks a token in the node's own HPO profile (condition domain
        # only) — so profile tokens vs EMERGENT co-riders are one glance apart.
        return [_trunc(nm, maxlen) + ("*" if mark and profile_idx is not None
                                      and i in profile_idx else "")
                for nm, _, i in tw]

    lead = cond_d if cond_d is not None else int(np.argmax(sh["dom_mass"][t]))
    main = render(lead, k, mark=(lead == cond_d))
    parts = "·".join(main) if main else "(flat)"
    drug_d = next((i for i, n in enumerate(dom_names) if "drug" in n.lower()), None)
    if drug_d is not None and drug_d != lead:
        dr = render(drug_d, 3)
        if dr:
            parts += " // " + "·".join(dr)
    return parts


def build_digest(run_dir, *, exemplars=8, t_words=6, bundle_meta_path=None,
                 vocab_path=None, names_path=None, readout_label="gated_pc",
                 grep_pattern=None, redundancy=False, profile_file=None):
    """A COMPACT single-block digest -- the copy-paste-to-chat view.

    Same inputs as build_report, but emits only: a one-line header (K / starved%
    / evidence spread / decoder), the sharpness-by-depth table with an auto
    cliff-marker, the top-`exemplars` best-fed topics (one line each, compact
    words), an optional `--grep` table (one line per match), and an optional
    one-line redundancy summary. No per-topic probability dumps, no background
    section, no borrowed-topics columns -- a few hundred tokens, not thousands.
    """
    npz, manifest = load_run(run_dir)
    lams = domain_lambdas(npz)
    K = int(manifest["K"]); C = int(manifest["C"])
    n_bg = int(manifest["n_bg"]); tpn = int(manifest["tpn"])
    dom_names = manifest.get("domain_names") or [f"dom{i}" for i in range(len(lams))]
    labels, topic2engine = topic_labels(manifest)
    sh = topic_sharpness(lams)
    name_by_id = {int(k): v for k, v in
                  manifest.get("corpus_manifest", {}).get("name_by_id", {}).items()}

    heads = load_readout_heads(run_dir, readout_label)
    decoder_src = (heads["src"] if heads is not None
                   else "co-fit w_CK (no readout heads sidecar)")

    # bundle meta -> depth + words (same guard-lite path as build_report)
    depths = parent_int = inv_maps = names = None
    meta = load_bundle_meta(bundle_meta_path) if bundle_meta_path else None
    if meta:
        if "parent_int" in meta:
            depths = node_depths(meta["parent_int"])
            parent_int = {int(c): [int(p) for p in ps]
                          for c, ps in meta["parent_int"].items()}
        if "name_by_id" in meta:
            name_by_id = {int(k): v for k, v in meta["name_by_id"].items()} or name_by_id
        if not vocab_path and "vocab_maps" in meta:
            inv_maps = [{int(idx): int(cid) for cid, idx in vm.items()}
                        for vm in meta["vocab_maps"]]
    if vocab_path:
        inv_maps = load_vocab_maps(vocab_path, len(lams))
    if names_path:
        names = load_concept_names(names_path)

    def depth_of(t):
        eng = topic2engine[t]
        return depths.get(eng, -1) if (depths and eng is not None) else -1

    # HPO-profile marker (`*` on a top word that is in the node's own positive
    # profile — insight 0084's legibility read, inline): needs the emit-eta TSV
    # and the meta's condition vocab map. Profile tokens vs EMERGENT co-riders
    # then read apart at a glance; the co-riders are the discovery signal.
    prof = None
    if profile_file and meta and "vocab_maps" in meta:
        vm0 = {str(kk): int(v) for kk, v in meta["vocab_maps"][0].items()}
        prof = _profile_vocab_sets(profile_file, manifest, vm0)

    def words(t):
        eng = topic2engine[t]
        pidx = prof.get(eng) if (prof and eng is not None) else None
        wl = _digest_words(t, lams, sh, inv_maps, names, name_by_id, dom_names,
                           k=t_words, profile_idx=pidx)
        return (" | " + wl) if wl else ""

    def line(t):
        dd = f"d{depth_of(t)} " if depths else ""
        st = "STARVED" if sh["support_frac"][t] > 0.5 else "fed"
        return (f"  {(labels[t] or '')[:34].ljust(34)} {dd}"
                f"ev{sh['evidence'][t]:.3g} f{sh['support_frac'][t]:.2f} {st}"
                f"{words(t)}")

    fg = np.arange(n_bg, K)
    frac = sh["support_frac"][fg]
    ev = sh["evidence"][fg]
    starved = int(np.sum(frac > 0.5))
    q = np.percentile(ev, [50, 90])

    L = []
    w = L.append
    w(f"# {Path(run_dir).name} — digest")
    w(f"K={K} ({n_bg} bg + {K - n_bg} node, tpn={tpn}) · C={C} · "
      f"{100 * starved / max(fg.size, 1):.0f}% starved (frac>0.5) · "
      f"ev min {ev.min():.3g} / med {q[0]:.3g} / p90 {q[1]:.3g} / max {ev.max():.3g}")
    w(f"decoder: {decoder_src}")
    if prof:
        w(f"`*` = token in the node's own HPO profile "
          f"({Path(profile_file).name}; {len(prof)} profiled nodes) — "
          f"unmarked condition terms are EMERGENT co-riders")
    w("")

    # depth rollup with an auto cliff-marker at the first depth whose median frac
    # crosses 0.5 (fed -> starved) -- the one line that answers the depth question.
    if depths:
        by_d = {}
        for t in fg:
            by_d.setdefault(depth_of(t), []).append(t)
        floor = ev.min()
        cliff = None
        w("depth ·   n · med-ev · med-frac")
        for dep in sorted(k for k in by_d if k >= 0):
            ts = by_d[dep]
            mfrac = float(np.median(sh["support_frac"][ts]))
            if cliff is None and mfrac > 0.5:
                cliff = dep
            mark = "  <- cliff" if dep == cliff else ""
            w(f"{dep:>5} · {len(ts):>3} · {np.median(sh['evidence'][ts]):>6.3g} · "
              f"{mfrac:.2f}{mark}")
        if cliff is not None:
            w(f"verdict: fed through depth {cliff - 1}; depth>={cliff} median at "
              f"prior floor (~{floor:.3g}).")
        w("")

    order = sorted(fg, key=lambda t: -sh["evidence"][t])[:exemplars]
    w(f"fed exemplars (top {len(order)} by ev):")
    for t in order:
        w(line(t))
    w("")

    if grep_pattern:
        rx = re.compile(grep_pattern, re.I)
        matched = sorted((t for t in fg if rx.search(labels[t] or "")),
                         key=lambda t: -sh["evidence"][t])
        w(f"grep '{grep_pattern}' — {len(matched)} match"
          + (f" (top 25 by ev)" if len(matched) > 25 else "") + ":")
        for t in matched[:25]:
            w(line(t))
        if not matched:
            w("  (no node label matched)")
        w("")

    if redundancy and parent_int is not None:
        rows = sibling_redundancy(parent_int, topic2engine, lams, sh, n_bg, K)
        scored = [r for r in rows if r["n_fed"] >= 2]
        if scored:
            nnames = node_names(manifest)
            n_col = sum(1 for r in scored if r["med_cos_fed"] > 0.8)
            worst = max(scored, key=lambda r: r["med_cos_fed"])
            w(f"redundancy: {len(scored)} parents >=2 fed · {n_col} collapsed "
              f"(med fed-cos>0.8) · worst "
              f"{nnames.get(worst['parent'], worst['parent'])} "
              f"{worst['med_cos_fed']:.2f} (max {worst['max_cos_fed']:.2f})")
        else:
            w("redundancy: no parent has >=2 fed children")
        w("")

    return "\n".join(L).rstrip() + "\n"


def _credited_engine_ids(credited_file, manifest):
    """Engine ids of the nodes named in a TSV's `mondo_id` column.

    ``credited_file`` is typically the probe's ``--emit-eta`` table (the fit's
    own credited set — plan D5's internal-control split for exp 0116), but any
    TSV with a `mondo_id` column works. Curie -> engine id mirrors
    `mondo_native_dag.mondo_cid` (the numeric part of `MONDO:%07d`; stable by
    construction) so this off-cluster tool needs no extra imports, then engine
    ids come from the manifest's int2cid — so the slice matches exactly the
    nodes the fit driver credited. Non-Mondo rows and nodes outside this run's
    DAG are silently skipped (they were never credited here either)."""
    header = None
    cids = set()
    with open(credited_file) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if header is None:
                header = parts
                if "mondo_id" not in header:
                    raise SystemExit(
                        f"[inspect_topics] {credited_file} has no mondo_id "
                        f"column (columns: {header})")
                mi = header.index("mondo_id")
                continue
            s = str(parts[mi])
            if s.startswith("MONDO:") and s[len("MONDO:"):].isdigit():
                cids.add(int(s[len("MONDO:"):]))
    _, int2cid = node_order(manifest)
    return {e for e, c in int2cid.items() if c in cids}


def _profile_vocab_sets(profile_file, manifest, vocab_map0):
    """{engine id: frozenset of CONDITION-domain vocab indices} of each profiled
    node's POSITIVE profile concepts (neg rows excluded — a NOT term is not what
    the topic should look like). ``profile_file`` is the probe's --emit-eta TSV
    (mondo_id, concept_id, weight, neg, coverage); ``vocab_map0`` is the bundle
    meta's condition-domain {concept_id: idx}. Concepts outside the vocab are
    dropped (they cannot appear in a topic either)."""
    header, rows = None, {}
    with open(profile_file) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if header is None:
                header = parts
                for col in ("mondo_id", "concept_id", "neg"):
                    if col not in header:
                        raise SystemExit(
                            f"[inspect_topics] {profile_file} has no {col} "
                            f"column (columns: {header}) — --profile-align "
                            f"needs the probe's --emit-eta TSV")
                mi, ci, ni = (header.index(c)
                              for c in ("mondo_id", "concept_id", "neg"))
                continue
            if str(parts[ni]).strip().lower() in ("1", "true"):
                continue
            s = str(parts[mi])
            if not (s.startswith("MONDO:") and s[len("MONDO:"):].isdigit()):
                continue
            idx = vocab_map0.get(str(parts[ci])) or vocab_map0.get(
                int(parts[ci]) if str(parts[ci]).isdigit() else -1)
            if idx is not None:
                rows.setdefault(int(s[len("MONDO:"):]), set()).add(int(idx))
    _, int2cid = node_order(manifest)
    return {e: frozenset(rows[c]) for e, c in int2cid.items() if c in rows}


def _align_scores(lam0, topic, prof_idx, top_m):
    """(profile mass, top-m overlap) of one topic against one profile set.

    Mass = sum of E[beta] (the topic's normalized condition-domain row) on the
    profile indices; overlap = fraction of the topic's top-m tokens that are
    profile tokens. Both in [0,1]; the flat-topic baseline for mass is
    |profile| / V_0 (printed for context by the caller)."""
    row = np.asarray(lam0[topic], dtype=np.float64)
    s = row.sum()
    beta = row / s if s > 0 else row
    idx = np.fromiter(prof_idx, dtype=np.int64)
    mass = float(beta[idx].sum())
    top = np.argsort(beta)[::-1][:top_m]
    overlap = float(len(set(int(i) for i in top) & prof_idx) / top_m)
    return mass, overlap


def build_profile_align(run_dir, profile_file, *, bundle_meta_path,
                        compare_dir=None, top_m=15, starved_frac=0.5):
    """Compact profile-ALIGNMENT scorecard: does each credited node's BOOSTED
    topic look like its HPO profile? (The quantitative form of 0116's eyeball
    finding that starved deep topics became phenotype-legible — insight 0084.)

    Per credited node, scores the FIRST topic of its block (the boosted one
    under `profile_eta_topics: 1`) against the node's positive profile tokens:
    E[beta] mass on the profile and top-`top_m` overlap. Pools medians over
    STARVED (support_frac > `starved_frac`, the digest's flatness read) vs FED
    topics, and — with ``compare_dir`` — recomputes the SAME topics/profiles on
    a baseline run so the alignment delta is paired. ~10 lines instead of a
    pasted digest; model params and counts-of-topics only (egress-safe)."""
    run_dir = Path(run_dir)
    npz, manifest = load_run(run_dir)
    meta = load_bundle_meta(bundle_meta_path) if bundle_meta_path else None
    if not meta or "vocab_maps" not in meta:
        raise SystemExit("[inspect_topics] --profile-align needs the bundle "
                         "meta (INSPECT_KEY auto-discovery or --bundle-meta)")
    vm0 = {str(k): int(v) for k, v in meta["vocab_maps"][0].items()}
    v0 = len(vm0)
    prof = _profile_vocab_sets(profile_file, manifest, vm0)
    if not prof:
        raise SystemExit("[inspect_topics] no profiled node maps into this "
                         "run's DAG/vocab — wrong TSV or wrong bundle?")
    nodes, _ = node_order(manifest)
    n_bg, tpn = int(manifest["n_bg"]), int(manifest["tpn"])
    boosted_topic = {e: n_bg + i * tpn for i, e in enumerate(nodes)}
    lams = domain_lambdas(npz)
    sh = topic_sharpness(lams)
    nnames = node_names(manifest)

    base_lam0 = None
    if compare_dir:
        cmp_dir = Path(resolve_run_dir(str(compare_dir)))
        base_lam0 = domain_lambdas(load_run(cmp_dir)[0])[0]

    rows = []
    for e, pidx in sorted(prof.items()):
        t = boosted_topic.get(e)
        if t is None:
            continue
        mass, ov = _align_scores(lams[0], t, pidx, top_m)
        r = {"eng": e, "topic": t, "n_prof": len(pidx),
             "starved": bool(sh["support_frac"][t] > starved_frac),
             "mass": mass, "overlap": ov, "flat_mass": len(pidx) / v0}
        if base_lam0 is not None:
            r["mass_b"], r["overlap_b"] = _align_scores(base_lam0, t, pidx,
                                                        top_m)
        rows.append(r)

    def _med(vs):
        return sorted(vs)[len(vs) // 2] if vs else float("nan")

    L = [f"# profile alignment — {run_dir.name} · {len(rows)} credited "
         f"node(s) scored · boosted topic vs its positive profile "
         f"(top-{top_m} overlap; mass = E[beta] on profile tokens; flat "
         f"baseline ≈ {_med([r['flat_mass'] for r in rows]):.4f})"]
    for tag in ("starved", "fed"):
        grp = [r for r in rows if r["starved"] == (tag == "starved")]
        if not grp:
            L.append(f"{tag}: 0 topics")
            continue
        line = (f"{tag}: n={len(grp)} median mass="
                f"{_med([r['mass'] for r in grp]):.3f} "
                f"overlap@{top_m}={_med([r['overlap'] for r in grp]):.2f}")
        if base_lam0 is not None:
            line += (f"  |  baseline mass={_med([r['mass_b'] for r in grp]):.3f} "
                     f"overlap={_med([r['overlap_b'] for r in grp]):.2f}")
        L.append(line)
    worst = sorted(rows, key=lambda r: r["overlap"])[:3]
    best = sorted(rows, key=lambda r: -r["overlap"])[:3]
    L.append("least aligned: " + "; ".join(
        f"{nnames.get(r['eng'], r['eng'])} ov={r['overlap']:.2f}" for r in worst))
    L.append("most aligned:  " + "; ".join(
        f"{nnames.get(r['eng'], r['eng'])} ov={r['overlap']:.2f}" for r in best))
    return "\n".join(L) + "\n"


def build_auc_slice(run_dir, *, bundle_meta_path=None, grep_pattern=None,
                    arm="gated_pc", credited_file=None, compare_dir=None):
    """Compact per-node readout-AUC slice from `results_readout.json`.

    Answers "which nodes does the readout rank WELL or BADLY?" off-cluster —
    e.g. whether the 0114/0115 anchor-misaligned node topics (grep their names)
    underperform the branch, which is the misalignment-cost question the 0114
    readout was run for. Reports AUC/AP and node counts ONLY; the per-node
    positive counts in the JSON stay in the run dir (egress floor) — nothing
    printed here is a patient count.

    ``credited_file`` (a TSV with a `mondo_id` column, e.g. the probe's
    --emit-eta table) splits every view into CREDITED vs UNCREDITED scored
    nodes, and ``compare_dir`` (a baseline run dir with the same arm in its
    results_readout.json) adds PAIRED per-node AUC deltas on the shared scored
    nodes — together they are exp 0116's pre-registered primary read: credited
    nodes up, uncredited (the internal control) ~0. Deltas are AUC arithmetic
    on already-disclosable per-node AUCs plus counts of nodes; nothing new is
    disclosed.
    """
    run_dir = Path(run_dir)
    res = json.loads((run_dir / "results_readout.json").read_text())
    if arm not in res:
        raise SystemExit(f"[inspect_topics] no arm {arm!r} in results_readout.json "
                         f"(has: {sorted(res)}); run gated-pc-readout first")
    per_node = {int(k): v for k, v in (res[arm].get("per_node") or {}).items()}
    if not per_node:
        raise SystemExit(f"[inspect_topics] arm {arm!r} carries no per_node block")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    nnames = node_names(manifest)
    depths = None
    meta = load_bundle_meta(bundle_meta_path) if bundle_meta_path else None
    if meta and meta.get("parent_int"):
        depths = node_depths(meta["parent_int"])

    aucs = {c: float(d["auc"]) for c, d in per_node.items()}
    vals = sorted(aucs.values())

    def _q(v, p):
        return v[min(len(v) - 1, int(p * len(v)))]

    def _fmt(d, keys):
        return " ".join(f"{k}={d[k]:.4f}" if isinstance(d.get(k), float)
                        else f"{k}={d.get(k)}" for k in keys if d.get(k) is not None)

    L = [f"# readout AUC slice — {run_dir.name} · arm={arm} · "
         f"{len(vals)} scored node(s)"]
    # Recall the recorded macro lines too, so this one command recovers a
    # readout whose terminal output is gone (results_readout.json is durable
    # precisely for that — see run_readout's docstring).
    rk = res[arm].get("ranking") or {}
    det = res[arm].get("detection") or {}
    if rk:
        L.append(f"recorded macro ranking: {_fmt(rk, ('auc', 'ap', 'n_nodes'))}")
    if det:
        L.append(f"recorded detection: {_fmt(det, ('auc', 'ap', 'prev', 'n'))}")
    L += [f"AUC quantiles: p10={_q(vals, .10):.3f} p25={_q(vals, .25):.3f} "
          f"median={_q(vals, .50):.3f} p75={_q(vals, .75):.3f} "
          f"p90={_q(vals, .90):.3f}", ""]

    credited = (_credited_engine_ids(credited_file, manifest)
                if credited_file else None)

    def _split(ids):
        cred = sorted(aucs[c] for c in ids if c in credited)
        uncr = sorted(aucs[c] for c in ids if c not in credited)
        return cred, uncr

    if credited is not None:
        cred, uncr = _split(aucs)
        L.append(f"## credited split ({Path(credited_file).name}) — "
                 f"{len(cred)} credited / {len(uncr)} uncredited scored node(s)")
        for tag, v in (("credited", cred), ("uncredited", uncr)):
            if v:
                L.append(f"{tag}: median AUC={_q(v, .5):.3f} "
                         f"(p25={_q(v, .25):.3f} p75={_q(v, .75):.3f})")
        L.append("")

    if compare_dir:
        # The baseline only needs results_readout.json (no fit npz required
        # for a paired compare), so resolve leniently: a directory that
        # carries it is taken as-is; otherwise fall through to the normal
        # run-dir resolution (IDs / globs).
        cmp_dir = Path(compare_dir)
        if not (cmp_dir.is_dir() and (cmp_dir / "results_readout.json").exists()):
            cmp_dir = Path(resolve_run_dir(str(compare_dir)))
        res_b = json.loads((cmp_dir / "results_readout.json").read_text())
        if arm not in res_b:
            raise SystemExit(f"[inspect_topics] no arm {arm!r} in "
                             f"{cmp_dir}/results_readout.json")
        base = {int(k): float(v["auc"])
                for k, v in (res_b[arm].get("per_node") or {}).items()}
        shared = sorted(set(aucs) & set(base))
        deltas = {c: aucs[c] - base[c] for c in shared}

        def _delta_line(tag, cs):
            dv = sorted(deltas[c] for c in cs)
            if not dv:
                return f"{tag}: 0 shared node(s)"
            up = sum(1 for d in dv if d > 0)
            dn = sum(1 for d in dv if d < 0)
            return (f"{tag}: n={len(dv)} median dAUC={_q(dv, .5):+.4f} "
                    f"mean={sum(dv) / len(dv):+.4f} "
                    f"(p25={_q(dv, .25):+.4f} p75={_q(dv, .75):+.4f}) "
                    f"up/down={up}/{dn}")

        L.append(f"## paired vs {cmp_dir.name} — {len(shared)} shared "
                 f"scored node(s) (this run minus baseline)")
        L.append(_delta_line("all", shared))
        if credited is not None:
            L.append(_delta_line("credited", [c for c in shared
                                              if c in credited]))
            L.append(_delta_line("uncredited (internal control)",
                                 [c for c in shared if c not in credited]))
        if depths:
            by_d: dict = {}
            for c in shared:
                by_d.setdefault(depths.get(c, -1), []).append(deltas[c])
            L.append("by depth (median dAUC): " + "  ".join(
                f"d{d}={_q(sorted(v), .5):+.3f}(n={len(v)})"
                for d, v in sorted(by_d.items())))
        L.append("")
    if depths:
        by_d: dict = {}
        for c, a in aucs.items():
            by_d.setdefault(depths.get(c, -1), []).append(a)
        L += ["| depth | n scored | median AUC | p25 | p75 |", "|---|---|---|---|---|"]
        for d in sorted(by_d):
            v = sorted(by_d[d])
            L.append(f"| {d} | {len(v)} | {_q(v, .5):.3f} | {_q(v, .25):.3f} | "
                     f"{_q(v, .75):.3f} |")
        L.append("")
    if grep_pattern:
        rx = re.compile(grep_pattern, re.IGNORECASE)
        hit = {c: a for c, a in aucs.items() if rx.search(str(nnames.get(c, "")))}
        rest = sorted(a for c, a in aucs.items() if c not in hit)
        L.append(f"## grep {grep_pattern!r} — {len(hit)} matched scored node(s)")
        if hit and rest:
            hv = sorted(hit.values())
            L.append(f"matched median AUC={_q(hv, .5):.3f} vs rest "
                     f"median={_q(rest, .5):.3f}")
        L.append("")
        for c, a in sorted(hit.items(), key=lambda kv: kv[1]):
            ap_ = per_node[c].get("ap")
            dep = f" d{depths.get(c, '?')}" if depths else ""
            L.append(f"- AUC={a:.3f}"
                     f"{'' if ap_ is None else f' AP={float(ap_):.3f}'}{dep} — "
                     f"{nnames.get(c, f'eng:{c}')}")
        L.append("")
    return "\n".join(L).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# --strip-audit: what the vocabulary leakage strip COULD have dropped          #
# --------------------------------------------------------------------------- #
def _node_cid_set(manifest, meta):
    """Every node concept id the run knows: the manifest's post-prune int2cid,
    unioned with the bundle meta's. The PRE-prune `before_dag.nodes()` the
    assembler actually strips over is not persisted, so this is a LOWER bound
    on the strip set — enough to decide the id-space question, which is what
    the audit is for (a Mondo id is a Mondo id before and after pruning)."""
    _, int2cid = node_order(manifest)
    cids = {int(c) for e, c in int2cid.items() if e != 0}
    if meta and "int2cid" in meta:
        cids |= {int(c) for e, c in meta["int2cid"].items() if int(e) != 0}
    return cids


def _profile_concepts(profile_file):
    """{mondo numeric id: set of POSITIVE concept ids} from the emit-eta TSV —
    ALL mapped concepts, in-vocab or not (the TSV is bundle-agnostic)."""
    header, out = None, {}
    with open(profile_file) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if header is None:
                header = parts
                need = ("mondo_id", "concept_id", "neg")
                if any(c not in header for c in need):
                    raise SystemExit(f"[inspect_topics] {profile_file} lacks "
                                     f"{need} (columns: {header})")
                mi, ci, ni = (header.index(c) for c in need)
                continue
            if str(parts[ni]).strip().lower() in ("1", "true"):
                continue
            s = str(parts[mi])
            if not (s.startswith("MONDO:") and s[len("MONDO:"):].isdigit()):
                continue
            if str(parts[ci]).strip().lstrip("-").isdigit():
                out.setdefault(int(s[len("MONDO:"):]), set()).add(int(parts[ci]))
    return out


def build_strip_audit(run_dir, *, bundle_meta_path, profile_file=None):
    """Off-YARN audit of the vocabulary LEAKAGE STRIP on this run's bundle.

    The assembler strips `{vm[c] for c in before_dag.nodes() if c in vm}` from
    every domain's features (`multi_domain.py`, step 6): the label DAG's node
    ids looked up in each domain's `{concept_id: idx}` vocab map. That is only
    a strip when the node ids and the vocab keys share an id space. On the
    anchor Mondo path (`dag_source: mondo`) node ids ARE OMOP concept ids; on
    the native path (`dag_source: mondo_native`, exp 0110+) node ids are the
    Mondo curie's numeric part (`MONDO:0004995` -> 4995 — `mondo_native_dag`'s
    recorded id-space deviation), which an OMOP-keyed vocab map only matches by
    numeric coincidence. This report counts, per domain, how many vocab dims a
    node id actually resolves to — i.e. how many dims the strip could have
    removed — so `strip_mode` in the front matter can be read as what it DID
    rather than what it says. With the emit-eta TSV it also reports how many
    of the credited profiles' concepts sit in the node-id set (would-be
    stripped) vs the condition vocab (live).

    Model/bundle metadata and counts of dims only (egress-safe)."""
    run_dir = Path(run_dir)
    _, manifest = load_run(run_dir)
    meta = load_bundle_meta(bundle_meta_path) if bundle_meta_path else None
    if not meta or "vocab_maps" not in meta:
        raise SystemExit("[inspect_topics] --strip-audit needs the bundle meta "
                         "(INSPECT_KEY auto-discovery or --bundle-meta)")
    cm = manifest.get("corpus_manifest", {})
    dag_source = str(cm.get("dag_source") or manifest.get("dag_source") or "?")
    strip_mode = str(cm.get("strip_mode") or manifest.get("strip_mode") or "?")
    window_mode = str(cm.get("window_mode") or manifest.get("window_mode") or "?")
    index_mode = str(cm.get("index_mode") or "?")
    dom_names = list(manifest.get("domain_names") or
                     [f"dom{m}" for m in range(len(meta["vocab_maps"]))])
    cids = _node_cid_set(manifest, meta)
    native = dag_source == "mondo_native"

    L = [f"# strip audit — {run_dir.name} · dag_source={dag_source} "
         f"strip_mode={strip_mode} window_mode={window_mode} "
         f"index_mode={index_mode} · {len(cids)} node ids (post-prune; the "
         f"assembler strips over the PRE-prune set, so dims below are a lower "
         f"bound)"]
    if native:
        L.append("node ids are Mondo numerics (MONDO:%07d -> int); the strip "
                 "resolves them against OMOP concept-id vocab maps, so a hit "
                 "below is a numeric coincidence, not a disease code")
    else:
        L.append("node ids are OMOP concept ids; hits below are the DAG-node "
                 "codes the strip removed from the features")
    total_hits = 0
    for m, vm in enumerate(meta["vocab_maps"]):
        keys = set()
        for k in vm.keys():
            ks = str(k).strip()
            if ks.lstrip("-").isdigit():
                keys.add(int(ks))
        hits = sorted(cids & keys)
        total_hits += len(hits)
        nm = dom_names[m] if m < len(dom_names) else f"dom{m}"
        frac = len(hits) / len(vm) if vm else float("nan")
        L.append(f"domain {m} ({nm}): {len(hits)} of {len(vm)} vocab dims match a "
                 f"node id ({100 * frac:.3f}%)")
    if profile_file:
        prof = _profile_concepts(profile_file)
        vm0 = set()
        for k in meta["vocab_maps"][0].keys():
            if str(k).strip().lstrip("-").isdigit():
                vm0.add(int(str(k).strip()))
        allc = set().union(*prof.values()) if prof else set()
        in_node = allc & cids
        in_vocab = allc & vm0
        L.append(f"profile (emit-eta, {len(prof)} nodes, {len(allc)} distinct "
                 f"positive concepts): {len(in_vocab)} in the condition vocab "
                 f"(live), {len(in_node)} equal to a node id (would be stripped), "
                 f"{len(allc - in_vocab - in_node)} in neither")
    if native and total_hits <= max(3, len(cids) // 100):
        L.append("VERDICT: the vocabulary strip is effectively a NO-OP on this "
                 "run — the disease's own condition codes stay in the features "
                 "wherever the window admits them; the leakage protection "
                 "actually in force is the pre-index feature window "
                 f"(window_mode={window_mode}) plus incident eligibility at eval")
    elif native:
        L.append(f"VERDICT: {total_hits} coincidental dims stripped — inspect "
                 "them before trusting anything else in this report")
    else:
        L.append(f"VERDICT: {total_hits} DAG-node dims stripped across domains")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# --collinearity: are the credited (profile-boosted) topics near-duplicates?   #
# --------------------------------------------------------------------------- #
def _ancestors(parent_int, e):
    """All proper ancestors of engine node `e` (root 0 excluded), walking the
    meta's `{child: [parents]}` map. Empty when the map is absent."""
    out, stack = set(), [e]
    while stack:
        c = stack.pop()
        for p in (parent_int or {}).get(int(c), []):
            p = int(p)
            if p != 0 and p not in out:
                out.add(p)
                stack.append(p)
    return out


def _descendants(parent_int, e):
    """All proper descendants of engine node `e` under the meta's
    `{child: [parents]}` map. Empty when the map is absent."""
    children = {}
    for c, ps in (parent_int or {}).items():
        for p in ps:
            children.setdefault(int(p), []).append(int(c))
    out, stack = set(), [e]
    while stack:
        c = stack.pop()
        for k in children.get(int(c), []):
            if k not in out:
                out.add(k)
                stack.append(k)
    return out


def _uniform_unit_vec(lams):
    """The `_topic_unit_vec` of a perfectly FLAT topic: each domain's row is
    uniform (1/V_m), concatenated and L2-normalized. Cosine against it is the
    'flatness floor' — two starved topics agree because both are ~this."""
    parts = [np.full(lam.shape[1], 1.0 / lam.shape[1]) for lam in lams]
    v = np.concatenate(parts)
    return v / np.linalg.norm(v)


def _jaccard(a, b):
    u = len(a | b)
    return len(a & b) / u if u else float("nan")


def build_collinearity(run_dir, profile_file, *, bundle_meta_path,
                       readout_label="gated_pc", starved_frac=0.5,
                       n_compare=40, seed=0):
    """Does the profile prior make the credited nodes' boosted topics into
    near-duplicates of each other and of the shared/ancestor topics — and does
    the readout decoder then route around them? Insight 0087's `self-w ≈ 0`
    read, quantified and given a control group.

    Four pooled sections, credited vs uncredited (the latter split fed/starved
    by the boosted topic's support_frac, the digest's flatness read):

      profiles   pairwise Jaccard of the credited nodes' in-vocab positive
                 profile token sets (max and median vs the other credited
                 nodes): near-1 = the prior is pinning many nodes to the SAME
                 words, so their topics cannot be distinct whatever the fit does.
      topics     cosine of each node's boosted topic (all-domain E[beta], as in
                 `sibling_redundancy`) vs: the other group members (max), the
                 background topics (max), its own ancestors' topics (max), and
                 the flat topic (the starvation floor — read the others against
                 it; a starved topic is trivially ~1 to another starved topic).
      decoder    from the readout heads (standardized W_std when the checkpoint
                 is on disk, else raw V): the share of |w| a node's head puts on
                 its OWN block vs the background vs its ancestors' blocks. A
                 median own-share near 0 with a high ancestor/background share is
                 the mechanical form of 'legible topic, unused topic'.
      evidence   the boosted topic's lambda mass vs the prior floor (the
                 minimum over all topics): a topic at the floor absorbed no
                 data, so its theta cannot vary between documents and no decoder
                 could load on it however legible it reads.

    `n_compare` bounds the pairwise work (each node is compared against up to
    that many sampled peers of its group, same for both groups). Model params
    and counts of nodes only (egress-safe)."""
    run_dir = Path(run_dir)
    npz, manifest = load_run(run_dir)
    meta = load_bundle_meta(bundle_meta_path) if bundle_meta_path else None
    if not meta or "vocab_maps" not in meta:
        raise SystemExit("[inspect_topics] --collinearity needs the bundle meta "
                         "(INSPECT_KEY auto-discovery or --bundle-meta)")
    vm0 = {str(k): int(v) for k, v in meta["vocab_maps"][0].items()}
    prof = _profile_vocab_sets(profile_file, manifest, vm0)
    credited = _credited_engine_ids(profile_file, manifest)
    parent_int = ({int(c): [int(p) for p in ps]
                   for c, ps in meta["parent_int"].items()}
                  if "parent_int" in meta else None)
    nodes, _ = node_order(manifest)
    n_bg, tpn = int(manifest["n_bg"]), int(manifest["tpn"])
    K = int(manifest["K"])
    block = {e: list(range(n_bg + i * tpn, n_bg + (i + 1) * tpn))
             for i, e in enumerate(nodes)}
    lams = domain_lambdas(npz)
    sh = topic_sharpness(lams)
    floor = float(np.min(sh["evidence"])) if K else float("nan")
    rng = np.random.default_rng(seed)

    cred = sorted(e for e in nodes if e in credited)
    uncred = [e for e in nodes if e not in credited]
    fed = [e for e in uncred if sh["support_frac"][block[e][0]] <= starved_frac]
    starved = [e for e in uncred if sh["support_frac"][block[e][0]] > starved_frac]
    groups = [("credited", cred), ("uncredited fed", fed),
              ("uncredited starved", starved)]

    vec_cache = {}

    def vec(t):
        if t not in vec_cache:
            vec_cache[t] = _topic_unit_vec(t, lams)
        return vec_cache[t]

    flat = _uniform_unit_vec(lams)
    bg_vecs = [vec(t) for t in range(n_bg)]

    def _med(vs):
        vs = [v for v in vs if v == v]           # drop NaN
        return float(np.median(vs)) if vs else float("nan")

    heads = load_readout_heads(run_dir, readout_label)
    W = np.abs(heads["W_load"]) if heads else None
    degen = heads.get("degenerate") if heads else None

    def group_stats(members):
        peers_pool = list(members)
        s = {"cos_peer": [], "cos_bg": [], "cos_anc": [], "cos_flat": [],
             "own": [], "bg": [], "anc": [], "desc": [], "other": [], "top10": [],
             "own_fed": [], "other_fed": [], "top10_fed": [], "n_fed_topics": 0,
             "top5_rel": {"own": 0, "anc": 0, "desc": 0, "bg": 0, "other": 0},
             "ev": [], "n_anc": 0, "n_dec": 0}
        for e in members:
            t = block[e][0]
            v = vec(t)
            others = [o for o in peers_pool if o != e]
            if len(others) > n_compare:
                others = list(rng.choice(others, size=n_compare, replace=False))
            if others:
                s["cos_peer"].append(max(float(v @ vec(block[o][0]))
                                         for o in others))
            if bg_vecs:
                s["cos_bg"].append(max(float(v @ b) for b in bg_vecs))
            anc = _ancestors(parent_int, e) if parent_int else set()
            anc_topics = [tt for a in anc for tt in block.get(a, [])]
            if anc_topics:
                s["n_anc"] += 1
                s["cos_anc"].append(max(float(v @ vec(tt)) for tt in anc_topics))
            s["cos_flat"].append(float(v @ flat))
            s["ev"].append(float(sh["evidence"][t]))
            if W is not None and e < W.shape[0] and not (
                    degen is not None and bool(degen[e])):
                row = W[e]
                tot = float(row.sum())
                if tot > 0:
                    s["n_dec"] += 1
                    desc = _descendants(parent_int, e) if parent_int else set()
                    desc_topics = [tt for d in desc for tt in block.get(d, [])]
                    own_s = float(row[block[e]].sum()) / tot
                    bg_s = float(row[:n_bg].sum()) / tot
                    anc_s = float(row[anc_topics].sum()) / tot if anc_topics else 0.0
                    desc_s = (float(row[desc_topics].sum()) / tot
                              if desc_topics else 0.0)
                    s["own"].append(own_s)
                    s["bg"].append(bg_s)
                    s["anc"].append(anc_s)
                    s["desc"].append(desc_s)
                    s["other"].append(max(0.0, 1.0 - own_s - bg_s - anc_s - desc_s))
                    order = np.argsort(-row)
                    s["top10"].append(float(row[order[:10]].sum()) / tot)
                    # FED-topics-only view: ~K starved topics are near-constant
                    # and, standardized, become unit-variance NOISE features
                    # that soak up |w| mass; restricting to topics with data
                    # (support_frac <= starved_frac) asks where the weight
                    # sits among features that could carry signal.
                    fed_mask = sh["support_frac"] <= starved_frac
                    fed_mask[:n_bg] = True
                    rf = row * fed_mask
                    tf = float(rf.sum())
                    if tf > 0:
                        s["n_fed_topics"] = int(fed_mask.sum())
                        s["own_fed"].append(float(rf[block[e]].sum()) / tf)
                        s["other_fed"].append(max(0.0, 1.0 - (
                            float(rf[block[e]].sum()) + float(rf[:n_bg].sum())
                            + (float(rf[anc_topics].sum()) if anc_topics else 0.0)
                            + (float(rf[desc_topics].sum()) if desc_topics else 0.0)
                        ) / tf))
                        s["top10_fed"].append(
                            float(np.sort(rf)[::-1][:10].sum()) / tf)
                    own_set, anc_set = set(block[e]), set(anc_topics)
                    desc_set = set(desc_topics)
                    for tt in order[:5]:
                        tt = int(tt)
                        rel = ("bg" if tt < n_bg else "own" if tt in own_set
                               else "anc" if tt in anc_set
                               else "desc" if tt in desc_set else "other")
                        s["top5_rel"][rel] += 1
        return s

    L = [f"# collinearity — {run_dir.name} · credited={len(cred)} "
         f"uncredited fed={len(fed)} starved={len(starved)} (boosted topic "
         f"support_frac > {starved_frac} = starved) · K={K} n_bg={n_bg} tpn={tpn}"]

    # 1. profile overlap among credited nodes
    pe = [e for e in cred if e in prof]
    if len(pe) >= 2:
        mx, md = [], []
        for e in pe:
            js = [_jaccard(prof[e], prof[o]) for o in pe if o != e]
            mx.append(max(js))
            md.append(float(np.median(js)))
        counts = {}
        for e in pe:
            for idx in prof[e]:
                counts[idx] = counts.get(idx, 0) + 1
        shared = sum(1 for c in counts.values() if c >= max(2, len(pe) // 2))
        L.append(f"profiles: {len(pe)} credited nodes with in-vocab tokens · "
                 f"pairwise Jaccard vs other credited: median-of-max="
                 f"{_med(mx):.2f} median-of-median={_med(md):.2f} · "
                 f"{shared} tokens sit in >= half of the profiles · median "
                 f"profile size {int(np.median([len(prof[e]) for e in pe]))}")
    else:
        L.append("profiles: fewer than 2 credited nodes map into this vocab")

    # 2-4. per group
    L.append("topics: max cosine of the boosted topic vs [group peers | "
             "background | own ancestors] and vs the FLAT topic (floor)")
    stats = {}
    for name, members in groups:
        if not members:
            L.append(f"  {name}: n=0")
            continue
        s = group_stats(members)
        stats[name] = s
        L.append(f"  {name}: n={len(members)} peer={_med(s['cos_peer']):.2f} "
                 f"bg={_med(s['cos_bg']):.2f} anc={_med(s['cos_anc']):.2f} "
                 f"(n_anc={s['n_anc']}) flat={_med(s['cos_flat']):.2f}")
    if W is not None:
        scale = ("standardized W_std" if heads.get("standardized")
                 else "raw-θ V (INFLATED)")
        L.append(f"decoder ({scale}): median share of |w| on [own block | "
                 "background | ancestors' | descendants' | other nodes' blocks], "
                 "top10 = share held by the 10 largest |w|; own<0.05 = head "
                 "ignores its own topic")
        if not heads.get("standardized"):
            L.append("  INCONCLUSIVE: only the raw-θ decoder V is on disk, and "
                     "V = W_std/sd explodes on low-variance (starved) topics, so "
                     "the |w| mass below is dominated by ~constant topics, not "
                     "by what the head uses. Re-run the readout under the "
                     "W_std-in-heads code (or keep the solver checkpoint) for "
                     "an honest read.")
        for name, members in groups:
            s = stats.get(name)
            if not s or not s["own"]:
                L.append(f"  {name}: no fittable heads")
                continue
            low = sum(1 for o in s["own"] if o < 0.05)
            L.append(f"  {name}: n={s['n_dec']} own={_med(s['own']):.2f} "
                     f"bg={_med(s['bg']):.2f} anc={_med(s['anc']):.2f} "
                     f"desc={_med(s['desc']):.2f} other={_med(s['other']):.2f} "
                     f"top10={_med(s['top10']):.2f} · own<0.05: {low}/{s['n_dec']}")
            if s["own_fed"]:
                L.append(f"    among FED topics only ({s['n_fed_topics']} of {K}): "
                         f"own={_med(s['own_fed']):.2f} other={_med(s['other_fed']):.2f} "
                         f"top10={_med(s['top10_fed']):.2f}")
            rel = s["top5_rel"]
            tot5 = max(1, sum(rel.values()))
            L.append("    top-5 loaded topics by relation: " + " ".join(
                f"{k}={100 * v / tot5:.0f}%" for k, v in rel.items()))
    else:
        L.append("decoder: no readout heads/checkpoint on disk (run "
                 "gated-pc-readout first)")
    L.append(f"evidence: boosted-topic lambda mass, median per group vs the "
             f"prior floor {floor:.1f} (min over all topics)")
    for name, members in groups:
        s = stats.get(name)
        if not s:
            continue
        at_floor = sum(1 for v in s["ev"] if v <= 1.05 * floor)
        L.append(f"  {name}: median={_med(s['ev']):.1f} · at floor (<=1.05x): "
                 f"{at_floor}/{len(s['ev'])}")
    return "\n".join(L) + "\n"



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
    ap.add_argument("--redundancy", type=int, default=0, metavar="N",
                    help="Add a SIBLING-REDUNDANCY section: per-parent cosine among "
                         "its fed children's topics (needs --bundle-meta parent_int); "
                         "flags parents whose children uniformly collapse. Shows the "
                         "top N most-collapsed parents. Try --redundancy 30.")
    ap.add_argument("--grep", default=None, metavar="REGEX",
                    help="Look up specific node topics by a case-insensitive regex on "
                         "their name (evidence + depth + words for each match). E.g. "
                         "--grep 'ischemic stroke|hemorrhoid|varicella'.")
    ap.add_argument("--digest", action="store_true",
                    help="Emit the COMPACT single-block digest (header + "
                         "sharpness-by-depth + auto cliff-marker + top fed "
                         "exemplars + optional --grep/--redundancy), a few "
                         "hundred tokens for copy-paste to chat. Honours "
                         "--bundle-meta (depth+words), --grep, --redundancy, "
                         "--top-words; writes <run-dir>/topics_digest.md by "
                         "default. Suppresses the verbose report.")
    ap.add_argument("--digest-exemplars", type=int, default=8, metavar="N",
                    help="How many best-fed topics to line-detail in --digest "
                         "(default 8).")
    ap.add_argument("--readout-auc", action="store_true",
                    help="Emit a compact PER-NODE READOUT-AUC slice from the run's "
                         "results_readout.json (run gated-pc-readout first): AUC "
                         "quantiles, by-depth medians (needs --bundle-meta), and "
                         "--grep'd nodes' AUCs vs the rest. AUC/AP only — no "
                         "patient counts. Suppresses the other reports.")
    ap.add_argument("--credited-file", default=None, metavar="TSV",
                    help="(--readout-auc) TSV with a mondo_id column (e.g. the "
                         "probe's --emit-eta table): split the slice into "
                         "CREDITED vs UNCREDITED scored nodes — exp 0116's "
                         "internal-control read.")
    ap.add_argument("--compare-run", default=None, metavar="RUN_DIR",
                    help="(--readout-auc) baseline run dir (same arm in its "
                         "results_readout.json): add PAIRED per-node AUC "
                         "deltas on shared scored nodes, split by "
                         "--credited-file when given. (--profile-align) "
                         "baseline run for paired alignment scores.")
    ap.add_argument("--strip-audit", action="store_true",
                    help="Emit the LEAKAGE-STRIP audit: per domain, how many "
                         "vocab dims the label DAG's node ids actually resolve "
                         "to (what strip_mode could have removed). On the "
                         "native-Mondo path node ids are Mondo numerics, so "
                         "the OMOP-keyed strip is expected to hit ~nothing. "
                         "Needs the bundle meta; --credited-file adds the "
                         "profile-concept split. Suppresses other reports.")
    ap.add_argument("--collinearity", action="store_true",
                    help="Emit the credited-topic COLLINEARITY report: profile "
                         "Jaccard among credited nodes, boosted-topic cosine vs "
                         "peers/background/ancestors/flat, readout-decoder "
                         "weight share on own/background/ancestor blocks, and "
                         "topic evidence vs the prior floor — credited vs "
                         "uncredited fed/starved. Needs --credited-file and "
                         "the bundle meta. Suppresses other reports.")
    ap.add_argument("--profile-align", action="store_true",
                    help="Emit the ~10-line profile-ALIGNMENT scorecard "
                         "(insight 0084's legibility read, quantified): each "
                         "credited node's boosted topic scored against its "
                         "positive profile tokens (E[beta] mass + top-15 "
                         "overlap), pooled starved vs fed, paired vs "
                         "--compare-run. Needs --credited-file (the emit-eta "
                         "TSV) and the bundle meta. Suppresses other reports.")
    args = ap.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    if args.strip_audit:
        report = build_strip_audit(run_dir, bundle_meta_path=args.bundle_meta,
                                   profile_file=args.credited_file)
        default_out = "strip_audit.md"
    elif args.collinearity:
        if not args.credited_file:
            raise SystemExit("[inspect_topics] --collinearity needs "
                             "--credited-file (the probe's --emit-eta TSV; "
                             "CREDITED=1 via the Makefile)")
        report = build_collinearity(
            run_dir, args.credited_file, bundle_meta_path=args.bundle_meta,
            readout_label=args.readout_label)
        default_out = "collinearity.md"
    elif args.profile_align:
        if not args.credited_file:
            raise SystemExit("[inspect_topics] --profile-align needs "
                             "--credited-file (the probe's --emit-eta TSV; "
                             "CREDITED=1 via the Makefile)")
        report = build_profile_align(
            run_dir, args.credited_file, bundle_meta_path=args.bundle_meta,
            compare_dir=args.compare_run)
        default_out = "profile_align.md"
    elif args.readout_auc:
        report = build_auc_slice(
            run_dir, bundle_meta_path=args.bundle_meta,
            grep_pattern=args.grep, arm=args.readout_label,
            credited_file=args.credited_file, compare_dir=args.compare_run)
        default_out = "readout_auc_slice.md"
    elif args.digest:
        report = build_digest(
            run_dir, exemplars=args.digest_exemplars, t_words=args.top_words,
            bundle_meta_path=args.bundle_meta, vocab_path=args.vocab_map,
            names_path=args.concept_names, readout_label=args.readout_label,
            grep_pattern=args.grep, redundancy=bool(args.redundancy),
            profile_file=args.credited_file)
        default_out = "topics_digest.md"
    else:
        report = build_report(
            run_dir, top_topics=args.top_topics, top_loadings=args.top_loadings,
            t_words=args.top_words, vocab_path=args.vocab_map,
            names_path=args.concept_names, sort_by=args.sort,
            bundle_meta_path=args.bundle_meta, readout_label=args.readout_label,
            tour_per_depth=args.tour, redundancy=args.redundancy,
            grep_pattern=args.grep)
        default_out = "topics_report.md"

    print(report)
    if args.out != "-":
        out = Path(args.out) if args.out else Path(run_dir) / default_out
        out.write_text(report + "\n")
        print(f"\n[inspect_topics] wrote {out}", flush=True)


if __name__ == "__main__":
    main()

"""The browsable export of a DAG-gated model: format `dag-gated-export/1`.

Spec: `docs/superpowers/specs/2026-11-11-dag-gated-dashboard-export.md` (the contract
the browser in `dag-dashboard/` reads). Pure Python + numpy, no Spark: a fitted
model's λ, its node set, the label DAG it was given, the vocabularies and the names in;
four minified JSON files out (`manifest.json`, `nodes.json`, `vocab.json`,
`topics.json`), columnar, every reference an integer index.

What is shipped, and why it stays small (target ~2 MB gzipped at K ~ 2,900):

- per topic and domain only a SELECTION of terms (`select_terms`): the top
  `top_beta` by β, plus up to `top_lift` more by lift β / p among the top `lift_pool`
  by β. The browser's relevance slider (λ log β + (1 - λ) log(β / p)) spans exactly
  these two ends: λ = 1 ranks by β, λ = 0 by lift. The lift pool keeps a rare term
  whose β sits at the prior level from ranking first on lift alone.
- term ids sorted ascending and DELTA-coded (`delta_encode`): small gaps, short
  digits, repetitive text that gzip folds well.
- weights as one LOG-BYTE each (`log_byte_encode`): b = round(255 (log x - lo) /
  (hi - lo)), with (lo, hi) the range of log x over every shipped value of a domain.
  A topic's β spans orders of magnitude and the browser reads it on a log scale
  (relevance is linear in log β), so a log grid spends the 256 levels evenly where
  they are read; the relative error is at most exp((hi - lo) / 510) - 1.
- p(w), the relevance formula's p, is the model's own domain-normalized term marginal
  p(w) ∝ sum_k λ_kw (`term_marginal`): no corpus data needed.

Disclosure (AGENTS.md, egress floor): model parameters over a vocabulary floored at
20 patients per term; node support is shown BANDED (`band_support`, 20-wide bands at
the midpoint); no document or patient data, and no per-node number beyond the banded
support, the topic and its sharpness (supportFraction).

Node topics: topic k < n_bg is background; the node with engine id e owns topics
n_bg + (e - 1) * tpn .. n_bg + e * tpn - 1 (`DagLayout`'s block order). With tpn > 1
every topic of a block is exported (`topics.node` names its node for each) and
`nodes.topic` is the FIRST topic of the block; `support_frac` is then the minimum over
the block (a block is flat only when every topic in it is, as `nodeStatistics`).
"""
from __future__ import annotations

import gzip
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = "dag-gated-export/1"
FILES = ("manifest.json", "nodes.json", "vocab.json", "topics.json")
STATUS_TOPIC, STATUS_PRUNED, STATUS_SPLICED, STATUS_UNSUPPORTED = 0, 1, 2, 3
SUPPORT_BAND = 20
DEFAULT_SELECTION = {"top_beta": 25, "top_lift": 25, "lift_pool": 200}
_JSON_SEPARATORS = (",", ":")


# --------------------------------------------------------------------------- #
# Codecs                                                                       #
# --------------------------------------------------------------------------- #

def log_byte_range(values) -> tuple:
    """`(lo, hi)`: the range of log x over the positive `values` (the codec's grid
    ends). An empty input gives (0, 0)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[v > 0]
    if v.size == 0:
        return 0.0, 0.0
    lv = np.log(v)
    return float(lv.min()), float(lv.max())


def log_byte_encode(values, lo, hi) -> np.ndarray:
    """Positive `values` -> bytes b in 0..255 on the log grid [lo, hi]:
    b = round(255 (log x - lo) / (hi - lo)), clipped. A degenerate range (hi == lo)
    encodes everything as 0, which decodes to exp(lo)."""
    v = np.asarray(values, dtype=np.float64)
    if hi <= lo:
        return np.zeros(v.shape, dtype=np.uint8)
    with np.errstate(divide="ignore"):
        t = (np.log(np.maximum(v, 1e-300)) - lo) / (hi - lo)
    return np.clip(np.rint(255.0 * t), 0, 255).astype(np.uint8)


def log_byte_decode(b, lo, hi) -> np.ndarray:
    """The inverse grid: x = exp(lo + (hi - lo) b / 255)."""
    return np.exp(lo + (hi - lo) * np.asarray(b, dtype=np.float64) / 255.0)


def log_byte_max_log_error(lo, hi) -> float:
    """Worst-case |log decoded - log x| of the codec: half a grid step."""
    return max(hi - lo, 0.0) / 510.0


def delta_encode(sorted_ids) -> list:
    """Ascending ids -> [first, gap, gap, ...] (the first value absolute)."""
    ids = [int(i) for i in sorted_ids]
    return [b - a for a, b in zip([0] + ids[:-1], ids)]


def delta_decode(deltas) -> list:
    """The inverse of `delta_encode` (a running sum)."""
    return [int(x) for x in np.cumsum(np.asarray(deltas, dtype=np.int64))]


def band_support(s):
    """A document count shown at its 20-wide band's midpoint: 0 -> 0, 1-20 -> 10,
    21-40 -> 30, ...; None stays None (unknown). The Mondo usage dashboard's banding,
    with the egress floor's width (a band never reveals a count below 20)."""
    if s is None:
        return None
    s = int(s)
    if s <= 0:
        return 0
    return SUPPORT_BAND * math.ceil(s / SUPPORT_BAND) - SUPPORT_BAND // 2


# --------------------------------------------------------------------------- #
# Term selection                                                               #
# --------------------------------------------------------------------------- #

def domain_beta(lam_dom) -> np.ndarray:
    """E[β] within one domain: each topic's λ row in the domain, normalized."""
    lam_dom = np.asarray(lam_dom, dtype=np.float64)
    s = lam_dom.sum(axis=1, keepdims=True)
    return lam_dom / np.where(s > 0, s, 1.0)


def term_marginal(lam_dom) -> np.ndarray:
    """p(w) within one domain: p(w) ∝ sum_k λ_kw, the model's own term marginal
    (expected word counts plus the prior; the relevance formula's p)."""
    m = np.asarray(lam_dom, dtype=np.float64).sum(axis=0)
    t = m.sum()
    return m / t if t > 0 else m


def select_terms(beta, pw, *, top_beta=25, top_lift=25, lift_pool=200) -> list:
    """Per topic (row of `beta`, one domain): the local term indices shipped, as an
    ascending int array.

    The top `top_beta` terms by β, plus up to `top_lift` more by lift β / p(w) among
    the top `lift_pool` terms by β (excluding those already chosen). Ties break by the
    lower term index (a stable sort), so the selection is deterministic even among the
    many prior-level terms of a sparse row, which tie exactly."""
    beta = np.asarray(beta, dtype=np.float64)
    pw = np.asarray(pw, dtype=np.float64)
    K, V = beta.shape
    pool = min(int(lift_pool), V)
    nb = min(int(top_beta), V)
    # Stable descending order by β: argsort of -β with kind="stable" keeps the lower
    # index first among equal β.
    order = np.argsort(-beta, axis=1, kind="stable")[:, :max(pool, nb)]
    with np.errstate(divide="ignore", invalid="ignore"):
        lift_all = beta / np.where(pw > 0, pw, np.inf)[None, :]
    out = []
    for k in range(K):
        head = order[k, :nb]
        rest = order[k, nb:pool]
        if top_lift > 0 and rest.size:
            lr = lift_all[k, rest]
            pick = rest[np.argsort(-lr, kind="stable")[:int(top_lift)]]
            sel = np.concatenate([head, pick])
        else:
            sel = head
        out.append(np.sort(sel.astype(np.int64)))
    return out


# --------------------------------------------------------------------------- #
# The node table                                                               #
# --------------------------------------------------------------------------- #

def _node_key(x):
    """A total order over mixed str / int node ids (ints first), the node set's."""
    return (0, x, "") if isinstance(x, int) else (1, 0, str(x))


def _longest_depth(parents) -> dict:
    """Longest path from the synthetic root over `{node: [parents]}` (top level = 1,
    the node set's and `nodeStatistics`' convention). Iterative; cycle-free input."""
    depth = {}
    for start in parents:
        if start in depth:
            continue
        stack = [(start, False)]
        while stack:
            n, done = stack.pop()
            if n in depth:
                continue
            ps = parents.get(n, ())
            if done or all(p in depth for p in ps):
                depth[n] = 1 + max((depth[p] for p in ps), default=0)
            else:
                stack.append((n, True))
                stack.extend((p, False) for p in ps if p not in depth)
    return depth


def _descendants(children, n) -> set:
    seen, stack = set(), list(children.get(n, ()))
    while stack:
        c = stack.pop()
        if c not in seen:
            seen.add(c)
            stack.extend(children.get(c, ()))
    return seen


def node_supports(given_dag, ns, extra_support=None) -> dict:
    """`{node: support | None}` over every node of the given DAG.

    - kept: the node set's own `support`;
    - unsupported (dropped after the fit): `extra_support` (the as-fitted
      `node_statistics.tsv` rows, which describe the model before the drop); None
      without it;
    - spliced: DERIVED, exactly. A spliced node was removed because its single kept
      child had EQUAL support; that child was kept at learn time or spliced in turn,
      so the chain ends at a learn-kept descendant (kept now, or unsupported) with the
      same support. Support never decreases going up, so that support is the
      MAXIMUM over the node's learn-kept descendants. None when one of those is an
      unsupported node of unknown support;
    - pruned: None. Their support is below `min_docs_per_node` and is recorded
      nowhere (the node set keeps kept supports only).
    """
    extra = {k: int(v) for k, v in (extra_support or {}).items() if v is not None}
    sup = {n: int(s) for n, s in ns.support.items()}
    for n in ns.unsupported:
        sup[n] = extra.get(n)
    children = {}
    for c, ps in given_dag.items():
        for p in ps:
            children.setdefault(p, []).append(c)
    learn_kept = set(ns.support) | set(ns.unsupported)
    for n in ns.spliced:
        below = [d for d in _descendants(children, n) if d in learn_kept]
        vals = [sup.get(d) for d in below]
        sup[n] = (None if (not vals or any(v is None for v in vals)) else max(vals))
    for n in ns.pruned:
        sup[n] = None
    return {n: sup.get(n) for n in given_dag}


def _support_fraction(lam_list):
    """Per-topic supportFraction (spark_vi.eval.topic.sharpness), lazily imported so
    this module's codecs need nothing beyond numpy."""
    from spark_vi.eval.topic.sharpness import support_fraction
    frac, _ = support_fraction({m: a for m, a in enumerate(lam_list)})
    return frac


# --------------------------------------------------------------------------- #
# Build                                                                        #
# --------------------------------------------------------------------------- #

def _code(c):
    """A vocab key as written in the bundle's vocab maps -> an int when it is one."""
    if isinstance(c, (int, np.integer)):
        return int(c)
    s = str(c)
    try:
        return int(s)
    except ValueError:
        return s


def build_export(lam, ns, *, n_bg, tpn, given_dag, domains, vocab_maps,
                 node_names=None, term_names=None, extra_support=None,
                 run="", mondo_version="", cdr="", created_utc=None,
                 top_beta=25, top_lift=25, lift_pool=200, notes=None) -> dict:
    """The four files of `dag-gated-export/1` as JSON-ready objects
    (`{file name: object}`; `manifest["sizes"]` is filled by `write_export`).

    `lam`: per-domain λ (`{m: (K, V_m)}` or a list in domain order), the model as
    saved (cleaned when it was). `ns`: its `spark_vi.dag.nodeset.NodeSet`. `given_dag`:
    the label DAG the fit was given, `{node: [parents]}` in user ids (every node a key).
    `domains`: domain names in domain order. `vocab_maps`: per domain `{code: local
    index}`. `node_names`: `{node id: label}`; `term_names`: `{code: name}` (codes as
    ints where they are ints). `extra_support`: `{node: support}` for nodes the node
    set holds no support for (the unsupported ones, as fitted)."""
    from spark_vi.eval.topic.sharpness import lambda_domains
    lam_list = lambda_domains(lam) if not isinstance(lam, list) else [
        np.asarray(a, dtype=np.float64) for a in lam]
    M = len(lam_list)
    if len(domains) != M or len(vocab_maps) != M:
        raise ValueError(f"{M} λ domains but {len(domains)} domain names and "
                         f"{len(vocab_maps)} vocab maps")
    K = lam_list[0].shape[0]
    n_bg, tpn = int(n_bg), int(tpn)
    N = len(ns.user_ids) - 1
    if K != n_bg + N * tpn:
        raise ValueError(f"λ has K={K} topics; the node set lays out n_bg={n_bg} + "
                         f"{N} nodes x tpn={tpn} = {n_bg + N * tpn}")
    for m, (a, vm) in enumerate(zip(lam_list, vocab_maps)):
        if a.shape[0] != K or a.shape[1] != len(vm):
            raise ValueError(f"domain {m} ({domains[m]}): λ is {a.shape}, vocab map "
                             f"has {len(vm)} terms")
    node_names = node_names or {}
    term_names = term_names or {}

    # ---- nodes ----------------------------------------------------------------
    given = {n: list(ps) for n, ps in given_dag.items()}
    for ps in list(given.values()):
        for p in ps:
            given.setdefault(p, [])
    status = {}
    for n in ns.nodes:
        status[n] = STATUS_TOPIC
    for n in ns.pruned:
        status[n] = STATUS_PRUNED
    for n in ns.spliced:
        status[n] = STATUS_SPLICED
    for n in ns.unsupported:
        status[n] = STATUS_UNSUPPORTED
    missing = [n for n in given if n not in status]
    extra_nodes = [n for n in status if n not in given]
    if missing or extra_nodes:
        raise ValueError(f"the node set and the given DAG disagree: {len(missing)} "
                         f"given node(s) with no status (e.g. {missing[:3]}), "
                         f"{len(extra_nodes)} node-set node(s) not in the DAG "
                         f"(e.g. {extra_nodes[:3]})")
    depth = _longest_depth(given)
    order = sorted(given, key=lambda n: (depth[n], _node_key(n)))
    idx = {n: i for i, n in enumerate(order)}
    sup = node_supports(given, ns, extra_support)

    lam_frac = _support_fraction(lam_list)
    first_topic = {}
    block_frac = {}
    topic_node = [None] * K
    for e, u in enumerate(ns.user_ids[1:], start=1):
        lo = n_bg + (e - 1) * tpn
        blk = list(range(lo, lo + tpn))
        first_topic[u] = lo if blk else None
        block_frac[u] = float(min(lam_frac[t] for t in blk)) if blk else None
        for t in blk:
            topic_node[t] = idx[u]

    def landing_of(n):
        st = status[n]
        if st == STATUS_PRUNED:
            land = ns.pruned[n]
        elif st == STATUS_UNSUPPORTED:
            land = ns.unsupported[n]
        else:   # spliced: the fit's own landing, engine ids -> user ids
            land = [ns.user_ids[e] for e in ns.frontier.landing.get(n, ())]
        return sorted(idx[x] for x in land)

    def kept_parents_of(n):
        from spark_vi.dag.nodeset import ROOT
        return sorted(idx[p] for p in ns.parents[n] if p is not ROOT)

    nodes = {
        "id": [str(n) for n in order],
        "name": [str(node_names.get(n, n)) for n in order],
        "parents": [sorted(idx[p] for p in given[n]) for n in order],
        "kept_parents": [kept_parents_of(n) if status[n] == STATUS_TOPIC else None
                         for n in order],
        "depth": [int(depth[n]) for n in order],
        "status": [int(status[n]) for n in order],
        "landing": [None if status[n] == STATUS_TOPIC else landing_of(n)
                    for n in order],
        "support": [band_support(sup[n]) for n in order],
        "support_frac": [(int(round(1000 * block_frac[n]))
                          if status[n] == STATUS_TOPIC and block_frac[n] is not None
                          else None) for n in order],
        "topic": [first_topic.get(n) if status[n] == STATUS_TOPIC else None
                  for n in order],
    }

    # ---- term selection per (topic, domain) ------------------------------------
    sel = {"top_beta": int(top_beta), "top_lift": int(top_lift),
           "lift_pool": int(lift_pool)}
    betas, pws, picks = [], [], []
    for a in lam_list:
        b, p = domain_beta(a), term_marginal(a)
        betas.append(b)
        pws.append(p)
        picks.append(select_terms(b, p, **sel))

    # ---- vocab: only shipped terms; index order (domain, times shipped desc, local
    # index) so the terms most topics ship get the smallest ids and the shortest gaps.
    vocab = {"domain": [], "code": [], "name": [], "freq": []}
    gidx = []          # per domain: local index -> global vocab index
    freq_codec, weight_codec, domain_meta = [], [], []
    for m, vm in enumerate(vocab_maps):
        inv = {}
        for c, i in vm.items():
            inv[int(i)] = _code(c)
        counts = np.bincount(np.concatenate(picks[m]) if picks[m] else
                             np.zeros(0, np.int64), minlength=len(vm))
        shipped = np.flatnonzero(counts)
        shipped = shipped[np.lexsort((shipped, -counts[shipped]))]
        g = np.full(len(vm), -1, dtype=np.int64)
        base = len(vocab["domain"])
        g[shipped] = base + np.arange(shipped.size)
        gidx.append(g)
        lo, hi = log_byte_range(pws[m][shipped])
        freq_codec.append({"lo": lo, "hi": hi})
        fb = log_byte_encode(pws[m][shipped], lo, hi)
        for j, li in enumerate(shipped):
            code = inv[int(li)]
            vocab["domain"].append(m)
            vocab["code"].append(code)
            vocab["name"].append(str(term_names.get(code, "")))
            vocab["freq"].append(int(fb[j]))
        w_all = np.concatenate([betas[m][k, picks[m][k]] for k in range(K)]) \
            if K else np.zeros(0)
        wlo, whi = log_byte_range(w_all)
        weight_codec.append({"lo": wlo, "hi": whi})
        domain_meta.append({"name": str(domains[m]), "shipped_terms": int(shipped.size)})

    # ---- topics -----------------------------------------------------------------
    d_all = []
    for k in range(K):
        per = []
        for m in range(M):
            loc = picks[m][k]
            gl = gidx[m][loc]
            o = np.argsort(gl, kind="stable")
            gl, loc = gl[o], loc[o]
            wb = log_byte_encode(betas[m][k, loc], weight_codec[m]["lo"],
                                 weight_codec[m]["hi"])
            per.append([delta_encode(gl), [int(x) for x in wb]])
        d_all.append(per)
    topics = {"node": topic_node, "d": d_all}

    manifest = {
        "schema": SCHEMA, "run": str(run),
        "created_utc": created_utc or datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "mondo_version": str(mondo_version or ""), "cdr": str(cdr or ""),
        "K": int(K), "n_bg": n_bg, "tpn": tpn,
        "domains": domain_meta,
        "selection": sel,
        "weight_codec": {"kind": "log-byte", "per_domain": weight_codec},
        "freq_codec": {"kind": "log-byte", "per_domain": freq_codec},
        "disclosure": {
            "min_cell": _egress_floor(),
            "support": "20-wide bands shown at the midpoint (1-20 -> 10 '<=20', "
                       "21-40 -> 30, ...)"},
        "min_docs_per_node": int(ns.min_docs_per_node),
        "sizes": {},
        "notes": notes if notes is not None else default_notes(),
    }
    return {"manifest.json": manifest, "nodes.json": nodes, "vocab.json": vocab,
            "topics.json": topics}


def _egress_floor() -> int:
    from charmpheno.egress import EGRESS_FLOOR
    return int(EGRESS_FLOOR)


def default_notes() -> str:
    return ("nodes: index order = (depth, id); depth = longest path from the synthetic "
            "root, top level = 1. support: banded; null = not recorded (pruned nodes: "
            "below min_docs_per_node; unsupported nodes without the as-fitted node "
            "statistics; spliced nodes derived as the max over their kept or "
            "unsupported descendants, null when one is unknown). support_frac and the "
            "relevance p are the model's own. With tpn > 1, nodes.topic is the first "
            "topic of the node's block and topics.node names the node for each. "
            "vocab.name '' = unresolved (show the code).")


# --------------------------------------------------------------------------- #
# Write / read / decode                                                        #
# --------------------------------------------------------------------------- #

def dumps(obj) -> bytes:
    """Minified JSON, UTF-8 (the format's one serialization)."""
    return json.dumps(obj, separators=_JSON_SEPARATORS, ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _gz_len(b: bytes) -> int:
    return len(gzip.compress(b, compresslevel=9, mtime=0))


def write_export(files: dict, out_dir) -> dict:
    """Write the four files to `out_dir`; fill `manifest["sizes"]` with
    `{file: [raw bytes, gzip bytes]}` for every file, the manifest's own included (its
    entry is iterated to a fixed point, since it counts itself). Returns the sizes."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    blobs = {name: dumps(obj) for name, obj in files.items() if name != "manifest.json"}
    sizes = {name: [len(b), _gz_len(b)] for name, b in blobs.items()}
    man = dict(files["manifest.json"])
    man_sizes = [0, 0]
    for _ in range(10):
        man["sizes"] = {**sizes, "manifest.json": man_sizes}
        mb = dumps(man)
        new = [len(mb), _gz_len(mb)]
        if new == man_sizes:
            break
        man_sizes = new
    files["manifest.json"]["sizes"] = man["sizes"]
    blobs["manifest.json"] = dumps(man)
    for name, b in blobs.items():
        (out / name).write_bytes(b)
    return dict(man["sizes"])


def read_export(out_dir) -> dict:
    """The four files of an export directory, parsed."""
    d = Path(out_dir)
    return {name: json.loads((d / name).read_text(encoding="utf-8")) for name in FILES}


def decode_topics(files) -> list:
    """`topics.json` decoded: per topic, per domain, `(term indices, β)` with the
    indices absolute (into vocab.json) and β from the weight codec. The browser's
    reading of the format, here for the tests."""
    man, top = files["manifest.json"], files["topics.json"]
    codec = man["weight_codec"]["per_domain"]
    out = []
    for per in top["d"]:
        row = []
        for m, (deltas, w) in enumerate(per):
            ids = delta_decode(deltas)
            row.append((ids, log_byte_decode(w, codec[m]["lo"], codec[m]["hi"])))
        out.append(row)
    return out


def decode_freq(files) -> np.ndarray:
    """vocab.json's p(w), decoded per term with its domain's freq codec."""
    man, voc = files["manifest.json"], files["vocab.json"]
    codec = man["freq_codec"]["per_domain"]
    dom = np.asarray(voc["domain"], dtype=np.int64)
    b = np.asarray(voc["freq"], dtype=np.float64)
    lo = np.asarray([codec[m]["lo"] for m in dom], dtype=np.float64)
    hi = np.asarray([codec[m]["hi"] for m in dom], dtype=np.float64)
    return np.exp(lo + (hi - lo) * b / 255.0)

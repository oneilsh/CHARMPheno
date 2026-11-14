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
support, the topic and its sharpness (supportFraction). The floors this rests on are
checked, not assumed: `build_export` refuses a node set with `min_docs_per_node` below
the floor and drops an NPMI result whose `min_pair_count` is below it
(`npmi_exportable`); the driver refuses a vocabulary floored below it.

Marks (optional, `positive_profile_sets` / `marks_local`): a node's OWN codes (the
code ladder landed on the fitted node set) and its POSITIVE HPO-profile concepts, as
indices into vocab.json in the condition domain, so the browser can badge a topic's
terms. Ontology / code-ladder mappings only: no count beyond counts of concepts.

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


def select_terms(beta, pw, *, top_beta=40, top_lift=40, lift_pool=200) -> list:
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


def npmi_block(npmi: dict, K: int):
    """`(topics.json "npmi" block, manifest "npmi" entry)` from a topic-NPMI result
    (`analysis/cloud/topic_npmi.py`'s `<run>/npmi_<tag>.json`: `{top_n, split,
    min_pair_count, rows: [{domain, topic, npmi_gated, npmi_all, ...}]}`).

    Per scored domain, two arrays aligned with the topics: the gated NPMI (over the
    documents each topic's gate admits) and the all-documents NPMI of the topic's top
    terms, as round(1000 * NPMI), null where a topic was not scored (NaN, or no row).
    Only these two scalars are shipped, no count. Rows are keyed by the model's topic
    index, which is the export's topic index.

    Disclosure: a topic's NPMI averages the word pairs whose joint document count is at
    least the result's `min_pair_count` (`topic_npmi.run_npmi`; its analysis default is
    3). The `floor` that `score` is given is the size of the gated reference set, not
    the pair cells. So the scalars rest on cells >= the egress floor only when
    `min_pair_count >= EGRESS_FLOOR`; `npmi_exportable` is that rule, and
    `build_export` drops the block otherwise. This function does not check it."""
    import math
    doms = sorted({int(r["domain"]) for r in npmi.get("rows", [])})
    gated = {d: [None] * K for d in doms}
    every = {d: [None] * K for d in doms}

    def q(v):
        return None if v is None or (isinstance(v, float) and math.isnan(v)) \
            else int(round(1000 * float(v)))
    for r in npmi.get("rows", []):
        k = int(r["topic"])
        if 0 <= k < K:
            gated[int(r["domain"])][k] = q(r.get("npmi_gated"))
            every[int(r["domain"])][k] = q(r.get("npmi_all"))
    block = {"domains": doms, "gated": [gated[d] for d in doms],
             "all": [every[d] for d in doms]}
    meta = {"top_n": npmi.get("top_n"), "split": npmi.get("split"),
            "min_pair_count": npmi.get("min_pair_count"), "domains": doms, "scale": 1000}
    return block, meta


def npmi_exportable(npmi: dict):
    """`(ok, reason)`: may this topic-NPMI result leave the workspace? Only when every
    pair it averaged had a joint count at or above the egress floor, i.e. its
    `min_pair_count >= EGRESS_FLOOR` (`npmi_block`'s docstring). `reason` says why not
    (empty when ok)."""
    floor = _egress_floor()
    mpc = npmi.get("min_pair_count") if isinstance(npmi, dict) else None
    try:
        mpc_i = int(mpc)
    except (TypeError, ValueError):
        return False, (f"the NPMI result records no usable min_pair_count ({mpc!r}); "
                       f"an exportable NPMI needs min_pair_count >= {floor}")
    if mpc_i < floor:
        return False, (f"the NPMI averaged word pairs with joint counts down to "
                       f"min_pair_count={mpc_i}, below the egress floor {floor}; "
                       f"re-run the readout with --npmi --npmi-min-pair-count {floor} "
                       "for an exportable NPMI")
    return True, ""


def positive_profile_sets(rows) -> tuple:
    """`({node: {concept id}}, stats)`: each node's POSITIVE HPO-profile concepts from
    profile-mark rows `(mondo_id, concept_id, neg)` (`hpo_profile_marks.py`'s table).

    The rule is `profile_eta.build_profile_seed`'s: a NOT-annotated (`neg`) phenotype
    is not what the node's topic should look like, so it never marks; a concept listed
    both ways for the same node is contested and dropped. `stats` counts rows and
    concepts only (`n_rows`, `n_neg_rows`, `n_contested`)."""
    pos, neg = {}, {}
    n_rows = n_neg = 0
    for mid, cid, ng in rows:
        n_rows += 1
        if ng:
            n_neg += 1
        (neg if ng else pos).setdefault(mid, set()).add(int(cid))
    out, n_contested = {}, 0
    for mid, s in pos.items():
        bad = s & neg.get(mid, set())
        n_contested += len(bad)
        if s - bad:
            out[mid] = s - bad
    return out, {"n_rows": n_rows, "n_neg_rows": n_neg, "n_contested": n_contested}


def marks_local(by_node, vocab_map) -> tuple:
    """`({node: sorted local term indices}, {node: count})` of a marks map
    (`{node: {concept code}}`) against one domain's vocab map (`{code: local index}`):
    only concepts in the model's vocabulary have an index; the count is of those. A
    node whose concepts are all outside the vocabulary is absent from both."""
    code2local = {_code(c): int(i) for c, i in vocab_map.items()}
    idx = {}
    for n, codes in by_node.items():
        loc = sorted({code2local[_code(c)] for c in codes if _code(c) in code2local})
        if loc:
            idx[n] = loc
    return idx, {n: len(v) for n, v in idx.items()}


def build_export(lam, ns, *, n_bg, tpn, given_dag, domains, vocab_maps,
                 node_names=None, term_names=None, extra_support=None,
                 run="", mondo_version="", cdr="", created_utc=None,
                 top_beta=40, top_lift=40, lift_pool=200, notes=None,
                 npmi=None, own_codes=None, profile=None, marks_meta=None,
                 marks_domain=0) -> dict:
    """The four files of `dag-gated-export/1` as JSON-ready objects
    (`{file name: object}`; `manifest["sizes"]` is filled by `write_export`).

    `lam`: per-domain λ (`{m: (K, V_m)}` or a list in domain order), the model as
    saved (cleaned when it was). `ns`: its `spark_vi.dag.nodeset.NodeSet`. `given_dag`:
    the label DAG the fit was given, `{node: [parents]}` in user ids (every node a key).
    `domains`: domain names in domain order. `vocab_maps`: per domain `{code: local
    index}`. `node_names`: `{node id: label}`; `term_names`: `{code: name}` (codes as
    ints where they are ints). `extra_support`: `{node: support}` for nodes the node
    set holds no support for (the unsupported ones, as fitted). `npmi`: a topic-NPMI
    result (`npmi_block`); dropped, with the reason printed, unless `npmi_exportable`.

    Marks (optional; the spec's "Marks" section): `own_codes` is `{node id: {concept
    id}}`, each node's OWN codes (the code ladder landed on the fitted node set, the
    fit's `code_map.tsv`); `profile` is `{node id: {concept id}}`, each node's POSITIVE
    HPO-profile concepts (`positive_profile_sets`). Both are read against the vocab of
    domain `marks_domain` (the condition domain): `nodes.json` gains `own` / `profile`
    (vocab.json indices, null when none; `profile` for nodes with a topic only) and
    `own_n` / `profile_n` (how many of the node's concepts are in the model's
    vocabulary, every node). A marked term that no topic list ships is added to
    vocab.json (after the domain's shipped terms) so the page can name it. `marks_meta`
    is `{"own": {"source"}, "profile": {"source", "hpo_release"}}` for the manifest.

    Disclosure: refuses (ValueError) a node set fitted with `min_docs_per_node` below
    the egress floor, since a kept node's banded support and its topic then rest on
    fewer documents than the floor. The vocabulary's own floor (`min_patient_count`)
    is not in the model; the driver checks it from the run manifest."""
    floor = _egress_floor()
    if int(ns.min_docs_per_node) < floor:
        raise ValueError(
            f"min_docs_per_node={int(ns.min_docs_per_node)} is below the egress floor "
            f"{floor}: the node set keeps nodes supported by fewer documents than the "
            "floor, so the export is not disclosable (refit with minDocsPerNode >= "
            f"{floor})")
    if npmi is not None:
        ok, why = npmi_exportable(npmi)
        if not ok:
            print(f"[export-dag-gated] NPMI dropped: {why}", flush=True)
            npmi = None
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
    # ---- marks (optional): the marked nodes' terms in the marks domain -----------
    has_marks = own_codes is not None or profile is not None
    md = int(marks_domain)
    if has_marks and not 0 <= md < M:
        raise ValueError(f"marks_domain={md} is not one of the {M} domains")
    own_loc, own_cnt = (marks_local(own_codes, vocab_maps[md]) if own_codes is not None
                        else ({}, {}))
    prof_loc, prof_cnt = (marks_local(profile, vocab_maps[md]) if profile is not None
                          else ({}, {}))
    # Profile LISTS ship for nodes with a topic only: a profile marks a topic's terms,
    # and the profiles of the whole label DAG would outweigh the topics. Counts ship
    # for every node.
    prof_loc = {n: v for n, v in prof_loc.items() if status.get(n) == STATUS_TOPIC}
    own_loc = {n: v for n, v in own_loc.items() if n in idx}
    own_cnt = {n: v for n, v in own_cnt.items() if n in idx}
    prof_cnt = {n: v for n, v in prof_cnt.items() if n in idx}
    marked_local = set()
    for d in (own_loc, prof_loc):
        for v in d.values():
            marked_local.update(v)

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
        n_topic_terms = int(shipped.size)
        if has_marks and m == md:
            # marked terms that no topic list ships: appended after the shipped ones
            # in local-index order, so the topics' delta-coded ids do not move
            extra = np.asarray(sorted(marked_local - set(shipped.tolist())),
                               dtype=np.int64)
            shipped = np.concatenate([shipped, extra])
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
        dmeta = {"name": str(domains[m]), "shipped_terms": n_topic_terms}
        if has_marks and m == md:
            dmeta["mark_terms"] = int(shipped.size) - n_topic_terms
        domain_meta.append(dmeta)

    marks_manifest = None
    if has_marks:
        gm = gidx[md]

        def _glist(loc):
            return sorted(int(gm[li]) for li in loc)
        marks_manifest = {"domain": md}
        meta = marks_meta or {}
        if own_codes is not None:
            nodes["own"] = [_glist(own_loc[n]) if n in own_loc else None for n in order]
            nodes["own_n"] = [int(own_cnt.get(n, 0)) for n in order]
            marks_manifest["own"] = {
                "source": str((meta.get("own") or {}).get("source", "code_map.tsv")),
                "nodes": sum(1 for x in nodes["own"] if x is not None)}
        if profile is not None:
            nodes["profile"] = [_glist(prof_loc[n]) if n in prof_loc else None
                                for n in order]
            nodes["profile_n"] = [int(prof_cnt.get(n, 0)) for n in order]
            pm = meta.get("profile") or {}
            marks_manifest["profile"] = {
                "source": str(pm.get("source", "profile_marks.tsv")),
                **({"hpo_release": str(pm["hpo_release"])} if pm.get("hpo_release")
                   else {}),
                "nodes": sum(1 for x in nodes["profile"] if x is not None)}

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
    npmi_meta = None
    if npmi is not None:
        topics["npmi"], npmi_meta = npmi_block(npmi, K)

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
        **({"npmi": npmi_meta} if npmi_meta else {}),
        **({"marks": marks_manifest} if marks_manifest else {}),
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


def embed_in_page(page_html: str, out_dir) -> str:
    """The browser page with the export's four files embedded, as one self-contained
    HTML file: a `<script id="dag-data" type="application/json">` block in `<head>`
    holding `{file name: parsed file}`, which the page reads instead of fetching.

    Why: some file servers (JupyterLab's, which sandboxes the HTML it serves) and
    `file://` block the page's fetches of the data files; a page that carries its data
    needs no fetch. `</` is escaped as `<\\/` (a valid JSON escape) so no string in the
    data can close the script element."""
    d = Path(out_dir)
    blob = {name: json.loads((d / name).read_text(encoding="utf-8")) for name in FILES}
    text = json.dumps(blob, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    tag = f'<script id="dag-data" type="application/json">{text}</script>\n'
    if "</head>" not in page_html:
        raise ValueError("the page has no </head> to embed the data before")
    return page_html.replace("</head>", tag + "</head>", 1)


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

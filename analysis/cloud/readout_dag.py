"""Readout-side Mondo hierarchy over a FLAT fit's label nodes (spec 2026-10-07
§D1/§D4, WP-B; insight 0096).

The ribbon fit (`--label-set`) sees a flat forest: every label node attaches to
the synthetic root, and the bundle's closure mask — "observe the active closure
and its DAG siblings" — therefore observes EVERY node on EVERY foreground
document (exp 0135: 70.7M cells, 4 h per readout pass, and a "within-cohort"
head whose cohort is the whole foreground). The hierarchy the fit was told not
to see is what the READ needs: negatives for a disease head are the patients
with its sibling diseases under the nearest Mondo ancestor, and the stacked
product needs ancestor heads to multiply.

This module builds that hierarchy from Mondo's own graph and applies it to the
scored frames AT READOUT TIME, so the fit, the bundle, the cache key and every
hashed module are untouched (AGENTS.md "cache-key landmine"):

  readout nodes = the fit's label nodes (engine ids 0..C_fit-1, kept as they
                  are, so every per-node number is still keyed by the fit's own
                  ids) ∪ ANCESTOR HEADS appended at C_fit.. — every Mondo
                  ancestor of a label node with >= `min_descendants` label-node
                  descendants, minus two kinds of redundancy:
                    * ROOT ALIASES: an ancestor over every label node ("disease",
                      "human disease") IS the root; folded into engine id 0;
                    * RUNGS: ancestors with the same label-descendant set are the
                      same head (same y column, same cohort); only the most
                      specific of each such group is kept.
  readout DAG   = `induced_hasse_parents` (the transitive reduction of Mondo's
                  order restricted to the readout nodes); a node with no readout
                  ancestor attaches to the root.
  y_r[n]        = 1 iff some label node in desc(n) ∪ {n} is active — the
                  closure-max the hierarchical read owes R1a (a hEDS patient is a
                  member of the EDS head here, and only here).
  mask_r[n]     = 1 iff a readout PARENT of n is active (root: iff the root is
                  active) — exactly `frontier_to_label`'s closure policy on the
                  readout DAG, written as "n is observed when its cohort is
                  active", which is the identity `obs(n) = ∪_{p ∈ parents(n)}
                  pos(p)`: a node is observed on the rows where it or a sibling
                  is positive, and on no other.

Both columns are pure column arithmetic over the fit's `label` array (ADR 0047:
no UDF, nothing array-shaped on a task closure; every descendant list is an
inlined literal). `label_mask_mode="full"` runs keep an all-ones mask.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "ReadoutDag", "build_readout_dag", "load_mondo_parent_adj",
    "widen_labels_to_readout", "widened_bundle_view", "format_readout_dag_report",
    "write_readout_dag", "resolve_readout_hierarchy",
]

ROOT = 0
FIT_ROOT_CID = -1          # mondo_native_dag.MONDO_NATIVE_ROOT_CID
_PREFIX = "MONDO:"


def _curie(cid) -> str:
    return f"{_PREFIX}{int(cid):07d}"


def _cid(curie) -> int:
    return int(str(curie)[len(_PREFIX):])


@dataclass
class ReadoutDag:
    """The readout node space. Engine ids `< C_fit` ARE the fit's; ancestors
    follow in sorted-curie order. `pos_sources[n]` are the fit engine ids whose
    activation makes readout node `n` positive (the label-node descendants of
    `n`, itself included when it is a label node); `parent_int` is the readout
    DAG in readout ids (root 0 -> [])."""
    C_fit: int
    C: int
    parent_int: dict
    int2cid: dict
    name_by_id: dict
    pos_sources: list
    root_aliases: list
    stats: dict = field(default_factory=dict)

    @property
    def n_ancestors(self) -> int:
        return self.C - self.C_fit

    def to_json(self) -> dict:
        return {
            "version": 1, "C_fit": self.C_fit, "C": self.C,
            "parent_int": {str(c): list(ps) for c, ps in self.parent_int.items()},
            "int2cid": {str(i): int(c) for i, c in self.int2cid.items()},
            "name_by_id": {str(c): n for c, n in self.name_by_id.items()},
            "pos_sources": [list(map(int, s)) for s in self.pos_sources],
            "root_aliases": list(self.root_aliases),
            "stats": self.stats,
        }

    @classmethod
    def from_json(cls, d) -> "ReadoutDag":
        return cls(C_fit=int(d["C_fit"]), C=int(d["C"]),
                   parent_int={int(c): [int(p) for p in ps]
                               for c, ps in d["parent_int"].items()},
                   int2cid={int(i): int(c) for i, c in d["int2cid"].items()},
                   name_by_id={int(c): n for c, n in d["name_by_id"].items()},
                   pos_sources=[[int(x) for x in s] for s in d["pos_sources"]],
                   root_aliases=list(d.get("root_aliases", [])),
                   stats=dict(d.get("stats", {})))


def load_mondo_parent_adj(mondo_version, mondo_cache_dir="data/mondo"):
    """Mondo's `{child: [parents]}` disease adjacency + `{curie: name}` from the
    SAME version-keyed download the fit used (`anchor_selection_cloud
    ._download_cached`; cache hit on the fit's cluster, re-fetched elsewhere).
    Import-only use of the hashed mapping module, as `mondo_native_dag` does."""
    import pandas as pd
    from anchor_selection_cloud import _download_cached
    from mondo_to_omop_mapping import _disease_child_adjacency
    from mondo_native_dag import parent_adjacency

    cache = Path(mondo_cache_dir)
    edges = pd.read_csv(_download_cached(mondo_version, "mondo_edges.tsv", cache),
                        sep="\t", low_memory=False)
    nodes = pd.read_csv(_download_cached(mondo_version, "mondo_nodes.tsv", cache),
                        sep="\t", low_memory=False)
    names = {str(i): str(n) for i, n in zip(nodes["id"], nodes["name"])}
    return parent_adjacency(_disease_child_adjacency(edges, nodes)), names


def build_readout_dag(int2cid_fit, parent_adj, *, names=None, min_descendants=2,
                      root_cid=FIT_ROOT_CID) -> ReadoutDag:
    """The readout DAG over the fit's label nodes (`int2cid_fit`: engine id ->
    Mondo cid, root included) from Mondo's full `parent_adj`. Pure; see the
    module docstring for the construction and the two redundancy rules."""
    from mondo_native_dag import ancestor_closure, induced_hasse_parents

    names = dict(names or {})
    label_eng = {int(e): int(c) for e, c in int2cid_fit.items()
                 if int(e) != ROOT and int(c) != root_cid}
    if ROOT not in {int(e) for e in int2cid_fit}:
        raise ValueError("int2cid has no root (engine id 0)")
    C_fit = max(int(e) for e in int2cid_fit) + 1
    term_of = {e: _curie(c) for e, c in label_eng.items()}
    eng_of_term = {t: e for e, t in term_of.items()}
    all_labels = frozenset(label_eng)

    # Label-node descendants of every Mondo ancestor (and of every label node).
    desc: dict = {}
    anc_of: dict = {}
    for e, t in term_of.items():
        a_set = ancestor_closure(t, parent_adj)
        anc_of[t] = a_set
        for a in a_set:
            desc.setdefault(a, set()).add(e)
    # `ancestor_closure` is reflexive, so every label term lists itself.
    n_nested_labels = sum(1 for t, e in eng_of_term.items()
                          if desc.get(t, set()) - {e})

    cands = {a for a, ds in desc.items()
             if a not in eng_of_term and len(ds) >= int(min_descendants)}
    root_aliases = sorted(a for a in cands if desc[a] == all_labels)
    cands -= set(root_aliases)

    # Rungs: same label-descendant set -> keep the most specific member(s).
    by_set: dict = {}
    for a in cands:
        by_set.setdefault(frozenset(desc[a]), []).append(a)
    kept_anc = []
    n_rungs = 0
    memo: dict = {}

    def _strict_anc(t):
        if t not in memo:
            memo[t] = ancestor_closure(t, parent_adj) - {t}
        return memo[t]

    for group in by_set.values():
        if len(group) == 1:
            kept_anc.extend(group)
            continue
        minimal = [a for a in group
                   if not any(a in _strict_anc(b) for b in group if b != a)]
        kept_anc.extend(minimal)
        n_rungs += len(group) - len(minimal)
    kept_anc = sorted(kept_anc, key=_cid)

    # Readout ids: label nodes keep theirs; ancestors appended in curie order.
    r_of_term = dict(eng_of_term)
    for i, a in enumerate(kept_anc):
        r_of_term[a] = C_fit + i
    C = C_fit + len(kept_anc)
    hasse = induced_hasse_parents(set(r_of_term), parent_adj)
    parent_int = {ROOT: []}
    for t, r in r_of_term.items():
        ps = [r_of_term[p] for p in hasse.get(t, [])]
        parent_int[r] = sorted(ps) if ps else [ROOT]
    # Unmapped fit ids (none expected) attach to the root so the vector is dense.
    for e in range(C_fit):
        parent_int.setdefault(e, [ROOT] if e != ROOT else [])

    pos_sources = [[] for _ in range(C)]
    pos_sources[ROOT] = [ROOT]
    for e in range(1, C_fit):
        pos_sources[e] = (sorted({e} | desc.get(term_of[e], set()))
                          if e in term_of else [e])
    for a in kept_anc:
        pos_sources[r_of_term[a]] = sorted(desc[a])

    int2cid = {int(e): int(c) for e, c in int2cid_fit.items()}
    name_by_id = {}
    for a in kept_anc:
        int2cid[r_of_term[a]] = _cid(a)
        name_by_id[_cid(a)] = names.get(a, a)

    children: dict = {}
    for c, ps in parent_int.items():
        for p in ps:
            children.setdefault(p, []).append(c)
    depth = _depths(parent_int, C)
    sib = [len(children.get(p, [])) for p in children]
    clo = [len(_closure(c, parent_int)) for c in range(C)]
    stats = {
        "C_fit": C_fit, "C": C, "n_label": len(label_eng),
        "n_ancestor_heads": len(kept_anc),
        "n_candidates": len(cands) + len(root_aliases) + n_rungs,
        "n_root_aliases": len(root_aliases), "n_rungs_dropped": n_rungs,
        "n_nested_labels": n_nested_labels,
        "min_descendants": int(min_descendants),
        "depth_hist": {str(d): sum(1 for c in range(C) if depth[c] == d)
                       for d in sorted(set(depth.values()))},
        "max_depth": max(depth.values()),
        "closure_terms": {"mean": sum(clo) / C, "max": max(clo)},
        "sibling_group": {"n": len(sib), "mean": (sum(sib) / len(sib)) if sib else 0,
                          "max": max(sib) if sib else 0,
                          "root_children": len(children.get(ROOT, []))},
        "multi_parent": sum(1 for ps in parent_int.values() if len(ps) > 1),
    }
    return ReadoutDag(C_fit=C_fit, C=C, parent_int=parent_int, int2cid=int2cid,
                      name_by_id=name_by_id, pos_sources=pos_sources,
                      root_aliases=root_aliases, stats=stats)


def _closure(c, parent_int, memo=None):
    memo = {} if memo is None else memo
    if c not in memo:
        acc = {c}
        for p in parent_int.get(c, []):
            acc |= _closure(p, parent_int, memo)
        memo[c] = acc
    return memo[c]


def _depths(parent_int, C):
    from collections import deque
    children = {c: [] for c in range(C)}
    for c in range(C):
        for p in parent_int.get(c, []):
            children[p].append(c)
    depth = {c: None for c in range(C)}
    dq = deque([(ROOT, 0)])
    while dq:
        n, d = dq.popleft()
        if depth[n] is not None and depth[n] <= d:
            continue
        depth[n] = d
        for ch in children[n]:
            dq.append((ch, d + 1))
    return {c: (-1 if d is None else d) for c, d in depth.items()}


def format_readout_dag_report(rdag: ReadoutDag) -> str:
    s = rdag.stats
    return (
        f"[readout dag]  Mondo hierarchy over the flat fit (spec 2026-10-07 §D1, "
        f"WP-B): label nodes={s['n_label']} + ancestor heads={s['n_ancestor_heads']} "
        f"-> C={s['C']} (fit C={s['C_fit']}); candidates={s['n_candidates']} "
        f"(root aliases folded={s['n_root_aliases']}, rungs dropped="
        f"{s['n_rungs_dropped']}); label nodes with a label-node ancestor="
        f"{s['n_nested_labels']}\n"
        f"[readout dag]  depth hist={s['depth_hist']} max={s['max_depth']}; "
        f"closure terms mean={s['closure_terms']['mean']:.2f} "
        f"max={s['closure_terms']['max']}; sibling groups={s['sibling_group']['n']} "
        f"mean={s['sibling_group']['mean']:.1f} max={s['sibling_group']['max']} "
        f"(root children={s['sibling_group']['root_children']}); "
        f"multi-parent nodes={s['multi_parent']}"
        + (f"\n[readout dag]  root aliases: {', '.join(rdag.root_aliases)}"
           if rdag.root_aliases else ""))


def write_readout_dag(run_dir, rdag: ReadoutDag, name="readout_dag.json") -> Path:
    """`<run>/readout_dag.json`: ids, names, parents, sources (ontology ids and
    labels only — no patient data), so the per-node rows of a hierarchical
    readout resolve to names and the record can be re-read."""
    p = Path(run_dir) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rdag.to_json()))
    return p


def resolve_readout_hierarchy(choice, manifest) -> bool:
    """`--readout-hierarchy {auto,mondo,flat}` -> whether the Mondo readout DAG
    is applied. `auto` (the default) turns it on exactly for label-set (flat
    ribbon) fits, read from the manifest's `corpus_manifest.label_set`, so every
    nested run's readout is byte-identical to what it was."""
    choice = str(choice or "auto")
    if choice == "mondo":
        return True
    if choice == "flat":
        return False
    cm = manifest.get("corpus_manifest") or {}
    return bool(cm.get("label_set"))


def widen_labels_to_readout(df, rdag: ReadoutDag, *, label_col="label",
                            mask_col="labelMask", mask_mode="closure"):
    """Replace the fit-width `label`/`labelMask` (C_fit doubles) with the
    readout-width pair (C doubles) on a scored frame. Column arithmetic only.

    `label_r[n]` is the closure-max over `pos_sources[n]`; `mask_r[n]` is the
    max over `n`'s readout parents' `label_r` (root: its own) — see the module
    docstring for why that is the closure mask on the readout DAG. Under
    `mask_mode="full"` the mask is all ones (a full-mask fit stays full)."""
    import json as _json
    from pyspark.sql import functions as F

    C = int(rdag.C)
    # Two constant index tables, each ONE string literal parsed by `from_json`
    # (constant-folded by the optimizer): `src[n]` = pos_sources[n], `par[n]` =
    # n's readout parents (the root lists itself). Every per-row computation is
    # then a single higher-order `transform` — a compact plan whatever C is. The
    # first version spelled out C `arrays_overlap` / `greatest` expressions; at
    # C=702 its generated Java blew the 64 KB method limit (0135 launch 3:
    # janino InternalCompilerException, interpreted fallback, ~2 min lost).
    src = [[int(x) for x in rdag.pos_sources[n]] for n in range(C)]
    par = [([int(p) for p in rdag.parent_int.get(n, [])] if n != ROOT else [ROOT])
           or [ROOT] for n in range(C)]
    src_lit = F.from_json(F.lit(_json.dumps(src)), "array<array<int>>")
    par_lit = F.from_json(F.lit(_json.dumps(par)), "array<array<int>>")

    lab = F.col(label_col)
    tmp = f"__{label_col}_r"
    out = df.withColumn(tmp, F.transform(
        src_lit,
        lambda ss: F.exists(ss, lambda i: lab[i] > 0).cast("double")))
    yr = F.col(tmp)
    if str(mask_mode) == "full":
        mask_expr = F.array_repeat(F.lit(1.0), C)
    else:
        mask_expr = F.transform(
            par_lit,
            lambda ps: F.exists(ps, lambda q: yr[q] > 0).cast("double"))
    return (out.withColumn(mask_col, mask_expr)
               .drop(label_col)
               .withColumnRenamed(tmp, label_col))


def widened_bundle_view(bundle, rdag: ReadoutDag):
    """A shallow copy of the bundle whose DAG bridge (`parent_int`, `int2cid`,
    `cid2int`, `name_by_id`) is the READOUT node space, for the driver's
    per-node reporters. Frames are untouched (the widening is a frame op)."""
    import copy
    view = copy.copy(bundle)
    view.parent_int = {int(c): list(ps) for c, ps in rdag.parent_int.items()}
    view.int2cid = dict(rdag.int2cid)
    view.cid2int = {c: i for i, c in rdag.int2cid.items()}
    nb = dict(getattr(bundle, "name_by_id", {}) or {})
    nb.update(rdag.name_by_id)
    view.name_by_id = nb
    return view

"""Write a small SYNTHETIC `dag-gated-export/1` directory for developing the browser page.

Not model output and not patient data: every name, weight and support band here is made
up by a seeded random generator. It exercises the format of
docs/superpowers/specs/2026-11-11-dag-gated-dashboard-export.md: ~60 nodes in a
multi-parent DAG with every status (0 topic, 1 pruned, 2 spliced, 3 unsupported),
3 domains, 5 background topics, columnar arrays, delta-coded ids and log-byte weights.

    python dag-dashboard/dev/make_sample.py [out_dir]   # default dag-dashboard/dev/sample
"""
import gzip
import json
import math
import os
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "sample")
rng = np.random.default_rng(11)
DOMAINS = ["condition", "drug", "measurement"]
V_PER = [240, 180, 120]
TOP_BETA, TOP_LIFT, LIFT_POOL = 25, 25, 200
N_BG = 5

# ---- the DAG: 4 systems, 3 branches each, 3 leaves per branch, plus cross-links ----
SYSTEMS = ["cardiovascular disorder", "respiratory disorder", "endocrine disorder", "renal disorder"]
BRANCH = ["inflammatory", "congenital", "degenerative"]
names, parents = [], []


def add(name, ps):
    names.append(name)
    parents.append([int(p) for p in ps])
    return len(names) - 1


root = add("disease (synthetic)", [])
sys_ix = [add(s, [root]) for s in SYSTEMS]
branch_ix = [add(f"{b} {s}", [si]) for s, si in zip(SYSTEMS, sys_ix) for b in BRANCH]
leaf_ix = [add(f"{names[bi]}, type {chr(65 + j)}", [bi]) for bi in branch_ix for j in range(3)]
for li in rng.choice(leaf_ix, 10, replace=False):          # multi-parent cross-links
    other = int(rng.choice([b for b in branch_ix if b not in parents[li]]))
    parents[li].append(other)
for li in rng.choice(leaf_ix, 6, replace=False):           # a few depth-4 nodes
    add(f"{names[li]}, subtype 1", [li])
N = len(names)

children = [[] for _ in range(N)]
for i, ps in enumerate(parents):
    for p in ps:
        children[p].append(i)
depth = [0] * N
for i in range(N):  # parents were added before children: one pass gives the shortest depth
    if parents[i]:
        depth[i] = 1 + min(depth[p] for p in parents[i])

# ---- statuses ---------------------------------------------------------------------
status = [0] * N
leaves = [i for i in range(N) if not children[i]]
for i in rng.choice(leaves, 7, replace=False):
    status[i] = 1                                           # pruned: too few documents
for i in rng.choice(branch_ix, 3, replace=False):
    status[i] = 2                                           # spliced into its children
for i in rng.choice([x for x in leaf_ix if status[x] == 0], 4, replace=False):
    status[i] = 3                                           # unsupported: flat topic
status[sys_ix[3]] = 2


def ancestors(x):
    a, s = set(), list(parents[x])
    while s:
        q = s.pop()
        if q not in a:
            a.add(q)
            s.extend(parents[q])
    return a


def topic_ancestors(i):
    """Nearest ancestors with a topic, transitively reduced (the kept DAG's parents)."""
    out, seen, st = set(), set(), list(parents[i])
    while st:
        p = st.pop()
        if p in seen:
            continue
        seen.add(p)
        if status[p] == 0:
            out.add(p)
            continue
        st.extend(parents[p])
    return sorted(p for p in out if not any(p in ancestors(q) for q in out if q != p))


kept_parents, landing = [], []
for i in range(N):
    if status[i] == 0:
        kept_parents.append(topic_ancestors(i))
        landing.append(None)
    elif status[i] == 2:
        kept_parents.append(None)
        land = [c for c in children[i] if status[c] == 0] or topic_ancestors(i)
        landing.append(sorted(land))
    else:
        kept_parents.append(None)
        landing.append(topic_ancestors(i) if rng.random() > 0.25 else [])   # [] = background

support_raw = [int(rng.lognormal(7 - 1.2 * depth[i], 1.0)) + 1 for i in range(N)]
for i in range(N):
    if status[i] == 1:
        support_raw[i] = int(rng.integers(1, 19))


def band(c):
    return 0 if c <= 0 else 20 * ((c - 1) // 20) + 10


support = [band(c) for c in support_raw]
support_frac = [int(round(1000 * rng.beta(2, 5))) if status[i] == 0 else None for i in range(N)]

# ---- vocabulary and topics --------------------------------------------------------
node_topic_ix = [i for i in range(N) if status[i] == 0]
K = N_BG + len(node_topic_ix)
lam = [np.zeros((K, v)) for v in V_PER]
bg_base = [rng.gamma(0.3, 1.0, v) for v in V_PER]
for k in range(N_BG):
    for d in range(3):
        lam[d][k] = rng.gamma(0.2, 1.0, V_PER[d]) * 5 + bg_base[d]
row_of = {}
for r, i in enumerate(node_topic_ix):   # a topic parent precedes its children
    row = N_BG + r
    row_of[i] = row
    for d in range(3):
        kp = kept_parents[i]
        base = np.mean([lam[d][row_of[p]] for p in kp], axis=0) if kp else bg_base[d]
        spike = np.zeros(V_PER[d])
        spike[rng.choice(V_PER[d], 6, replace=False)] = rng.gamma(2, 8, 6)
        lam[d][row] = 0.6 * base * rng.gamma(4, 0.25, V_PER[d]) + spike + 0.01
beta = [lm / lm.sum(axis=1, keepdims=True) for lm in lam]
pw = [lm.sum(axis=0) / lm.sum() for lm in lam]          # the model's own term marginal

shipped = []                            # per topic, per domain: local term ids
for k in range(K):
    per = []
    for d in range(3):
        order = list(np.argsort(-beta[d][k]))
        top = order[:TOP_BETA]
        pool = [w for w in order[:LIFT_POOL] if w not in top]
        lift = sorted(pool, key=lambda w: -beta[d][k][w] / pw[d][w])[:TOP_LIFT]
        per.append(sorted(int(w) for w in set(top) | set(lift)))
    shipped.append(per)
used = [sorted({w for k in range(K) for w in shipped[k][d]}) for d in range(3)]


def enc(x, lo, hi):
    return int(round(255 * (math.log(x) - lo) / (hi - lo))) if hi > lo else 0


lo_f = [math.log(pw[d][used[d]].min()) for d in range(3)]
hi_f = [math.log(pw[d][used[d]].max()) for d in range(3)]
lo_w = [min(math.log(beta[d][k][w]) for k in range(K) for w in shipped[k][d]) for d in range(3)]
hi_w = [max(math.log(beta[d][k][w]) for k in range(K) for w in shipped[k][d]) for d in range(3)]
vocab_index, voc = {}, {"domain": [], "code": [], "name": [], "freq": []}
for d in range(3):
    for w in used[d]:
        vocab_index[(d, w)] = len(voc["domain"])
        voc["domain"].append(d)
        voc["code"].append(f"RX{w:04d}" if d == 1 else 1000 * (d + 1) + w)
        voc["name"].append(f"{DOMAINS[d]} term {w:03d}")
        voc["freq"].append(enc(pw[d][w], lo_f[d], hi_f[d]))
topics = {"node": [None] * N_BG + node_topic_ix, "d": []}
for k in range(K):
    per = []
    for d in range(3):
        gids = sorted(vocab_index[(d, w)] for w in shipped[k][d])
        local = {vocab_index[(d, w)]: w for w in shipped[k][d]}
        delta = [gids[0]] + [b - a for a, b in zip(gids, gids[1:])]
        per.append([delta, [enc(beta[d][k][local[g]], lo_w[d], hi_w[d]) for g in gids]])
    topics["d"].append(per)
nodes = {
    "id": [f"MONDO:99{i:05d}" for i in range(N)],
    "name": names, "parents": parents, "kept_parents": kept_parents, "depth": depth,
    "status": status, "landing": landing, "support": support, "support_frac": support_frac,
    "topic": [N_BG + node_topic_ix.index(i) if status[i] == 0 else None for i in range(N)],
}

os.makedirs(OUT, exist_ok=True)
sizes = {}
for fn, obj in [("nodes.json", nodes), ("vocab.json", voc), ("topics.json", topics)]:
    raw = json.dumps(obj, separators=(",", ":")).encode()
    with open(os.path.join(OUT, fn), "wb") as f:
        f.write(raw)
    sizes[fn] = [len(raw), len(gzip.compress(raw))]
manifest = {
    "schema": "dag-gated-export/1", "run": "SYNTHETIC sample (dev/make_sample.py)",
    "created_utc": "2026-11-11T00:00:00Z", "mondo_version": "synthetic", "cdr": "synthetic",
    "K": K, "n_bg": N_BG, "tpn": len(node_topic_ix),
    "domains": [{"name": d, "shipped_terms": len(used[i])} for i, d in enumerate(DOMAINS)],
    "selection": {"top_beta": TOP_BETA, "top_lift": TOP_LIFT, "lift_pool": LIFT_POOL},
    "weight_codec": {"kind": "log-byte",
                     "per_domain": [{"lo": lo_w[d], "hi": hi_w[d]} for d in range(3)]},
    "freq_codec": {"kind": "log-byte",
                   "per_domain": [{"lo": lo_f[d], "hi": hi_f[d]} for d in range(3)]},
    "disclosure": {"min_cell": 20, "support": "20-wide bands shown at the midpoint "
                   "(1-20 -> 10 '<=20', 21-40 -> 30, ...)"},
    "sizes": sizes,
    "notes": "SYNTHETIC: seeded random names, weights and supports; no model, no patient data.",
}
with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, separators=(",", ":"))
print(f"wrote {OUT}: {N} nodes, statuses {[status.count(s) for s in range(4)]}, K={K}, "
      f"vocab={len(voc['domain'])}, sizes={sizes}")

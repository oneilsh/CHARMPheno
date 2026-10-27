"""Leaves-first gate schedule vs the single-round gated fit, on a planted tree.

Exp 0152 read 3b: on the whole-Mondo nested DAG a parent block sometimes holds
its dominant child's vocabulary (cardiomyopathy reads as HCM; migraine disorder
reads as epilepsy) while the child block fills with comorbidity. The idea under
test: fit in ROUNDS, opening the gate by node HEIGHT (leaves first, height 0,
then their parents, ...), so each child claims its own words before the parent
block exists, and the parent opens onto what is left — the shared stratum.

Step 1 of that plan: does a planted tree reproduce the failure under the
current single-round gate, and does the schedule fix it? The fit runs through
the production `VIRunner` on LOCAL SPARK with the production mini-batch regime
(subsample 0.1, tau0 64, kappa 0.51; full-batch EM is a different optimizer and
can land elsewhere — `--full-batch` keeps it only as a comparison arm). The
model is the production `DagGatedOnlineLDA` E-step/M-step with the schedule added.

Schedule mechanics (prototype; nothing in spark-vi changes):
- the gate state rides `global_params` (the per-iteration broadcast), so the
  executors see it every iteration: `gate_iter` counts iterations and
  `gate_open_at[k]` is the iteration topic k opened (-1 = closed);
- `allowed_set` keeps only closure nodes of height <= the current level;
- a closed block's lambda is FROZEN (its zero sufficient statistics would
  otherwise pull it to the flat prior every step);
- `+seed`: a block is seeded the moment it opens with the word distribution of
  its subtree's documents, scaled to a typical open node topic's mass (an
  unseeded late block opens flat onto sharp competitors, wins no
  responsibility, and decays to the prior: dead on arrival);
- `--block-clock`: each open topic's Robbins-Monro step counts from when it
  opened, rho_k = (tau0 + t - t_open_k + 1)^-kappa, so a block opened late is
  not stuck with the global schedule's already-small step.

Metrics per node block u (its topic row, normalized): `own` = mass on u's
planted signature; `kids` = mass on its children's signatures (the leak this
targets); `comorb` = mass on the comorbidity clusters. A parent that "claims its
child" has high `kids`; a child that "lost its words" has low `own`, high `comorb`.

    PYTHONPATH=spark-vi poetry run python analysis/local/sim_leaves_first.py --seeds 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "spark-vi"))

from spark_vi.models.topic.dag_placement import DagLayout  # noqa: E402
from spark_vi.models.topic.dag_gated_lda import DagGatedOnlineLDA  # noqa: E402
from spark_vi.models.topic.types import GatedBOWDocument  # noqa: E402

# Planted tree: two branches under root. P(1) has a dominant child (3), a minor
# child (4), a rare child (5); 3 has its own subtype (8). Q(2) has two children.
PARENT = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 3}
NAMES = {1: "P", 2: "Q", 3: "P.dominant", 4: "P.minor", 5: "P.rare",
         6: "Q.a", 7: "Q.b", 8: "P.dominant.sub"}
# Share of documents whose MOST-SPECIFIC code is each node (0 = background doc).
LABEL_SHARE = {0: 0.40, 1: 0.05, 2: 0.04, 3: 0.20, 4: 0.06, 5: 0.015,
               6: 0.10, 7: 0.08, 8: 0.05}
V = 400


def heights(lay):
    """Longest path down to a leaf (leaves 0). The schedule's clock: by height,
    not depth, because real leaves sit at very different depths."""
    h = {}

    def rec(u):
        if u not in h:
            ch = [c for c in lay.children.get(u, []) if c != 0]
            h[u] = 0 if not ch else 1 + max(rec(c) for c in ch)
        return h[u]
    for u in lay.nodes:
        rec(u)
    return h


class StagedLayout(DagLayout):
    """`allowed_set` restricted to closure nodes of height <= `max_height`
    (None = every node: the production gate)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.height = heights(self)
        self.max_height = None
        self.topic_height = np.full(self.K, -1)          # bg topics: -1, always open
        for u in self.nodes:
            self.topic_height[self.block[u]] = self.height[u]

    def allowed_set(self, frontier):
        if self.max_height is None:
            return super().allowed_set(frontier)
        al = set(range(self.n_bg))
        for f in frontier:
            for u in self.closure(f):
                if u != 0 and self.height[u] <= self.max_height:
                    al.update(self.block[u])
        return np.array(sorted(al), dtype=int)


class StagedGated(DagGatedOnlineLDA):
    """DagGatedOnlineLDA plus a height schedule carried in global_params.

    `schedule[t]` is the max open height at iteration t (None = all open).
    tau0 / kappa are only used by the per-block clock."""

    def __init__(self, lay, vocab_size, *, schedule, block_clock=False,
                 tau0=64.0, kappa=0.51, open_seed=None, **kw):
        super().__init__(lay, vocab_size, **kw)
        self.schedule = list(schedule)
        self.block_clock = bool(block_clock)
        self.tau0, self.kappa = float(tau0), float(kappa)
        # {topic k: (V,) word distribution} to seed a block the moment it opens
        # (None = keep its init). A block opened onto already-sharp children and
        # background starts flat, wins no responsibility and decays to the
        # prior; a seed gives it a competitive start.
        self.open_seed = open_seed

    def _level(self, t):
        return self.schedule[t] if t < len(self.schedule) else None

    def _open_mask(self, level):
        th = self.lay.topic_height
        return np.ones(self.K, bool) if level is None else th <= level

    def initialize_global(self, data_summary):
        gp = super().initialize_global(data_summary)
        gp["gate_iter"] = np.array(0)
        gp["gate_open_at"] = np.where(self._open_mask(self._level(0)), 0, -1)
        return gp

    def local_update(self, rows, global_params):
        self.lay.max_height = self._level(int(global_params["gate_iter"]))
        return super().local_update(rows, global_params)

    def update_global(self, global_params, target_stats, learning_rate):
        t = int(global_params["gate_iter"])
        open_now = global_params["gate_open_at"] >= 0
        lam_old = global_params["lambda"]
        new = super().update_global(global_params, target_stats, learning_rate)
        lam = new["lambda"]
        if self.block_clock:
            # Re-blend each open topic with its own step: the parent's natural
            # gradient target is recovered from its blend at the global step.
            tgt = (lam - (1.0 - learning_rate) * lam_old) / learning_rate
            age = t - global_params["gate_open_at"]
            rho_k = (self.tau0 + age + 1.0) ** -self.kappa
            blended = (1.0 - rho_k[:, None]) * lam_old + rho_k[:, None] * tgt
            lam = np.where(open_now[:, None], blended, lam)
        lam = np.where(open_now[:, None], lam, lam_old)        # freeze closed blocks
        open_at = global_params["gate_open_at"].copy()
        nxt = self._open_mask(self._level(t + 1))
        newly = nxt & (open_at < 0)
        open_at[newly] = t + 1
        if self.open_seed is not None and newly.any():
            # Seed at the mass of a typical open node topic, so the new block
            # competes on equal footing (scale-free otherwise).
            node_open = open_now.copy()
            node_open[:self.lay.n_bg] = False
            mass = float(lam[node_open].sum(axis=1).mean()) if node_open.any() \
                else float(lam.sum(axis=1).mean())
            lam = lam.copy()
            for k in np.flatnonzero(newly):
                lam[k] = self.eta + mass * self.open_seed[k]
        new.update({"lambda": lam, "gate_iter": np.array(t + 1),
                    "gate_open_at": open_at})
        return new


def plant(seed, *, n_docs=4000, doc_len=60, sig=8, n_comorb=8, comorb_len=12,
          parent_share=0.35):
    """Documents over the planted tree. Each doc: a common pool draw (bg), two
    random comorbidity clusters (not tied to the label — the stuff a starved
    block absorbs), and signature draws along closure(label) where the label's
    OWN signature gets `1 - parent_share` and its ancestors split the rest
    (an ancestor's words are the shared stratum of its subtree)."""
    rng = np.random.default_rng(seed)
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    nodes = lay.nodes
    common = np.arange(0, 100)
    comorb = [np.arange(100 + i * 10, 100 + (i + 1) * 10) for i in range(n_comorb)]
    base = 100 + n_comorb * 10
    sigs = {u: np.arange(base + i * sig, base + (i + 1) * sig)
            for i, u in enumerate(nodes)}
    assert base + len(nodes) * sig <= V
    labels = np.array(list(LABEL_SHARE))
    p = np.array([LABEL_SHARE[u] for u in labels])
    p /= p.sum()
    comorb_p = rng.dirichlet(np.ones(n_comorb) * 2.0)
    docs = []
    for _ in range(n_docs):
        y = int(rng.choice(labels, p=p))
        toks = [rng.choice(common, size=doc_len // 2)]
        for c in rng.choice(n_comorb, size=2, replace=False, p=comorb_p):
            toks.append(rng.choice(comorb[c], size=comorb_len // 2))
        if y != 0:
            n_sig = doc_len - doc_len // 2
            anc = [u for u in lay.closure(y) if u not in (0, y)]
            n_own = int(round(n_sig * (1 - parent_share))) if anc else n_sig
            toks.append(rng.choice(sigs[y], size=n_own))
            for a in anc:
                toks.append(rng.choice(sigs[a], size=max(1, (n_sig - n_own) // len(anc))))
        idx, cnt = np.unique(np.concatenate(toks), return_counts=True)
        docs.append(GatedBOWDocument(idx.astype(np.int32), cnt.astype(np.float64),
                                     int(cnt.sum()),
                                     frontier=frozenset() if y == 0 else frozenset({y})))
    return docs, sigs, comorb


def block_metrics(lam, lay, sigs, comorb):
    beta = lam / lam.sum(axis=1, keepdims=True)
    com = np.concatenate(comorb)
    rows = {}
    for u in lay.nodes:
        b = beta[lay.block[u][0]]
        kids = [c for c in lay.children.get(u, []) if c != 0]
        rows[u] = {
            "own": float(b[sigs[u]].sum()),
            "kids": float(sum(b[sigs[c]].sum() for c in kids)) if kids else 0.0,
            "comorb": float(b[com].sum()),
        }
    return rows


def schedule_for(lay, total, mode, warm):
    """Per-iteration max height. `single`: None throughout (production).
    `leaves`: `warm` iterations at each height level, then everything open."""
    if mode == "single":
        return [None] * total
    out = []
    for h in range(max(lay.height.values())):
        out += [h] * warm
    return (out + [None] * total)[:total]


def subtree_seed(docs, lay):
    """{topic k: normalized word frequency over the documents in node u's
    subtree} for each node block — the empirical distribution the block's
    gate admits, available from one pass (the same per-node closure counts the
    spectral seed already gathers)."""
    sub = {u: lay.subtree(u) for u in lay.nodes}
    out = {}
    for u in lay.nodes:
        acc = np.zeros(V)
        for d in docs:
            if any(f in sub[u] for f in d.frontier):
                acc[d.indices] += d.counts
        out[lay.block[u][0]] = acc / max(acc.sum(), 1.0)
    return out


def fit(sc, docs, mode, *, seed, total, warm, frac, tau0, kappa, block_clock,
        n_bg, full_batch, open_seed=False):
    from spark_vi.core import VIConfig, VIRunner
    lay = StagedLayout(PARENT, n_bg=n_bg, tpn=1)
    m = StagedGated(lay, V, schedule=schedule_for(lay, total, mode, warm),
                    block_clock=block_clock, tau0=tau0, kappa=kappa,
                    open_seed=subtree_seed(docs, lay) if open_seed else None,
                    alpha=0.1, eta=0.02, random_seed=seed)
    np.random.seed(seed)
    cfg = VIConfig(max_iterations=total, learning_rate_tau0=tau0,
                   learning_rate_kappa=kappa,
                   mini_batch_fraction=None if full_batch else frac,
                   convergence_tol=1e-12, random_seed=seed)
    rdd = sc.parallelize(docs, 4).cache()
    rdd.count()
    res = VIRunner(m, config=cfg).fit(rdd)
    rdd.unpersist()
    return res.global_params["lambda"], lay


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--total", type=int, default=100)
    ap.add_argument("--warm", type=int, default=25,
                    help="iterations per height level before the next opens")
    ap.add_argument("--frac", type=float, default=0.1, help="mini-batch fraction")
    ap.add_argument("--tau0", type=float, default=64.0)
    ap.add_argument("--kappa", type=float, default=0.51)
    ap.add_argument("--n-docs", type=int, default=4000)
    ap.add_argument("--parent-share", type=float, default=0.35)
    ap.add_argument("--n-bg", type=int, default=2)
    ap.add_argument("--block-clock", action="store_true",
                    help="add a leaves+seed arm with per-block step clocks")
    ap.add_argument("--full-batch", action="store_true",
                    help="also run full-batch EM arms (comparison only)")
    args = ap.parse_args(argv)

    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.master("local[4]").appName("sim-leaves-first")
             .config("spark.ui.enabled", "false")
             .config("spark.sql.shuffle.partitions", "4").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    # (mode, block_clock, full_batch, open_seed)
    arms = [("single", False, False, False), ("leaves", False, False, False),
            ("leaves", False, False, True)]
    if args.block_clock:
        arms.append(("leaves", True, False, True))
    if args.full_batch:
        arms += [("single", False, True, False), ("leaves", False, True, True)]
    res = {}

    def label(m, c, fb, sd):
        return m + ("+seed" if sd else "") + ("+clock" if c else "") + (" FB" if fb else "")
    try:
        for s in range(args.seeds):
            docs, sigs, comorb = plant(s, n_docs=args.n_docs,
                                       parent_share=args.parent_share)
            for arm in arms:
                mode, clock, fb, seeded = arm
                lam, lay = fit(spark.sparkContext, docs, mode, seed=s,
                               total=args.total, warm=args.warm, frac=args.frac,
                               tau0=args.tau0, kappa=args.kappa, block_clock=clock,
                               n_bg=args.n_bg, full_batch=fb, open_seed=seeded)
                for u, r in block_metrics(lam, lay, sigs, comorb).items():
                    res.setdefault(arm, {}).setdefault(u, []).append(r)
                print(f"[sim] seed {s} {label(*arm)} done", flush=True)
    finally:
        spark.stop()

    print(f"\n# leaves-first sim — VIRunner local Spark, mini-batch {args.frac}, "
          f"tau0 {args.tau0}, kappa {args.kappa}, {args.total} iters, "
          f"{args.warm}/level, {args.seeds} seeds, {args.n_docs} docs, "
          f"parent_share {args.parent_share}, n_bg {args.n_bg}")
    print("mean over seeds, per cell: own · kids · comorb (mass on own signature · "
          "children's signatures · comorbidity clusters)")
    lay = StagedLayout(PARENT, n_bg=2, tpn=1)
    print(f"{'node':<16} h " + "".join(f"| {label(*a):<17}" for a in arms))
    for u in lay.nodes:
        cells = []
        for key in arms:
            rs = res[key][u]
            vals = [np.mean([r[k] for r in rs]) for k in ("own", "kids", "comorb")]
            cells.append("| " + " ".join(f"{v:.2f}" for v in vals) + "    ")
        print(f"{NAMES[u]:<16} {lay.height[u]} " + "".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Sibling-support diagnostic for the residual degenerate nodes (exp 0109).

Exp 0109's splice removed exactly the STRUCTURAL only-child class nodes (143 of
them, every one degenerate) and left 620 degenerates standing — refuting the
"{root} ∪ only-children exactly" characterization. The standing hypothesis is
OBSERVATIONAL only-childness: a class node whose siblings exist in the graph but
contribute no observed rows is an only child in the data, and under closure
masking its observed train cell is single-class exactly as if the siblings were
not there. This tool tests that hypothesis directly, offline, against the run's
own cached bundle:

  1. one pass over the TRAIN split's label/labelMask columns → per-node
     (n_obs, n_pos) — the same quantities the readout's moments pass derives,
     minus the θ moments it doesn't need;
  2. the readout's exact degeneracy rule ((n_obs<=0)|(n_pos<=0)|(n_pos>=n_obs));
  3. each degenerate node bucketed by what the bundle's own `parent_int`
     children map says about it:
       root                      — structural, stays degenerate by design;
       leaf                      — a terminal with a degenerate cell (its own
                                   story: no positives, or no negatives, in the
                                   train split);
       class ≤1 supported child  — HYPOTHESIS-CONSISTENT (an observational
                                   only-child: at most one child has any train
                                   positives);
       class ≥2 supported chn    — UNEXPLAINED (a third mechanism; named in the
                                   output for eyeballing).

If the ≥2-supported bucket is ~empty, the hypothesis holds and the data-aware
collapse (splice on observed sibling support, not graph structure) is justified
as the reduction's v2. If it is large, there is another mechanism to find first.

Bundle located exactly like gated_pc_readout: recompute the cache key from the
run's manifest and REQUIRE a HIT (a diagnostic should never pay a rebuild — run
the readout first if the cache is cold).
"""
from __future__ import annotations

import argparse

import numpy as np

from _driver_common import _phase, configure_logging, make_spark_session
from gated_pc_readout import bundle_key_from_manifest, resolve_run_dir

import json


def classify_degenerates(n_obs, n_pos, parent_int, C):
    """Bucket every degenerate node by the sibling-support hypothesis.

    Pure function of the counts and the parent map so it is unit-testable
    without Spark. Returns ``(buckets, per_node)`` where `buckets` maps bucket
    name -> sorted engine ids and `per_node` maps engine id ->
    ``(n_children, n_supported_children)`` for the class-node buckets.
    """
    n_obs = np.asarray(n_obs, float)
    n_pos = np.asarray(n_pos, float)
    degenerate = (n_obs <= 0) | (n_pos <= 0) | (n_pos >= n_obs)

    children: dict[int, list[int]] = {c: [] for c in range(C)}
    roots = []
    for c in range(C):
        ps = parent_int.get(c, [])
        if not ps:
            roots.append(c)
        for p in ps:
            if 0 <= int(p) < C:
                children[int(p)].append(c)

    buckets = {"root": [], "leaf": [], "class_le1_supported": [],
               "class_ge2_supported": []}
    per_node = {}
    for c in range(C):
        if not degenerate[c]:
            continue
        if c in roots:
            buckets["root"].append(c)
            continue
        kids = children[c]
        if not kids:
            buckets["leaf"].append(c)
            continue
        supported = [k for k in kids if n_pos[k] > 0]
        per_node[c] = (len(kids), len(supported))
        key = ("class_le1_supported" if len(supported) <= 1
               else "class_ge2_supported")
        buckets[key].append(c)
    return buckets, per_node


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--cache-uri", default=None)
    p.add_argument("--label-col", default="label")
    p.add_argument("--mask-col", default="labelMask")
    args = p.parse_args(argv)
    configure_logging()

    run_dir = resolve_run_dir(args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    C = int(manifest["C"])
    cm = manifest.get("corpus_manifest") or {}
    cache_uri = args.cache_uri or cm.get("cache_uri")
    key = bundle_key_from_manifest(manifest)

    with make_spark_session(app_name="diag-sibling-support") as spark:
        from _case_finding_cache import try_load

        with _phase("load cached bundle"):
            bundle = try_load(spark, cache_uri, key)
            if bundle is None:
                print(f"[diag] ERROR: cache MISS at {cache_uri}/{key} — this "
                      "diagnostic never rebuilds; run gated_pc_readout first so "
                      "the bundle is cached.", flush=True)
                return 2

        with _phase("train label counts"):
            # One (C,)+(C,) aggregate over label/mask — the moments pass minus
            # the θ terms it does not need. Vector columns arrive as Spark ML
            # vectors or plain sequences depending on the writer; toArray covers
            # both.
            cols = (args.label_col, args.mask_col)

            def _local(rows, _C=C, _cols=cols):
                n_obs = np.zeros(_C)
                n_pos = np.zeros(_C)
                for r in rows:
                    y = np.asarray(getattr(r[_cols[0]], "toArray",
                                           lambda: r[_cols[0]])(), float)
                    m = np.asarray(getattr(r[_cols[1]], "toArray",
                                           lambda: r[_cols[1]])(), float)
                    n_obs += m
                    n_pos += y * m
                return [(n_obs, n_pos)]

            n_obs, n_pos = (bundle.train_df.select(*cols).rdd
                            .mapPartitions(_local)
                            .treeAggregate((np.zeros(C), np.zeros(C)),
                                           lambda a, b: (a[0] + b[0], a[1] + b[1]),
                                           lambda a, b: (a[0] + b[0], a[1] + b[1]),
                                           depth=2))

        buckets, per_node = classify_degenerates(n_obs, n_pos,
                                                 bundle.parent_int, C)
        n_deg = sum(len(v) for v in buckets.values())
        print(f"[diag] degenerate nodes on the TRAIN split: {n_deg}/{C} "
              "(compare against the readout banner's count)", flush=True)
        for k in ("root", "leaf", "class_le1_supported", "class_ge2_supported"):
            print(f"[diag]   {k}: {len(buckets[k])}", flush=True)
        print("[diag] hypothesis-consistent = root + leaf + class_le1_supported; "
              "class_ge2_supported is the UNEXPLAINED bucket.", flush=True)

        def _name(c):
            cid = bundle.int2cid.get(c, c)
            return f"{cid} {bundle.name_by_id.get(cid, '?')}"

        for c in buckets["class_ge2_supported"][:20]:
            nk, ns = per_node[c]
            print(f"[diag]   UNEXPLAINED {_name(c)}: {ns}/{nk} children "
                  f"supported, n_obs={n_obs[c]:.0f} n_pos={n_pos[c]:.0f}",
                  flush=True)
        # A few consistent examples too, so the pattern is eyeball-checkable.
        for c in buckets["class_le1_supported"][:5]:
            nk, ns = per_node[c]
            print(f"[diag]   consistent {_name(c)}: {ns}/{nk} children "
                  f"supported, n_obs={n_obs[c]:.0f} n_pos={n_pos[c]:.0f}",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

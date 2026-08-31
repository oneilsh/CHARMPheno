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
  3. each degenerate node bucketed by SIBLING support (see
     `classify_degenerates` — v2 of this diagnostic; v1 bucketed by CHILD
     support and its own first production run refuted it): root / no_pos
     (split-or-label starvation, a separate story) / all-positive with no
     supported sibling (the unified cohort-collapse mechanism) / all-positive
     DESPITE a supported sibling (genuinely unexplained, named).

If the with-supported-sibling bucket is ~empty, one mechanism explains every
all-positive degenerate — cohort == own closure because no sibling contributes
observed rows — and the data-aware reduction (v2 of the collapse) has its spec.
The first production run's arithmetic already leans that way: 763 (exp 0104) =
1 root + 143 structural only-children (spliced) + 606 leaves + 13 all-positive
classes, and the 13 all had n_pos == n_obs exactly.

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
    """Bucket every degenerate node by SIBLING support (v2 of this diagnostic).

    v1 bucketed by CHILD support and its first production run refuted itself:
    the 13 "unexplained" all-positive classes all had fully-supported children —
    because degeneracy does not run through a node's children at all. Under
    closure masking a node's observed cohort comes from its PARENT's closure,
    so its cell is all-positive exactly when no SIBLING contributes observed
    rows: the cohort collapses to the node's own closure, where everyone is
    positive by construction. (The smoking gun in that run: a degenerate child
    whose n_obs equaled its parent's almost exactly — its cohort WAS the
    parent's closure.) The structural only-children the splice removed are the
    special case with no siblings at all; this test covers the general case,
    leaves included — a terminal with unsupported siblings degenerates by the
    same mechanism as a class.

    Pure function of the counts and the parent map so it is unit-testable
    without Spark. Returns ``(buckets, per_node)``: buckets map name -> sorted
    engine ids; per_node maps engine id -> (n_siblings, n_supported_siblings).
    Buckets:
      root              — structural, stays degenerate by design;
      no_pos            — zero observed train positives (a different story:
                          the split, or labels, starved the node — NOT the
                          cohort-collapse mechanism);
      allpos_no_sibling_support — all-positive with NO supported sibling under
                          any parent: the unified observational-only-child
                          mechanism, hypothesis-CONSISTENT;
      allpos_with_supported_sibling — all-positive DESPITE a supported sibling:
                          genuinely unexplained, named in the output.
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

    buckets = {"root": [], "no_pos": [], "allpos_no_sibling_support": [],
               "allpos_with_supported_sibling": []}
    per_node = {}
    for c in range(C):
        if not degenerate[c]:
            continue
        if c in roots:
            buckets["root"].append(c)
            continue
        if n_pos[c] <= 0 or n_obs[c] <= 0:
            buckets["no_pos"].append(c)
            continue
        siblings = {k for p in parent_int.get(c, [])
                    for k in children.get(int(p), []) if k != c}
        supported = [k for k in sorted(siblings) if n_pos[k] > 0]
        per_node[c] = (len(siblings), len(supported))
        key = ("allpos_no_sibling_support" if not supported
               else "allpos_with_supported_sibling")
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
        for k in ("root", "no_pos", "allpos_no_sibling_support",
                  "allpos_with_supported_sibling"):
            print(f"[diag]   {k}: {len(buckets[k])}", flush=True)
        print("[diag] the cohort-collapse mechanism = allpos_no_sibling_support "
              "(+ root); allpos_with_supported_sibling is UNEXPLAINED; no_pos is "
              "a separate split/label starvation story.", flush=True)

        def _name(c):
            cid = bundle.int2cid.get(c, c)
            return f"{cid} {bundle.name_by_id.get(cid, '?')}"

        for c in buckets["allpos_with_supported_sibling"][:20]:
            nk, ns = per_node[c]
            print(f"[diag]   UNEXPLAINED {_name(c)}: {ns}/{nk} siblings "
                  f"supported, n_obs={n_obs[c]:.0f} n_pos={n_pos[c]:.0f}",
                  flush=True)
        # A few consistent + no_pos examples so the patterns are eyeball-checkable.
        for c in buckets["allpos_no_sibling_support"][:5]:
            nk, ns = per_node[c]
            print(f"[diag]   consistent {_name(c)}: {ns}/{nk} siblings "
                  f"supported, n_obs={n_obs[c]:.0f} n_pos={n_pos[c]:.0f}",
                  flush=True)
        for c in buckets["no_pos"][:5]:
            print(f"[diag]   no_pos {_name(c)}: n_obs={n_obs[c]:.0f} "
                  f"n_pos={n_pos[c]:.0f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

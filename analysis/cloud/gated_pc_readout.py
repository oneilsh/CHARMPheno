"""Standalone post-hoc case-finding readout on a FINISHED gated_pc run (no re-fit).

Loads a completed run's saved globals (gated_pc_result.npz: lambda/alpha/w_CK) +
manifest.json, reconstructs the OnlinePCLDAModel, reloads the cached
CaseFindingBundle the run was fit against, re-transforms train+test, and prints
the FULL case-finding readout for the gated_pc arm:

  - pc_topics_lr:   ranking AUC/AP + per-node precision@recall / recall@FDR +
                    case-vs-background detection (AUC/AP + precision@recall).
  - co-fit head:    the same readout on the head's own P(node)=sigmoid(w_CK·θ).

The other arms' summary AUC/AP are echoed from manifest["results"] for context
(only the gated_pc arm's globals are saved, so only it can be re-scored here; a
run fit with the current driver already computes+stores every arm's full readout
inline — this script is for re-reading, richer operating points, or a run whose
inline readout predates this eval).

The bundle is located by RECOMPUTING its assembly cache key from the manifest. On
a MISS the tool no longer gives up: if the manifest carries the full assembly
parameters (every run of the current driver does — `corpus_manifest` is the corpus
spec verbatim) it REBUILDS the corpus through the same seam the fit uses, Mondo
DAG build and SNOMED-climb included, writes it through to the cache and proceeds.
That is the difference between "re-score a saved fit on a cluster whose cache is
empty" and "re-run the fit". `--no-rebuild` restores the old fail-fast behaviour,
and `--bundle-path` still short-circuits the key recompute entirely.

Rebuilding re-runs the assembly, which is deterministic in practice but not
guaranteed to be (the CDR moves; assembly source changes), so a rebuilt corpus is
CHECKED against the saved fit before anything is scored: per-domain vocabulary
sizes against the saved λ's own V dimensions, and the rebuilt engine-id -> concept
map against the one the manifest recorded. A mismatch aborts with the drift named,
rather than reporting numbers for a model scored on a corpus it was not fit on.

Both readout paths of ADR 0046 are available here, with the same
`--readout-mode {driver,distributed,auto}` / `--readout-ab-check` semantics as the
fit driver: this tool IS the recovery path when a finished fit's readout output was
lost (exp 0103), and at whole-Mondo C the driver-collect readout is exactly what
does not fit — so re-reading has to be able to run distributed without re-fitting.

SUBMIT NOTE: the distributed path's mapPartitions kernels pickle by MODULE
REFERENCE, so `analysis/cloud/distributed_readout.py` must be importable on the
EXECUTORS as top-level `distributed_readout` — i.e. it must ride in --py-files
(the `gated-pc-readout` Makefile target does this; a hand-rolled spark-submit must
add `--py-files ...,<repo>/analysis/cloud/distributed_readout.py` or the
distributed mode dies unpickling on the executors).

Cluster-covered (Spark + live cache); the pure helpers (build_parser,
bundle_key_from_manifest, reconstruct_model) are unit-tested and the scoring body
(run_readout) is covered against the driver path on a local-Spark fixture in
tests/scripts/test_readout_integration.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import os
from pathlib import Path

import numpy as np

from _driver_common import (_phase, configure_logging, hard_exit,
                            install_stdout_tee, make_spark_session)
from disk_telemetry import start_disk_telemetry
from gated_pc_cloud import (
    _DRIVER_READOUT_MAX_C, _MONDO_DAG_SOURCES, _collect_head_proba,
    _collect_lean_proba, _collect_theta_labels, _dump_partial_results,
    _fit_readout_heads, _write_readout_heads, distributed_ranking_readout,
    distributed_score_arm, format_arm_readout, format_incident_readout,
    format_stacked_readout, incident_readout, multidomain_cache_key,
    multidomain_load_or_build, readout_ab_report, readout_from_proba,
    resolve_eval_path, resolve_readout_mode, score_arm, stacked_readout,
)

# What the batched solve got before fits recorded their own cap. Runs older than
# `readout_max_iter` in the manifest were, in practice, record runs at the driver's
# own default of 200 — a DEV smoke from that era has to be recovered with an
# explicit --readout-max-iter 60, which is exactly the retyping the manifest field
# removes for every fit written since.
_LEGACY_READOUT_MAX_ITER = 200


def _cprint(msg, **kw) -> None:
    """print() through term_colors.highlight_line (ERROR/WARN/HIT/MISS/
    rebuild/macro-AUC/eta-boost -- see term_colors.py). Identity when color
    is disabled or nothing matches, so this is safe on every "[readout]"
    status line in this module.

    LOCAL import, not module-level: this module rides in a self-referencing
    `--py-files` chain from other diagnostics (`hpoa_stage2_probe.py` imports
    it at module level and itself ships its own mapPartitions kernel on
    `--py-files`) -- term_colors must never become a top-level dependency an
    executor could be asked to resolve. See term_colors.py's module
    docstring."""
    from term_colors import highlight_line
    print(highlight_line(msg), **kw)


def resolve_readout_max_iter(cli_value, manifest):
    """Pick the batched-L-BFGS iteration cap for a re-readout, and say who won.

    Precedence: explicit CLI > the fit's own recorded ``readout_max_iter`` >
    the legacy default. Same doctrine as ``readout_theta_topm``: a recovery should
    REPRODUCE the run it is rescuing (the fit's cap is what its lost readout would
    have used, CHARM_DEV capping already folded in), while the CLI stays available
    for deliberately re-solving a run harder or cheaper than it was.

    Returns ``(max_iter, source)``; `source` is a human label for the log line, so
    a recovery's output states on its face which budget it ran under — the number
    alone cannot be read back as "the fit's" or "mine".
    """
    if cli_value is not None:
        return int(cli_value), "CLI"
    recorded = manifest.get("readout_max_iter")
    if recorded:
        return int(recorded), "manifest"
    return _LEGACY_READOUT_MAX_ITER, "legacy default"


_FEATURE_MASK_MODES = ("all", "own", "own-bg", "bg", "closure", "drop-own",
                       "drop-closure", "family", "family-closure")


def build_readout_feature_mask(mode, C, K, n_bg, tpn, parent_int=None):
    """The (C,K) per-node topic mask for a readout FEATURE ABLATION, or None
    for `all`. Rows are engine node ids 0..C-1 (root 0 owns no block; its row
    keeps only the background under the bg-including modes and is degenerate
    anyway); columns follow `DagLayout`: `[0, n_bg)` background, then one
    `tpn`-wide block per non-root node in sorted engine-id order.

      own          the node's own block only
      own-bg       own block + background
      bg           background only (0081's "the signal is shared" test)
      closure      own + every ancestor's block + background — exactly the
                   gate's allowed set for a document attested at this node
      drop-own     everything EXCEPT the own block
      drop-closure everything except own + ancestors' blocks (background kept)
      family       own + SIBLINGS' blocks (other children of the node's parents)
                   + background — the sibling-contrast decoder: under the closure
                   label mask a node's negatives are its siblings, so their own
                   topics firing is the direct evidence AGAINST this node
      family-closure  family + every ancestor's block

    `parent_int` (`{child: [parents]}` in engine ids) is needed by the closure
    and family modes only. Pure numpy; unit-tested off-cluster."""
    if mode not in _FEATURE_MASK_MODES:
        raise ValueError(f"unknown feature mask mode {mode!r}; "
                         f"choose from {_FEATURE_MASK_MODES}")
    if mode == "all":
        return None
    C, K, n_bg, tpn = int(C), int(K), int(n_bg), int(tpn)
    nodes = sorted(e for e in range(1, C))
    block = {e: list(range(n_bg + i * tpn, n_bg + (i + 1) * tpn))
             for i, e in enumerate(nodes)}
    parents = {int(c): [int(p) for p in ps]
               for c, ps in (parent_int or {}).items()}
    needs_dag = ("closure", "drop-closure", "family", "family-closure")
    if mode in needs_dag and not parents:
        raise ValueError(f"feature mask mode {mode!r} needs parent_int")
    children = {}
    for c, ps in parents.items():
        for q in ps:
            children.setdefault(q, []).append(c)

    def siblings(e):
        out = set()
        for q in parents.get(e, []):
            out.update(children.get(q, []))
        out.discard(e)
        return out

    def ancestors(e):
        out, stack = set(), [e]
        while stack:
            for q in parents.get(stack.pop(), []):
                if q != 0 and q not in out:
                    out.add(q); stack.append(q)
        return out

    m = np.zeros((C, K), dtype=bool)
    for e in range(C):
        own = block.get(e, [])
        anc = ([t for a in ancestors(e) for t in block.get(a, [])]
               if mode in ("closure", "drop-closure", "family-closure") else [])
        sib = ([t for a in siblings(e) for t in block.get(a, [])]
               if mode in ("family", "family-closure") else [])
        if mode == "own":
            m[e, own] = True
        elif mode == "own-bg":
            m[e, own] = True; m[e, :n_bg] = True
        elif mode == "bg":
            m[e, :n_bg] = True
        elif mode == "closure":
            m[e, own] = True; m[e, anc] = True; m[e, :n_bg] = True
        elif mode == "drop-own":
            m[e, :] = True; m[e, own] = False
        elif mode == "drop-closure":
            m[e, :] = True; m[e, own] = False; m[e, anc] = False
        elif mode == "family":
            m[e, own] = True; m[e, sib] = True; m[e, :n_bg] = True
        elif mode == "family-closure":
            m[e, own] = True; m[e, sib] = True; m[e, anc] = True; m[e, :n_bg] = True
    return m


def resolve_readout_l2(cli_value, manifest):
    """Ridge strength for a re-readout: explicit CLI > the fit's recorded
    ``readout_l2`` > the legacy 1.0. Same reproduce-the-run doctrine as
    `resolve_readout_max_iter`. Returns ``(l2, source)``."""
    if cli_value is not None:
        return float(cli_value), "CLI"
    recorded = manifest.get("readout_l2")
    if recorded is not None:
        return float(recorded), "manifest"
    return 1.0, "legacy default"


def observe_root_everywhere(df, C, mask_col="labelMask"):
    """Return `df` with `mask_col[0]` (the ROOT's observation flag) set to 1.0 on
    every row; every other element untouched.

    The stacked arm's whole mechanism (spec 2026-10-06 Part A): under
    `label_mask_mode=closure` the root is observed only on foreground rows, all
    positive, so its readout head is the degenerate constant 1.0 and the closure
    product `Π_{a ∈ closure(c)} σ(z_a)` carries no population prior. Observing the
    root on every row makes its head a case-vs-background logistic — the one
    HSLDA's root conditional is — while the batched solve stays per-node
    independent (`solve_batched_lr`), so the heads of nodes ≥ 1 are byte-identical
    to the prevalent arm's. Applied to the TRAIN split only; the test mask is the
    corpus's own, so the ranking axis scores the same cells it always did.

    Pure column arithmetic on the `array<double>` mask (no UDF): `[1.0] ++
    mask[1:]`."""
    from pyspark.sql import functions as F
    C = int(C)
    tail = (F.slice(F.col(mask_col), 2, C - 1) if C > 1
            else F.array().cast("array<double>"))
    return df.withColumn(mask_col, F.concat(F.array(F.lit(1.0)), tail))


def incident_mask_kernel(get_v2n):
    """The per-row kernel of `mask_prior_cells`: `(features vector, mask) -> mask` with
    every node the patient has PRIOR evidence for (a mapped attestation code of the
    node or a label descendant in the pre-index condition features — exactly
    `prevalence_check.prior_matrix`'s rule) set to 0.0. The root (node 0) is never
    touched. `get_v2n()` returns {vocab idx: [nodes]}; on the cluster it reads a
    broadcast, so nothing array-shaped rides the task closure (ADR 0047).

    A plain nested function on purpose: cloudpickle ships it by value, so no module
    needs to be on --py-files."""
    def kernel(vec, mask):
        if mask is None:
            return None
        v2n = get_v2n()
        idx = getattr(vec, "indices", None)
        if idx is None:
            arr = vec.toArray() if hasattr(vec, "toArray") else (vec or [])
            idx = [i for i, x in enumerate(arr) if x]
        out = list(mask)
        for v in idx:
            for n in v2n.get(int(v), ()):
                if n != 0:
                    out[n] = 0.0
        return out
    return kernel


def mask_prior_cells(df, v2n, *, feat_col="features_0", mask_col="labelMask"):
    """INCIDENT label rule (insight 0100; exp 0144), readout-side: a patient with prior
    evidence for node X is neither a positive nor a negative for X's head, so the solve
    learns "will X be newly coded", not "will X be coded again". Python UDF over the
    widened `array<double>` mask; the caller caches and materializes the result (the
    solver makes repeated passes)."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import ArrayType, DoubleType
    bc = df.sparkSession.sparkContext.broadcast(
        {int(k): [int(n) for n in ns] for k, ns in v2n.items()})
    kern = incident_mask_kernel(lambda: bc.value)
    udf = F.udf(kern, ArrayType(DoubleType()))
    return df.withColumn(mask_col, udf(F.col(feat_col), F.col(mask_col)))


def run_conditional_from_heads(test_scored, run_dir, tag, *, C, parent_int, int2cid,
                               name_by_id, min_count=20):
    """CONDITIONAL DIAGNOSIS on held-out patients from a finished readout's saved
    heads (spec 2026-10-07 §D4, WP-D part 2): "this patient is in Mondo group P;
    which child?" scored per readout-DAG edge P -> c within P's test cohort
    (`gated_pc_cloud.conditional_readout`, all-ones mask, cond AUC the sober column,
    top-1 next to the majority baseline, ECE), for both the flat heads sigma(z_c)
    and the stacked product P_stack(c). No solve: the heads are the record's own
    (`readout_heads_gated_pc_<tag>.npz`), so this re-reads, never re-fits.

    The two rows are not rival models. Under the closure mask every head of depth >= 1
    is fit on its parent's positives, so sigma(z_c) IS P(c | parent) =
    P_stack(c) / P_stack(parent); that is the conditional-diagnosis answer. The
    P_stack(c) row ranks the MARGINAL within the parent cohort: the ancestor product
    varies across a cohort that is already known to have the parent and carries no
    which-child information, so it only adds noise, more with each level of depth.
    It is kept as a reference for that reason (0138's first run, 2026-10-12).

    Returns the results dict; writes `results_conditional_<tag>.json`. Counts stay in
    the run dir (egress floor); the printed tables are the existing pooled format."""
    from gated_pc_cloud import (closure_matrix, conditional_readout,
                                format_conditional_readout, stacked_proba)
    run_dir = Path(run_dir)
    hp = run_dir / f"readout_heads_gated_pc_{tag}.npz"
    if not hp.exists():
        raise SystemExit(f"[readout] no saved heads at {hp}")
    z = np.load(hp)
    if int(z["C"]) != int(C):
        raise SystemExit(f"[readout] heads {hp.name} are C={int(z['C'])} but the "
                         f"widened frames are C={int(C)}: different readout DAG")
    topm = int(z["theta_topm"])
    _cprint(f"[readout]   conditional diagnosis from {hp.name} (C={int(C)} "
            f"K={int(z['K'])} topm={topm}); test split only", flush=True)
    proba, y, _m, _keys, _e = _collect_lean_proba(
        test_scored, int(C), z["V"], z["b_raw"], degenerate=z["degenerate"],
        const=z["const"], theta_topm=topm)
    ones = np.ones_like(y)
    out = {"heads": hp.name, "min_count": int(min_count)}
    out["flat"] = conditional_readout(proba, y, ones, parent_int, int(C),
                                      min_count=int(min_count))
    print(format_conditional_readout(out["flat"], int2cid, name_by_id).replace(
        "[conditional sharpening]", "[conditional diagnosis: P(c|parent) = sigma(z_c), the conditional head — THE ANSWER]"),
        flush=True)
    P = stacked_proba(proba, closure_matrix(parent_int, int(C)))
    out["stacked"] = conditional_readout(P, y, ones, parent_int, int(C),
                                         min_count=int(min_count))
    print(format_conditional_readout(out["stacked"], int2cid, name_by_id).replace(
        "[conditional sharpening]", "[reference: marginal P_stack(c) ranked within the parent cohort — the ancestor product is noise here]"),
        flush=True)
    _dump_partial_results(run_dir, out, name=f"results_conditional_{tag}.json")
    _cprint(f"[readout]   wrote {run_dir / f'results_conditional_{tag}.json'}",
            flush=True)
    return out


def collect_set_patients(test_df, members, n_domains, *, max_patients=5000, seed=0,
                         label_col="label", length="distinct"):
    """Test-split persons positive on any of `members` (fit-width label indices):
    `[(truth positions into members, {d: {vocab idx: count}}, {d: length})]`.

    Filter and label slice are column arithmetic over a JSON literal (the
    `widen_labels_to_readout` pattern: nothing array-shaped rides a closure); the
    collect carries only the members' label cells and the sparse feature vectors.
    Capped at `max_patients` by a seeded Spark sample. `length` is the record length
    the presence model gets per domain: `distinct` codes (default) or `tokens`."""
    import json as _json
    from pyspark.sql import functions as F
    lit = F.from_json(F.lit(_json.dumps([int(m) for m in members])), "array<int>")
    lab = F.col(label_col)
    fcols = ([f"features_{d}" for d in range(int(n_domains))]
             if f"features_0" in test_df.columns else ["features"])
    df = (test_df.where(F.exists(lit, lambda i: lab[i] > 0))
                 .select(F.transform(lit, lambda i: lab[i]).alias("__y"), *fcols))
    n = df.count()
    if n > int(max_patients):
        df = df.sample(False, min(1.0, 1.2 * int(max_patients) / n), seed=int(seed))
    out = []
    for row in df.collect()[:int(max_patients)]:
        truth = [j for j, v in enumerate(row["__y"]) if v and v > 0]
        bags, lens = {}, {}
        for d, c in enumerate(fcols):
            vec = row[c]
            if vec is None:
                continue
            idx = getattr(vec, "indices", None)
            if idx is None:
                arr = np.asarray(vec.toArray() if hasattr(vec, "toArray") else vec)
                idx = np.flatnonzero(arr)
                vals = arr[idx]
            else:
                vals = vec.values
            bags[d] = {int(i): float(c) for i, c in zip(idx, vals)}
            lens[d] = (float(np.sum(vals)) if length == "tokens"
                       else float(len(idx)))
        out.append((truth, bags, lens))
    return out, n


def run_acquisition(test_df, run_dir, pattern, *, adhoc=False, prior="population",
                    results_name="results_readout_stacked_mass99.json",
                    domains=("condition", "drug"), pool=150, steps=20,
                    max_patients=5000, include_strata=False, thin_frac=0.1,
                    own_top=5, rel_lambda=0.6, rho=0.3, seed=0, tag=None,
                    length="distinct", tau="auto", engine="presence",
                    heads_tag="stacked_mass99"):
    """WP-D part 3: simulated acquisition on held-out patients (`voi_acquisition`).
    Profiles, pool and prior come from the run dir exactly as part 1's tables do
    (`voi_tables.load_voi_context`); Spark is used only to collect the test
    patients' code sets per candidate set. Writes `acquisition[_<tag>].md/.json`
    (pooled numbers only).

    `engine="model"` (part 3b, `model_acquisition`) keeps the pool and the question
    orders and swaps the posterior for the record's own: θ re-inferred from the
    revealed codes (the deployment fold-in over the saved λ and α) scored by the
    saved readout heads `readout_heads_gated_pc_<heads_tag>.npz`."""
    import voi_acquisition as va
    import voi_tables as vt
    run_dir = Path(run_dir)
    mfit = None
    if engine == "model":
        from gated_pc_cloud import closure_matrix
        npz = np.load(run_dir / "gated_pc_result.npz")
        lam_keys = sorted((k for k in npz.files if k.startswith("lambda_")),
                          key=lambda k: int(k.split("_", 1)[1]))
        lams_fit = [npz[k] for k in lam_keys] if lam_keys else [npz["lambda"]]
        hp = run_dir / f"readout_heads_gated_pc_{heads_tag}.npz"
        if not hp.exists():
            raise SystemExit(f"[acquisition] --acq-engine model needs {hp}")
        hz = np.load(hp)
        rd = json.loads((run_dir / "readout_dag.json").read_text())
        if int(hz["C"]) != int(rd["C"]):
            raise SystemExit(f"[acquisition] heads are C={int(hz['C'])} but "
                             f"readout_dag.json is C={rd['C']}")
        M = closure_matrix({int(c): [int(x) for x in ps]
                            for c, ps in rd["parent_int"].items()}, int(rd["C"]))
        mfit = (lams_fit, npz["alpha"],
                (hz["V"], hz["b_raw"], hz["degenerate"], hz["const"],
                 int(hz["theta_topm"])), M)
    bm = run_dir / "bundle_meta.json"
    cm = run_dir / "code_map.tsv"
    ctx = vt.load_voi_context(
        run_dir, bundle_meta_path=str(bm) if bm.exists() else None,
        own_codes_path=str(cm) if cm.exists() else None, results_name=results_name,
        thin_frac=thin_frac, own_top=own_top, rel_lambda=rel_lambda, prior=prior,
        include_strata=include_strata)
    doms = [i for i, nm in enumerate(ctx.dom_names) if nm in set(domains)]
    if not doms:
        raise SystemExit(f"[acquisition] none of {domains} in {ctx.dom_names}")
    sets = vt.select_sets(ctx, pattern, adhoc=adhoc)
    if not sets:
        raise SystemExit(f"[acquisition] no candidate set matches {pattern!r}")
    L_ = [f"# {run_dir.name} — simulated acquisition (WP-D part 3)",
          f"pattern {pattern!r}{' (ad hoc)' if adhoc else ''} · domains "
          f"{[ctx.dom_names[d] for d in doms]} · prior {prior} · rho {rho:g} · "
          + (f"ENGINE: θ fold-in + saved heads ({heads_tag}) · " if mfit else
             f"L = the patient's {length} codes per domain · temper {tau} · ")
          + f"profiles (pool only): signatures"
          f"{' + strata' if include_strata else ''} · pool {pool}/domain · "
          f"steps {steps}", ""]
    js = {}
    for n, gname, members in sets:
        profs = {}
        used = None
        for d in doms:
            pr, bg = vt.set_profiles(ctx, members, d)
            u = [m for m in members if pr.get(m) is not None]
            used = u if used is None else [m for m in used if m in u]
            profs[d] = (pr, bg)
        if len(used) < 2:
            L_.append(f"## {gname}\n(fewer than 2 members with a signature profile)")
            continue
        pc = ctx.prior_counts
        pri = np.array([float(pc.get(m, 0) or 0) for m in used])
        pri = pri / pri.sum() if pri.sum() > 0 else np.full(len(used), 1 / len(used))
        excl = {0: vt.set_exclusions(ctx, n, members)} if 0 in doms else {}
        pool_items, mix = va.build_pool(
            pri, {d: np.stack([profs[d][0][m] for m in used]) for d in doms},
            {d: profs[d][1] for d in doms}, excl, pool=pool, rho=rho)
        pats, n_all = collect_set_patients(test_df, used, len(ctx.lams),
                                           max_patients=max_patients, seed=seed,
                                           length=length)
        pats = [p for p in pats if p[0]]
        _cprint(f"[acquisition]   {gname}: {len(pats)} patients (of {n_all} "
                f"positive), pool {len(pool_items)}", flush=True)
        named = [ctx.rd_names.get(m, str(m))[:28] for m in used]
        if mfit is not None:
            import model_acquisition as ma
            lams_fit, alpha_fit, heads, M = mfit
            eng = ma.ModelEngine(
                lams_fit, alpha_fit, heads,
                {m: [int(a) for a in np.flatnonzero(M[m])] for m in used}, used)
            res = ma.simulate_set_model(eng, pool_items, pats, steps=steps, seed=seed)
            L_.append(ma.format_model_acquisition(gname, named, res))
        else:
            res = va.simulate_set(pri, pool_items, mix, pats, steps=steps, seed=seed,
                                  tau=tau)
            L_.append(va.format_acquisition(gname, named, pri, res))
        L_.append("")
        js[gname] = res if res["n"] >= va.EGRESS_FLOOR else {"suppressed": True}
    text = "\n".join(L_).rstrip() + "\n"
    stem = "acquisition" + (f"_{tag}" if tag else "")
    (run_dir / f"{stem}.md").write_text(text)
    (run_dir / f"{stem}.json").write_text(json.dumps(js, indent=1, default=str))
    print(text, flush=True)
    _cprint(f"[acquisition]   wrote {run_dir / (stem + '.md')}", flush=True)
    return js


def run_differential(test_df, train_df, run_dir, pattern, *, adhoc=False,
                     results_name="results_readout_stacked_mass99.json",
                     heads_tag="stacked_mass99", max_patients=600, max_train=2000,
                     thin_frac=0.1, own_top=5, rel_lambda=0.6, rho=0.3, seed=0,
                     tag=None, next_question=False, nq_pool=100, nq_hide=0.2):
    """Framing 1 — DIFFERENTIAL WITH REASONS (spec 2026-10-13 clinician framings,
    R0/R1; `differential.py`). Four engines on the same held-out patients positive on
    exactly one member: E1 presence model with subtype profiles, E2 renormalized
    stacked heads, E3 a multiclass head fit on the train split, E4 the gated
    per-disease likelihood. Records lose attestation codes and `[measured]` tokens
    first. Tempers (E1, E4) are fit on one half of the patients and every engine is
    scored on the other. Writes `differential[_<tag>].md/.json` (pooled only)."""
    import time
    import differential as dfm
    import model_acquisition as ma
    import voi_tables as vt
    from gated_pc_cloud import closure_matrix
    run_dir = Path(run_dir)
    bm, cm = run_dir / "bundle_meta.json", run_dir / "code_map.tsv"
    names_csv = run_dir / "concept_names.csv"
    ctx = vt.load_voi_context(
        run_dir, bundle_meta_path=str(bm) if bm.exists() else None,
        own_codes_path=str(cm) if cm.exists() else None, results_name=results_name,
        names_path=str(names_csv) if names_csv.exists() else None,
        thin_frac=thin_frac, own_top=own_top, rel_lambda=rel_lambda)
    npz = np.load(run_dir / "gated_pc_result.npz")
    lam_keys = sorted((k for k in npz.files if k.startswith("lambda_")),
                      key=lambda k: int(k.split("_", 1)[1]))
    lams_fit = [npz[k] for k in lam_keys] if lam_keys else [npz["lambda"]]
    alpha = npz["alpha"]
    hz = np.load(run_dir / f"readout_heads_gated_pc_{heads_tag}.npz")
    rd = json.loads((run_dir / "readout_dag.json").read_text())
    M = closure_matrix({int(c): [int(x) for x in ps] for c, ps in rd["parent_int"].items()},
                       int(rd["C"]))
    heads = (hz["V"], hz["b_raw"], hz["degenerate"], hz["const"], int(hz["theta_topm"]))
    measured = set()
    if "measurement" in ctx.dom_names and ctx.inv_maps:
        dm = ctx.dom_names.index("measurement")
        measured = {int(v) for v, tok in ctx.inv_maps[dm].items() if int(tok) % 100 == 0}
    ev = np.asarray(ctx.sh["evidence"], dtype=np.float64)
    sets = vt.select_sets(ctx, pattern, adhoc=adhoc)
    if not sets:
        raise SystemExit(f"[differential] no candidate set matches {pattern!r}")
    L_ = [f"# {run_dir.name} — differential with reasons (framing 1)",
          f"pattern {pattern!r}{' (ad hoc)' if adhoc else ''} · records minus attestation "
          f"codes and {len(measured):,} [measured] tokens · E1 presence subtypes · E2 "
          f"stacked heads ({heads_tag}) · E3 multiclass head (train split) · E4 gated "
          f"likelihood · tempers fit on one half, all scored on the other", ""]
    js = {}
    eng = None
    for n, gname, members in sets:
        t0 = time.time()
        excl = {0: vt.set_exclusions(ctx, n, members)}
        # Name-matched attestation (0141): a condition concept whose name IS a member's
        # or the group's name ("Hypertrophic cardiomyopathy" 40565500, "Cardiomyopathy")
        # but which the code map missed is attestation too, not a finding.
        nm_targets = {str(ctx.rd_names.get(m, "")).strip().lower() for m in members}
        nm_targets.add(str(gname).strip().lower())
        if ctx.names and ctx.inv_maps:
            for v, cid in ctx.inv_maps[0].items():
                if str(ctx.names.get(int(cid), "")).strip().lower() in nm_targets:
                    excl[0].add(int(v))
        if measured:
            excl[dm] = measured
        subtypes = {}
        for m in members:
            sig = ctx.sig.get(m) or [t for t, k, _, _ in ctx.kinds.get(m, [])
                                     if k != "starved"]
            subtypes[m] = [(t, ev[t]) for t in sig]
        if eng is None:
            eng = ma.ModelEngine(lams_fit, alpha, heads,
                                 {m: [int(a) for a in np.flatnonzero(M[m])]
                                  for m in members}, members)
        else:
            eng.__init__(lams_fit, alpha, heads,
                         {m: [int(a) for a in np.flatnonzero(M[m])] for m in members},
                         members)
        e1 = dfm.PresenceSubtypeEngine(lams_fit, subtypes, ctx.n_bg, ev, members, excl,
                                       rho=rho)
        bg = list(range(ctx.n_bg))
        e4 = dfm.GatedLikelihoodEngine(eng.eb, alpha,
                                       {m: bg + list(ctx.block[m]) for m in members},
                                       members)
        feat = bg + sorted({t for m in members for t in ctx.block[m]})

        def prep(rows):
            out = []
            for truth, bags, _ in rows:
                if len(truth) != 1:
                    continue
                fb, lens = dfm.filter_record(bags, excl)
                gi = [eng.gidx(d, v) for d, b in fb.items() for v in b]
                gc = [c for d, b in fb.items() for c in b.values()]
                out.append((truth, fb, lens, gi, gc))
            return out
        te_raw, n_te = collect_set_patients(test_df, members, len(ctx.lams),
                                            max_patients=max_patients, seed=seed)
        tr_raw, n_tr = collect_set_patients(train_df, members, len(ctx.lams),
                                            max_patients=max_train, seed=seed)
        n_multi = sum(len(t) > 1 for t, _, _ in te_raw)
        te, tr = prep(te_raw), prep(tr_raw)
        _cprint(f"[differential]   {gname}: {len(te)} single-positive test patients "
                f"({n_multi} multi-positive set aside), {len(tr)} train", flush=True)
        if len(te) < 2 * dfm.EGRESS_FLOOR:
            L_.append(f"## {gname}\n(fewer than {2 * dfm.EGRESS_FLOOR} single-positive "
                      f"held-out patients; suppressed)")
            continue
        def theta(row):
            g = eng.fold_in(row[3], row[4])
            return g / g.sum()
        th_te = np.stack([theta(r) for r in te])
        gsum_te = [float(eng.alpha.sum() + sum(r[4])) for r in te]
        truths = [r[0] for r in te]
        e3 = None
        ytr = [r[0][0] for r in tr]
        if len(set(ytr)) >= 2:
            th_tr = np.stack([theta(r) for r in tr])
            e3 = dfm.MulticlassHead(feat, len(members), seed=seed).fit(th_tr, ytr)
        s1 = [e1.scores(e1.subtype_ll(r[1], r[2])) for r in te]
        s4, lphi4 = zip(*[e4.elbos(r[3], r[4]) for r in te])
        # Prior over members from the TRAIN split's single-positive patients, add-one
        # smoothed: a member absent from the scored-results file (0140 first run) got
        # prior 1e-12 and could never be called, and a test-derived prior leaks.
        pri = np.bincount(np.asarray(ytr, dtype=np.int64),
                          minlength=len(members)).astype(np.float64) + 1.0
        pri = pri / pri.sum()
        lp = np.log(pri)
        fit_i, ev_i = list(range(0, len(te), 2)), list(range(1, len(te), 2))
        tau1, _ = dfm.fit_tau_scores(lp, [s1[i] for i in fit_i], [truths[i] for i in fit_i])
        tau4, _ = dfm.fit_tau_scores(lp, [s4[i] for i in fit_i], [truths[i] for i in fit_i])
        tv = [truths[i] for i in ev_i]
        res = {"n_eval": len(ev_i), "n_fit": len(fit_i), "n_train": len(tr),
               "tau_e1": tau1, "tau_e4": tau4,
               "e3_lambda": (e3.lam if e3 is not None else None)}
        rows = {"prior": [pri for _ in ev_i],
                "E1 presence subtypes": [dfm.posterior_from_scores(lp, s1[i], tau1)
                                         for i in ev_i],
                "E2 stacked heads": [eng.posterior(th_te[i]) for i in ev_i],
                "E4 gated likelihood": [dfm.posterior_from_scores(lp, s4[i], tau4)
                                        for i in ev_i]}
        if e3 is not None:
            rows["E3 multiclass head"] = list(e3.predict(th_te[ev_i]))
        res["metrics"] = {k: dfm.metrics(v, tv) for k, v in rows.items()}
        # reasons on correctly called evaluation patients
        r1, r3, r4 = [], [], []
        for j, i in enumerate(ev_i):
            t = truths[i][0]
            m = members[t]
            if e3 is not None and int(np.argmax(rows["E3 multiclass head"][j])) == t \
                    and te[i][3]:
                c3 = dfm.e3_reasons(e3, th_te[i], gsum_te[i], eng.eb[:, te[i][3]],
                                    te[i][4], t)
                r3.append((m, {_split_gidx(eng, g): float(x)
                               for g, x in zip(te[i][3], c3)}))
            if int(np.argmax(rows["E1 presence subtypes"][j])) == t:
                r1.append((m, e1.reasons(e1.subtype_ll(te[i][1], te[i][2]), m)))
            if int(np.argmax(rows["E4 gated likelihood"][j])) == t:
                rs = e4.reasons(te[i][3], te[i][4], lphi4[i], m)
                r4.append((m, {_split_gidx(eng, g): x for g, x in rs.items()}))
        res["reasons_e3"] = dfm.pool_reasons(r3)
        res["reasons_e1"] = dfm.pool_reasons(r1)
        res["reasons_e4"] = dfm.pool_reasons(r4)
        named = {m: ctx.rd_names.get(m, str(m)) for m in members}
        L_.append(f"## {gname} — {len(members)} members · {len(ev_i)} scored patients "
                  f"(τ fit on {len(fit_i)}) · {len(tr)} train patients for E3 · "
                  f"τ E1 {tau1:.3g}, E4 {tau4:.3g}")
        L_.append("")
        L_.append("| engine | top-1 | top-3 | bal top-1 | log loss | ECE |")
        L_.append("|---|--:|--:|--:|--:|--:|")
        for k, mtr in res["metrics"].items():
            L_.append(f"| {k} | {mtr['top1']:.3f} | {mtr['top3']:.3f} | "
                      f"{mtr['bal_top1']:.3f} | {mtr['logloss']:.3f} | {mtr['ece']:.3f} |")
        for key, lab in (("reasons_e3", "E3"), ("reasons_e1", "E1"),
                         ("reasons_e4", "E4")):
            if res[key]:
                L_.append("")
                L_.append(f"{lab} reasons (top supporting codes, correctly called "
                          f"patients, members with >= {dfm.EGRESS_FLOOR}):")
                for m, codes in res[key].items():
                    L_.append(f"- {named[m][:40]}: " + " · ".join(
                        ctx.code_name(d, v) for d, v in codes))
        L_.append("")
        if next_question and e3 is not None:
            nq = _next_question_block(ctx, eng, e3, members, te, ev_i, truths, pri,
                                      excl, named, nq_pool, seed, frac=nq_hide,
                                      e1=e1, e1_lp=lp, e1_tau=tau1)
            L_.extend(nq["lines"])
            res["next_question"] = nq["res"]
        _cprint(f"[differential]   {gname}: done in {time.time() - t0:.0f}s", flush=True)
        js[gname] = {k: v for k, v in res.items() if not k.startswith("reasons")}
    text = "\n".join(L_).rstrip() + "\n"
    stem = "differential" + (f"_{tag}" if tag else "")
    (run_dir / f"{stem}.md").write_text(text)
    (run_dir / f"{stem}.json").write_text(json.dumps(js, indent=1, default=str))
    print(text, flush=True)
    _cprint(f"[differential]   wrote {run_dir / (stem + '.md')}", flush=True)
    return js


def _next_question_block(ctx, eng, head, members, te, ev_i, truths, pri, excl, named,
                         pool_per_dom, seed, *, frac=0.2, e1=None, e1_lp=None,
                         e1_tau=1.0):
    """Framing 2 (spec R2) on the framing-1 evaluation half: the candidate pool is the
    population-EIG table over condition, drug and measurement-result codes (attestation
    and [measured] excluded), the differential is E3's. Candidates whose presence
    alone would give one member >= 90% (presence model, prior `pri`) are flagged as
    likely treatments / diagnostic consequences (R2.2)."""
    import differential as dfm
    import voi_acquisition as va
    import voi_tables as vt
    doms = [i for i, nm in enumerate(ctx.dom_names)
            if nm in ("condition", "drug", "measurement")]
    profs, bgs, used = {}, {}, None
    for d in doms:
        pr, bg = vt.set_profiles(ctx, members, d)
        u = [j for j, m in enumerate(members) if pr.get(m) is not None]
        used = u if used is None else [j for j in used if j in u]
        profs[d], bgs[d] = pr, bg
    lines = [f"### Next question (framing 2; {100 * frac:.0f}% of each record hidden; "
             f"differentials E3 and E1)"]
    if not used or len(used) < 2:
        return {"lines": lines + ["(fewer than 2 members with a profile)"], "res": None}
    pu = np.asarray(pri)[used]
    pu = pu / pu.sum()
    P = {d: np.stack([profs[d][members[j]] for j in used]) for d in doms}
    pool_items, _ = va.build_pool(pu, P, bgs, excl, pool=pool_per_dom)
    flagged = {}
    for j, (d, v) in enumerate(pool_items):
        pm = vt.presence_prob(P[d][:, v], bgs[d][v])
        q = pu * pm
        flagged[j] = bool(q.max() / q.sum() >= 0.9)
    pats = [(truths[i], te[i][1]) for i in ev_i]
    rng = np.random.default_rng(int(seed))
    r = dfm.evaluate_next_question(eng, head, pats, pool_items, flagged, rng, frac=frac,
                                   e1=e1, e1_lp=e1_lp, e1_tau=e1_tau)
    if r["n"] < dfm.EGRESS_FLOOR:
        return {"lines": lines + ["(fewer than 20 evaluable patients; suppressed)"],
                "res": None}
    v, f = r["visible"], r["full"]
    lines += [f"{r['n']} patients · pool {len(pool_items)} codes (condition, drug, "
              f"measurement results) · likely-treatment flags in VOI top 3: "
              f"{100 * r['flag_share_top3']:.0f}%", "",
              "| strategy | k | yield | P(true) E3 | top-1 E3 | P(true) E1 | top-1 E1 |",
              "|---|--:|--:|--:|--:|--:|--:|",
              f"| visible record ({100 - 100 * frac:.0f}%) | — | — | {v['p_true']:.3f} | "
              f"{v['top1']:.3f} | {v['p_true_e1']:.3f} | {v['top1_e1']:.3f} |"]
    for k in r["ks"]:
        for s_ in tuple(r["strategies"]) + ("random hidden",):
            x = r["by"][f"{s_}@{k}"]
            lines.append(f"| {s_} | {k} | {x['yield']:.3f} | {x['p_true']:.3f} | "
                         f"{x['top1']:.3f} | {x['p_true_e1']:.3f} | {x['top1_e1']:.3f} |")
    lines.append(f"| full record | — | — | {f['p_true']:.3f} | {f['top1']:.3f} | "
                 f"{f['p_true_e1']:.3f} | {f['top1_e1']:.3f} |")
    top = sorted(((j, c) for j, c in r["recommended"].items()
                  if c >= dfm.EGRESS_FLOOR), key=lambda x: -x[1])[:10]
    if top:
        lines += ["", "Most often in the VOI (E3) top 3 (>= 20 patients; † = likely "
                  "treatment / consequence):"]
        lines.append(" · ".join(ctx.code_name(*pool_items[j]) + (" †" if flagged[j] else "")
                                for j, _ in top))
    lines.append("")
    res = {k: v for k, v in r.items() if k != "recommended"}
    return {"lines": lines, "res": res}


def _split_gidx(eng, g):
    """Global concatenated vocab id -> (domain, vocab idx)."""
    d = int(np.searchsorted(eng.offsets, g, side="right") - 1)
    return d, int(g - eng.offsets[d])


def run_lookalikes(test_scored, run_dir, pattern, *, C, parent_int, adhoc=False,
                   heads_tag="stacked_mass99", top_n=200, pos_cap=1000, bg_n=3000,
                   seed=0, tag=None, thin_frac=0.1, own_top=5, rel_lambda=0.6):
    """Framing 3 — UNDIAGNOSED LOOK-ALIKES (spec 2026-10-13, R3; `lookalikes.py`).
    For every member X of the candidate sets matching `pattern`: rank held-out
    patients WITHOUT an X label by the record's stacked readout P_stack(X) (saved
    heads, test split only, no solve), and profile the top `top_n` against a random
    test sample and X's own patients. Also reports how often the look-alikes carry a
    concept NAMED like X that the code map missed (true diagnoses the label never
    saw). Writes `lookalikes[_<tag>].md/.json` (pooled; rates on >= 20 patients)."""
    import lookalikes as lk
    import voi_tables as vt
    from gated_pc_cloud import (_collect_lean_proba, _doc_key_column, closure_matrix,
                                stacked_proba)
    from pyspark.sql import functions as F
    run_dir = Path(run_dir)
    bm, cm = run_dir / "bundle_meta.json", run_dir / "code_map.tsv"
    names_csv = run_dir / "concept_names.csv"
    ctx = vt.load_voi_context(
        run_dir, bundle_meta_path=str(bm) if bm.exists() else None,
        own_codes_path=str(cm) if cm.exists() else None,
        names_path=str(names_csv) if names_csv.exists() else None,
        thin_frac=thin_frac, own_top=own_top, rel_lambda=rel_lambda)
    sets = vt.select_sets(ctx, pattern, adhoc=adhoc)
    targets = sorted({m for _, _, mem in sets for m in mem})
    if not targets:
        raise SystemExit(f"[lookalikes] no candidate set matches {pattern!r}")
    hp = run_dir / f"readout_heads_gated_pc_{heads_tag}.npz"
    z = np.load(hp)
    if int(z["C"]) != int(C):
        raise SystemExit(f"[lookalikes] heads {hp.name} are C={int(z['C'])}, frames C={C}")
    _cprint(f"[lookalikes]   scoring the test split with {hp.name} for "
            f"{len(targets)} target(s)", flush=True)
    proba, y, _m, keys, _e = _collect_lean_proba(
        test_scored, int(C), z["V"], z["b_raw"], degenerate=z["degenerate"],
        const=z["const"], theta_topm=int(z["theta_topm"]))
    P = stacked_proba(proba, closure_matrix(parent_int, int(C)))
    rng = np.random.default_rng(int(seed))
    keys = [int(k) for k in keys]
    plan, want = {}, set()
    for X in targets:
        yx = y[:, X].astype(bool)
        if int(yx.sum()) < lk.EGRESS_FLOOR:
            continue
        top, cut = lk.top_lookalikes(P[:, X], yx, keys, top_n)
        pos_i = np.flatnonzero(yx)
        pos = [keys[i] for i in rng.choice(pos_i, size=min(pos_cap, pos_i.size),
                                           replace=False)]
        plan[X] = {"auc": lk.auc(P[:, X], yx), "n_pos": int(yx.sum()), "cut": cut,
                   "top": top, "pos": pos}
        want.update(top); want.update(pos)
    bg = [keys[i] for i in rng.choice(len(keys), size=min(bg_n, len(keys)),
                                      replace=False)]
    want.update(bg)
    fcols = sorted((c for c in test_scored.columns if c.startswith("features_")),
                   key=lambda c: int(c.split("_")[1])) or ["features"]
    spark = test_scored.sparkSession
    kdf = spark.createDataFrame([(int(k),) for k in want], ["doc_key"])
    rows = (test_scored.withColumn("doc_key", _doc_key_column(test_scored, "person_id"))
            .join(F.broadcast(kdf), "doc_key").select("doc_key", *fcols).collect())
    bags = {}
    for r in rows:
        b = {}
        for d, c in enumerate(fcols):
            vec = r[c]
            if vec is None:
                continue
            idx = getattr(vec, "indices", None)
            if idx is None:
                arr = np.asarray(vec.toArray() if hasattr(vec, "toArray") else vec)
                idx = np.flatnonzero(arr)
            b[d] = {int(i): 1.0 for i in idx}
        bags[int(r["doc_key"])] = b
    measured = set()
    if "measurement" in ctx.dom_names and ctx.inv_maps:
        dm = ctx.dom_names.index("measurement")
        measured = {int(v) for v, tok in ctx.inv_maps[dm].items() if int(tok) % 100 == 0}
    bgb = [bags[k] for k in bg if k in bags]
    L_ = [f"# {run_dir.name} — undiagnosed look-alikes (framing 3)",
          f"pattern {pattern!r}{' (ad hoc)' if adhoc else ''} · ranking P_stack(X) "
          f"({heads_tag}) over test patients without an X label · top {top_n} "
          f"profiled against {len(bgb)} random test patients and up to {pos_cap} of X's "
          f"own · [measured] and X's attestation codes excluded from profiles · rates "
          f"shown only on >= {lk.EGRESS_FLOOR} patients", ""]
    js = {}
    for X, pl in plan.items():
        nm = str(ctx.rd_names.get(X, X))
        named = set()
        if ctx.names and ctx.inv_maps:
            for v, cid in ctx.inv_maps[0].items():
                if str(ctx.names.get(int(cid), "")).strip().lower() == nm.strip().lower():
                    named.add(int(v))
        excl = {0: set(ctx.own_idx.get(X, ())) | named}
        if measured:
            excl[dm] = measured
        topb = [bags[k] for k in pl["top"] if k in bags]
        posb = [bags[k] for k in pl["pos"] if k in bags]
        # Who are the look-alikes? Features are the PRE-index history and labels the
        # year after it (lookback mode), so "no X label" includes patients whose X
        # diagnosis is in their history but was not re-coded in the label year.
        own = set(ctx.own_idx.get(X, ()))
        unmapped = named - own
        grp = lk.split_lookalikes(topb, own, unmapped)
        und = grp["undiagnosed"]
        prof = lk.feature_profile(topb, posb, bgb, excl_by_dom=excl)
        prof_u = (lk.feature_profile(und, posb, bgb, excl_by_dom=excl)
                  if len(und) >= lk.EGRESS_FLOOR else [])
        nbg_prior, nB = lk.share_with(bgb, {0: own})
        fmt = lambda n, N: (f"{100 * n / N:.1f}%" if n >= lk.EGRESS_FLOOR else
                            ("0" if n == 0 else f"<{lk.EGRESS_FLOOR} pts"))
        nT = len(topb)
        L_.append(f"## {nm} — AUC {pl['auc']:.3f} · {pl['n_pos']} test positives · "
                  f"look-alike cut P_stack ≥ {pl['cut']:.3f}")
        L_.append(f"top {nT} look-alikes: X code in their pre-index history "
                  f"{fmt(len(grp['prior_dx']), nT)} (random {fmt(nbg_prior, nB)}) · "
                  f"only an UNMAPPED X-named concept {fmt(len(grp['unmapped_only']), nT)} · "
                  f"neither (undiagnosed by code) {fmt(len(und), nT)}"
                  + (f" · X-named concepts: {len(named & own)} mapped, {len(unmapped)} "
                     f"unmapped" if named else ""))
        for title, rows in (("all look-alikes", prof),
                            ("undiagnosed-by-code look-alikes only", prof_u)):
            L_.append("")
            if not rows:
                L_.append(f"{title}: no code carried by >= {lk.EGRESS_FLOOR} of them is "
                          f"X-like" + ("" if title.startswith("all") or
                                       len(und) >= lk.EGRESS_FLOOR else
                                       f" (fewer than {lk.EGRESS_FLOOR} patients)"))
                continue
            L_.append(f"{title}:")
            L_.append("| code | look-alikes | X patients | random |")
            L_.append("|---|--:|--:|--:|")
            for (d, v), rt, rp, rb, lift in rows:
                L_.append(f"| {ctx.code_name(d, v)} | {100 * rt:.0f}% | {100 * rp:.0f}% "
                          f"| {100 * rb:.1f}% |")
        L_.append("")
        js[nm] = {"auc": pl["auc"], "n_pos": pl["n_pos"], "cut": pl["cut"],
                  "n_top": nT, "prior_dx": len(grp["prior_dx"]),
                  "unmapped_only": len(grp["unmapped_only"]), "undiagnosed": len(und),
                  "prior_dx_bg": [nbg_prior, nB],
                  "named_mapped": len(named & own), "named_unmapped": len(unmapped),
                  "profile": [[ctx.code_name(d, v), rt, rp, rb, lift]
                              for (d, v), rt, rp, rb, lift in prof],
                  "profile_undiagnosed": [[ctx.code_name(d, v), rt, rp, rb, lift]
                                          for (d, v), rt, rp, rb, lift in prof_u]}
    text = "\n".join(L_).rstrip() + "\n"
    stem = "lookalikes" + (f"_{tag}" if tag else "")
    (run_dir / f"{stem}.md").write_text(text)
    (run_dir / f"{stem}.json").write_text(json.dumps(js, indent=1, default=str))
    print(text, flush=True)
    _cprint(f"[lookalikes]   wrote {run_dir / (stem + '.md')}", flush=True)
    return js


def run_prevalence_check(test_scored, run_dir, *, C, parent_int,
                         heads_tag="stacked_mass99", tag=None, top=15):
    """How much of the case-finding AUC is prevalent recall? (insight 0100;
    `prevalence_check.py`). Scores the test split with the saved heads (no solve), flags
    each (patient, node) with a mapped attestation code of the node or a label
    descendant in the PRE-INDEX condition features, and reports per-node AUC over all
    patients vs with those patients set aside — flat heads and the stacked product.
    Writes `prevalence_check[_<tag>].md/.json`."""
    import prevalence_check as pv
    import voi_tables as vt
    from gated_pc_cloud import (_collect_lean_proba, _doc_key_column, closure_matrix,
                                stacked_proba)
    run_dir = Path(run_dir)
    bm, cm = run_dir / "bundle_meta.json", run_dir / "code_map.tsv"
    if not cm.exists():
        raise SystemExit("[prevalence] needs <run>/code_map.tsv (the attestation map)")
    ctx = vt.load_voi_context(run_dir, bundle_meta_path=str(bm) if bm.exists() else None,
                              own_codes_path=str(cm))
    hp = run_dir / f"readout_heads_gated_pc_{heads_tag}.npz"
    z = np.load(hp)
    if int(z["C"]) != int(C):
        raise SystemExit(f"[prevalence] heads {hp.name} are C={int(z['C'])}, frames C={C}")
    proba, y, _m, keys, _e = _collect_lean_proba(
        test_scored, int(C), z["V"], z["b_raw"], degenerate=z["degenerate"],
        const=z["const"], theta_topm=int(z["theta_topm"]))
    P = stacked_proba(proba, closure_matrix(parent_int, int(C)))
    col = "features_0" if "features_0" in test_scored.columns else "features"

    def _codes(row):
        v = row[col]
        idx = getattr(v, "indices", None)
        if idx is None:
            arr = np.asarray(v.toArray() if hasattr(v, "toArray") else v)
            idx = np.flatnonzero(arr)
        return int(row["doc_key"]), [int(i) for i in idx]
    rows = (test_scored.withColumn("doc_key", _doc_key_column(test_scored, "person_id"))
            .select("doc_key", col).rdd.map(_codes).collect())
    codes = dict(rows)
    v2n = pv.vocab_to_nodes(ctx.own_idx, ctx.rdag["pos_sources"])
    prior = pv.prior_matrix([int(k) for k in keys], codes, v2n, int(C))
    nodes = list(range(1, int(C)))
    res = {}
    L_ = [f"# {run_dir.name} — prevalence check: AUC with prior-diagnosis patients set "
          f"aside", f"heads {hp.name} · {len(keys):,} test patients · prior = a mapped "
          f"attestation code of the node (or a label descendant) in the 5-year pre-index "
          f"condition features (a lower bound: older diagnoses count as incident)", ""]
    for arm, score in (("stacked", P), ("flat", proba)):
        per = pv.node_aucs(score, y, prior, nodes)
        sm = pv.summarize(per)
        res[arm] = {"summary": sm, "per_node": per}
        if not sm.get("n_nodes"):
            L_.append(f"## {arm}: no node with >= {pv.EGRESS_FLOOR} positives both ways")
            continue
        L_ += [f"## {arm} — {sm['n_nodes']} nodes · macro AUC all {sm['macro_auc_all']:.4f}"
               f" → incident {sm['macro_auc_incident']:.4f} · gap median "
               f"{sm['gap_median']:+.4f}, p90 {sm['gap_p90']:+.4f}, > 0.05 on "
               f"{sm['n_gap_over_0p05']} nodes · median prior share of positives "
               f"{sm['prior_share_pos_median']:.2f}", "",
               "| prior share of positives | nodes | AUC all | AUC incident | median gap |",
               "|---|--:|--:|--:|--:|"]
        for band, b in sm["by_prior_share"].items():
            L_.append(f"| {band} | {b['n_nodes']} | {b['auc_all']:.3f} | "
                      f"{b['auc_incident']:.3f} | {b['median_gap']:+.3f} |")
        worst = sorted(per.items(), key=lambda kv: -(kv[1]["auc_all"]
                                                     - kv[1]["auc_incident"]))[:int(top)]
        L_ += ["", f"largest gaps ({arm}):"]
        for c, v in worst:
            L_.append(f"- {str(ctx.rd_names.get(c, c))[:40]}: {v['auc_all']:.3f} → "
                      f"{v['auc_incident']:.3f} · prior share of positives "
                      f"{v['prior_share_pos']:.2f} · positives {v['n_pos_all']} → "
                      f"{v['n_pos_incident']}")
        L_.append("")
    text = "\n".join(L_).rstrip() + "\n"
    stem = "prevalence_check" + (f"_{tag}" if tag else "")
    (run_dir / f"{stem}.md").write_text(text)
    (run_dir / f"{stem}.json").write_text(json.dumps(res, indent=1, default=str))
    print(text, flush=True)
    _cprint(f"[prevalence]   wrote {run_dir / (stem + '.md')}", flush=True)
    return res


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Post-hoc case-finding readout on a finished gated_pc run "
                    "(no re-fit): pc_topics_lr + precision@recall / recall@FDR / "
                    "detection for the gated_pc arm.",
        epilog="Submit via `make gated-pc-readout ID=N` (analysis/cloud/Makefile). A "
               "hand-rolled spark-submit MUST pass --py-files "
               "<repo>/analysis/cloud/distributed_readout.py alongside the source "
               "zips: --readout-mode distributed ships its partition kernels by "
               "module reference and they must import on the executors.")
    p.add_argument("--run-dir", required=True,
                   help="Run dir with gated_pc_result.npz + manifest.json.")
    p.add_argument("--cache-uri", default=None,
                   help="Bundle cache root the fit used (its --cache-uri). Defaults "
                        "to the one the manifest recorded; required only when "
                        "neither that nor --bundle-path is available.")
    p.add_argument("--bundle-path", default=None,
                   help="Exact cached bundle dir ({cache_uri}/{key}); bypasses the "
                        "key recompute if the manifest is incomplete.")
    p.add_argument("--doc-min-length", type=int, default=None,
                   help="Override doc_min_length for the cache key (older manifests "
                        "did not record it; the current driver does).")
    p.add_argument("--conditional-from-heads", default=None, metavar="TAG",
                   help="WP-D part 2: CONDITIONAL DIAGNOSIS from a finished readout's "
                        "saved heads (readout_heads_gated_pc_<TAG>.npz, e.g. "
                        "stacked_mass99) — no solve. Transforms the TEST split only, "
                        "widens it to the readout DAG, collects the per-doc proba and "
                        "scores, per Mondo parent -> child edge of the readout DAG, "
                        "the within-parent-cohort cond AUC, top-1 vs majority and "
                        "ECE, for the flat heads and the stacked product. Writes "
                        "results_conditional_<TAG>.json.")
    p.add_argument("--acquisition", default=None, metavar="REGEX",
                   help="WP-D part 3: SIMULATED ACQUISITION on held-out patients for "
                        "the candidate sets matching REGEX (as inspect_topics --voi): "
                        "ask codes in greedy-EIG / static / random order, score the "
                        "posterior on the true disease. Needs the run's "
                        "readout_dag.json, code_map.tsv; no transform, no solve. "
                        "Writes acquisition[_<tag>].md/.json.")
    p.add_argument("--differential", default=None, metavar="REGEX",
                   help="Framing 1 (spec 2026-10-13): DIFFERENTIAL WITH REASONS for the "
                        "candidate sets matching REGEX — four engines (presence "
                        "subtypes, stacked heads, multiclass head, gated likelihood) "
                        "scored on held-out single-positive patients. --acq-adhoc, "
                        "--acq-heads and --acq-tag apply. Writes "
                        "differential[_<tag>].md/.json.")
    p.add_argument("--lookalikes", default=None, metavar="REGEX",
                   help="Framing 3 (spec 2026-10-13): UNDIAGNOSED LOOK-ALIKES for every "
                        "member of the candidate sets matching REGEX — test patients "
                        "without the label ranked by the saved stacked heads "
                        "(--acq-heads), the top --look-top profiled. Transforms the "
                        "test split only; --acq-adhoc and --acq-tag apply.")
    p.add_argument("--look-top", type=int, default=200)
    p.add_argument("--prevalence-check", action="store_true",
                   help="Insight 0100: per-node AUC of the saved heads (--acq-heads) "
                        "over all test patients vs with prior-diagnosis patients set "
                        "aside (mapped own code in the pre-index features). Transforms "
                        "the test split only.")
    p.add_argument("--readout-incident", action="store_true",
                   help="Exp 0144 (insight 0100): INCIDENT heads on the saved fit — "
                        "every (patient, node) cell with prior evidence for the node "
                        "(mapped own code, or a label descendant's, in the pre-index "
                        "condition features) is masked in BOTH splits before the "
                        "solve; outputs are tagged 'incident_…'; then the prevalence "
                        "check runs on the new heads over the unmasked test split. "
                        "Needs the Mondo readout DAG and <run>/code_map.tsv.")
    p.add_argument("--diff-max-patients", type=int, default=600)
    p.add_argument("--diff-max-train", type=int, default=2000)
    p.add_argument("--diff-next-question", action="store_true",
                   help="(--differential) also run framing 2 — the next-question "
                        "evaluation (hide --diff-nq-hide of each record; yield and gain "
                        "of the top-k recommendations under the E3 and E1 "
                        "differentials)")
    p.add_argument("--diff-nq-hide", type=float, default=0.2,
                   help="(--diff-next-question) fraction of each record hidden; 0.8 = "
                        "early in a workup (20%% visible)")
    p.add_argument("--acq-adhoc", action="store_true",
                   help="(--acquisition) REGEX picks label nodes as ONE set")
    p.add_argument("--acq-prior", choices=("population", "uniform"),
                   default="population")
    p.add_argument("--acq-domains", default="condition,drug",
                   help="(--acquisition) comma list of domains to ask about")
    p.add_argument("--acq-pool", type=int, default=150)
    p.add_argument("--acq-steps", type=int, default=20)
    p.add_argument("--acq-max-patients", type=int, default=5000)
    p.add_argument("--acq-include-strata", action="store_true",
                   help="(--acquisition) census-rule sensitivity: strata join the "
                        "signature profiles")
    p.add_argument("--acq-tag", default=None)
    p.add_argument("--acq-length", choices=("distinct", "tokens"), default="distinct",
                   help="(--acquisition) record length L_i: distinct codes (default) "
                        "or token count (0138's first run; overconfident)")
    p.add_argument("--acq-engine", choices=("presence", "model"), default="presence",
                   help="(--acquisition) posterior engine: the per-code presence model "
                        "(part 3) or the record's own θ fold-in + saved readout heads "
                        "(part 3b; --acq-heads names the heads)")
    p.add_argument("--acq-heads", default="stacked_mass99",
                   help="(--acq-engine model) readout_heads_gated_pc_<TAG>.npz")
    p.add_argument("--acq-tau", default="auto",
                   help="(--acquisition) likelihood temper: 'auto' (fit on half the "
                        "patients, scored on the other half) or a float (1 = none)")
    p.add_argument("--bundle-only", action="store_true",
                   help="Load (or rebuild) the bundle, write <run>/bundle_meta.json "
                        "and <run>/code_map.tsv, and stop: no transform, no solve. "
                        "What a fresh cluster runs before `inspect-topics` on an old "
                        "run (~20 min on a MISS).")
    p.add_argument("--no-rebuild", action="store_true",
                   help="FAIL FAST on a cache miss instead of re-assembling the "
                        "corpus from the manifest's parameters. The old behaviour: "
                        "use it when a miss should be investigated (a wrong "
                        "--cache-uri, a drifted assembly source) rather than paid "
                        "for — a whole-Mondo rebuild is ~5 min of BigQuery plus the "
                        "assembly itself.")
    # Rebuild inputs a manifest may predate. Each defaults to the manifest value;
    # pass them when re-reading a run whose manifest was written before the corpus
    # spec was recorded in full (they are cache-KEY inputs, so a wrong value misses
    # the cache rather than mis-scoring).
    p.add_argument("--billing", default=None,
                   help="Billing project for a rebuild (default: the manifest's).")
    p.add_argument("--dag-source", choices=["snomed", "mondo", "mondo_native"],
                   default=None,
                   help="The fit's --dag-source. Default: the manifest's, which for "
                        "runs predating this field is absent — so a MONDO run from "
                        "before it was recorded must pass --dag-source mondo, or it "
                        "keys (and would rebuild) as the disease-anchored SNOMED "
                        "corpus instead. Every exp-0110 (mondo_native) run records "
                        "it, so an override there can only contradict the manifest "
                        "and is rejected (exit 2).")
    p.add_argument("--cache-write", choices=["on", "off"], default="on",
                   help="write a rebuilt bundle through to --cache-uri (default on, "
                        "so the next readout of this run is a HIT). 'off' keeps the "
                        "rebuild in memory only.")
    p.add_argument("--mondo-version", default=None,
                   help="mondo runs: the fit's --mondo-version (default: manifest).")
    p.add_argument("--mondo-branch", default=None,
                   help="mondo runs: the fit's --mondo-branch, '' = whole Mondo "
                        "(default: manifest).")
    p.add_argument("--min-positives", type=int, default=None,
                   help="mondo runs: the fit's --min-positives (default: manifest).")
    p.add_argument("--mondo-cache-dir", default=None,
                   help="mondo runs: local Mondo OBO cache dir for a rebuild "
                        "(default: the manifest's, else data/mondo).")
    p.add_argument("--label-set", default=None,
                   help="ribbon runs: where the fit's label-set TSV lives NOW, for a "
                        "rebuild on a cache MISS (default: the manifest's path). The "
                        "ribbon's identity is always the manifest's; a TSV whose "
                        "identity differs is rejected at build time.")
    p.add_argument("--dag-collapse", choices=["on", "off"], default=None,
                   help="mondo runs: whether the fit used the exp-0109 "
                        "splice-to-fixpoint DAG reduction (--dag-collapse). It is a "
                        "cache-KEY input and it changes the DAG a rebuild produces, "
                        "so a wrong value misses the cache and would rebuild a "
                        "different corpus (the drift gate catches it). Default: the "
                        "manifest's, else off — every fit predating the flag.")
    p.add_argument("--preindex-closure", choices=["on", "off"], default=None,
                   help="mondo runs: whether the fit's corpus carries E1's "
                        "pre-index closure column (--preindex-closure). It is a "
                        "cache-KEY input and it changes what a rebuild produces, "
                        "so a wrong value misses the cache and would rebuild a "
                        "different bundle. Default: the manifest's, else off — "
                        "every fit predating the flag.")
    p.add_argument("--recall-targets", default="0.5,0.8,0.9")
    p.add_argument("--fdr-targets", default="0.1,0.25,0.5")
    p.add_argument("--min-label-count", type=int, default=None,
                   help="Small-cell mask floor; default = the run's own value.")
    p.add_argument("--readout-mode", choices=["driver", "distributed", "auto"],
                   default="auto",
                   help="where the pc_topics_lr readout is fit, same semantics as "
                        "gated_pc_cloud's flag (ADR 0046): 'driver' collects per-doc "
                        "theta + dense label/mask for BOTH splits and fits C sklearn "
                        "LRs on the driver; 'distributed' fits all C heads in ONE "
                        "batched L-BFGS on the executors and collects only the lean "
                        "float32/uint8 test-split eval bundle; 'auto' (default) = "
                        f"driver at C<={_DRIVER_READOUT_MAX_C}, else distributed. "
                        "Note the run's OWN --readout-mode is not consulted: the "
                        "collect that has to fit is this driver's, not the fit's.")
    p.add_argument("--readout-ab-check", action="store_true",
                   help="run BOTH readout paths on the same re-transformed frames and "
                        "print the deltas (macro AUC/AP, per-node |dAUC|, sampled max "
                        "|dproba|). Report only — never asserts. Ignored unless the "
                        f"mode resolves to distributed AND C<={_DRIVER_READOUT_MAX_C} "
                        "(the driver path must still be affordable to compare to).")
    p.add_argument("--eval-path", choices=["driver", "distributed"], default="driver",
                   help="EVAL path for the gated_pc ranking arms (exp 0111 WP-B), "
                        "orthogonal to --readout-mode. 'distributed' scores the "
                        "prevalent + incident per-node ranking via the cell explode "
                        "(no O(N.C) driver collect); 'driver' (default) uses the "
                        "shipped lean collect. Only meaningful when the readout "
                        "resolves to distributed. This is the flag the WP-B CLUSTER "
                        "parity run flips against the 0110 corpus.")
    p.add_argument("--readout-theta-mass", type=float, default=None,
                   help="OVERRIDE: resolve the top-m width from a theta-MASS target "
                        "on the train frame (spec 2026-10-07 §D4; smallest m in "
                        "{16,32,...,K} with p10 coverage >= target). Tags outputs "
                        "mass<pct>. Default: the manifest's resolved "
                        "readout_theta_topm; a fit-only manifest that recorded a "
                        "readout_theta_mass but no resolved width re-resolves it.")
    p.add_argument("--readout-theta-topm", type=int, default=None,
                   help="OVERRIDE the manifest's readout_theta_topm for this re-readout "
                        "(0 forces full-K). Default: the manifest's value, which "
                        "reproduces the fit's own design matrix. The override exists "
                        "for exactly one job: PRICING truncation — re-read a full-K "
                        "run (e.g. the 0103 cardiovascular record) with top-m forced "
                        "on and compare per-node AUC against its recorded numbers, so "
                        "the whole-Mondo enablement decision rests on a measured "
                        "delta, not the mass-coverage heuristic alone.")
    p.add_argument("--readout-max-iter", type=int, default=None,
                   help="OVERRIDE the batched L-BFGS iteration cap for the "
                        "distributed re-readout. Default: the manifest's recorded "
                        "readout_max_iter — the cap the fit itself used, CHARM_DEV "
                        "capping included, so a recovery reproduces the run it is "
                        "rescuing without being told. Manifests written before that "
                        f"field fall back to {_LEGACY_READOUT_MAX_ITER} (the "
                        "record-run budget); recovering a DEV smoke from one of "
                        "those still needs an explicit 60.")
    p.add_argument("--readout-l2", type=float, default=None,
                   help="OVERRIDE the ridge strength of the per-node readout "
                        "heads (SUMMED log-loss; 1.0 == sklearn C=1.0). Default: "
                        "the manifest's recorded readout_l2, else 1.0 — so a "
                        "recovery reproduces the fit's own readout. At record "
                        "cell counts 1.0 is near-unregularized over K "
                        "standardized features: 0120 went 0.7555 -> 0.7927 at "
                        "100 (insight 0089). A sweep is a named re-readout.")
    p.add_argument("--readout-feature-mask", choices=_FEATURE_MASK_MODES,
                   default="all",
                   help="FEATURE ABLATION of the per-node readout heads: restrict "
                        "each node's head to its own topic block (own), own+background "
                        "(own-bg), background only (bg), the gate's closure (closure), "
                        "or everything but its own block / closure (drop-own, "
                        "drop-closure). Answers whether the gated per-node block "
                        "carries the node's case-finding signal at all (insight "
                        "0089: full heads put ~0 weight on it). Distributed mode "
                        "only. Writes results_readout_<mode>.json and "
                        "readout_heads_gated_pc_<mode>.npz — the record files are "
                        "never touched.")
    p.add_argument("--readout-stacked", action="store_true",
                   help="Add the STACKED arm (spec 2026-10-06 Part A): the root "
                        "head is fit on every train row (case vs background) in "
                        "the same batched solve, and each node is scored by the "
                        "closure PRODUCT of its ancestors' heads, "
                        "P_stack(c)=prod_{a in closure(c)} sigma(z_a) — HSLDA's "
                        "label-side gating on post-hoc heads. Outputs go to "
                        "results_readout_stacked.json / "
                        "readout_heads_gated_pc_stacked.npz (the record files are "
                        "untouched); the block carries the flat, root-only and "
                        "stacked detection side by side. Distributed readout + "
                        "driver eval path only.")
    p.add_argument("--readout-hierarchy", choices=["auto", "mondo", "flat"],
                   default="auto",
                   help="WP-B (spec 2026-10-07 §D1/§D4, insight 0096): read a FLAT "
                        "label-set fit through the Mondo hierarchy. 'mondo' builds "
                        "the readout DAG over the fit's label nodes + every Mondo "
                        "ancestor grouping >= 2 of them (root aliases folded, rungs "
                        "dropped), widens label/labelMask on the scored frames so a "
                        "node is observed only where it or a sibling is positive "
                        "(negatives = siblings under the nearest grouping ancestor), "
                        "appends the ancestor heads, and feeds that DAG to the "
                        "stacked product. 'auto' (default) = 'mondo' for label-set "
                        "fits, 'flat' otherwise (every nested run is byte-identical). "
                        "'flat' on a label-set fit tags its outputs _flat. Writes "
                        "<run>/readout_dag.json. Incompatible with "
                        "--readout-feature-mask; the incident arm is skipped.")
    return p


def resolve_run_dir(pattern):
    """Resolve --run-dir to a single gated_pc run directory.

    An exact dir that holds gated_pc_result.npz is used as-is. Otherwise the value
    is treated as a glob (the Makefile passes `.../<id>-*` QUOTED so the shell does
    not pre-expand it) and filtered to matched dirs that contain gated_pc_result.npz
    — so a runs/ numbering COLLISION with a non-gated_pc experiment of the same id
    (e.g. a stale `0076-multidomain-...` from another branch) is discarded
    automatically. Requires exactly one gated_pc match; a clear error otherwise."""
    p = Path(pattern)
    if p.is_dir() and (p / "gated_pc_result.npz").exists():
        return p
    matches = [Path(m) for m in glob.glob(str(pattern))]
    hits = [m for m in matches if m.is_dir() and (m / "gated_pc_result.npz").exists()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SystemExit(
            f"[readout] no run dir with gated_pc_result.npz matches {pattern!r} "
            f"(matched: {[m.name for m in matches]}). Has the fit finished + saved? "
            f"Pass an exact --run-dir / GPR_RUN_DIR.")
    raise SystemExit(
        f"[readout] {len(hits)} gated_pc run dirs match {pattern!r}: "
        f"{[m.name for m in hits]}. Pass an exact --run-dir / GPR_RUN_DIR.")


# The single-domain assembler's kwargs that are also cache-key inputs (billing is
# an assembly input only). Kept next to the spec builder so "what the key needs"
# and "what a rebuild needs" cannot drift apart.
#
# `doc_spec` is NOT here for the same reason it is not in
# `gated_pc_cloud._SPEC_ASSEMBLY_KEYS`: these names are forwarded to the
# assembler (`rebuild_bundle` below), which takes no doc-spec kwarg. It is
# key-only, and is passed explicitly at each of the two sites that need it.
_SNOMED_KEY_KEYS = (
    "source_table", "person_mod", "vocab_size", "min_df", "min_patient_count",
    "doc_min_length", "prior_obs_days", "window_days", "disease", "min_n",
    "holdout_frac", "n_bg", "tpn", "cdr", "strip_mode", "window_mode",
    "lookback_days", "label_window_days", "emit_labels", "label_mask_mode",
)


def corpus_spec_from_manifest(manifest: dict, *, doc_min_length=None, billing=None,
                              dag_source=None, mondo_version=None, mondo_branch=None,
                              min_positives=None, mondo_cache_dir=None,
                              dag_collapse=None, preindex_closure=None,
                              label_set_path=None) -> dict:
    """The corpus SPEC a gated_pc manifest describes — key inputs + rebuild inputs.

    The current driver writes this dict into `corpus_manifest` verbatim, so for any
    recent run this is just a read-back. Older manifests kept some fields only at
    the top level (and the Mondo build inputs not at all), so every field falls back
    `corpus_manifest` -> top level -> a documented default, and the CLI overrides
    supply what neither has. Raises KeyError naming the field when a required one is
    missing everywhere."""
    from _case_finding_cache import DEFAULT_DOC_SPEC

    cm = manifest.get("corpus_manifest", {})

    _MISSING = object()

    def _pick(name, default=_MISSING):
        if name in cm and cm[name] is not None:
            return cm[name]
        if name in manifest and manifest[name] is not None:
            return manifest[name]
        if default is _MISSING:
            raise KeyError(
                f"{name} is not in the manifest (neither corpus_manifest nor the "
                f"top level); it is a cache-key input, so the bundle cannot be "
                f"located. Pass --bundle-path at the exact cached dir.")
        return default

    dml = doc_min_length if doc_min_length is not None else cm.get("doc_min_length")
    if dml is None:
        raise KeyError(
            "doc_min_length is not in the manifest and no --doc-min-length was "
            "given; it is a cache-key input, so the bundle cannot be located. "
            "Pass --doc-min-length (the fit's value) or --bundle-path.")
    dag_source = str(dag_source if dag_source is not None
                     else _pick("dag_source", "snomed"))
    # Both Mondo flavours (exp 0088/0104's anchor hierarchy and exp 0110's native
    # label space) share the population index, min_n=0 and the Mondo build inputs.
    mondo = dag_source in _MONDO_DAG_SOURCES
    extra_domains = list(_pick("extra_domains", []) or [])
    spec = {
        "dag_source": dag_source,
        "extra_domains": extra_domains,
        # The Mondo fit indexes the whole POPULATION and passes min_n=0 (its DAG is
        # already powered); both are defaults here so a manifest written before
        # corpus_manifest carried them still recomputes the fit's own key.
        "index_mode": _pick("index_mode", "population" if mondo else "disease"),
        "min_n": (cm["min_n"] if "min_n" in cm
                  else (0 if mondo else manifest["min_n"])),
        "disease": _pick("disease"),
        "cdr": _pick("cdr", None),
        "billing": billing if billing is not None else _pick("billing", None),
        "source_table": _pick("source_table", "condition_era"),
        "person_mod": _pick("person_mod"), "vocab_size": _pick("vocab_size"),
        "min_df": _pick("min_df"), "min_patient_count": _pick("min_patient_count"),
        "doc_min_length": int(dml),
        "prior_obs_days": _pick("prior_obs_days", 0),
        "window_days": _pick("window_days", 0),
        "holdout_frac": _pick("holdout_frac"),
        "n_bg": _pick("n_bg"), "tpn": _pick("tpn"),
        "strip_mode": _pick("strip_mode"), "window_mode": _pick("window_mode"),
        "lookback_days": _pick("lookback_days"),
        "label_window_days": _pick("label_window_days"),
        "label_mask_mode": _pick("label_mask_mode", "full"),
        "emit_labels": True,               # a gated_pc corpus always carries labels
        "mondo_version": (mondo_version if mondo_version is not None
                          else _pick("mondo_version", "")),
        "mondo_branch": (mondo_branch if mondo_branch is not None
                         else _pick("mondo_branch", "")),
        "min_positives": (min_positives if min_positives is not None
                          else _pick("min_positives", 0)),
        "mondo_cache_dir": (mondo_cache_dir if mondo_cache_dir is not None
                            else _pick("mondo_cache_dir", "data/mondo")),
        # exp 0109. Defaults to False, which is what EVERY manifest written before
        # the flag existed means — so an old run still recomputes its own key.
        # On the exp-0110 native path the splice is intrinsic to the build, so the
        # flag is pinned False there and cannot double-apply (this mirrors
        # `gated_pc_cloud.multidomain_corpus_spec`, which is what wrote the field).
        "dag_collapse": (False if dag_source == "mondo_native" else
                         bool(dag_collapse if dag_collapse is not None
                              else _pick("dag_collapse", False))),
        # Spec 2026-10-07 §D1: the ribbon. `label_set` is the KEY identity and is
        # always the manifest's (a readout cannot be told to key under a different
        # ribbon than the fit used); `label_set_path` is the rebuild input and may
        # be overridden when the TSV moved. Both default to '' — what every
        # manifest written before the field existed means.
        "label_set": str(_pick("label_set", "") or ""),
        "label_set_path": str(label_set_path if label_set_path is not None
                              else _pick("label_set_path", "") or ""),
        # The DOC UNIT (R5.3 / audit seam 4). A cache-key input as of this
        # change, and the default is what EVERY manifest written before the field
        # existed means — every corpus in the repo was assembled under it — so an
        # old run still recomputes its own byte-identical key. There is no CLI
        # override: unlike the Mondo build inputs, the doc unit is not something a
        # rebuild can be told to differ on, because the assembler hard-codes it.
        "doc_spec": str(_pick("doc_spec", DEFAULT_DOC_SPEC)),
        # exp 0111 WP-D2: the external episode/random index identity. Defaults to
        # None (absent) — what every manifest written before the arms existed means,
        # so an old run recomputes its byte-identical key — and is folded into the
        # bundle key ONLY on the `index_mode="external"` path (multidomain_cache_key),
        # so a re-readout of an episode bundle recomputes the fit's OWN key. The
        # sidecar URI rides too but is NOT a key input; it only tells a re-readout
        # MISS where to reload/rebuild the index's sidecar.
        "episode_sampling": _pick("episode_sampling", None),
        "episode_sidecar_uri": _pick("episode_sidecar_uri", ""),
        # exp 0111 WP-D3 (D13): the gate/label-separation identity. Defaults to None
        # (absent) — what every manifest written before D13 means, so an old run
        # recomputes its byte-identical key — and is folded into the bundle key ONLY
        # on the external path (multidomain_cache_key), so a re-readout of a gated
        # episode bundle recomputes the fit's OWN key. The readout reads the 365-day
        # `label`/`labelMask` regardless; this field only routes the cache key.
        "gate_frontier_mode": _pick("gate_frontier_mode", None),
        # E1 (exp E-census onward). Mondo-only, defaults to False — what every
        # manifest written before the flag existed means — and the CLI wins in
        # both directions for a fit whose manifest omits or contradicts it, the
        # same tri-state shape `dag_collapse` uses. It is a cache-key input, so a
        # wrong value MISSES the cache rather than mis-scoring.
        "preindex_closure": (bool(preindex_closure
                                  if preindex_closure is not None
                                  else _pick("preindex_closure", False))
                             if mondo else False),
    }
    return spec


def spec_is_multidomain(spec) -> bool:
    """True when the corpus came from the multi-domain assembler (either Mondo
    flavour, or plain extra domains) — the shapes whose cache key carries the
    extra fold. Missing `mondo_native` here would recompute a SNOMED key for an
    exp-0110 run, i.e. a guaranteed MISS followed by a ~20-minute rebuild of the
    wrong corpus, which is exactly the failure `mondo_spec_mismatch` was written
    for after exp 0104's fresh-cluster recovery."""
    return (str(spec.get("dag_source", "snomed")) in _MONDO_DAG_SOURCES
            or bool(spec.get("extra_domains")))


def native_spec_mismatch(spec, manifest) -> bool:
    """True when the SAVED FIT is an exp-0110 native run but the rebuild spec
    resolved to something else.

    The sibling of `mondo_spec_mismatch` for the native label space, and it needs
    its own witness: a native fit's `int2cid` values are plain ints (the Mondo
    curie's numeric part — see `mondo_native_dag`'s docstring on why they cannot
    be curie strings), so the `MONDO:`-prefix test cannot see it. The manifest's
    own `corpus_manifest.dag_source` is the witness instead, which every native
    run records; the only way to reach here is a CLI override that contradicts
    it, and silently rebuilding a DIFFERENT label space under a wrong key is worth
    an exit 2 rather than a drift-gate failure 20 minutes later."""
    cm = manifest.get("corpus_manifest") or {}
    saved = str(cm.get("dag_source") or manifest.get("dag_source") or "")
    return saved == "mondo_native" and str(spec.get("dag_source")) != "mondo_native"


def mondo_spec_mismatch(spec, manifest) -> bool:
    """True when the rebuild spec says snomed but the SAVED FIT is a Mondo run.

    A manifest written before ``corpus_manifest`` recorded ``dag_source`` resolves
    to the snomed default, and a Mondo fit then keys — and, on the MISS that wrong
    key guarantees — REBUILDS the wrong corpus. That cost a fresh-cluster recovery
    of exp 0104 a rebuild attempt (it also surfaced as a billing NPE, but the
    corpus was already wrong before BigQuery was ever reached). The fit itself is
    the witness the manifest lacks: a Mondo run's label space is Mondo ids, so
    ``int2cid`` carries the ``MONDO:`` prefix. Detecting the mismatch here — BEFORE
    the key is computed — turns ~20 minutes of rebuilding the wrong thing (caught
    only by the drift gate, exit 3) into an immediate exit 2 naming the flags."""
    int2cid = manifest.get("int2cid") or []
    cids = int2cid.values() if isinstance(int2cid, dict) else int2cid
    looks_mondo = any(str(c).startswith("MONDO:") for c in cids)
    return looks_mondo and str(spec.get("dag_source")) != "mondo"


def bundle_key_from_manifest(manifest: dict, *, doc_min_length=None, **spec_over):
    """Recompute the bundle cache key from a gated_pc manifest.

    Routes on the corpus the manifest describes: a mondo / extra-domains run gets
    the MULTI-DOMAIN key (the same `multidomain_cache_key` the fit computes, folding
    extra_domains / index_mode / the Mondo build inputs), anything else the
    single-domain key, byte-identical to what it has always been. `doc_min_length`
    and the `spec_over` fields override the manifest (older manifests omit them)."""
    from _case_finding_cache import compute_bundle_cache_key

    spec = corpus_spec_from_manifest(manifest, doc_min_length=doc_min_length,
                                     **spec_over)
    if spec_is_multidomain(spec):
        return multidomain_cache_key(spec)
    return compute_bundle_cache_key(doc_spec=spec["doc_spec"],
                                    **{k: spec[k] for k in _SNOMED_KEY_KEYS})


def rebuild_bundle(spark, spec, *, cache_uri=None):
    """Re-assemble the corpus a finished fit was built against, and write it through.

    The SAME seam the fit uses — `multidomain_load_or_build` (Mondo DAG build +
    SNOMED-climb provider included, on the miss it is about to take) for a
    multi-domain corpus, `load_or_build_case_finding_bundle` for a single-domain
    one — so the bundle this produces lands under the key the fit's own bundle would
    have, and the next readout (or fit) of this corpus is a HIT."""
    if spec_is_multidomain(spec):
        return multidomain_load_or_build(spark, spec, cache_uri=cache_uri)
    from _case_finding_cache import load_or_build_case_finding_bundle
    params = {k: spec[k] for k in _SNOMED_KEY_KEYS}
    return load_or_build_case_finding_bundle(
        spark, cache_uri=cache_uri, billing=spec["billing"],
        # key-only, exactly as the fit passes it (gated_pc_cloud's single-domain
        # call site), so a rebuild lands under the key the fit stored under.
        _key_extra={"doc_spec": spec["doc_spec"]}, **params)


def lambda_vocab_sizes(global_params) -> list:
    """The V dimension of each domain's λ, in domain order — the saved fit's own
    record of how wide each vocabulary was. Multi-domain λ is a {domain: (K, V_m)}
    dict; single-domain is one (K, V) array."""
    lam = global_params["lambda"]
    if isinstance(lam, dict):
        return [int(lam[m].shape[1]) for m in sorted(lam)]
    return [int(lam.shape[1])]


def bundle_drift_report(bundle, manifest, fit_vocab_sizes) -> list:
    """Reasons the corpus in hand cannot be scored against the saved fit (empty = ok).

    Re-assembly is deterministic in practice but nothing guarantees it: the CDR
    advances, an assembly-source edit changes the split or the vocabulary fit, a
    Mondo release re-parents a branch. The λ that comes back from the npz is
    (K, V_m) per domain and the per-node head is (C, K) over engine ids, so the two
    things that MUST still line up are each domain's vocabulary width and the
    engine-id -> concept-id map. Both are recorded — the widths implicitly in λ, the
    map explicitly in the manifest — so drift is detectable before a single number
    is computed, and reporting AUCs for a model scored on a corpus it was not fit on
    is not a failure mode this tool has to have."""
    problems = []
    vocab_maps = getattr(bundle, "vocab_maps", None)
    if vocab_maps is None:
        vocab_maps = [bundle.vocab_map]
    sizes = [len(vm) for vm in vocab_maps]
    if len(sizes) != len(fit_vocab_sizes):
        problems.append(
            f"the saved fit has {len(fit_vocab_sizes)} domain(s) "
            f"(lambda {fit_vocab_sizes}) but the corpus has {len(sizes)} "
            f"(vocab sizes {sizes})")
    else:
        for i, (v_fit, v_corpus) in enumerate(zip(fit_vocab_sizes, sizes)):
            if int(v_fit) != int(v_corpus):
                problems.append(
                    f"domain {i}: the fit's lambda is V={v_fit} wide, the corpus "
                    f"vocabulary has {v_corpus} concepts")
    got = {int(i): int(c) for i, c in bundle.int2cid.items()}
    C = manifest.get("C")
    if C is not None and len(got) != int(C):
        problems.append(
            f"the fit has C={int(C)} label heads, the corpus DAG has {len(got)} nodes")
    saved = (manifest.get("corpus_manifest") or {}).get("int2cid")
    if saved:
        want = {int(i): int(c) for i, c in saved.items()}
        if got != want:
            diff = sorted(i for i in set(got) | set(want)
                          if got.get(i) != want.get(i))
            problems.append(
                f"the engine-id -> concept-id map differs at {len(diff)} node(s) "
                f"(first: {diff[:5]}); the label DAG is not the one the head was "
                f"fit against")
    return problems


def reconstruct_model(run_dir: Path, manifest: dict):
    """Rebuild a scoreable OnlinePCLDAModel from the saved gated_pc_result.npz.

    The driver saves raw globals (lambda/alpha/w_CK), not a persisted model, so we
    wrap them in a VIResult and an OnlinePCLDAModel and set weightY>0 + numLabels so
    transform appends both topicDistribution (θ) and the head probability. The CAVI
    read-out knobs (gammaShape/caviMaxIter/caviTol) come from the model defaults,
    which match the fit's defaults for this experiment family.

    MULTI-DOMAIN (the Mondo path): `np.savez` cannot store a dict, so `_save_fit`
    writes a per-domain λ as `lambda_0, lambda_1, ...`; it is reassembled here into
    the `{domain: (K, V_m)}` dict `_transform` fuses, and `featuresCols` is set to
    the matching `features_0..` so the transform reads the per-domain BOW columns
    instead of a `features` column the multi-domain corpus does not have."""
    from spark_vi.core.result import VIResult
    from spark_vi.mllib.topic.pc import OnlinePCLDAModel

    npz = np.load(run_dir / "gated_pc_result.npz")
    files = set(npz.files)
    if "lambda" in files:
        lam = npz["lambda"]
    else:
        dom = sorted(int(f.split("_", 1)[1]) for f in files
                     if f.startswith("lambda_"))
        if not dom:
            raise KeyError(
                f"{run_dir}/gated_pc_result.npz has no lambda: found {sorted(files)}")
        lam = {m: npz[f"lambda_{m}"] for m in dom}
    gp = {"lambda": lam, "alpha": npz["alpha"], "w_CK": npz["w_CK"]}
    if "b_CK" in files:
        # The per-node intercept (--head-intercept) is part of the head; without it
        # the head arm's P(node) would be the intercept-free sigmoid of a model that
        # was fit with one. Absent on runs that predate the save.
        gp["b_CK"] = npz["b_CK"]
    result = VIResult(global_params=gp, elbo_trace=[],
                      n_iterations=int(manifest.get("max_iter", 0)), converged=True)
    model = OnlinePCLDAModel(result)
    model._set(weightY=float(manifest.get("weight_y", 1.0)),
               numLabels=int(manifest["C"]), closureParents="")
    if isinstance(lam, dict):
        model._set(featuresCols=[f"features_{i}" for i in range(len(lam))])
    return model


def run_readout(train_scored, test_scored, manifest, *, recall_targets, fdr_targets,
                min_count, readout_mode="auto", ab_check=False, out_dir=None,
                theta_topm=None, readout_max_iter=None, elig_col=None,
                eval_path="driver", readout_l2=None, feature_mask=None,
                mask_tag=None, stacked=False, parent_int=None, theta_mass=None,
                C=None):
    """Score both gated_pc arms off two already-TRANSFORMED splits. No argparse.

    The whole body of this tool that is worth testing: given the frames a finished
    fit's model produced and that fit's manifest, it routes the pc_topics_lr arm
    through the driver or the distributed readout (ADR 0046), optionally runs the
    A/B equality report, adds the co-fit head arm, and returns
    `{"gated_pc": ..., "gated_pc_head": ...}`. Taking DataFrames rather than a
    SparkSession + run dir is what makes it callable against `score_arm` on a local
    Spark fixture (same reason `distributed_score_arm` is shaped that way);
    argparse, run-dir resolution and the bundle-cache reload stay cluster-covered.

    K comes from the manifest (`lay.K` at fit time) and is only needed by the
    distributed solver; runs old enough to lack it fall back to the width of the
    transform's own theta, which is the same number by construction. `theta_topm`
    and `readout_max_iter` are both None-means-"ask the manifest" for the same
    reason (see `resolve_readout_max_iter`): a recovery reproduces the fit's own
    readout unless explicitly told otherwise.

    `out_dir` gets a `results_readout.json` after EACH arm lands — a re-readout is
    a recovery action (exp 0103 lost a 4h fit's readout to an empty summary), so
    its output has to survive the terminal it was printed to. Deliberately NOT
    results_partial.json: that file belongs to the fit's own record. It is also
    where the SOLVER checkpoint goes (`readout_ckpt_gated_pc.npz`), so a recovery
    that dies mid-solve resumes rather than restarting; it is removed when the
    solve completes.

    `elig_col` (E2/WP4) is E1's pre-index closure column name, resolved by the
    CALLER from the bundle's witness (`main` does it; `None` means "this corpus has
    none"). Given it, the pc_topics_lr arm's lean collect picks up a fourth CSR run
    and this returns a `gated_pc_incident` block beside the prevalent one — which is
    how the incident numbers are actually produced: a re-readout of a SAVED FIT, no
    re-fit, warm bundle. Without it the incident arm is SKIPPED with a printed line
    and every other number is byte-identical to what it was.

    `stacked=True` (spec 2026-10-06 Part A) adds the `gated_pc_stacked` block: the
    TRAIN mask gets the root observed on every row (`observe_root_everywhere`), the
    same batched solve then yields the prevalent heads for nodes ≥ 1 plus a real
    case-vs-background root head, and the closure product of the per-doc
    probabilities is scored by `gated_pc_cloud.stacked_readout`. Needs
    `parent_int` (the bundle's engine-id parent map) and the distributed readout
    on the driver eval path (the product is formed on the collected (D,C) proba).
    The caller tags the outputs (`mask_tag="stacked"`) so the record's
    `results_readout.json` / `readout_heads_gated_pc.npz` are never touched.

    `C` (WP-B) overrides the manifest's label width when the frames have been
    widened to the READOUT node space (`readout_dag.widen_labels_to_readout`);
    `parent_int` is then the readout DAG. None = the fit's own C."""
    C = int(manifest["C"] if C is None else C)
    mode = resolve_readout_mode(readout_mode, C)
    K = int(manifest.get("K") or 0)
    if mode == "distributed" and not K:
        # Only the batched solver needs K, so only it pays for the peek.
        K = int(train_scored.select("topicDistribution").head()[0].size)
    _cprint(f"[readout]   readout_mode={readout_mode} -> {mode} (C={C}, "
          f"K={K or 'n/a'}, driver-collect ceiling C<={_DRIVER_READOUT_MAX_C})",
          flush=True)
    ab = bool(ab_check) and mode == "distributed"
    if bool(ab_check) and not ab:
        _cprint("[readout]   readout_ab_check ignored (readout_mode resolved to driver "
              "— there is nothing to compare against)", flush=True)
    elif ab and C > _DRIVER_READOUT_MAX_C:
        _cprint(f"[readout]   readout_ab_check SKIPPED: C={C} exceeds the driver path's "
              f"own ceiling ({_DRIVER_READOUT_MAX_C}); the gate is meant to run at "
              "cardiovascular scale (C=444)", flush=True)
        ab = False

    arm = "gated_pc" if not mask_tag else f"gated_pc_{mask_tag}"
    results_name = ("results_readout.json" if not mask_tag
                    else f"results_readout_{mask_tag}.json")
    if feature_mask is not None and mode != "distributed":
        raise SystemExit("[readout] --readout-feature-mask needs the distributed "
                         "readout (the driver-collect sklearn path has no mask)")
    if stacked:
        if mode != "distributed" or resolve_eval_path(eval_path, mode) != "driver":
            raise SystemExit("[readout] --readout-stacked needs the distributed "
                             "readout on the DRIVER eval path (the closure product "
                             "is formed on the collected (D,C) proba)")
        if parent_int is None:
            raise SystemExit("[readout] --readout-stacked needs the bundle's "
                             "parent_int (no DAG, no closure)")
        # The one change to the solve: the root is observed on every train row,
        # so its head is case-vs-background instead of the degenerate constant.
        train_scored = observe_root_everywhere(train_scored, C)
        _cprint("[readout]   STACKED arm: root head fit on EVERY train row (case vs "
              "background); heads of nodes >= 1 unchanged (per-node independent "
              "solve)", flush=True)

    def _dump(results):
        if out_dir is not None:
            _dump_partial_results(Path(out_dir), results, name=results_name)

    results = {}
    dist = None
    # E2/WP4. The eligibility column rides the LEAN collect, which only the
    # distributed path performs, and only a corpus built with --preindex-closure
    # carries it. Both misses print a line here rather than failing later: the
    # incident arm is an ADDITION to this tool's output, never a precondition of it.
    if elig_col is not None and mode != "distributed":
        _cprint(f"[readout]   INCIDENT arm (E2) SKIPPED: readout_mode resolved to "
              f"{mode!r} and the eligibility column rides the LEAN (distributed) "
              "collect. Re-run with --readout-mode distributed for the incident "
              "block.", flush=True)
        elig_col = None
    elif elig_col is None:
        _cprint("[readout]   INCIDENT arm (E2) SKIPPED: this corpus carries no "
              "pre-index closure witness (built without --preindex-closure). The "
              "prevalent arms below are unaffected.", flush=True)
    if mode == "distributed":
        # No theta collect: the per-node LRs are fit on the executors and only the
        # lean test-split eval bundle comes back. theta_topm defaults to the
        # MANIFEST's value: a re-readout reproduces the fit's own design matrix —
        # silently refitting a top-m run at full K would change the estimator
        # (and crawl at whole-Mondo, the scale where top-m is on). An explicit
        # override (--readout-theta-topm) exists for the truncation-PRICING
        # experiment: force top-m onto a recorded full-K run and read the delta.
        _mass = None
        if theta_mass is not None and float(theta_mass) > 0:
            _mass = float(theta_mass)
            _cprint(f"[readout]   theta top-m by MASS target {_mass:g} (CLI OVERRIDE)",
                  flush=True)
        elif theta_topm is None and not int(manifest.get("readout_theta_topm", 0) or 0) \
                and float(manifest.get("readout_theta_mass", 0) or 0) > 0:
            # A fit-only manifest: the fit recorded the mass target but died before
            # the transform resolved it. Re-resolve, as the fit would have.
            _mass = float(manifest["readout_theta_mass"])
            _cprint(f"[readout]   theta top-m by MASS target {_mass:g} (from the "
                  "manifest; width not yet resolved there)", flush=True)
        if _mass is not None:
            import distributed_readout as _drm
            _K = int(manifest.get("K") or 0) or None
            theta_topm, _cov = _drm.resolve_theta_topm_by_mass(
                train_scored, _K, _mass)
            _cprint("[readout]   theta top-m by mass: "
                  + " ".join(f"m={m}:{mean:.3f}/{p10:.3f}"
                             for m, (mean, p10) in sorted(_cov.items()))
                  + f" (mean/p10) -> " + (f"m={theta_topm}" if theta_topm else
                                          f"FULL K={_K}"), flush=True)
        elif theta_topm is None:
            theta_topm = int(manifest.get("readout_theta_topm", 0) or 0)
            if theta_topm:
                _cprint(f"[readout]   theta top-m={theta_topm} (from manifest)",
                      flush=True)
        else:
            theta_topm = int(theta_topm)
            _cprint(f"[readout]   theta top-m={theta_topm} (CLI OVERRIDE — deltas vs "
                  "the recorded run price the truncation)", flush=True)
        # Only the batched solver has an iteration cap, so this is the only place
        # the resolution matters — and the log line lands right where the number is
        # about to be spent, next to top-m's.
        readout_max_iter, _mi_src = resolve_readout_max_iter(readout_max_iter, manifest)
        readout_l2, _l2_src = resolve_readout_l2(readout_l2, manifest)
        _cprint(f"[readout]   readout l2={readout_l2:g} (from {_l2_src})", flush=True)
        _cprint(f"[readout]   readout max_iter={readout_max_iter} (from {_mi_src})",
              flush=True)
        _epath = resolve_eval_path(eval_path, mode)
        if _epath == "distributed" and int(theta_topm or 0) > 0:
            _cprint(f"[readout]   eval_path=distributed IGNORED (readout_theta_topm="
                  f"{theta_topm}>0; the cell explode is dense-θ only)", flush=True)
            _epath = "driver"
        if _epath == "distributed":
            # WP-B: fit the heads on the executors, then score COLLECT-FREE via the
            # cell explode (no (D_te,C) proba reaches the driver — audit §5f). This
            # is the `make gated-pc-readout ID=110 --eval-path distributed` path the
            # orchestrator points at the real 0110 corpus for the CLUSTER parity run.
            _cprint("[readout]   eval_path=distributed (WP-B): ranking arms via "
                  "score_cells_arms_df/per_node_metric_arms_rows (no driver "
                  "collect); conditional/detection/PR axes need the collect and are "
                  "skipped — run --eval-path driver for them.", flush=True)
            _ck = Path(out_dir) / f"readout_ckpt_{arm}.npz" if out_dir else None
            _V, _b, _const, _deg, _info = _fit_readout_heads(
                train_scored, C, K, label=arm, max_iter=readout_max_iter,
                l2=readout_l2, theta_topm=theta_topm, checkpoint_path=_ck,
                checkpoint_every=10, feature_mask=feature_mask)
            if out_dir:
                _write_readout_heads(out_dir, arm, _V, _b, _const, _deg,
                                     C, K, theta_topm, W_std=_info.get("W_std"))
            results["gated_pc"], _inc_block = distributed_ranking_readout(
                test_scored, C, _V, _b,
                recall_targets=recall_targets, fdr_targets=fdr_targets,
                min_count=min_count, elig_col=elig_col,
                arm_label=f"{arm} (pc_topics_lr)")
            dist = None
        else:
            dist = distributed_score_arm(
                train_scored, test_scored, C, K, recall_targets=recall_targets,
                fdr_targets=fdr_targets, min_count=min_count, label=arm,
                feature_mask=feature_mask,
                theta_topm=theta_topm, max_iter=readout_max_iter, l2=readout_l2,
                # Same dir as `results_readout.json`, and for the same reason one
                # step earlier in the pipeline: this tool IS the recovery path, and
                # its solve is the multi-hour part. A death mid-solve (the 08-28
                # preemption-wave job abort cost 9,112s) now costs at most
                # `checkpoint_every` iterations, because the next invocation finds
                # the checkpoint next to the manifest it already reads. `None` when
                # no out_dir was given — nowhere durable to put it.
                checkpoint_dir=out_dir, elig_col=elig_col)
            results["gated_pc"] = dist[0]
            _inc_block = None
    else:
        _inc_block = None
        Pi_tr, y_tr, m_tr, _ = _collect_theta_labels(train_scored, C)
        Pi_te, y_te, m_te, _ = _collect_theta_labels(test_scored, C)
        results["gated_pc"] = score_arm(
            Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te, C, recall_targets=recall_targets,
            fdr_targets=fdr_targets, min_count=min_count)
    _dump(results)
    print(format_arm_readout(f"{arm} (pc_topics_lr)", results["gated_pc"]),
          flush=True)
    # E2/WP4 incident block. Two sources, one shape: the collect path scores the
    # incident-masked (D,C) proba here; the eval_path=distributed path already built
    # the block collect-free from the arms explode (`_inc_block`). This is the cheap
    # path the program actually uses — `make gated-pc-readout ID=110` against a warm
    # bundle and a saved fit.
    _inc = _inc_block
    if _inc is None and dist is not None and dist[6] is not None:
        _inc = incident_readout(
            dist[1], dist[2], dist[3], dist[6], C, recall_targets=recall_targets,
            fdr_targets=fdr_targets, min_count=min_count,
            prevalent=results["gated_pc"], arm_label="gated_pc (pc_topics_lr)")
    if _inc is not None:
        results["gated_pc_incident"] = _inc
        _dump(results)
        print(format_incident_readout(_inc).replace("[driver]", "[readout]"),
              flush=True)
    if stacked and dist is not None:
        # Same collected proba the prevalent arm was scored on — one closure
        # product, no second collect, no second solve.
        _stk = stacked_readout(
            dist[1], dist[2], dist[3], parent_int, C, recall_targets=recall_targets,
            fdr_targets=fdr_targets, min_count=min_count,
            prevalent=results["gated_pc"], doc_keys=dist[4], arm_label=arm)
        results["gated_pc_stacked"] = _stk
        _dump(results)
        print(format_stacked_readout(_stk).replace("[driver]", "[readout]"),
              flush=True)
    if ab and dist is not None:
        # Reuses the distributed result just computed, so the gate costs one extra
        # (driver-path) readout, not two.
        readout_ab_report(train_scored, test_scored, C, K,
                          recall_targets=recall_targets, fdr_targets=fdr_targets,
                          min_count=min_count, label="gated_pc", distributed=dist)

    # Co-fit head arm. The scaled-back mainline (unsup gate + post-hoc readout) fits
    # at weightY=0, whose transform appends NO `probability` column — the head does
    # not exist, and asking for it used to fail the whole re-readout with an
    # AnalysisException AFTER the expensive arm above had already been printed. Skip
    # it on either witness (the manifest's weight_y, or the column itself, which also
    # covers a manifest too old to record weight_y).
    weight_y = float(manifest.get("weight_y", 1.0))
    if weight_y == 0.0 or "probability" not in test_scored.columns:
        _cprint(f"[readout]   co-fit head arm SKIPPED (weight_y={weight_y:g}, "
              f"probability column "
              f"{'present' if 'probability' in test_scored.columns else 'absent'}): "
              "an unsupervised gate has no co-fit head to read out.", flush=True)
        return results
    if mode == "distributed":
        # No LR involved — `probability` IS the per-doc (C,) P(node) — so the
        # distributed variant is just the lean collector over that column.
        hp, hy, hm, ids_hd, _ = _collect_lean_proba(
            test_scored, C, score_col="probability")
    else:
        # WP-A4 residue (plan 2026-09-03 note): the co-fit head arm's OWN detection
        # pool must dedup to persons under multi-doc too — collect the doc keys and
        # thread them, exactly as the pc_topics_lr arm does. `with_doc_keys=True`
        # is opt-in so the arity change is explicit (see `_collect_head_proba`).
        hp, hy, hm, ids_hd = _collect_head_proba(
            test_scored, C, with_doc_keys=True)
    results["gated_pc_head"] = readout_from_proba(
        hp, hy, hm, C, recall_targets=recall_targets, fdr_targets=fdr_targets,
        min_count=min_count, doc_keys=ids_hd)
    _dump(results)
    print(format_arm_readout("gated_pc (co-fit head)", results["gated_pc_head"]),
          flush=True)
    return results


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    run_dir = resolve_run_dir(args.run_dir)
    # Every [readout]/[driver] line is teed to a durable readout_log.md IN THE
    # RUN DIR (the workspace disk survives cluster restarts; ~ and /tmp do not).
    # Appended, so a re-readout, an ablation sweep, and a ridge sweep all land
    # in one chronological file beside the results they produced. Separate from
    # the fit's driver_log.md, which `gated_pc_report --summary` digests.
    install_stdout_tee(run_dir / "readout_log.md")
    _cprint(f"[readout]   run dir: {run_dir}", flush=True)
    _argv = list(sys.argv[1:]) if argv is None else list(argv)
    _cprint(f"[readout]   argv: {' '.join(_argv)}", flush=True)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    C = int(manifest["C"])
    rt = [float(x) for x in args.recall_targets.split(",") if x]
    ft = [float(x) for x in args.fdr_targets.split(",") if x]
    min_count = (args.min_label_count if args.min_label_count is not None
                 else int(manifest.get("min_label_count", 20)))

    spec_over = dict(billing=args.billing, dag_source=args.dag_source,
                     mondo_version=args.mondo_version,
                     mondo_branch=args.mondo_branch,
                     min_positives=args.min_positives,
                     mondo_cache_dir=args.mondo_cache_dir,
                     label_set_path=args.label_set,
                     dag_collapse=(None if args.dag_collapse is None
                                   else args.dag_collapse == "on"),
                     preindex_closure=(None if args.preindex_closure is None
                                       else args.preindex_closure == "on"))
    # The fit records the cache root it used, so a recovery command need not
    # remember it; an explicit --cache-uri still wins.
    cm = manifest.get("corpus_manifest") or {}
    cache_uri = args.cache_uri or cm.get("cache_uri")

    with make_spark_session(app_name="gated-pc-readout") as spark:
        # Driver-disk telemetry, IN-BAND (see disk_telemetry's docstring): this
        # is the solve that keeps dying of ENOSPC from `sc.broadcast` with every
        # ADR 0047 destroy fix active, and the two local `diskwatch` loops that
        # were supposed to catch leak #2 both died with their cluster. Printing
        # from inside the driver puts the disk history in the persisted job log
        # and in driver_log.md, where it survives the machine.
        start_disk_telemetry(
            extra_dirs=[d for d in spark.sparkContext.getConf()
                        .get("spark.local.dir", "").split(",") if d],
            log=lambda msg: _cprint(f"[readout] {msg}", flush=True))

        from _case_finding_cache import try_load

        with _phase("reload cached bundle"):
            spec = None
            if args.bundle_path:
                base = args.bundle_path.rstrip("/")
                cache_uri, key = base.rsplit("/", 1)
            else:
                if not cache_uri:
                    _cprint("[readout] ERROR: pass --cache-uri or --bundle-path.",
                          flush=True)
                    return 1
                spec = corpus_spec_from_manifest(
                    manifest, doc_min_length=args.doc_min_length, **spec_over)
                if native_spec_mismatch(spec, manifest):
                    _cprint("[readout] ERROR: the saved fit is an exp-0110 NATIVE "
                          "Mondo run (corpus_manifest.dag_source=mondo_native), "
                          "but the rebuild spec resolved dag_source="
                          f"{spec['dag_source']!r} — that keys, and on the MISS "
                          "rebuilds, a different label space. Drop the override, "
                          "or pass --dag-source mondo_native.", flush=True)
                    return 2
                if mondo_spec_mismatch(spec, manifest):
                    _cprint("[readout] ERROR: the saved fit's label space is MONDO "
                          "ids, but the rebuild spec resolved dag_source="
                          f"{spec['dag_source']!r} — this manifest predates "
                          "corpus_manifest recording the Mondo build inputs, so "
                          "the defaults would key (and on the MISS that wrong key "
                          "guarantees, rebuild) the WRONG corpus. Pass the fit's "
                          "own values from the experiment doc's front matter, "
                          "e.g. for exp 0104:", flush=True)
                    _cprint("[readout]   --dag-source mondo --mondo-version "
                          "2026-06-02 --min-positives 100", flush=True)
                    return 2
                key = bundle_key_from_manifest(
                    manifest, doc_min_length=args.doc_min_length, **spec_over)
            bundle = try_load(spark, cache_uri, key)
            if bundle is None and (args.no_rebuild or spec is None):
                # --no-rebuild, or a --bundle-path whose dir does not hold a bundle
                # (there is no spec to rebuild FROM in that case).
                _cprint(f"[readout] ERROR: bundle cache MISS at {cache_uri}/{key}. "
                      "The assembly source may have changed since the fit, or a "
                      "key field differs. Pass --bundle-path at the exact cached "
                      "dir, or --doc-min-length if it was omitted"
                      + (" (drop --no-rebuild to re-assemble it instead)."
                         if args.no_rebuild else "."), flush=True)
                return 2
            if bundle is None:
                # LOAD-OR-BUILD: a cold cache (a fresh cluster, a cleared bucket, or
                # a fit that predates the corpus being cached at all) is not a reason
                # to lose a finished fit. Rebuild through the fit's own seam and
                # write it through, so this is paid once.
                _cprint("[readout] cache MISS — rebuilding bundle from manifest "
                      "params (~20 min at whole-Mondo)", flush=True)
                # billing/cdr are ENVIRONMENT, not corpus identity: the fit takes
                # them from the sourced .workspace_env, and manifests from before
                # they were recorded carry neither. Fall back to the same env vars
                # here — the Makefile target sources .workspace_env before this
                # tool runs — rather than handing spark-bigquery a null
                # parentProject, which dies as an opaque py4j NPE (exp 0104's
                # fresh-cluster recovery, 08-28).
                for field, env_var, flag in (("billing", "GOOGLE_CLOUD_PROJECT",
                                              "--billing"),
                                             ("cdr", "WORKSPACE_CDR", None)):
                    if spec.get(field) is None:
                        env_val = os.environ.get(env_var)
                        if env_val:
                            spec[field] = env_val
                            _cprint(f"[readout]   {field} not in the manifest; "
                                  f"using {env_var} from the environment (the "
                                  "same source the fit used)", flush=True)
                        else:
                            _cprint(f"[readout] ERROR: a rebuild needs {field}, "
                                  "but the manifest predates recording it and "
                                  f"{env_var} is unset"
                                  + (f" — pass {flag}." if flag else
                                     " — source .workspace_env (make setup)."),
                                  flush=True)
                            return 2
                _cprint(f"[readout]   corpus: dag_source={spec['dag_source']} "
                      f"extra_domains={spec['extra_domains']} "
                      f"index_mode={spec['index_mode']} min_n={spec['min_n']} "
                      f"mondo_branch={spec['mondo_branch'] or 'ALL'} "
                      f"min_positives={spec['min_positives']} "
                      f"dag_collapse={spec['dag_collapse']} "
                      f"preindex_closure={spec['preindex_closure']}", flush=True)
                bundle = rebuild_bundle(
                    spark, spec,
                    cache_uri=(cache_uri if args.cache_write == "on" else None))
                _cprint(f"[readout]   bundle REBUILT"
                      + (f" and written to {cache_uri}/{key} (the next readout of "
                         "this run is a HIT)" if args.cache_write == "on"
                         else " (--cache-write off: not persisted)")
                      + f"; C={C}", flush=True)
            else:
                _cprint(f"[readout]   bundle loaded ({cache_uri}/{key}); C={C}",
                      flush=True)
            from gated_pc_cloud import write_bundle_meta
            _bmp = write_bundle_meta(run_dir, bundle)
            _cprint(f"[readout]   wrote bundle meta -> {_bmp}", flush=True)
            _ncm = getattr(bundle, "native_code_map", None)
            if _ncm and (args.bundle_only or not (run_dir / "code_map.tsv").exists()):
                # WP-C': a fit from before the code map rode the bundle gets it
                # on its first re-readout, so `--profile-census --own-codes`
                # works on 0135 without a refit.
                from gated_pc_cloud import write_code_map
                _cmp = write_code_map(run_dir, _ncm)
                _cprint(f"[readout]   wrote native code map ({len(_ncm)} pairs) -> "
                      f"{_cmp}", flush=True)

        if args.differential:
            with _phase("differential with reasons (framing 1)"):
                run_differential(
                    bundle.test_df, bundle.train_df, run_dir, args.differential,
                    adhoc=args.acq_adhoc, heads_tag=args.acq_heads,
                    max_patients=args.diff_max_patients,
                    max_train=args.diff_max_train, tag=args.acq_tag,
                    next_question=args.diff_next_question,
                    nq_hide=args.diff_nq_hide)
            return 0

        if args.acquisition:
            with _phase("simulated acquisition (WP-D part 3)"):
                run_acquisition(
                    bundle.test_df, run_dir, args.acquisition, adhoc=args.acq_adhoc,
                    prior=args.acq_prior,
                    domains=tuple(x.strip() for x in args.acq_domains.split(",")),
                    pool=args.acq_pool, steps=args.acq_steps,
                    max_patients=args.acq_max_patients,
                    include_strata=args.acq_include_strata, tag=args.acq_tag,
                    length=args.acq_length, engine=args.acq_engine,
                    heads_tag=args.acq_heads,
                    tau=(args.acq_tau if args.acq_tau == "auto" else float(args.acq_tau)))
            return 0

        if args.bundle_only:
            _cprint("[readout]   --bundle-only: bundle meta + code map written; "
                  "stopping before the transform", flush=True)
            return 0

        with _phase("reconstruct model + transform"):
            # The theta collect (driver mode) now happens inside run_readout, so the
            # distributed mode can skip it entirely rather than paying for it here.
            model = reconstruct_model(run_dir, manifest)
            # DRIFT GATE, before a single number is computed: the corpus in hand
            # (cached or rebuilt) must be the one this λ/head was fit against.
            drift = bundle_drift_report(
                bundle, manifest, lambda_vocab_sizes(model.result.global_params))
            if drift:
                _cprint("[readout] ERROR: the corpus does not match the saved fit — "
                      "it has DRIFTED since the fit (a data refresh, or an "
                      "assembly-source change). Refusing to score:", flush=True)
                for line in drift:
                    _cprint(f"[readout]     - {line}", flush=True)
                _cprint("[readout]   Pass --bundle-path at the bundle this run was "
                      "actually fit against, or re-fit.", flush=True)
                return 3
            train_scored = model.transform(bundle.train_df).cache()
            test_scored = model.transform(bundle.test_df).cache()

        # WP-B (spec 2026-10-07 §D1/§D4): a label-set fit is FLAT, and the
        # bundle's closure mask on a flat forest is a full mask (insight 0096:
        # 70.7M cells). The read gets the hierarchy the fit was denied: the
        # Mondo readout DAG over the label nodes + grouping ancestors, applied to
        # the SCORED frames as column arithmetic. Nothing in the bundle, the
        # cache or a hashed module moves; the manifest's C stays the fit's.
        _rdag = None
        _C_ro = C
        _parent_ro = getattr(bundle, "parent_int", None)
        _cm = manifest.get("corpus_manifest") or {}
        from readout_dag import resolve_readout_hierarchy
        if resolve_readout_hierarchy(args.readout_hierarchy, manifest):
            with _phase("readout DAG (Mondo hierarchy over the flat fit)"):
                from readout_dag import (
                    build_readout_dag, format_readout_dag_report,
                    load_mondo_parent_adj, widen_labels_to_readout,
                    write_readout_dag)
                if args.readout_feature_mask != "all":
                    raise SystemExit("[readout] --readout-feature-mask ablations "
                                     "are defined on the FIT's blocks; not "
                                     "available with --readout-hierarchy mondo")
                _mv = args.mondo_version or _cm.get("mondo_version")
                _mcd = args.mondo_cache_dir or _cm.get("mondo_cache_dir") or "data/mondo"
                if not _mv:
                    raise SystemExit("[readout] --readout-hierarchy mondo needs the "
                                     "Mondo version (manifest corpus_manifest."
                                     "mondo_version or --mondo-version)")
                _padj, _mnames = load_mondo_parent_adj(_mv, _mcd)
                _rdag = build_readout_dag(bundle.int2cid, _padj, names=_mnames)
                _cprint(format_readout_dag_report(_rdag), flush=True)
                _rp = write_readout_dag(run_dir, _rdag)
                _cprint(f"[readout]   wrote {_rp}", flush=True)
                _mm = str(_cm.get("label_mask_mode") or
                          manifest.get("label_mask_mode") or "full")
                _tr_w = widen_labels_to_readout(train_scored, _rdag, mask_mode=_mm).cache()
                _te_w = widen_labels_to_readout(test_scored, _rdag, mask_mode=_mm).cache()
                # Materialize the widened frames off the cached transforms, then
                # drop the fit-width copies: the solver's repeated passes read
                # the widened train frame, never these.
                # Conditional-from-heads scores the test split only: never
                # materialize the (much larger) train transform for it.
                _n_tr = ("not materialized"
                         if (args.conditional_from_heads or args.lookalikes
                             or args.prevalence_check)
                         else _tr_w.count())
                _n_te = _te_w.count()
                train_scored.unpersist(); test_scored.unpersist()
                train_scored, test_scored = _tr_w, _te_w
                _C_ro, _parent_ro = int(_rdag.C), _rdag.parent_int
                _cprint(f"[readout]   frames widened to the readout node space: "
                      f"C={_C_ro} (fit C={C}), mask mode={_mm}, train docs={_n_tr} "
                      f"test docs={_n_te}", flush=True)

        if args.prevalence_check:
            if _rdag is None:
                raise SystemExit("[readout] --prevalence-check needs the Mondo readout DAG")
            with _phase("prevalence check (insight 0100)"):
                run_prevalence_check(test_scored, run_dir, C=_C_ro,
                                     parent_int=_parent_ro, heads_tag=args.acq_heads,
                                     tag=args.acq_tag)
            return 0

        if args.lookalikes:
            if _rdag is None:
                raise SystemExit("[readout] --lookalikes needs the Mondo readout DAG")
            with _phase("undiagnosed look-alikes (framing 3)"):
                run_lookalikes(test_scored, run_dir, args.lookalikes, C=_C_ro,
                               parent_int=_parent_ro, adhoc=args.acq_adhoc,
                               heads_tag=args.acq_heads, top_n=args.look_top,
                               tag=args.acq_tag)
            return 0

        if args.conditional_from_heads:
            if _rdag is None:
                raise SystemExit("[readout] --conditional-from-heads needs the Mondo "
                                 "readout DAG (a label-set fit, or "
                                 "--readout-hierarchy mondo)")
            with _phase("conditional diagnosis from saved heads"):
                _nb = dict(getattr(bundle, "name_by_id", {}) or {})
                _nb.update(_rdag.name_by_id)
                run_conditional_from_heads(
                    test_scored, run_dir, args.conditional_from_heads, C=_C_ro,
                    parent_int=_parent_ro, int2cid=_rdag.int2cid, name_by_id=_nb,
                    min_count=min_count)
            return 0

        _test_unmasked = test_scored
        if args.readout_incident:
            if _rdag is None:
                raise SystemExit("[readout] --readout-incident needs the Mondo readout DAG")
            with _phase("incident label rule: mask prior-evidence cells (exp 0144)"):
                import prevalence_check as _pv
                import voi_tables as _vt
                _bm, _cmp = run_dir / "bundle_meta.json", run_dir / "code_map.tsv"
                if not _cmp.exists():
                    raise SystemExit("[readout] --readout-incident needs "
                                     "<run>/code_map.tsv (the attestation map)")
                _ctx = _vt.load_voi_context(
                    run_dir, bundle_meta_path=str(_bm) if _bm.exists() else None,
                    own_codes_path=str(_cmp))
                _v2n = _pv.vocab_to_nodes(_ctx.own_idx, _rdag.pos_sources)
                _fc = ("features_0" if "features_0" in train_scored.columns
                       else "features")
                _tr_i = mask_prior_cells(train_scored, _v2n, feat_col=_fc).cache()
                _te_i = mask_prior_cells(test_scored, _v2n, feat_col=_fc).cache()
                _cprint(f"[readout]   incident masking: {len(_v2n):,} condition codes "
                        f"attest readout nodes; train docs={_tr_i.count()} test "
                        f"docs={_te_i.count()} (masked frames materialized)", flush=True)
                train_scored.unpersist()
                train_scored, test_scored = _tr_i, _te_i

        with _phase("score"):
            # E2/WP4: the incident arm's eligibility comes from the BUNDLE (spec
            # R2.3 — a corpus property; nothing this run computes may enter it),
            # and its availability is decided by E1's WITNESS, not by a column
            # probe. `bundle_preindex_witness` answers None for every corpus built
            # before E1 or without the flag, and `run_readout` then prints a skip
            # line instead of raising. Deliberately NOT `require_preindex_closure`:
            # that one RAISES, which is right for the census (whose whole job is
            # the eligibility census) and wrong here (whose job is the prevalent
            # readout, with the incident block as an addition).
            from preindex_closure import bundle_preindex_witness
            _pw = bundle_preindex_witness(bundle)
            _elig_col = str(_pw.get("col_name")) if _pw else None
            if _elig_col and _rdag is not None:
                _cprint("[readout]   INCIDENT arm (E2) SKIPPED under "
                      "--readout-hierarchy mondo: the pre-index closure column is "
                      "in the FIT's node space (widening it is a WP-B follow-up).",
                      flush=True)
                _elig_col = None
            if _elig_col:
                _cprint(f"[readout]   incident eligibility column: {_elig_col!r} "
                      f"({_pw.get('version')})", flush=True)
            _fmask, _mtag = None, None
            if args.readout_theta_topm is not None:
                # A CLI top-m override is a pricing experiment, not the record:
                # tag its outputs so they never overwrite results_readout.json
                # (or a mask run at the manifest's top-m).
                _mtag = f"topm{int(args.readout_theta_topm)}"
            if args.readout_theta_mass is not None and float(args.readout_theta_mass) > 0:
                if args.readout_theta_topm is not None:
                    raise SystemExit("[readout] --readout-theta-mass and "
                                     "--readout-theta-topm are mutually exclusive")
                _mtag = f"mass{int(round(float(args.readout_theta_mass) * 100))}"
            if args.readout_feature_mask != "all":
                _fmask = build_readout_feature_mask(
                    args.readout_feature_mask, C, int(manifest["K"]),
                    int(manifest["n_bg"]), int(manifest["tpn"]),
                    parent_int=getattr(bundle, "parent_int", None))
                _mtag = (args.readout_feature_mask.replace("-", "_")
                         + (f"_{_mtag}" if _mtag else ""))
                _cprint(f"[readout]   FEATURE ABLATION mode={args.readout_feature_mask}: "
                      f"results -> results_readout_{_mtag}.json, heads -> "
                      f"readout_heads_gated_pc_{_mtag}.npz (record untouched)",
                      flush=True)
            if args.readout_hierarchy == "flat" and _cm.get("label_set"):
                # The hierarchical read IS a label-set fit's record; an explicit
                # flat read of one is the ablation and must not overwrite it.
                _mtag = "flat" + (f"_{_mtag}" if _mtag else "")
                _cprint(f"[readout]   FLAT read of a label-set fit: results -> "
                      f"results_readout_{_mtag}.json (record untouched)", flush=True)
            if args.readout_incident:
                _mtag = "incident" + (f"_{_mtag}" if _mtag else "")
            if args.readout_stacked:
                # Its own tag: the arm re-solves the SAME heads plus a changed
                # root head, and the record's results/heads files stay the
                # prevalent arm's.
                _mtag = "stacked" + (f"_{_mtag}" if _mtag else "")
                _cprint(f"[readout]   STACKED arm requested: results -> "
                      f"results_readout_{_mtag}.json, heads -> "
                      f"readout_heads_gated_pc_{_mtag}.npz (record untouched)",
                      flush=True)
            _rname = ("results_readout.json" if not _mtag
                      else f"results_readout_{_mtag}.json")
            run_readout(train_scored, test_scored, manifest, recall_targets=rt,
                        fdr_targets=ft, min_count=min_count,
                        readout_mode=args.readout_mode,
                        ab_check=args.readout_ab_check, out_dir=run_dir,
                        theta_topm=args.readout_theta_topm,
                        readout_max_iter=args.readout_max_iter,
                        elig_col=_elig_col, eval_path=args.eval_path,
                        readout_l2=args.readout_l2, feature_mask=_fmask,
                        mask_tag=_mtag, stacked=bool(args.readout_stacked),
                        parent_int=_parent_ro,
                        theta_mass=args.readout_theta_mass, C=_C_ro)
            _cprint(f"[readout]   arm results written to {run_dir / _rname}",
                  flush=True)
            if args.readout_incident and args.readout_stacked:
                # The comparison that matters: these heads vs the record's on the
                # SAME unmasked test split under the same prior rule (0143: the
                # record's stacked incident macro AUC 0.702, flat 0.616).
                with _phase("prevalence check on the incident heads (exp 0144)"):
                    run_prevalence_check(_test_unmasked, run_dir, C=_C_ro,
                                         parent_int=_parent_ro, heads_tag=_mtag,
                                         tag=_mtag)
                _test_unmasked.unpersist()
            train_scored.unpersist(); test_scored.unpersist()

            # Echo the other arms' stored summary (only gated_pc can be re-scored).
            # `results` is legitimately absent-or-null on a FIT-ONLY manifest:
            # `gated_pc_cloud` writes the npz + manifest as soon as the fit lands
            # (`partial="fit-only"`) so a readout death cannot cost the fit, which
            # is precisely the run this tool exists to rescue — so say what is
            # missing rather than echoing nothing, and never index into it.
            if manifest.get("partial"):
                _cprint(f"[readout]   manifest marked partial="
                      f"{manifest['partial']!r}: the fit landed but its own "
                      "readout did not, so there are no stored arm results to "
                      "echo (the numbers above are this re-readout's).",
                      flush=True)
            for name, res in (manifest.get("results") or {}).items():
                if name in ("gated_pc", "gated_pc_head"):
                    continue
                rk = res.get("ranking", res)   # tolerate old flat-macro manifests
                auc = rk.get("auc"); ap = rk.get("ap")
                _cprint(f"[readout]   (manifest) {name}: AUC="
                      f"{'n/a' if auc is None else f'{auc:.4f}'} "
                      f"AP={'n/a' if ap is None else f'{ap:.4f}'}", flush=True)
    return 0


if __name__ == "__main__":
    # A client-mode spark-submit driver does not end when main() returns: the
    # SparkSubmit JVM that owns this process waits on a non-daemon Dataproc
    # metrics-publisher thread stuck in a socket write (jstack, 2026-09-12),
    # and a CHAINED sweep waits on the JVM. `hard_exit` flushes the durable
    # log, takes the JVM down through the gateway, then os._exit's — see its
    # docstring in _driver_common for the full diagnosis.
    try:
        _rc = main()
    except SystemExit as _e:                      # argparse errors etc.
        _rc = _e.code if isinstance(_e.code, int) else 1
    except BaseException:                         # a crash must not wedge the sweep either
        import traceback
        traceback.print_exc()
        _rc = 1
    hard_exit(_rc)

"""The clinician framings over a finished DAG-gated fit (spec 2026-10-13 clinician
framings; ADR 0051 keeps them here, in `analysis/cloud`, beside the drivers).

  * framing 1 — DIFFERENTIAL WITH REASONS: `run_differential` (engines E2 stacked
    heads, E3 multiclass head on θ, E5 code head, E6 codes + θ; `differential.py`),
    incident test patients, cross-fit, the survey table (`differential_survey`);
  * framing 2 — WORKUP (exp 0167; spec 2026-10-28): every fact recorded between the
    index and the first diagnosis, ranked by EIG under the patient's belief
    (`workup_frame`, `_workup_block`; the sidecar is `workup_sidecar.py`);
  * framing 3 — NEW DIAGNOSES: `run_prevalence_check`, per-node AUC with the
    patients diagnosed before the index set aside (`prevalence_check.py`).

`dag_gated_readout.py` dispatches here (`--differential`, `--prevalence-check`). The
prevalence check's block functions run on the executors: this module and
`prevalence_check.py` ride the readout's `--py-files`."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dag_gated_fit import _cprint


def most_specific_members(members, parent_int):
    """Members that are not an ancestor of another member. On a NESTED label space
    (exp 0152's whole-Mondo fit) a group's label descendants include intermediate
    terms whose labels are the closure of their children's, so "which member" needs the
    most specific ones; a patient coded only at an intermediate level has no member
    truth and is not scored. `parent_int` is the readout DAG (int or str keys)."""
    mem = [int(m) for m in members]
    anc = set()
    for m in mem:
        stack, seen = list(parent_int.get(m, parent_int.get(str(m), []))), set()
        while stack:
            a = int(stack.pop())
            if a in seen:
                continue
            seen.add(a)
            stack.extend(parent_int.get(a, parent_int.get(str(a), [])))
        anc |= seen
    return [m for m in mem if m not in anc]


def top_members(members, parent_int, group=None):
    """Members with no proper ancestor among the other members, the group node itself
    excluded: the group's TOP-LEVEL label terms. On a nested label space a term's label
    is the closure of its descendants', so a top-level member rolls up its subtypes (a
    patient coded at chronic hepatitis C is a hepatitis C patient). This is the
    natural differential; `most_specific_members` instead scores only the leaves and
    drops every patient coded at an intermediate term (exp 0152 read 4: viral
    hepatitis lost hepatitis C, its bulk). `parent_int` is the readout DAG."""
    mem = [int(m) for m in members if group is None or int(m) != int(group)]
    ms = set(mem)
    out = []
    for m in mem:
        stack, seen, hit = list(parent_int.get(m, parent_int.get(str(m), []))), set(), False
        while stack and not hit:
            a = int(stack.pop())
            if a in seen:
                continue
            seen.add(a)
            hit = a in ms
            stack.extend(parent_int.get(a, parent_int.get(str(a), [])))
        if not hit:
            out.append(m)
    return out


def collect_set_patients(test_df, members, n_domains, *, max_patients=5000, seed=0,
                         label_col="label", length="distinct", extra_cols=()):
    """Test-split persons positive on any of `members` (fit-width label indices):
    `[(truth positions into members, {d: {vocab idx: count}}, {d: length})]`.

    Filter and label slice are column arithmetic over a JSON literal (the
    `widen_labels_to_readout` pattern: nothing array-shaped rides a closure); the
    collect carries only the members' label cells and the sparse feature vectors.
    Capped at `max_patients` by a seeded Spark sample. `length` is the record length
    the presence model gets per domain: `distinct` codes (default) or `tokens`.

    `extra_cols`: columns of `test_df` carried through per row as a 4th element, a
    `{name: value}` dict (exp 0167: the workup window's codes and length)."""
    import json as _json
    from pyspark.sql import functions as F
    lit = F.from_json(F.lit(_json.dumps([int(m) for m in members])), "array<int>")
    lab = F.col(label_col)
    fcols = ([f"features_{d}" for d in range(int(n_domains))]
             if f"features_0" in test_df.columns else ["features"])
    keep = F.exists(lit, lambda i: lab[i] > 0)
    df = (test_df.where(keep)
          .select(F.transform(lit, lambda i: lab[i]).alias("__y"), *fcols, *extra_cols))
    n = df.count()
    if n > int(max_patients):
        df = df.sample(False, min(1.0, 1.2 * int(max_patients) / n), seed=int(seed))
    out = []
    for row in df.collect()[:int(max_patients)]:
        truth = sorted(j for j, v in enumerate(row["__y"]) if v and v > 0)
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
        if extra_cols:
            out.append((truth, bags, lens, {c: row[c] for c in extra_cols}))
        else:
            out.append((truth, bags, lens))
    return out, n


def workup_frame(first, horizon, workup, node_cids, target_cids, parent_adj, *,
                 label_window_days=365, inclusive=False):
    """Exp 0167: `(person_id, __wk: array<struct<d,t>>, __wdays)` for every person whose
    FIRST diagnosis of a group member (any of `target_cids` or a Mondo term under
    one, from the conversion sidecar) falls in the label year `[index, index + W)`:
    the `(dom, token)` pairs the workup sidecar has first recorded in
    `[index, t_dx)` (`inclusive`: `[index, t_dx]`), and the window's length in days.
    Persons diagnosed before the index (prevalent) or never get no row: they have no
    workup window."""
    import prevalence_check as pv
    from pyspark.sql import functions as F
    tgt = {i + 1: int(c) for i, c in enumerate(target_cids)}
    hit = pv.sidecar_nodes_to_readout(node_cids, tgt, parent_adj, root_cid=-10**9)
    spark = first.sparkSession
    if not hit:
        return spark.createDataFrame(
            [], "person_id long, __wk array<struct<d:int,t:long>>, __wdays int")
    ndf = spark.createDataFrame([(int(x),) for x in hit], "node_cid long")
    hz = horizon.select("person_id", "index_date")
    dx = (first.join(F.broadcast(ndf), "node_cid").join(hz, "person_id")
          .where(F.col("first_attested_date") >= F.col("index_date"))
          .where(F.col("first_attested_date")
                 < F.date_add(F.col("index_date"), int(label_window_days)))
          .groupBy("person_id", "index_date")
          .agg(F.min("first_attested_date").alias("t_dx")))
    before = (F.col("first_date") <= F.col("t_dx")) if inclusive else (
        F.col("first_date") < F.col("t_dx"))
    wk = (workup.join(dx.select("person_id", "t_dx"), "person_id").where(before)
          .groupBy("person_id")
          .agg(F.collect_list(F.struct(F.col("dom").cast("int").alias("d"),
                                       F.col("token").alias("t"))).alias("__wk")))
    return (dx.withColumn("__wdays", F.datediff("t_dx", "index_date"))
            .join(wk, "person_id", "left")
            .select("person_id",
                    F.coalesce("__wk", F.array().cast("array<struct<d:int,t:long>>"))
                    .alias("__wk"), "__wdays"))


def _test_prior_pairs(test_df, run_dir, sidecar_uri=None):
    """`(person_id, node_cid)` for test persons: every sidecar Mondo node first attested
    BEFORE the person's index (index_horizon). Exp 0151."""
    from pyspark.sql import functions as F
    from conversion_sidecar import try_load_index_horizon, try_load_sidecar
    wit = json.loads((Path(run_dir) / "conversion_sidecar.json").read_text())
    uri = sidecar_uri or wit.get("sidecar_uri")
    spark = test_df.sparkSession
    first = try_load_sidecar(spark, uri, wit["key"])
    hz = try_load_index_horizon(spark, uri, wit["key"])
    if first is None or hz is None:
        raise SystemExit(f"[differential] sidecar MISS at {uri}/{wit['key']} — rebuild "
                         "it on this cluster (make build-conversion-sidecar ID=N)")
    persons = test_df.select("person_id").distinct()
    return (first.join(hz.select("person_id", "index_date").join(persons, "person_id"),
                       "person_id")
            .where(F.col("first_attested_date") < F.col("index_date"))
            .select("person_id", "node_cid").distinct())


def _incident_test_frame(test_df, prior_pairs, prior_nodes, target_cids, parent_adj):
    """`test_df` minus the persons with a pre-index attestation of any target Mondo
    term (a group's members and the group node) or anything under one — the INCIDENT
    rule of exp 0151 (insight 0103): their pre-index record is pre-diagnosis."""
    import prevalence_check as pv
    from pyspark.sql import functions as F
    tgt = {i + 1: int(c) for i, c in enumerate(target_cids)}
    hit = pv.sidecar_nodes_to_readout(prior_nodes, tgt, parent_adj, root_cid=-10**9)
    if not hit:
        return test_df
    ndf = test_df.sparkSession.createDataFrame([(int(n),) for n in hit], "node_cid long")
    # The excluded persons are collected and BROADCAST back, so the (large) test frame
    # is filtered in place, never shuffled (0151 launch 1: ~440 s per group from a
    # sort-merge anti-join that shuffled every feature vector twice).
    prev = [int(r[0]) for r in prior_pairs.join(F.broadcast(ndf), "node_cid")
            .select("person_id").distinct().collect()]
    if not prev:
        return test_df
    pdf = test_df.sparkSession.createDataFrame([(x,) for x in prev], "person_id long")
    return test_df.join(F.broadcast(pdf), "person_id", "left_anti")


def _workup_setup(spark, run_dir, ctx, sidecar_uri=None):
    """Exp 0167: the frames and maps the workup framing reads once per survey — the
    conversion sidecar (first attestations, index horizon), the workup sidecar, the
    corpus vocabularies (so answers stay in the space the engines read) and the
    measurement domain's index."""
    from conversion_sidecar import try_load_index_horizon, try_load_sidecar
    from workup_sidecar import try_load_workup
    run_dir = Path(run_dir)
    for f, how in (("conversion_sidecar.json", "build-conversion-sidecar"),
                   ("workup_sidecar.json", "build-workup-sidecar")):
        if not (run_dir / f).exists():
            raise SystemExit(f"[differential] --diff-workup needs <run>/{f} "
                             f"(make {how} ID=N)")
    cw = json.loads((run_dir / "conversion_sidecar.json").read_text())
    ww = json.loads((run_dir / "workup_sidecar.json").read_text())
    uri = sidecar_uri or cw.get("sidecar_uri")
    first = try_load_sidecar(spark, uri, cw["key"])
    hz = try_load_index_horizon(spark, uri, cw["key"])
    wk = try_load_workup(spark, ww["sidecar_uri"], ww["key"])
    if first is None or hz is None or wk is None:
        raise SystemExit("[differential] a sidecar is MISSING on this cluster — rebuild "
                         "(make build-conversion-sidecar ID=N, then "
                         "build-workup-sidecar ID=N)")
    if ww.get("conversion_key") != cw["key"]:
        raise SystemExit("[differential] the workup sidecar was built against a "
                         "different conversion sidecar — rebuild it (--force)")
    first = first.cache()
    nodes = [int(r[0]) for r in first.select("node_cid").distinct().collect()]
    if not ctx.inv_maps:
        raise SystemExit("[differential] --diff-workup needs the bundle's vocab maps")
    tok2idx = [{int(t): int(i) for i, t in m.items()} for m in ctx.inv_maps]
    meas = (ctx.dom_names.index("measurement") if "measurement" in ctx.dom_names
            else None)
    return {"first": first, "hz": hz, "workup": wk, "nodes": nodes,
            "n_nodes": len(nodes), "window": int(ww.get("label_window_days") or 365),
            "tok2idx": tok2idx, "vocab_sets": {d: set(m) for d, m in enumerate(tok2idx)},
            "meas_dom": meas}


def _workup_block(dfm, ctx, wk_ctx, tr, te, ytr, members, excl, halves, truths,
                  q0_by_engine, pri, *, drop_done=False, new_only=True,
                  length_edges=()):
    """Exp 0167 for one group: answers, candidates, likelihoods, then per engine the
    temper on each cross-fit fitting half and R1-R4 on its evaluation half (all
    patients once), plus coverage and the group's top facts. None when too few
    windows exist to score.

    `new_only` (run 2, default): a fact already in the patient's pre-index record is
    not a candidate for that patient, and the likelihoods count only the training
    patients for whom it was new. `length_edges` (days, e.g. (30, 120)): likelihoods
    per window-length bin, shrunk to the pooled one, each patient read with its own
    window's bin."""
    tok2idx = wk_ctx["tok2idx"]
    md = wk_ctx["meas_dom"]
    excl_tok = {d: {ctx.inv_maps[d][int(i)] for i in ex if int(i) in ctx.inv_maps[d]}
                for d, ex in excl.items()}

    def _ans(extra):
        if extra is None or extra.get("__wdays") is None:
            return None
        return dfm.workup_answers([(r["d"], r["t"]) for r in (extra["__wk"] or [])],
                                  meas_dom=md, vocab_sets=wk_ctx["vocab_sets"],
                                  excl=excl_tok, drop_done=drop_done)

    def _known(row):
        return (dfm.known_facts(row[1], ctx.inv_maps, meas_dom=md) if new_only
                else set())
    tr_ans = [_ans(r[5]) for r in tr]
    keep_tr = [i for i, a in enumerate(tr_ans) if a is not None]
    te_ans = [_ans(r[5]) for r in te]
    keep_te = [i for i, a in enumerate(te_ans) if a is not None]
    out = {"n_train_windows": len(keep_tr), "n_test_windows": len(keep_te),
           "new_only": bool(new_only), "length_edges": list(length_edges)}
    if len(keep_te) < dfm.EGRESS_FLOOR or len(keep_tr) < dfm.EGRESS_FLOOR:
        out["skipped"] = "fewer than 20 train or test patients with a window"
        return out
    tr_a = [tr_ans[i] for i in keep_tr]
    tr_k = [_known(tr[i]) for i in keep_tr]
    Y = dfm.label_matrix([ytr[i] for i in keep_tr], len(members))
    cands = dfm.workup_candidates(tr_a, known=tr_k)
    out["n_candidates"] = len(cands)
    if not cands:
        out["skipped"] = "no fact recorded (new) for 20 training patients"
        return out
    L = dfm.answer_likelihoods(tr_a, Y, cands, known=tr_k)
    days = {i: te[i][5]["__wdays"] for i in keep_te}
    if length_edges:
        tr_bin = np.array([dfm.length_bin(tr[i][5]["__wdays"], length_edges)
                           for i in keep_tr])
        Lb = []
        for b in range(len(length_edges) + 1):
            sel = np.flatnonzero(tr_bin == b)
            Lb.append(dfm.answer_likelihoods([tr_a[j] for j in sel], Y[sel], cands,
                                             known=[tr_k[j] for j in sel], prior=L)
                      if sel.size else L)
        L_of = {i: Lb[dfm.length_bin(days[i], length_edges)] for i in keep_te}
        out["train_windows_per_bin"] = [int((tr_bin == b).sum())
                                        for b in range(len(length_edges) + 1)]
    else:
        L_of = {i: L for i in keep_te}
    A = {i: dfm.answer_vector(te_ans[i], cands) for i in keep_te}
    E = {i: dfm.eligible_vector(_known(te[i]), cands) for i in keep_te}
    out["coverage"] = dfm.workup_coverage([days[i] for i in keep_te],
                                          [A[i] * E[i] for i in keep_te])
    out["coverage"]["eligible_median"] = float(np.median([E[i].sum() for i in keep_te]))
    kt = set(keep_te)
    for name, q0 in q0_by_engine.items():
        if q0 is None:
            continue
        ev, taus = [], []
        for fi, ei in halves:
            f = [i for i in fi if i in kt]
            tau = (dfm.fit_workup_tau([q0[i] for i in f], [truths[i][0] for i in f],
                                      [A[i] for i in f], [L_of[i] for i in f],
                                      eligible=[E[i] for i in f]) if f else 1.0)
            for i in ei:
                if i in kt:
                    ev.append(i)
                    taus.append(tau)
        if len(ev) < dfm.EGRESS_FLOOR:
            continue
        out[name] = dfm.evaluate_workup(
            [q0[i] for i in ev], [truths[i][0] for i in ev], [A[i] for i in ev],
            [L_of[i] for i in ev], pri, np.asarray(taus),
            eligible=[E[i] for i in ev], L_pop=L)
    # Run-2 diagnostic (the sign of R1): likelihood transfer, gain by answer type,
    # R1 by window activity, under the first engine's beliefs.
    name0 = next((k for k, v in q0_by_engine.items() if v is not None), None)
    if name0 is not None and out.get(name0):
        ev0 = [i for i in keep_te]
        tau_d = float(np.mean(out[name0]["tau"]))
        tr_ix = list(range(len(keep_tr)))
        tr_t = [int(np.argmax(Y[j])) for j in tr_ix]
        out["diagnostics"] = dfm.workup_diagnostics(
            [q0_by_engine[name0][i] for i in ev0], [truths[i][0] for i in ev0],
            [A[i] for i in ev0], [L_of[i] for i in ev0], tau_d,
            eligible=[E[i] for i in ev0],
            activity=[len(te[i][5]["__wk"] or []) for i in ev0],
            train=([dfm.answer_vector(tr_a[j], cands) for j in tr_ix], tr_t,
                   [dfm.eligible_vector(tr_k[j], cands) for j in tr_ix]))
        out["diagnostics"]["engine"] = name0
    # The group's most valuable facts under the first engine's beliefs: mean EIG over
    # the scored patients for whom each fact is new; every listed fact has >= 20
    # training patients answering it.
    name0 = next((k for k, v in q0_by_engine.items() if v is not None), None)
    if name0 is not None:
        tau0 = float(np.mean(out.get(name0, {}).get("tau") or [1.0]))
        tot, cnt = np.zeros(len(cands)), np.zeros(len(cands))
        for i in keep_te:
            post, pa = dfm.workup_posteriors(q0_by_engine[name0][i], L_of[i], tau0)
            tot += dfm.workup_eig(q0_by_engine[name0][i], post, pa) * E[i]
            cnt += E[i]
        mean_eig = np.divide(tot, cnt, out=np.zeros_like(tot), where=cnt > 0)
        names = []
        for j in np.argsort(-mean_eig)[:5]:
            k = cands[int(j)][0]
            if k[0] == "c":
                names.append(ctx.code_name(k[1], tok2idx[k[1]].get(k[2], -1)))
            else:
                t = next((t for t in tok2idx[md] if t // 100 == k[1]), None)
                nm = ctx.code_name(md, tok2idx[md][t]) if t is not None else str(k[1])
                names.append(f"test {nm.split(' [')[0]}")
        out["top_facts"] = names
    return out


def _format_workup(w, window):
    """The group's workup block (exp 0167) as markdown lines."""
    L_ = [f"### Workup (framing 2, exp 0167; window [index, first diagnosis"
          f"{']' if window == 'inclusive' else ')'}"
          f"{'; new facts only' if w.get('new_only') else ''}"
          f"{'; length bins ' + ','.join(str(x) for x in w['length_edges']) + ' days' if w.get('length_edges') else ''})"]
    if w.get("skipped"):
        return L_ + [f"({w['skipped']}: train {w['n_train_windows']}, test "
                     f"{w['n_test_windows']})", ""]
    c = w["coverage"]
    dq = " / ".join(f"{x:.0f}" for x in c["window_days_quartiles"])
    L_ += [f"{w['n_test_windows']} test / {w['n_train_windows']} train patients with a "
           f"window · {w['n_candidates']} candidate facts · window days {dq} · "
           f"{100 * c['share_with_any']:.0f}% with any non-default answer (median "
           f"{c['median_non_default']:.0f}; {c.get('eligible_median', float('nan')):.0f} "
           f"candidates new to the median patient)", "",
           "| q0 | n | tau | per-patient R1 Spearman (q1/med/q3) | gain top-1 | top-3 | random | "
           "population | oracle | R3 slope | top-1 changes call | toward truth |",
           "|---|--:|---|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for e in ("E6", "E3"):
        r = w.get(e)
        if not r:
            continue
        r1 = " / ".join(f"{x:+.2f}" for x in r["R1_spearman_quartiles"])
        g = r["R2"]
        L_.append(f"| {e} | {r['n']} | {', '.join(f'{t:.2f}' for t in r['tau'])} | {r1} | "
                  f"{g['top1']:+.4f} | {g['top3']:+.4f} | {g['random']:+.4f} | "
                  f"{g['population_top1']:+.4f} | {g['oracle']:+.4f} | "
                  f"{r['R3']['slope']:.2f} | {r['R4']['top1_changes_call']:.2f} | "
                  f"{r['R4']['top1_moves_toward_truth']:.2f} |")
    for e in ("E6", "E3"):
        f = (w.get(e) or {}).get("facts") or {}
        if "spearman_mean_eig_vs_mean_gain" in f:
            L_.append(f"fact-level ranking ({e}, {f['n_facts']} facts new to >= 20 "
                      f"patients): Spearman mean EIG vs mean realized gain "
                      f"{f['spearman_mean_eig_vs_mean_gain']:+.2f} · top-5 facts' mean "
                      f"gain {f['top5_mean_gain']:+.4f} vs all {f['all_mean_gain']:+.4f}"
                      f" · slope {f['slope']:.2f}")
    if w.get("top_facts"):
        L_ += ["", "most informative facts (mean EIG, E6): " + " · ".join(w["top_facts"])]
    d = w.get("diagnostics")
    if d:
        L_ += ["", f"diagnostic ({d['engine']}): log P(true answer | true member) "
               f"test {d['ll_test']:.4f} vs train {d.get('ll_train', float('nan')):.4f} · "
               f"non-default rate predicted {d['pred_nondefault']:.3f} vs observed "
               f"{d['obs_nondefault']:.3f} · default-answer pairs "
               f"{d['default_pairs']['n']} (gain {d['default_pairs']['mean_gain']:+.5f}, "
               f"EIG {d['default_pairs']['mean_eig']:.5f}), non-default "
               f"{d['nondefault_pairs']['n']} (gain "
               f"{d['nondefault_pairs']['mean_gain']:+.5f}, EIG "
               f"{d['nondefault_pairs']['mean_eig']:.5f}) · per patient sum EIG "
               f"{d['sum_eig_mean']:.4f} vs sum gain {d['sum_gain_mean']:+.4f}"]
        for b in d.get("by_activity", []):
            L_.append(f"- activity {b['tercile']} (median {b['activity_median']:.0f} "
                      f"codes, n {b['n']}): R1 median {b['R1_median']:+.2f} · top-1 gain "
                      f"{b['top1_gain']:+.4f}")
    return L_ + [""]


def run_differential(test_df, train_df, run_dir, pattern, *, adhoc=False,
                     results_name="results_readout_stacked_mass99.json",
                     heads_tag="stacked_mass99", max_patients=600, max_train=2000,
                     thin_frac=0.1, own_top=5, rel_lambda=0.6, seed=0, tag=None,
                     crossfit=False, sidecar_uri=None, max_members=None,
                     incident=False, groups_like=None, member_rule="auto",
                     code_select="df", code_lambda=0.6, workup=False,
                     workup_window="strict", workup_drop_done=False,
                     workup_new_only=True, workup_length_edges=()):
    """Framing 1 — DIFFERENTIAL WITH REASONS (spec 2026-10-13 clinician framings,
    R0/R1; `differential.py`), with framing 2 as workup (`workup`, exp 0167). Engines
    on the same held-out patients positive on exactly one member: E2 renormalized
    stacked heads, E3 a multiclass head on θ, E5 a code head, E6 codes + θ (E3/E5/E6
    fit on the train split). Records lose attestation codes and `[measured]` tokens
    first. The patients split in two halves (the workup's temper is fit on one and
    scored on the other; `crossfit` swaps them so every patient is scored once).
    Writes `differential[_<tag>].md/.json` (pooled only), and with `max_members` the
    survey table `<tag>_survey.md`.

    (The presence-subtype and gated-likelihood engines E1/E4, ever-coded training
    members, multi-positive soft targets and the hidden-record next-question
    protocol were removed in the 2026-10-29 cleanup; last at commit 29556f3d.)"""
    import time
    import differential as dfm
    from model_engine import ModelEngine
    import voi_tables as vt
    from spark_vi.readout import closure_matrix
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
    member_cid, parent_adj = {}, None
    prior_pairs, prior_nodes = None, []
    if incident or workup:
        # Exp 0149: members' Mondo ids (bundle int2cid) and the Mondo is-a, so the
        # sidecar's finer terms roll up onto the members.
        from readout_dag import load_mondo_parent_adj
        bmj = json.loads(bm.read_text())
        member_cid = {int(e): int(c) for e, c in bmj["int2cid"].items()}
        cm_ = (json.loads((run_dir / "manifest.json").read_text())
               .get("corpus_manifest") or {})
        parent_adj, _nm = load_mondo_parent_adj(
            cm_.get("mondo_version"), cm_.get("mondo_cache_dir") or "data/mondo")
        member_cid.update({int(e): int(c) for e, c in
                           json.loads((run_dir / "readout_dag.json").read_text())
                           ["int2cid"].items()})
    if incident:
        # Exp 0151: (person, Mondo node) first attested BEFORE the person's index, for
        # the test persons — computed once; each group then drops the persons with
        # any such node at or under one of its members or the group node itself.
        prior_pairs = _test_prior_pairs(test_df, run_dir, sidecar_uri).cache()
        _np = prior_pairs.count()                      # materialize the cache ONCE
        prior_nodes = [int(r[0]) for r in
                       prior_pairs.select("node_cid").distinct().collect()]
        _cprint(f"[differential]   incident rule: {_np:,} (test person, node) pairs "
                f"attested before the index over {len(prior_nodes):,} Mondo nodes",
                flush=True)
    wk_ctx = None
    if workup:
        wk_ctx = _workup_setup(train_df.sparkSession, run_dir, ctx, sidecar_uri)
        _cprint(f"[differential]   workup (exp 0167): window [index, t_dx"
                f"{']' if workup_window == 'inclusive' else ')'}"
                f"{', order-only answers dropped' if workup_drop_done else ''} · "
                f"{wk_ctx['n_nodes']:,} sidecar nodes", flush=True)
    sets = vt.select_sets(ctx, pattern, adhoc=adhoc)
    # Member rule. A flat label set scores every label member ("all"). On a nested
    # label space (no label set: exp 0152) "top" scores the group's top-level label
    # terms, each rolling up its subtypes; "most-specific" scores the leaves only.
    if member_rule == "auto":
        _mf = json.loads((run_dir / "manifest.json").read_text())
        member_rule = ("all" if (_mf.get("corpus_manifest") or {}).get("label_set")
                       else "top")
    all_members = {}
    if member_rule != "all":
        _pi = ctx.rdag["parent_int"]
        _s2 = []
        for n_, g_, m_ in sets:
            ms = (most_specific_members(m_, _pi) if member_rule == "most-specific"
                  else top_members(m_, _pi, group=n_))
            all_members[g_] = list(m_)
            if len(ms) >= 2:
                _s2.append((n_, g_, ms))
        sets = _s2
    if groups_like:
        # Exp 0152 read 4: the groups a previous survey scored, matched by name.
        _names = set(json.loads(Path(groups_like).read_text()))
        sets = [x for x in sets if x[1] in _names]
    if max_members:
        # Exp 0150 survey: every Mondo group up to `max_members` label members (the
        # broad body-system groups are not a differential).
        sets = [x for x in sets if len(x[2]) <= int(max_members)]
    if not sets:
        raise SystemExit(f"[differential] no candidate set matches {pattern!r}")
    _cprint(f"[differential]   member rule: {member_rule}", flush=True)
    _cprint(f"[differential]   {len(sets)} candidate set(s)", flush=True)
    L_ = [f"# {run_dir.name} — differential with reasons (framing 1)",
          f"pattern {pattern!r}{' (ad hoc)' if adhoc else ''} · records minus attestation "
          f"codes and {len(measured):,} [measured] tokens · E2 stacked heads "
          f"({heads_tag}) · E3 multiclass head on θ · E5/E6 codes by "
          f"{'group document frequency' if code_select == 'df' else f'within-group relevance (λ={code_lambda:g})'}"
          f" (E3/E5/E6 fit on the train split)", ""]
    js = {}
    for n, gname, members in sets:
        t0 = time.time()
        sub = all_members.get(gname, members)        # every label node in the group
        excl = {0: vt.set_exclusions(ctx, n, sub)}
        # Name-matched attestation (0141): a condition concept whose name IS a member's
        # or the group's name ("Hypertrophic cardiomyopathy" 40565500, "Cardiomyopathy")
        # but which the code map missed is attestation too, not a finding.
        nm_targets = {str(ctx.rd_names.get(m, "")).strip().lower() for m in sub}
        nm_targets.add(str(gname).strip().lower())
        if ctx.names and ctx.inv_maps:
            for v, cid in ctx.inv_maps[0].items():
                if str(ctx.names.get(int(cid), "")).strip().lower() in nm_targets:
                    excl[0].add(int(v))
        if measured:
            excl[dm] = measured
        eng = ModelEngine(lams_fit, alpha, heads,
                          {m: [int(a) for a in np.flatnonzero(M[m])] for m in members},
                          members)
        bg = list(range(ctx.n_bg))
        feat = bg + sorted({t for m in members for t in ctx.block[m]})

        def prep(rows):
            out = []
            for row in rows:
                truth, bags = row[0], row[1]
                if len(truth) != 1:                    # single-member patients only
                    continue
                fb, lens = dfm.filter_record(bags, excl)
                gi = [eng.gidx(d, v) for d, b in fb.items() for v in b]
                gc = [c for d, b in fb.items() for c in b.values()]
                out.append((truth, fb, lens, gi, gc, row[3] if len(row) > 3 else None))
            return out
        te_src = test_df
        if incident:
            te_src = _incident_test_frame(test_df, prior_pairs, prior_nodes,
                                          [member_cid[int(m)] for m in sub]
                                          + ([member_cid[int(n)]] if int(n) > 0
                                             and int(n) in member_cid else []),
                                          parent_adj)
        tr_src, wk_cols = train_df, ()
        if wk_ctx is not None:
            # Exp 0167: each person's workup window for THIS group (index to the first
            # diagnosis of any member), joined on before the collect.
            from pyspark.sql import functions as F
            wkf = workup_frame(wk_ctx["first"], wk_ctx["hz"], wk_ctx["workup"],
                               wk_ctx["nodes"], [member_cid[int(m)] for m in members],
                               parent_adj, label_window_days=wk_ctx["window"],
                               inclusive=(workup_window == "inclusive")).cache()
            te_src = te_src.join(F.broadcast(wkf), "person_id", "left")
            tr_src = train_df.join(F.broadcast(wkf), "person_id", "left")
            wk_cols = ("__wk", "__wdays")
        te_raw, n_te = collect_set_patients(te_src, members, len(ctx.lams),
                                            max_patients=max_patients, seed=seed,
                                            extra_cols=wk_cols)
        tr_raw, n_tr = collect_set_patients(tr_src, members, len(ctx.lams),
                                            max_patients=max_train, seed=seed,
                                            extra_cols=wk_cols)
        if wk_ctx is not None:
            wkf.unpersist()
        n_multi = sum(len(r[0]) > 1 for r in te_raw)
        te = prep(te_raw)
        tr = prep(tr_raw)
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
        e3 = e5 = e6 = None
        ytr = [r[0][0] for r in tr]
        Ytr = dfm.label_matrix(ytr, len(members))
        if int((Ytr.sum(axis=0) > 0).sum()) >= 2:
            th_tr = np.stack([theta(r) for r in tr])
            e3 = dfm.MulticlassHead(feat, len(members), seed=seed).fit(th_tr, ytr)
            # Exp 0148 (insight 0101): the same softmax on the record's CODES (E5) and
            # on codes + log θ (E6) — same filtered records, same patients.
            # Exp 0166: `code_select` picks E5/E6's codes by group frequency (df) or
            # by within-group relevance (lift mixed with frequency, `code_lambda`).
            ck = dict(seed=seed, select=code_select, rel_lambda=code_lambda)
            e5 = dfm.CodeHead(len(members), **ck).fit([r[1] for r in tr], ytr)
            e6 = dfm.CodeHead(len(members), theta_topics=feat, **ck).fit(
                [r[1] for r in tr], ytr, th_tr)
        # Prior over members from the TRAIN split (soft counts), add-one smoothed: a
        # member absent from the scored-results file (0140 first run) got prior 1e-12
        # and could never be called, and a test-derived prior leaks.
        pri = Ytr.sum(axis=0) + 1.0
        pri = pri / pri.sum()

        def engine_rows(idx):
            out = {"prior": [pri for _ in idx],
                   "E2 stacked heads": [eng.posterior(th_te[i]) for i in idx]}
            if e3 is not None:
                out["E3 multiclass head"] = list(e3.predict(th_te[idx]))
                out["E5 code head"] = list(e5.predict([te[i][1] for i in idx]))
                out["E6 θ + codes"] = list(e6.predict([te[i][1] for i in idx],
                                                       th_te[idx]))
            return out
        # Exp 0149 item 3: two halves; with CROSS-FIT they swap, so every patient is
        # scored once (the workup fits its temper on one half, scores the other).
        halves = [(list(range(0, len(te), 2)), list(range(1, len(te), 2)))]
        if crossfit:
            halves.append((halves[0][1], halves[0][0]))
        rows, tv, ev_all = {}, [], []
        for fi, ei in halves:
            for k, v in engine_rows(ei).items():
                rows.setdefault(k, []).extend(v)
            tv += [truths[i] for i in ei]
            ev_all += ei
        fit_i = halves[0][0]
        res = {"members": [int(m) for m in members], "member_rule": member_rule,
               "n_eval": len(ev_all), "n_fit": len(fit_i), "n_train": len(tr),
               "crossfit": bool(crossfit), "code_select": code_select,
               "code_lambda": (code_lambda if code_select == "relevance" else None),
               "e3_lambda": (e3.lam if e3 is not None else None),
               "e5_lambda": (e5.lam if e5 is not None else None),
               "e6_lambda": (e6.lam if e6 is not None else None),
               "e5_n_codes": (e5.p_codes if e5 is not None else None)}
        res["metrics"] = {k: dfm.metrics(v, tv) for k, v in rows.items()}
        # reasons on correctly called evaluation patients
        r3, r5 = [], []
        for j, i in enumerate(ev_all):
            t = truths[i][0]
            m = members[t]
            if e3 is not None and int(np.argmax(rows["E3 multiclass head"][j])) == t \
                    and te[i][3]:
                c3 = dfm.e3_reasons(e3, th_te[i], gsum_te[i], eng.eb[:, te[i][3]],
                                    te[i][4], t)
                r3.append((m, {_split_gidx(eng, g): float(x)
                               for g, x in zip(te[i][3], c3)}))
            if e5 is not None and int(np.argmax(rows["E5 code head"][j])) == t:
                r5.append((m, e5.reasons(te[i][1], t)))
        res["reasons_e3"] = dfm.pool_reasons(r3)
        res["reasons_e5"] = dfm.pool_reasons(r5)
        named = {m: ctx.rd_names.get(m, str(m)) for m in members}
        L_.append(f"## {gname} — {len(members)} members · {len(ev_all)} scored patients "
                  f"({'cross-fit, both halves' if crossfit else 'one half'})"
                  f" · {len(tr)} train patients")
        L_.append("")
        L_.append("| engine | top-1 | top-3 | bal top-1 | log loss | ECE |")
        L_.append("|---|--:|--:|--:|--:|--:|")
        for k, mtr in res["metrics"].items():
            L_.append(f"| {k} | {mtr['top1']:.3f} | {mtr['top3']:.3f} | "
                      f"{mtr['bal_top1']:.3f} | {mtr['logloss']:.3f} | {mtr['ece']:.3f} |")
        for key, lab in (("reasons_e3", "E3"), ("reasons_e5", "E5")):
            if res[key]:
                L_.append("")
                L_.append(f"{lab} reasons (top supporting codes, correctly called "
                          f"patients, members with >= {dfm.EGRESS_FLOOR}):")
                for m, codes in res[key].items():
                    L_.append(f"- {named[m][:40]}: " + " · ".join(
                        ctx.code_name(d, v) for d, v in codes))
        L_.append("")
        if wk_ctx is not None and e3 is not None:
            wres = _workup_block(
                dfm, ctx, wk_ctx, tr, te, ytr, members, excl, halves, truths,
                {"E6": e6.predict([r[1] for r in te], th_te), "E3": e3.predict(th_te)},
                pri, drop_done=workup_drop_done, new_only=workup_new_only,
                length_edges=tuple(workup_length_edges))
            res["workup"] = wres
            L_ += _format_workup(wres, workup_window)
        _cprint(f"[differential]   {gname}: done in {time.time() - t0:.0f}s", flush=True)
        js[gname] = {k: v for k, v in res.items() if not k.startswith("reasons")}
    text = "\n".join(L_).rstrip() + "\n"
    stem = "differential" + (f"_{tag}" if tag else "")
    (run_dir / f"{stem}.md").write_text(text)
    (run_dir / f"{stem}.json").write_text(json.dumps(js, indent=1, default=str))
    if max_members:
        sv = differential_survey(js, sets, run_dir)
        (run_dir / f"{stem}_survey.md").write_text(sv)
        print(sv, flush=True)
    print(text, flush=True)
    _cprint(f"[differential]   wrote {run_dir / (stem + '.md')}", flush=True)
    return js


def differential_survey(js, sets, run_dir, *, top=40):
    """Exp 0150: one row per surveyed Mondo group — framing 1 (balanced top-1 of the
    prior, E3 θ, E5 codes, E6 θ + codes; the best engine's lift over chance = bal × M;
    E6 top-3 vs the prior's), framing 2 when the workup ran (E6's fact-level ranking:
    Spearman of mean EIG vs mean realized gain, and the top-5 facts' gain vs all),
    framing 3 (members' mean new-diagnosis AUC from the whole-history prevalence
    checks, θ and raw codes, when those files are in the run dir). Pooled per group;
    groups under the egress floor never reach `js`."""
    from prevalence_check import framing3_sources
    th, rw = framing3_sources(run_dir)

    def f3(members, src):
        if not src or "history_stacked" not in src:
            return float("nan")
        pn = src["history_stacked"]["per_node"]
        v = [pn[str(m)]["auc_incident"] for m in members if str(m) in pn]
        return float(np.mean(v)) if v else float("nan")

    def f2(r):
        f = ((r.get("workup") or {}).get("E6") or {}).get("facts") or {}
        return (f.get("spearman_mean_eig_vs_mean_gain", float("nan")),
                f.get("top5_mean_gain", float("nan")), f.get("all_mean_gain", float("nan")))
    mem = {g: m for _n, g, m in sets}
    rows = []
    for g, r in js.items():
        mt = r.get("metrics") or {}
        if "E6 θ + codes" not in mt:
            continue
        M = len(mem.get(g, []))
        bal = {k: mt[k]["bal_top1"] for k in ("prior", "E3 multiclass head",
                                              "E5 code head", "E6 θ + codes")}
        best = max(("E3 multiclass head", "E5 code head", "E6 θ + codes"),
                   key=lambda k: bal[k])
        rho, g5, gall = f2(r)
        rows.append({"group": g, "M": M, "n": r["n_eval"], "prior": bal["prior"],
                     "e3": bal["E3 multiclass head"], "e5": bal["E5 code head"],
                     "e6": bal["E6 θ + codes"], "best": best.split()[0],
                     "lift": bal[best] * M, "top3_e6": mt["E6 θ + codes"]["top3"],
                     "top3_prior": mt["prior"]["top3"], "wk_rho": rho, "wk_top5": g5,
                     "wk_all": gall, "f3_theta": f3(mem.get(g, []), th),
                     "f3_raw": f3(mem.get(g, []), rw)})
    if not rows:
        return "# differential survey\n(no group scored)\n"
    rows.sort(key=lambda x: -x["lift"])
    med = lambda k: float(np.nanmedian([x[k] for x in rows]))
    win = {e: sum(x["best"] == e for x in rows) for e in ("E3", "E5", "E6")}
    L = ["# differential survey (exp 0150) — framings 1-3 across Mondo groups", "",
         f"{len(rows)} groups scored · best engine by balanced top-1: E3 {win['E3']}, "
         f"E5 {win['E5']}, E6 {win['E6']} · median lift over chance {med('lift'):.2f}× · "
         f"median E6 − E3 balanced top-1 "
         f"{float(np.median([x['e6'] - x['e3'] for x in rows])):+.3f}"
         + (f" · workup (E6) fact-level Spearman median {med('wk_rho'):+.2f}"
            if any(np.isfinite(x["wk_rho"]) for x in rows) else ""), "",
         f"Top {min(top, len(rows))} by lift (balanced top-1 of the best engine × members):",
         "", "| group | M | n | prior | E3 | E5 | E6 | lift | top-3 E6 (prior) | "
         "F2 workup ρ · top-5 / all gain | F3 AUC θ / raw |",
         "|---|--:|--:|--:|--:|--:|--:|--:|---|---|---|"]
    for x in rows[:top]:
        L.append(f"| {str(x['group'])[:40]} | {x['M']} | {x['n']} | {x['prior']:.2f} | "
                 f"{x['e3']:.2f} | {x['e5']:.2f} | {x['e6']:.2f} | {x['lift']:.2f} | "
                 f"{x['top3_e6']:.2f} ({x['top3_prior']:.2f}) | {x['wk_rho']:+.2f} · "
                 f"{x['wk_top5']:+.4f} / {x['wk_all']:+.4f} | "
                 f"{x['f3_theta']:.2f} / {x['f3_raw']:.2f} |")
    L.append("")
    return "\n".join(L) + "\n"


def _split_gidx(eng, g):
    """Global concatenated vocab id -> (domain, vocab idx)."""
    d = int(np.searchsorted(eng.offsets, g, side="right") - 1)
    return d, int(g - eng.offsets[d])


def _whole_history_prior(test_scored, run_dir, *, readout_int2cid, parent_adj,
                         sidecar_uri=None):
    """Exp 0145 (spec 2026-10-13 label redesign R1): the WHOLE-HISTORY prior from the
    E4 sidecar — a first attestation of X's Mondo term or a descendant dated BEFORE
    the person's index (the sidecar's own index_horizon, re-derived by the
    assembler's index builder at sidecar build). Not the 5-year lookback.

    Returns `(test_scored + an "_prior_hist" array<int> column of readout nodes, None
    where the person has none; receipt)`: a join, so nothing per person reaches the
    driver (ADR 0051 step 4)."""
    import prevalence_check as pv
    from pyspark.sql import functions as F
    from conversion_sidecar import try_load_index_horizon, try_load_sidecar
    wp = Path(run_dir) / "conversion_sidecar.json"
    if not wp.exists():
        raise SystemExit("[prevalence] --prev-history needs <run>/conversion_sidecar.json "
                         "(make build-conversion-sidecar ID=N)")
    wit = json.loads(wp.read_text())
    uri = sidecar_uri or wit.get("sidecar_uri")
    spark = test_scored.sparkSession
    first = try_load_sidecar(spark, uri, wit["key"])
    hz = try_load_index_horizon(spark, uri, wit["key"])
    if first is None or hz is None:
        raise SystemExit(f"[prevalence] sidecar MISS at {uri}/{wit['key']} — rebuild it "
                         "on this cluster (make build-conversion-sidecar ID=N)")
    persons = test_scored.select("person_id").distinct()
    hz_t = hz.join(persons, "person_id").select("person_id", "index_date").cache()
    n_idx = hz_t.count()
    pairs = (first.join(hz_t, "person_id")
             .where(F.col("first_attested_date") < F.col("index_date"))
             .select("person_id", "node_cid").distinct().cache())
    nodes = [int(r[0]) for r in pairs.select("node_cid").distinct().collect()]
    n2r = pv.sidecar_nodes_to_readout(nodes, readout_int2cid, parent_adj)
    mrows = [(int(n), int(r)) for n, rs in n2r.items() for r in rs]
    mdf = spark.createDataFrame(mrows, "node_cid long, rnode int")
    hist = (pairs.join(F.broadcast(mdf), "node_cid").groupBy("person_id")
            .agg(F.collect_set("rnode").alias("_prior_hist")))
    out = test_scored.join(hist, "person_id", "left")
    return out, {"n_with_index": n_idx, "n_sidecar_nodes_prior": len(nodes),
                 "n_mapped_to_readout": len(n2r)}, (pairs, hz_t)


def run_prevalence_check(test_scored, run_dir, *, C, parent_int,
                         heads_tag="stacked_mass99", tag=None, top=15,
                         history=None, node_subset=None):
    """How much of the case-finding AUC is prevalent recall? (insight 0100;
    `prevalence_check.py`). Scores the test split with the saved heads (no solve), flags
    each (patient, node) with a mapped attestation code of the node or a label
    descendant in the PRE-INDEX condition features, and reports per-node AUC over all
    patients vs with those patients set aside — flat heads and the stacked product.
    Writes `prevalence_check[_<tag>].md/.json`.

    The scores, the prior flags and the closure product stay on the executors as score
    blocks, and the per-node AUCs come from merged tallies (ADR 0051 step 4): the
    driver never holds a (patients, nodes) matrix.

    `history` (exp 0145): a dict of `_whole_history_prior` kwargs; adds the
    WHOLE-HISTORY prior rule beside the lookback one, with a containment check."""
    import prevalence_check as pv
    import voi_tables as vt
    from dag_gated_fit import _doc_key_column
    from spark_vi.eval.metrics import local as ml
    from spark_vi.readout import closure_matrix
    from spark_vi.readout import evaluate as ev
    from spark_vi.readout.spark import score_blocks
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
    C = int(C)
    col = "features_0" if "features_0" in test_scored.columns else "features"
    df = test_scored.withColumn("doc_key", _doc_key_column(test_scored, "person_id"))
    res = {}
    rec, cached = None, ()
    extra = [col]
    if history:
        df, rec, cached = _whole_history_prior(df, run_dir, **history)
        extra.append("_prior_hist")
    rdd, bheads = score_blocks(df, C, z["V"], z["b_raw"], degenerate=z["degenerate"],
                               const=z["const"], theta_topm=int(z["theta_topm"]),
                               extra_cols=extra)
    blocks = ev.Blocks(rdd, bcasts=[bheads])
    bv2n = blocks.broadcast(pv.vocab_to_nodes(ctx.own_idx, ctx.rdag["pos_sources"]))

    def _priors(b, _bv=bv2n, _col=col, _C=C, _hist=bool(history)):
        def _idx(v):
            i = getattr(v, "indices", None)
            if i is None:
                a = np.asarray(v.toArray() if hasattr(v, "toArray") else v)
                i = np.flatnonzero(a)
            return [int(x) for x in i]
        out = {k: b[k] for k in ("ids", "proba", "y", "mask")}
        out["prior"] = pv.prior_rows([_idx(v) for v in b[_col]], ev._val(_bv), _C)
        if _hist:
            out["prior_hist"] = pv.rows_from_node_lists(b["_prior_hist"], _C)
        return out
    blocks = blocks.map(_priors)
    blocks._bcasts.append(bv2n)
    blocks = ev.with_stacked(blocks, closure_matrix(parent_int, C)).persist()
    nodes = list(range(1, C))
    if node_subset is not None:
        # Exp 0152: only the solved heads — masked heads are constants (AUC 0.5).
        nodes = [n for n in nodes if node_subset[n]]
    rules = [("", "prior")] + ([("history_", "prior_hist")] if history else [])
    arms = (("stacked", "stacked"), ("flat", "proba"))

    def _pieces(b, _nodes=tuple(nodes), _rules=tuple(r for _, r in rules)):
        for arm, sc in arms:
            yield from ml.column_pieces(b[sc], b["y"], columns=_nodes,
                                        key=lambda c, _a=arm: (_a, "all", c))
            for rule in _rules:
                yield from ml.column_pieces(b[sc], b["y"], ~b[rule], columns=_nodes,
                                            key=lambda c, _a=arm, _r=rule: (_a, _r, c))
    try:
        per_key = blocks.metrics(_pieces)

        def _counts(b, _h=bool(history)):
            a = b["prior"]
            hh = b["prior_hist"] if _h else a
            return np.array([a.shape[0], a.sum(), hh.sum(), (a & hh).sum()],
                            dtype=np.int64)
        n_docs, n_lb, n_h, n_both = (int(x) for x in blocks.reduce(
            _counts, lambda u, v: u + v))
    finally:
        blocks.unpersist()
        for d in cached:
            d.unpersist()
    L_ = [f"# {run_dir.name} — prevalence check: AUC with prior-diagnosis patients set "
          f"aside", f"heads {hp.name} · {n_docs:,} test patients · LOOKBACK prior = a "
          f"mapped attestation code of the node (or a label descendant) in the 5-year "
          f"pre-index condition features (a lower bound: older diagnoses count as "
          f"incident)", ""]
    if history:
        rec = {"n_test_docs": n_docs, **rec}
        res["history_receipt"] = rec
        res["containment_lookback_in_history"] = (n_both / n_lb if n_lb
                                                  else float("nan"))
        L_ += [f"WHOLE-HISTORY prior (exp 0145) = a sidecar first attestation of the "
               f"node's Mondo term or a descendant before the index · test docs "
               f"{rec['n_test_docs']:,}, with a sidecar index {rec['n_with_index']:,} · "
               f"prior cells: lookback {n_lb:,}, whole history {n_h:,} · lookback "
               f"cells inside whole-history "
               f"{res['containment_lookback_in_history']:.3f} (expect ~1; low = index "
               f"or node-id mismatch)", ""]
    for pre, rule in rules:
        for arm0, _sc in arms:
            arm = pre + arm0
            m_all = {c: per_key[(arm0, "all", c)] for c in nodes
                     if (arm0, "all", c) in per_key}
            m_inc = {c: per_key[(arm0, rule, c)] for c in nodes
                     if (arm0, rule, c) in per_key}
            per = pv.node_aucs_from_metrics(m_all, m_inc, nodes)
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

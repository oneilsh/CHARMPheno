"""Code-level value of information over Mondo candidate sets (spec 2026-10-07 §D4,
WP-D part 1; the 2026-08-14 VOI metrics report §4).

The clinician's question, at population level: "this patient is somewhere in Mondo
group G (connective tissue disorders, cardiomyopathies); which not-yet-observed code
would most sharpen the call between G's diseases?" Off-cluster and pure numpy over a
saved fit — no patient data is read, only model parameters, the readout DAG and pooled
per-node counts (used as the prior, never printed as counts).

THE MODEL (why each piece has the shape it has)
-----------------------------------------------
* Candidate set S: the label-node members of one readout-DAG node (a Mondo ancestor
  head, or a label node with label descendants) — `readout_dag.json`'s
  `pos_sources`, so S is exactly the set the stacked readout already scores under
  that ancestor.
* Disease phenotype beta_d(v): the evidence-weighted mixture of disease d's SIGNATURE
  topics (the census's own-code rule + thin floor: strata, duplicates, thin and
  starved topics excluded), per domain. Strata are excluded on purpose: a care-context
  topic that every disease shares carries no between-disease information and would
  only dilute the contrast (the report's prerequisite (2), "sibling-contrastive beta").
* Background beta_bg(v): the evidence-weighted mean of the background topics — what a
  patient's codes look like when they are not about d.
* Presence model: a patient of disease d with L tokens in the domain, a share rho of
  them from d's own profile, shows code v at least once with probability
      p_d(v) = 1 - exp(-L * (rho * beta_d(v) + (1 - rho) * beta_bg(v))),
  the Poisson limit of L draws from the mixture. The background term is what keeps
  a code that d never emits from being a perfect negative (absence is weak evidence
  in EHR data). L and rho are KNOBS, not fitted (defaults 40 and 0.3); the EIG
  ranking is insensitive to them over a wide range because they scale every
  disease's p_d(v) alike.
* Prior pi over S: proportional to each member's positive count among scored test
  persons (the results file's per-node `n_pos`), uniform when absent. A population
  prior, i.e. this table answers "for a patient known to be in G, before anything
  else is known about them".
* EIG(v) = H(pi) - P(x=1) H(pi | x=1) - P(x=0) H(pi | x=0), in bits — Nelson 2005's
  expected information gain of observing whether v is present.

By default the members' OWN codes are excluded (`exclude_own`): the code that names
the diagnosis is trivially the most informative "test" and says nothing a clinician
does not already know. They are reported in their own short list. The candidate
set's OWN node is excluded the same way when it is itself a label node ("Cardiomyopathy"
is the given, not a finding).

`adhoc=True` builds ONE set from the label nodes whose names match the pattern, for
contrasts no single readout node holds (Marfan vs EDS: their nearest common ancestor is
a 19-member grouping).

`prior="uniform"` swaps the population prior for a flat one. Under the population prior
EIG is dominated by the common members (a 0% member can never be "favoured" enough to
move the expected entropy), so rare-disease contrasts (Marfan vs EDS inside a large
group) only show under the flat prior: "if it were equally likely to be any of these".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["disease_profile", "background_profile", "presence_prob", "eig_table",
           "candidate_sets", "voi_for_set", "build_voi_report", "load_voi_context",
           "select_sets", "set_profiles", "set_exclusions"]


def _norm(row):
    row = np.asarray(row, dtype=np.float64)
    s = row.sum()
    return row / s if s > 0 else row


def disease_profile(lam_dom, topics, evidence):
    """Evidence-weighted mixture of the E[beta] rows of `topics` in one domain."""
    if not topics:
        return None
    w = np.array([float(evidence[t]) for t in topics], dtype=np.float64)
    rows = np.stack([_norm(lam_dom[t]) for t in topics])
    return (w[:, None] * rows).sum(axis=0) / w.sum()


def background_profile(lam_dom, n_bg, evidence):
    return disease_profile(lam_dom, list(range(int(n_bg))), evidence)


def presence_prob(beta_d, beta_bg, *, L=40.0, rho=0.3):
    """P(code v appears at least once | disease d) — see the module docstring."""
    return 1.0 - np.exp(-float(L) * (float(rho) * np.asarray(beta_d)
                                     + (1.0 - float(rho)) * np.asarray(beta_bg)))


def _entropy(p, axis=0):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-300, 1.0)
    return -(p * np.log2(p)).sum(axis=axis)


def eig_table(prior, P):
    """EIG (bits) of observing presence of each code. `prior` (S,), `P` (S, V) the
    per-disease presence probabilities. Returns `(eig (V,), post1 (S, V))` where
    `post1[:, v]` is the posterior over S after observing v present."""
    prior = np.asarray(prior, dtype=np.float64)
    prior = prior / prior.sum()
    P = np.clip(np.asarray(P, dtype=np.float64), 1e-12, 1 - 1e-12)
    p1 = prior @ P
    post1 = prior[:, None] * P / p1
    post0 = prior[:, None] * (1.0 - P) / (1.0 - p1)
    eig = _entropy(prior) - p1 * _entropy(post1) - (1.0 - p1) * _entropy(post0)
    return eig, post1


def candidate_sets(rdag, pattern, *, names, min_members=2):
    """`[(readout node id, name, [member label-node ids])]` for every readout-DAG node
    whose name matches `pattern` and that has >= `min_members` label-node members.
    `rdag` is the parsed `readout_dag.json`; `names` `{readout id: name}`."""
    import re
    rx = re.compile(pattern, re.I)
    C_fit = int(rdag["C_fit"])
    out = []
    for n, srcs in enumerate(rdag["pos_sources"]):
        if n == 0 or not rx.search(str(names.get(n, ""))):
            continue
        members = sorted({int(e) for e in srcs if 0 < int(e) < C_fit})
        if len(members) >= int(min_members):
            out.append((n, names.get(n, str(n)), members))
    return out


def voi_for_set(members, profiles, beta_bg, prior_counts, *, L=40.0, rho=0.3,
                exclude=None, top=15):
    """Rank codes for one candidate set in one domain. `profiles` `{node: beta}`
    (members without a profile are dropped and reported), `prior_counts`
    `{node: count}`. Returns a dict with the members used, the prior, and the top
    codes as `(vocab_idx, eig_bits, favoured_member, prior_of_it, post_of_it)`."""
    used = [m for m in members if profiles.get(m) is not None]
    dropped = [m for m in members if profiles.get(m) is None]
    if len(used) < 2:
        return {"used": used, "dropped": dropped, "rows": [], "prior": []}
    prior = np.array([float(prior_counts.get(m, 0) or 0) for m in used])
    if prior.sum() <= 0:
        prior = np.ones(len(used))
    prior = prior / prior.sum()
    P = np.stack([presence_prob(profiles[m], beta_bg, L=L, rho=rho) for m in used])
    eig, post1 = eig_table(prior, P)
    order = np.argsort(eig)[::-1]
    excl = set(exclude or ())
    rows = []
    for v in order:
        if int(v) in excl:
            continue
        j = int(np.argmax(post1[:, v] - prior))
        rows.append((int(v), float(eig[v]), used[j], float(prior[j]),
                     float(post1[j, v])))
        if len(rows) >= int(top):
            break
    return {"used": used, "dropped": dropped, "rows": rows,
            "prior": [float(x) for x in prior], "h_prior": float(_entropy(prior))}


def load_voi_context(run_dir, *, bundle_meta_path=None, vocab_path=None,
                     names_path=None, own_codes_path=None, results_name=None,
                     results_arm="gated_pc_stacked.marginal_ranking.stacked",
                     thin_frac=0.1, own_top=None, rel_lambda=None,
                     prior="population", include_strata=False):
    """Everything the VOI tables and the acquisition simulation (`voi_acquisition`)
    share, read once from a finished run dir: λ per domain, the census's per-node
    signature topics, own-code vocab indices, readout-DAG names, the prior counts.

    `include_strata=True` lets a node's STRATUM topics into its profile too (the
    census-rule sensitivity: a stratum by the own-code rule may still be a real
    profile of the disease, e.g. hEDS's hypermobility-POTS topic)."""
    from types import SimpleNamespace
    import inspect_topics as it

    run_dir = Path(run_dir)
    npz, manifest = it.load_run(run_dir)
    lams = it.domain_lambdas(npz)
    sh = it.topic_sharpness(lams)
    n_bg, tpn = int(manifest["n_bg"]), int(manifest["tpn"])
    nodes, int2cid = it.node_order(manifest)
    block = {e: list(range(n_bg + i * tpn, n_bg + (i + 1) * tpn))
             for i, e in enumerate(nodes)}
    dom_names = manifest.get("domain_names") or [f"dom{i}" for i in range(len(lams))]
    meta = it.load_bundle_meta(bundle_meta_path) if bundle_meta_path else None
    name_by_id = {int(k): v for k, v in
                  manifest.get("corpus_manifest", {}).get("name_by_id", {}).items()}
    if meta and "name_by_id" in meta:
        name_by_id.update({int(k): v for k, v in meta["name_by_id"].items()})
    inv_maps = None
    if meta and "vocab_maps" in meta:
        inv_maps = [{int(i): int(c) for c, i in vm.items()} for vm in meta["vocab_maps"]]
    elif vocab_path:
        inv_maps = it.load_vocab_maps(vocab_path, len(lams))
    names = it.load_concept_names(names_path) if names_path else {}

    rd_path = run_dir / "readout_dag.json"
    if not rd_path.exists():
        raise SystemExit("[voi] needs <run>/readout_dag.json (a --readout-hierarchy "
                         "mondo readout of this run)")
    rdag = json.loads(rd_path.read_text())
    rd_names = {}
    for n, cid in rdag["int2cid"].items():
        nm = (rdag.get("name_by_id") or {}).get(str(cid)) or name_by_id.get(int(cid))
        rd_names[int(n)] = nm or f"cid:{cid}"

    own_idx = {}
    if own_codes_path and inv_maps:
        vm0 = {c: i for i, c in inv_maps[0].items()}
        own_idx = it.own_code_vocab_idx(it.load_code_map(own_codes_path), int2cid, vm0)
    kinds = {}
    _pw0 = it.model_word_marginal(lams[0])
    for e, blk in block.items():
        kinds[e] = it.classify_block_topics(
            blk, lams, sh, n_bg, own_idx=(own_idx.get(e) or None) if own_idx else None,
            thin_frac=thin_frac, own_top=own_top, pw0=_pw0, rel_lambda=rel_lambda)
    keep = {"signature", "stratum"} if include_strata else {"signature"}
    sig = {e: [t for t, k, _, _ in r if k in keep] for e, r in kinds.items()}

    prior_counts = {}
    if prior == "population" and results_name and (run_dir / results_name).exists():
        res = json.loads((run_dir / results_name).read_text())
        node = res
        for part in results_arm.split("."):
            node = node.get(part, {}) if isinstance(node, dict) else {}
        prior_counts = {int(k): v.get("n_pos") for k, v in
                        (node.get("per_node") or {}).items()}

    def code_name(d, v):
        cid = inv_maps[d].get(int(v)) if inv_maps and d < len(inv_maps) else None
        nm = (names.get(cid) if cid is not None else None) or (
            name_by_id.get(cid) if cid is not None else None)
        return it._trunc(nm or (f"cid:{cid}" if cid is not None else f"idx:{v}"), 48)

    return SimpleNamespace(run_dir=run_dir, lams=lams, sh=sh, n_bg=n_bg,
                           dom_names=dom_names, rdag=rdag, rd_names=rd_names,
                           own_idx=own_idx, sig=sig, prior_counts=prior_counts,
                           code_name=code_name, results_name=results_name,
                           C_fit=int(rdag["C_fit"]), inv_maps=inv_maps)


def select_sets(ctx, pattern, *, adhoc=False):
    """`[(readout node or -1, name, members)]`: readout-DAG groups matching the
    pattern, or with `adhoc` ONE set of the label nodes whose names match."""
    if adhoc:
        import re
        rx = re.compile(pattern, re.I)
        mem = [m for m in range(1, ctx.C_fit) if rx.search(ctx.rd_names.get(m, ""))]
        return [(-1, f"ad hoc set /{pattern}/", mem)] if len(mem) >= 2 else []
    return candidate_sets(ctx.rdag, pattern, names=ctx.rd_names)


def set_profiles(ctx, members, d):
    """`({member: beta or None}, beta_bg)` for one domain."""
    lam = ctx.lams[d]
    prof = {m: (disease_profile(lam, ctx.sig.get(m, []), ctx.sh["evidence"])
                if ctx.sig.get(m) else None) for m in members}
    return prof, background_profile(lam, ctx.n_bg, ctx.sh["evidence"])


def set_exclusions(ctx, n, members):
    """Domain-0 vocab indices excluded as ATTESTATION, not finding: the members' own
    codes and the set node's own (when it is a label node)."""
    excl = set()
    for m in list(members) + ([n] if 0 < n < ctx.C_fit else []):
        excl |= set(ctx.own_idx.get(m, ()))
    return excl


def build_voi_report(run_dir, pattern, *, bundle_meta_path=None, vocab_path=None,
                     names_path=None, own_codes_path=None, results_name=None,
                     results_arm="gated_pc_stacked.marginal_ranking.stacked",
                     L=40.0, rho=0.3, top=15, thin_frac=0.1, own_top=None,
                     domains=None, include_own=False, rel_lambda=None,
                     prior="population", adhoc=False):
    """The VOI report for every candidate set matching `pattern` (markdown)."""
    import inspect_topics as it

    ctx = load_voi_context(run_dir, bundle_meta_path=bundle_meta_path,
                           vocab_path=vocab_path, names_path=names_path,
                           own_codes_path=own_codes_path, results_name=results_name,
                           results_arm=results_arm, thin_frac=thin_frac,
                           own_top=own_top, rel_lambda=rel_lambda, prior=prior)
    rd_names, code_name = ctx.rd_names, ctx.code_name
    doms = [i for i, n in enumerate(ctx.dom_names)
            if domains is None or n in set(domains)]
    sets = select_sets(ctx, pattern, adhoc=adhoc)
    L_ = [f"# {ctx.run_dir.name} — code-level VOI over Mondo candidate sets",
          f"pattern {pattern!r} · {len(sets)} set(s) · presence model L={L:g}, "
          f"rho={rho:g} · signatures only (own-code rule, thin < {thin_frac:g}) · "
          f"prior = " + (f"scored test positives per member ({results_name})"
                         if ctx.prior_counts else "uniform") + " · "
          f"own codes {'INCLUDED' if include_own else 'excluded (listed apart)'}", ""]
    for n, gname, members in sets:
        L_.append(f"## {gname} — {len(members)} member(s)")
        for d in doms:
            profiles, bg = set_profiles(ctx, members, d)
            excl = (set_exclusions(ctx, n, members)
                    if d == 0 and not include_own else set())
            r = voi_for_set(members, profiles, bg, ctx.prior_counts, L=L, rho=rho,
                            exclude=excl, top=top)
            if not r["rows"]:
                L_.append(f"- {ctx.dom_names[d]}: fewer than 2 members with a signature "
                          f"profile; skipped")
                continue
            if d == doms[0]:
                L_.append(f"H(prior)={r['h_prior']:.2f} bits · members (prior), "
                          f"most likely first:")
                mp = sorted(zip(r["used"], r["prior"]), key=lambda x: -x[1])
                for i in range(0, len(mp), 4):
                    L_.append("  " + " · ".join(
                        f"{it._trunc(rd_names.get(m, str(m)), 32)} {100 * p:.0f}%"
                        for m, p in mp[i:i + 4]))
                if r["dropped"]:
                    L_.append("  no signature profile: " + ", ".join(
                        it._trunc(rd_names.get(m, str(m)), 32) for m in r["dropped"]))
            L_.append(f"### {ctx.dom_names[d]}: top {len(r['rows'])} codes by EIG")
            L_.append("| code | EIG (bits) | if present, favours | prior → posterior |")
            L_.append("|---|--:|---|--:|")
            for v, e_, m, p0, p1 in r["rows"]:
                L_.append(f"| {code_name(d, v)} | {e_:.3f} | "
                          f"{it._trunc(rd_names.get(m, str(m)), 32)} | "
                          f"{100 * p0:.0f}% → {100 * p1:.0f}% |")
            if d == 0 and not include_own and excl:
                own_v = sorted(excl)
                L_.append("own codes (excluded above): " + " · ".join(
                    code_name(0, v) for v in own_v[:12]))
        L_.append("")
    if not sets:
        L_.append("(no readout-DAG node matched with >= 2 label-node members)")
    return "\n".join(L_).rstrip() + "\n"

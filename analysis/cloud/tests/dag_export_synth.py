"""Synthetic DAG-gated runs for the export tests, the committed browser fixture and the
size measurement. SYNTHETIC ONLY: invented node ids, names, terms and document counts;
no patient data, no real model.

A run dir is written as the fit driver writes one (`dag_gated_fit.py`): `model/result`
(`spark_vi.io.export.save_result`, the node set in its metadata), `label_dag.json`,
`bundle_meta.json`, `manifest.json`, `node_statistics.tsv` (the model AS FITTED, before
the drop), plus a Mondo-style `mondo_nodes.tsv` in a cache dir and a concept-names CSV.

  # regenerate the committed fixture (deterministic, byte-identical)
  PYTHONPATH=.:spark-vi:charmpheno python analysis/cloud/tests/dag_export_synth.py \
      --fixture dag-dashboard/fixture
  # measure the export at the real scale (2,895 topics; 5000/5000/1601 terms)
  PYTHONPATH=.:spark-vi:charmpheno python analysis/cloud/tests/dag_export_synth.py \
      --measure /tmp/scale_run
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_CLOUD = _HERE.parent.parent
for _p in (str(_CLOUD), str(_CLOUD.parent.parent / "spark-vi"),
           str(_CLOUD.parent.parent / "charmpheno"), str(_CLOUD.parent.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FIXTURE_CREATED_UTC = "2026-11-11T00:00:00Z"
DOMAINS = ("condition", "drug", "measurement")
MONDO_VERSION = "2026-06-02"

# --------------------------------------------------------------------------- #
# The fixture's label DAG: ~40 invented disease nodes, multi-parent in places. #
# --------------------------------------------------------------------------- #
# (local key, name, [parent keys])
_FIXTURE_NODES = [
    ("cv", "cardiovascular disorder", []),
    ("resp", "respiratory system disorder", []),
    ("met", "metabolic disorder", []),
    ("renal", "kidney disorder", []),
    ("heart", "heart disorder", ["cv"]),
    ("vasc", "vascular disorder", ["cv"]),
    ("arrh", "cardiac arrhythmia", ["heart"]),
    ("af", "atrial fibrillation", ["arrh"]),
    ("af_parox", "paroxysmal atrial fibrillation", ["af"]),
    ("brady", "sinus bradycardia", ["arrh"]),
    ("hf", "heart failure", ["heart"]),
    ("hfref", "heart failure with reduced ejection fraction", ["hf"]),
    ("hfpef", "heart failure with preserved ejection fraction", ["hf"]),
    ("valve", "heart valve disorder", ["heart"]),
    ("aortic_st", "aortic valve stenosis", ["valve"]),
    ("htn", "hypertensive disorder", ["vasc"]),
    ("ess_htn", "essential hypertension", ["htn"]),
    ("pulm_htn", "pulmonary hypertension", ["vasc", "resp"]),
    ("pad", "peripheral artery disease", ["vasc"]),
    ("airway", "lower airway disorder", ["resp"]),
    ("asthma", "asthma", ["airway"]),
    ("asthma_allergic", "allergic asthma", ["asthma"]),
    ("copd", "chronic obstructive pulmonary disease", ["airway"]),
    ("emph", "emphysema", ["copd"]),
    ("ild", "interstitial lung disease", ["resp"]),
    ("ipf", "idiopathic pulmonary fibrosis", ["ild"]),
    ("dm", "diabetes mellitus", ["met"]),
    ("t2dm", "type 2 diabetes mellitus", ["dm"]),
    ("t1dm", "type 1 diabetes mellitus", ["dm"]),
    ("dm_neph", "diabetic nephropathy", ["t2dm", "ckd"]),
    ("lipid", "lipid metabolism disorder", ["met"]),
    ("hyperchol", "hypercholesterolemia", ["lipid"]),
    ("fam_hyperchol", "familial hypercholesterolemia", ["hyperchol"]),
    ("metsyn", "metabolic syndrome", ["met", "cv"]),
    ("obes", "obesity disorder", ["met"]),
    ("ckd", "chronic kidney disease", ["renal"]),
    ("ckd3", "chronic kidney disease stage 3", ["ckd"]),
    ("ckd5", "chronic kidney disease stage 5", ["ckd"]),
    ("aki", "acute kidney injury", ["renal"]),
    ("cardiorenal", "cardiorenal syndrome", ["hf", "ckd"]),
    ("gout", "gout", ["met", "renal"]),
]
# Documents labelled with each node (synthetic). Pruned: < MIN_DOCS reach the node;
# spliced: `af` has exactly one kept child (`af_parox`) and no documents of its own.
_FIXTURE_DOCS = {
    "af_parox": 140, "brady": 35, "hfref": 90, "hfpef": 60, "aortic_st": 7,
    "ess_htn": 300, "pulm_htn": 45, "pad": 12, "asthma": 110, "asthma_allergic": 4,
    "copd": 80, "emph": 50, "ipf": 9, "ild": 3, "t2dm": 260, "t1dm": 40,
    "dm_neph": 55, "hyperchol": 150, "fam_hyperchol": 6, "metsyn": 70, "obes": 120,
    "ckd3": 85, "ckd5": 15, "aki": 64, "cardiorenal": 33, "gout": 41, "hf": 25,
    "valve": 30,
}
MIN_DOCS = 25
# Kept nodes dropped after the fit as unsupported (one with a kept child, `ild` is
# pruned already; `copd` keeps `emph` below it).
_FIXTURE_UNSUPPORTED = ("copd", "brady")

_ADJ = ["Chronic", "Acute", "Recurrent", "Mild", "Severe", "Secondary", "Primary",
        "Persistent", "Transient", "Early-onset"]
_ORGAN = ["renal", "cardiac", "pulmonary", "hepatic", "vascular", "metabolic",
          "endocrine", "respiratory", "bronchial", "arterial", "venous", "coronary"]
_NOUN = ["insufficiency", "finding", "inflammation", "disorder", "obstruction",
         "dysfunction", "lesion", "syndrome", "complication", "abnormality"]
_DRUG_STEM = ["lorva", "metfo", "ateno", "furose", "salbu", "lisino", "amlo", "spiro",
              "insu", "warfa", "apixa", "predni", "tiotro", "budeso", "allopu",
              "empagli", "carve", "rosuva", "hydrochlo", "losa"]
_DRUG_END = ["statin", "rmin", "lol", "mide", "tamol", "pril", "dipine", "nolactone",
             "lin", "rin", "ban", "sone", "pium", "nide", "rinol", "flozin"]
_ANALYTE = ["Creatinine", "Hemoglobin A1c", "LDL cholesterol", "Potassium",
            "Sodium", "BNP", "Troponin I", "eGFR", "Glucose", "Uric acid", "ALT",
            "Albumin", "FEV1", "Oxygen saturation", "Platelets", "Triglycerides"]
_STATES = [(1, "low"), (2, "normal"), (3, "high"), (0, "measured")]


def _term_tables(sizes, rng):
    """Per domain: codes (ints) and names, invented."""
    codes, names = [], []
    # condition
    V = sizes[0]
    cs, ns_ = [], []
    for i in range(V):
        cs.append(3100000 + 17 * i)
        ns_.append(f"{_ADJ[i % len(_ADJ)]} {_ORGAN[(i // 3) % len(_ORGAN)]} "
                   f"{_NOUN[(i // 7) % len(_NOUN)]}" + (f" {i // 840 + 1}" if i >= 840
                                                        else ""))
    codes.append(cs); names.append(ns_)
    V = sizes[1]
    cs, ns_ = [], []
    for i in range(V):
        cs.append(1100000 + 23 * i)
        stem = _DRUG_STEM[i % len(_DRUG_STEM)] + _DRUG_END[(i // 5) % len(_DRUG_END)]
        dose = [5, 10, 20, 40, 100, 250, 500][(i // 3) % 7]
        form = ["Oral Tablet", "Oral Capsule", "Injection", "Inhalant"][(i // 11) % 4]
        ns_.append(f"{stem.capitalize()} {dose} MG {form}"
                   + (f" ({i // 2000 + 1})" if i >= 2000 else ""))
    codes.append(cs); names.append(ns_)
    V = sizes[2]
    cs, ns_ = [], []
    for i in range(V):
        a = i // len(_STATES)
        st, lab = _STATES[i % len(_STATES)]
        cid = 3000000 + 31 * a
        cs.append(cid * 100 + st)
        ana = _ANALYTE[a % len(_ANALYTE)] + (f" (panel {a // len(_ANALYTE)})"
                                             if a >= len(_ANALYTE) else "")
        ns_.append(f"{ana} [{lab}]")
    codes.append(cs); names.append(ns_)
    return codes, names


def _lambda(K, n_bg, sizes, rng, *, groups=None, eta=0.05):
    """Heavy-tailed per-domain λ: each topic writes a Zipf-weighted, mostly topic-own
    set of terms (a group's terms when `groups` maps topic -> group) on top of the
    prior η, with lognormal masses; background topics spread wide."""
    lam = {}
    for m, V in enumerate(sizes):
        pop = 1.0 / np.arange(1, V + 1) ** 0.9            # term popularity (Zipf)
        A = np.full((K, V), eta)
        for k in range(K):
            mass = float(np.exp(rng.normal(6.5 if k >= n_bg else 9.0, 1.2)))
            mass *= [1.0, 0.5, 0.3][m % 3]
            n_terms = int(min(V, rng.integers(60, 600) if k >= n_bg else V // 2))
            if groups is not None and k in groups:
                g_rng = np.random.default_rng(1000 * m + groups[k])
                own = g_rng.choice(V, size=min(V, n_terms), replace=False, p=pop / pop.sum())
                own = own[: max(10, n_terms // 2)]
                extra = rng.choice(V, size=n_terms - own.size, replace=False,
                                   p=pop / pop.sum()) if n_terms > own.size else []
                support = np.unique(np.concatenate([own, np.asarray(extra, int)]))
            else:
                support = rng.choice(V, size=n_terms, replace=False, p=pop / pop.sum())
            w = rng.pareto(1.3, size=support.size) + 1e-3
            w *= pop[support] ** 0.5
            A[k, support] += mass * w / w.sum()
        lam[m] = A
    return lam


def _drop_topics(lam, ns_before, ns_after, n_bg, tpn):
    """λ subset to the surviving topics, background first then the surviving blocks in
    the NEW engine order (as `DagGatedLDAModel._without_nodes`)."""
    keep = list(range(n_bg))
    for u in ns_after.nodes:
        lo = n_bg + (ns_before.engine_id[u] - 1) * tpn
        keep.extend(range(lo, lo + tpn))
    idx = np.asarray(keep)
    return {m: a[idx] for m, a in lam.items()}


def write_run(run, *, dag, ns_fitted, ns, lam, n_bg, tpn, vocab_codes, node_names,
              term_names, mondo_cache, domains=DOMAINS, cdr="synthetic-project.synthetic_cdr"):
    """A run dir as the fit driver writes it (the pieces the exporter reads)."""
    from spark_vi.core.result import VIResult
    from spark_vi.io.export import save_result
    from spark_vi.eval.topic.sharpness import support_fraction
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    K = n_bg + len(ns.nodes) * tpn
    res = VIResult(global_params={"lambda": lam, "alpha": np.full(K, 0.1)},
                   elbo_trace=[], n_iterations=1, converged=True,
                   metadata={"model_class": "DagGatedOnlineLDA", "K": K,
                             "dag_node_set": {**ns.to_dict(), "n_bg": n_bg, "tpn": tpn}})
    save_result(res, str(run / "model" / "result"))
    edges = []
    for n in sorted(dag):
        ps = sorted(dag[n])
        if not ps:
            edges.append([n, None])
        edges.extend([n, p] for p in ps)
    (run / "label_dag.json").write_text(json.dumps(
        {"dag": edges, "labels_col": "labels", "node_set": ns.to_dict()}))
    vocab_maps = [{str(c): i for i, c in enumerate(cs)} for cs in vocab_codes]
    kept_names = {"-1": "root"}
    for u in ns.nodes:
        kept_names[str(int(u.split(":")[1]))] = node_names[u]
    (run / "bundle_meta.json").write_text(json.dumps(
        {"vocab_maps": vocab_maps, "name_by_id": kept_names, "label_dag": edges}))
    (run / "manifest.json").write_text(json.dumps({
        "model_class": "gated_pc", "n_bg": n_bg, "tpn": tpn, "K": K,
        "domain_names": list(domains),
        "node_statistics": {"file": "node_statistics.tsv", "dropped": True},
        "corpus_manifest": {"cdr": cdr, "billing": "synthetic-billing",
                            "min_patient_count": 20,
                            "mondo_version": MONDO_VERSION,
                            "mondo_cache_dir": str(mondo_cache)}}))
    # node statistics of the model AS FITTED (support per kept-before-drop node)
    lines = ["node\tdepth\tsupport\tsupportFraction\tmass\tmassToParent\tflat"]
    for u in ns_fitted.nodes:
        lines.append(f"{u}\t1\t{ns_fitted.support[u]}\t0.5\t1\t\tfalse")
    (run / "node_statistics.tsv").write_text("\n".join(lines) + "\n")
    # topic NPMI as topic_npmi.py writes it (two scored domains; a few unscored topics)
    nrng = np.random.default_rng(K)
    rows = []
    for d in (0, 1):
        for k in range(K):
            g = float(nrng.normal(0.2, 0.08)) if k % 17 else float("nan")
            rows.append({"domain": d, "topic": k, "kind": "bg" if k < n_bg else "node",
                         "npmi_gated": g, "npmi_all": g - 0.03})
    (run / "npmi_top10.json").write_text(json.dumps(
        {"top_n": 10, "split": "train", "min_pair_count": 20, "rows": rows}))
    Path(mondo_cache).mkdir(parents=True, exist_ok=True)
    with open(Path(mondo_cache) / f"{MONDO_VERSION}_mondo_nodes.tsv", "w") as fh:
        fh.write("id\tcategory\tname\n")
        for n in sorted(node_names):
            fh.write(f"{n}\tbiolink:Disease\t{node_names[n]}\n")
    with open(run / "concept_names.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["concept_id", "concept_name"])
        for cs, ns_ in zip(vocab_codes, term_names):
            w.writerows(zip(cs, ns_))
    return run


def fixture_run(root, *, seed=0):
    """The ~40-node fixture run (all four statuses, multi-parent nodes, 3 domains)."""
    from spark_vi.dag.nodeset import drop_nodes, fit_node_set
    rng = np.random.default_rng(seed)
    ids = {key: f"MONDO:99{1000 + i:05d}" for i, (key, _, _) in enumerate(_FIXTURE_NODES)}
    dag = {ids[k]: [ids[p] for p in ps] for k, _, ps in _FIXTURE_NODES}
    names = {ids[k]: nm for k, nm, _ in _FIXTURE_NODES}
    labels = []
    for k, n in _FIXTURE_DOCS.items():
        labels.extend([[ids[k]]] * n)
    ns_fit = fit_node_set(dag, labels, min_docs_per_node=MIN_DOCS)
    ns = drop_nodes(ns_fit, [ids[k] for k in _FIXTURE_UNSUPPORTED])
    n_bg, tpn = 3, 1
    sizes = (600, 400, 160)
    K_fit = n_bg + len(ns_fit.nodes) * tpn
    groups = {n_bg + e - 1: e for e in range(1, len(ns_fit.nodes) + 1)}
    lam = _drop_topics(_lambda(K_fit, n_bg, sizes, rng, groups=groups), ns_fit, ns,
                       n_bg, tpn)
    codes, tnames = _term_tables(sizes, rng)
    root = Path(root)
    return write_run(root / "0999-synthetic-fixture", dag=dag, ns_fitted=ns_fit, ns=ns,
                     lam=lam, n_bg=n_bg, tpn=tpn, vocab_codes=codes, node_names=names,
                     term_names=tnames, mondo_cache=root / "mondo")


def scale_run(root, *, n_bg=200, n_kept=2695, n_given=7000, sizes=(5000, 5000, 1601),
              seed=1):
    """A run at the real scale: n_bg + n_kept topics (2,895), a given DAG of `n_given`
    nodes (pruned, spliced and unsupported among them), heavy-tailed λ rows."""
    from spark_vi.dag.nodeset import drop_nodes, learn_node_set
    rng = np.random.default_rng(seed)
    # a random multi-parent DAG: node i's parents among earlier nodes
    ids = [f"MONDO:{1000000 + 7 * i:07d}" for i in range(n_given)]
    dag = {}
    for i, n in enumerate(ids):
        if i < 30:
            dag[n] = []
            continue
        k = 1 if rng.random() < 0.6 else int(rng.integers(2, 4))
        ps = set(int(x) for x in rng.integers(max(0, i - 400), i, size=k))
        dag[n] = [ids[p] for p in sorted(ps)]
    # supports: monotone up (sum over children-ish), heavy-tailed at the leaves
    children = {n: [] for n in ids}
    for c, ps in dag.items():
        for p in ps:
            children[p].append(c)
    own = {n: int(rng.pareto(1.1) * 30) for n in ids}
    sup = {}
    for n in reversed(ids):
        cs = sorted((sup[c] for c in children[n]), reverse=True)
        sup[n] = own[n] + (cs[0] + sum(cs[1:]) // 4 if cs else 0)
    # choose min_docs so about n_kept + some survive
    vals = np.sort(np.asarray(list(sup.values())))[::-1]
    min_docs = int(vals[min(len(vals) - 1, n_kept + 250)]) + 1
    ns_fit = learn_node_set(dag, sup, n_docs=100000, min_docs_per_node=min_docs)
    n_drop = max(0, len(ns_fit.nodes) - n_kept)
    drop = list(rng.choice(np.asarray(ns_fit.nodes, dtype=object), size=n_drop,
                           replace=False)) if n_drop else []
    ns = drop_nodes(ns_fit, drop)
    tpn = 1
    K_fit = n_bg + len(ns_fit.nodes)
    lam = _drop_topics(_lambda(K_fit, n_bg, sizes, rng), ns_fit, ns, n_bg, tpn)
    codes, tnames = _term_tables(sizes, rng)
    words = ["syndrome", "disease", "disorder", "deficiency", "type", "familial",
             "congenital", "autosomal", "dominant", "recessive", "acquired", "chronic",
             "neoplasm", "carcinoma", "infection", "of", "the", "with", "and", "due to"]
    names = {n: " ".join(words[int(j)] for j in rng.integers(0, len(words),
                                                              size=int(rng.integers(2, 7))))
             for n in ids}
    root = Path(root)
    return write_run(root / "0998-synthetic-scale", dag=dag, ns_fitted=ns_fit, ns=ns,
                     lam=lam, n_bg=n_bg, tpn=tpn, vocab_codes=codes, node_names=names,
                     term_names=tnames, mondo_cache=root / "mondo")


def export_run(run, out, *, created_utc=None):
    """Run the driver's pipeline on a synthetic run dir (names from its files)."""
    import export_dag_gated as drv
    argv = ["--run-dir", str(run), "--out", str(out),
            "--names", str(Path(run) / "concept_names.csv")]
    if created_utc:
        argv += ["--created-utc", created_utc]
    return drv.main(argv)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", help="write the fixture export to this dir")
    ap.add_argument("--measure", help="build the scale run under this dir and export it")
    args = ap.parse_args(argv)
    import tempfile
    if args.fixture:
        with tempfile.TemporaryDirectory() as td:
            run = fixture_run(td)
            export_run(run, args.fixture, created_utc=FIXTURE_CREATED_UTC)
    if args.measure:
        run = scale_run(args.measure)
        export_run(run, Path(run) / "dag_export")


if __name__ == "__main__":
    main()

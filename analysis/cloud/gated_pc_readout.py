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
import os
from pathlib import Path

import numpy as np

from _driver_common import _phase, configure_logging, make_spark_session
from disk_telemetry import start_disk_telemetry
from gated_pc_cloud import (
    _DRIVER_READOUT_MAX_C, _MONDO_DAG_SOURCES, _collect_head_proba,
    _collect_lean_proba, _collect_theta_labels, _dump_partial_results,
    distributed_score_arm, format_arm_readout, multidomain_cache_key,
    multidomain_load_or_build, readout_ab_report, readout_from_proba,
    resolve_readout_mode, score_arm,
)

# What the batched solve got before fits recorded their own cap. Runs older than
# `readout_max_iter` in the manifest were, in practice, record runs at the driver's
# own default of 200 — a DEV smoke from that era has to be recovered with an
# explicit --readout-max-iter 60, which is exactly the retyping the manifest field
# removes for every fit written since.
_LEGACY_READOUT_MAX_ITER = 200


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
    p.add_argument("--dag-collapse", choices=["on", "off"], default=None,
                   help="mondo runs: whether the fit used the exp-0109 "
                        "splice-to-fixpoint DAG reduction (--dag-collapse). It is a "
                        "cache-KEY input and it changes the DAG a rebuild produces, "
                        "so a wrong value misses the cache and would rebuild a "
                        "different corpus (the drift gate catches it). Default: the "
                        "manifest's, else off — every fit predating the flag.")
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
                              dag_collapse=None) -> dict:
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
        # The DOC UNIT (R5.3 / audit seam 4). A cache-key input as of this
        # change, and the default is what EVERY manifest written before the field
        # existed means — every corpus in the repo was assembled under it — so an
        # old run still recomputes its own byte-identical key. There is no CLI
        # override: unlike the Mondo build inputs, the doc unit is not something a
        # rebuild can be told to differ on, because the assembler hard-codes it.
        "doc_spec": str(_pick("doc_spec", DEFAULT_DOC_SPEC)),
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
                theta_topm=None, readout_max_iter=None):
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
    solve completes."""
    C = int(manifest["C"])
    mode = resolve_readout_mode(readout_mode, C)
    K = int(manifest.get("K") or 0)
    if mode == "distributed" and not K:
        # Only the batched solver needs K, so only it pays for the peek.
        K = int(train_scored.select("topicDistribution").head()[0].size)
    print(f"[readout]   readout_mode={readout_mode} -> {mode} (C={C}, "
          f"K={K or 'n/a'}, driver-collect ceiling C<={_DRIVER_READOUT_MAX_C})",
          flush=True)
    ab = bool(ab_check) and mode == "distributed"
    if bool(ab_check) and not ab:
        print("[readout]   readout_ab_check ignored (readout_mode resolved to driver "
              "— there is nothing to compare against)", flush=True)
    elif ab and C > _DRIVER_READOUT_MAX_C:
        print(f"[readout]   readout_ab_check SKIPPED: C={C} exceeds the driver path's "
              f"own ceiling ({_DRIVER_READOUT_MAX_C}); the gate is meant to run at "
              "cardiovascular scale (C=444)", flush=True)
        ab = False

    def _dump(results):
        if out_dir is not None:
            _dump_partial_results(Path(out_dir), results,
                                  name="results_readout.json")

    results = {}
    dist = None
    if mode == "distributed":
        # No theta collect: the per-node LRs are fit on the executors and only the
        # lean test-split eval bundle comes back. theta_topm defaults to the
        # MANIFEST's value: a re-readout reproduces the fit's own design matrix —
        # silently refitting a top-m run at full K would change the estimator
        # (and crawl at whole-Mondo, the scale where top-m is on). An explicit
        # override (--readout-theta-topm) exists for the truncation-PRICING
        # experiment: force top-m onto a recorded full-K run and read the delta.
        if theta_topm is None:
            theta_topm = int(manifest.get("readout_theta_topm", 0) or 0)
            if theta_topm:
                print(f"[readout]   theta top-m={theta_topm} (from manifest)",
                      flush=True)
        else:
            theta_topm = int(theta_topm)
            print(f"[readout]   theta top-m={theta_topm} (CLI OVERRIDE — deltas vs "
                  "the recorded run price the truncation)", flush=True)
        # Only the batched solver has an iteration cap, so this is the only place
        # the resolution matters — and the log line lands right where the number is
        # about to be spent, next to top-m's.
        readout_max_iter, _mi_src = resolve_readout_max_iter(readout_max_iter, manifest)
        print(f"[readout]   readout max_iter={readout_max_iter} (from {_mi_src})",
              flush=True)
        dist = distributed_score_arm(
            train_scored, test_scored, C, K, recall_targets=recall_targets,
            fdr_targets=fdr_targets, min_count=min_count, label="gated_pc",
            theta_topm=theta_topm, max_iter=readout_max_iter,
            # Same dir as `results_readout.json`, and for the same reason one step
            # earlier in the pipeline: this tool IS the recovery path, and its solve
            # is the multi-hour part. A death mid-solve (the 08-28 preemption-wave
            # job abort cost 9,112s) now costs at most `checkpoint_every`
            # iterations, because the next invocation finds the checkpoint next to
            # the manifest it already reads. `None` when no out_dir was given —
            # nowhere durable to put it.
            checkpoint_dir=out_dir)
        results["gated_pc"] = dist[0]
    else:
        Pi_tr, y_tr, m_tr, _ = _collect_theta_labels(train_scored, C)
        Pi_te, y_te, m_te, _ = _collect_theta_labels(test_scored, C)
        results["gated_pc"] = score_arm(
            Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te, C, recall_targets=recall_targets,
            fdr_targets=fdr_targets, min_count=min_count)
    _dump(results)
    print(format_arm_readout("gated_pc (pc_topics_lr)", results["gated_pc"]),
          flush=True)
    if ab:
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
        print(f"[readout]   co-fit head arm SKIPPED (weight_y={weight_y:g}, "
              f"probability column "
              f"{'present' if 'probability' in test_scored.columns else 'absent'}): "
              "an unsupervised gate has no co-fit head to read out.", flush=True)
        return results
    if mode == "distributed":
        # No LR involved — `probability` IS the per-doc (C,) P(node) — so the
        # distributed variant is just the lean collector over that column.
        hp, hy, hm, _ = _collect_lean_proba(test_scored, C, score_col="probability")
    else:
        hp, hy, hm = _collect_head_proba(test_scored, C)
    results["gated_pc_head"] = readout_from_proba(
        hp, hy, hm, C, recall_targets=recall_targets, fdr_targets=fdr_targets,
        min_count=min_count)
    _dump(results)
    print(format_arm_readout("gated_pc (co-fit head)", results["gated_pc_head"]),
          flush=True)
    return results


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    run_dir = resolve_run_dir(args.run_dir)
    print(f"[readout]   run dir: {run_dir}", flush=True)
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
                     dag_collapse=(None if args.dag_collapse is None
                                   else args.dag_collapse == "on"))
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
            log=lambda msg: print(f"[readout] {msg}", flush=True))

        from _case_finding_cache import try_load

        with _phase("reload cached bundle"):
            spec = None
            if args.bundle_path:
                base = args.bundle_path.rstrip("/")
                cache_uri, key = base.rsplit("/", 1)
            else:
                if not cache_uri:
                    print("[readout] ERROR: pass --cache-uri or --bundle-path.",
                          flush=True)
                    return 1
                spec = corpus_spec_from_manifest(
                    manifest, doc_min_length=args.doc_min_length, **spec_over)
                if native_spec_mismatch(spec, manifest):
                    print("[readout] ERROR: the saved fit is an exp-0110 NATIVE "
                          "Mondo run (corpus_manifest.dag_source=mondo_native), "
                          "but the rebuild spec resolved dag_source="
                          f"{spec['dag_source']!r} — that keys, and on the MISS "
                          "rebuilds, a different label space. Drop the override, "
                          "or pass --dag-source mondo_native.", flush=True)
                    return 2
                if mondo_spec_mismatch(spec, manifest):
                    print("[readout] ERROR: the saved fit's label space is MONDO "
                          "ids, but the rebuild spec resolved dag_source="
                          f"{spec['dag_source']!r} — this manifest predates "
                          "corpus_manifest recording the Mondo build inputs, so "
                          "the defaults would key (and on the MISS that wrong key "
                          "guarantees, rebuild) the WRONG corpus. Pass the fit's "
                          "own values from the experiment doc's front matter, "
                          "e.g. for exp 0104:", flush=True)
                    print("[readout]   --dag-source mondo --mondo-version "
                          "2026-06-02 --min-positives 100", flush=True)
                    return 2
                key = bundle_key_from_manifest(
                    manifest, doc_min_length=args.doc_min_length, **spec_over)
            bundle = try_load(spark, cache_uri, key)
            if bundle is None and (args.no_rebuild or spec is None):
                # --no-rebuild, or a --bundle-path whose dir does not hold a bundle
                # (there is no spec to rebuild FROM in that case).
                print(f"[readout] ERROR: bundle cache MISS at {cache_uri}/{key}. "
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
                print("[readout] cache MISS — rebuilding bundle from manifest "
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
                            print(f"[readout]   {field} not in the manifest; "
                                  f"using {env_var} from the environment (the "
                                  "same source the fit used)", flush=True)
                        else:
                            print(f"[readout] ERROR: a rebuild needs {field}, "
                                  "but the manifest predates recording it and "
                                  f"{env_var} is unset"
                                  + (f" — pass {flag}." if flag else
                                     " — source .workspace_env (make setup)."),
                                  flush=True)
                            return 2
                print(f"[readout]   corpus: dag_source={spec['dag_source']} "
                      f"extra_domains={spec['extra_domains']} "
                      f"index_mode={spec['index_mode']} min_n={spec['min_n']} "
                      f"mondo_branch={spec['mondo_branch'] or 'ALL'} "
                      f"min_positives={spec['min_positives']} "
                      f"dag_collapse={spec['dag_collapse']}", flush=True)
                bundle = rebuild_bundle(
                    spark, spec,
                    cache_uri=(cache_uri if args.cache_write == "on" else None))
                print(f"[readout]   bundle REBUILT"
                      + (f" and written to {cache_uri}/{key} (the next readout of "
                         "this run is a HIT)" if args.cache_write == "on"
                         else " (--cache-write off: not persisted)")
                      + f"; C={C}", flush=True)
            else:
                print(f"[readout]   bundle loaded ({cache_uri}/{key}); C={C}",
                      flush=True)

        with _phase("reconstruct model + transform"):
            # The theta collect (driver mode) now happens inside run_readout, so the
            # distributed mode can skip it entirely rather than paying for it here.
            model = reconstruct_model(run_dir, manifest)
            # DRIFT GATE, before a single number is computed: the corpus in hand
            # (cached or rebuilt) must be the one this λ/head was fit against.
            drift = bundle_drift_report(
                bundle, manifest, lambda_vocab_sizes(model.result.global_params))
            if drift:
                print("[readout] ERROR: the corpus does not match the saved fit — "
                      "it has DRIFTED since the fit (a data refresh, or an "
                      "assembly-source change). Refusing to score:", flush=True)
                for line in drift:
                    print(f"[readout]     - {line}", flush=True)
                print("[readout]   Pass --bundle-path at the bundle this run was "
                      "actually fit against, or re-fit.", flush=True)
                return 3
            train_scored = model.transform(bundle.train_df).cache()
            test_scored = model.transform(bundle.test_df).cache()

        with _phase("score"):
            run_readout(train_scored, test_scored, manifest, recall_targets=rt,
                        fdr_targets=ft, min_count=min_count,
                        readout_mode=args.readout_mode,
                        ab_check=args.readout_ab_check, out_dir=run_dir,
                        theta_topm=args.readout_theta_topm,
                        readout_max_iter=args.readout_max_iter)
            print(f"[readout]   arm results written to "
                  f"{run_dir / 'results_readout.json'}", flush=True)
            train_scored.unpersist(); test_scored.unpersist()

            # Echo the other arms' stored summary (only gated_pc can be re-scored).
            # `results` is legitimately absent-or-null on a FIT-ONLY manifest:
            # `gated_pc_cloud` writes the npz + manifest as soon as the fit lands
            # (`partial="fit-only"`) so a readout death cannot cost the fit, which
            # is precisely the run this tool exists to rescue — so say what is
            # missing rather than echoing nothing, and never index into it.
            if manifest.get("partial"):
                print(f"[readout]   manifest marked partial="
                      f"{manifest['partial']!r}: the fit landed but its own "
                      "readout did not, so there are no stored arm results to "
                      "echo (the numbers above are this re-readout's).",
                      flush=True)
            for name, res in (manifest.get("results") or {}).items():
                if name in ("gated_pc", "gated_pc_head"):
                    continue
                rk = res.get("ranking", res)   # tolerate old flat-macro manifests
                auc = rk.get("auc"); ap = rk.get("ap")
                print(f"[readout]   (manifest) {name}: AUC="
                      f"{'n/a' if auc is None else f'{auc:.4f}'} "
                      f"AP={'n/a' if ap is None else f'{ap:.4f}'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

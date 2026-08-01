"""Experiment-tracking runner: reads a docs/experiments/NNNN-slug.md
record, merges defaults, dispatches lda_bigquery_cloud.py via spark-submit,
captures sanitized stdout to summary.md in the run dir, then runs eval.

See docs/superpowers/specs/2026-05-28-experiment-tracking-design.md.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "docs" / "experiments"
DEFAULTS_DIR = REPO_ROOT / "experiments" / "defaults"
COVARIATE_CACHE_MISS_EXIT = 42  # MUST match analysis/cloud/build_dashboard_cloud.py


def read_frontmatter(path: Path) -> dict:
    """Parse YAML frontmatter from a markdown file.

    Frontmatter is delimited by leading and trailing `---` lines on their own.
    Returns the parsed dict. Raises ValueError if absent or unterminated.
    """
    text = path.read_text()
    if not text.startswith("---\n"):
        raise ValueError(f"{path}: missing frontmatter block (expected leading '---')")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise ValueError(f"{path}: unterminated frontmatter block (no trailing '---')")
    yaml_text = text[4:end]
    return yaml.safe_load(yaml_text) or {}


def merge_config(base: dict, override: dict) -> dict:
    """Shallow merge: returns a new dict with override taking precedence over base."""
    out = dict(base)
    out.update(override)
    return out


def load_defaults(cohort: str, defaults_dir: Path) -> dict:
    """Load _base.yaml then <cohort>.yaml and merge.

    Raises FileNotFoundError if either file is missing.
    """
    base_path = defaults_dir / "_base.yaml"
    cohort_path = defaults_dir / f"{cohort}.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"missing defaults file: {base_path}")
    if not cohort_path.exists():
        raise FileNotFoundError(f"missing defaults file: {cohort_path}")
    base = yaml.safe_load(base_path.read_text()) or {}
    cohort_overrides = yaml.safe_load(cohort_path.read_text()) or {}
    return merge_config(base, cohort_overrides)


def _list_experiment_files(experiments_dir: Path) -> list[Path]:
    """All files matching NNNN-*.md in experiments_dir, sorted by id."""
    pattern = re.compile(r"^(\d{4})-.+\.md$")
    out = []
    for p in experiments_dir.iterdir():
        if pattern.match(p.name):
            out.append(p)
    out.sort(key=lambda p: p.name)
    return out


def find_next_pending(experiments_dir: Path) -> Path | None:
    """Return the lowest-id experiment file with status: pending, or None."""
    for p in _list_experiment_files(experiments_dir):
        try:
            fm = read_frontmatter(p)
        except ValueError:
            continue
        if fm.get("status") == "pending":
            return p
    return None


def find_by_id(experiments_dir: Path, exp_id: int) -> Path:
    """Return the experiment file with the given id. Raises FileNotFoundError if absent."""
    prefix = f"{exp_id:04d}-"
    for p in _list_experiment_files(experiments_dir):
        if p.name.startswith(prefix):
            return p
    raise FileNotFoundError(f"no experiment with id {exp_id:04d} in {experiments_dir}")


_RUN_DIR_PATTERN = re.compile(r"^(\d{4})-")


def find_most_recent_fit(runs_dir: Path) -> int | None:
    """Return the experiment id of the most-recently-checkpointed fit.

    Walks `runs_dir/<NNNN-slug>/manifest.json`, picks the one with the latest
    mtime, parses the leading NNNN from the directory name. Returns None if
    no checkpoint exists (empty / nonexistent runs_dir, or no manifest.json
    files in any subdirectory).

    The dir-naming convention `NNNN-slug` is enforced by `main()` when it
    creates save_dir; this function trusts that convention and skips any
    directory whose name doesn't start with four digits + hyphen.
    """
    if not runs_dir.is_dir():
        return None
    best_mtime = -1.0
    best_id: int | None = None
    for manifest in runs_dir.glob("*/manifest.json"):
        m = _RUN_DIR_PATTERN.match(manifest.parent.name)
        if m is None:
            continue
        mtime = manifest.stat().st_mtime
        if mtime > best_mtime:
            best_mtime = mtime
            best_id = int(m.group(1))
    return best_id


def find_most_recent_fit_needing_build(runs_dir: Path) -> int | None:
    """Return exp id of the most-recently-fit experiment whose dashboard
    bundle is missing or stale.

    Compares each `<NNNN-slug>/manifest.json` mtime against the same dir's
    `dashboard_bundle/corpus_stats.json` mtime (the last JSON written by
    build_dashboard_cloud before the zip step). If the marker is absent or
    older than the manifest, the fit is a build candidate. Among all
    candidates, returns the one with the latest manifest mtime — the
    freshest fit that still wants a build. Returns None if no candidates
    exist (all bundles current, or no fits at all).

    Caller invariant: dir-naming convention `NNNN-slug` is enforced by
    `main()` at save_dir creation; this function trusts it and skips any
    sibling whose name doesn't match.
    """
    if not runs_dir.is_dir():
        return None
    best_mtime = -1.0
    best_id: int | None = None
    for manifest in runs_dir.glob("*/manifest.json"):
        m = _RUN_DIR_PATTERN.match(manifest.parent.name)
        if m is None:
            continue
        marker = manifest.parent / "dashboard_bundle" / "corpus_stats.json"
        manifest_mtime = manifest.stat().st_mtime
        if marker.exists() and marker.stat().st_mtime >= manifest_mtime:
            continue
        if manifest_mtime > best_mtime:
            best_mtime = manifest_mtime
            best_id = int(m.group(1))
    return best_id


# Patterns that indicate per-patient row-level info. Belt-and-suspenders for
# driver-side stripping; if a new driver path re-introduces patient prints,
# these catch them at the wrapper boundary before they reach summary.md.
PATIENT_PATTERNS: list[re.Pattern] = [
    re.compile(r"person_hash", re.IGNORECASE),
    re.compile(r"person_id\s*=\s*\S+"),
    re.compile(r"\bhash:[0-9a-f]{6,}", re.IGNORECASE),
    re.compile(r"transform sample", re.IGNORECASE),  # the phase marker bracketing it
]

# Cluster log noise that has no signal value in the committed record.
# These match the standard log4j-style timestamp prefix that Spark, Hadoop,
# YARN, and the GCS connector emit. The wrapper drops them when writing to
# summary.md but still tees them to stdout so they're visible during live
# debugging.
NOISE_PATTERNS: list[re.Pattern] = [
    # log4j format: "YY/MM/DD HH:MM:SS LEVEL Logger: message"
    re.compile(r"^\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2} (INFO|WARN|DEBUG) "),
    # Bracketed CONTEXT ratelimit_period footers from GCS connector
    re.compile(r"\[CONTEXT ratelimit_period="),
]

# Composed: what summary.md drops. Callers pass DROP_PATTERNS at the sanitize
# boundary; the split between PATIENT_PATTERNS (privacy) and NOISE_PATTERNS
# (readability) is documentary.
DROP_PATTERNS: list[re.Pattern] = PATIENT_PATTERNS + NOISE_PATTERNS


def sanitize_line(line: str, patterns: list[re.Pattern]) -> str | None:
    """Return the line if safe to commit, or None to drop.

    Drops any line matching any patient-info pattern. Aggregate counts and
    topic-level prints (no per-patient identifiers) pass through.
    """
    for pat in patterns:
        if pat.search(line):
            return None
    return line


_ITER_MARKER = re.compile(r"^\[driver\]\s+iter\s+(\d+)/\d+:")


def parse_iter_marker(line: str) -> int | None:
    """Return the iter number from a `[driver]   iter N/M: ...` line, or None.

    Used by the SIGTERM trap to know which iter was last in flight so the
    killed marker can be informative.
    """
    m = _ITER_MARKER.match(line)
    return int(m.group(1)) if m else None


def build_lda_args(
    effective: dict, save_dir: Path, resume_from: Path | None,
) -> list[str]:
    """Build the CLI arg list for lda_bigquery_cloud.py from an effective config.

    The driver's argparse defaults handle anything not set here; we explicitly
    pass everything in `effective` to keep the cluster's behavior reproducible
    from the config alone.
    """
    args: list[str] = [
        "--save-dir", str(save_dir),
        "--save-interval", str(effective["save_interval"]),
        "--source-table", str(effective["source_table"]),
        "--doc-unit", str(effective["doc_unit"]),
        "--doc-min-length", str(effective["doc_min_length"]),
        "--K", str(effective["K"]),
        "--max-iter", str(effective["max_iter"]),
        "--vocab-size", str(effective["vocab_size"]),
        "--min-df", str(effective["min_df"]),
        "--min-patient-count", str(effective["min_patient_count"]),
        "--subsampling-rate", str(effective["subsampling_rate"]),
        "--tau0", str(effective["tau0"]),
        "--kappa", str(effective["kappa"]),
        "--print-topics-every", str(effective["print_topics_every"]),
        "--person-mod", str(effective["person_mod"]),
        "--top-n-tokens", str(effective["top_n_tokens"]),
        "--seed", str(effective["seed"]),
        # `cohort_def` is the driver-side argparse value (e.g. first_dementia_year);
        # `cohort` is the display id used for record-keeping and defaults lookup.
        "--cohort", str(effective["cohort_def"]),
        "--prior-obs-days", str(effective.get("prior_obs_days", 365)),
    ]
    if effective.get("cache_uri"):
        args += ["--corpus-cache-uri", str(effective["cache_uri"])]
    # BooleanOptionalAction in the driver: --optimize-doc-concentration / --no-...
    if effective.get("optimize_doc_concentration", True):
        args.append("--optimize-doc-concentration")
    else:
        args.append("--no-optimize-doc-concentration")
    if effective.get("optimize_topic_concentration", False):
        args.append("--optimize-topic-concentration")
    else:
        args.append("--no-optimize-topic-concentration")
    if resume_from is not None:
        args += ["--resume-from", str(resume_from)]
    return args


def validate_frontmatter(fm: dict) -> None:
    """Validate experiment frontmatter; exit(2) on missing/invalid fields."""
    required = ["id", "slug", "cohort", "model_class"]
    for key in required:
        if key not in fm:
            print(f"[run-exp] ERROR: frontmatter missing required field {key!r}",
                  flush=True)
            sys.exit(2)

    model_class = fm["model_class"]
    if model_class not in ("lda", "stm", "dag_placement", "multidomain"):
        print(f"[run-exp] ERROR: model_class {model_class!r} not supported "
              f"(currently: lda, stm, dag_placement, multidomain; hdp planned)",
              flush=True)
        sys.exit(2)

    if model_class == "stm":
        stm_required = ["covariate_formula", "categorical_cols", "continuous_cols"]
        for key in stm_required:
            if key not in fm:
                print(f"[run-exp] ERROR: model_class={model_class} requires "
                      f"frontmatter field {key!r}", flush=True)
                sys.exit(2)


def build_fit_driver_path(effective: dict) -> str:
    """Return path (relative to repo root) to the fit driver for this model_class."""
    model_class = effective.get("model_class", "lda")
    base = "analysis/cloud"
    if model_class == "lda":
        return f"{base}/lda_bigquery_cloud.py"
    if model_class == "stm":
        return f"{base}/stm_bigquery_cloud.py"
    if model_class == "dag_placement":
        return f"{base}/dag_placement_cloud.py"
    if model_class == "multidomain":
        return f"{base}/multidomain_cloud.py"
    raise ValueError(f"no fit driver for model_class={model_class!r}")


def build_fit_args(
    effective: dict, out_dir: str, resume_from: Path | None = None,
) -> list[str]:
    """Build argv for the fit driver, dispatching on model_class.

    resume_from (when set) is threaded to whichever driver runs so a re-run
    continues a prior checkpoint (LDA and STM both support --resume-from;
    --max-iter is then ADDITIONAL iterations on top of the loaded count).
    """
    model_class = effective.get("model_class", "lda")
    if model_class == "lda":
        return build_lda_args(effective, Path(out_dir), resume_from)
    if model_class == "stm":
        return build_stm_args(effective, out_dir, resume_from)
    if model_class == "dag_placement":
        return build_dag_placement_args(effective, out_dir, resume_from)
    if model_class == "multidomain":
        return build_multidomain_args(effective, out_dir, resume_from)
    raise ValueError(f"unknown model_class: {model_class!r}")


def _norm_cohort(c):
    """Normalize the 'none' sentinel / empty string to Python None so a
    checkpoint and config that mean 'no cohort' compare equal."""
    return None if c in (None, "none", "") else c


def _resume_corpus_mismatches(checkpoint_manifest: dict, effective: dict) -> list[str]:
    """Compare a checkpoint's corpus_manifest to the current effective config.

    Returns a list of human-readable mismatches on the fields that determine
    corpus MEMBERSHIP; empty list = compatible. Warm-starting onto a checkpoint
    whose corpus differs (e.g. a different person_mod sample or cohort lookback)
    silently trains on the wrong data, so the caller refuses to resume on any
    mismatch. Only fields PRESENT in the checkpoint are compared, so a
    checkpoint predating a field (e.g. prior_obs_days) is not penalized.

    vocab_size is deliberately NOT compared: STM stores the realized vocab
    count (post min_df/min_patient_count pruning) while config carries the
    CountVectorizer cap, so they legitimately differ for the same corpus. A
    genuine vocab-dimension change fails loudly at warm-start (the K x V lambda
    shape won't match), so it doesn't need a silent-corruption guard.
    """
    expected = {
        "person_mod": effective.get("person_mod"),
        "source_table": effective.get("source_table"),
        "cohort": _norm_cohort(effective.get("cohort_def", effective.get("cohort"))),
        "prior_obs_days": effective.get("prior_obs_days"),
    }
    mismatches: list[str] = []
    for key, want in expected.items():
        if key not in checkpoint_manifest:
            continue  # checkpoint predates this field; can't verify, don't block
        got = checkpoint_manifest[key]
        got_cmp = _norm_cohort(got) if key == "cohort" else got
        if got_cmp != want:
            mismatches.append(f"{key}: checkpoint={got!r} != config={want!r}")
    # doc_spec is stored as a manifest dict; compare just its spec name.
    ck_doc = checkpoint_manifest.get("doc_spec")
    if isinstance(ck_doc, dict) and "name" in ck_doc:
        want_doc = effective.get("doc_unit", "patient_year")
        if ck_doc["name"] != want_doc:
            mismatches.append(
                f"doc_spec: checkpoint={ck_doc['name']!r} != config={want_doc!r}"
            )
    if "topic_block_spec" in checkpoint_manifest:
        # Compare topic-block specs as plain dicts. This function runs in the
        # orchestration python, which has NO spark_vi (it ships only inside
        # spark-submit's py-files), so we must NOT import TopicBlockPartition
        # here — doing so crashed `make exp ID=N` resume with ModuleNotFoundError.
        # Mirror TopicBlockPartition.to_dict()'s shape:
        # {group_var, background_k, foreground: [[label, size], ...]}.
        ck_spec = checkpoint_manifest["topic_block_spec"] or None
        want_spec = None
        if effective.get("background_k") is not None and effective.get("foreground"):
            fg = [[lbl.strip(), int(sz)] for lbl, _, sz in
                  (piece.partition(":") for piece in
                   str(effective["foreground"]).split(","))]
            want_spec = {
                "group_var": str(effective.get("group_var", "source_cohort")),
                "background_k": int(effective["background_k"]),
                "foreground": fg,
            }

        def _norm(spec):
            # Normalize for comparison: foreground entries to lists, ints to
            # ints, so a JSON-loaded checkpoint (lists) and a config-built spec
            # compare structurally regardless of list/tuple provenance.
            if not spec:
                return None
            return (
                str(spec.get("group_var")),
                int(spec.get("background_k")),
                tuple((str(g), int(sz)) for g, sz in spec.get("foreground", [])),
            )

        if _norm(ck_spec) != _norm(want_spec):
            mismatches.append(
                f"topic_block_spec: checkpoint={ck_spec!r} != config={want_spec!r}")
    return mismatches


def check_resume_compat(save_dir: Path, effective: dict) -> None:
    """Exit(2) if resuming would warm-start onto a corpus that no longer
    matches the experiment config. No-op when the checkpoint has no readable
    corpus_manifest (nothing to verify)."""
    import json
    try:
        manifest = json.loads((save_dir / "manifest.json").read_text())
    except (OSError, ValueError):
        return
    corpus = (manifest.get("metadata") or {}).get("corpus_manifest")
    if not corpus:
        return
    mismatches = _resume_corpus_mismatches(corpus, effective)
    if mismatches:
        print("[run-exp] ERROR: refusing to resume — the checkpoint's corpus "
              "differs from the current experiment config:", flush=True)
        for m in mismatches:
            print(f"[run-exp]   - {m}", flush=True)
        print(f"[run-exp] Revert the config to match, or clear the checkpoint "
              f"dir to start fresh: rm -rf {save_dir}", flush=True)
        sys.exit(2)


def _require_workspace_env() -> tuple[str, str]:
    """Read the BigQuery CDR + billing project from the workspace environment.

    These are environment-specific (set by the workspace setup), never part of
    the committed experiment config — so they are sourced here, not from
    `effective`. The LDA driver reads the same two env vars itself; the STM
    drivers take them as --cdr/--billing args, so we resolve them here.
    """
    cdr = os.environ.get("WORKSPACE_CDR")
    billing = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not (cdr and billing):
        print("[run-exp] ERROR: WORKSPACE_CDR and GOOGLE_CLOUD_PROJECT must be set "
              "in env for STM runs (source the workspace env first).", flush=True)
        sys.exit(2)
    return cdr, billing


def build_covariates_args(effective: dict) -> list[str]:
    """Build argv for analysis/cloud/build_stm_covariates.py from an effective config."""
    cdr, billing = _require_workspace_env()
    args = [
        "--cdr", cdr,
        "--billing", billing,
        "--source-table", str(effective["source_table"]),
        "--person-mod", str(effective["person_mod"]),
        "--cache-uri", str(effective["cache_uri"]),
        "--covariate-formula", str(effective["covariate_formula"]),
        "--categorical-cols", ",".join(effective.get("categorical_cols", [])),
        "--continuous-cols", ",".join(effective.get("continuous_cols", [])),
    ]
    cohort_def = effective.get("cohort_def", effective.get("cohort", ""))
    if cohort_def and cohort_def != "none":
        args += ["--cohort", str(cohort_def)]
    args += ["--prior-obs-days", str(effective.get("prior_obs_days", 365))]
    # A gated experiment keys the covariate sidecar on (person_id, group) so
    # the gated dashboard prevalence can group by the label — pass the same
    # --group-var the fit driver uses (build_stm_args). Gating is signalled by
    # background_k + foreground being set.
    if effective.get("background_k") is not None and effective.get("foreground"):
        args += ["--group-var", str(effective.get("group_var", "source_cohort"))]
    if effective.get("known_sex_only"):
        args += ["--known-sex-only"]
    return args


def _dispatch_covariate_build(effective: dict, *, force: bool = False) -> int:
    """Run analysis/cloud/build_stm_covariates.py via spark-submit; return its rc.

    Factored out of the `--build-covariates-only` dispatch so
    `_build_only_with_auto_covariates` can reuse it (via a lambda) to
    auto-rebuild the covariate cache when `--build-only` reports a
    COVARIATE_CACHE_MISS_EXIT.
    """
    cov_script = REPO_ROOT / "analysis" / "cloud" / "build_stm_covariates.py"
    cov_args = build_covariates_args(effective)
    if force:
        cov_args.append("--force")
    cov_cmd = build_spark_submit_cmd(str(cov_script), cov_args, REPO_ROOT)
    print(f"[run-exp] build-covariates spark-submit: {' '.join(cov_cmd)}", flush=True)
    import subprocess as _sp
    result = _sp.run(cov_cmd, check=False)
    if result.returncode != 0:
        print(f"[run-exp] build-covariates exited non-zero ({result.returncode})",
              flush=True)
    return result.returncode


def _build_only_with_auto_covariates(*, effective, auto, force, run_build, dispatch_cov):
    """Run the dashboard build, with two covariate-cache behaviors folded in
    so both are covered by the same Spark-free injected-callable unit tests:

    1. Forced pre-build: if `force` and this is a gated-STM build with a
       cache_uri, rebuild the covariate cache FIRST (before the build ever
       runs). If that forced rebuild fails, abort immediately and return its
       rc — the build never runs against a rebuild that didn't happen.
    2. Miss-retry: if the build then exits COVARIATE_CACHE_MISS_EXIT and
       auto-rebuild is enabled for a gated STM build with a cache_uri,
       rebuild the covariate cache once (passing `force` through — after a
       forced pre-build a miss is unexpected, but the existing semantics of
       passing `force` are preserved) and re-run the build a single time.

    run_build() -> int and dispatch_cov(force: bool) -> int are injected so
    this is Spark-free and testable. Returns the final build return code."""
    if force and effective.get("model_class") == "stm" and effective.get("cache_uri"):
        print("[run-exp] --build-only --force-covariates: rebuilding covariate "
              "cache before the build", flush=True)
        pre_rc = dispatch_cov(True)
        if pre_rc != 0:
            print(f"[run-exp] forced covariate pre-build failed ({pre_rc}); "
                  "not running the dashboard build", flush=True)
            return pre_rc
    rc = run_build()
    if (rc == COVARIATE_CACHE_MISS_EXIT and auto
            and effective.get("model_class") == "stm" and effective.get("cache_uri")):
        print("[run-exp] --build-only: covariate cache MISS (exit %d); "
              "auto-rebuilding the covariate cache, then retrying the build once."
              % rc, flush=True)
        cov_rc = dispatch_cov(force)
        if cov_rc != 0:
            print(f"[run-exp] covariate rebuild failed ({cov_rc}); not retrying "
                  "the dashboard build", flush=True)
            return cov_rc
        print("[run-exp] covariate cache rebuilt; re-running the dashboard build",
              flush=True)
        rc = run_build()
    return rc


def build_stm_args(
    effective: dict, out_dir: str, resume_from: Path | None = None,
) -> list[str]:
    """Build argv for analysis/cloud/stm_bigquery_cloud.py."""
    cdr, billing = _require_workspace_env()
    common = [
        "--cdr", cdr,
        "--billing", billing,
        "--source-table", str(effective["source_table"]),
        "--doc-spec", str(effective.get("doc_unit", "patient_year")),
        "--doc-min-length", str(effective["doc_min_length"]),
        "--K", str(effective["K"]),
        "--max-iter", str(effective["max_iter"]),
        "--vocab-size", str(effective["vocab_size"]),
        "--min-df", str(effective["min_df"]),
        "--min-patient-count", str(effective["min_patient_count"]),
        "--subsampling-rate", str(effective["subsampling_rate"]),
        "--tau0", str(effective["tau0"]),
        "--kappa", str(effective["kappa"]),
        "--save-interval", str(effective["save_interval"]),
        "--print-topics-every", str(effective.get("print_topics_every", 10)),
        "--person-mod", str(effective["person_mod"]),
        "--cohort", str(effective.get("cohort_def", effective.get("cohort", ""))),
        "--prior-obs-days", str(effective.get("prior_obs_days", 365)),
        "--out-dir", str(out_dir),
    ]
    if effective.get("cache_uri"):
        common.extend(["--cache-uri", str(effective["cache_uri"])])
    if effective.get("random_seed") is not None:
        common.extend(["--random-seed", str(effective["random_seed"])])
    gating: list[str] = []
    if effective.get("background_k") is not None and effective.get("foreground"):
        gating = [
            "--background-k", str(effective["background_k"]),
            "--foreground", str(effective["foreground"]),
            "--group-var", str(effective.get("group_var", "source_cohort")),
        ]
    hardening: list[str] = []
    # reference_topic + spectral_init default to True (insight 0030). Emit the
    # explicit boolean so the driver receives the intended value, not its own default.
    ref = effective.get("reference_topic", True)
    hardening.append("--reference-topic" if ref else "--no-reference-topic")
    spec = effective.get("spectral_init", True)
    hardening.append("--spectral-init" if spec else "--no-spectral-init")
    # spectral_method defaults to "auto" (ADR 0037: dense below a vocab threshold,
    # scalable above). Emit the flag only for an explicit override; "auto" is the
    # driver default and stays implicit.
    method = effective.get("spectral_method", "auto")
    if method != "auto":
        hardening.extend(["--spectral-method", str(method)])
    if effective.get("spectral_d") is not None:
        hardening.extend(["--spectral-d", str(effective["spectral_d"])])
    if effective.get("spectral_min_doc_freq") is not None:
        hardening.extend(["--spectral-min-doc-freq", str(effective["spectral_min_doc_freq"])])
    if effective.get("min_pair_support") is not None:
        hardening.extend(["--min-pair-support", str(effective["min_pair_support"])])
    if effective.get("estimate_sigma_diagonal"):
        hardening.append("--estimate-sigma-diagonal")
    if effective.get("estimate_global_scale"):
        hardening.append("--estimate-global-scale")
    if "global_scale_step_cap" in effective:
        hardening.extend(["--global-scale-step-cap", str(effective["global_scale_step_cap"])])
    pin = effective.get("sigma_diagonal_pin")
    if pin is not None and float(pin) != 1.0:
        hardening += ["--sigma-diagonal-pin", str(float(pin))]
    if effective.get("known_sex_only"):
        hardening.append("--known-sex-only")
    return common + [
        "--covariate-formula", str(effective["covariate_formula"]),
        "--categorical-cols", ",".join(effective.get("categorical_cols", [])),
        "--continuous-cols", ",".join(effective.get("continuous_cols", [])),
        "--sigma-init", str(effective.get("sigma_init", 1.0)),
        "--sigma-ridge", str(effective.get("sigma_ridge", 1e-6)),
        "--lbfgs-max-iter", str(effective.get("lbfgs_max_iter", 50)),
        "--lbfgs-tol", str(effective.get("lbfgs_tol", 1e-4)),
    ] + hardening + gating + (["--resume-from", str(resume_from)] if resume_from is not None else [])


def build_dag_placement_args(
    effective: dict, out_dir: str, resume_from: Path | None = None,
) -> list[str]:
    """Build argv for analysis/cloud/dag_placement_cloud.py (gated-SVI case-finding).

    K is emergent (n_bg + surviving-DAG-nodes * tpn), so there is NO --K. Resume is
    unsupported (GatedLDAModel is not persistable in v1); resume_from is ignored.
    """
    cdr, billing = _require_workspace_env()
    args = [
        "--cdr", cdr,
        "--billing", billing,
        "--source-table", str(effective["source_table"]),
        "--person-mod", str(effective["person_mod"]),
        "--vocab-size", str(effective["vocab_size"]),
        "--min-df", str(effective["min_df"]),
        "--min-patient-count", str(effective["min_patient_count"]),
        "--doc-min-length", str(effective["doc_min_length"]),
        "--prior-obs-days", str(effective.get("prior_obs_days", 365)),
        "--window-days", str(effective.get("window_days", 365)),
        "--window-mode", str(effective.get("window_mode", "forward")),
        "--lookback-days", str(effective.get("lookback_days", 365)),
        "--label-window-days", str(effective.get("label_window_days", 365)),
        "--disease", str(effective.get("disease", "diabetes")),
        "--min-n", str(effective["min_n"]),
        "--holdout-frac", str(effective.get("holdout_frac", 0.2)),
        "--n-bg", str(effective["n_bg"]),
        "--tpn", str(effective["tpn"]),
        "--node-alpha-scale", str(effective.get("node_alpha_scale", 1.0)),
        "--transform-alpha-mode", str(effective.get("transform_alpha_mode", "fitted")),
        "--transform-alpha", str(effective.get("transform_alpha", 0.0)),
        "--transform-bg-weight", str(effective.get("transform_bg_weight", 0.5)),
        "--mini-batch-fraction", str(effective.get("mini_batch_fraction", 0.0)),
        "--learning-rate-tau0", str(effective.get("learning_rate_tau0", 1.0)),
        "--learning-rate-kappa", str(effective.get("learning_rate_kappa", 0.7)),
        "--max-iter", str(effective["max_iter"]),
        "--cavi-max-iter", str(effective.get("cavi_max_iter", 100)),
        "--cavi-tol", str(effective.get("cavi_tol", 1e-3)),
        "--init", str(effective.get("init", "random")),
        "--spectral-max-vocab", str(effective.get("spectral_max_vocab", 8000)),
        "--spectral-method", str(effective.get("spectral_method", "auto")),
        "--anchor-scope", str(effective.get("anchor_scope", "closure")),
        "--spectral-topo-order", str(effective.get("spectral_topo_order", "forward")),
        "--strip-mode", str(effective.get("strip_mode", "test_only")),
        # Per-iter top-terms logging reuses the shared _base.yaml knobs (STM parity).
        "--print-topics-every", str(effective.get("print_topics_every", 0)),
        "--top-n-tokens", str(effective.get("top_n_tokens", 8)),
        "--out-dir", str(out_dir),
    ]
    if effective.get("seed") is not None:
        args.extend(["--seed", str(effective["seed"])])
    if effective.get("cache_uri"):
        args.extend(["--cache-uri", str(effective["cache_uri"])])
    # Store-true flag: learn the asymmetric per-node alpha (optimizeDocConcentration).
    # node_alpha_scale is the initial alpha; the gated Newton step refines it.
    if effective.get("optimize_doc_concentration"):
        args.append("--optimize-doc-concentration")
    return args


def build_multidomain_args(
    effective: dict, out_dir: str, resume_from: Path | None = None,
) -> list[str]:
    """Build argv for analysis/cloud/multidomain_cloud.py (two-domain gated fit).

    Like dag_placement, K is EMERGENT (n_bg + surviving-DAG-nodes * tpn), so there
    is NO --K, and resume is unsupported (GatedLDAModel is not persistable in v1);
    resume_from is ignored (accepted for signature parity with the other builders).

    --seed is REQUIRED by the driver (insight 0070: the scalable spectral init is
    seed-fragile, so the seed must be a deliberate, recorded choice); it comes from
    _base.yaml's `seed`. omega / eta_per_domain are emitted ONLY when set, because
    the driver's None default routes to the shim's pre-multi-domain scalar default —
    passing an empty/degenerate list would instead assert a per-domain vector.
    """
    cdr, billing = _require_workspace_env()
    # Comma-string-or-list normalizer: a frontmatter value may be written as a
    # comma-string ("1.0,0.5") or a YAML list, and both must normalize to the
    # comma-string the driver's argparse expects. Used below by omega /
    # eta_per_domain (emitted only when set) AND obs_exclude_vocab (always
    # emitted -- a YAML list here must not silently str()-serialize to
    # "['PPI', 'SNOMED']", which would parse into garbage tuple entries and make
    # the vocabulary strip silently no-op). Defined BEFORE the args list so it is
    # bound before use, not just before the omega/eta_per_domain call sites.
    def _as_comma(v):
        if isinstance(v, (list, tuple)):
            return ",".join(str(x) for x in v)
        return str(v)
    args = [
        "--cdr", cdr,
        "--billing", billing,
        "--out-dir", str(out_dir),
        "--seed", str(effective["seed"]),
        "--source-table-cond", str(effective.get("source_table_cond", "condition_era")),
        "--source-table-drug", str(effective.get("source_table_drug", "drug_era")),
        "--person-mod", str(effective["person_mod"]),
        "--disease", str(effective.get("disease", "diabetes")),
        "--min-n", str(effective["min_n"]),
        "--holdout-frac", str(effective.get("holdout_frac", 0.2)),
        "--strip-mode", str(effective.get("strip_mode", "test_only")),
        "--doc-min-length", str(effective["doc_min_length"]),
        "--prior-obs-days", str(effective.get("prior_obs_days", 365)),
        "--window-days", str(effective.get("window_days", 365)),
        # Per-domain vocabulary controls (independent natural sizes; SP3b design).
        "--cond-vocab-size", str(effective.get("cond_vocab_size", 5000)),
        "--cond-min-df", str(effective.get("cond_min_df", 20)),
        "--cond-min-patient-count", str(effective.get("cond_min_patient_count", 20)),
        "--drug-vocab-size", str(effective.get("drug_vocab_size", 2000)),
        "--drug-min-df", str(effective.get("drug_min_df", 20)),
        "--drug-min-patient-count", str(effective.get("drug_min_patient_count", 20)),
        "--n-bg", str(effective["n_bg"]),
        "--tpn", str(effective["tpn"]),
        "--max-iter", str(effective["max_iter"]),
        "--cavi-max-iter", str(effective.get("cavi_max_iter", 100)),
        "--cavi-tol", str(effective.get("cavi_tol", 1e-3)),
        "--init", str(effective.get("init", "spectral")),
        "--spectral-max-vocab", str(effective.get("spectral_max_vocab", 8000)),
        "--spectral-method", str(effective.get("spectral_method", "auto")),
        "--anchor-scope", str(effective.get("anchor_scope", "closure")),
        "--spectral-topo-order", str(effective.get("spectral_topo_order", "forward")),
        "--min-peak-ratio", str(effective.get("min_peak_ratio", 5.0)),
        # Final per-domain topic dump: top-N tokens per topic (reuses the shared
        # _base.yaml top_n_tokens knob; 0 disables the dump).
        "--top-n-tokens", str(effective.get("top_n_tokens", 8)),
        "--domains", str(effective.get("domains", "drug_era")),
        "--window-mode", str(effective.get("window_mode", "forward")),
        "--lookback-days", str(effective.get("lookback_days", 365)),
        "--label-window-days", str(effective.get("label_window_days", 365)),
        "--obs-vocab-size", str(effective.get("obs_vocab_size", 1500)),
        "--obs-min-df", str(effective.get("obs_min_df", 20)),
        "--obs-min-patient-count", str(effective.get("obs_min_patient_count", 20)),
        "--obs-exclude-vocab", _as_comma(effective.get("obs_exclude_vocab", "")),
        # Measurement domain (value-aware synthetic tokens; binary per-doc). The
        # driver defaults these too, so they only matter when `measurement` is in
        # `domains`.
        "--meas-vocab-size", str(effective.get("meas_vocab_size", 2000)),
        "--meas-min-df", str(effective.get("meas_min_df", 20)),
        "--meas-min-patient-count", str(effective.get("meas_min_patient_count", 20)),
        # Anchor hierarchy above anchors (default none = flat forest).
        "--anchor-hierarchy", str(effective.get("anchor_hierarchy", "none")),
        "--hier-concept-class", str(effective.get("hier_concept_class", "")),
        "--hier-restrict-under", str(effective.get("hier_restrict_under", "")),
        "--hier-min-class-size", str(effective.get("hier_min_class_size", 2)),
        "--hier-max-class-fraction", str(effective.get("hier_max_class_fraction", 1.0)),
        # SVI schedule (mirrors build_dag_placement_args). Default 0.0 = full-batch;
        # _base.yaml sets 0.1, so a multidomain experiment inherits mini-batch
        # UNLESS its frontmatter pins mini_batch_fraction: 0.0 (exps 0070/0071 do).
        "--mini-batch-fraction", str(effective.get("mini_batch_fraction", 0.0)),
        "--learning-rate-tau0", str(effective.get("learning_rate_tau0", 1.0)),
        "--learning-rate-kappa", str(effective.get("learning_rate_kappa", 0.7)),
    ]
    # omega / eta_per_domain: emit ONLY when set (None -> shim scalar default).
    # _as_comma (defined above, before the args list) normalizes the frontmatter
    # comma-string-or-YAML-list to the comma-string the driver's
    # _parse_float_list expects.
    if effective.get("omega") is not None:
        args.extend(["--omega", _as_comma(effective["omega"])])
    if effective.get("eta_per_domain") is not None:
        args.extend(["--eta-per-domain", _as_comma(effective["eta_per_domain"])])
    # store_true flag: emit only when truthy in frontmatter.
    if effective.get("rollup_attestation"):
        args.append("--rollup-attestation")
    return args


def build_eval_args(checkpoint_dir: Path, effective: dict) -> list[str]:
    """Build the CLI arg list for eval_coherence_cloud.py."""
    return [
        "--checkpoint", str(checkpoint_dir),
        "--model-class", str(effective.get("model_class", "lda")),
    ]


def build_dashboard_args(
    effective: dict, checkpoint_dir: Path, zip_name: str,
) -> list[str]:
    """Build the CLI arg list for build_dashboard_cloud.py.

    `zip_name` is the basename (not full path) written as a sibling of the
    bundle directory inside the checkpoint dir. Callers should pass
    `f"{exp_id:04d}-{slug}-dashboard.zip"`.

    Signature note: parameter order is `(effective, checkpoint_dir, zip_name)`
    rather than `build_eval_args`'s `(checkpoint_dir, effective)` because
    `zip_name` is naturally last (it's caller-supplied per-experiment), and
    putting `effective` first keeps the config-derived args grouped together.

    Dict-access asymmetry: `model_class` uses `.get(..., "lda")` to match the
    `build_eval_args` convention (the wrapper always sets model_class anyway,
    but a fallback is harmless). `vocab_top_n` and `top_n_codes_for_npmi` use
    direct `[...]` access because they live in `_base.yaml` — if they're
    missing, a KeyError is the right failure mode (the YAML chain is broken).
    """
    args = [
        "--checkpoint", str(checkpoint_dir),
        "--model-class", str(effective.get("model_class", "lda")),
        "--zip-name", zip_name,
        "--vocab-top-n", str(effective["vocab_top_n"]),
        "--top-n-codes-for-npmi", str(effective["top_n_codes_for_npmi"]),
    ]
    # STM uses the covariate cache to compute the faithful corpus-mean
    # corpus_prevalence; pass it through when configured (ignored for LDA/HDP,
    # which need no covariate sidecar). Mirrors the build/eval arg convention.
    if effective.get("cache_uri"):
        args.extend(["--cache-uri", str(effective["cache_uri"])])
    return args


# Spark configuration constants. Match the existing per-cohort Makefile targets.
# If these need to vary per-experiment, lift them into _base.yaml in a future
# increment.
SPARK_SUBMIT_FLAGS = [
    "--master", "yarn",
    "--deploy-mode", "client",
    "--driver-memory", "4g",
    "--conf", "spark.executor.cores=4",
    "--conf", "spark.executor.memory=6g",
    "--conf", "spark.executor.memoryOverhead=2g",
    "--conf", "spark.python.worker.memory=2g",
]


def cluster_overlay_path(repo_root: Path) -> Path:
    """Path to the cluster dependency-overlay zip.

    Carries the pure-Python runtime deps the AoU Dataproc image does NOT ship
    (formulaic and its pure-Python leaves), built by the Makefile
    `cluster-overlay` target from cluster-requirements.txt. The image already
    provides numpy/pandas/scipy/pyarrow, so the overlay carries only the
    leaves and rides on --py-files atop the image's own Python.
    """
    return repo_root / "analysis" / "cloud" / "dist" / "formulaic_overlay.zip"


def build_spark_submit_cmd(
    script: str, script_args: list[str], repo_root: Path,
) -> list[str]:
    """Build the full spark-submit command line.

    All Python travels via --py-files on the image's own interpreter -- no
    interpreter override, no --files. Two zips of our own source (spark_vi,
    charmpheno) ride so editing them only needs a fast `make zip`. A third,
    the dependency overlay, carries the pure-Python runtime deps the image
    does NOT ship (e.g. `formulaic`, needed by the STM covariate/formula path)
    built on the master from cluster-requirements.txt (see the Makefile
    `cluster-overlay` target). The image supplies numpy/pandas/scipy/pyarrow,
    so the overlay holds only the pure-Python leaves and never shadows the
    image's compiled libs. The overlay is absent for LDA/HDP runs that never
    import formulaic -- then only the source zips ride.
    """
    py_files = [
        repo_root / "spark-vi" / "dist" / "spark_vi.zip",
        repo_root / "charmpheno" / "dist" / "charmpheno.zip",
    ]
    overlay = cluster_overlay_path(repo_root)
    if overlay.exists():
        py_files.append(overlay)
    py_files_val = ",".join(str(p) for p in py_files)

    return (
        ["spark-submit"]
        + SPARK_SUBMIT_FLAGS
        + ["--py-files", py_files_val, script]
        + script_args
    )


def _count_existing_fit_sessions(summary_text: str) -> int:
    """Count occurrences of '## Fit session N' headers in existing summary."""
    return len(re.findall(r"^## Fit session \d+", summary_text, re.MULTILINE))


def write_summary_header(
    summary_path: Path, *, exp_id: int, slug: str, effective: dict,
) -> None:
    """Write or append the per-session header to summary.md.

    If the file doesn't exist: writes the experiment-level header (title + effective
    config block) plus '## Fit session 1' header.
    If the file exists: appends '## Fit session N' header where N is (count+1).
    """
    started = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    if not summary_path.exists():
        lines = [
            f"# Experiment {exp_id:04d} — {slug}",
            "",
            "## Effective config",
        ]
        for k in sorted(effective.keys()):
            lines.append(f"{k}: {effective[k]}")
        lines += ["", "## Fit session 1", f"Started: {started}", ""]
        summary_path.write_text("\n".join(lines) + "\n")
    else:
        existing = summary_path.read_text()
        n = _count_existing_fit_sessions(existing) + 1
        with summary_path.open("a") as f:
            f.write(f"\n## Fit session {n}\nStarted: {started}\n\n")


class _SignalReceived(Exception):
    """Raised inside the tee loop when SIGTERM or SIGINT arrives."""
    def __init__(self, signum: int):
        super().__init__(f"signal {signum}")
        self.signum = signum


def run_subprocess_tee_sanitize(
    cmd: list[str], summary_path: Path, patterns: list[re.Pattern],
) -> int:
    """Run `cmd` as a subprocess; stream stdout line-by-line.

    Each line is printed to this process's stdout (live debugging) AND, if
    `sanitize_line` returns non-None, appended to `summary_path`.

    If SIGTERM or SIGINT arrives during streaming, writes a
    `### Killed at iter N (signal: ...)` marker to summary_path, terminates
    the child, and returns 130 (standard signal-exit code). Otherwise returns
    the child's exit code.

    Tracks the last `[driver]   iter N/M:` line seen so the killed marker can
    name the iter that was in flight.
    """
    def _handler(signum, frame):  # noqa: ARG001 — frame unused
        raise _SignalReceived(signum)

    prev_term = signal.signal(signal.SIGTERM, _handler)
    prev_int = signal.signal(signal.SIGINT, _handler)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True,
    )
    assert proc.stdout is not None
    last_iter: int | None = None
    try:
        with summary_path.open("a") as fout:
            for line in proc.stdout:
                # Live debugging: always print to terminal
                sys.stdout.write(line)
                sys.stdout.flush()
                # Track iter for the killed-marker, even on lines we don't
                # commit (so we know which iter was in flight when the signal
                # arrived).
                m = parse_iter_marker(line)
                if m is not None:
                    last_iter = m
                # Committed record: sanitized only
                clean = sanitize_line(line, patterns)
                if clean is not None:
                    fout.write(clean)
                    fout.flush()
        return proc.wait()
    except _SignalReceived as sig:
        with summary_path.open("a") as fout:
            iter_str = f"iter {last_iter}" if last_iter is not None else "unknown iter"
            fout.write(f"\n### Killed at {iter_str} (signal: {sig.signum})\n")
            fout.flush()
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return 130
    finally:
        signal.signal(signal.SIGTERM, prev_term)
        signal.signal(signal.SIGINT, prev_int)


def append_eval_section(
    summary_path: Path, eval_stdout: str, *, exit_code: int = 0,
) -> None:
    """Append a sanitized, timestamped '## Eval (NPMI)' section to summary_path.

    Header includes a UTC timestamp so multiple eval runs against the same
    experiment (e.g. `make eval-exp ID=N` re-runs) are distinguishable in
    the same summary file. A `### Eval complete (exit N)` marker is appended
    at the end for consistency with the fit `### Session complete` pattern.
    """
    started = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with summary_path.open("a") as f:
        f.write(f"\n## Eval (NPMI) — {started}\n")
        for line in eval_stdout.splitlines(keepends=True):
            clean = sanitize_line(line, DROP_PATTERNS)
            if clean is not None:
                f.write(clean)
        if not eval_stdout.endswith("\n"):
            f.write("\n")
        f.write(f"\n### Eval complete (exit {exit_code})\n")


def write_build_section_header(summary_path: Path) -> None:
    """Append a timestamped '## Dashboard build — <UTC ts>' header.

    The build dispatch then streams its sanitized stdout via
    `run_subprocess_tee_sanitize` (same pattern as fit). After the dispatch
    completes, `main()` appends `### Build complete (exit N)` for symmetry
    with the eval pattern.
    """
    started = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with summary_path.open("a") as f:
        f.write(f"\n## Dashboard build — {started}\n")


# RUNS_DIR mirrors the existing Makefile constant. Override via --runs-dir CLI
# for local testing. On the cluster, the GCS-mounted path is the canonical home.
DEFAULT_RUNS_DIR = "/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=False)
    selector.add_argument("--next", action="store_true",
                          help="Pick the lowest-id experiment with status: pending.")
    selector.add_argument("--id", type=int, default=None,
                          help="Run the experiment with the given numeric id.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip fit dispatch; only run eval against the existing "
                             "checkpoint at $RUNS_DIR/NNNN-slug/. If --id is omitted, "
                             "auto-selects the most-recently-fit experiment "
                             "(latest manifest.json mtime under $RUNS_DIR).")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip the eval dispatch at the end of the run. Fit only.")
    parser.add_argument("--build-only", action="store_true",
                        help="Skip fit and eval; only run the dashboard build "
                             "against the existing checkpoint at $RUNS_DIR/NNNN-slug/. "
                             "If --id is omitted, auto-selects the most recent "
                             "fit whose dashboard_bundle/corpus_stats.json is "
                             "missing or older than manifest.json.")
    parser.add_argument("--build-covariates-only", action="store_true",
                        help="Rebuild the STM covariate cache for the given experiment "
                             "without re-running the fit. Requires --id N and "
                             "model_class=stm.")
    parser.add_argument("--force-covariates", action="store_true",
                        help="With --build-covariates-only: delete the existing "
                             "covariate cache dir before rebuilding so the formula "
                             "change is picked up. With --build-only: force-rebuild "
                             "the covariate cache before the build.")
    parser.add_argument("--auto-covariates", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="On --build-only, auto-rebuild the STM covariate cache "
                             "if the build reports a covariate-cache miss, then "
                             "retry once (default: on). --no-auto-covariates "
                             "restores fail-fast.")
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR,
                        help="Base directory for run output. Default: %(default)s")
    parser.add_argument("--experiments-dir", type=Path, default=EXPERIMENTS_DIR,
                        help="Where docs/experiments/NNNN-*.md files live.")
    parser.add_argument("--defaults-dir", type=Path, default=DEFAULTS_DIR,
                        help="Where experiments/defaults/*.yaml files live.")
    args = parser.parse_args(argv)

    if args.no_eval and args.eval_only:
        print("[run-exp] ERROR: --no-eval and --eval-only are contradictory", flush=True)
        return 2

    if args.build_only and (args.eval_only or args.no_eval):
        print("[run-exp] ERROR: --build-only is contradictory with --eval-only "
              "and --no-eval", flush=True)
        return 2

    if args.build_covariates_only and (args.eval_only or args.no_eval or args.build_only):
        print("[run-exp] ERROR: --build-covariates-only is contradictory with "
              "--eval-only, --no-eval, and --build-only", flush=True)
        return 2

    if args.force_covariates and not (args.build_covariates_only or args.build_only):
        print("[run-exp] ERROR: --force-covariates requires --build-covariates-only "
              "or --build-only", flush=True)
        return 2

    # 1. Select experiment file
    if args.next:
        exp_path = find_next_pending(args.experiments_dir)
        if exp_path is None:
            print("[run-exp] no pending experiments found", flush=True)
            return 1
    elif args.id is not None:
        try:
            exp_path = find_by_id(args.experiments_dir, args.id)
        except FileNotFoundError as e:
            print(f"[run-exp] ERROR: {e}", flush=True)
            return 2
    elif args.eval_only:
        # No --id with --eval-only: auto-select most-recently-fit experiment.
        runs_dir_path = Path(args.runs_dir)
        auto_id = find_most_recent_fit(runs_dir_path)
        if auto_id is None:
            print(f"[run-exp] ERROR: --eval-only without --id requires at least one "
                  f"checkpoint under {runs_dir_path}; none found", flush=True)
            return 2
        print(f"[run-exp] --eval-only auto-selected most-recent-fit id={auto_id}",
              flush=True)
        try:
            exp_path = find_by_id(args.experiments_dir, auto_id)
        except FileNotFoundError as e:
            print(f"[run-exp] ERROR: {e}", flush=True)
            return 2
    elif args.build_only:
        # No --id with --build-only: auto-select the freshest fit whose
        # bundle is missing or stale.
        runs_dir_path = Path(args.runs_dir)
        auto_id = find_most_recent_fit_needing_build(runs_dir_path)
        if auto_id is None:
            print(f"[run-exp] no fits need building under {runs_dir_path} "
                  "(all bundles current or no checkpoints yet)", flush=True)
            return 0
        print(f"[run-exp] --build-only auto-selected id={auto_id} "
              "(fit newer than dashboard bundle)", flush=True)
        try:
            exp_path = find_by_id(args.experiments_dir, auto_id)
        except FileNotFoundError as e:
            print(f"[run-exp] ERROR: {e}", flush=True)
            return 2
    elif args.build_covariates_only:
        # --build-covariates-only without --id has no auto-discovery path.
        print("[run-exp] ERROR: --build-covariates-only requires --id N", flush=True)
        return 2
    else:
        print("[run-exp] ERROR: provide --next, --id N, --eval-only, "
              "--build-only, or --build-covariates-only",
              flush=True)
        return 2
    print(f"[run-exp] experiment: {exp_path}", flush=True)

    # 2. Read frontmatter + merge defaults
    try:
        fm = read_frontmatter(exp_path)
    except ValueError as e:
        print(f"[run-exp] ERROR: {e}", flush=True)
        return 2
    validate_frontmatter(fm)
    try:
        defaults = load_defaults(fm["cohort"], args.defaults_dir)
    except FileNotFoundError as e:
        print(f"[run-exp] ERROR: {e}", flush=True)
        return 2
    effective = merge_config(defaults, fm)

    # 2b. --build-covariates-only: dispatch to standalone script and return.
    if args.build_covariates_only:
        if effective.get("model_class") != "stm":
            print(f"[run-exp] ERROR: --build-covariates-only requires model_class=stm, "
                  f"got {effective.get('model_class')!r}", flush=True)
            return 2
        if not effective.get("cache_uri"):
            print("[run-exp] ERROR: --build-covariates-only requires cache_uri in "
                  "effective config", flush=True)
            return 2
        return _dispatch_covariate_build(effective, force=args.force_covariates)

    # 3. Resolve save_dir, detect resume
    # Resume only when there's an actual checkpoint (manifest.json) inside the
    # save_dir, not merely when the directory exists. A prior failed attempt
    # may have created the directory + summary.md without ever writing a
    # checkpoint; treating that as a resume causes the driver to fail looking
    # for manifest.json.
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    save_dir = runs_dir / f"{fm['id']:04d}-{fm['slug']}"
    has_checkpoint = (save_dir / "manifest.json").exists()
    resume_from: Path | None = save_dir if has_checkpoint else None
    print(f"[run-exp] save_dir: {save_dir}  resume: {resume_from is not None}",
          flush=True)
    # Guard: a fit resume warm-starts onto the checkpoint's params, so refuse
    # when the checkpoint's corpus no longer matches the config (eval-only /
    # build-only just read the checkpoint, so they don't warm-start and skip
    # the check). Exits(2) on mismatch with remediation guidance.
    if resume_from is not None and not (args.eval_only or args.build_only):
        check_resume_compat(save_dir, effective)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 4. Write summary header (new file or append session marker)
    #    Skip when --eval-only or --build-only — those modes don't open a
    #    new fit session; they each append their own section header later.
    summary_path = save_dir / "summary.md"
    if not (args.eval_only or args.build_only):
        write_summary_header(
            summary_path, exp_id=fm["id"], slug=fm["slug"], effective=effective,
        )

    # 5. Dispatch fit (unless --eval-only or --build-only)
    if args.eval_only or args.build_only:
        if not (save_dir / "manifest.json").exists():
            mode = "--eval-only" if args.eval_only else "--build-only"
            print(f"[run-exp] ERROR: {mode} requires checkpoint at "
                  f"{save_dir}/manifest.json (none found)", flush=True)
            return 2
        mode = "--eval-only" if args.eval_only else "--build-only"
        print(f"[run-exp] {mode}: skipping fit dispatch", flush=True)
    else:
        fit_script = REPO_ROOT / build_fit_driver_path(effective)
        # Both LDA and STM support --resume-from; build_fit_args threads it.
        fit_args = build_fit_args(effective, str(save_dir), resume_from)
        fit_cmd = build_spark_submit_cmd(str(fit_script), fit_args, REPO_ROOT)
        # Display-only join; cmd is passed as list to Popen/run, not via shell.
        print(f"[run-exp] spark-submit: {' '.join(fit_cmd)}", flush=True)
        fit_rc = run_subprocess_tee_sanitize(fit_cmd, summary_path, DROP_PATTERNS)
        if fit_rc != 0:
            print(f"[run-exp] fit exited non-zero ({fit_rc}); skipping eval", flush=True)
            with summary_path.open("a") as f:
                f.write(f"\n### Session ended with exit code {fit_rc}\n")
            return fit_rc
        with summary_path.open("a") as f:
            f.write("\n### Session complete (exit 0)\n")

    # Skip eval dispatch when --no-eval (fit-only mode) or --build-only
    # (build is the only stage that runs).
    if args.no_eval:
        print("[run-exp] --no-eval: skipping eval dispatch", flush=True)
        print(f"[run-exp] DONE. summary at: {summary_path}", flush=True)
        return 0
    if args.build_only:
        print("[run-exp] --build-only: skipping eval dispatch", flush=True)
        # Fall through to the build-dispatch block below.
    elif effective.get("model_class") in ("dag_placement", "multidomain"):
        # dag_placement and multidomain both save an npz + manifest (a
        # methods-experiment artifact), not a topic-word bundle the NPMI eval
        # driver can read. Their metrics live in manifest.json (placement AUC/MRR;
        # dead-node init read + corpus_stats for multidomain) + the fit log.
        mc = effective.get("model_class")
        print(f"[run-exp] model_class={mc}: NPMI eval not wired for the npz "
              "result; skipping eval (see manifest.json + fit log).", flush=True)
    else:
        # 6. Dispatch eval (capture stdout into a string for sanitized append)
        eval_script = REPO_ROOT / "analysis" / "cloud" / "eval_coherence_cloud.py"
        eval_args = build_eval_args(save_dir, effective)
        eval_cmd = build_spark_submit_cmd(str(eval_script), eval_args, REPO_ROOT)
        # Display-only join; cmd is passed as list to Popen/run, not via shell.
        print(f"[run-exp] eval spark-submit: {' '.join(eval_cmd)}", flush=True)
        eval_proc = subprocess.run(
            eval_cmd, capture_output=True, text=True, check=False,
        )
        sys.stdout.write(eval_proc.stdout)
        sys.stdout.flush()
        sys.stderr.write(eval_proc.stderr)
        sys.stderr.flush()
        if eval_proc.returncode != 0:
            print(f"[run-exp] eval exited non-zero ({eval_proc.returncode}); "
                  "appending captured output anyway", flush=True)
        append_eval_section(summary_path, eval_proc.stdout, exit_code=eval_proc.returncode)

    # 7. Dispatch build (only when --build-only).
    if args.build_only:
        build_script = REPO_ROOT / "analysis" / "cloud" / "build_dashboard_cloud.py"
        zip_name = f"{fm['id']:04d}-{fm['slug']}-dashboard.zip"
        b_args = build_dashboard_args(effective, save_dir, zip_name)
        build_cmd = build_spark_submit_cmd(str(build_script), b_args, REPO_ROOT)
        # Display-only join; cmd is passed as list to Popen, not via shell.
        print(f"[run-exp] build spark-submit: {' '.join(build_cmd)}", flush=True)
        write_build_section_header(summary_path)
        if (args.force_covariates and not (
                effective.get("model_class") == "stm" and effective.get("cache_uri"))):
            print("[run-exp] --force-covariates has no effect here: not an STM "
                  "build with a cache_uri; skipping covariate rebuild", flush=True)
        build_rc = _build_only_with_auto_covariates(
            effective=effective, auto=args.auto_covariates,
            force=args.force_covariates,
            run_build=lambda: run_subprocess_tee_sanitize(build_cmd, summary_path, DROP_PATTERNS),
            dispatch_cov=lambda force: _dispatch_covariate_build(effective, force=force),
        )
        with summary_path.open("a") as f:
            f.write(f"\n### Build complete (exit {build_rc})\n")
        if build_rc != 0:
            print(f"[run-exp] build exited non-zero ({build_rc})", flush=True)
            return build_rc

    print(f"[run-exp] DONE. summary at: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Generate concise human-readable labels for phenotypes in a dashboard bundle.

For each phenotype, picks the top-N codes by Sievert-Shirley lambda-relevance
(matches the dashboard's lambda=0.6 default), sends them to an LLM via
pydantic-ai, and writes the returned label back into phenotypes.json's `label`
field.

The provider is auto-selected from the first per-provider env var that is
set, in this priority order:

    CHARMPHENO_LABEL_KEY_ANTHROPIC  -> anthropic:claude-haiku-4-5
    CHARMPHENO_LABEL_KEY_OPENAI     -> openai:gpt-4o-mini
    CHARMPHENO_LABEL_KEY_GOOGLE     -> google-gla:gemini-2.5-flash

You can put all three in your .env and the script will pick the first
available; delete a key from .env to fall through to the next provider.
Override the model explicitly with `--model <prefix>:<name>` — the script
will use the env var matching that prefix.

The script intentionally does NOT read provider-specific env vars like
ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY — that keeps the
labeling keys in their own namespace so they can't accidentally cross-bill
with another SDK caller running in the same shell (notably Claude Code,
which auto-picks up ANTHROPIC_API_KEY).

Sources of the key, in order:
  1. --api-key-file <path>            (file contents; requires --model too)
  2. CHARMPHENO_LABEL_KEY_* env vars  (process env)
  3. .env file (auto-loaded)          (only populates THIS process's env,
                                       does not export to the parent shell)

Usage:
    # one-time install of the labeling deps
    poetry install --with labeling

    # .env in the repo root with any of:
    #   CHARMPHENO_LABEL_KEY_ANTHROPIC=sk-ant-...
    #   CHARMPHENO_LABEL_KEY_OPENAI=sk-...
    #   CHARMPHENO_LABEL_KEY_GOOGLE=...
    poetry run python scripts/label_phenotypes.py \\
        --bundle-dir dashboard/public/data

    # force a specific provider:
    poetry run python scripts/label_phenotypes.py \\
        --model openai:gpt-4o-mini

    # key from a file (never enters env at all):
    poetry run python scripts/label_phenotypes.py \\
        --model anthropic:claude-haiku-4-5 \\
        --api-key-file ~/.charmpheno-label-key

The script is idempotent: phenotypes that already have a non-empty `label`
are skipped unless --force.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

from typing import Literal

from pydantic import BaseModel, Field


DEFAULT_TOP_N = 15
DEFAULT_LAMBDA = 0.6
DEFAULT_MAX_WORDS = 6

# Provider priority order. The first entry whose env var is set wins when
# `--model` is not provided. When `--model` IS provided, its prefix selects
# which env var to read (and the entry's default name is ignored).
#   (model_prefix, env_var, default_model_name)
PROVIDER_CHAIN: list[tuple[str, str, str]] = [
    ("anthropic", "CHARMPHENO_LABEL_KEY_ANTHROPIC", "claude-haiku-4-5"),
    ("openai", "CHARMPHENO_LABEL_KEY_OPENAI", "gpt-4o-mini"),
    ("google-gla", "CHARMPHENO_LABEL_KEY_GOOGLE", "gemini-2.5-flash"),
]

QualityCategory = Literal["phenotype", "background", "anchor", "mixed", "dead"]


def _build_system_prompt(max_words: int) -> str:
    # The {MAX_WORDS} substitution is done at build time so the resulting
    # string is byte-identical across all topics in a run (enabling prompt
    # caching where the provider supports it).
    return f"""\
You are a clinical informatics expert interpreting learned topics from \
an LDA topic model trained over patient records. Each topic is a \
distribution over clinical concepts; you will be given a topic's top \
concepts ranked by Sievert-Shirley relevance, plus three topic-level \
statistics.

The current model is trained over patient conditions only (no drugs, \
procedures, measurements, or labs); your interpretations should reflect \
that — describe conditions and their clinical relationships, not \
treatments or workups.

## Topic-level statistics

- **NPMI** (normalized pointwise mutual information): a scalar in \
[-1, 1] summarizing how strongly the topic's top-N concepts co-occur \
in the reference corpus, relative to chance. Higher = more coherent \
top-N. NPMI alone is not enough — common conditions always co-occur, \
so a topic dominated by common comorbidities can score moderate-to- \
high NPMI without representing a real cluster.
- **pair_coverage**: fraction of the topic's top-N concept pairs that \
contributed to the NPMI calculation (cleared the minimum-joint-count \
threshold in the reference corpus). Low pair_coverage means most of \
the top-N pairs are too rare to score; NPMI is then averaged over \
only a few pairs and is less reliable.
- **usage**: total mass this topic carries across the corpus. Near- \
floor usage means the topic is essentially unused regardless of how \
its top-N looks.

## Quality categories

Classify each topic into exactly one of:

- **phenotype**: a recognizable disease/condition cluster with a \
coherent clinical theme. Moderate-to-high usage; top concepts share \
a clinical story. The common case for clinically useful topics.

- **background**: a large catch-all carrying a substantial slice of \
corpus mass (high usage). Typical flavors include chronic-comorbidity \
(e.g. HTN/HLD/T2DM/GERD/anxiety), acute-presentation (e.g. pain/ \
chest pain/nausea/SoB/vomiting), or metabolic-syndrome. Coherent but \
not a discrete phenotype; useful for cohort exclusion, not selection.

- **anchor**: dominated by a single specific concept (one term carries \
most of the within-topic weight). Narrow but interpretable.

- **mixed**: top concepts span unrelated clinical themes; no shared \
theme survives a clinician's reading. Often a sign of insufficient \
model capacity rather than a real cluster.

- **dead**: an essentially unused topic sitting at the prior-smoothing \
floor (near-floor usage). Two sub-flavors that look different but \
are equally unusable:
  (a) top concepts are rare or weird and the topic never accumulated \
      meaningful data — NPMI is typically low and pair_coverage low \
      because the rare top-N pairs don't clear the threshold;
  (b) top concepts are common comorbidities and NPMI is moderate-to- \
      high with high pair_coverage, but usage is near-floor. The top-N \
      reflects baseline corpus smoothing, not a co-occurrence pattern \
      in actual documents.
Either way: no useful clinical interpretation.

Read the three statistics jointly. Useful tells:
- High NPMI + high usage                          → phenotype or background
- High NPMI + low pair_coverage + low usage       → narrow rare-condition \
                                                    cluster; trust cautiously
- Moderate-to-high NPMI + high pair_coverage + \
  near-floor usage                                → dead (case b)
- Low NPMI + low pair_coverage + low usage        → dead (case a)
- Low NPMI with no other clear pattern            → mixed

## Output fields

- **label**: a concise clinician-voice label, at most {max_words} words. \
Examples: "Type 2 diabetes care", "Chronic comorbidity catch-all". \
For mixed/dead, use something honest: "Mixed clinical themes", \
"Unused / low-signal topic".

- **description**: 2-3 sentences, clinician voice, plain clinical \
English. No model terminology, no references to topics, codes, or \
statistics. For phenotype/background/anchor, name the clinical \
pattern and what it suggests about the patients. For mixed/dead, \
briefly say why the topic isn't usable without using technical \
vocabulary (e.g. "no shared clinical theme across the leading \
conditions" rather than "low NPMI").

- **quality**: one of phenotype, background, anchor, mixed, dead.

Voice rules:
- Use clinical terminology a clinician would recognize, not lay phrasing.
- Be specific. "Cardiovascular" alone is too vague; "Heart failure with \
arrhythmia" is better.
- No hedging ("possibly...", "likely related to..."). State the theme.
- Do not mention the model, the codes, this prompt, or any statistics \
in the label or description — those live in the structured `quality` \
field for downstream filtering."""


class PhenotypeLabel(BaseModel):
    """Structured output schema — pydantic-ai enforces this against the
    provider's structured-output / function-calling mechanism."""

    label: str = Field(description="Concise clinical label.")
    description: str = Field(
        description=(
            "2-3 sentences, clinician voice, plain clinical English. "
            "No model terminology or references to topics/codes/statistics."
        ),
    )
    quality: QualityCategory = Field(
        description=(
            "One of phenotype, background, anchor, mixed, dead. "
            "See the system prompt for category definitions."
        ),
    )


def _maybe_load_dotenv(explicit_path: str | None) -> Path | None:
    """Load a .env file into THIS process's os.environ. Does not export to
    the parent shell. Searches, in order:
      1. --env-file <path> if explicitly passed
      2. .env in CWD
      3. .env at the repo root (inferred from this script's path)
    """
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    else:
        candidates.append(Path.cwd() / ".env")
        candidates.append(Path(__file__).resolve().parent.parent / ".env")

    existing = [c for c in candidates if c.is_file()]

    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        if existing:
            print(
                f"[label] WARNING: found .env at {existing[0]} but "
                f"python-dotenv is not installed.\n"
                f"  Run `make install-labeling` (or "
                f"`poetry install --with labeling`) to enable .env support.",
                file=sys.stderr,
                flush=True,
            )
        return None

    for c in existing:
        # override=False so a real env var still wins over the file.
        load_dotenv(c, override=False)
        return c
    return None


def _read_key_file(path: str) -> str:
    p = Path(path).expanduser()
    if not p.is_file():
        raise SystemExit(f"--api-key-file not found: {p}")
    key = p.read_text().strip()
    if not key:
        raise SystemExit(f"--api-key-file is empty: {p}")
    return key


def _resolve_provider_and_key(args: argparse.Namespace) -> tuple[str, str]:
    """Decide which provider+model to use and return (model_string, api_key).

    Resolution:
      1. If --api-key-file is set: require --model so we know which provider
         to talk to; use that file's contents as the key.
      2. Else if --model is set: look up its prefix in PROVIDER_CHAIN and
         read the corresponding env var.
      3. Else: walk PROVIDER_CHAIN in priority order, use the first entry
         whose env var is set (with its default model name).

    Never reads provider-specific env vars like ANTHROPIC_API_KEY etc. —
    keys must be in the CHARMPHENO_LABEL_KEY_* namespace.
    """
    if args.api_key_file:
        if not args.model:
            raise SystemExit(
                "--api-key-file requires --model so the provider is "
                "unambiguous (e.g. --model anthropic:claude-haiku-4-5)."
            )
        return args.model, _read_key_file(args.api_key_file)

    if args.model:
        if ":" not in args.model:
            raise SystemExit(
                f"--model must be prefixed, e.g. 'anthropic:claude-haiku-4-5'. "
                f"Got: {args.model!r}"
            )
        prefix = args.model.split(":", 1)[0]
        match = next((e for e in PROVIDER_CHAIN if e[0] == prefix), None)
        if match is None:
            supported = ", ".join(p for p, _, _ in PROVIDER_CHAIN)
            raise SystemExit(
                f"unsupported model prefix {prefix!r}. "
                f"Supported: {supported}."
            )
        _, env_var, _ = match
        key = os.environ.get(env_var)
        if not key:
            raise SystemExit(
                f"--model selected provider {prefix!r}, but env var "
                f"{env_var} is not set (and no --api-key-file given).\n"
                f"  Set {env_var}=... in your shell or .env."
            )
        return args.model, key

    # Auto-pick: first env var set wins.
    for prefix, env_var, default_name in PROVIDER_CHAIN:
        key = os.environ.get(env_var)
        if key:
            return f"{prefix}:{default_name}", key

    seen = sorted(
        k for k in os.environ
        if "CHARMPHENO" in k.upper() or "LABEL" in k.upper()
    )
    seen_str = ", ".join(seen) if seen else "(none)"
    expected = "\n    ".join(v for _, v, _ in PROVIDER_CHAIN)
    raise SystemExit(
        f"No API key found.\n"
        f"  Set one of these env vars (or put them in .env):\n"
        f"    {expected}\n"
        f"  Related env vars in this process: {seen_str}\n"
        f"  (This script intentionally does not read provider-specific env "
        f"vars like ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY.)"
    )


def _build_agent(model_str: str, api_key: str, max_words: int):
    """Build a pydantic-ai Agent for the given model string with the API
    key passed in explicitly (not via env), structured output, and the
    shared system prompt.

    The provider prefix (`openai:`, `anthropic:`, `google-gla:`, ...)
    controls which provider class is instantiated. We pass api_key=
    directly so pydantic-ai never consults env vars under any provider.
    """
    from pydantic_ai import Agent

    if ":" not in model_str:
        raise SystemExit(
            f"--model must be prefixed, e.g. 'google-gla:gemini-2.5-flash', "
            f"'openai:gpt-4o-mini', 'anthropic:claude-haiku-4-5'. "
            f"Got: {model_str!r}"
        )
    prefix, name = model_str.split(":", 1)

    if prefix == "google-gla":
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider
        model = GoogleModel(name, provider=GoogleProvider(api_key=api_key))
    elif prefix == "openai":
        from pydantic_ai.models.openai import OpenAIModel
        from pydantic_ai.providers.openai import OpenAIProvider
        model = OpenAIModel(name, provider=OpenAIProvider(api_key=api_key))
    elif prefix == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider
        model = AnthropicModel(
            name, provider=AnthropicProvider(api_key=api_key),
        )
    else:
        raise SystemExit(
            f"unsupported model prefix {prefix!r}. "
            f"Supported: google-gla, openai, anthropic."
        )

    return Agent(
        model,
        system_prompt=_build_system_prompt(max_words),
        output_type=PhenotypeLabel,
    )


def _lambda_relevance(pwk: float, pw: float, lam: float) -> float:
    """Sievert-Shirley relevance: lam * log p(w|k) + (1 - lam) * log(p(w|k)/p(w)).

    Matches dashboard/src/lib/inference.ts. Returns -inf if pwk <= 0 (the
    code does not appear in the topic).
    """
    if pwk <= 0:
        return -math.inf
    log_pwk = math.log(pwk)
    if pw <= 0:
        return lam * log_pwk + (1.0 - lam) * 1e6
    return lam * log_pwk + (1.0 - lam) * math.log(pwk / pw)


def _top_codes_for_phenotype(
    *,
    beta_row: list[float],
    vocab: list[dict],
    lam: float,
    n: int,
) -> list[dict]:
    """Pick top-N concepts by Sievert-Shirley relevance. Each entry
    carries description, within-topic weight (%), and lift = p(w|k)/p(w)
    over the corpus. Domain is dropped — the current model is conditions-
    only, so it would be a constant column."""
    pw = [c.get("corpus_freq", 0.0) for c in vocab]
    scored = [
        (i, _lambda_relevance(beta_row[i], pw[i], lam))
        for i in range(len(beta_row))
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    out: list[dict] = []
    for idx, _score in scored[:n]:
        c = vocab[idx]
        pwk = beta_row[idx]
        pw_i = pw[idx]
        lift = (pwk / pw_i) if pw_i > 0 else float("inf")
        out.append({
            "description": c.get("description") or c.get("code"),
            "weight_pct": round(pwk * 100.0, 3),
            "lift": lift,
        })
    return out


def _build_user_message(
    *,
    phenotype_id: int,
    top_codes: list[dict],
    npmi: float,
    pair_coverage: float,
    usage_frac: float,
    max_words: int,
) -> str:
    lines = [
        f"Topic id: {phenotype_id}",
        f"Label budget: at most {max_words} words.",
        "",
        "Topic statistics:",
        f"  NPMI:           {npmi:.3f}",
        f"  pair_coverage:  {pair_coverage:.0%} of top-N pairs scored",
        f"  usage:          {usage_frac * 100:.2f}% of total corpus mass",
        "",
        "Top conditions (description · within-topic weight · lift over corpus):",
    ]
    for r, c in enumerate(top_codes, start=1):
        lift_str = "×∞" if c["lift"] == float("inf") else f"×{c['lift']:.1f}"
        lines.append(
            f"  {r:>2}. {c['description']}  ·  {c['weight_pct']:.2f}%  "
            f"·  {lift_str}"
        )
    return "\n".join(lines)


def _label_one(
    *,
    agent,
    phenotype_id: int,
    top_codes: list[dict],
    npmi: float,
    pair_coverage: float,
    usage_frac: float,
    max_words: int,
) -> tuple[PhenotypeLabel, dict]:
    """One labeling call. Returns (output, usage_dict)."""
    user_text = _build_user_message(
        phenotype_id=phenotype_id, top_codes=top_codes,
        npmi=npmi, pair_coverage=pair_coverage, usage_frac=usage_frac,
        max_words=max_words,
    )
    result = agent.run_sync(user_text)
    out: PhenotypeLabel = result.output
    u = result.usage()
    usage = {
        "input": getattr(u, "input_tokens", 0) or 0,
        "output": getattr(u, "output_tokens", 0) or 0,
    }
    return out, usage


def _write_atomic(path: Path, data: dict) -> None:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8",
        dir=str(path.parent), prefix=path.name + ".",
        suffix=".tmp", delete=False,
    )
    try:
        json.dump(data, tmp)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.replace(tmp.name, path)
    except Exception:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bundle-dir", type=Path, default=Path("dashboard/public/data"),
        help="Dir containing model.json, vocab.json, phenotypes.json",
    )
    parser.add_argument(
        "--model", default=None,
        help="pydantic-ai model string, e.g. 'anthropic:claude-haiku-4-5', "
             "'openai:gpt-4o-mini', 'google-gla:gemini-2.5-flash'. "
             "If omitted, auto-selects based on which CHARMPHENO_LABEL_KEY_* "
             "env var is set (anthropic > openai > google).",
    )
    parser.add_argument(
        "--top-n", type=int, default=DEFAULT_TOP_N,
        help="Top-N codes by relevance to send per phenotype",
    )
    parser.add_argument(
        "--lambda", dest="lam", type=float, default=DEFAULT_LAMBDA,
        help="Sievert-Shirley lambda for code ranking",
    )
    parser.add_argument(
        "--max-words", type=int, default=DEFAULT_MAX_WORDS,
        help="Label length budget in words",
    )
    parser.add_argument(
        "--api-key-file", type=str, default=None,
        help="Path to a file containing the API key (alternative to env var). "
             "Requires --model so the provider is unambiguous.",
    )
    parser.add_argument(
        "--env-file", type=str, default=None,
        help="Path to a .env file to load (only into this process). "
             "If omitted, looks for .env in CWD then at the repo root.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-label phenotypes that already have a label.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only label the first N phenotypes (for cost-bound dry runs).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be sent but don't call the API.",
    )
    args = parser.parse_args(argv)

    loaded = _maybe_load_dotenv(args.env_file)
    if loaded:
        print(f"[label] loaded env from {loaded}", flush=True)

    bundle_dir: Path = args.bundle_dir
    model_p = bundle_dir / "model.json"
    vocab_p = bundle_dir / "vocab.json"
    phens_p = bundle_dir / "phenotypes.json"
    for p in (model_p, vocab_p, phens_p):
        if not p.is_file():
            raise SystemExit(f"missing bundle file: {p}")

    model_b = json.loads(model_p.read_text())
    vocab_b = json.loads(vocab_p.read_text())
    phens_b = json.loads(phens_p.read_text())

    beta = model_b["beta"]
    vocab_codes = vocab_b["codes"]
    phenotypes = phens_b["phenotypes"]

    K = len(phenotypes)
    V_disp = len(vocab_codes)
    if len(beta) != K:
        raise SystemExit(
            f"length mismatch: {len(beta)} beta rows vs {K} phenotypes",
        )
    if any(len(row) != V_disp for row in beta):
        raise SystemExit(
            f"beta rows are not all width {V_disp} (the displayed vocab "
            f"size); is this a pre-trim bundle?",
        )

    todo: list[int] = []
    for i, p in enumerate(phenotypes):
        existing = (p.get("label") or "").strip()
        if existing and not args.force:
            continue
        todo.append(i)
        if args.limit is not None and len(todo) >= args.limit:
            break

    print(f"[label] {len(todo)}/{K} phenotypes to label "
          f"(--force={args.force}, --limit={args.limit})", flush=True)
    if not todo:
        print("[label] nothing to do", flush=True)
        return 0

    # Required stats fed to the model. Missing stats are a bundle bug; we
    # error explicitly rather than substituting defaults so a stale bundle
    # can't silently produce labels with no statistical grounding.
    for i, p in enumerate(phenotypes):
        for key in ("npmi", "pair_coverage", "corpus_prevalence"):
            if key not in p:
                raise SystemExit(
                    f"phenotype {i} missing {key!r} — re-export the bundle "
                    f"with a current write_phenotypes_bundle."
                )

    def _stats_for(i: int) -> tuple[float, float, float]:
        p = phenotypes[i]
        return (
            float(p["npmi"]),
            float(p["pair_coverage"]),
            float(p["corpus_prevalence"]),
        )

    if args.dry_run:
        for i in todo[:3]:
            top = _top_codes_for_phenotype(
                beta_row=beta[i], vocab=vocab_codes,
                lam=args.lam, n=args.top_n,
            )
            npmi, pcov, usage_frac = _stats_for(i)
            print(f"\n--- phenotype {i} (dry-run preview) ---", flush=True)
            print(_build_user_message(
                phenotype_id=i, top_codes=top,
                npmi=npmi, pair_coverage=pcov, usage_frac=usage_frac,
                max_words=args.max_words,
            ), flush=True)
        if len(todo) > 3:
            print(f"\n... and {len(todo) - 3} more "
                  f"(use --limit for a partial real run)", flush=True)
        print(f"\n[label] dry-run only; no API calls made", flush=True)
        return 0

    try:
        import pydantic_ai  # noqa: F401, WPS433
    except ImportError:
        raise SystemExit(
            "pydantic-ai not installed. Run: poetry install --with labeling"
        )

    model_str, api_key = _resolve_provider_and_key(args)
    agent = _build_agent(model_str, api_key, max_words=args.max_words)
    print(f"[label] using model {model_str}", flush=True)

    total = {"input": 0, "output": 0}
    for n, i in enumerate(todo, start=1):
        top = _top_codes_for_phenotype(
            beta_row=beta[i], vocab=vocab_codes,
            lam=args.lam, n=args.top_n,
        )
        npmi, pcov, usage_frac = _stats_for(i)
        try:
            out, usage = _label_one(
                agent=agent,
                phenotype_id=i, top_codes=top,
                npmi=npmi, pair_coverage=pcov, usage_frac=usage_frac,
                max_words=args.max_words,
            )
        except Exception as e:
            print(f"[label] phenotype {i}: error: {e}", flush=True)
            _write_atomic(phens_p, phens_b)
            raise

        for k in total:
            total[k] += usage[k]

        phenotypes[i]["label"] = out.label.strip()
        phenotypes[i]["description"] = out.description.strip()
        phenotypes[i]["quality"] = out.quality

        print(
            f"[label] {n:>3}/{len(todo)}  k={i:<3}  [{out.quality:<10}]  "
            f"{out.label}",
            flush=True,
        )

    _write_atomic(phens_p, phens_b)

    print(
        f"\n[label] wrote {len(todo)} labels to {phens_p}\n"
        f"[label] tokens — input {total['input']}, output {total['output']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

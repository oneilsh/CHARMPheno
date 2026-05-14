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


DEFAULT_TOP_N = 20
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


def _build_system_prompt(
    *,
    max_words: int,
    alpha_min: float,
    alpha_median: float,
    alpha_max: float,
    alpha_floor_threshold: float,
    kl_min: float,
    kl_median: float,
    kl_max: float,
) -> str:
    # All substitutions happen once per run so the resulting string is
    # byte-identical across topics (enabling prompt caching).
    return f"""\
You are a clinical informatics expert interpreting learned topics from \
an LDA topic model trained over patient records. Each topic is a \
distribution over clinical concepts; you will be given each topic's \
top concepts under two different rankings, plus Bayesian and \
information-theoretic statistics about the topic's distinctiveness.

The current model is trained over patient conditions only (no drugs, \
procedures, measurements, or labs); your interpretations should reflect \
that — describe conditions and their clinical relationships, not \
treatments or workups.

## Two rankings

For each topic you will see the top concepts ranked two ways:

- **by within-topic frequency**: the concepts that carry the most of \
this topic's mass (highest p(w|k)). These are what the topic IS made of.
- **by lift over the corpus**: the concepts most concentrated in this \
topic relative to the corpus baseline (highest p(w|k)/p(w)). These are \
what the topic SINGLES OUT relative to other topics.

The relationship between the two lists is itself diagnostic:

- **Strong overlap (the same concepts dominate both lists)** → real \
phenotype, IF the distinctiveness stats below confirm.
- **Frequency list = common comorbidities (HTN/HLD/T2DM/GERD/anxiety), \
lift list = unrelated outliers with tiny weight** → background catch-all \
OR dead-baseline (case b below). The distinctiveness stats disambiguate.
- **Both lists span unrelated clinical themes** → mixed.
- **A single concept dominates the frequency list AND tops the lift \
list** → anchor.

## Topic-level statistics

Two distinctiveness signals govern classification. They reflect what \
the model itself "thinks" of each topic and how far it sits from the \
corpus baseline:

- **alpha (α)** — the per-topic asymmetric Dirichlet prior weight. \
The fitter ran asymmetric-α optimization, which actively pushes α down \
for slots that aren't carrying real data and keeps it up for slots that \
are. α at the floor means the model itself has zeroed out that slot. \
Distribution across this fit:
    min     = {alpha_min:.4f}    ← floor (asymmetric-α optimizer's lower bound)
    median  = {alpha_median:.4f}
    max     = {alpha_max:.4f}
α within ≈10% of the min (i.e. α ≤ {alpha_floor_threshold:.4f}) \
indicates the slot is at floor. ALWAYS classify such topics as **dead** \
regardless of how the top-N reads — the optimizer has already declared \
them unused.

- **KL(β ‖ corpus)** — KL divergence between the topic's word \
distribution and the corpus marginal, in nats. Quantifies whether the \
topic is distinguishable from the corpus baseline:
    near 0       → β ≈ corpus marginal; the top-N is just baseline \
                  smoothing of common comorbidities, not a learned pattern.
    moderate     → β amplifies a subset of concepts ~2-3× over their \
                  corpus rates (typical of catch-all background flavors).
    high         → β sharply concentrates on concepts that are rare or \
                  much-overweighted in the corpus (real phenotype).
Distribution across this fit:
    min = {kl_min:.3f}, median = {kl_median:.3f}, max = {kl_max:.3f}.

Two more contextual stats:

- **NPMI** ([-1, 1]): co-occurrence coherence of the top-N concepts in \
the corpus. Helpful but not decisive — common comorbidities always \
co-occur, so moderate NPMI is achievable without a real cluster.
- **pair_coverage** (0..100%): fraction of top-N concept pairs that \
cleared the joint-count threshold. Low coverage = NPMI averaged over \
few pairs and less reliable.
- **usage** (% of corpus mass): how much of the corpus this topic \
carries. Useful for distinguishing real catch-alls (high usage) from \
narrow phenotypes (lower usage), but a low-usage topic with α \
well above floor and high KL is still a real phenotype.

## Quality categories

Classify each topic into exactly one of:

- **phenotype** — α well above floor AND KL above the median (relative \
to this fit). Top concepts share a coherent clinical theme. The common \
case for clinically useful topics; usage can range from <1% (rare \
condition) to several percent (common condition) — usage does NOT \
disqualify a topic from being a phenotype as long as α and KL show it \
is distinct from prior and from baseline.

- **background** — α above floor AND KL moderate (near the median) AND \
top concepts are common comorbidities or generic acute symptoms (e.g. \
HTN/HLD/T2DM/GERD/anxiety; or pain/chest pain/nausea/SoB/vomiting). \
Typically high usage (several % to tens of % of corpus mass). The \
topic carries real data but the pattern is the corpus's chronic or \
acute baseline, useful for cohort *exclusion*, not selection.

- **anchor** — one concept dominates both rankings; the topic \
essentially names that concept.

- **mixed** — α above floor BUT the two rankings together span \
unrelated clinical themes with no story a clinician would write down. \
Often a sign of insufficient model capacity.

- **dead** — either:
  (a) α at floor (within ≈10% of the minimum across this fit); OR
  (b) α above floor BUT KL near 0 (close to the minimum across this \
      fit). The slot carries some data but the data is the corpus \
      baseline — top-N is HTN/HLD/T2DM-style common comorbidities not \
      because the topic learned them but because η-smoothing landed \
      there.
Either way: no useful clinical interpretation.

## Decision order

To avoid rationalizing a label first and then picking a quality, \
classify in this order:

1. Check α. At floor → `dead`, stop.
2. Check KL. Near the minimum across the fit → `dead`, stop.
3. Check whether one concept dominates both rankings → `anchor`.
4. Check whether the rankings span unrelated themes → `mixed`.
5. Otherwise it is `phenotype` or `background`. Phenotype if KL is \
above the median; background if KL is moderate AND top concepts are \
generic comorbidities/symptoms.

## Output fields

- **quality**: one of phenotype, background, anchor, mixed, dead. \
Decide this FIRST using the rule above.

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
conditions" rather than "low KL").

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
    provider's structured-output / function-calling mechanism.

    Field order is deliberate: `quality` first so the model commits to
    the classification (using α and KL per the system prompt's decision
    rule) BEFORE crafting prose. Reversing the order encourages the
    model to write a plausible label first and rationalize the quality
    label to match it.
    """

    quality: QualityCategory = Field(
        description=(
            "One of phenotype, background, anchor, mixed, dead. Decide "
            "this FIRST using the alpha + KL decision rule in the "
            "system prompt — before writing the label and description."
        ),
    )
    description: str = Field(
        description=(
            "2-3 sentences, clinician voice, plain clinical English. "
            "No model terminology or references to topics/codes/statistics."
        ),
    )
    label: str = Field(description="Concise clinical label.")


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


def _build_agent(
    model_str: str,
    api_key: str,
    *,
    max_words: int,
    alpha_dist: tuple[float, float, float],
    kl_dist: tuple[float, float, float],
):
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
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
        model = OpenAIChatModel(name, provider=OpenAIProvider(api_key=api_key))
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

    a_min, a_med, a_max = alpha_dist
    k_min, k_med, k_max = kl_dist
    # "At floor" = within ~10% of the minimum. For asymmetric-α fits with
    # a hard lower bound, the min IS the floor and a 10% margin avoids
    # FP from numerical jitter.
    alpha_floor_threshold = a_min * 1.10
    return Agent(
        model,
        system_prompt=_build_system_prompt(
            max_words=max_words,
            alpha_min=a_min, alpha_median=a_med, alpha_max=a_max,
            alpha_floor_threshold=alpha_floor_threshold,
            kl_min=k_min, kl_median=k_med, kl_max=k_max,
        ),
        output_type=PhenotypeLabel,
    )


def _kl_div_topic_vs_corpus(
    beta_row: list[float], corpus_freq: list[float],
) -> float:
    """KL(β[k] ‖ p(w)) over the displayed vocab, in nats.

    β[k] is the topic-term distribution restricted to the displayed
    vocab (already row-stochastic per the export normalization). p(w)
    is the corpus marginal restricted to the same vocab; we renormalize
    it to a proper probability distribution over the displayed support
    before computing KL so both sides live on the same simplex.

    Terms with β[k][i] = 0 contribute 0 to the sum (since β log β → 0).
    Terms with p(w)[i] = 0 would make the divergence infinite; the
    export's small-cell guard ensures the displayed vocab has no zeros
    in practice, but we guard defensively by skipping such terms with
    a tiny epsilon.
    """
    total_pw = sum(corpus_freq)
    if total_pw <= 0:
        return 0.0
    eps = 1e-12
    kl = 0.0
    for b_i, p_i in zip(beta_row, corpus_freq):
        if b_i <= 0:
            continue
        p_norm = p_i / total_pw
        if p_norm <= eps:
            p_norm = eps
        kl += b_i * math.log(b_i / p_norm)
    return kl


def _top_codes_by_metric(
    *,
    beta_row: list[float],
    vocab: list[dict],
    n: int,
    metric: str,
) -> list[dict]:
    """Pick top-N concepts ranked by ``metric``:

      - ``"frequency"``: by within-topic mass p(w|k).
      - ``"lift"``: by lift p(w|k)/p(w) over the corpus.

    Each entry carries description, within-topic weight (%), and lift.
    Domain is dropped — conditions-only model means it'd be a constant
    column.
    """
    pw = [c.get("corpus_freq", 0.0) for c in vocab]

    def _score(i: int) -> float:
        pwk = beta_row[i]
        if pwk <= 0:
            return -math.inf
        if metric == "frequency":
            return pwk
        if metric == "lift":
            if pw[i] <= 0:
                # Code never appears in corpus; lift infinite. Saturate so
                # the comparison still orders rather than ties at inf.
                return 1e12 * pwk
            return pwk / pw[i]
        raise ValueError(f"unknown metric: {metric!r}")

    scored = sorted(range(len(beta_row)), key=_score, reverse=True)
    out: list[dict] = []
    for idx in scored[:n]:
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


def _format_code_list(codes: list[dict]) -> list[str]:
    lines = []
    for r, c in enumerate(codes, start=1):
        lift_str = "×∞" if c["lift"] == float("inf") else f"×{c['lift']:.1f}"
        lines.append(
            f"  {r:>2}. {c['description']}  ·  {c['weight_pct']:.2f}%  "
            f"·  {lift_str}"
        )
    return lines


def _build_user_message(
    *,
    phenotype_id: int,
    top_by_freq: list[dict],
    top_by_lift: list[dict],
    alpha: float,
    kl: float,
    npmi: float,
    pair_coverage: float,
    usage_frac: float,
    max_words: int,
) -> str:
    # Stats are listed in decision order (alpha and KL first — the
    # primary classifiers — then the supporting contextual stats).
    lines = [
        f"Topic id: {phenotype_id}",
        f"Label budget: at most {max_words} words.",
        "",
        "Distinctiveness statistics (primary classifiers):",
        f"  alpha:           {alpha:.4f}    "
        f"(check against the fit's min/floor in the system prompt)",
        f"  KL(beta||corpus): {kl:.3f}    "
        f"(check against the fit's min/median/max in the system prompt)",
        "",
        "Contextual statistics:",
        f"  NPMI:            {npmi:.3f}",
        f"  pair_coverage:   {pair_coverage:.0%} of top-N pairs scored",
        f"  usage:           {usage_frac * 100:.2f}% of total corpus mass",
        "",
        "Top conditions by within-topic frequency "
        "(description · weight · lift):",
    ]
    lines.extend(_format_code_list(top_by_freq))
    lines.append("")
    lines.append(
        "Top conditions by lift over the corpus "
        "(description · weight · lift):"
    )
    lines.extend(_format_code_list(top_by_lift))
    return "\n".join(lines)


def _label_one(
    *,
    agent,
    phenotype_id: int,
    top_by_freq: list[dict],
    top_by_lift: list[dict],
    alpha: float,
    kl: float,
    npmi: float,
    pair_coverage: float,
    usage_frac: float,
    max_words: int,
) -> tuple[PhenotypeLabel, dict]:
    """One labeling call. Returns (output, usage_dict)."""
    user_text = _build_user_message(
        phenotype_id=phenotype_id,
        top_by_freq=top_by_freq, top_by_lift=top_by_lift,
        alpha=alpha, kl=kl,
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
        help="Top-N codes per ranking (two rankings: by frequency and by "
             "lift) sent per topic.",
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
    alpha_arr = model_b.get("alpha")
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
    if not alpha_arr or len(alpha_arr) != K:
        raise SystemExit(
            f"model.json missing 'alpha' array of length {K}; the labeling "
            f"rubric needs the per-topic Dirichlet prior weights to detect "
            f"floor-α (dead) topics.",
        )

    # Per-topic distinctiveness signals.
    #   alpha[k] reflects the asymmetric-α optimizer's verdict on topic k.
    #   KL(β[k]||p(w)) reflects how far the topic departs from the corpus
    #     marginal — low KL = baseline pseudo-coherent (dead case b).
    corpus_freq_disp = [c.get("corpus_freq", 0.0) for c in vocab_codes]
    kl_arr = [
        _kl_div_topic_vs_corpus(beta[k], corpus_freq_disp)
        for k in range(K)
    ]
    alpha_arr = [float(a) for a in alpha_arr]
    alpha_sorted = sorted(alpha_arr)
    kl_sorted = sorted(kl_arr)

    def _median(xs: list[float]) -> float:
        n = len(xs)
        if n == 0:
            return 0.0
        mid = n // 2
        if n % 2 == 1:
            return xs[mid]
        return 0.5 * (xs[mid - 1] + xs[mid])

    alpha_dist = (alpha_sorted[0], _median(alpha_sorted), alpha_sorted[-1])
    kl_dist = (kl_sorted[0], _median(kl_sorted), kl_sorted[-1])
    print(
        f"[label] alpha[K={K}] min={alpha_dist[0]:.4f} "
        f"median={alpha_dist[1]:.4f} max={alpha_dist[2]:.4f}",
        flush=True,
    )
    print(
        f"[label] KL(beta||corpus)[K={K}] min={kl_dist[0]:.3f} "
        f"median={kl_dist[1]:.3f} max={kl_dist[2]:.3f}",
        flush=True,
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

    def _two_rankings(i: int) -> tuple[list[dict], list[dict]]:
        top_freq = _top_codes_by_metric(
            beta_row=beta[i], vocab=vocab_codes,
            n=args.top_n, metric="frequency",
        )
        top_lift = _top_codes_by_metric(
            beta_row=beta[i], vocab=vocab_codes,
            n=args.top_n, metric="lift",
        )
        return top_freq, top_lift

    if args.dry_run:
        for i in todo[:3]:
            top_freq, top_lift = _two_rankings(i)
            npmi, pcov, usage_frac = _stats_for(i)
            print(f"\n--- phenotype {i} (dry-run preview) ---", flush=True)
            print(_build_user_message(
                phenotype_id=i,
                top_by_freq=top_freq, top_by_lift=top_lift,
                alpha=alpha_arr[i], kl=kl_arr[i],
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
    agent = _build_agent(
        model_str, api_key,
        max_words=args.max_words,
        alpha_dist=alpha_dist, kl_dist=kl_dist,
    )
    print(f"[label] using model {model_str}", flush=True)

    total = {"input": 0, "output": 0}
    for n, i in enumerate(todo, start=1):
        top_freq, top_lift = _two_rankings(i)
        npmi, pcov, usage_frac = _stats_for(i)
        try:
            out, usage = _label_one(
                agent=agent,
                phenotype_id=i,
                top_by_freq=top_freq, top_by_lift=top_lift,
                alpha=alpha_arr[i], kl=kl_arr[i],
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

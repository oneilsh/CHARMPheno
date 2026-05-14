"""Generate concise human-readable labels for phenotypes in a dashboard bundle.

For each phenotype, picks the top-N codes by Sievert-Shirley lambda-relevance
(matches the dashboard's λ=0.6 default), sends them to a Claude model, and
writes the returned label back into phenotypes.json's `label` field.

By design, the script does NOT use the standard ANTHROPIC_API_KEY env var.
Instead it reads CHARMPHENO_LABEL_KEY (rename via --api-key-env), or a key
file path (--api-key-file). This keeps the labeling key isolated from other
Anthropic SDK callers (notably Claude Code) that might be active in the
same shell.

Sources of the key, in order:
  1. --api-key-file <path>            (file contents)
  2. --api-key-env <NAME>             (process env, default CHARMPHENO_LABEL_KEY)
  3. .env file (auto-loaded)          (only populates THIS process's env,
                                       does not export to the parent shell)

Even if your .env happens to also contain ANTHROPIC_API_KEY, this script
ignores it — and the SDK client is constructed with api_key= explicitly,
so the SDK's own env-var fallback is never triggered.

Usage:
    # one-time install of the labeling deps
    poetry install --with labeling

    # option A: .env in the repo root
    echo 'CHARMPHENO_LABEL_KEY=sk-ant-...' >> .env
    poetry run python scripts/label_phenotypes.py \\
        --bundle-dir dashboard/public/data

    # option B: env var
    export CHARMPHENO_LABEL_KEY=sk-ant-...
    poetry run python scripts/label_phenotypes.py \\
        --bundle-dir dashboard/public/data

    # option C: key file (never enters env at all)
    poetry run python scripts/label_phenotypes.py \\
        --bundle-dir dashboard/public/data \\
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


DEFAULT_API_KEY_ENV = "CHARMPHENO_LABEL_KEY"
DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_TOP_N = 15
DEFAULT_LAMBDA = 0.6
DEFAULT_MAX_WORDS = 6

# System prompt is stable across all phenotypes -> cache it once and reread
# on every subsequent request. The breakpoint goes on the system block so
# any per-phenotype variation in the user message doesn't invalidate it.
SYSTEM_PROMPT = """\
You are a clinical informatics expert summarizing learned phenotypes from \
an LDA topic model trained over OMOP healthcare records.

You will be given a phenotype's top weighted codes (concept descriptions \
ranked by Sievert-Shirley relevance — a mix of frequency within the topic \
and lift over the corpus). Each code is one of: condition, drug, procedure, \
measurement, or observation.

Your task: produce a single concise label that captures the clinical theme \
shared by these codes.

Guidelines:
- Use clinical terminology a clinician would recognize ("Type 2 diabetes \
care", "Postoperative orthopedic recovery"), not lay phrasing.
- Be specific. "Cardiovascular" alone is too vague; "Heart failure & \
arrhythmia" is better.
- If the codes do not form a coherent clinical theme (mixed unrelated \
diagnoses, generic preventive codes, or apparent noise), set \
`is_coherent` to false and use a label like "Mixed / low-coherence".
- Avoid hedging language ("Possibly...", "Likely related to..."). State \
the theme plainly.
- Do not mention the model, the codes, or this prompt in the label.\
"""

# Constrains the response to a clean JSON object we can parse and assign.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "description": "Concise clinical label, up to MAX_WORDS words.",
        },
        "is_coherent": {
            "type": "boolean",
            "description": "False if the codes do not form a coherent theme.",
        },
    },
    "required": ["label", "is_coherent"],
    "additionalProperties": False,
}


def _maybe_load_dotenv(explicit_path: str | None) -> Path | None:
    """Load a .env file into THIS process's os.environ. Does not export to
    the parent shell. Searches, in order:
      1. --env-file <path> if explicitly passed
      2. .env in CWD
      3. .env at the repo root (inferred from this script's path)
    Returns the path that was loaded, or None if no .env was found / no
    dotenv lib is installed.

    Silent on import failure — dotenv is an optional convenience, the
    script still works fine if it's not installed.
    """
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        return None

    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    else:
        candidates.append(Path.cwd() / ".env")
        # scripts/label_phenotypes.py -> repo root
        candidates.append(Path(__file__).resolve().parent.parent / ".env")

    for c in candidates:
        if c.is_file():
            # override=False so a real env var still wins over the file —
            # matters if the user wants to temporarily override .env for
            # one invocation without editing the file.
            load_dotenv(c, override=False)
            return c
    return None


def _resolve_api_key(args: argparse.Namespace) -> str:
    """Get the API key from --api-key-file or the configured env var.

    Explicitly does NOT fall back to ANTHROPIC_API_KEY — keeping the
    labeling key in its own namespace avoids accidental cross-billing
    with other Anthropic SDK callers running in the same shell.
    """
    if args.api_key_file:
        p = Path(args.api_key_file).expanduser()
        if not p.is_file():
            raise SystemExit(f"--api-key-file not found: {p}")
        key = p.read_text().strip()
        if not key:
            raise SystemExit(f"--api-key-file is empty: {p}")
        return key
    key = os.environ.get(args.api_key_env)
    if not key:
        raise SystemExit(
            f"No API key found.\n"
            f"  Looked for: {args.api_key_env} env var, --api-key-file, "
            f"or a .env file with {args.api_key_env}=...\n"
            f"  (This script intentionally does not read ANTHROPIC_API_KEY.)"
        )
    return key


def _lambda_relevance(pwk: float, pw: float, lam: float) -> float:
    """Sievert-Shirley relevance: lam * log p(w|k) + (1 - lam) * log(p(w|k)/p(w)).

    Matches dashboard/src/lib/inference.ts. Returns -inf if pwk <= 0 (the
    code does not appear in the topic).
    """
    if pwk <= 0:
        return -math.inf
    log_pwk = math.log(pwk)
    if pw <= 0:
        # Code never appears in the corpus; lift is infinite. Saturate to a
        # large finite value rather than -inf so sorting still works.
        return lam * log_pwk + (1.0 - lam) * 1e6
    return lam * log_pwk + (1.0 - lam) * math.log(pwk / pw)


def _top_codes_for_phenotype(
    *,
    beta_row: list[float],
    vocab: list[dict],
    lam: float,
    n: int,
) -> list[dict]:
    """Pick top-N codes for a phenotype by relevance. vocab is the
    full vocab.json `codes` array (index-aligned with beta columns)."""
    pw = [c.get("corpus_freq", 0.0) for c in vocab]
    scored = [
        (i, _lambda_relevance(beta_row[i], pw[i], lam))
        for i in range(len(beta_row))
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    out: list[dict] = []
    for idx, _score in scored[:n]:
        c = vocab[idx]
        out.append({
            "code": c.get("code"),
            "description": c.get("description") or c.get("code"),
            "domain": c.get("domain", "unknown"),
            "weight_pct": round(beta_row[idx] * 100.0, 3),
        })
    return out


def _build_user_message(*, phenotype_id: int, top_codes: list[dict],
                        max_words: int) -> str:
    """Per-phenotype user prompt. Everything that changes per phenotype
    lives here; the cached system prompt above stays byte-identical."""
    lines = [
        f"Phenotype id: {phenotype_id}",
        f"Label budget: at most {max_words} words.",
        "",
        "Top codes (description · domain · within-topic weight):",
    ]
    for r, c in enumerate(top_codes, start=1):
        lines.append(
            f"  {r:>2}. {c['description']}  ·  {c['domain']}  "
            f"·  {c['weight_pct']:.2f}%"
        )
    return "\n".join(lines)


def _label_one(
    *,
    client,
    model: str,
    phenotype_id: int,
    top_codes: list[dict],
    max_words: int,
) -> tuple[str, bool, dict]:
    """Make one Messages API call. Returns (label, is_coherent, usage_dict)."""
    user_text = _build_user_message(
        phenotype_id=phenotype_id, top_codes=top_codes, max_words=max_words,
    )
    resp = client.messages.create(
        model=model,
        max_tokens=128,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_text}],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": RESPONSE_SCHEMA,
            }
        },
    )
    text = next(b.text for b in resp.content if b.type == "text")
    parsed = json.loads(text)
    usage = {
        "input": resp.usage.input_tokens,
        "output": resp.usage.output_tokens,
        "cache_read": resp.usage.cache_read_input_tokens,
        "cache_write": resp.usage.cache_creation_input_tokens,
    }
    return parsed["label"].strip(), bool(parsed["is_coherent"]), usage


def _write_atomic(path: Path, data: dict) -> None:
    """Write JSON via temp-file + rename so a crash mid-write can't truncate
    the bundle the dashboard is reading."""
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
        "--model", default=DEFAULT_MODEL,
        help="Anthropic model id for labeling",
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
        "--api-key-env", default=DEFAULT_API_KEY_ENV,
        help="Env var to read the API key from. Default is a non-standard "
             "name so it never collides with ANTHROPIC_API_KEY.",
    )
    parser.add_argument(
        "--api-key-file", type=str, default=None,
        help="Path to a file containing the API key (alternative to env var).",
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

    # Pull .env into this process's environ before reading any *_KEY var.
    # No-op (silent) if python-dotenv isn't installed or no .env exists.
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

    # Plan the work — skip phenotypes that already have a label unless --force.
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

    if args.dry_run:
        for i in todo[:3]:
            top = _top_codes_for_phenotype(
                beta_row=beta[i], vocab=vocab_codes,
                lam=args.lam, n=args.top_n,
            )
            print(f"\n--- phenotype {i} (dry-run preview) ---", flush=True)
            print(_build_user_message(
                phenotype_id=i, top_codes=top, max_words=args.max_words,
            ), flush=True)
        if len(todo) > 3:
            print(f"\n... and {len(todo) - 3} more "
                  f"(use --limit for a partial real run)", flush=True)
        print(f"\n[label] dry-run only; no API calls made", flush=True)
        return 0

    # Resolve the key only when we actually need to make calls.
    try:
        import anthropic  # noqa: WPS433
    except ImportError:
        raise SystemExit(
            "anthropic SDK not installed. Run: poetry install --with labeling"
        )

    api_key = _resolve_api_key(args)
    client = anthropic.Anthropic(api_key=api_key)

    total = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    for n, i in enumerate(todo, start=1):
        top = _top_codes_for_phenotype(
            beta_row=beta[i], vocab=vocab_codes,
            lam=args.lam, n=args.top_n,
        )
        try:
            label, coherent, usage = _label_one(
                client=client, model=args.model,
                phenotype_id=i, top_codes=top, max_words=args.max_words,
            )
        except anthropic.APIError as e:
            print(f"[label] phenotype {i}: API error: {e}", flush=True)
            # Persist what we've labeled so far before re-raising.
            _write_atomic(phens_p, phens_b)
            raise

        for k in total:
            total[k] += usage[k]

        phenotypes[i]["label"] = label
        # If the model judged the cluster incoherent, prefer that signal over
        # whatever junk_flag was set by NPMI alone.
        if not coherent:
            phenotypes[i]["junk_flag"] = True

        cache_hit = "HIT " if usage["cache_read"] else "miss"
        print(
            f"[label] {n:>3}/{len(todo)}  k={i:<3}  "
            f"[{cache_hit}]  {label}",
            flush=True,
        )

    # One final atomic write at the end.
    _write_atomic(phens_p, phens_b)

    print(
        f"\n[label] wrote {len(todo)} labels to {phens_p}\n"
        f"[label] tokens — input {total['input']}, output {total['output']}, "
        f"cache_read {total['cache_read']}, cache_write {total['cache_write']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

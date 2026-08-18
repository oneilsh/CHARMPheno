"""Compact post-fit report for a gated_pc run — so a run can be pasted back to chat
without the multi-thousand-line per-parent tables.

Reads a run's ``summary.md`` (the full sanitized stdout ``run_experiment`` captures)
and emits a short, signal-DENSE digest:

  * a WEIRDNESS scan up top (tracebacks, OOM / maxResultSize, lost executors / bad
    nodes, non-zero exits, NaN/inf, |w| blowup, a STARVED supervised correction);
  * a FIT-HEALTH trajectory (ELBO, |w_CK|max, corr_relΔλ at first / mid / last / peak);
  * then the kept signal lines VERBATIM in order — [mondo], ledger, [cost], the
    per-arm readout summary lines, the per-depth conditional-AUC tables, the
    head-formulation ladder, the HEADLINE shaping ablation.

It DROPS the bulk that eats context: the per-parent multiclass tables (``top1=… /
majority=…`` rows), the per-node domain λ-mass tables, and the per-iteration α/η/λ
diagnostic spam (kept only as the sampled trajectory). Whole lines are kept as-is —
no re-derivation — so it can't silently mis-report a number.

Run:  make -C analysis/cloud report ID=91
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

# Lines that are the BULK (dropped from the verbatim echo; the trajectory extractor
# below still reads the per-iter lines before they are dropped).
_DROP = (
    lambda ln: "top1=" in ln and "majority=" in ln,               # per-parent table row
    lambda ln: re.search(r"\S+\s+\d\.\d{3}\s+\d\.\d{3}\s+\d\.\d{3}\s*$", ln)  # λ-mass row
    is not None,
    lambda ln: "α[min=" in ln or "diagnostics: alpha=" in ln,     # per-iter diag spam
    lambda ln: re.search(r"iter \d+/\d+: ELBO=", ln) is not None,  # per-iter ELBO line
    lambda ln: ln.strip() in ("", "..."),                         # blanks / paste elisions
)

# Weirdness patterns (case-insensitive substring / regex over the whole log).
_FLAG_PATTERNS = (
    ("traceback", r"Traceback \(most recent call last\)"),
    ("exception", r"\bError\b|\bException\b|py4j\.protocol\.Py4JJavaError"),
    ("oom", r"OutOfMemory|maxResultSize|Container killed|memory limit"),
    ("executor-loss", r"Lost executor|bad node|Killed by external signal"),
    ("nonzero-exit", r"exited non-zero|non-zero exit|Error 1\d\d\b"),
    ("nan/inf", r"\bnan\b|\bNaN\b|[^A-Za-z]inf[^A-Za-z]"),
)


def _fmt_traj(name, vals, *, lo_is_bad=False, starved=None):
    if not vals:
        return None
    first, last = vals[0], vals[-1]
    peak = max(vals, key=abs)
    n = len(vals)
    mid = vals[n // 2]
    s = (f"  {name:12s} i1={first:.3g}  i{n // 2 + 1}={mid:.3g}  "
         f"i{n}={last:.3g}  peak={peak:.3g}")
    if starved is not None and starved:
        s += "   ⚠ STARVED (~0: supervision not moving topics)"
    return s


def _last_run_section(text: str) -> tuple[str, str]:
    """summary.md is APPEND-ONLY (each fit/eval appends under a ``## `` H2 heading),
    so a file holds many runs. Return (text-of-last-section, its-heading) so the
    report digests only the MOST RECENT run — not a mix of every run's trajectory /
    flags. No heading ⇒ the whole text (a single-run or hand-pasted log)."""
    lines = text.splitlines(keepends=True)
    idx = next((i for i in range(len(lines) - 1, -1, -1)
                if lines[i].startswith("## ")), None)
    if idx is None:
        return text, ""
    return "".join(lines[idx:]), lines[idx].strip()


def build_report(text: str, *, title: str = "", all_sections: bool = False) -> str:
    heading = ""
    if not all_sections:
        text, heading = _last_run_section(text)
    lines = text.splitlines()
    out: list[str] = []

    # --- flags scan ---
    flags: list[str] = []
    low = text
    for label, pat in _FLAG_PATTERNS:
        hits = [m.group(0) for m in re.finditer(pat, low)]
        if hits:
            # one representative + count
            flags.append(f"  {label}: {len(hits)}× e.g. {hits[0].strip()[:80]}")

    # --- trajectory extraction (before dropping the per-iter lines) ---
    elbo = [float(m) for m in re.findall(r"iter \d+/\d+: ELBO=(-?[\d.]+)", text)]
    wmax = [float(m) for m in re.findall(r"\|w_CK\|max=([\d.eE+-]+)", text)]
    corr = [float(m) for m in re.findall(r"corr_relΔλ=([\d.eE+-]+)", text)]
    if wmax and max(wmax) > 200:
        flags.append(f"  head-blowup: |w_CK|max peaked at {max(wmax):.0f} (>200; a "
                     "converged localized head sits ~70)")
    corr_starved = bool(corr) and max(corr) < 1e-3
    if corr_starved:
        flags.append(f"  pc-no-op: corr_relΔλ max={max(corr):.1e} — weight_y not "
                     "shaping topics")

    hdr = f"POST-FIT REPORT  {title}".rstrip()
    out.append("=" * 78)
    out.append(hdr)
    if heading:
        out.append(f"(latest run section: {heading})")
    out.append("=" * 78)
    out.append("FLAGS: " + ("none" if not flags else ""))
    out.extend(flags)

    traj = [t for t in (
        _fmt_traj("ELBO", elbo),
        _fmt_traj("|w_CK|max", wmax),
        _fmt_traj("corr_relΔλ", corr, starved=corr_starved),
    ) if t]
    if traj:
        out.append("FIT-HEALTH (first / mid / last / peak):")
        out.extend(traj)
        if len(elbo) >= 2:
            # ELBO is the variational bound → MAXIMIZED, so a healthy fit RISES
            # (0090: −5.9e7 → −2.6e7). A net fall is the thing to check.
            trend = "rising ✓" if elbo[-1] >= elbo[0] else "⚠ FELL (check divergence)"
            out.append(f"  ELBO trend: {trend}")
    out.append("-" * 78)

    # --- verbatim signal lines (bulk dropped) ---
    for ln in lines:
        if any(pred(ln) for pred in _DROP):
            continue
        out.append(ln)
    return "\n".join(out) + "\n"


def _resolve_summary(args) -> Path:
    if args.summary:
        return Path(args.summary)
    if not args.id:
        raise SystemExit("provide --id N (with --runs-dir) or --summary <path>")
    pat = str(Path(args.runs_dir) / f"{int(args.id):04d}-*" / "summary.md")
    hits = sorted(glob.glob(pat))
    if not hits:
        raise SystemExit(f"no summary.md at {pat}")
    return Path(hits[-1])


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--id", type=int)
    p.add_argument("--runs-dir", default=".")
    p.add_argument("--summary", help="path to a summary.md (overrides --id/--runs-dir)")
    p.add_argument("--all-sections", action="store_true",
                   help="digest the WHOLE append-only summary.md (default: only the "
                        "most recent ## run section)")
    args = p.parse_args(argv)
    path = _resolve_summary(args)
    sys.stdout.write(build_report(
        path.read_text(), title=path.parent.name, all_sections=args.all_sections))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

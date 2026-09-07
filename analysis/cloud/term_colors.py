"""Tasteful ANSI 256-color helpers for cluster-driver log output.

DRIVER-SIDE ONLY. This module is never added to any Makefile `--py-files`
list and must never be imported at MODULE TOP LEVEL from a file whose
functions run executor-side (a mapPartitions kernel, a UDF) — see the
`--py-files` rule in AGENTS.md. A driver that itself rides `--py-files`
(e.g. `hpoa_stage2_probe.py`, which defines its own mapPartitions kernel)
must import this module only inside a function that runs on the driver
(e.g. `main()`), never at the top of the file, so an executor importing
that module to resolve the kernel never needs this module to exist.

The user reads these logs LIVE in a 256-color terminal and has trouble
finding the handful of lines that matter in a wall of driver output: phase
boundaries, the per-iteration progress line, a wiring sentinel token, a
cache hit/miss, and anything gone wrong. Color is a findability aid, not
decoration — see `highlight_line` for the actual rules, applied sparingly.

WHEN COLOR IS EMITTED (decided once, cached — see `enabled`):
    CHARM_COLOR=1   forces color ON regardless of isatty (e.g. `| tee`,
                    or scripts/run_experiment.py relaying a live terminal
                    through a pipe — see its comment at the fit dispatch).
    CHARM_COLOR=0   forces color OFF regardless of isatty.
    NO_COLOR set    (https://no-color.org) forces color OFF, UNLESS
                    CHARM_COLOR=1 overrode it above.
    otherwise       color iff `sys.stdout.isatty()`.

With color disabled every helper here is IDENTITY: piped logs, files, and
pasted terminal text stay byte-identical to today's plain output. This is
the one property every caller of this module leans on, and it is tested
directly in `analysis/cloud/tests/test_term_colors.py`.
"""
from __future__ import annotations

import os
import re
import sys

_RESET = "\x1b[0m"

_BOLD = "1"
_DIM = "2"
_RED = "31"
_GREEN = "32"
_YELLOW = "33"
_CYAN = "36"

_enabled_cache: bool | None = None


def _decide() -> bool:
    override = os.environ.get("CHARM_COLOR")
    if override == "1":
        return True
    if override == "0":
        return False
    if "NO_COLOR" in os.environ:
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        # A stdout that can't answer isatty (some capture shims) is not a
        # human's terminal — default to plain, never raise into a driver.
        return False


def enabled() -> bool:
    """Whether color escapes should be emitted, decided once and cached.

    Cached at first call so the decision can't jitter mid-run — e.g. if
    `_driver_common.install_stdout_tee` swaps `sys.stdout` for a tee object
    later, a call that already happened keeps its answer, and a later call
    reuses it via the delegating `isatty` on the tee (see its `__getattr__`)
    rather than re-deciding against a different-looking stream.
    """
    global _enabled_cache
    if _enabled_cache is None:
        _enabled_cache = _decide()
    return _enabled_cache


def c(text: str, code: str) -> str:
    """Wrap `text` in ANSI SGR `code` (e.g. "1;32"), or return it unchanged
    when color is disabled. This is the ONE place that ever emits an escape
    — every other helper here is a thin wrapper around it."""
    if not enabled():
        return text
    return f"\x1b[{code}m{text}{_RESET}"


def bold(text: str) -> str:
    return c(text, _BOLD)


def dim(text: str) -> str:
    return c(text, _DIM)


def red(text: str) -> str:
    return c(text, _RED)


def green(text: str) -> str:
    return c(text, _GREEN)


def yellow(text: str) -> str:
    return c(text, _YELLOW)


def cyan(text: str) -> str:
    return c(text, _CYAN)


def bold_red(text: str) -> str:
    return c(text, f"{_BOLD};{_RED}")


def bold_green(text: str) -> str:
    return c(text, f"{_BOLD};{_GREEN}")


def bold_cyan(text: str) -> str:
    return c(text, f"{_BOLD};{_CYAN}")


# --------------------------------------------------------------------------- #
# Line-level rules, shared by the logging Formatter (_driver_common) and the  #
# plain print() sites in gated_pc_cloud / gated_pc_readout / hpoa_stage2_probe. #
# Every function here is identity on a non-matching line, and identity        #
# (via `c`) when color is disabled — the substring-absent / disabled cases    #
# are the same code path, not a separate branch, so they can't drift.         #
# --------------------------------------------------------------------------- #

# spark_vi.core.runner's per-iter line: "iter %d/%d: ELBO=..., ..., %.1fs".
# Captured as three pieces so only the iter marker and the trailing timing
# change color; the ELBO/rho/batch middle stays plain and legible.
_ITER_RE = re.compile(r"(iter \d+/\d+)(:.*, )(\d+\.\d+s)$")

# gated_lda.GatedOnlineLDA._eta_boost_summary's wiring sentinel, e.g.
# "..., η_boost[topics=3 nnz=120 mass=45.6]". Model-owned text (spark-vi) —
# never edit that module; this only recolors the string at the print site.
_ETA_BOOST_RE = re.compile(r"η_boost\[[^\]]*\]")

# Cache/bundle status tokens as driver print()s spell them (word-boundary so
# "MISSING" or "prebuild" don't light up by accident). MISS is matched
# case-SENSITIVE (it is a spelled-out sentinel, like HIT — a stray lowercase
# "miss" in ordinary prose must not light up); "rebuild"/"REBUILT" appears in
# both cases across these drivers, so that half is case-insensitive.
_HIT_RE = re.compile(r"\bHIT\b")
_MISS_RE = re.compile(r"\bMISS\b")
_REBUILD_RE = re.compile(r"\brebuil(?:d|ds|ding|t)\b", re.IGNORECASE)


def colorize_iter_line(line: str) -> str:
    """Bold the `iter N/M` marker and dim the trailing per-iter timing on a
    spark_vi progress line; any other line (including one that merely
    mentions "iter" without the runner's exact shape) is returned unchanged."""
    m = _ITER_RE.search(line)
    if not m:
        return line
    marker, middle, timing = m.groups()
    return (line[: m.start()] + bold(marker) + middle + dim(timing)
            + line[m.end():])


def colorize_eta_boost(line: str) -> str:
    """Green-highlight the `η_boost[...]` wiring-sentinel substring, wherever
    it lands in the line; identity when the substring is absent."""
    if not enabled() or "η_boost[" not in line:
        return line
    return _ETA_BOOST_RE.sub(lambda m: green(m.group(0)), line)


def highlight_line(line: str) -> str:
    """The driver print-site rules, applied SPARINGLY and in order:

      1. "ERROR" anywhere         -> bold red, whole line.
      2. "WARN" anywhere          -> yellow, whole line (also catches
                                     "WARNING", which is the spelling this
                                     codebase actually uses).
      3. a readout headline       -> bold green, whole line ("macro AUC="
                                     or "detection (case vs bg)").
      4. otherwise, token-level:  HIT green, MISS/rebuild yellow, and an
                                     eta-boost sentinel green, independently
                                     — a line can carry more than one.

    Identity when nothing matches, and identity (via the underlying `c`)
    when color is disabled — never used on markdown report bodies or
    per-node data dumps, only the driver's own short status lines."""
    if "ERROR" in line:
        return bold_red(line)
    if "WARN" in line:
        return yellow(line)
    if "macro AUC=" in line or "detection (case vs bg)" in line:
        return bold_green(line)
    out = _HIT_RE.sub(lambda m: green(m.group(0)), line)
    out = _MISS_RE.sub(lambda m: yellow(m.group(0)), out)
    out = _REBUILD_RE.sub(lambda m: yellow(m.group(0)), out)
    out = colorize_eta_boost(out)
    return out

"""Tests for the driver-side ANSI color helpers.

`enabled()` caches its decision in module state (by design — see its
docstring), so every test resets the cache first; nothing here talks to
Spark or a real terminal.
"""

import pytest


@pytest.fixture(autouse=True)
def _reset_color_cache(monkeypatch):
    """Isolate the CHARM_COLOR/NO_COLOR env and the cached enabled() decision
    so tests can't see each other's state."""
    import term_colors
    monkeypatch.delenv("CHARM_COLOR", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    term_colors._enabled_cache = None
    yield
    term_colors._enabled_cache = None


def _tty(monkeypatch, value):
    """Make sys.stdout.isatty() return `value` for the duration of a test."""
    import sys
    monkeypatch.setattr(sys.stdout, "isatty", lambda: value, raising=False)


def test_disabled_by_default_when_not_a_tty(monkeypatch):
    import term_colors
    _tty(monkeypatch, False)
    assert term_colors.enabled() is False


def test_enabled_when_isatty_and_nothing_overrides(monkeypatch):
    import term_colors
    _tty(monkeypatch, True)
    assert term_colors.enabled() is True


def test_no_color_wins_over_isatty(monkeypatch):
    """https://no-color.org: NO_COLOR set at all (any value) disables color
    even on a real tty, as long as CHARM_COLOR doesn't override it."""
    import term_colors
    _tty(monkeypatch, True)
    monkeypatch.setenv("NO_COLOR", "1")
    assert term_colors.enabled() is False


def test_charm_color_1_forces_on_over_a_pipe(monkeypatch):
    """The `| tee` / run_experiment.py relay case: stdout is not a tty, but
    the operator (or the relay) forces color anyway."""
    import term_colors
    _tty(monkeypatch, False)
    monkeypatch.setenv("CHARM_COLOR", "1")
    assert term_colors.enabled() is True


def test_charm_color_0_forces_off_on_a_real_tty(monkeypatch):
    import term_colors
    _tty(monkeypatch, True)
    monkeypatch.setenv("CHARM_COLOR", "0")
    assert term_colors.enabled() is False


def test_charm_color_1_overrides_no_color_too(monkeypatch):
    """CHARM_COLOR is the explicit operator override and wins over both the
    NO_COLOR convention and isatty -- it exists precisely to force color
    where the default heuristic would say no."""
    import term_colors
    _tty(monkeypatch, False)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("CHARM_COLOR", "1")
    assert term_colors.enabled() is True


def test_decision_is_cached_across_calls(monkeypatch):
    """enabled() decides once; a later isatty() change doesn't retroactively
    flip it (see the docstring: `install_stdout_tee` can swap sys.stdout
    mid-run)."""
    import term_colors
    _tty(monkeypatch, True)
    assert term_colors.enabled() is True
    _tty(monkeypatch, False)
    assert term_colors.enabled() is True  # still the cached answer


def test_helpers_are_identity_when_disabled(monkeypatch):
    import term_colors
    _tty(monkeypatch, False)
    text = "plain text"
    assert term_colors.c(text, "1;32") == text
    assert term_colors.bold(text) == text
    assert term_colors.dim(text) == text
    assert term_colors.red(text) == text
    assert term_colors.green(text) == text
    assert term_colors.yellow(text) == text
    assert term_colors.cyan(text) == text
    assert term_colors.bold_red(text) == text
    assert term_colors.bold_green(text) == text
    assert term_colors.bold_cyan(text) == text


def test_helpers_emit_escapes_when_enabled(monkeypatch):
    import term_colors
    _tty(monkeypatch, True)
    out = term_colors.green("HIT")
    assert out != "HIT"
    assert out.startswith("\x1b[")
    assert out.endswith("\x1b[0m")
    assert "HIT" in out


def test_c_uses_the_exact_code_given(monkeypatch):
    import term_colors
    _tty(monkeypatch, True)
    assert term_colors.c("x", "1;32") == "\x1b[1;32mx\x1b[0m"


# --------------------------------------------------------------------------- #
# colorize_iter_line: the spark_vi per-iter progress line.                    #
# --------------------------------------------------------------------------- #

_ITER_LINE = "[driver]   iter 3/50: ELBO=-1234.5678, batch=2000, rho=0.1234, 12.3s"


def test_colorize_iter_line_identity_when_disabled(monkeypatch):
    import term_colors
    _tty(monkeypatch, False)
    assert term_colors.colorize_iter_line(_ITER_LINE) == _ITER_LINE


def test_colorize_iter_line_bolds_marker_and_dims_timing(monkeypatch):
    import term_colors
    _tty(monkeypatch, True)
    out = term_colors.colorize_iter_line(_ITER_LINE)
    assert out != _ITER_LINE
    assert term_colors.bold("iter 3/50") in out
    assert term_colors.dim("12.3s") in out
    # the ELBO/rho middle stays untouched, plain text
    assert "ELBO=-1234.5678, batch=2000, rho=0.1234, " in out


def test_colorize_iter_line_leaves_non_matching_lines_untouched(monkeypatch):
    import term_colors
    _tty(monkeypatch, True)
    line = "[driver]   some other status line, nothing to see here"
    assert term_colors.colorize_iter_line(line) == line
    line2 = "iter without the runner's exact shape"
    assert term_colors.colorize_iter_line(line2) == line2


# --------------------------------------------------------------------------- #
# colorize_eta_boost: the eta-boost wiring sentinel from spark_vi model code. #
# --------------------------------------------------------------------------- #

_ETA_LINE = ("[driver]   α[min=0.01 max=0.5], η_boost[topics=3 nnz=120 "
             "mass=45.6]")


def test_colorize_eta_boost_identity_when_disabled(monkeypatch):
    import term_colors
    _tty(monkeypatch, False)
    assert term_colors.colorize_eta_boost(_ETA_LINE) == _ETA_LINE


def test_colorize_eta_boost_highlights_the_token(monkeypatch):
    import term_colors
    _tty(monkeypatch, True)
    out = term_colors.colorize_eta_boost(_ETA_LINE)
    assert out != _ETA_LINE
    assert term_colors.green("η_boost[topics=3 nnz=120 mass=45.6]") in out
    # everything before the token is untouched
    assert out.startswith("[driver]   α[min=0.01 max=0.5], ")


def test_colorize_eta_boost_leaves_non_matching_lines_untouched(monkeypatch):
    import term_colors
    _tty(monkeypatch, True)
    line = "[driver]   iter 3/50: ELBO=-1234.5678, batch=2000, rho=0.1234, 12.3s"
    assert term_colors.colorize_eta_boost(line) == line


# --------------------------------------------------------------------------- #
# highlight_line: the print-site rules shared by gated_pc_cloud /             #
# gated_pc_readout / hpoa_stage2_probe / _case_finding_cache.                 #
# --------------------------------------------------------------------------- #

def test_highlight_line_identity_when_disabled(monkeypatch):
    import term_colors
    _tty(monkeypatch, False)
    lines = [
        "[driver]   ERROR: bundle cache MISS",
        "[episode]   WARNING: sidecar write failed",
        "gated_pc: macro AUC=0.9123 AP=0.5000 (over 400 nodes)",
        "[driver]   case-finding-cache HIT",
        "[readout] cache MISS — rebuilding bundle from manifest",
        "[probe]   bundle HIT: C=812",
        "a line with nothing interesting in it",
    ]
    for line in lines:
        assert term_colors.highlight_line(line) == line


def test_highlight_line_error_is_bold_red_whole_line(monkeypatch):
    import term_colors
    _tty(monkeypatch, True)
    line = "[driver]   ERROR: bundle cache MISS at gs://x/y"
    assert term_colors.highlight_line(line) == term_colors.bold_red(line)


def test_highlight_line_warn_is_yellow_whole_line(monkeypatch):
    import term_colors
    _tty(monkeypatch, True)
    line = "[episode]   WARNING: sidecar write to gs://x failed"
    assert term_colors.highlight_line(line) == term_colors.yellow(line)


def test_highlight_line_readout_headline_is_bold_green_whole_line(monkeypatch):
    import term_colors
    _tty(monkeypatch, True)
    auc_line = "gated_pc (pc_topics_lr): macro AUC=0.9123 AP=0.5000 (over 400 nodes)"
    assert term_colors.highlight_line(auc_line) == term_colors.bold_green(auc_line)
    det_line = ("gated_pc (pc_topics_lr): detection (case vs bg) (812 documents) "
                "AUC=0.8000 AP=0.4000 prev=0.010")
    assert term_colors.highlight_line(det_line) == term_colors.bold_green(det_line)


def test_highlight_line_hit_token_is_green(monkeypatch):
    import term_colors
    _tty(monkeypatch, True)
    line = "[driver]   case-finding-cache HIT"
    out = term_colors.highlight_line(line)
    assert out != line
    assert term_colors.green("HIT") in out
    assert "[driver]   case-finding-cache " in out  # prefix untouched


def test_highlight_line_miss_and_rebuild_tokens_are_yellow(monkeypatch):
    import term_colors
    _tty(monkeypatch, True)
    miss_line = "[readout] cache MISS — rebuilding bundle from manifest data"
    out = term_colors.highlight_line(miss_line)
    assert term_colors.yellow("MISS") in out
    assert term_colors.yellow("rebuilding") in out

    rebuilt_line = "[readout]   bundle REBUILT (C=812); the sidecar for this run is a HIT"
    out2 = term_colors.highlight_line(rebuilt_line)
    assert term_colors.yellow("REBUILT") in out2
    assert term_colors.green("HIT") in out2


def test_highlight_line_does_not_light_up_partial_words(monkeypatch):
    """Word-boundary matching: MISSING / prebuild must not be mistaken for
    the MISS / rebuild sentinels."""
    import term_colors
    _tty(monkeypatch, True)
    line = "[driver]   the profile field is MISSING from the prebuild config"
    assert term_colors.highlight_line(line) == line


def test_highlight_line_leaves_plain_lines_untouched(monkeypatch):
    import term_colors
    _tty(monkeypatch, True)
    line = "[driver]   corpus: V=(5000) vocab, C=812 nodes, 40000 docs"
    assert term_colors.highlight_line(line) == line

"""Tests for the in-band driver disk telemetry.

No Spark and no sleeping: the ticker's loop is a `while True: _tick(); sleep()`
around a single pure-ish function, so the tests drive `_tick` directly. What
matters here is the three properties the module promises the run it watches:
ONE greppable line per tick, silence about directories that vanish under it,
and never raising into the driver.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def _reset_fail_state():
    """The failure rate-limiter is module state; keep tests independent."""
    from disk_telemetry import _fail_state
    saved = dict(_fail_state)
    _fail_state.update({"ticks": 0, "last_logged": None})
    yield
    _fail_state.update(saved)


def _tree(tmp_path):
    """A small dir tree with two top-level entries and a broadcast pickle."""
    d = tmp_path / "sparktmp"
    (d / "blockmgr-abc").mkdir(parents=True)
    (d / "blockmgr-abc" / "shuffle_0.data").write_bytes(b"x" * 4096)
    (d / "spark-xyz").mkdir()
    (d / "spark-xyz" / "broadcast_7").write_bytes(b"y" * 2048)
    return d


def test_tick_emits_one_line_with_mount_and_du_segments(tmp_path):
    from disk_telemetry import _tick
    d = _tree(tmp_path)
    lines = []
    _tick([str(d)], lines.append)

    assert len(lines) == 1
    line = lines[0]
    assert line.startswith("disk_telemetry:")
    assert "\n" not in line
    # mount segment: mount0(<dir>)=<used>G/<avail>G
    assert f"mount0({d})=" in line
    assert "G/" in line
    # du segment: the top-level entries, in megabytes
    assert "blockmgr-abc=" in line and "M" in line
    assert "spark-xyz=" in line
    # the broadcast pickle is found two levels deep
    assert "bcast_files=1" in line


def test_tick_dedupes_filesystems_by_st_dev(tmp_path):
    """Two dirs on the same filesystem report ONE mount, not the same number
    twice (a master where /tmp, /var/log and the Spark dir all share /)."""
    from disk_telemetry import _tick
    a = tmp_path / "a"; a.mkdir()
    b = tmp_path / "b"; b.mkdir()
    lines = []
    _tick([str(a), str(b)], lines.append)
    assert lines[0].count("mount") == 1


def test_missing_and_vanished_dirs_are_skipped_silently(tmp_path):
    from disk_telemetry import _tick
    d = _tree(tmp_path)
    gone = tmp_path / "does-not-exist"
    lines = []
    _tick([str(gone), str(d)], lines.append)      # must not raise
    assert len(lines) == 1
    assert "tick failed" not in lines[0]
    assert str(gone) not in lines[0]
    assert f"mount0({d})=" in lines[0]            # the real dir still reported


def test_tick_swallows_injected_exception_and_rate_limits(monkeypatch, tmp_path):
    from disk_telemetry import _FAIL_LOG_EVERY, _tick
    d = _tree(tmp_path)

    def _boom(path):
        raise RuntimeError("statvfs exploded\nsecond line of the message")

    monkeypatch.setattr(os, "statvfs", _boom)
    lines = []
    _tick([str(d)], lines.append)                 # must not propagate
    assert len(lines) == 1
    assert lines[0] == ("disk_telemetry: tick failed: RuntimeError: "
                        "statvfs exploded")
    # ...and it must not spam: the next failing ticks stay quiet until
    # _FAIL_LOG_EVERY ticks have passed since the one that logged.
    for _ in range(_FAIL_LOG_EVERY - 1):
        _tick([str(d)], lines.append)
    assert len(lines) == 1
    _tick([str(d)], lines.append)
    assert len(lines) == 2


def test_du_timeout_drops_only_that_segment(monkeypatch, tmp_path):
    """A wedged du costs its segment, not the tick."""
    import subprocess

    from disk_telemetry import _tick
    d = _tree(tmp_path)

    def _timeout(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="du", timeout=60)

    monkeypatch.setattr(subprocess, "run", _timeout)
    lines = []
    _tick([str(d)], lines.append)
    assert len(lines) == 1
    assert "mount0(" in lines[0]
    assert "blockmgr-abc=" not in lines[0]
    assert "bcast_files=1" in lines[0]             # unaffected by du


def test_resolve_dirs_dedupes_and_drops_nonexistent(monkeypatch, tmp_path):
    from disk_telemetry import resolve_dirs
    real = tmp_path / "spark-local"
    real.mkdir()
    monkeypatch.setenv("SPARK_LOCAL_DIRS", f"{real},{tmp_path / 'nope'}")
    dirs = resolve_dirs(extra_dirs=[str(real), str(tmp_path / "also-nope")])
    assert dirs.count(str(real)) == 1
    assert all(os.path.isdir(p) for p in dirs)
    assert not any("nope" in p for p in dirs)


def test_start_disk_telemetry_returns_daemon_thread_and_announces(tmp_path):
    from disk_telemetry import start_disk_telemetry
    d = _tree(tmp_path)
    lines = []
    thread = start_disk_telemetry(extra_dirs=[str(d)], interval_s=3600,
                                  log=lines.append)
    assert thread.daemon
    assert lines[0].startswith("disk_telemetry: watching ")
    assert str(d) in lines[0]
    assert "every 3600s" in lines[0]

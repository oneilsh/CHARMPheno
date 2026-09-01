"""In-band driver-side disk telemetry: one stdout line every couple of minutes.

WHY THIS LIVES IN THE DRIVER PROCESS (and not in a shell loop on the master).
Long whole-Mondo fits/readouts keep dying of driver-local ENOSPC
(`OSError: [Errno 28] No space left on device`, raised from pyspark's
`broadcast.py::dump`). ADR 0047 closed one leak class — per-iteration Python
broadcasts must be `destroy()`ed, not `unpersist()`ed — but runs with every
destroy fix verifiably active still eat ~100 GB of master disk over a few
hours, so a SECOND consumer exists and has to be caught in the act: the
crash's SparkContext shutdown hook deletes the evidence, so a post-hoc `du`
on the corpse tells you nothing.

Twice the ad-hoc `nohup diskwatch` loop that was supposed to catch it wrote to
the master's local `~` and the cluster was torn down before anyone read it —
the autopsy died with the machine. Dataproc's job driver stdout, by contrast,
is persisted to the staging bucket automatically. So the watcher belongs
INSIDE the driver, printing to stdout: it then rides the persisted run log
(and `_driver_common.install_stdout_tee`'s durable `driver_log.md`) with no
operator step to forget and nothing to lose when the cluster goes away.

Design constraints that follow from that:

- **stdlib only, no pyspark import.** This module ships in the same `--py-files`
  payload as executor-side code; an import-time dependency on the driver's
  environment would break that. It has no import-time side effects either.
- **ONE line per tick**, prefixed `disk_telemetry:` — so a whole run's disk
  history is `grep disk_telemetry: driver_log.md` and plots straight from the
  captured log.
- **The thread never raises and never spams.** It runs beside a multi-hour
  solve; a telemetry bug must not be able to kill the run it is protecting, and
  a wedged `du` must not turn into a log flood. The whole tick body is guarded
  and failures are rate-limited to one line per 10 ticks.

What the line discriminates: JVM block-manager state (`blockmgr-*` — the
ContextCleaner-gated suspect for leak #2) from Python broadcast temp files
(`spark-*/pyspark-*`, `broadcast*`) from ordinary log growth, per filesystem.
That is the whole point: the mount totals say the disk is filling, the
top-level `du` says WHO is filling it.
"""
from __future__ import annotations

import glob
import os
import subprocess
import tempfile
import threading
import time

# The Dataproc default Spark local dir. Hard fallback: on a cluster the driver
# JVM's scratch lands here even when SPARK_LOCAL_DIRS is unset in the Python
# process's environment (YARN sets it for the container, not always for us).
_DATAPROC_SPARK_TMP = "/hadoop/spark/tmp"

_TOP_N = 6              # per watched dir, report only the biggest entries
_DU_MAX_ENTRIES = 50    # cap the du argv: a spark tmp dir can hold thousands
_DU_TIMEOUT_S = 60      # a wedged du must not stall the ticker forever
_FAIL_LOG_EVERY = 10    # at most one failure line per this many ticks
_GIB = float(1 << 30)

# Rate-limit state for the tick-failed line. Module-level because the failure
# it guards against is a repeating one (the same bad dir, every tick).
_fail_state = {"ticks": 0, "last_logged": None}


def _default_log(msg):
    """Print to stdout, flushing EVERY line.

    The job driver's stdout is captured by Dataproc but buffered by Python; an
    unflushed line is exactly the line you lose when the process dies of the
    thing you were watching for.
    """
    print(msg, flush=True)


def resolve_dirs(extra_dirs=()):
    """The de-duplicated, existing set of directories to watch.

    Union of: the Spark local dir(s) from ``SPARK_LOCAL_DIRS`` (comma- or
    colon-separated, as Spark writes it), the Dataproc default
    ``/hadoop/spark/tmp``, the process temp dir (where pyspark's
    ``broadcast.py`` writes its pickles when Spark's own dirs are not in play),
    ``/var/log``, and whatever the caller passes (drivers pass
    ``spark.local.dir`` from the live conf, which is the authoritative value
    and may differ from the environment).

    Non-existent entries are dropped here rather than at tick time so the
    startup banner names what is actually being watched.
    """
    cand = []
    for env_val in (os.environ.get("SPARK_LOCAL_DIRS", ""),):
        for part in env_val.replace(":", ",").split(","):
            cand.append(part)
    cand.append(_DATAPROC_SPARK_TMP)
    cand.append(tempfile.gettempdir())
    cand.append("/var/log")
    cand.extend(extra_dirs or ())

    out = []
    seen = set()
    for d in cand:
        # An unset SPARK_LOCAL_DIRS / an empty `spark.local.dir` splits to "",
        # and "" must NOT become "/" — du-ing the root's top level every two
        # minutes is both expensive and not what anyone asked to watch.
        d = (d or "").strip().rstrip("/")
        if not d or d in seen:
            continue
        seen.add(d)
        if os.path.isdir(d):
            out.append(d)
    return out


def _mount_segments(dirs):
    """``mount<n>(<dir>)=<used>G/<avail>G`` for each DISTINCT filesystem.

    De-duplicated by ``st_dev`` so a master where /tmp, /var/log and the Spark
    dir all live on the root filesystem reports one number, not three copies of
    the same number. A dir that vanished between resolve and tick is skipped
    silently (Spark deletes its own scratch on shutdown); a statvfs that fails
    for any other reason is a real anomaly and rides the outer handler.
    """
    segs = []
    seen_dev = set()
    for d in dirs:
        try:
            dev = os.stat(d).st_dev
        except OSError:
            continue
        if dev in seen_dev:
            continue
        seen_dev.add(dev)
        vfs = os.statvfs(d)
        used = (vfs.f_blocks - vfs.f_bfree) * vfs.f_frsize / _GIB
        avail = vfs.f_bavail * vfs.f_frsize / _GIB
        segs.append(f"mount{len(segs)}({d})={used:.1f}G/{avail:.1f}G")
    return segs


def _du_segment(d):
    """``<dir>: name=<M>M ...`` for the ``_TOP_N`` biggest top-level entries.

    One ``du -sm`` over the (capped) top-level entries, not a recursive walk of
    our own: du is the cheap way to get this and the cap bounds it. Returns ""
    when there is nothing to report, and drops the segment (rather than the
    tick) if du times out or is missing.
    """
    entries = sorted(glob.glob(os.path.join(d, "*")))[:_DU_MAX_ENTRIES]
    if not entries:
        return ""
    try:
        proc = subprocess.run(
            ["du", "-sm"] + entries,
            capture_output=True, text=True, timeout=_DU_TIMEOUT_S)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""
    sized = []
    for line in proc.stdout.splitlines():
        # `du -sm` prints "<megabytes>\t<path>"; it also prints per-entry
        # errors to stderr and keeps going, so a partial stdout is normal.
        size, _, path = line.partition("\t")
        try:
            sized.append((int(size), os.path.basename(path.rstrip("/"))))
        except ValueError:
            continue
    if not sized:
        return ""
    sized.sort(reverse=True)
    return d + ": " + " ".join(f"{name}={mb}M" for mb, name in sized[:_TOP_N])


def _bcast_files(dirs):
    """Count files whose basename starts with ``broadcast``, two levels deep.

    pyspark's broadcast pickles land as ``<local-dir>/<spark-app-tmp>/broadcast*``
    (and occasionally one level up). Two globs instead of a recursive walk: the
    Spark scratch tree holds tens of thousands of shuffle blocks and walking it
    every two minutes would itself be the load.
    """
    n = 0
    for d in dirs:
        n += len(glob.glob(os.path.join(d, "broadcast*")))
        n += len(glob.glob(os.path.join(d, "*", "broadcast*")))
    return n


def _quick_used_bytes(dirs):
    """Total used bytes across the watched dirs' DISTINCT filesystems.

    The cheap (statvfs-only) measurement the quiet-mode gate reads every
    interval; the expensive `du` line only runs when this says something moved.
    """
    total = 0
    seen_dev = set()
    for d in dirs:
        try:
            dev = os.stat(d).st_dev
        except OSError:
            continue
        if dev in seen_dev:
            continue
        seen_dev.add(dev)
        vfs = os.statvfs(d)
        total += (vfs.f_blocks - vfs.f_bfree) * vfs.f_frsize
    return total


def _should_print(prev_used, used, ticks_since_print, growth_mb,
                  heartbeat_every):
    """The quiet-mode gate, as a pure function so it is testable.

    Print when: no baseline yet (first tick), the measurement itself failed
    (fail OPEN — an anomaly must surface, not be gated away), used space moved
    by >= `growth_mb` since the last PRINTED line, or `heartbeat_every` ticks
    passed silently (so a reader can still tell the watcher is alive and flat).
    """
    if prev_used is None or used is None:
        return True
    if abs(used - prev_used) >= growth_mb * (1 << 20):
        return True
    return ticks_since_print >= heartbeat_every


def _tick(dirs, log):
    """Emit ONE telemetry line for `dirs`. Never raises.

    Split out from the loop so the whole thing is testable without a thread, a
    sleep, or a SparkSession.
    """
    _fail_state["ticks"] += 1
    try:
        parts = ["disk_telemetry:"]
        parts.extend(_mount_segments(dirs))
        for d in dirs:
            seg = _du_segment(d)
            if seg:
                parts.append("| " + seg)
        parts.append(f"| bcast_files={_bcast_files(dirs)}")
        log(" ".join(parts))
    except Exception as exc:                      # never kill the run we watch
        last = _fail_state["last_logged"]
        if last is None or (_fail_state["ticks"] - last) >= _FAIL_LOG_EVERY:
            _fail_state["last_logged"] = _fail_state["ticks"]
            first_line = str(exc).splitlines()[0] if str(exc) else ""
            log(f"disk_telemetry: tick failed: {type(exc).__name__}: "
                f"{first_line}")


def start_disk_telemetry(extra_dirs=(), interval_s=120, log=None,
                         growth_mb=512, heartbeat_every=15):
    """Start the daemon ticker and return its Thread.

    Always on, no CLI flag: the whole reason it exists is that the run which
    needed it was never the run someone remembered to enable it for. Daemon so
    it never holds the driver open past `main`.

    QUIET BY DEFAULT (2026-09-01, after leak #2 was caught and fixed): a
    healthy run's disk line is flat, and a flat line every 2 minutes is log
    noise that makes the interesting lines harder to copy out of a paste. The
    ticker still MEASURES every `interval_s` (statvfs only — cheap), but only
    PRINTS the full line when used space moved by >= `growth_mb` since the
    last printed line, on a measurement failure (fail open), or as an
    every-`heartbeat_every`-ticks liveness line (default 15 x 120s = one line
    per ~30 quiet minutes). A recurrence of the leak class therefore still
    hits the log within one interval of its first half-gigabyte.

    `extra_dirs` is for the live Spark conf's ``spark.local.dir`` (the drivers
    pass it right after the session exists) — the conf value wins over the
    environment on a cluster where they disagree.
    """
    if log is None:
        log = _default_log
    dirs = resolve_dirs(extra_dirs)
    log(f"disk_telemetry: watching {' '.join(dirs) or '(none)'} "
        f"every {int(interval_s)}s (prints on >={int(growth_mb)}M movement "
        f"or every {int(heartbeat_every)} quiet ticks)")

    def _loop():
        # Tick FIRST: the launch-time baseline is what every later line is
        # read against, and a run that dies in its first two minutes still
        # leaves one measurement behind.
        prev_used = None
        since = heartbeat_every
        while True:
            try:
                used = _quick_used_bytes(dirs)
            except Exception:
                used = None
            since += 1
            if _should_print(prev_used, used, since, growth_mb,
                             heartbeat_every):
                _tick(dirs, log)
                prev_used = used
                since = 0
            time.sleep(interval_s)

    thread = threading.Thread(target=_loop, name="disk-telemetry", daemon=True)
    thread.start()
    return thread

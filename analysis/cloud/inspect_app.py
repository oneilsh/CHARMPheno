"""Live polling dashboard for a running Spark app on a Dataproc cluster.

Polls the driver's Spark REST API (localhost:4040) and the YARN
ResourceManager REST API (localhost:8088) and renders a compact refreshing
view: executor utilization + GC health, active stage throughput, recent
completed stages, and per-worker YARN container/vcore/memory usage.

Use from a JupyterLab terminal on the Dataproc master while a smoke
target is running in a sibling terminal. No install — stdlib only.

    python inspect_app.py                  # auto-detect the running app
    python inspect_app.py --interval 3
    python inspect_app.py --app-id application_..._0004
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

YARN_BASE = "http://localhost:8088/ws/v1/cluster"
CLEAR = "\x1b[2J\x1b[H"


_POLL_TIMEOUT = 15.0  # generous: live driver can be slow when locked in a long stage


def _get(url: str, timeout: float = _POLL_TIMEOUT):
    with urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def discover_spark_base(host: str | None,
                         port: int | None) -> str | None:
    """Probe the live driver UI for a Spark REST API base URL.

    Spark binds the driver UI to whichever host `spark.driver.host`
    resolves to (often the FQDN, not localhost) on the first free port
    starting at 4040. We shotgun across the usual host aliases × the
    standard port range and return the first that responds.
    """
    hosts = [host] if host else ["localhost", "127.0.0.1", socket.gethostname()]
    live_ports = [port] if port else list(range(4040, 4046))
    for h in hosts:
        for p in live_ports:
            url = f"http://{h}:{p}/api/v1"
            try:
                _get(f"{url}/applications", timeout=1.5)
                return url
            except (HTTPError, URLError, ConnectionError, OSError):
                continue
    return None


def _fmt_bytes(b: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024:
            return f"{b:>6.1f}{u}"
        b /= 1024
    return f"{b:>6.1f}PB"


def _fmt_ms(ms: float) -> str:
    if ms < 1000:
        return f"{ms:>4.0f}ms"
    s = ms / 1000
    if s < 60:
        return f"{s:>5.1f}s"
    return f"{s/60:>5.1f}m"


# Errors that should NOT take down the loop — the live driver REST endpoint
# can stall transiently when the driver thread is busy (long stages, GC).
_NETWORK_ERRORS = (HTTPError, URLError, ConnectionError, TimeoutError, OSError)


def find_app_id(spark_base: str, explicit: str | None) -> str | None:
    """Pick the newest in-progress app on the live driver UI.

    Returns None if no in-progress app is listed. The live driver UI
    only exists for the duration of an actual driver process, so any
    listed app is by definition still running — no zombie filtering
    needed.
    """
    if explicit:
        return explicit
    try:
        apps = _get(f"{spark_base}/applications")
    except _NETWORK_ERRORS:
        return None
    if not apps:
        return None

    def _start(a):
        return (a.get("attempts") or [{}])[0].get("startTimeEpoch", 0)

    in_progress = [
        a for a in apps
        if not (a.get("attempts") or [{}])[0].get("completed", True)
    ]
    if not in_progress:
        return None
    in_progress.sort(key=_start, reverse=True)
    return in_progress[0]["id"]


def _sustained_cpu_pct(e: dict) -> float:
    """Fraction of wallclock that this executor's cores spent running tasks.

    `totalDuration` is the sum across all tasks ever assigned. Dividing by
    (lifetime * cores) gives the sustained utilization since `addTime` —
    the cleanest "is this CPU-bound" signal Spark exposes. Near 1.0 means
    the executor was busy ~always; <0.5 means the bottleneck is elsewhere
    (driver-side compute, shuffle wait, scheduling gaps).
    """
    cores = e.get("totalCores", 0) or 0
    total_dur_ms = e.get("totalDuration", 0) or 0
    add_str = e.get("addTime", "")
    if not (cores and add_str):
        return 0.0
    try:
        # Spark formats as "2026-05-04T18:55:30.123GMT"; isoformat needs an offset.
        add_dt = datetime.fromisoformat(add_str.replace("GMT", "+00:00"))
        wall_ms = time.time() * 1000 - add_dt.timestamp() * 1000
    except (ValueError, TypeError):
        return 0.0
    if wall_ms <= 0:
        return 0.0
    return total_dur_ms / (wall_ms * cores)


def render_executors(execs: list) -> None:
    print("Executors  (cpu% sustained since addTime; util% = right now)")
    for e in execs:
        cores = e.get("totalCores", 0) or 0
        active = e.get("activeTasks", 0) or 0
        util = (active / cores) if cores else 0.0
        cpu = _sustained_cpu_pct(e)
        gc_pct = (
            100 * e["totalGCTime"] / e["totalDuration"]
            if e.get("totalDuration") else 0.0
        )
        print(f"  {e['id']:<8}  cores={cores:>2}  active={active:>2}/{cores:<2}"
              f"  util={util:>4.0%}  cpu={cpu:>4.0%}"
              f"  done={e.get('completedTasks',0):>5}"
              f"  failed={e.get('failedTasks',0):>3}"
              f"  GC={gc_pct:>4.1f}%"
              f"  mem={_fmt_bytes(e.get('memoryUsed',0))}")


def render_active_stages(stages: list) -> None:
    print("\nActive stages  (throughput = (input + shuffleRead) / executorRunTime)")
    if not stages:
        print("  (none — driver between stages, doing local work, or post-fit)")
        return
    for s in stages[:5]:
        done = s.get("numCompleteTasks", 0)
        total = s.get("numTasks", 0)
        in_b = s.get("inputBytes", 0)
        shR = s.get("shuffleReadBytes", 0)
        shW = s.get("shuffleWriteBytes", 0)
        run_ms = max(s.get("executorRunTime", 0), 1)
        thr = (in_b + shR) * 1000 / run_ms
        name = s.get("name", "")[:50]
        print(f"  [{s['stageId']:>3}] {done:>4}/{total:<4}"
              f"  in={_fmt_bytes(in_b)}  shR={_fmt_bytes(shR)}  shW={_fmt_bytes(shW)}"
              f"  ~{_fmt_bytes(thr)}/s  {name}")


def render_recent_complete(stages: list, n: int = 3) -> None:
    completed = [s for s in stages if s.get("status") == "COMPLETE"]
    if not completed:
        return
    print(f"\nLast {n} completed stages:")
    for s in sorted(completed, key=lambda s: s["stageId"])[-n:]:
        run_ms = s.get("executorRunTime", 0)
        in_b = s.get("inputBytes", 0)
        shR = s.get("shuffleReadBytes", 0)
        shW = s.get("shuffleWriteBytes", 0)
        thr = (in_b + shR) * 1000 / max(run_ms, 1)
        print(f"  [{s['stageId']:>3}] {s.get('name','')[:38]:<38}"
              f"  exec={_fmt_ms(run_ms)}  in={_fmt_bytes(in_b)}"
              f"  shR={_fmt_bytes(shR)}  shW={_fmt_bytes(shW)}  ~{_fmt_bytes(thr)}/s")


def render_recent_sql(spark_base: str, app_id: str, n: int = 3) -> None:
    """Hit the SQL endpoint and flag broadcast vs sort-merge joins."""
    sqls = _safe_get(f"{spark_base}/applications/{app_id}/sql?length=10")
    if not sqls:
        return
    print(f"\nRecent SQL queries (last {n}):")
    for q in sqls[-n:]:
        # Only ask for the full plan on the few we actually print.
        full = _safe_get(f"{spark_base}/applications/{app_id}/sql/{q['id']}")
        plan = (full or {}).get("planDescription", "")
        flags = []
        if "BroadcastHashJoin" in plan or "BroadcastExchange" in plan:
            flags.append("BROADCAST")
        if "SortMergeJoin" in plan:
            flags.append("sort-merge")
        if "Exchange" in plan and "BroadcastExchange" not in plan:
            flags.append("shuffle")
        flag_s = ",".join(flags) or "—"
        dur = q.get("duration", 0)
        desc = q.get("description", "")[:50]
        print(f"  q{q['id']:>3}  {q.get('status','?'):<8}  {_fmt_ms(dur)}  [{flag_s}]  {desc}")


def render_yarn_nodes() -> None:
    payload = _safe_get(f"{YARN_BASE}/nodes")
    nodes = (payload or {}).get("nodes", {}).get("node", [])
    if not nodes:
        return
    print("\nYARN nodes  (containers/vcores/memory in use vs total available on each)")
    for n in nodes:
        host = n.get("nodeHostName", "?").split(".")[0]
        used_v = n.get("usedVirtualCores", 0)
        avail_v = n.get("availableVirtualCores", 0)
        used_m = n.get("usedMemoryMB", 0)
        avail_m = n.get("availMemoryMB", 0)
        print(f"  {host:<40}  state={n.get('state','?'):<8}"
              f"  containers={n.get('numContainers',0):>2}"
              f"  vcores={used_v}/{used_v + avail_v}"
              f"  mem={used_m}/{used_m + avail_m}MB")


def _safe_get(url: str):
    """Single request; on transient failure, return None instead of raising."""
    try:
        return _get(url)
    except _NETWORK_ERRORS:
        return None


def render_progress(yarn_app: dict | None,
                     execs: list | None,
                     all_stages: list | None,
                     state: dict) -> None:
    """One-line progress summary: app state, elapsed, throughput deltas.

    `state` carries previous-poll values across calls so we can compute
    tasks/sec and stage-rate deltas without keeping a separate timeseries.
    """
    bits = []
    if yarn_app:
        app_state = yarn_app.get("state", "?")
        elapsed_ms = yarn_app.get("elapsedTime", 0)
        progress_pct = yarn_app.get("progress", 0.0)
        bits.append(f"yarn={app_state}")
        bits.append(f"elapsed={_fmt_ms(elapsed_ms)}")
        # YARN's progress for Spark client-mode apps often sits at 10% for the
        # whole run (the AM is a placeholder), so flag it as a hint, not truth.
        bits.append(f"yarn-progress={progress_pct:.0f}%*")

    now = time.time()
    if execs is not None:
        completed = sum(e.get("completedTasks", 0) for e in execs)
        prev_completed = state.get("completed_tasks")
        prev_t = state.get("t")
        if prev_completed is not None and prev_t is not None and now > prev_t:
            rate = (completed - prev_completed) / (now - prev_t)
            bits.append(f"tasks=+{rate:.1f}/s")
        bits.append(f"total_tasks={completed}")
        state["completed_tasks"] = completed
        state["t"] = now

    if all_stages:
        max_stage = max((s.get("stageId", 0) for s in all_stages), default=0)
        prev_stage = state.get("max_stage")
        if prev_stage is not None:
            delta = max_stage - prev_stage
            bits.append(f"stage={max_stage} (+{delta})")
        else:
            bits.append(f"stage={max_stage}")
        state["max_stage"] = max_stage

    if bits:
        print("Progress  " + "  ".join(bits) + "\n")


def render_once(spark_base: str, app_id: str, started: float,
                 state: dict) -> None:
    print(CLEAR, end="")
    print(f"=== {app_id}   {time.strftime('%H:%M:%S')}   "
          f"{time.time() - started:6.1f}s polled   ({spark_base}) ===\n")
    execs = _safe_get(f"{spark_base}/applications/{app_id}/executors")
    active = _safe_get(f"{spark_base}/applications/{app_id}/stages?status=ACTIVE")
    all_stages = _safe_get(f"{spark_base}/applications/{app_id}/stages?length=20")
    yarn_app = _safe_get(f"{YARN_BASE}/apps/{app_id}")
    yarn_app = (yarn_app or {}).get("app")  # unwrap {"app": {...}}
    if execs is None and active is None and all_stages is None:
        print(f"(all spark UI requests timed out — cluster busy, will retry)")
        return

    render_progress(yarn_app, execs, all_stages, state)
    if execs is not None:
        render_executors(execs)
    if active is not None:
        render_active_stages(active)
    if all_stages is not None:
        render_recent_complete(all_stages)
    render_recent_sql(spark_base, app_id)
    render_yarn_nodes()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interval", type=float, default=5.0,
                    help="seconds between refreshes (default: 5)")
    p.add_argument("--app-id", default=None,
                    help="explicit app id (default: first app on the Spark UI)")
    p.add_argument("--host", default=None,
                    help="Spark UI host (default: probe localhost, 127.0.0.1, $HOSTNAME)")
    p.add_argument("--port", type=int, default=None,
                    help="Spark UI port (default: probe 4040..4045)")
    args = p.parse_args()

    spark_base = discover_spark_base(args.host, args.port)
    if not spark_base:
        print("No live Spark driver UI found. Probed:", file=sys.stderr)
        hosts = [args.host] if args.host else ["localhost", "127.0.0.1", socket.gethostname()]
        ports = [args.port] if args.port else list(range(4040, 4046))
        for h in hosts:
            for pn in ports:
                print(f"  http://{h}:{pn}/", file=sys.stderr)
        print("\nFind the actual port with:    ss -tlnp | grep ':40'",
              file=sys.stderr)
        print("Then re-run with:             --host <h> --port <p>",
              file=sys.stderr)
        return 1

    app_id = find_app_id(spark_base, args.app_id)
    if not app_id:
        print(f"Spark UI reachable at {spark_base} but no apps listed.",
              file=sys.stderr)
        return 1
    started = time.time()
    print(f"Polling {app_id} on {spark_base} every {args.interval}s — "
          f"Ctrl-C to quit.")
    time.sleep(0.5)
    try:
        # Re-detect each tick when --app-id wasn't pinned: the newest
        # in-progress app can change between iterations (a new job may
        # be submitted after inspect was launched). Without re-detection
        # we'd stick on the previous app and never pick up the new one.
        last_app_id = app_id
        # Per-app cross-poll state for delta metrics (tasks/sec, stage rate).
        # Reset whenever the target app changes so deltas don't span jobs.
        state: dict = {}
        while True:
            current = (args.app_id
                       if args.app_id
                       else find_app_id(spark_base, None) or last_app_id)
            if current != last_app_id:
                print(f"\n*** switching: {last_app_id} -> {current} ***\n",
                      flush=True)
                last_app_id = current
                state = {}
            render_once(spark_base, current, started, state)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

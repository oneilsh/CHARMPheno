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
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

YARN_BASE = "http://localhost:8088/ws/v1/cluster"
CLEAR = "\x1b[2J\x1b[H"


def _get(url: str, timeout: float = 3.0):
    with urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def discover_spark_base(host: str | None,
                         port: int | None) -> tuple[str | None, str]:
    """Probe live driver UIs first, then fall back to the History Server.

    Returns (base_url, mode) where mode is "live" (driver UI on 4040+)
    or "history" (Spark History Server on 18080). The History Server
    serves the same REST shape but its data is event-log-flush delayed
    (~5s on Dataproc due to GCS hflush rate limits).
    """
    hosts = [host] if host else ["localhost", "127.0.0.1", socket.gethostname()]
    live_ports = [port] if port else list(range(4040, 4046))
    for h in hosts:
        for p in live_ports:
            url = f"http://{h}:{p}/api/v1"
            try:
                _get(f"{url}/applications", timeout=1.5)
                return url, "live"
            except (HTTPError, URLError, ConnectionError, OSError):
                continue
    # History Server fallback. Skip if the user pinned a specific port that
    # isn't 18080 — they're probably trying to hit a specific live driver.
    if port is None or port == 18080:
        for h in (hosts if host else ["localhost"]):
            url = f"http://{h}:18080/api/v1"
            try:
                _get(f"{url}/applications", timeout=1.5)
                return url, "history"
            except (HTTPError, URLError, ConnectionError, OSError):
                continue
    return None, "unknown"


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


def find_app_id(spark_base: str, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    try:
        apps = _get(f"{spark_base}/applications")
    except (HTTPError, URLError, ConnectionError):
        return None
    if not apps:
        return None
    # Prefer an in-progress app (matters for History Server which lists all).
    for a in apps:
        attempts = a.get("attempts") or [{}]
        if not attempts[0].get("completed", True):
            return a["id"]
    return apps[0]["id"]


def render_executors(execs: list) -> None:
    print("Executors  (active/cores = current utilization, GC% from totalGCTime/totalDuration)")
    for e in execs:
        cores = e.get("totalCores", 0) or 0
        active = e.get("activeTasks", 0) or 0
        util = (active / cores) if cores else 0.0
        gc_pct = (
            100 * e["totalGCTime"] / e["totalDuration"]
            if e.get("totalDuration") else 0.0
        )
        print(f"  {e['id']:<8}  cores={cores:>2}  active={active:>2}/{cores:<2}"
              f"  util={util:>4.0%}  done={e.get('completedTasks',0):>5}"
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
    try:
        sqls = _get(f"{spark_base}/applications/{app_id}/sql?length=10")
    except (HTTPError, URLError):
        return
    if not sqls:
        return
    print(f"\nRecent SQL queries (last {n}):")
    for q in sqls[-n:]:
        # Only ask for the full plan on the few we actually print.
        try:
            full = _get(f"{spark_base}/applications/{app_id}/sql/{q['id']}")
            plan = full.get("planDescription", "")
        except (HTTPError, URLError):
            plan = ""
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
    try:
        payload = _get(f"{YARN_BASE}/nodes")
        nodes = payload.get("nodes", {}).get("node", [])
    except (HTTPError, URLError, KeyError, ConnectionError):
        return
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


def render_once(spark_base: str, app_id: str, started: float) -> None:
    print(CLEAR, end="")
    print(f"=== {app_id}   {time.strftime('%H:%M:%S')}   "
          f"{time.time() - started:6.1f}s polled   ({spark_base}) ===\n")
    try:
        execs = _get(f"{spark_base}/applications/{app_id}/executors")
        active = _get(f"{spark_base}/applications/{app_id}/stages?status=ACTIVE")
        all_stages = _get(f"{spark_base}/applications/{app_id}/stages?length=20")
    except (HTTPError, URLError, ConnectionError) as e:
        print(f"(spark UI unreachable at {spark_base}: {e})")
        return

    render_executors(execs)
    render_active_stages(active)
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

    spark_base, mode = discover_spark_base(args.host, args.port)
    if not spark_base:
        print("No Spark UI or History Server found. Probed:", file=sys.stderr)
        hosts = [args.host] if args.host else ["localhost", "127.0.0.1", socket.gethostname()]
        ports = [args.port] if args.port else list(range(4040, 4046)) + [18080]
        for h in hosts:
            for pn in ports:
                print(f"  http://{h}:{pn}/", file=sys.stderr)
        print("\nFind the actual port with:    ss -tlnp | grep -E ':4|:18080'",
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
    note = "  (event-log lag ~5s)" if mode == "history" else ""
    print(f"Polling {app_id} on {spark_base} [{mode}{note}] "
          f"every {args.interval}s — Ctrl-C to quit.")
    time.sleep(0.5)
    try:
        while True:
            render_once(spark_base, app_id, started)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

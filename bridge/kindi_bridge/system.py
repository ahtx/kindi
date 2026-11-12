# Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess

import psutil


def processes():
    rows = []
    for p in psutil.process_iter(["pid", "ppid", "name", "username", "memory_info", "cpu_percent", "create_time", "cmdline"]):
        try:
            x = p.info
            rows.append(dict(id=f'{x["pid"]}:{x["create_time"]}', pid=x["pid"], born=x["create_time"],
                             ppid=x["ppid"], name=x["name"] or "?", user=x["username"] or "?",
                             cpu=x["cpu_percent"] or 0, mb=(x["memory_info"].rss / 1048576 if x["memory_info"] else 0),
                             command=" ".join(x["cmdline"] or []) or x["name"]))
        except (psutil.Error, TypeError):
            continue
    return sorted(rows, key=lambda p: (-p["mb"], p["pid"]))


def listeners():
    rows = []
    for c in psutil.net_connections(kind="inet"):
        if not c.laddr or (c.type == 1 and c.status != psutil.CONN_LISTEN):
            continue
        # UDP has bound sockets rather than a LISTEN state.
        try:
            p = psutil.Process(c.pid) if c.pid else None
            born, name = (p.create_time(), p.name()) if p else (None, "unknown (permissions)")
        except psutil.Error:
            born, name = None, "exited/unknown"
        proto = "TCP" if c.type == 1 else "UDP"
        host, port = c.laddr.ip, c.laddr.port
        rows.append(dict(id=f"{proto}:{host}:{port}:{c.pid}:{born}", proto=proto, host=host, port=port,
                         pid=c.pid, born=born, name=name,
                         exposure="wildcard" if host in ("0.0.0.0", "::") else "bound"))
    return sorted(rows, key=lambda r: (r["port"], r["host"], r["proto"]))


def terminate(pid, born, force=False):
    """Signal precisely the inspected process, refusing PID reuse and control-path processes."""
    if not pid or born is None:
        raise ValueError("Process owner is unknown; cannot signal it")
    p = psutil.Process(int(pid))
    if p.create_time() != born:
        raise ValueError("Process changed since selection; refresh and select again")
    protected = {os.getpid(), 1, *(p.pid for p in psutil.Process().parents())}
    if p.pid in protected or p.name() in {"sshd", "tmux: server", "systemd"}:
        raise ValueError("Protected control/session process")
    p.kill() if force else p.terminate()


def metrics():
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(str(Path.home()))
    return f"CPU {psutil.cpu_percent():.0f}%  RAM {mem.percent:.0f}%  disk {disk.percent:.0f}%  load {'/'.join(f'{x:.1f}' for x in os.getloadavg()) if hasattr(os, 'getloadavg') else '-'}"


def gpu_metrics():
    return run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv,noheader"], timeout=3)


def run(args, timeout=5):
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError((p.stderr or p.stdout).strip()[:1000])
    return p.stdout


def schedule_inventory():
    parts = []
    for label, cmd in [("User cron", ["crontab", "-l"]), ("System timers", ["systemctl", "list-timers", "--all", "--no-pager"])]:
        try:
            parts.append(label + "\n" + run(cmd))
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as e:
            parts.append(f"{label}: {e}")
    return "\n\n".join(parts)

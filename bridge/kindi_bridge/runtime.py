# Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time

from .store import Store, state_dir
from .system import run

CHECK_PROMPT = "Was the previous job successful? If not, plan an alternate strategy and resolve it. If it is complete, confirm completion and give the evidence."
_service_processes = []


def pi_command():
    pi = shutil.which("pi")
    if not pi:
        matches = sorted((Path.home() / ".nvm/versions/node").glob("*/bin/pi"), reverse=True)
        pi = str(matches[0]) if matches else "pi"
    return [pi, "--mode", "rpc"]


def child_env(command=None):
    env = os.environ.copy()
    if command and Path(command[0]).is_absolute():
        env["PATH"] = str(Path(command[0]).parent) + os.pathsep + env.get("PATH", "")
    return env


TMUX_CONF = Path(__file__).parent / "assets" / "tmux.conf"


def tmux(*args):
    # -f applies only when this call starts the server: then extended keys are on before the first pane (pi) starts
    return run(["tmux", "-L", os.environ.get("BRIDGE_TMUX_SOCKET", "kindi-bridge"), "-f", str(TMUX_CONF), *args])


def configure_tmux():
    tmux("set-option", "-s", "extended-keys", "on")   # modified Enter keys reach pi (tmux 3.2+)
    tmux("set-option", "-t", "agents", "status-right", "Ctrl-b ?:Menu | v:Tile z:Zoom r:Refit d:Back")
    tmux("set-window-option", "-t", "agents", "window-size", "latest")
    for key, command in [("v", ["select-layout", "tiled"]), ("r", ["select-layout", "tiled"]), ("z", ["resize-pane", "-Z"])]:
        tmux("bind-key", key, *command)
    for key, direction in [("Left","L"),("Right","R"),("Up","U"),("Down","D")]:
        tmux("bind-key", "-r", key, "resize-pane", "-"+direction, "3")
    tmux("bind-key", "?", "display-menu", "-T", "Kindi Bridge", "Choose agent", "a", "choose-tree -Z", "Tile", "v", "select-layout tiled", "Zoom", "z", "resize-pane -Z", "Refit", "r", "select-layout tiled", "Return to Bridge", "d", "detach-client")


def launch_agent(store: Store, name):
    agent = store.agent(name)
    if store.setting('native_workbench')=='1':
        from .workbench import launch_native
        return launch_native(store,agent)
    if agent["heartbeat"] and time.time() - agent["heartbeat"] < 10:
        return agent
    command = shlex.join([sys.executable, "-m", "kindi_bridge", "worker", agent["id"]])
    env = f"BRIDGE_STATE={shlex.quote(str(state_dir()))} "
    try:
        tmux("has-session", "-t", "agents")
        pane = tmux("split-window", "-d", "-P", "-F", "#{pane_id}", "-t", "agents", "-c", agent["cwd"], env + command).strip()
    except RuntimeError:
        pane = tmux("new-session", "-d", "-s", "agents", "-P", "-F", "#{pane_id}", "-c", agent["cwd"], env + command).strip()
    store.execute("UPDATE agents SET pane=?,state='starting' WHERE id=?", (pane, agent["id"]))
    tmux("set-option", "-t", "agents", "remain-on-exit", "on")
    configure_tmux()
    tmux("select-layout", "-t", "agents", "tiled")
    return store.agent(agent["id"])


def ensure_service():
    # The service itself takes a kernel lock; simultaneous UIs cannot start two schedulers.
    proc = subprocess.Popen([sys.executable, "-m", "kindi_bridge", "serve"], stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True, env=os.environ.copy())
    _service_processes[:]=[p for p in _service_processes if p.poll() is None]
    _service_processes.append(proc)
    return proc.pid


def serve(stop=None):
    import fcntl
    stop = stop or threading.Event()
    with open(state_dir() / "service.lock", "a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        store = Store()
        store.setting("service_pid", str(os.getpid()))
        # Indexing stays outside the scheduler loop. A slow extractor/model download cannot delay due jobs.
        index_thread = None
        store.execute("UPDATE collections SET state='pending' WHERE state='indexing'")
        last_rescan = time.time()
        while not stop.wait(0.25):
            now = time.time()
            store.setting("service_heartbeat", str(now))
            for agent in store.rows("SELECT * FROM agents WHERE state NOT IN ('offline','failed')"):
                if agent["heartbeat"] and now - agent["heartbeat"] > 15:
                    store.execute("UPDATE agents SET state='offline' WHERE id=?", (agent["id"],))
                    store.execute("UPDATE jobs SET state='uncertain',error='Worker lost; reconcile acceptance before retrying' WHERE agent=? AND state IN ('sending','accepted')", (agent["id"],))
            store.execute("UPDATE jobs SET state='exhausted',error='Deadline reached' WHERE deadline IS NOT NULL AND deadline<? AND state='scheduled'", (now,))
            if now - last_rescan > 300:
                store.execute("UPDATE collections SET state='pending' WHERE state IN ('ready','keyword-only')")
                last_rescan = now
            if index_thread is None or not index_thread.is_alive():
                pending = store.rows("SELECT * FROM collections WHERE state='pending' LIMIT 1")
                if pending:
                    def index_one(c=pending[0]):
                        from .knowledge import index_collection
                        try:
                            index_collection(Store(), c)
                        except Exception as e:
                            Store().execute("UPDATE collections SET state='error',error=? WHERE id=?", (str(e), c["id"]))
                    index_thread = threading.Thread(target=index_one, daemon=True)
                    index_thread.start()


def completion_result(text, task, check):
    """Parse only an assistant's explicit task-bound completion record."""
    for line in reversed(text.splitlines()):
        if line.startswith("BRIDGE_RESULT "):
            try:
                result = json.loads(line[len("BRIDGE_RESULT "):])
                if result.get("task") == task and result.get("check") == check and result.get("status") in {"complete", "incomplete", "blocked"} and isinstance(result.get("evidence"), str) and result["evidence"].strip():
                    return result
            except (ValueError, TypeError, AttributeError):
                pass
    return None


def prompt_message(store, agent, job):
    text=job['prompt']
    try:
        from .learn import nudge_for
        nudge = nudge_for(store, agent['id'])
        if nudge:
            text = nudge + "\n\n" + text
            store.event('nudge_injected', nudge[:500], agent['id'], job['id'])
    except Exception:  # the learning layer must never break a prompt
        pass
    if job['kind']=='check':
        text+=f'\n\nBound job objective/identity: {job["task"]}\nPrevious check result:\n{job["response"] or "No previous result"}\nFinish with one line: BRIDGE_RESULT '+json.dumps({'task':job['task'],'check':job['id'],'status':'complete|incomplete|blocked','evidence':'specific evidence or reason'})
    if agent['collection']:
        try:
            from .knowledge import context
            block=context(store,job['task'] or text,agent['collection']);text+=block
            if block:store.event('retrieved_context',block,agent['id'],job['id'])
        except Exception as e:store.event('retrieval_error',e,agent['id'],job['id'])
    return text


def settle_job(store, job, text, error=None, now=None):
    now = now or time.time()
    status = "failed" if error else "completed"
    result = completion_result(text, job["task"], job["id"]) if job["kind"] == "check" else None
    if job["kind"] == "check" and not error:
        status = "confirmed" if result and result["status"] == "complete" else "scheduled"
    elif job["interval"] and not error:
        status = "scheduled"
    if status == "scheduled" and (job["attempts"] >= job["max_checks"] or (job["deadline"] and now >= job["deadline"])):
        status = "exhausted"
    due = max(now, (job["accepted"] or now) + job["interval"]) if status == "scheduled" else job["due"]
    store.execute("UPDATE jobs SET state=?,response=?,error=?,finished=?,due=? WHERE id=? AND state IN ('sending','accepted')", (status, text[-50000:], error, now, due, job["id"]))
    store.event(status, text or error or "No assistant response", job["agent"], job["id"])


def worker(ident):
    store = Store()
    agent = store.agent(ident)
    import fcntl
    lock = open(state_dir() / f"agent-{ident}.lock", "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    command = json.loads(agent["command"])
    events = queue.Queue()
    process = None
    def request_stop(*_):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, request_stop)
    current, answer, provider_error = None, "", None
    store.execute("UPDATE jobs SET state='uncertain',error='Worker restarted; delivery requires reconciliation' WHERE agent=? AND state IN ('sending','accepted')", (ident,))
    try:
        process = subprocess.Popen(command, cwd=agent["cwd"], env=child_env(command), stdin=subprocess.PIPE,
                                   start_new_session=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        def reader(stream, diagnostic=False):
            for line in iter(stream.readline, b""):
                try:
                    event = {"type": "diagnostic", "text": line.decode("utf-8", errors="replace")} if diagnostic else json.loads(line)
                    events.put(event)
                except ValueError:
                    events.put({"type": "diagnostic", "text": line.decode("utf-8", errors="replace")})
        threading.Thread(target=reader, args=(process.stdout,), daemon=True).start()
        threading.Thread(target=reader, args=(process.stderr, True), daemon=True).start()
        store.execute("UPDATE agents SET state='ready',pid=?,heartbeat=? WHERE id=?", (os.getpid(), time.time(), ident))
        store.event("agent_started", f"{agent['name']} in {agent['cwd']}", ident)
        print(f"KINDI // {agent['name']} — prompts arrive through Bridge Console", flush=True)
        last_beat = 0
        while process.poll() is None:
            now = time.time()
            if current and now - current.get("sent_at", now) > 3600:
                raise TimeoutError("One-hour turn timeout; reconcile this job before retrying")
            if now - last_beat >= 1:
                store.execute("UPDATE agents SET heartbeat=? WHERE id=?", (now, ident))
                last_beat = now
            if current is None:
                with store.db() as db:
                    db.execute("BEGIN IMMEDIATE")
                    row = db.execute("SELECT * FROM jobs WHERE agent=? AND state='scheduled' AND due<=? AND (deadline IS NULL OR deadline>=?) ORDER BY due,created LIMIT 1", (ident, now, now)).fetchone()
                    if row:
                        current = dict(row)
                        db.execute("UPDATE jobs SET state='sending',error=NULL WHERE id=?", (current["id"],))
                if current:
                    answer, provider_error = "", None
                    current['sent_at'] = now
                    text = current["prompt"]
                    if current["kind"] == "check":
                        text += f'\n\nBound job objective/identity: {current["task"]}\nPrevious check result:\n{current["response"] or "No previous result"}\nFinish with one line: BRIDGE_RESULT ' + json.dumps({"task": current["task"], "check": current["id"], "status": "complete|incomplete|blocked", "evidence": "specific evidence or reason"})
                    agent = store.agent(ident)
                    if agent["collection"]:
                        try:
                            from .knowledge import context
                            block = context(store, current["task"] or text, agent["collection"])
                            text += block
                            if block:
                                store.event("retrieved_context", block, ident, current["id"])
                        except Exception as e:
                            store.event("retrieval_error", e, ident, current["id"])
                    process.stdin.write((json.dumps({"id": current["id"], "type": "prompt", "message": text}) + "\n").encode())
                    process.stdin.flush()
                    store.event("sending", current["prompt"], ident, current["id"])
                    store.execute("UPDATE agents SET state='waiting for inference' WHERE id=?", (ident,))
            try:
                event = events.get(timeout=0.1)
            except queue.Empty:
                if current and now - (current.get("sent_at") or current.setdefault("sent_at", now)) > 3600:
                    raise TimeoutError("One-hour turn timeout; reconcile this job before retrying")
                continue
            kind = event.get("type", "unknown")
            if kind == "diagnostic":
                store.event(kind, event["text"], ident, current["id"] if current else None)
                print(event["text"], end="", flush=True)
            elif kind == "response" and current and event.get("id") == current["id"]:
                if event.get("success"):
                    store.execute("UPDATE jobs SET state='accepted',accepted=?,attempts=attempts+1 WHERE id=? AND state='sending'", (time.time(), current["id"]))
                    current = store.rows("SELECT * FROM jobs WHERE id=?", (current["id"],))[0]
                    current['sent_at'] = now
                    if (event.get("data") or {}).get("disposition") == "handled":
                        settle_job(store, current, "", error="Prompt handled without an agent run")
                        current = None
                else:
                    settle_job(store, current, "", error=str(event.get("error", "RPC rejected prompt")))
                    current = None
                    store.execute("UPDATE agents SET state='ready' WHERE id=?", (ident,))
            elif kind == "message_update":
                update = event.get("assistantMessageEvent", {})
                if update.get("type") == "text_delta":
                    delta = update.get("delta", "")
                    print("".join(c for c in delta if c.isprintable() or c in '\n\t'), end="", flush=True)
                    store.execute("UPDATE agents SET state='generating' WHERE id=?", (ident,))
            elif kind == "message_end":
                message = event.get("message", {})
                if message.get("role") == "assistant":
                    content = message.get("content", [])
                    answer = content if isinstance(content, str) else "\n".join(x.get("text", "") for x in content if x.get("type") == "text")
                    if message.get("stopReason") in {"error", "aborted"}:
                        provider_error = message.get("errorMessage") or message["stopReason"]
            elif kind in {"tool_execution_start", "tool_execution_end", "agent_start", "agent_end"}:
                store.event(kind, json.dumps(event), ident, current["id"] if current else None)
                if kind == "tool_execution_start":
                    store.execute("UPDATE agents SET state='running tool' WHERE id=?", (ident,))
                    print(f'\n[tool: {event.get("toolName", "?")}]', flush=True)
            elif kind == "extension_ui_request":
                store.execute("UPDATE agents SET state='needs input' WHERE id=?", (ident,))
                store.event("unsupported_extension_ui", json.dumps(event), ident, current["id"] if current else None)
            elif kind == "agent_settled":
                if current:
                    # A settled event can precede the async prompt response: acceptance is known from the run.
                    saved = store.rows("SELECT * FROM jobs WHERE id=?", (current["id"],))[0]
                    if saved["state"] == "sending":
                        store.execute("UPDATE jobs SET state='accepted',accepted=?,attempts=attempts+1 WHERE id=?", (time.time(), current["id"]))
                        saved = store.rows("SELECT * FROM jobs WHERE id=?", (current["id"],))[0]
                    settle_job(store, saved, answer, provider_error)
                    print("\n[settled]", flush=True)
                    current = None
                store.execute("UPDATE agents SET state='ready' WHERE id=?", (ident,))
    except Exception as e:
        store.event("worker_error", e, ident, current["id"] if current else None)
        print(f"\nBridge worker error: {e}", flush=True)
    finally:
        if current:
            store.execute("UPDATE jobs SET state='uncertain',error='Worker exited before confirmed completion; inspect history' WHERE id=? AND state IN ('sending','accepted')", (current["id"],))
        store.execute("UPDATE agents SET state='offline',heartbeat=? WHERE id=?", (time.time(), ident))
        if process:
            import psutil
            descendants = []
            with contextlib.suppress(psutil.Error):
                descendants = psutil.Process(process.pid).children(recursive=True)
            with contextlib.suppress(OSError):
                process.stdin.close()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=3)
            for child in descendants:
                with contextlib.suppress(psutil.Error):
                    child.terminate()
        lock.close()

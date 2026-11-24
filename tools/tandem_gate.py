#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Opt-in, reproducible GPU memory gate for the plan-engine tandem owner.

The gate answers one question on a host that has both the pack and the card:

    With ``--tandem-gpu plan`` and a per-card ceiling of N MiB, where does ``auto`` put the
    split, and what does the process actually allocate?

It does not assume a 27B result.  It reads the pack's own tensor sizes, asks the same
``plan_cpu_layers`` / ``vram_budget_bytes`` functions the runtime uses, builds a real
``TandemEngine`` with the plan owner, samples free device memory through the load and a
fixed greedy run, and writes a JSON record: pack identity (model name, source hash and
dimensions - never the absolute pack path), card, requested ceiling/reserve, chosen CPU and
GPU layers, the planner's estimate, the measured peak and minimum free memory, the greedy
reply tokens, an optional ``kindi verify`` result, and a verdict.

The record is evidence only for the machine and pack it names.  A 27B claim still needs a
27B record from a host that has the 27B pack; a dry run or a smaller pack is not one.

Run it only where the pack and card exist, for example:

    python tools/tandem_gate.py --pack ~/.kindi/packs/bonsai-27b \
        --ctx 24576 --vram-budget-mib 7168 --reserve-mib 512 \
        --json tandem-27b-24576.json

``--dry-run`` performs the prerequisite checks and prints the planner's split without
building an engine.  Exit codes: 0 the measured run stayed inside the budget, kept the
requested context, and verify passed or had no reference; 1 the measurement or verify
failed; 2 a missing pack/card/argument or an unusable plan (nothing was measured).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import threading
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:                     # `python tools/tandem_gate.py` from any cwd
    sys.path.insert(0, ROOT)

from kindi.cpu import VOCAB                  # noqa: E402 - after sys.path
from kindi.tandem import MIB, gpu_estimate, pack_bytes, plan_cpu_layers, vram_budget_bytes  # noqa: E402

SCHEMA = "kindi.tandem-gate/1"
DEFAULT_PACK = "~/.kindi/packs/bonsai-27b"
REJECT_BACKENDS = ("kindi_gemma4", "kindi_nanbeige", "kindi_lightning")

GPU_PROBE = r"""
import json
import cupy as cp

count = int(cp.cuda.runtime.getDeviceCount())
if count < 1:
    raise SystemExit("no CUDA device")
dev = cp.cuda.Device(0)
with dev:
    free, total = (int(x) for x in cp.cuda.runtime.memGetInfo())
props = cp.cuda.runtime.getDeviceProperties(0)
name = props.get("name", "GPU")
if isinstance(name, bytes):
    name = name.decode("utf-8", "replace")
cc = dev.compute_capability
if isinstance(cc, (tuple, list)):
    major, minor = int(cc[0]), int(cc[1])
else:
    major, minor = divmod(int(cc), 10)
print(json.dumps({
    "count": count,
    "name": str(name),
    "cc": [major, minor],
    "free": free,
    "total": total,
    "cupy": getattr(cp, "__version__", "?"),
    "driver": int(cp.cuda.runtime.driverGetVersion()),
    "runtime": int(cp.cuda.runtime.runtimeGetVersion()),
}))
"""


class GateError(Exception):
    """A prerequisite or planning failure; ``code`` is the process exit code."""

    def __init__(self, message, code=2):
        Exception.__init__(self, message)
        self.code = int(code)


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _non_negative_int(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return value


def _token_ids(text):
    parts = [part.strip() for part in text.replace(";", ",").split(",")]
    if not parts or any(not part for part in parts):
        raise argparse.ArgumentTypeError("must contain at least one token id per comma")
    try:
        values = [int(part) for part in parts]
    except ValueError:
        raise argparse.ArgumentTypeError("must be comma-separated token ids") from None
    bad = [value for value in values if not 0 <= value < VOCAB]
    if bad:
        raise argparse.ArgumentTypeError("token id %d is outside 0..%d" % (bad[0], VOCAB - 1))
    return values


def _path(value):
    return os.path.abspath(os.path.expanduser(value))


def _log(message):
    print(message, file=sys.stderr, flush=True)


def pack_identity(pack):
    """-> (absolute pack path, manifest, path-free identity).

    The identity is what the JSON record carries: the model/source names, the source
    hash and the tensor dimensions.  It deliberately has no directory, so a record can be
    attached to a PR without publishing the machine's filesystem layout.
    """
    path = _path(pack)
    manifest_path = os.path.join(path, "kindi_pack.json")
    if not os.path.isdir(path):
        raise GateError("pack directory not found: %s (convert the model first)" % path)
    if not os.path.isfile(manifest_path):
        raise GateError("pack has no kindi_pack.json: %s (run `kindi convert` first)" % path)
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as exc:
        raise GateError("cannot read %s: %s" % (manifest_path, exc)) from None
    backend = manifest.get("backend")
    if backend in REJECT_BACKENDS:
        raise GateError("pack backend %r is not a dense Bonsai pack; the tandem split does not support it"
                        % backend)
    dims = dict(manifest.get("dimensions") or {})
    try:
        layers = int(dims.get("layers", 0))
    except (TypeError, ValueError):
        raise GateError("pack dimensions.layers is not an integer") from None
    if layers < 2:
        raise GateError("pack has %d layer(s); the tandem split needs at least two" % layers)
    identity = {
        "name": os.path.basename(os.path.normpath(path)),
        "model": manifest.get("model"),
        "source_file": manifest.get("source_file"),
        "source_sha256": manifest.get("source_sha256"),
        "layers": layers,
        "width": dims.get("width"),
        "ffn": dims.get("ffn"),
        "heads": dims.get("heads"),
        "kv_heads": dims.get("kv_heads"),
        "head_dim": dims.get("head_dim"),
        "full_attention_interval": dims.get("full_attention_interval"),
        "hadamard_block": manifest.get("hadamard_block", 512),
        "bcodes_bytes": manifest.get("bcodes_bytes"),
        "dpm_bytes": manifest.get("dpm_bytes"),
        "f32_bytes": manifest.get("f32_bytes"),
    }
    return path, manifest, identity


def probe_gpu():
    """Probe CUDA/CuPy in a short-lived child, so a missing driver is a clean exit 2."""
    try:
        proc = subprocess.run([sys.executable, "-c", GPU_PROBE], cwd=ROOT, env=os.environ.copy(),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=90)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GateError("CUDA/CuPy probe failed: %s" % exc) from None
    if proc.returncode:
        lines = (proc.stderr or "").strip().splitlines()
        raise GateError("CUDA/CuPy is not usable on this host: %s"
                        % (lines[-1] if lines else "probe exited %d" % proc.returncode))
    try:
        info = json.loads((proc.stdout or "").strip().splitlines()[-1])
    except (IndexError, ValueError):
        raise GateError("CUDA/CuPy probe returned no device information") from None
    try:
        info["free"], info["total"], info["count"] = int(info["free"]), int(info["total"]), int(info["count"])
    except (KeyError, TypeError, ValueError):
        raise GateError("CUDA/CuPy probe returned an incomplete device record") from None
    return info


def pack_sizes(pack, manifest):
    try:
        return pack_bytes(pack, manifest)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise GateError("cannot read the pack's tensor indexes for the planner: %s" % exc) from None


def plan_for_gate(dims, sizes, ctx, free_bytes, reserve_mib, ceiling_mib, target_mib=None):
    """The planner's split and budget for one pack/context/card, with no engine built."""
    budget = int(vram_budget_bytes(free_bytes, reserve_mib, ceiling_mib))
    if budget <= 0:
        raise GateError("the reserve and ceiling leave no usable GPU memory "
                        "(free %d MiB, reserve %d MiB, ceiling %d MiB)"
                        % (free_bytes // MIB, reserve_mib, ceiling_mib))
    full = int(gpu_estimate(dims, sizes, ctx, (), owner="plan"))
    try:
        cpu = plan_cpu_layers(dims, sizes, ctx, free_bytes, reserve_mib=reserve_mib,
                              target_mib=target_mib, owner="plan", vram_budget_mib=ceiling_mib)
    except SystemExit as exc:
        raise GateError(str(exc)) from None
    except Exception as exc:                                   # noqa: BLE001 - report, do not invent
        raise GateError("the planner failed: %s" % exc) from None
    if not cpu:
        raise GateError("this model already fits under the %d MiB ceiling; a split would only be slower. "
                        "Lower --vram-budget-mib or pass --target-mib to force headroom" % ceiling_mib)
    cpu = tuple(int(layer) for layer in cpu)
    gpu = tuple(layer for layer in range(int(dims["layers"])) if layer not in frozenset(cpu))
    estimate = int(gpu_estimate(dims, sizes, ctx, cpu, owner="plan"))
    return {
        "requested_ctx": int(ctx),
        "free_bytes": int(free_bytes),
        "free_mib": round(free_bytes / MIB, 1),
        "ceiling_mib": int(ceiling_mib),
        "reserve_mib": int(reserve_mib),
        "target_mib": target_mib,
        "budget_bytes": budget,
        "budget_mib": round(budget / MIB, 1),
        "cpu_layers": list(cpu),
        "gpu_layers": list(gpu),
        "estimate_bytes": estimate,
        "estimate_mib": round(estimate / MIB, 1),
        "full_estimate_bytes": full,
        "full_estimate_mib": round(full / MIB, 1),
    }


def peak_summary(free_start, total, samples):
    """Turn sampled free-memory readings into the run's measured allocation."""
    values = [int(value) for value in samples if value is not None]
    if not values:
        values = [int(free_start)]
    min_free = min(values)
    return {
        "free_start_bytes": int(free_start),
        "free_start_mib": round(free_start / MIB, 1),
        "min_free_bytes": min_free,
        "min_free_mib": round(min_free / MIB, 1),
        "peak_alloc_bytes": max(0, int(free_start) - min_free),
        "peak_alloc_mib": round(max(0, int(free_start) - min_free) / MIB, 1),
        "peak_used_bytes": max(0, int(total) - min_free),
        "peak_used_mib": round(max(0, int(total) - min_free) / MIB, 1),
        "samples": len(values),
    }


def gate_verdict(plan, measure, engine, error, verify, reserve_mib, requested_ctx):
    """The pass/fail reasons.  Every reason names the measurement it came from."""
    reasons = []
    if error:
        reasons.append(error)
    if measure is not None:
        budget_mib = plan.get("budget_mib", round(plan["budget_bytes"] / MIB, 1))
        if measure["peak_alloc_bytes"] > plan["budget_bytes"]:
            reasons.append("measured peak %d MiB is over the %d MiB budget"
                           % (measure["peak_alloc_mib"], budget_mib))
        if measure["min_free_bytes"] < int(reserve_mib) * MIB:
            reasons.append("minimum free memory %d MiB is below the %d MiB reserve"
                           % (measure["min_free_mib"], reserve_mib))
    if engine is not None and int(engine["context"]) < int(requested_ctx):
        reasons.append("the GPU owner reduced the context from %d to %d tokens; the requested context was not measured"
                       % (requested_ctx, engine["context"]))
    if engine is None and measure is not None and not error:
        reasons.append("the engine was not built")
    if verify is not None:
        if verify.get("status") == "fail":
            reasons.append("kindi verify failed (exit %s)" % verify.get("exit_code"))
        elif verify.get("status") == "timeout":
            reasons.append("kindi verify timed out")
    return {"fits": not reasons, "reasons": reasons}


class _Sampler(threading.Thread):
    """Sample device free memory while the engine loads and runs."""

    def __init__(self, cp, interval):
        threading.Thread.__init__(self, daemon=True)
        self.cp = cp
        self.interval = float(interval)
        self.stop_event = threading.Event()
        self.samples = []
        self.pool = []
        self.errors = []

    def run(self):
        while not self.stop_event.is_set():
            try:
                self.samples.append(int(self.cp.cuda.runtime.memGetInfo()[0]))
                self.pool.append(int(self.cp.get_default_memory_pool().used_bytes()))
            except Exception as exc:                            # noqa: BLE001 - record a sampler failure
                self.errors.append(str(exc))
            self.stop_event.wait(self.interval)

    def stop(self):
        self.stop_event.set()


def _engine_record(eng):
    return {
        "mode": eng.mode,
        "owner": eng.gpu_kind,
        "kv": getattr(eng.gpu, "kv", None),
        "context": int(eng.ctx),
        "cpu_layers": list(eng.cpu_range),
        "gpu_layers": list(eng.gpu_range),
        "cpu_threads": getattr(eng.cpu, "threads", None),
        "cpu_isa": getattr(eng.cpu, "isa", None),
        "fingerprint": eng.cache_fingerprint(),
    }


def _verify_command(args, pack, cpu_layers):
    return [
        sys.executable, "-m", "kindi", "--pack", pack, "verify",
        "--tandem", args.mode, "--tandem-cpu-layers", ",".join(str(layer) for layer in cpu_layers),
        "--tandem-gpu", "plan",
    ]


def _verify_env(args):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("KINDI_") or key in ("KINDI_CPU_CACHE",)}
    env.update({
        "KINDI_DEVICE": "gpu",
        "KINDI_CARDS": "1",
        "KINDI_TANDEM_GPU": "plan",
        "KINDI_TANDEM_VRAM_MB": str(args.vram_budget_mib),
        "KINDI_TANDEM_RESERVE_MB": str(args.reserve_mib),
        "KINDI_HEAD": "exact",
        "KINDI_SPEC": "0",
    })
    return env


def run_verify(args, pack, cpu_layers):
    """Run the existing llama.cpp reference check with the split the gate measured."""
    cmd = _verify_command(args, pack, cpu_layers)
    shown = [("<pack>" if value == pack else value) for value in cmd]
    _log("tandem gate: running kindi verify with %d CPU layers" % len(cpu_layers))
    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=_verify_env(args), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=args.verify_timeout)
    except subprocess.TimeoutExpired as exc:
        output = exc.output or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", "replace")
        return {"status": "timeout", "exit_code": None, "command": shown,
                "tail": _tail(output, pack), "timeout_s": args.verify_timeout}
    except OSError as exc:
        return {"status": "fail", "exit_code": None, "command": shown, "tail": str(exc)}
    if proc.returncode == 0:
        status = "pass"
    elif proc.returncode == 2:
        status = "skipped"                    # no reference file for this pack
    else:
        status = "fail"
    return {"status": status, "exit_code": int(proc.returncode), "command": shown,
            "tail": _tail(proc.stdout or "", pack)}


def _tail(text, pack, lines=40):
    text = text.replace(pack, "<pack>").replace(ROOT, "<repo>").replace(os.path.expanduser("~"), "~")
    return "\n".join(text.strip().splitlines()[-lines:])


def _record(args, identity, card, plan, engine, measure, reply, verify, verdict, error, notes):
    return {
        "schema": SCHEMA,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": "plan-owner-vram",
        "pack": identity,
        "card": {
            "name": card.get("name"),
            "cc": card.get("cc"),
            "visible": card.get("count"),
            "total_bytes": card.get("total"),
            "total_mib": round(int(card["total"]) / MIB, 1) if card.get("total") is not None else None,
            "free_at_probe_bytes": card.get("free"),
            "free_at_probe_mib": round(int(card["free"]) / MIB, 1) if card.get("free") is not None else None,
            "cupy": card.get("cupy"),
            "driver": card.get("driver"),
            "runtime": card.get("runtime"),
        },
        "request": {
            "mode": args.mode,
            "ctx": args.ctx,
            "vram_budget_mib": args.vram_budget_mib,
            "reserve_mib": args.reserve_mib,
            "target_mib": args.target_mib,
            "tokens": args.tokens,
            "prompt_ids": args.prompt_ids,
            "verify": bool(args.verify),
            "sample_ms": args.sample_ms,
            "dry_run": bool(args.dry_run),
        },
        "host": {"python": sys.version.split()[0], "platform": platform.platform()},
        "plan": plan,
        "engine": engine,
        "measure": measure,
        "run": {"reply_ids": reply, "reply_tokens": len(reply) if reply else 0},
        "verify": verify,
        "verdict": verdict,
        "error": error,
        "notes": notes,
    }


def _dry_run(args, identity, card, dims, sizes):
    plan = plan_for_gate(dims, sizes, args.ctx, int(card["free"]), args.reserve_mib,
                         args.vram_budget_mib, args.target_mib)
    verdict = {"fits": True, "dry_run": True, "reasons": []}
    notes = ["dry run: no engine was built and no memory was measured; card free memory is the probe's figure"]
    return _record(args, identity, card, plan, None, None, None, None, verdict, None, notes)


def _measure(args, identity, card, dims, sizes):
    # A tandem plan owner is one card; a multi-card host still measures card 0.
    os.environ["KINDI_CARDS"] = "1"
    try:
        import cupy as cp
    except ImportError as exc:
        raise GateError("CuPy is not installed: %s" % exc) from None
    try:
        cp.cuda.Device(0).use()
        cp.cuda.Device().synchronize()
        free_start, total = (int(value) for value in cp.cuda.runtime.memGetInfo())
    except Exception as exc:                                  # noqa: BLE001 - a clean prerequisite error
        raise GateError("CUDA device 0 is not usable in this process: %s" % exc) from None
    plan = plan_for_gate(dims, sizes, args.ctx, free_start, args.reserve_mib,
                         args.vram_budget_mib, args.target_mib)
    sampler = _Sampler(cp, max(args.sample_ms, 1) / 1000.0)
    sampler.start()
    eng = None
    engine = None
    reply = None
    error = None
    try:
        from kindi.tandem import TandemEngine
        cpu_layers = "auto:%d" % args.target_mib if args.target_mib else "auto"
        eng = TandemEngine(args.pack, ctx=args.ctx, cpu_layers=cpu_layers, mode=args.mode,
                           log=_log, gpu="plan", vram_budget_mib=args.vram_budget_mib,
                           reserve_mib=args.reserve_mib)
        engine = _engine_record(eng)
        engine["requested_context"] = args.ctx
        # The engine ran its own auto planner after this pre-plan.  Record the split it actually
        # chose and price that split, so the measured peak is compared with the real placement.
        plan["planned_cpu_layers"] = list(plan["cpu_layers"])
        plan["matches_planner"] = plan["planned_cpu_layers"] == engine["cpu_layers"]
        plan["cpu_layers"] = list(engine["cpu_layers"])
        plan["gpu_layers"] = list(engine["gpu_layers"])
        plan["estimate_bytes"] = int(gpu_estimate(dims, sizes, engine["context"],
                                                   engine["cpu_layers"], owner="plan"))
        plan["estimate_mib"] = round(plan["estimate_bytes"] / MIB, 1)
        plan["full_estimate_bytes"] = int(gpu_estimate(dims, sizes, engine["context"], (), owner="plan"))
        plan["full_estimate_mib"] = round(plan["full_estimate_bytes"] / MIB, 1)
        reply = [int(token) for token in eng.generate(args.prompt_ids, args.tokens)]
    except (Exception, SystemExit) as exc:                    # noqa: BLE001 - record the failure, then gate it
        error = ("%s: %s" % (type(exc).__name__, exc)).replace(args.pack, "<pack>").replace(ROOT, "<repo>")
    finally:
        sampler.stop()
        sampler.join(timeout=5.0)
        try:
            cp.cuda.Device().synchronize()
        except Exception:                                      # noqa: BLE001 - best-effort after an error
            pass
        if eng is not None:
            del eng
        gc.collect()
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:                                      # noqa: BLE001 - best-effort teardown
            pass
    measure = peak_summary(free_start, total, sampler.samples)
    if sampler.pool:
        measure["pool_peak_bytes"] = max(sampler.pool)
        measure["pool_peak_mib"] = round(max(sampler.pool) / MIB, 1)
    if sampler.errors:
        measure["sample_errors"] = sampler.errors[:3]
    measure["sample_ms"] = args.sample_ms
    measure["free_at_probe_mib"] = round(int(card["free"]) / MIB, 1) if card.get("free") is not None else None

    memory_ok = (error is None and engine is not None and int(engine["context"]) >= int(args.ctx)
                 and measure["peak_alloc_bytes"] <= plan["budget_bytes"]
                 and measure["min_free_bytes"] >= int(args.reserve_mib) * MIB)
    verify = None
    if args.verify:
        if memory_ok:
            verify = run_verify(args, args.pack, engine["cpu_layers"])
        else:
            verify = {"status": "skipped", "reason": "the memory gate failed before verify", "exit_code": None}
    verdict = gate_verdict(plan, measure, engine, error, verify, args.reserve_mib, args.ctx)
    notes = [
        "peak_alloc is free_start - min(sampled free), so it is a sampled lower bound on the true peak",
        "other processes sharing the card can change the free-memory readings during the run",
    ]
    return _record(args, identity, card, plan, engine, measure, reply, verify, verdict, error, notes)


def _write(record, path):
    if not path:
        return None
    if path == "-":
        print(json.dumps(record, indent=2))
        return "-"
    target = _path(path)
    temporary = target + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, target)
    return target


def _layers_text(layers):
    if not layers:
        return "(none)"
    layers = sorted(int(layer) for layer in layers)
    runs = []
    start = previous = layers[0]
    for layer in layers[1:]:
        if layer == previous + 1:
            previous = layer
            continue
        runs.append(str(start) if start == previous else "%d-%d" % (start, previous))
        start = previous = layer
    runs.append(str(start) if start == previous else "%d-%d" % (start, previous))
    return ",".join(runs)


def _summary(record):
    plan = record.get("plan") or {}
    card = record.get("card") or {}
    measure = record.get("measure")
    engine = record.get("engine")
    print("tandem gate: %s (%s layers, sha256 %s)" % (
        record["pack"].get("model"), record["pack"].get("layers"),
        (record["pack"].get("source_sha256") or "?")[:12]))
    print("card: %s (sm_%s%s), %s MiB free of %s at probe" % (
        card.get("name"), (card.get("cc") or ["?", "?"])[0], (card.get("cc") or ["?", "?"])[1],
        card.get("free_at_probe_mib"), card.get("total_mib")))
    print("request: %s, ctx %d, ceiling %d MiB, reserve %d MiB, budget %s MiB" % (
        record["request"]["mode"], record["request"]["ctx"], record["request"]["vram_budget_mib"],
        record["request"]["reserve_mib"], plan.get("budget_mib")))
    print("plan: CPU %s; GPU %s; estimate %s MiB (full %s MiB)" % (
        _layers_text(plan.get("cpu_layers")), _layers_text(plan.get("gpu_layers")),
        plan.get("estimate_mib"), plan.get("full_estimate_mib")))
    if engine is not None:
        print("engine: context %s, owner %s, K/V %s" % (
            engine.get("context"), engine.get("owner"), engine.get("kv")))
    if measure is not None:
        print("measured: peak %s MiB (free_start %s, min free %s), %s samples" % (
            measure.get("peak_alloc_mib"), measure.get("free_start_mib"),
            measure.get("min_free_mib"), measure.get("samples")))
    if record.get("verify") is not None:
        print("verify: %s (exit %s)" % (record["verify"].get("status"), record["verify"].get("exit_code")))
    verdict = record.get("verdict") or {}
    print("verdict: %s%s" % ("PASS" if verdict.get("fits") else "FAIL",
                             " (dry run)" if verdict.get("dry_run") else ""))
    for reason in verdict.get("reasons") or []:
        print("  - %s" % reason)


def _parser():
    parser = argparse.ArgumentParser(
        prog="tandem_gate.py",
        description="Measure a plan-engine tandem split against a per-card VRAM ceiling (opt-in; needs the pack and card).")
    parser.add_argument("--pack", default=DEFAULT_PACK,
                        help="dense Bonsai pack directory (default: %s)" % DEFAULT_PACK)
    parser.add_argument("--ctx", type=_positive_int, default=24576,
                        help="requested context in tokens (default: 24576)")
    parser.add_argument("--vram-budget-mib", type=_positive_int, required=True, metavar="N",
                        help="per-card ceiling in MiB; the planner uses min(free - reserve, N)")
    parser.add_argument("--reserve-mib", type=_non_negative_int, default=512, metavar="N",
                        help="free card memory left for the display/driver (default: 512)")
    parser.add_argument("--target-mib", type=_positive_int, default=None, metavar="N",
                        help="ask the split to free at least N MiB even when the model already fits")
    parser.add_argument("--mode", choices=("cpu-first", "gpu-first"), default="cpu-first",
                        help="tandem placement (default: cpu-first)")
    parser.add_argument("--tokens", type=_positive_int, default=8,
                        help="greedy reply tokens to generate (default: 8)")
    parser.add_argument("--prompt-ids", type=_token_ids, default=[1, 2, 3, 4],
                        help="comma-separated prompt token ids (default: 1,2,3,4)")
    parser.add_argument("--no-verify", dest="verify", action="store_false",
                        help="skip the separate `kindi verify` reference run")
    parser.set_defaults(verify=True)
    parser.add_argument("--verify-timeout", type=_positive_int, default=3600, metavar="S",
                        help="timeout for the verify child in seconds (default: 3600)")
    parser.add_argument("--sample-ms", type=_positive_int, default=10, metavar="MS",
                        help="free-memory sampling interval in milliseconds (default: 10)")
    parser.add_argument("--json", metavar="FILE",
                        help="write the JSON record here ('-' prints it to stdout)")
    parser.add_argument("--dry-run", action="store_true",
                        help="check the pack/card and print the planner's split without building an engine")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        pack, manifest, identity = pack_identity(args.pack)
        card = probe_gpu()
        sizes = pack_sizes(pack, manifest)
        dims = dict(manifest.get("dimensions") or {})
        args.pack = pack
        if args.dry_run:
            record = _dry_run(args, identity, card, dims, sizes)
        else:
            record = _measure(args, identity, card, dims, sizes)
    except GateError as exc:
        print("tandem gate: %s" % exc, file=sys.stderr)
        return exc.code
    _summary(record)
    path = _write(record, args.json)
    if path and path != "-":
        print("record: %s" % path)
    return 0 if record["verdict"].get("fits") else 1


if __name__ == "__main__":
    raise SystemExit(main())

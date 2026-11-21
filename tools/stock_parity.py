#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Re-run the official v0.6.1 vs this-fork greedy parity matrix.

The fixed prompts are the first 100, 2,000 and 8,000 token ids of one tracked
document.  The official tree generates 32 greedy tokens from each prefix on the
2B and 9B; those ids are the tracked goldens.  This command runs the same
prefixes through stock ``v0.6.1`` and this worktree and compares token ids.

A stock tree is extracted from the local git tag after the tag is verified as
the pinned commit.  Stock needs the same native-Windows loader shim used by the
earlier stock page; the worker applies it before importing stock Kindi.  Both
sides use the plain engine, exact head and no speculation/tensor-core/tandem
settings.

Usage:
    python tools/stock_parity.py --check
    python tools/stock_parity.py --capture
    python tools/stock_parity.py --write-prompts
"""
import argparse
import atexit
import ctypes
import hashlib
import inspect
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Iterable, cast

ROOT = Path(__file__).resolve().parents[1]
TOOL = Path(__file__).resolve()
BENCH_DIR = ROOT / "docs" / "benchmarks" / "2026-11-20-i7-8086k-rtx4060ti-stock-parity"
DEFAULT_PROMPTS = BENCH_DIR / "prompts.json"
DEFAULT_GOLDENS = BENCH_DIR / "goldens.json"
DEFAULT_REPORT = BENCH_DIR / "parity-report.json"
DEFAULT_PACK_ROOT = Path.home() / ".kindi" / "packs"
STOCK_TAG = "v0.6.1"
STOCK_COMMIT = "7fececbff13adc88bf21b8f07dea4e1c91996a9d"
MODELS = ("2b", "9b")
POSITIONS = (100, 2000, 8000)
GENERATED_TOKENS = 32
CONTEXT = 8192
SOURCE_TEMPLATE = ("The quick brown fox jumps over the lazy dog. "
                   "Pack my box with five dozen liquor jugs. "
                   "How vexingly quick daft zebras jump! ")
SOURCE_REPEATS = 1200


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp, path)


def _pack_for(args, model):
    explicit = getattr(args, "pack_" + model, None)
    if explicit:
        return Path(explicit).expanduser().resolve()
    return (Path(args.pack_root).expanduser() / ("bonsai-" + model)).resolve()


def _check_stock_tag():
    """Refuse a local tag that is not the pinned stock commit."""
    try:
        got = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", STOCK_TAG + "^{commit}"],
            stderr=subprocess.STDOUT, text=True, timeout=30).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("could not resolve local tag %s: %s" % (STOCK_TAG, exc)) from exc
    if got != STOCK_COMMIT:
        raise RuntimeError("local tag %s is %s, not pinned %s" % (STOCK_TAG, got, STOCK_COMMIT))


def _extract_stock(path):
    """Extract the pinned tag into a fresh directory."""
    try:
        archive = subprocess.check_output(
            ["git", "-C", str(ROOT), "archive", "--format=tar", STOCK_TAG],
            stderr=subprocess.STDOUT, timeout=120)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("could not read git tag %s: %s" % (STOCK_TAG, exc)) from exc
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            try:
                tar.extractall(path, filter="data")
            except TypeError:                       # Python < 3.12 has no filter argument
                tar.extractall(path)
    except (OSError, tarfile.TarError) as exc:
        raise RuntimeError("could not extract %s into %s: %s" % (STOCK_TAG, path, exc)) from exc
    if not (path / "kindi" / "engine.py").is_file():
        raise RuntimeError("extracted %s has no kindi/engine.py" % STOCK_TAG)


def _stock_tree():
    """A fresh stock tree extracted from the verified tag."""
    _check_stock_tag()
    path = Path(tempfile.mkdtemp(prefix="kindi-stock-%s-" % STOCK_TAG))
    atexit.register(shutil.rmtree, str(path), ignore_errors=True)
    try:
        _extract_stock(path)
    except Exception:
        shutil.rmtree(path, ignore_errors=True)
        raise
    return path


def _worker_env(is_fork):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("KINDI_DEVICE", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if is_fork:
        # Match stock's plain path: no speculation, tensor cores, or plan engine.
        env["KINDI_TP"] = "0"
        env["KINDI_SPEC"] = "0"
        env["KINDI_TC"] = "0"
        env["KINDI_FD"] = "0"
        env["KINDI_ATT_TC"] = "0"
    return env


def _run_worker(tree, args, out_path, model="2b", emit_prompts=None, source_file=None,
                source_template=None, source_repeats=None, is_fork=False) -> dict:
    cmd = [sys.executable, str(TOOL), "--worker", "--tree", str(tree),
           "--pack", str(_pack_for(args, model)), "--model", model,
           "--out", str(out_path), "--prompts-path", str(args.prompts),
           "--ctx", str(args.ctx), "--generated-tokens", str(args.generated_tokens),
           "--positions", ",".join(str(p) for p in args.positions),
           "--models", ",".join(args.models)]
    if emit_prompts:
        cmd += ["--emit-prompts", str(emit_prompts), "--source-file", str(source_file),
                "--source-template", source_template, "--source-repeats", str(source_repeats)]
    try:
        proc = subprocess.run(cmd, cwd=str(tree), env=_worker_env(is_fork),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=args.timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("worker failed to start in %s: %s" % (tree, exc)) from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-12:]
        raise RuntimeError("worker in %s exited %d:\n%s"
                           % (tree, proc.returncode, "\n".join(tail)))
    if emit_prompts:
        return {}
    try:
        with open(out_path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        raise RuntimeError("worker in %s wrote no usable result: %s" % (tree, exc)) from exc


def _prompt_ids(source_text, tokenizer, positions):
    ids = tokenizer.encode(source_text)
    if len(ids) < max(positions):
        raise RuntimeError("source is only %d tokens; need %d" % (len(ids), max(positions)))
    return {p: ids[:p] for p in positions}


def _worker_main(args):
    """Child process: import one tree's Kindi and emit prompts or generate."""
    if os.name == "nt":
        cdll = ctypes.CDLL

        def cdll_windows(name, *rest, **kwargs):
            if name == "libcuda.so.1":
                name = "nvcuda.dll"
            return cdll(name, *rest, **kwargs)

        ctypes.CDLL = cdll_windows

    tree = Path(args.tree).resolve()
    tools_dir = str(Path(__file__).resolve().parent)
    repo_dir = str(ROOT)
    sys.path[:] = [str(tree)] + [
        p for p in sys.path
        if os.path.abspath(p or os.getcwd()) not in (tools_dir, repo_dir)
    ]
    from kindi.tokenizer import Tok

    pack = Path(args.pack).expanduser().resolve()
    with open(pack / "vocab.json", encoding="utf-8") as handle:
        vocab = json.load(handle)
    tok = Tok(vocab, str(pack / "merges.txt"))
    positions = tuple(sorted(int(x) for x in args.positions.split(",") if x))

    if args.emit_prompts:
        with open(args.source_file, encoding="utf-8") as handle:
            source = handle.read()
        by_position = _prompt_ids(source, tok, positions)
        prompts = []
        for pos in positions:
            ids = by_position[pos]
            prompts.append({
                "id": "pos-%d" % pos,
                "position": pos,
                "ids": [int(x) for x in ids],
                "sha256": _sha256_text(_canonical([int(x) for x in ids])),
            })
        doc = {
            "format": 1,
            "description": "Fixed token-id prefixes for stock v0.6.1 parity",
            "source_template": args.source_template,
            "source_repeats": int(args.source_repeats),
            "source_sha256": _sha256_text(source),
            "generated_tokens": int(args.generated_tokens),
            "context": int(args.ctx),
            "models": [m for m in args.models.split(",") if m],
            "prompts": prompts,
        }
        _write_json(args.emit_prompts, doc)
        return 0

    import numpy as np

    from kindi.engine import make_engine

    clusters = None
    cluster_path = pack / "head_clusters.npz"
    if cluster_path.is_file():
        clusters = dict(np.load(cluster_path))
    params = inspect.signature(make_engine).parameters
    kwargs = {"pack": str(pack), "ctx": args.ctx, "clusters": clusters,
              "log": lambda msg: sys.stderr.write("[%s] %s\n" % (args.model, msg))}
    if "tandem" in params:
        kwargs["tandem"] = "off"
    if "kv_cache" in params:
        kwargs["kv_cache"] = None
    eng = make_engine(**kwargs)
    if int(getattr(eng, "ctx", args.ctx)) < max(positions) + args.generated_tokens:
        raise RuntimeError("engine context %s is too small for %d + %d"
                           % (getattr(eng, "ctx", None), max(positions), args.generated_tokens))
    outputs = {}
    prompts_path = Path(args.prompts)
    with open(prompts_path, encoding="utf-8") as handle:
        prompt_doc = json.load(handle)
    by_pos = {int(p["position"]): p for p in prompt_doc["prompts"]}
    for pos in positions:
        if pos not in by_pos:
            raise RuntimeError("tracks prompts have no position %d" % pos)
        ids = [int(x) for x in by_pos[pos]["ids"]]
        if len(ids) != pos:
            raise RuntimeError("prompt %d has %d ids" % (pos, len(ids)))
        t0 = time.time()
        generated = cast(Iterable[int], eng.generate(ids, args.generated_tokens))
        got = [int(x) for x in generated]
        outputs[str(pos)] = {
            "ids": got,
            "text": tok.decode(got),
            "seconds": round(time.time() - t0, 2),
            "prompt_sha256": by_pos[pos]["sha256"],
        }
    try:
        import cupy as cp
        gpu = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
        gpu = gpu.decode() if isinstance(gpu, bytes) else str(gpu)
    except Exception:                                # noqa: BLE001 - the run already failed if unavailable
        gpu = "unknown"
    result = {
        "tree": args.tree,
        "model": args.model,
        "gpu": gpu,
        "engine": type(eng).__name__,
        "positions": outputs,
    }
    _write_json(args.out, result)
    return 0


def _write_prompts(args, stock):
    source = SOURCE_TEMPLATE * SOURCE_REPEATS
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt",
                                     delete=False) as handle:
        handle.write(source)
        source_path = handle.name
    out = Path(args.prompts)
    try:
        _run_worker(stock, args, out.with_name("prompts-worker.json"), model="2b",
                    emit_prompts=out, source_file=source_path,
                    source_template=SOURCE_TEMPLATE, source_repeats=SOURCE_REPEATS)
    finally:
        os.unlink(source_path)
        try:
            out.with_name("prompts-worker.json").unlink()
        except OSError:
            pass
    return out


def _load_prompts(path):
    with open(path, encoding="utf-8") as handle:
        doc = json.load(handle)
    if doc.get("format") != 1:
        raise RuntimeError("%s: unsupported prompt format" % path)
    if (doc.get("source_template") != SOURCE_TEMPLATE
            or doc.get("source_repeats") != SOURCE_REPEATS):
        raise RuntimeError("%s: prompt source is not the fixed template" % path)
    source = SOURCE_TEMPLATE * SOURCE_REPEATS
    if doc.get("source_sha256") != _sha256_text(source):
        raise RuntimeError("%s: prompt source hash mismatch" % path)
    prompts = doc.get("prompts") or []
    by_pos = {int(p["position"]): p for p in prompts}
    for p in POSITIONS:
        if p not in by_pos:
            raise RuntimeError("%s: no prompt at position %d" % (path, p))
        ids = by_pos[p].get("ids") or []
        if len(ids) != p:
            raise RuntimeError("%s: prompt %d has %d ids" % (path, p, len(ids)))
        if by_pos[p].get("sha256") != _sha256_text(_canonical([int(x) for x in ids])):
            raise RuntimeError("%s: prompt %d hash mismatch" % (path, p))
    return doc


def _prompts_sha256(doc):
    return _sha256_text(_canonical([(int(p["position"]), [int(x) for x in p["ids"]])
                                    for p in doc["prompts"]]))


def _check_prompt_config(args, prompts):
    if prompts.get("context") != args.ctx:
        raise RuntimeError("prompts use context %s, not %s" % (prompts.get("context"), args.ctx))
    if prompts.get("generated_tokens") != args.generated_tokens:
        raise RuntimeError("prompts use %s generated tokens, not %s"
                           % (prompts.get("generated_tokens"), args.generated_tokens))
    models = prompts.get("models") or []
    if not set(args.models).issubset(models):
        raise RuntimeError("prompts do not cover model(s): %s"
                           % ", ".join(sorted(set(args.models) - set(models))))


def _manifest_sha256(pack):
    with open(Path(pack) / "kindi_pack.json", "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _capture(args, stock, prompts):
    _check_stock_tag()
    _check_prompt_config(args, prompts)
    models = {}
    try:
        for model in args.models:
            pack = _pack_for(args, model)
            if not (pack / "kindi_pack.json").is_file():
                raise RuntimeError("missing pack for %s: %s" % (model, pack))
            result = _run_worker(stock, args, BENCH_DIR / (".capture-%s.json" % model),
                                 model=model, is_fork=False)
            outputs = {}
            for pos in args.positions:
                row = result["positions"][str(pos)]
                outputs[str(pos)] = {"ids": row["ids"], "text": row["text"]}
            models[model] = {
                "pack": "bonsai-" + model,
                "manifest_sha256": _manifest_sha256(pack),
                "engine": result.get("engine"),
                "positions": outputs,
            }
        goldens = {
            "format": 1,
            "captured": time.strftime("%Y-%m-%d"),
            "stock": {
                "tag": STOCK_TAG,
                "commit": STOCK_COMMIT,
                "loader": "native-Windows libcuda.so.1 -> nvcuda.dll shim",
                "context": args.ctx,
                "generated_tokens": args.generated_tokens,
                "prompts_sha256": _prompts_sha256(prompts),
                "source_sha256": prompts["source_sha256"],
            },
            "models": models,
        }
        _write_json(args.goldens, goldens)
    finally:
        for model in args.models:
            try:
                (BENCH_DIR / (".capture-%s.json" % model)).unlink()
            except OSError:
                pass
    return goldens


def _compare(goldens, result, side, models=MODELS, positions=POSITIONS):
    rows, mismatches = [], []
    for model in models:
        if model not in goldens.get("models", {}):
            continue
        want_model = goldens["models"][model]["positions"]
        got_model = result.get("models", {}).get(model, {}).get("positions", {})
        for pos in positions:
            want = want_model.get(str(pos), {}).get("ids")
            got = got_model.get(str(pos), {}).get("ids")
            if want is None:
                continue
            if got is None:
                rows.append((model, pos, side, "missing", None))
                mismatches.append((model, pos, side, None))
                continue
            if want == got:
                rows.append((model, pos, side, "exact", None))
                continue
            first = next((i for i, (a, b) in enumerate(zip(want, got, strict=False))
                          if a != b), min(len(want), len(got)))
            rows.append((model, pos, side, "MISMATCH", first))
            mismatches.append((model, pos, side, first))
    return rows, mismatches


def _check_goldens(goldens, args, prompts):
    stock = goldens.get("stock") or {}
    if goldens.get("format") != 1:
        raise RuntimeError("goldens have an unsupported format")
    if stock.get("tag") != STOCK_TAG or stock.get("commit") != STOCK_COMMIT:
        raise RuntimeError("goldens were not captured from pinned %s" % STOCK_TAG)
    if stock.get("context") != args.ctx:
        raise RuntimeError("goldens use context %s, not %s" % (stock.get("context"), args.ctx))
    if stock.get("generated_tokens") != args.generated_tokens:
        raise RuntimeError("goldens use %s generated tokens, not %s"
                           % (stock.get("generated_tokens"), args.generated_tokens))
    if stock.get("source_sha256") != prompts.get("source_sha256"):
        raise RuntimeError("goldens and prompts use different source documents")
    if stock.get("prompts_sha256") != _prompts_sha256(prompts):
        raise RuntimeError("tracked prompts do not match the goldens' prompt hash")
    for model in args.models:
        row = goldens.get("models", {}).get(model)
        if not row:
            raise RuntimeError("goldens have no model %s" % model)
        for pos in args.positions:
            out = row.get("positions", {}).get(str(pos))
            if out is None:
                raise RuntimeError("goldens have no %s row at position %d" % (model, pos))
            if len(out.get("ids") or []) != args.generated_tokens:
                raise RuntimeError("goldens %s/%d do not contain %d tokens"
                                   % (model, pos, args.generated_tokens))


def _run_check(args, stock, prompts, goldens):
    _check_prompt_config(args, prompts)
    _check_goldens(goldens, args, prompts)
    rows = []
    try:
        for model in args.models:
            pack = _pack_for(args, model)
            manifest = pack / "kindi_pack.json"
            if not manifest.is_file():
                raise RuntimeError("missing pack for %s: %s" % (model, pack))
            want = goldens["models"][model]["manifest_sha256"]
            if want != _manifest_sha256(pack):
                raise RuntimeError("pack manifest changed for %s; re-capture the goldens" % model)
            stock_out = _run_worker(stock, args, BENCH_DIR / (".check-stock-%s.json" % model),
                                    model=model, is_fork=False)
            fork_out = _run_worker(ROOT, args, BENCH_DIR / (".check-fork-%s.json" % model),
                                   model=model, is_fork=True)
            rows.extend(_compare(goldens, {"models": {model: stock_out}}, "stock",
                                 models=(model,), positions=args.positions)[0])
            rows.extend(_compare(goldens, {"models": {model: fork_out}}, "fork",
                                 models=(model,), positions=args.positions)[0])
    finally:
        for model in args.models:
            for name in (".check-stock-%s.json" % model, ".check-fork-%s.json" % model):
                try:
                    (BENCH_DIR / name).unlink()
                except OSError:
                    pass
    expected = len(args.models) * len(args.positions) * 2
    if len(rows) != expected:
        raise RuntimeError("expected %d parity rows, got %d" % (expected, len(rows)))
    mismatches = [row for row in rows if row[3] != "exact"]
    return rows, mismatches


def _parse_list(text, cast):
    return tuple(cast(x.strip()) for x in text.split(",") if x.strip())


def _is_default_file(value, default):
    return Path(value).expanduser().resolve() == Path(default).resolve()


def _canonical_run(args):
    return (tuple(args.models) == MODELS and tuple(args.positions) == POSITIONS
            and args.generated_tokens == GENERATED_TOKENS and args.ctx == CONTEXT
            and not args.pack_2b and not args.pack_9b
            and _is_default_file(args.pack_root, DEFAULT_PACK_ROOT)
            and _is_default_file(args.prompts, DEFAULT_PROMPTS))


def _report_path(args):
    """The report to write: an explicit --json, else the tracked gate only for the canonical run."""
    if args.json:
        return Path(args.json)
    if _canonical_run(args) and _is_default_file(args.goldens, DEFAULT_GOLDENS):
        return DEFAULT_REPORT
    return None


def _write_failure_report(args):
    path = _report_path(args)
    if path is None:
        return
    try:
        _write_json(path, {"ok": False, "rows_exact": 0, "mismatch_count": 0,
                           "rows": [], "error": "check failed; see stderr"})
        print("report: %s (error)" % path)
    except OSError as exc:
        print("stock_parity: could not write failure report: %s" % exc, file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Stock Kindi v0.6.1 vs fork greedy parity goldens")
    parser.add_argument("--check", action="store_true",
                        help="run stock and fork against the tracked goldens (default)")
    parser.add_argument("--capture", action="store_true",
                        help="capture goldens from official v0.6.1")
    parser.add_argument("--write-prompts", action="store_true",
                        help="recreate prompts.json from the fixed source document")
    parser.add_argument("--pack-root", default=str(DEFAULT_PACK_ROOT))
    parser.add_argument("--pack-2b")
    parser.add_argument("--pack-9b")
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--goldens", default=str(DEFAULT_GOLDENS))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--positions", default=",".join(str(p) for p in POSITIONS))
    parser.add_argument("--generated-tokens", type=int, default=GENERATED_TOKENS)
    parser.add_argument("--ctx", type=int, default=CONTEXT)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", help="write the machine-readable report here instead of the tracked gate")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--tree", help=argparse.SUPPRESS)
    parser.add_argument("--pack", help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--out", help=argparse.SUPPRESS)
    parser.add_argument("--prompts-path", dest="prompts_path", help=argparse.SUPPRESS)
    parser.add_argument("--emit-prompts", dest="emit_prompts", help=argparse.SUPPRESS)
    parser.add_argument("--source-file", dest="source_file", help=argparse.SUPPRESS)
    parser.add_argument("--source-template", dest="source_template", help=argparse.SUPPRESS)
    parser.add_argument("--source-repeats", dest="source_repeats", type=int,
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        if args.emit_prompts:
            return _worker_main(args)
        # The parent passes one model at a time; keep its pack/model label.
        if not args.model:
            args.model = Path(args.pack).name.replace("bonsai-", "")
        if not args.prompts_path:
            args.prompts_path = str(DEFAULT_PROMPTS)
        return _worker_main(args)

    args.models = _parse_list(args.models, str)
    args.positions = _parse_list(args.positions, int)
    args.prompts = str(Path(args.prompts).expanduser().resolve())
    args.goldens = str(Path(args.goldens).expanduser().resolve())
    if not args.models or not args.positions:
        parser.error("--models and --positions must not be empty")
    try:
        stock = _stock_tree()
        prompts_path = Path(args.prompts)
        if args.write_prompts or not prompts_path.is_file():
            _write_prompts(args, stock)
        prompts = _load_prompts(prompts_path)
        if args.capture:
            if Path(args.goldens).exists() and not args.force:
                raise RuntimeError("%s exists; use --force to replace it" % args.goldens)
            if (_is_default_file(args.goldens, DEFAULT_GOLDENS)
                    and not _canonical_run(args)):
                raise RuntimeError("refusing to replace tracked goldens from a narrowed or "
                                   "non-default capture; pass --goldens FILE")
            _capture(args, stock, prompts)
            print("wrote %s from official %s" % (args.goldens, STOCK_TAG))
            return 0
        with open(args.goldens, encoding="utf-8") as handle:
            goldens = json.load(handle)
        rows, mismatches = _run_check(args, stock, prompts, goldens)
        print("%-5s %-6s %-6s %s" % ("model", "position", "side", "result"))
        for model, pos, side, state, first in rows:
            note = "" if first is None else " at token %d" % first
            print("%-5s %-6d %-6s %s%s" % (model, pos, side, state, note))
        exact = len(rows) - len(mismatches)
        report = {"ok": not mismatches, "rows_exact": exact,
                  "mismatch_count": len(mismatches),
                  "rows": [{"model": m, "position": p, "side": s, "result": r,
                            "first_difference": d} for m, p, s, r, d in rows]}
        report_path = _report_path(args)
        if report_path is not None:
            _write_json(report_path, report)
            print("report: %s" % report_path)
        if mismatches:
            print("FAIL: %d mismatch(es) against the %s goldens" % (len(mismatches), STOCK_TAG))
            return 1
        print("PASS: %d/%d stock and fork rows match the tracked %s goldens"
              % (exact, len(rows), STOCK_TAG))
        return 0
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        print("stock_parity: %s" % exc, file=sys.stderr)
        if not args.capture:
            _write_failure_report(args)
        return 2


if __name__ == "__main__":
    sys.exit(main())

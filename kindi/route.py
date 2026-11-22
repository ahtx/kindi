"""kindi route: one OpenAI endpoint in front of every server of the plan (docs/dev/F6-PLACEMENT.md section 7).

  kindi route [--plan ~/.kindi/plan.json] [--host 127.0.0.1] [--port 8090] [--api-key K]
  kindi route --to bonsai2-27b=http://127.0.0.1:8093 --to deepseek-v4.1-flash=http://127.0.0.1:8101

- POST /v1/chat/completions, /v1/completions, /v1/tokenize, /v1/detokenize, /v1/kindi/...: the request's "model"
  picks the server (the plan's model id, the server's name, or an id the server lists in /v1/models). With one
  server, a request without "model" goes to it. An unknown model: HTTP 404 with the models there are.
- GET /v1/models: the models of every server, merged. GET /health: each server's health.
- The response passes through unchanged (streams, progress comments and keep-alives too).
Standard library only (asyncio), Python 3.6+.

Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
"""
import argparse
import asyncio
import json
import os
import sys
import time
import urllib.request

HOP = {"connection", "keep-alive", "proxy-connection", "transfer-encoding", "upgrade", "host", "content-length"}


class Router(object):
    def __init__(self, backends, api_key=None, log=None):
        """backends: {name: (base url 'http://host:port', [model ids])}"""
        self.backends = dict(backends)
        self.key = api_key
        self.log = log or (lambda m: print("[kindi route] %s" % m, file=sys.stderr, flush=True))
        self._listed, self._listed_at = {}, 0.0
        self._turn = {}                                  # model -> the next of its servers (several servers: in turn)

    # ---- the model -> server map
    def _fetch(self, url, timeout=5):
        req = urllib.request.Request(url, headers={"Authorization": "Bearer %s" % self.key} if self.key else {})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))

    def listed(self, refresh=False):
        """{name: [ids the server lists in /v1/models]} (cached for 30 s)"""
        if refresh or time.time() - self._listed_at > 30:
            out = {}
            for name, (base, _) in self.backends.items():
                try:
                    out[name] = [m.get("id") for m in self._fetch(base + "/v1/models").get("data", []) if m.get("id")]
                except Exception:  # noqa: BLE001 - a server that is down lists nothing
                    out[name] = []
            self._listed, self._listed_at = out, time.time()
        return self._listed

    def pick(self, model):
        """the server name for a request's model, or None"""
        if not model:
            return next(iter(self.backends)) if len(self.backends) == 1 else None
        if model in self.backends:
            return model
        hits = [name for name, (base, ids) in self.backends.items() if model in ids]
        if not hits:
            for refresh in (False, True):
                hits = [name for name, ids in self.listed(refresh).items() if model in ids]
                if hits:
                    break
        if not hits:
            return None
        # one model on several servers (a 27B on each of two cards): the requests go to them in turn
        k = self._turn.get(model, 0)
        self._turn[model] = k + 1
        return hits[k % len(hits)]

    def models(self):
        data, seen = [], set()
        listed = self.listed()
        for name, (base, ids) in self.backends.items():
            for i in list(ids) + listed.get(name, []):
                if i not in seen:
                    seen.add(i)
                    data.append({"id": i, "object": "model", "owned_by": "kindi", "server": name})
        return {"object": "list", "data": data}

    # ---- HTTP
    async def handle(self, reader, writer):
        try:
            await self._handle(reader, writer)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        except Exception as e:  # noqa: BLE001 - one bad request never stops the router
            self.log("error: %r" % e)
            try:
                await self._reply(writer, 502, {"error": {"message": "kindi route: %s" % e, "type": "server_error"}})
            except Exception:  # noqa: BLE001
                pass
        finally:
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass

    async def _reply(self, writer, code, obj):
        body = json.dumps(obj).encode()
        reason = {200: "OK", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found", 411: "Length Required",
                  502: "Bad Gateway"}.get(code, "Error")
        writer.write(("HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n"
                      % (code, reason, len(body))).encode() + body)
        await writer.drain()

    async def _handle(self, reader, writer):
        head = await reader.readuntil(b"\r\n\r\n")
        lines = head.decode("latin-1").split("\r\n")
        method, path, _ = (lines[0].split(" ", 2) + ["", ""])[:3]
        hdr = []
        for ln in lines[1:]:
            if ":" in ln:
                k, v = ln.split(":", 1)
                hdr.append((k.strip(), v.strip()))
        hl = {k.lower(): v for k, v in hdr}
        if hl.get("transfer-encoding", "").lower() == "chunked":
            return await self._reply(writer, 411, {"error": {"message": "send a Content-Length", "type": "invalid_request_error"}})
        n = int(hl.get("content-length") or 0)
        body = await reader.readexactly(n) if n else b""
        if self.key and hl.get("authorization", "") != "Bearer %s" % self.key:
            return await self._reply(writer, 401, {"error": {"message": "wrong or missing API key", "type": "invalid_request_error"}})
        p = path.split("?", 1)[0]
        if method == "GET" and p in ("/v1/models", "/models"):
            return await self._reply(writer, 200, await asyncio.get_event_loop().run_in_executor(None, self.models))
        if method == "GET" and p == "/health":
            def health():
                out = {}
                for name, (base, _) in self.backends.items():
                    try:
                        out[name] = self._fetch(base + "/health", timeout=3)
                    except Exception as e:  # noqa: BLE001
                        out[name] = {"ok": False, "error": str(e)}
                return {"ok": all(v.get("ok", True) is not False for v in out.values()), "servers": out}
            return await self._reply(writer, 200, await asyncio.get_event_loop().run_in_executor(None, health))
        model = None
        if body:
            try:
                model = json.loads(body.decode("utf-8")).get("model")
            except (ValueError, AttributeError):
                return await self._reply(writer, 400, {"error": {"message": "the body is not JSON", "type": "invalid_request_error"}})
        name = await asyncio.get_event_loop().run_in_executor(None, self.pick, model)
        if name is None:
            avail = [m["id"] for m in (await asyncio.get_event_loop().run_in_executor(None, self.models))["data"]]
            return await self._reply(writer, 404, {"error": {
                "message": "no server for model %r; the models here: %s" % (model, ", ".join(avail)) if model else
                "several servers: say which one with \"model\" (%s)" % ", ".join(avail),
                "type": "invalid_request_error", "code": "model_not_found"}})
        base = self.backends[name][0]
        hostport = base.split("//", 1)[1].rstrip("/")
        host, _, port = hostport.partition(":")
        r2, w2 = await asyncio.open_connection(host, int(port or 80))
        out = ["%s %s HTTP/1.1" % (method, path), "Host: %s" % hostport, "Connection: close",
               "Content-Length: %d" % len(body)]
        out += ["%s: %s" % (k, v) for k, v in hdr if k.lower() not in HOP]
        w2.write(("\r\n".join(out) + "\r\n\r\n").encode("latin-1") + body)
        await w2.drain()

        async def pump():
            while True:
                b = await r2.read(65536)
                if not b:
                    break
                writer.write(b)
                await writer.drain()
        # the client going away stops the copy and closes the server connection (the server then cancels)
        try:
            await pump()
        finally:
            w2.close()


def backends_from_plan(path):
    p = json.load(open(os.path.expanduser(path)))
    ids = {}
    for mid, names in p.get("front", {}).get("models", {}).items():
        for name in ([names] if isinstance(names, str) else names):
            ids.setdefault(name, []).append(mid)
    return {s["name"]: ("http://127.0.0.1:%d" % s["port"], ids.get(s["name"], [s.get("id")])) for s in p["servers"]}


def serve(router, host, port, ready=None):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    srv = loop.run_until_complete(asyncio.start_server(router.handle, host, port))
    router.log("on http://%s:%d for %s" % (host, port, ", ".join("%s (%s)" % (n, b[0]) for n, b in router.backends.items())))
    if ready is not None:
        ready.set()
    try:
        loop.run_forever()
    finally:
        srv.close()
        loop.close()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kindi route")
    ap.add_argument("--plan", default="~/.kindi/plan.json")
    ap.add_argument("--to", action="append", default=[], help="MODEL=URL (instead of a plan)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int)
    ap.add_argument("--api-key", dest="api_key", default=os.environ.get("KINDI_API_KEY"))
    a = ap.parse_args(argv)
    if a.to:
        be = {}
        for t in a.to:
            m, _, url = t.partition("=")
            be[m] = (url.rstrip("/").replace("/v1", ""), [m])
        port = a.port or 8090
    else:
        be = backends_from_plan(a.plan)
        port = a.port or json.load(open(os.path.expanduser(a.plan))).get("front", {}).get("port", 8090)
    serve(Router(be, a.api_key), a.host, port)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Admission limits in front of the server (PR #63, atimelessrage): a cap on the requests held at once, a cap on the
request body, and an optional per-client request rate.

- queue_max: a generation request (chat completions, messages) that arrives while every decode slot is busy and
  queue_max requests already wait gets HTTP 429 with code server_busy and Retry-After: 1, before the server reads
  its prompt. 0: no cap (all requests queue, as before 0.8).
- max_body_mb: a body larger than this gets HTTP 413 (code request_too_large) before the server parses it; the body
  is never held past the cap.
- rate_limit_per_min: at most this many /v1 requests per client address in any 60 seconds; more gets HTTP 429 with
  code rate_limit_exceeded and the seconds to wait in Retry-After. 0: off.

This is ASGI middleware, not code in each endpoint: a request is counted from its arrival until the last byte of its
answer is sent (or the client goes away), streamed answers included, so no path can forget to give its place back.
No FastAPI import; standard library only."""
import collections
import json
import threading
import time

GENERATE = ("/v1/chat/completions", "/v1/messages")


def _body(message, typ, code):
    return {"error": {"message": message, "type": typ, "param": None, "code": code}}


class RateLimiter(object):
    """at most per_min requests per key in any window_s seconds"""

    def __init__(self, per_min, window_s=60.0, clock=time.monotonic):
        self.per_min, self.window_s, self.clock = max(0, int(per_min)), float(window_s), clock
        self.seen, self.lk, self.refused = {}, threading.Lock(), 0

    def wait_s(self, key):
        """0 when the request may go now (and counts it), else the whole seconds until it may"""
        if self.per_min <= 0:
            return 0
        now = self.clock()
        with self.lk:
            q = self.seen.setdefault(key, collections.deque())
            while q and q[0] <= now - self.window_s:
                q.popleft()
            if len(q) >= self.per_min:
                self.refused += 1
                return max(1, int(q[0] + self.window_s - now + 0.999))
            q.append(now)
            if len(self.seen) > 4096:                       # forget clients that sent nothing for a window
                for k in [k for k, v in self.seen.items() if not v or v[-1] <= now - self.window_s]:
                    del self.seen[k]
        return 0


class Admission(object):
    """ASGI middleware. running(): the decode slots (requests that run at once); the cap is running() + queue_max."""

    def __init__(self, app, queue_max=64, max_body_mb=32, rate_limit_per_min=0, running=lambda: 1):
        self.app = app
        self.queue_max = max(0, int(queue_max))
        self.max_body = max(1, int(max_body_mb)) << 20
        self.limiter = RateLimiter(rate_limit_per_min)
        self.running = running
        self.held, self.refused_busy, self.refused_large = 0, 0, 0

    def stats(self):
        return {"held": self.held, "queue_max": self.queue_max, "refused_busy": self.refused_busy,
                "refused_too_large": self.refused_large, "refused_rate": self.limiter.refused,
                "rate_limit_per_min": self.limiter.per_min}

    @staticmethod
    async def _reply(send, status, body, headers=()):
        raw = json.dumps(body).encode()
        h = [(b"content-type", b"application/json"), (b"content-length", str(len(raw)).encode())]
        h += [(k.encode(), v.encode()) for k, v in headers]
        await send({"type": "http.response.start", "status": status, "headers": h})
        await send({"type": "http.response.body", "body": raw})

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "") if scope.get("type") == "http" else ""
        if not path.startswith("/v1/") or scope.get("method") != "POST":
            return await self.app(scope, receive, send)
        client = (scope.get("client") or ("unknown", 0))[0]
        wait = self.limiter.wait_s(client)
        if wait:
            return await self._reply(send, 429, _body("too many requests from %s: at most %d a minute; retry in %d s"
                                                      % (client, self.limiter.per_min, wait), "rate_limit_exceeded",
                                                      "rate_limit_exceeded"), [("Retry-After", str(wait))])
        # the body, up to the cap: a declared length past it is refused before a byte is read
        n = dict(scope.get("headers") or []).get(b"content-length")
        if n is not None and n.isdigit() and int(n) > self.max_body:
            return await self._too_large(send)
        parts, size, more = [], 0, True
        while more:
            m = await receive()
            if m["type"] == "http.disconnect":
                return
            parts.append(m.get("body", b""))
            size += len(parts[-1])
            if size > self.max_body:
                return await self._too_large(send)
            more = m.get("more_body", False)
        body, sent = b"".join(parts), [False]

        async def replay():
            if not sent[0]:
                sent[0] = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()                          # after the body: the disconnect, when it comes
        if path not in GENERATE:
            return await self.app(scope, replay, send)
        if self.queue_max and self.held >= self.running() + self.queue_max:
            self.refused_busy += 1
            return await self._reply(send, 429, _body("the server is busy: %d requests running or waiting (%d decode "
                                                      "slot(s) + queue_max %d); retry shortly"
                                                      % (self.held, self.running(), self.queue_max),
                                                      "server_busy", "server_busy"), [("Retry-After", "1")])
        self.held += 1                                      # one event loop: no lock needed
        try:
            await self.app(scope, replay, send)
        finally:
            self.held -= 1

    async def _too_large(self, send):
        self.refused_large += 1
        return await self._reply(send, 413, _body("the request body is larger than max_body_mb (%d MiB)"
                                                  % (self.max_body >> 20), "invalid_request_error",
                                                  "request_too_large"))

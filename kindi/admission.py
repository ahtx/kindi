# SPDX-License-Identifier: AGPL-3.0-only
"""Small admission-control primitives for the one-engine HTTP server.

The server runs one generation at a time. A bounded queue and a per-client
request budget keep a burst from growing the waiting heap without limit, and a
body cap keeps a single client from making the host hold an arbitrary upload.
The module has no FastAPI or engine import, so it is CPU-testable.
"""
import collections
import threading
import time


class QueueFull(RuntimeError):
    """The bounded wait behind the active request is full."""


class RequestTooLarge(ValueError):
    """The request body exceeds the configured cap."""


class Gate(object):
    """Count admitted requests: one running plus up to ``max_queue`` waiting."""

    def __init__(self, max_queue):
        self.max_queue = max(0, int(max_queue))
        self.admitted = 0
        self.rejected = 0
        self._lock = threading.Lock()

    @property
    def waiting(self):
        with self._lock:
            return max(0, self.admitted - 1)

    def enter(self):
        with self._lock:
            if self.admitted >= self.max_queue + 1:
                self.rejected += 1
                raise QueueFull("the server is busy: %d request(s) queued (KINDI_QUEUE_MAX=%d)"
                                % (self.admitted - 1, self.max_queue))
            self.admitted += 1

    def release(self):
        with self._lock:
            if self.admitted > 0:
                self.admitted -= 1


class RateLimiter(object):
    """At most ``per_minute`` requests per client in a rolling window."""

    def __init__(self, per_minute, window_s=60.0):
        self.per_minute = max(0, int(per_minute))
        self.window_s = float(window_s)
        self.rejected = 0
        self._seen = {}
        self._lock = threading.Lock()

    def check(self, client):
        """Return seconds to retry after, or 0 when the request is allowed."""
        if self.per_minute <= 0:
            return 0
        now = time.monotonic()
        key = client or "unknown"
        with self._lock:
            rows = self._seen.setdefault(key, collections.deque())
            while rows and rows[0] <= now - self.window_s:
                rows.popleft()
            if len(rows) >= self.per_minute:
                self.rejected += 1
                return max(1, int(rows[0] + self.window_s - now + 0.999))
            rows.append(now)
        return 0

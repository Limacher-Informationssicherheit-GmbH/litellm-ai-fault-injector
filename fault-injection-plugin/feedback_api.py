# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reaction-capture service — the second half of the awareness tool.

The injector records *what* it manipulated; this service records *how the user
reacted*, so ``report.py`` can join the two on ``request_id`` and compute a
"noticed" rate. Without this half the tool cannot measure the thing it exists to
measure.

**The ``request_id`` a client sends here MUST be the ``id`` field from the
response body it received** (the completion id, e.g. ``chatcmpl-...``). The
injector keys its audit log on that same client-visible id, so this is what
makes the injections⋈feedback join succeed.

Signals are open-ended (``noticed``, ``corrected``, ``reasked``, ``thumbs_down``,
``none``); only the first four are credited as "noticed" by ``report.py``.

Config:
- Feedback path: ``FAULT_FEEDBACK_LOG`` env, else ``feedback_log_path`` from the
  proxy config (``FAULT_INJECTION_CONFIG``, else the YAML carrying a
  ``fault_injection:`` block beside this file).
- Auth: set ``FAULT_FEEDBACK_TOKEN`` to require ``Authorization: Bearer
  <token>``. Unset = **open** — the service then logs a startup warning and you
  must bind it to localhost. Do not expose an untokenised instance on 0.0.0.0.
- Rate limit: ``FAULT_FEEDBACK_RATE_LIMIT`` requests per minute per client IP
  (default 60; ``0`` disables). Bounds append-volume on the shared audit volume
  from an unauthenticated or token-holding client alike.

Run alongside the LiteLLM proxy, bound to loopback:

    uvicorn feedback_api:app --host 127.0.0.1 --port 8181
"""

from __future__ import annotations

import hmac
import logging
import os
import sys
import threading
import time
from collections import OrderedDict, deque
from datetime import datetime, timezone
from typing import Deque, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

# uvicorn is normally launched from the plugin root, but nothing guarantees it —
# and this module resolves its config relative to its own file, so it must be
# importable from any working directory too.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

import audit_log  # noqa: E402  (import needs the path bootstrap above)
from config import FaultInjectionConfig  # noqa: E402

logger = logging.getLogger("fault_injection.feedback")


def _resolve_feedback_path() -> str:
    """Single source of truth: env override, else the shared config value.

    Uses the same content-based discovery as the injector, so both halves of
    the tool agree on where feedback.jsonl lives even when the two processes
    have different working directories — otherwise report.py joins two files
    that were never in the same tree, and reports nothing, with no error.
    """
    env = os.getenv("FAULT_FEEDBACK_LOG")
    if env:
        return env
    return FaultInjectionConfig.discover(_HERE).feedback_log_path


def _resolve_rate_limit() -> int:
    raw = os.getenv("FAULT_FEEDBACK_RATE_LIMIT", "60")
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("FAULT_FEEDBACK_RATE_LIMIT=%r is not an integer; using 60", raw)
        return 60


FEEDBACK_PATH = _resolve_feedback_path()
FEEDBACK_TOKEN = os.getenv("FAULT_FEEDBACK_TOKEN")
RATE_LIMIT_PER_MIN = _resolve_rate_limit()

if not FEEDBACK_TOKEN:
    logger.warning(
        "FAULT_FEEDBACK_TOKEN is unset — POST /feedback is UNAUTHENTICATED. "
        "Bind this service to 127.0.0.1 (uvicorn --host 127.0.0.1) or set a "
        "token; an open endpoint lets anyone forge reaction signals and skew "
        "the noticed-rate the tool exists to measure."
    )

app = FastAPI(title="Fault-Injection Feedback API")


class _RateLimiter:
    """Sliding-window per-client-IP limiter, bounded by an LRU of clients.

    In-process only: it bounds one worker, not a fleet, and it is not a defence
    against a distributed source. Its job is to stop a single misbehaving or
    unauthenticated client from filling the shared audit volume with appends.

    The client map is an explicit LRU. A previous version kept a
    ``defaultdict(deque)`` and tried to reclaim entries whose window had
    emptied — but every code path that drains a window appends to it in the
    same call, so an empty window was never observable and the reclaim branch
    freed nothing, ever. On a token-less endpoint (the default) that is one
    permanent dict entry per distinct source IP: the limiter added to bound
    growth was itself the unbounded resource.
    """

    #: distinct clients tracked before the least-recently-seen is dropped.
    #: Dropping only forgives history, it never grants extra allowance beyond
    #: one fresh window, so eviction cannot be used to bypass the limit.
    _MAX_CLIENTS = 10_000

    def __init__(self, per_minute: int, max_clients: int = _MAX_CLIENTS) -> None:
        self._per_minute = per_minute
        self._max_clients = max_clients
        self._lock = threading.Lock()
        self._hits: "OrderedDict[str, Deque[float]]" = OrderedDict()

    def allow(self, client: str, now: float) -> bool:
        if self._per_minute <= 0:
            return True
        cutoff = now - 60.0
        with self._lock:
            window = self._hits.get(client)
            if window is None:
                window = deque()
                self._hits[client] = window
            self._hits.move_to_end(client)
            while window and window[0] < cutoff:
                window.popleft()
            if len(window) >= self._per_minute:
                return False
            window.append(now)
            while len(self._hits) > self._max_clients:
                self._hits.popitem(last=False)
            return True


LIMITER = _RateLimiter(RATE_LIMIT_PER_MIN)


async def enforce_rate_limit(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    if not LIMITER.allow(client, time.monotonic()):
        raise HTTPException(status_code=429, detail="rate limit exceeded")


async def require_auth(authorization: Optional[str] = Header(default=None)) -> None:
    """Require a bearer token iff FAULT_FEEDBACK_TOKEN is configured.

    Uses a constant-time compare so the token can't be recovered byte-by-byte
    via response-timing.
    """
    if not FEEDBACK_TOKEN:
        return
    # Compare as bytes: compare_digest on str raises TypeError unless both
    # sides are ASCII, and Starlette decodes headers as latin-1 — so a
    # non-ASCII Authorization header would 500 instead of 401.
    expected = f"Bearer {FEEDBACK_TOKEN}".encode("utf-8")
    got = (authorization or "").encode("utf-8")
    if not hmac.compare_digest(got, expected):
        raise HTTPException(status_code=401, detail="unauthorized")


class Feedback(BaseModel):
    # bounds prevent unauthenticated disk-fill on the shared audit volume
    request_id: str = Field(..., min_length=1, max_length=200)
    signal: str = Field(..., min_length=1, max_length=32)
    note: Optional[str] = Field(default=None, max_length=2000)
    session_id: Optional[str] = Field(default=None, max_length=200)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/feedback")
async def submit_feedback(
    fb: Feedback,
    _auth: None = Depends(require_auth),
    _rate: None = Depends(enforce_rate_limit),
) -> dict:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": fb.request_id,
        "signal": fb.signal,
        "note": fb.note,
        "session_id": fb.session_id,
    }
    audit_log.append_json(FEEDBACK_PATH, entry)
    return {"status": "recorded", "request_id": fb.request_id}

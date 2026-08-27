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
  proxy config (``FAULT_INJECTION_CONFIG``, default: ``proxy_config.yaml`` beside
  this file).
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
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Deque, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

import audit_log  # reuse the atomic JSONL append writer
from config import FaultInjectionConfig

logger = logging.getLogger("fault_injection.feedback")

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG_PATH = os.path.join(_HERE, "proxy_config.yaml")


def _resolve_feedback_path() -> str:
    """Single source of truth: env override, else the shared config value."""
    env = os.getenv("FAULT_FEEDBACK_LOG")
    if env:
        return env
    try:
        cfg = FaultInjectionConfig.from_yaml(
            os.getenv("FAULT_INJECTION_CONFIG", _DEFAULT_CONFIG_PATH)
        )
        return cfg.feedback_log_path
    except FileNotFoundError:
        return "./audit/feedback.jsonl"


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
    """Sliding-window per-client-IP limiter.

    In-process only: it bounds one worker, not a fleet, and it is not a defence
    against a distributed source. Its job is to stop a single misbehaving or
    unauthenticated client from filling the shared audit volume with appends.
    """

    def __init__(self, per_minute: int) -> None:
        self._per_minute = per_minute
        self._lock = threading.Lock()
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def allow(self, client: str, now: float) -> bool:
        if self._per_minute <= 0:
            return True
        with self._lock:
            window = self._hits[client]
            cutoff = now - 60.0
            while window and window[0] < cutoff:
                window.popleft()
            if len(window) >= self._per_minute:
                return False
            window.append(now)
            # keep the map from growing without bound across many client IPs
            if len(self._hits) > 10_000:
                for key in [k for k, v in self._hits.items() if not v][:5_000]:
                    del self._hits[key]
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
    if FEEDBACK_TOKEN and not hmac.compare_digest(
        authorization or "", f"Bearer {FEEDBACK_TOKEN}"
    ):
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

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
  proxy config (``FAULT_INJECTION_CONFIG``, default ``config/proxy_config.yaml``).
- Optional auth: set ``FAULT_FEEDBACK_TOKEN`` to require ``Authorization: Bearer
  <token>``. Unset = open (bind to localhost/private network in that case).

Run alongside the LiteLLM proxy:

    uvicorn feedback_api:app --port 8181
"""

from __future__ import annotations

import hmac
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

import audit_log  # reuse the atomic JSONL append writer
from config import FaultInjectionConfig


def _resolve_feedback_path() -> str:
    """Single source of truth: env override, else the shared config value."""
    env = os.getenv("FAULT_FEEDBACK_LOG")
    if env:
        return env
    try:
        cfg = FaultInjectionConfig.from_yaml(
            os.getenv("FAULT_INJECTION_CONFIG", "config/proxy_config.yaml")
        )
        return cfg.feedback_log_path
    except FileNotFoundError:
        return "./audit/feedback.jsonl"


FEEDBACK_PATH = _resolve_feedback_path()
FEEDBACK_TOKEN = os.getenv("FAULT_FEEDBACK_TOKEN")

app = FastAPI(title="Fault-Injection Feedback API")


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
async def submit_feedback(fb: Feedback, _: None = Depends(require_auth)) -> dict:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": fb.request_id,
        "signal": fb.signal,
        "note": fb.note,
        "session_id": fb.session_id,
    }
    audit_log.append_json(FEEDBACK_PATH, entry)
    return {"status": "recorded", "request_id": fb.request_id}

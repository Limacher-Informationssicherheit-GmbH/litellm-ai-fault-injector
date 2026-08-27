# SPDX-License-Identifier: AGPL-3.0-or-later
import json

import pytest
from fastapi import HTTPException

import feedback_api


async def test_auth_noop_when_no_token(monkeypatch):
    monkeypatch.setattr(feedback_api, "FEEDBACK_TOKEN", None)
    await feedback_api.require_auth(None)  # must not raise


async def test_auth_rejects_wrong_token(monkeypatch):
    monkeypatch.setattr(feedback_api, "FEEDBACK_TOKEN", "s3cret")
    with pytest.raises(HTTPException) as ei:
        await feedback_api.require_auth("Bearer wrong")
    assert ei.value.status_code == 401


async def test_auth_rejects_missing_header(monkeypatch):
    monkeypatch.setattr(feedback_api, "FEEDBACK_TOKEN", "s3cret")
    with pytest.raises(HTTPException):
        await feedback_api.require_auth(None)


async def test_auth_accepts_correct_token(monkeypatch):
    monkeypatch.setattr(feedback_api, "FEEDBACK_TOKEN", "s3cret")
    await feedback_api.require_auth("Bearer s3cret")  # must not raise


async def test_submit_feedback_writes_line(tmp_path, monkeypatch):
    path = tmp_path / "fb.jsonl"
    monkeypatch.setattr(feedback_api, "FEEDBACK_PATH", str(path))
    fb = feedback_api.Feedback(request_id="chatcmpl-1", signal="noticed", note="x")
    result = await feedback_api.submit_feedback(fb)
    assert result["status"] == "recorded"
    row = json.loads(path.read_text().strip())
    assert row["request_id"] == "chatcmpl-1"
    assert row["signal"] == "noticed"


def test_feedback_length_bounds_reject_oversized():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        feedback_api.Feedback(request_id="r", signal="noticed", note="x" * 3000)


def test_rate_limiter_bounds_appends_per_client():
    limiter = feedback_api._RateLimiter(per_minute=3)
    assert [limiter.allow("1.2.3.4", 100.0) for _ in range(4)] == [True, True, True, False]
    # a different client is unaffected
    assert limiter.allow("5.6.7.8", 100.0) is True
    # the window slides
    assert limiter.allow("1.2.3.4", 161.0) is True


def test_rate_limiter_disabled_when_zero():
    limiter = feedback_api._RateLimiter(per_minute=0)
    assert all(limiter.allow("1.2.3.4", 100.0) for _ in range(100))


async def test_rate_limited_request_gets_429(monkeypatch):
    class _Client:
        host = "9.9.9.9"

    class _Req:
        client = _Client()

    monkeypatch.setattr(feedback_api, "LIMITER", feedback_api._RateLimiter(1))
    await feedback_api.enforce_rate_limit(_Req())
    with pytest.raises(HTTPException) as ei:
        await feedback_api.enforce_rate_limit(_Req())
    assert ei.value.status_code == 429


def test_rate_limiter_client_map_is_bounded():
    # The previous reclaim branch scanned for empty windows, but every path
    # that drains a window appends in the same call — so it never freed
    # anything and the map grew one entry per source IP forever, on an
    # endpoint that is unauthenticated by default.
    limiter = feedback_api._RateLimiter(per_minute=5, max_clients=100)
    for i in range(5_000):
        limiter.allow(f"10.0.{i // 256}.{i % 256}", 100.0)
    assert len(limiter._hits) <= 100


def test_rate_limiter_still_limits_a_persistent_client_under_churn():
    limiter = feedback_api._RateLimiter(per_minute=2, max_clients=100)
    assert limiter.allow("1.1.1.1", 100.0) is True
    assert limiter.allow("1.1.1.1", 100.0) is True
    assert limiter.allow("1.1.1.1", 100.0) is False


async def test_auth_rejects_non_ascii_header_without_raising(monkeypatch):
    # Starlette decodes headers as latin-1; compare_digest on str raises
    # TypeError unless both sides are ASCII, which would 500 instead of 401.
    monkeypatch.setattr(feedback_api, "FEEDBACK_TOKEN", "s3cret")
    with pytest.raises(HTTPException) as ei:
        await feedback_api.require_auth("Bearer ünicode")
    assert ei.value.status_code == 401

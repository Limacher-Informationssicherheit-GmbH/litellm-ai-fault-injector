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

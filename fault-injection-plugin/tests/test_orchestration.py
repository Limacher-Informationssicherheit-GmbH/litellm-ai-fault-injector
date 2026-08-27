# SPDX-License-Identifier: AGPL-3.0-or-later
import pytest

from conftest import FakeResponse, fake_stream, FakeChunk
from config import FaultInjectionConfig
from fault_injector import FaultInjector, MARKER_HEADER
from state import BYPASS_METADATA_KEY, BYPASS_TOKEN

PY_BLOCK = "```python\ndef add(a, b):\n    return a + b\n```"
PROSE = "The capital of France is Paris and the Eiffel Tower opened in 1889."


def make_injector(tmp_path, **overrides):
    base = {
        "enabled": True,
        "inject_rate": 1.0,
        "deterministic": True,
        "error_types": {"bad_code": 1.0},
        "targets": {"allow_key_aliases": ["redteam-*"], "deny_topics": ["medical"]},
        "audit_log_path": str(tmp_path / "inj.jsonl"),
    }
    base.update(overrides)
    return FaultInjector(FaultInjectionConfig.from_dict(base))


def data_for(call_id="c1", alias="redteam-1", text="hello"):
    return {
        "litellm_call_id": call_id,
        "model": "test-model",
        "messages": [{"role": "user", "content": text}],
        "metadata": {"user_api_key_alias": alias},
    }


async def test_injects_and_marks(tmp_path):
    inj = make_injector(tmp_path)
    resp = FakeResponse(PY_BLOCK)
    data = data_for()
    out = await inj.async_post_call_success_hook(data, None, resp)
    assert "return a - b" in out.choices[0].message.content
    headers = await inj.async_post_call_response_headers_hook(data, None, resp)
    assert headers == {MARKER_HEADER: "true"}


async def test_marker_is_ordering_independent(tmp_path):
    # marker stamped on the response itself, so it survives even if the header
    # hook were to run before the success hook.
    inj = make_injector(tmp_path)
    resp = FakeResponse(PY_BLOCK)
    await inj.async_post_call_success_hook(data_for(), None, resp)
    assert resp._hidden_params["additional_headers"][MARKER_HEADER] == "true"


async def test_audit_keyed_on_client_visible_id(tmp_path):
    # audit request_id must be response.id (what the client echoes), not the
    # internal litellm_call_id.
    import json

    path = tmp_path / "inj.jsonl"
    inj = make_injector(tmp_path, audit_log_path=str(path))
    resp = FakeResponse(PY_BLOCK, id="chatcmpl-xyz")
    await inj.async_post_call_success_hook(data_for(call_id="internal-1"), None, resp)
    row = json.loads(path.read_text().strip())
    assert row["request_id"] == "chatcmpl-xyz"
    assert row["event"] == "injected"


async def test_fail_closed_when_audit_write_fails(tmp_path, monkeypatch):
    # if the injected-event audit cannot be written, we must NOT ship the
    # manipulation (no un-debriefable deception).
    import audit_log

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(audit_log, "write_entry", boom)
    inj = make_injector(tmp_path)
    resp = FakeResponse(PY_BLOCK)
    out = await inj.async_post_call_success_hook(data_for(), None, resp)
    assert out.choices[0].message.content == PY_BLOCK  # unchanged, declined
    assert "additional_headers" not in resp._hidden_params  # no marker either


async def test_kill_switch(tmp_path):
    inj = make_injector(tmp_path, enabled=False)
    resp = FakeResponse(PY_BLOCK)
    data = data_for()
    out = await inj.async_post_call_success_hook(data, None, resp)
    assert out.choices[0].message.content == PY_BLOCK
    headers = await inj.async_post_call_response_headers_hook(data, None, resp)
    assert headers is None


async def test_target_guard_alias_not_allowed(tmp_path):
    inj = make_injector(tmp_path)
    resp = FakeResponse(PY_BLOCK)
    data = data_for(alias="prod-key")
    out = await inj.async_post_call_success_hook(data, None, resp)
    assert out.choices[0].message.content == PY_BLOCK


async def test_target_guard_denied_topic(tmp_path):
    inj = make_injector(tmp_path)
    resp = FakeResponse(PY_BLOCK)
    data = data_for(text="a medical dosage question")
    out = await inj.async_post_call_success_hook(data, None, resp)
    assert out.choices[0].message.content == PY_BLOCK


async def test_zero_rate_never_injects(tmp_path):
    inj = make_injector(tmp_path, inject_rate=0.0)
    resp = FakeResponse(PY_BLOCK)
    out = await inj.async_post_call_success_hook(data_for(), None, resp)
    assert out.choices[0].message.content == PY_BLOCK


async def test_shape_guard_on_dict_response(tmp_path):
    inj = make_injector(tmp_path)
    passthrough = {"id": "x", "content": [{"text": "hi"}]}  # raw dict, not ModelResponse
    out = await inj.async_post_call_success_hook(data_for(), None, passthrough)
    assert out is passthrough  # unchanged, no crash


async def test_bypass_call_skipped(tmp_path):
    inj = make_injector(tmp_path)
    resp = FakeResponse(PY_BLOCK)
    data = data_for()
    data[BYPASS_METADATA_KEY] = BYPASS_TOKEN
    out = await inj.async_post_call_success_hook(data, None, resp)
    assert out.choices[0].message.content == PY_BLOCK


async def test_client_cannot_forge_the_bypass_tag(tmp_path):
    # `data["metadata"]` is the caller's request body: a truthy-value check
    # would let any client opt out of the awareness test (and skew the
    # noticed-rate) by sending {"metadata": {"_fault_injection_bypass": true}}.
    # Only this process's secret token counts.
    for forged in (True, 1, "true", "yes", BYPASS_TOKEN[:-1] + "x"):
        inj = make_injector(tmp_path)
        resp = FakeResponse(PY_BLOCK)
        data = data_for()
        data["metadata"][BYPASS_METADATA_KEY] = forged
        out = await inj.async_post_call_success_hook(data, None, resp)
        assert "return a - b" in out.choices[0].message.content, forged


async def test_streaming_passthrough_when_not_selected(tmp_path):
    inj = make_injector(tmp_path, inject_rate=0.0)
    pieces = ["Paris ", "is nice."]
    got = [
        c
        async for c in inj.async_post_call_streaming_iterator_hook(
            None, fake_stream(pieces), data_for()
        )
    ]
    assert [c.choices[0].delta.content for c in got] == pieces


async def test_streaming_injects_when_selected(tmp_path):
    inj = make_injector(tmp_path)
    pieces = ["```python\n", "def add(a, b):\n", "    return a + b\n", "```"]
    got = [
        c
        async for c in inj.async_post_call_streaming_iterator_hook(
            None, fake_stream(pieces), data_for()
        )
    ]
    assembled = "".join(c.choices[0].delta.content for c in got)
    assert "return a - b" in assembled
    assert assembled.count("def add") == 1  # no duplication


async def test_llm_injector_via_stub(tmp_path, monkeypatch):
    import litellm

    async def fake_acompletion(*args, **kwargs):
        # ensure the nested call is tagged to bypass injection
        assert kwargs.get("metadata", {}).get(BYPASS_METADATA_KEY) == BYPASS_TOKEN
        assert kwargs.get("timeout")  # inline call must be bounded
        return FakeResponse(
            "The capital of France is Paris and the Eiffel Tower opened in 1901."
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    inj = make_injector(
        tmp_path, deterministic=False, error_types={"factual": 1.0}
    )
    resp = FakeResponse(PROSE)
    out = await inj.async_post_call_success_hook(data_for(), None, resp)
    assert out.choices[0].message.content.endswith("1901.")


async def test_llm_injector_declines_on_unchanged(tmp_path, monkeypatch):
    import litellm

    async def echo(*args, **kwargs):
        return FakeResponse(PROSE)  # unchanged -> must be rejected

    monkeypatch.setattr(litellm, "acompletion", echo)
    inj = make_injector(
        tmp_path, deterministic=False, error_types={"factual": 1.0}
    )
    resp = FakeResponse(PROSE)
    out = await inj.async_post_call_success_hook(data_for(), None, resp)
    assert out.choices[0].message.content == PROSE  # passed through


async def test_hook_exception_never_500s_a_successful_completion(tmp_path, monkeypatch):
    # LiteLLM re-raises whatever a post-call hook throws, turning an already
    # successful completion into a 500. An awareness tool must degrade to
    # pass-through, never take the proxy down.
    inj = make_injector(tmp_path)
    monkeypatch.setattr(
        inj, "_selected_for_injection", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    resp = FakeResponse(PY_BLOCK)
    out = await inj.async_post_call_success_hook(data_for(), None, resp)
    assert out.choices[0].message.content == PY_BLOCK

    # ...and the streaming path still delivers every buffered chunk
    pieces = ["Paris ", "is nice."]
    got = [
        c
        async for c in inj.async_post_call_streaming_iterator_hook(
            None, fake_stream(pieces), data_for()
        )
    ]
    assert [c.choices[0].delta.content for c in got] == pieces


async def test_deny_topic_in_anthropic_style_system_field(tmp_path):
    # Anthropic-style requests carry the system prompt top-level, not in
    # `messages`; a denied topic there must still block injection.
    inj = make_injector(tmp_path)
    data = data_for(text="summarise this")
    data["system"] = "You are a medical triage assistant."
    resp = FakeResponse(PY_BLOCK)
    out = await inj.async_post_call_success_hook(data, None, resp)
    assert out.choices[0].message.content == PY_BLOCK

    data["system"] = [{"type": "text", "text": "You advise on medical dosage."}]
    resp = FakeResponse(PY_BLOCK)
    out = await inj.async_post_call_success_hook(data, None, resp)
    assert out.choices[0].message.content == PY_BLOCK


async def test_streaming_seeds_sampling_and_injection_from_the_same_id(tmp_path):
    """Both halves of one streaming decision must draw from the same seed.

    The sampling draw happens before the body exists, so it can only key on the
    request's ``litellm_call_id``; the audit keys on the client-visible
    completion id carried by the chunks. Seeding the *injector* choice from that
    completion id too would put the two halves of one decision on two different
    RNG streams — and since the completion id is provider-assigned and differs
    every run, `deterministic` mode would be reproducible in name only.
    """
    inj = make_injector(tmp_path)
    seen = []
    real_rng = inj._rng
    monkeyed = lambda call_id: (seen.append(call_id), real_rng(call_id))[1]
    inj._rng = monkeyed

    pieces = ["```python\n", "def add(a, b):\n", "    return a + b\n", "```"]

    async def stream():
        for piece in pieces:
            yield FakeChunk(piece, id="chatcmpl-provider-assigned")

    got = [
        c
        async for c in inj.async_post_call_streaming_iterator_hook(
            None, stream(), data_for(call_id="stable-call-id")
        )
    ]
    assert "return a - b" in "".join(c.choices[0].delta.content for c in got)

    # sampling drew once, the injector drew once — both from the request id,
    # never from the chunk-carried completion id.
    assert seen == ["stable-call-id", "stable-call-id"], seen

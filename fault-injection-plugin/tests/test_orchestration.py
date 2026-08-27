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


async def test_streaming_seeds_from_the_request_id_not_the_completion_id(tmp_path):
    """Sampling and injector choice must key on the same *call id* — the
    request's, not the chunk-carried completion id, which is provider-assigned
    and differs on every run — while drawing from independent streams."""
    inj = make_injector(tmp_path)
    seen = []
    real_rng = inj._rng
    inj._rng = lambda call_id, purpose: (
        seen.append((call_id, purpose)),
        real_rng(call_id, purpose),
    )[1]

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
    assert seen == [("stable-call-id", "sample"), ("stable-call-id", "choose")], seen


async def test_configured_error_type_weights_are_actually_honoured(tmp_path):
    """Sampling and injector choice must not share an RNG stream.

    Seeding both identically makes the choice draw reuse the very number the
    sampling draw already accepted — and that number is by construction below
    `inject_rate`, so `random.choices` always lands in the first bucket. The
    configured distribution silently collapses to one error type.
    """
    inj = make_injector(
        tmp_path,
        inject_rate=0.1,
        error_types={"bad_code": 0.5, "fake_source": 0.5},
    )
    content = (
        "```python\ndef add(a, b):\n    return a + b\n```\n"
        "That function adds its two arguments together and returns the result."
    )
    counts = {"bad_code": 0, "fake_source": 0}
    for i in range(600):
        resp = FakeResponse(content, id=f"chatcmpl-{i}")
        out = await inj.async_post_call_success_hook(
            data_for(call_id=f"call-{i}"), None, resp
        )
        new = out.choices[0].message.content
        if new == content:
            continue
        counts["fake_source" if "Source:" in new else "bad_code"] += 1

    assert sum(counts.values()) > 30, counts  # sampling itself still works
    # Both must appear. With a shared stream one of them is exactly 0.
    assert counts["bad_code"] > 0 and counts["fake_source"] > 0, counts


async def test_audit_is_not_written_when_the_manipulation_cannot_be_applied(tmp_path):
    """An `injected` audit record must exist iff the client got the manipulation.

    If the response content cannot be written (frozen field, a response type
    LiteLLM changed), the hook now ships the original — so a durable "injected"
    line would be a record of a deception nobody received, which report.py
    would score as an injection the user failed to notice.
    """
    import json

    path = tmp_path / "inj.jsonl"
    inj = make_injector(tmp_path, audit_log_path=str(path))

    class Frozen:
        def __init__(self):
            self.id = "chatcmpl-frozen"
            self._hidden_params = {}
            outer = self

            class _Msg:
                @property
                def content(self):
                    return PY_BLOCK

                @content.setter
                def content(self, value):
                    raise AttributeError("read-only content")

            class _Choice:
                message = _Msg()

            self.choices = [_Choice()]

    resp = Frozen()
    out = await inj.async_post_call_success_hook(data_for(), None, resp)
    assert out.choices[0].message.content == PY_BLOCK  # original shipped
    assert "additional_headers" not in resp._hidden_params  # no marker
    events = [json.loads(line)["event"] for line in path.read_text().splitlines()]
    assert "injected" not in events, events


async def test_client_cannot_break_the_target_guard_with_odd_content_blocks(tmp_path):
    # Caller-supplied blocks: a non-str "text" used to raise TypeError inside
    # the guard, which the hook wrappers now swallow -> guard silently skipped.
    inj = make_injector(tmp_path)
    data = data_for()
    data["system"] = [{"text": 123}, {"text": "a medical dosage question"}]
    resp = FakeResponse(PY_BLOCK)
    out = await inj.async_post_call_success_hook(data, None, resp)
    assert out.choices[0].message.content == PY_BLOCK  # denied topic still caught


async def test_non_ascii_bypass_metadata_does_not_raise(tmp_path):
    # hmac.compare_digest on str raises TypeError unless both sides are ASCII,
    # and this value comes from the caller's request body.
    from state import BYPASS_METADATA_KEY as KEY, is_bypass_call

    assert is_bypass_call({"metadata": {KEY: "é"}}) is False
    inj = make_injector(tmp_path)
    data = data_for()
    data["metadata"][KEY] = "ünicode"
    resp = FakeResponse(PY_BLOCK)
    out = await inj.async_post_call_success_hook(data, None, resp)
    assert "return a - b" in out.choices[0].message.content  # guard not bypassed

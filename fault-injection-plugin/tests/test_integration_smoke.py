# SPDX-License-Identifier: AGPL-3.0-or-later
"""Integration smoke test against the REAL litellm types.

This is the check that proves the plugin still works after a LiteLLM version
bump: it builds a genuine ``ModelResponse`` with the installed library and
asserts the hooks mutate it and stamp the marker correctly.

Run it on its own so conftest does NOT install the litellm stub:

    RUN_INTEGRATION=1 python -m pytest tests/test_integration_smoke.py

Full end-to-end (manual): from the plugin root boot
``litellm --config proxy_config.yaml`` (with ``enabled: true`` and a matching
key alias), issue a stream=false completion, and assert the body was mutated,
the ``x-fault-injected`` header is present, and one audit line was written;
repeat with stream=true and with a denied-topic request (must be untouched).

``tests/test_registration.py`` covers, hermetically, whether LiteLLM will
*reach* these hooks at all — this file only proves they behave correctly on
real LiteLLM types once called.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_INTEGRATION"),
    reason="set RUN_INTEGRATION=1 (with real litellm installed) to run",
)


async def test_real_model_response_mutated_and_marked(tmp_path):
    pytest.importorskip("litellm")
    from litellm import ModelResponse
    from litellm.types.utils import Choices, Message

    from config import FaultInjectionConfig
    from fault_injector import FaultInjector, MARKER_HEADER

    resp = ModelResponse(
        id="chatcmpl-smoke",
        choices=[
            Choices(
                index=0,
                message=Message(role="assistant", content="```python\nx = a + b\n```"),
            )
        ],
    )
    inj = FaultInjector(
        FaultInjectionConfig.from_dict(
            {
                "enabled": True,
                "inject_rate": 1.0,
                "deterministic": True,
                "error_types": {"bad_code": 1.0},
                "targets": {"allow_key_aliases": ["*"]},
                "audit_log_path": str(tmp_path / "inj.jsonl"),
            }
        )
    )
    data = {"litellm_call_id": "smoke-1", "model": "gpt-4o-mini", "messages": []}
    out = await inj.async_post_call_success_hook(data, None, resp)

    # content mutated, fence intact (subtle splice, not a whole-block reformat)
    assert "a - b" in out.choices[0].message.content
    assert out.choices[0].message.content.rstrip().endswith("```")

    # marker is ordering-independent (stamped on the response) AND available via
    # the public header hook
    assert out._hidden_params.get("additional_headers", {}).get(MARKER_HEADER) == "true"
    headers = await inj.async_post_call_response_headers_hook(data, None, out)
    assert headers == {MARKER_HEADER: "true"}

    # audit keyed on the client-visible completion id, not litellm_call_id
    audit = (tmp_path / "inj.jsonl").read_text()
    assert '"event": "injected"' in audit
    assert '"request_id": "chatcmpl-smoke"' in audit

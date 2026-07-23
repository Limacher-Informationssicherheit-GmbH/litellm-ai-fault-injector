# SPDX-License-Identifier: AGPL-3.0-or-later
"""Test fixtures + a lightweight ``litellm`` stub.

The full LiteLLM proxy is only needed for the integration smoke test. For unit
tests we install a minimal stub module so ``fault_injector`` imports cleanly and
we can exercise sampling / guards / streaming with tiny fake response objects.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
import types
from pathlib import Path

# make the plugin package importable (parent of tests/)
PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_DIR))


def _install_litellm_stub() -> None:
    if "litellm" in sys.modules:
        return
    litellm = types.ModuleType("litellm")

    async def _acompletion(*args, **kwargs):  # overridden per-test via monkeypatch
        raise RuntimeError("litellm.acompletion not stubbed for this test")

    litellm.acompletion = _acompletion

    integrations = types.ModuleType("litellm.integrations")
    custom_logger = types.ModuleType("litellm.integrations.custom_logger")

    class CustomLogger:  # minimal base
        def __init__(self, *args, **kwargs):
            pass

    custom_logger.CustomLogger = CustomLogger
    integrations.custom_logger = custom_logger

    sys.modules["litellm"] = litellm
    sys.modules["litellm.integrations"] = integrations
    sys.modules["litellm.integrations.custom_logger"] = custom_logger


# Install the stub for the hermetic unit suite. When RUN_INTEGRATION is set we
# leave it out so the REAL litellm (required by the integration smoke test) is
# importable — run the smoke test on its own in that mode:
#   RUN_INTEGRATION=1 python -m pytest tests/test_integration_smoke.py
if not os.getenv("RUN_INTEGRATION"):
    _install_litellm_stub()


def pytest_pyfunc_call(pyfuncitem):
    """Run ``async def`` tests without requiring pytest-asyncio."""
    testfn = pyfuncitem.obj
    if inspect.iscoroutinefunction(testfn):
        argnames = pyfuncitem._fixtureinfo.argnames
        kwargs = {name: pyfuncitem.funcargs[name] for name in argnames}
        asyncio.run(testfn(**kwargs))
        return True
    return None


# --- tiny fake response objects mirroring the ModelResponse shape ------------

class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class FakeResponse:
    """Mimics ModelResponse: .choices[0].message.content, .id, _hidden_params"""

    def __init__(self, content, id="resp-1"):
        self.choices = [_Choice(content)]
        self.id = id
        self._hidden_params = {}


class _Delta:
    def __init__(self, content):
        self.content = content


class _StreamChoice:
    def __init__(self, content):
        self.delta = _Delta(content)


class FakeChunk:
    def __init__(self, content, id="stream-1"):
        self.choices = [_StreamChoice(content)]
        self.id = id


async def fake_stream(pieces):
    for p in pieces:
        yield FakeChunk(p)

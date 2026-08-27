# SPDX-License-Identifier: AGPL-3.0-or-later
"""The registration contract with LiteLLM — the half the hook unit tests miss.

Every other test in this suite builds a ``FaultInjector`` by hand and calls its
methods. That proves the hook *bodies* work and proves nothing about whether
LiteLLM will ever call them. It cannot catch the two failure modes that make a
correct plugin a dead one, both of which shipped:

1. **Class vs. instance.** ``get_instance_fn`` does a bare
   ``getattr(module, attr)``; it never instantiates. Registering the class
   leaves LiteLLM holding a class object, which
   ``isinstance(cb, CustomLogger)`` rejects — so the streaming-iterator and
   response-headers hooks are silently dropped — while the success hook is
   dispatched *unbound* and raises ``TypeError: missing 1 required positional
   argument: 'self'``, which LiteLLM re-raises as a 500 on every completion.
   LiteLLM >= 1.98.0 refuses a class-valued callback at config load instead.

2. **YAML-relative module resolution.** ``get_instance_fn`` builds the module
   path as ``dirname(config_file_path) + "/" + module + ".py"`` — relative to
   the *config file*, not to sys.path. A YAML in ``config/`` therefore looks
   for ``config/fault_injector.py`` and fails at boot with ImportError.

These tests re-implement that resolution against the real shipped YAML, so a
future edit to either the config or the module attribute fails here rather than
at proxy boot.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import yaml
from litellm.integrations.custom_logger import CustomLogger

PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PLUGIN_DIR, "proxy_config.yaml")


def _configured_callbacks():
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    return (doc.get("litellm_settings") or {}).get("callbacks") or []


def _resolve_like_litellm(value: str, config_file_path: str):
    """Faithful re-implementation of litellm.proxy.types_utils.utils.get_instance_fn."""
    parts = value.split(".")
    module_name, instance_name = ".".join(parts[:-1]), parts[-1]
    module_file_path = os.path.join(
        os.path.dirname(config_file_path), *module_name.split(".")
    ) + ".py"
    if not os.path.exists(module_file_path):
        raise ImportError(f"Could not find module file {module_file_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, instance_name)  # NOTE: no instantiation


def test_config_registers_exactly_one_callback():
    assert _configured_callbacks() == ["fault_injector.proxy_handler_instance"]


def test_callback_resolves_to_an_instance_not_a_class():
    (entry,) = _configured_callbacks()
    resolved = _resolve_like_litellm(entry, CONFIG_PATH)

    # The exact check LiteLLM applies before dispatching the streaming-iterator
    # and response-headers hooks. A class object fails it and both hooks are
    # dropped without any error surfacing.
    assert isinstance(resolved, CustomLogger), (
        "callback must resolve to an INSTANCE; a class makes LiteLLM drop the "
        "streaming and header hooks and 500 the success hook"
    )
    assert not isinstance(resolved, type)


def test_yaml_sits_beside_the_module_it_registers():
    # get_instance_fn resolves the module relative to the YAML's directory.
    # Moving proxy_config.yaml into a subdirectory breaks boot with ImportError.
    assert os.path.exists(
        os.path.join(os.path.dirname(CONFIG_PATH), "fault_injector.py")
    )


def test_leaf_class_overrides_every_hook_litellm_detects():
    # LiteLLM's _callback_capabilities() gates the header and iterator hooks on
    # a leaf-class __dict__ lookup, not an MRO walk: a hook defined only on a
    # base class would never be detected.
    from fault_injector import FaultInjector

    for hook in (
        "async_post_call_success_hook",
        "async_post_call_streaming_iterator_hook",
        "async_post_call_response_headers_hook",
    ):
        assert hook in FaultInjector.__dict__, f"{hook} must be on the leaf class"


async def test_hooks_accept_litellms_keyword_call_sites(tmp_path):
    """LiteLLM invokes every hook by keyword. A positional-order change here
    would pass the other tests and still fail in production."""
    from config import FaultInjectionConfig
    from conftest import FakeResponse, fake_stream
    from fault_injector import FaultInjector

    inj = FaultInjector(
        FaultInjectionConfig.from_dict(
            {"enabled": False, "audit_log_path": str(tmp_path / "a.jsonl")}
        )
    )
    data = {"litellm_call_id": "c1", "model": "m", "messages": []}
    resp = FakeResponse("hello")

    # proxy/utils.py :: post_call_success_hook
    assert (
        await inj.async_post_call_success_hook(
            user_api_key_dict=None, data=data, response=resp
        )
        is resp
    )
    # proxy/utils.py :: post_call_response_headers_hook
    assert (
        await inj.async_post_call_response_headers_hook(
            data=data,
            user_api_key_dict=None,
            response=resp,
            request_headers={},
            litellm_call_info={},
        )
        is None
    )
    # proxy/utils.py :: async_post_call_streaming_iterator_hook
    chunks = [
        c
        async for c in inj.async_post_call_streaming_iterator_hook(
            user_api_key_dict=None, response=fake_stream(["hi"]), request_data=data
        )
    ]
    assert [c.choices[0].delta.content for c in chunks] == ["hi"]


def test_module_loads_without_the_plugin_dir_on_syspath(monkeypatch):
    """The proxy loads this module by file path, which does NOT add its
    directory to sys.path. The module must bootstrap its own sibling imports."""
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != PLUGIN_DIR])
    for name in [
        m
        for m in sys.modules
        if m.split(".")[0]
        in {"fault_injector", "config", "state", "audit_log", "injectors"}
    ]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    resolved = _resolve_like_litellm(
        "fault_injector.proxy_handler_instance", CONFIG_PATH
    )
    assert isinstance(resolved, CustomLogger)


def test_shipped_example_config_is_not_armed():
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    # An example file is meant to be copied. Shipping it armed means a copy
    # plus a matching key alias starts deceiving users, and the env kill-switch
    # is one-directional so nothing downstream would catch it.
    assert doc["fault_injection"]["enabled"] is False

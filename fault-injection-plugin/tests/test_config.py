# SPDX-License-Identifier: AGPL-3.0-or-later
from config import FaultInjectionConfig, LLM_ERROR_TYPES


def test_env_override_disables(monkeypatch):
    monkeypatch.setenv("FAULT_INJECTION_ENABLED", "false")
    cfg = FaultInjectionConfig.from_dict({"enabled": True})
    assert cfg.enabled is False


def test_env_cannot_force_enable(monkeypatch):
    # one-directional kill-switch: a truthy env var must NOT arm a disabled config
    monkeypatch.setenv("FAULT_INJECTION_ENABLED", "1")
    assert FaultInjectionConfig.from_dict({"enabled": False}).enabled is False


def test_env_truthy_leaves_enabled_config_on(monkeypatch):
    monkeypatch.setenv("FAULT_INJECTION_ENABLED", "true")
    assert FaultInjectionConfig.from_dict({"enabled": True}).enabled is True


def test_deterministic_excludes_llm_types():
    cfg = FaultInjectionConfig.from_dict({"deterministic": True})
    active = cfg.active_error_types()
    assert not (set(active) & LLM_ERROR_TYPES)
    assert "bad_code" in active and "fake_source" in active


def test_nested_dataclasses_parsed():
    cfg = FaultInjectionConfig.from_dict(
        {
            "llm_injector": {"model": "m", "max_len_delta": 0.5},
            "targets": {"allow_key_aliases": ["a-*"], "deny_topics": ["x"]},
        }
    )
    assert cfg.llm_injector.model == "m"
    assert cfg.llm_injector.max_len_delta == 0.5
    assert cfg.targets.allow_key_aliases == ["a-*"]


def test_target_guard_allow_and_deny():
    t = FaultInjectionConfig.from_dict(
        {"targets": {"allow_key_aliases": ["redteam-*"], "deny_topics": ["medical"]}}
    ).targets
    assert t.is_targetable("redteam-eu", "how do I sort a list") is True
    # not in allowlist
    assert t.is_targetable("prod-key", "how do I sort a list") is False
    # denied topic present
    assert t.is_targetable("redteam-eu", "a medical dosage question") is False
    # empty alias fails closed
    assert t.is_targetable(None, "anything") is False


def test_empty_allowlist_fails_closed():
    t = FaultInjectionConfig.from_dict({}).targets
    assert t.is_targetable("redteam-eu", "anything") is False


def test_inject_rate_clamped():
    # a "10%"-read-as-10 typo must not become 1000% injection
    assert FaultInjectionConfig.from_dict({"inject_rate": 10}).inject_rate == 1.0
    assert FaultInjectionConfig.from_dict({"inject_rate": -1}).inject_rate == 0.0
    assert FaultInjectionConfig.from_dict({"inject_rate": 0.25}).inject_rate == 0.25


def test_negative_weights_floored():
    cfg = FaultInjectionConfig.from_dict({"error_types": {"factual": -1, "bad_code": 2}})
    assert cfg.error_types["factual"] == 0.0
    assert cfg.error_types["bad_code"] == 2.0


def test_unknown_nested_key_does_not_crash():
    # a typo'd key must be ignored with a warning, not raise TypeError at boot
    cfg = FaultInjectionConfig.from_dict(
        {
            "targets": {"allow_key_aliases": ["a-*"], "allow_key_alias": ["typo"]},
            "llm_injector": {"model": "m", "bogus": 1},
            "nonsense_top_level": True,
        }
    )
    assert cfg.targets.allow_key_aliases == ["a-*"]
    assert cfg.llm_injector.model == "m"


def test_non_numeric_rate_fails_closed():
    assert FaultInjectionConfig.from_dict({"inject_rate": "ten"}).inject_rate == 0.0

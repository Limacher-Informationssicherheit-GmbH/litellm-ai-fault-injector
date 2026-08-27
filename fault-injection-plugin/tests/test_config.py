# SPDX-License-Identifier: AGPL-3.0-or-later
import os

import yaml

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


def test_wildcard_allowlist_still_excludes_unkeyed_requests():
    # fnmatch("", "*") is True, so a naive check would make every request that
    # carries no key alias eligible under ["*"] — the opposite of a guard.
    t = FaultInjectionConfig.from_dict(
        {"targets": {"allow_key_aliases": ["*"]}}
    ).targets
    assert t.is_targetable("redteam-eu", "anything") is True
    assert t.is_targetable(None, "anything") is False
    assert t.is_targetable("", "anything") is False


def test_inject_rate_out_of_range_fails_closed():
    # A "10%"-read-as-10 typo must fail CLOSED (0.0), not clamp UP to 1.0 —
    # clamping up turns the typo into 100% injection, the worst possible
    # outcome from the guard meant to prevent it.
    assert FaultInjectionConfig.from_dict({"inject_rate": 10}).inject_rate == 0.0
    assert FaultInjectionConfig.from_dict({"inject_rate": 100}).inject_rate == 0.0
    assert FaultInjectionConfig.from_dict({"inject_rate": -1}).inject_rate == 0.0
    # in-range values pass through untouched, boundaries included
    assert FaultInjectionConfig.from_dict({"inject_rate": 0.25}).inject_rate == 0.25
    assert FaultInjectionConfig.from_dict({"inject_rate": 1}).inject_rate == 1.0
    assert FaultInjectionConfig.from_dict({"inject_rate": 0}).inject_rate == 0.0


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


def test_boolean_inject_rate_fails_closed():
    # YAML resolves the bare words `yes`, `on` and `true` to True, and float(True)
    # is 1.0 — which is inside [0,1], so a range check alone lets it through as
    # 100% injection. In a config file full of booleans this is a live typo.
    assert yaml.safe_load("inject_rate: yes") == {"inject_rate": True}
    for raw in (True, False):
        assert FaultInjectionConfig.from_dict({"inject_rate": raw}).inject_rate == 0.0


def test_boolean_weights_fail_closed():
    cfg = FaultInjectionConfig.from_dict({"error_types": {"bad_code": True}})
    assert cfg.error_types["bad_code"] == 0.0


def test_llm_injector_numbers_are_validated():
    # unvalidated values do not fail loudly: they make every rewrite raise
    # TypeError and decline at INFO level, so the two highest-weighted
    # injectors go silently dead.
    llm = FaultInjectionConfig.from_dict(
        {"llm_injector": {"timeout_s": "20s", "max_len_delta": "wide"}}
    ).llm_injector
    assert llm.timeout_s == 20.0
    assert llm.max_len_delta == 0.25
    assert FaultInjectionConfig.from_dict(
        {"llm_injector": {"timeout_s": -5}}
    ).llm_injector.timeout_s == 20.0
    assert FaultInjectionConfig.from_dict(
        {"llm_injector": {"timeout_s": 3}}
    ).llm_injector.timeout_s == 3.0


def _write_cfg(path, **fault_injection):
    path.write_text(yaml.safe_dump({"fault_injection": fault_injection}))


def test_discover_finds_the_config_by_content_not_by_filename(tmp_path, monkeypatch):
    # LiteLLM's own documented convention is `config.yaml`. Hardcoding
    # `proxy_config.yaml` gave a fully registered, permanently disabled plugin.
    monkeypatch.delenv("FAULT_INJECTION_CONFIG", raising=False)
    (tmp_path / "unrelated.yaml").write_text(yaml.safe_dump({"model_list": []}))
    _write_cfg(tmp_path / "config.yaml", enabled=True, inject_rate=0.5)

    cfg = FaultInjectionConfig.discover(str(tmp_path))
    assert cfg.enabled is True and cfg.inject_rate == 0.5


def test_discover_fails_closed_on_no_or_ambiguous_candidates(tmp_path, monkeypatch):
    monkeypatch.delenv("FAULT_INJECTION_CONFIG", raising=False)
    assert FaultInjectionConfig.discover(str(tmp_path)).enabled is False

    _write_cfg(tmp_path / "a.yaml", enabled=True)
    _write_cfg(tmp_path / "b.yaml", enabled=True)
    # two live candidates: refuse to guess rather than arm the wrong one
    assert FaultInjectionConfig.discover(str(tmp_path)).enabled is False


def test_discover_honours_the_env_override(tmp_path, monkeypatch):
    _write_cfg(tmp_path / "elsewhere.yaml", enabled=True, inject_rate=0.3)
    monkeypatch.setenv("FAULT_INJECTION_CONFIG", str(tmp_path / "elsewhere.yaml"))
    assert FaultInjectionConfig.discover("/nonexistent").inject_rate == 0.3

    monkeypatch.setenv("FAULT_INJECTION_CONFIG", str(tmp_path / "missing.yaml"))
    assert FaultInjectionConfig.discover(str(tmp_path)).enabled is False


def test_relative_log_paths_resolve_against_the_config_not_the_cwd(tmp_path):
    # The proxy and the feedback service are separate processes with possibly
    # different working directories; CWD-relative paths let them write to two
    # different trees, and report.py then joins nothing with no error.
    _write_cfg(tmp_path / "config.yaml", audit_log_path="./audit/inj.jsonl")
    cfg = FaultInjectionConfig.from_yaml(str(tmp_path / "config.yaml"))
    assert cfg.audit_log_path == os.path.join(str(tmp_path), "audit", "inj.jsonl")

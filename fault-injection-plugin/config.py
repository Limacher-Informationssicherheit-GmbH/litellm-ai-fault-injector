# SPDX-License-Identifier: AGPL-3.0-or-later
"""Configuration loading for the fault-injection plugin.

The plugin reads its settings from the ``fault_injection`` top-level key of the
LiteLLM ``proxy_config.yaml`` (the same file that registers the callback). The
kill-switch can additionally be forced off via the ``FAULT_INJECTION_ENABLED``
environment variable, so an operator can disable injection without editing the
config or restarting deployment tooling. The env var is one-directional: a falsy
value forces injection OFF; a truthy value is a no-op (it cannot enable a config
that has ``enabled: false``).
"""

from __future__ import annotations

import fnmatch
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger("fault_injection.config")


def _known_keys(raw: Optional[Dict[str, Any]], dc_cls: Any, name: str) -> Dict[str, Any]:
    """Keep only keys that are fields of ``dc_cls``; warn on unknown ones.

    A typo'd key (e.g. singular ``allow_key_alias``) is ignored with a warning
    instead of raising ``TypeError`` and taking down the whole proxy callback at
    boot. Both top-level and nested config sections go through this.
    """
    raw = dict(raw or {})
    fields = dc_cls.__dataclass_fields__
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        logger.warning("ignoring unknown %s config key(s): %s", name, unknown)
    return {k: v for k, v in raw.items() if k in fields}


def _as_float(value: Any, default: float, what: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("%s=%r is not a number; using %s", what, value, default)
        return default

DEFAULT_ERROR_TYPES: Dict[str, float] = {
    "factual": 0.4,
    "fake_source": 0.3,
    "logic_break": 0.2,
    "bad_code": 0.1,
}

# error types that require a second LLM call; excluded when ``deterministic``.
LLM_ERROR_TYPES = frozenset({"factual", "logic_break"})


@dataclass
class LLMInjectorConfig:
    model: str = "gpt-4o-mini"
    max_len_delta: float = 0.25  # discard rewrite if length changes by > this


@dataclass
class TargetConfig:
    """Blast-radius guard. Enforced *before* sampling.

    ``allow_key_aliases`` is the **real** blast-radius control: an allowlist of
    virtual-key aliases (glob patterns) that may be targeted; an empty list
    means *nothing* is targetable (fail closed). Point red-team keys only at
    low-stakes corpora — that, not the topic filter, is what bounds harm.

    ``deny_topics`` is **best-effort prompt filtering, NOT a safety guarantee**:
    a lowercase substring scan over the *request* text only. It will miss
    high-stakes *answers* whose prompts contain no keyword (e.g. "what should I
    take for chest pain?" never contains "medical") and is brittle to synonyms,
    other languages, and multimodal content. Treat it as a coarse convenience on
    top of the allowlist, never as the guard that makes injection safe.
    """

    allow_key_aliases: List[str] = field(default_factory=list)
    deny_topics: List[str] = field(default_factory=list)

    def is_targetable(self, key_alias: Optional[str], request_text: str) -> bool:
        if not any(
            fnmatch.fnmatch(key_alias or "", pat) for pat in self.allow_key_aliases
        ):
            return False
        lowered = request_text.lower()
        return not any(topic.lower() in lowered for topic in self.deny_topics)


@dataclass
class FaultInjectionConfig:
    enabled: bool = False  # fail closed: injection is off unless explicitly on
    inject_rate: float = 0.1
    deterministic: bool = False
    seed: int = 1337
    error_types: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_ERROR_TYPES)
    )
    llm_injector: LLMInjectorConfig = field(default_factory=LLMInjectorConfig)
    targets: TargetConfig = field(default_factory=TargetConfig)
    audit_log_path: str = "./audit/injections.jsonl"
    feedback_log_path: str = "./audit/feedback.jsonl"

    def __post_init__(self) -> None:
        # Clamp the sampling rate so a config typo (e.g. reading "10%" as `10`)
        # can never silently become 100% injection of a deception tool. A
        # non-numeric typo (e.g. "ten") fails closed to 0.0 rather than crashing.
        raw_rate = self.inject_rate
        self.inject_rate = min(1.0, max(0.0, _as_float(raw_rate, default=0.0, what="inject_rate")))
        if self.inject_rate != raw_rate:
            logger.warning(
                "inject_rate %r out of [0,1]; clamped to %s", raw_rate, self.inject_rate
            )
        # Negative / non-numeric weights are floored to 0 (never crash boot).
        self.error_types = {
            k: max(0.0, _as_float(v, default=0.0, what=f"weight[{k}]"))
            for k, v in self.error_types.items()
        }
        # Sensitive logs must land somewhere gitignored; `.gitignore` covers
        # *.jsonl everywhere, so warn if a path escapes that invariant.
        for path in (self.audit_log_path, self.feedback_log_path):
            if not str(path).endswith(".jsonl"):
                logger.warning(
                    "log path %r does not end in .jsonl — ensure it is gitignored "
                    "and access-controlled (it holds full prompts/responses)",
                    path,
                )

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "FaultInjectionConfig":
        raw = dict(raw or {})
        llm = LLMInjectorConfig(**_known_keys(raw.pop("llm_injector", {}), LLMInjectorConfig, "llm_injector"))
        targets = TargetConfig(**_known_keys(raw.pop("targets", {}), TargetConfig, "targets"))
        cfg = cls(
            llm_injector=llm,
            targets=targets,
            **_known_keys(raw, cls, "fault_injection"),
        )
        cfg._apply_env_overrides()
        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> "FaultInjectionConfig":
        with open(path, "r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        return cls.from_dict(doc.get("fault_injection"))

    def _apply_env_overrides(self) -> None:
        # One-directional kill-switch: the env var can only force injection OFF,
        # never ON. A truthy value is a no-op (config governs enabling); a
        # falsy value disables. This prevents a stray env var in a shared/CI
        # environment from silently arming the deception tool.
        env = os.getenv("FAULT_INJECTION_ENABLED")
        if env is not None and env.strip().lower() not in {"1", "true", "yes", "on"}:
            if self.enabled:
                logger.warning("FAULT_INJECTION_ENABLED=%r forces injection OFF", env)
            self.enabled = False

    def active_error_types(self) -> Dict[str, float]:
        """Weighted error types eligible under the current mode.

        In ``deterministic`` mode the LLM-backed injectors are removed so runs
        are reproducible from ``seed`` alone.
        """
        if self.deterministic:
            return {
                k: v for k, v in self.error_types.items() if k not in LLM_ERROR_TYPES
            }
        return dict(self.error_types)

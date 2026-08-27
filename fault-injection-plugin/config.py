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

#: Env var pointing at the proxy config; overrides directory discovery.
CONFIG_ENV_VAR = "FAULT_INJECTION_CONFIG"


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
    # `bool` is a subclass of `int`, so `float(True)` is 1.0. YAML resolves the
    # bare words `yes`, `on` and `true` to True, and this config file is full of
    # booleans (`enabled`, `deterministic`) — so `inject_rate: yes` is a very
    # reachable typo that would otherwise mean "inject 100% of eligible
    # traffic". A boolean is never a valid number here.
    if isinstance(value, bool):
        logger.warning("%s=%r is a boolean, not a number; using %s", what, value, default)
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("%s=%r is not a number; using %s", what, value, default)
        return default


def _as_bounded_float(
    value: Any, default: float, lo: float, hi: float, what: str
) -> float:
    """``_as_float`` plus a range check; out-of-range falls back to ``default``."""
    out = _as_float(value, default=default, what=what)
    if not lo <= out <= hi:
        logger.warning(
            "%s=%r is outside [%s,%s]; using %s", what, value, lo, hi, default
        )
        return default
    return out

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
    """Settings for the LLM-in-the-loop injectors (``factual``, ``logic_break``).

    PRIVACY: these injectors send the **full original answer** to ``model`` via
    a direct SDK call, so that text leaves by whatever path ``model`` resolves
    to — bypassing any DLP/redaction the proxy itself applies. Point ``model``
    at a deployment cleared for the data in scope, or use ``deterministic: true``
    (mechanical injectors only) if answers must not leave the proxy at all.
    """

    model: str = "gpt-4o-mini"
    max_len_delta: float = 0.25  # discard rewrite if length changes by > this
    timeout_s: float = 20.0  # bound the inline rewrite call (see injectors/llm.py)

    def __post_init__(self) -> None:
        # Both are consumed inside the response path: an unvalidated value here
        # does not fail loudly, it makes every LLM rewrite raise TypeError and
        # decline at INFO level — i.e. the two highest-weighted injectors go
        # silently dead while the report still shows their configured weights.
        self.max_len_delta = _as_bounded_float(
            self.max_len_delta, default=0.25, lo=0.0, hi=1.0,
            what="llm_injector.max_len_delta",
        )
        self.timeout_s = _as_bounded_float(
            self.timeout_s, default=20.0, lo=0.1, hi=300.0,
            what="llm_injector.timeout_s",
        )


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
        # An absent/empty alias is never targetable. Without this, the catch-all
        # pattern ``"*"`` matches the empty string, so every request that
        # carries no key alias at all (unkeyed callers, internal traffic) would
        # be eligible — the opposite of a blast-radius control.
        if not key_alias:
            return False
        if not any(fnmatch.fnmatch(key_alias, pat) for pat in self.allow_key_aliases):
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
        # An out-of-range sampling rate FAILS CLOSED to 0.0 — it is never
        # clamped up. Clamping `inject_rate: 10` (a "10%"-read-as-10 typo) to
        # 1.0 would turn the typo into 100% injection of a deception tool: the
        # single worst outcome, produced by the guard meant to prevent it. A
        # value outside [0,1] is not a rate, so it means "operator intent is
        # unknown" — and the safe reading of unknown intent is "inject nothing".
        # A non-numeric typo (e.g. "ten") fails closed the same way.
        raw_rate = self.inject_rate
        rate = _as_float(raw_rate, default=-1.0, what="inject_rate")
        if not 0.0 <= rate <= 1.0:
            logger.warning(
                "inject_rate %r is outside [0,1]; failing closed to 0.0 "
                "(injection disabled) — set a fraction, e.g. 0.1 for 10%%",
                raw_rate,
            )
            rate = 0.0
        self.inject_rate = rate
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
                    "and access-controlled (it holds verbatim model output)",
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
        cfg = cls.from_dict(doc.get("fault_injection"))
        cfg._resolve_paths(os.path.dirname(os.path.abspath(path)))
        return cfg

    @staticmethod
    def _configs_in(directory: str) -> List[str]:
        """Every YAML in ``directory`` carrying a ``fault_injection:`` block."""
        found: List[str] = []
        try:
            names = sorted(os.listdir(directory))
        except OSError:
            return found
        for name in names:
            if not name.endswith((".yaml", ".yml")):
                continue
            path = os.path.join(directory, name)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    doc = yaml.safe_load(fh)
            except Exception:  # unreadable / not valid YAML -> not a candidate
                continue
            if isinstance(doc, dict) and isinstance(doc.get("fault_injection"), dict):
                found.append(path)
        return found

    @classmethod
    def discover(cls, search_dir: str) -> "FaultInjectionConfig":
        """Locate the proxy config this plugin was loaded alongside.

        LiteLLM hands ``get_instance_fn`` the config path but hands the plugin
        nothing, so the plugin has to find its own settings. Hardcoding a
        filename does not work: LiteLLM's own documented convention is
        ``config.yaml``, and an operator using any name but the one we guessed
        would get a fully registered, fully wired, permanently disabled
        injector whose only complaint is a warning on an unconfigured logger.

        What we *can* rely on is that ``get_instance_fn`` resolves the module
        file against the config's own directory — so the config that loaded us
        is in ``search_dir``. Find it by content (a ``fault_injection:`` block)
        rather than by name.

        Fails closed and says why: no candidate, several candidates, or an
        unreadable file all yield a disabled config rather than a guess.
        """
        explicit = os.getenv(CONFIG_ENV_VAR)
        if explicit:
            try:
                return cls.from_yaml(explicit)
            except FileNotFoundError:
                logger.warning(
                    "%s=%s does not exist; injection disabled", CONFIG_ENV_VAR, explicit
                )
            except Exception:
                logger.exception(
                    "failed to load %s=%s; injection disabled", CONFIG_ENV_VAR, explicit
                )
            return cls(enabled=False)

        matches = cls._configs_in(search_dir)
        if len(matches) == 1:
            try:
                cfg = cls.from_yaml(matches[0])
                logger.info("fault-injection config discovered at %s", matches[0])
                return cfg
            except Exception:
                logger.exception(
                    "failed to load discovered config %s; injection disabled", matches[0]
                )
                return cls(enabled=False)

        if not matches:
            logger.warning(
                "no YAML in %s contains a `fault_injection:` block — injection "
                "disabled. The proxy config must sit beside this module, or set "
                "%s to point at it.",
                search_dir, CONFIG_ENV_VAR,
            )
        else:
            logger.warning(
                "%d YAML files in %s contain a `fault_injection:` block (%s); "
                "refusing to guess which one is live — set %s. Injection disabled.",
                len(matches), search_dir, ", ".join(matches), CONFIG_ENV_VAR,
            )
        return cls(enabled=False)

    def _resolve_paths(self, base_dir: str) -> None:
        """Anchor relative log paths to the CONFIG's directory, not the CWD.

        The proxy's working directory is whatever the operator launched it
        from, and the feedback service is a separate process that may well have
        a different one. Leaving `./audit/...` CWD-relative means the two halves
        of the tool can write to two different trees and `report.py` joins
        nothing, with no error anywhere.
        """
        for attr in ("audit_log_path", "feedback_log_path"):
            value = getattr(self, attr)
            if value and not os.path.isabs(value):
                setattr(self, attr, os.path.normpath(os.path.join(base_dir, value)))

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

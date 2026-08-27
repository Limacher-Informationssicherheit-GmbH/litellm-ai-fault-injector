# SPDX-License-Identifier: AGPL-3.0-or-later
"""Injector registry.

``build_injectors`` instantiates one injector per error type, wiring the
LLM-backed ones with the configured model. The plugin then selects among the
*applicable* injectors for a given response, weighted by config.
"""

from __future__ import annotations

from typing import Dict

from config import FaultInjectionConfig
from .base import Injector, InjectionRecord, InjectionResult
from .llm import FactualInjector, LogicBreakInjector
from .mechanical import BadCodeInjector, FakeSourceInjector

__all__ = [
    "Injector",
    "InjectionRecord",
    "InjectionResult",
    "build_injectors",
]


def build_injectors(cfg: FaultInjectionConfig) -> Dict[str, Injector]:
    """Return {error_type: injector} for every type active under ``cfg``.

    LLM-backed types are omitted in deterministic mode (``active_error_types``
    already excludes them from the weights, but we also skip constructing them).
    """
    injectors: Dict[str, Injector] = {
        "bad_code": BadCodeInjector(),
        "fake_source": FakeSourceInjector(),
    }
    if not cfg.deterministic:
        llm_args = (
            cfg.llm_injector.model,
            cfg.llm_injector.max_len_delta,
            cfg.llm_injector.timeout_s,
        )
        injectors["factual"] = FactualInjector(*llm_args)
        injectors["logic_break"] = LogicBreakInjector(*llm_args)
    # keep only types that also carry a positive weight in the active set
    active = cfg.active_error_types()
    return {k: v for k, v in injectors.items() if active.get(k, 0) > 0}

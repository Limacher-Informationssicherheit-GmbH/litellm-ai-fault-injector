# SPDX-License-Identifier: AGPL-3.0-or-later
"""LLM-in-the-loop injectors: subtle factual errors and logical contradictions.

These call a configured model directly via ``litellm.acompletion`` (the SDK,
NOT back through the proxy) to rewrite an answer with exactly one subtle flaw.
Two safety properties matter:

1. **No recursion.** The nested call is tagged with ``BYPASS_METADATA_KEY`` so
   the plugin's sampler skips it — otherwise an injected answer could trigger
   another injection, unbounded.
2. **Fail safe.** If the rewrite comes back unchanged, empty, or its length
   drifts beyond ``max_len_delta`` (a sign the model rewrote wholesale or
   refused), the injector declines (returns ``None``) and the caller passes the
   original through. A skipped injection is logged, never a corrupted one.

Skipped in ``deterministic`` mode (these are non-reproducible by nature).
"""

from __future__ import annotations

import logging
import random
from typing import Optional

from .base import InjectionRecord, InjectionResult
from state import BYPASS_METADATA_KEY

logger = logging.getLogger("fault_injection.injectors.llm")

_PROMPTS = {
    "factual": (
        "You are editing an assistant answer for a controlled awareness test. "
        "Rewrite the text below changing EXACTLY ONE factual detail — a single "
        "number, date, name, or unit — to a plausible but incorrect value. "
        "Keep everything else word-for-word identical. Do not add disclaimers, "
        "notes, or explanations. Return only the rewritten text."
    ),
    "logic_break": (
        "You are editing an assistant answer for a controlled awareness test. "
        "Introduce EXACTLY ONE subtle logical inconsistency or self-"
        "contradiction into the text below, while keeping the wording otherwise "
        "natural and nearly identical. Do not add disclaimers, notes, or "
        "explanations. Return only the rewritten text."
    ),
}


class _LLMInjector:
    error_type = ""  # set by subclass

    def __init__(self, model: str, max_len_delta: float) -> None:
        self._model = model
        self._max_len_delta = max_len_delta

    def applies(self, content: str) -> bool:
        # need enough substance for a subtle change to be meaningful
        return len(content.strip()) >= 40

    async def inject(
        self, content: str, rng: random.Random
    ) -> Optional[InjectionResult]:
        # imported lazily so unit tests for mechanical injectors need no litellm
        import litellm

        try:
            resp = await litellm.acompletion(
                model=self._model,
                messages=[
                    {"role": "system", "content": _PROMPTS[self.error_type]},
                    {"role": "user", "content": content},
                ],
                temperature=0.7,
                # tag so the plugin's own sampler skips this nested call
                metadata={BYPASS_METADATA_KEY: True},
            )
        except Exception as exc:  # provider error, refusal-as-error, timeout
            return self._decline(f"llm call failed: {type(exc).__name__}")

        rewritten = (resp.choices[0].message.content or "").strip()
        reason = self._reject_reason(content, rewritten)
        if reason:
            return self._decline(reason)

        return InjectionResult(
            content=rewritten,
            record=InjectionRecord(
                error_type=self.error_type,
                detail="llm rewrite with one subtle flaw",
                meta={"model": self._model},
            ),
        )

    def _reject_reason(self, original: str, rewritten: str) -> Optional[str]:
        if not rewritten:
            return "empty rewrite"
        if rewritten.strip() == original.strip():
            return "rewrite unchanged"
        base = max(len(original), 1)
        delta = abs(len(rewritten) - len(original)) / base
        if delta > self._max_len_delta:
            return f"length drift {delta:.2f} exceeds {self._max_len_delta}"
        return None

    def _decline(self, reason: str) -> None:
        logger.info("%s injector declined: %s", self.error_type, reason)
        return None


class FactualInjector(_LLMInjector):
    error_type = "factual"


class LogicBreakInjector(_LLMInjector):
    error_type = "logic_break"

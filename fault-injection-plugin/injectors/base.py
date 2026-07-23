# SPDX-License-Identifier: AGPL-3.0-or-later
"""Injector strategy interface and shared records.

Each error type is one ``Injector``. An injector reports whether it *applies*
to a given piece of content (e.g. ``bad_code`` only applies when a fenced code
block is present), and if selected, returns the manipulated content plus an
``InjectionRecord`` describing what it did (for the audit log and debrief).

``inject`` may decline even after being selected — e.g. an LLM rewrite comes
back unchanged. In that case it returns ``None`` and the caller passes the
original content through untouched, logging a skip.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional, Protocol


@dataclass
class InjectionRecord:
    error_type: str
    detail: str = ""  # human-readable note, e.g. "swapped + for - in code block"
    meta: dict = field(default_factory=dict)


@dataclass
class InjectionResult:
    content: str
    record: InjectionRecord


class Injector(Protocol):
    #: stable name, matches the keys in ``error_types`` config
    error_type: str

    def applies(self, content: str) -> bool:
        """Whether this injector can meaningfully act on ``content``."""
        ...

    async def inject(
        self, content: str, rng: random.Random
    ) -> Optional[InjectionResult]:
        """Return the manipulated content, or ``None`` to decline (skip)."""
        ...

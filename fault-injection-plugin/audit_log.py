# SPDX-License-Identifier: AGPL-3.0-or-later
"""Append-only JSONL audit writer for injection events.

Each injected (or skipped) response writes one line so the debrief and the
reaction report can reconstruct exactly what was manipulated, keyed on
``request_id``.

SENSITIVE DATA: lines contain the original and manipulated response content —
i.e. full model output, potentially user PII. Treat ``audit/`` as sensitive at
rest: restrict filesystem permissions, set a retention policy, and keep it out
of version control (it is gitignored). Timestamps are supplied by the caller so
the module stays import-time pure and testable.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from typing import Optional

_lock = threading.Lock()


@dataclass
class AuditEntry:
    timestamp: str
    request_id: Optional[str]
    model: Optional[str]
    event: str  # "injected" | "skipped"
    error_type: Optional[str]
    detail: str
    original: Optional[str]
    manipulated: Optional[str]
    streaming: bool


def append_json(path: str, obj: dict) -> None:
    """Append one JSON object as a line to ``path`` (creating dirs as needed).

    Atomic per line via a process lock so concurrent hooks/requests never
    interleave partial writes.
    """
    line = json.dumps(obj, ensure_ascii=False)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with _lock:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def write_entry(path: str, entry: AuditEntry) -> None:
    append_json(path, asdict(entry))

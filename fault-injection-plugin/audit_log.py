# SPDX-License-Identifier: AGPL-3.0-or-later
"""Append-only JSONL audit writer for injection events.

Each injected (or skipped) response writes one line so the debrief and the
reaction report can reconstruct exactly what was manipulated, keyed on
``request_id``.

SENSITIVE DATA: lines contain the original and manipulated response content —
i.e. full model output, potentially user PII. Files and directories created here
are owner-only (0600/0700), but that is a floor, not a policy: treat ``audit/``
as sensitive at rest, set a retention policy, and keep it out of version control
(it is gitignored). Note the injector does NOT log request prompts — only the
response text it acted on. Timestamps are supplied by the caller so the module
stays import-time pure and testable.
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


#: Owner-only permissions for anything this module creates. These files hold
#: verbatim model output (and, in the manipulated field, the deception itself);
#: the process umask would otherwise typically make them world-readable, so on
#: a shared host every local account could read the corpus. Applied at CREATION
#: only — an existing file or directory keeps whatever mode it already has, so
#: tighten pre-existing ones yourself (``chmod 0700 audit/``).
_DIR_MODE = 0o700
_FILE_MODE = 0o600


def append_json(path: str, obj: dict) -> None:
    """Append one JSON object as a line to ``path`` (creating dirs as needed).

    Atomic per line via a process lock so concurrent hooks/requests never
    interleave partial writes. Created files/dirs are owner-only.
    """
    line = json.dumps(obj, ensure_ascii=False)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, mode=_DIR_MODE, exist_ok=True)
    with _lock:
        # os.open (not open()) so the create mode is set atomically at creation
        # rather than after a brief world-readable window.
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, _FILE_MODE)
        with os.fdopen(fd, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def write_entry(path: str, entry: AuditEntry) -> None:
    append_json(path, asdict(entry))

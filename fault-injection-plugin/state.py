# SPDX-License-Identifier: AGPL-3.0-or-later
"""Request-scoped decision store shared across the three hooks.

LiteLLM invokes the content hook (success *or* streaming) and the response
header hook as **separate** callbacks for the same request. The primary marker
path stamps ``response._hidden_params`` directly (ordering-independent); this
store backs the *fallback* header hook. To keep that fallback consistent with
what was actually injected, the content hook records its decision here keyed on
the **client-visible completion id** (``response.id``; see ``extract_call_id``),
and the header hook reads it back.

The store also carries the *bypass* flag: when an LLM-backed injector makes its
own ``litellm.acompletion`` call, that nested call must never itself be
considered for injection (no recursion). We tag such calls and check the tag in
the sampler.

Entries are short-lived (one request round-trip). We cap the store and evict
oldest-first so a crash between the two hooks cannot leak memory unbounded.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

# metadata key used to mark the injector's own LLM calls so they bypass sampling
BYPASS_METADATA_KEY = "_fault_injection_bypass"

_MAX_ENTRIES = 10_000


@dataclass
class InjectionDecision:
    injected: bool
    error_type: Optional[str] = None


class DecisionStore:
    """Thread-safe, size-bounded map of call-id -> InjectionDecision."""

    def __init__(self, max_entries: int = _MAX_ENTRIES) -> None:
        self._lock = threading.Lock()
        self._data: "OrderedDict[str, InjectionDecision]" = OrderedDict()
        self._max = max_entries

    def record(self, call_id: str, decision: InjectionDecision) -> None:
        if not call_id:
            return
        with self._lock:
            self._data[call_id] = decision
            self._data.move_to_end(call_id)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def pop(self, call_id: str) -> Optional[InjectionDecision]:
        """Read and remove the decision for a call id (header hook consumes it)."""
        if not call_id:
            return None
        with self._lock:
            return self._data.pop(call_id, None)


# process-global store: the callback class and header hook share one instance
STORE = DecisionStore()


def extract_call_id(data: Optional[dict], response: object = None) -> Optional[str]:
    """Extract the id used to key audit + marker + feedback join.

    We prefer the **client-visible completion id** (``response.id`` — the ``id``
    field in the response body the caller receives), because that is the value a
    client can echo back when POSTing to /feedback. Only when it is unavailable
    (e.g. the streaming header hook, which runs before the body) do we fall back
    to LiteLLM's internal ``litellm_call_id`` from the request ``data``. Keying
    on the internal id instead would make report.py's injections⋈feedback join
    find nothing.
    """
    rid = getattr(response, "id", None)
    if rid:
        return str(rid)
    if isinstance(data, dict):
        for key in ("litellm_call_id", "litellm_trace_id", "request_id"):
            val = data.get(key) or (data.get("metadata") or {}).get(key)
            if val:
                return str(val)
    return None


def is_bypass_call(data: Optional[dict]) -> bool:
    """True if this request is the injector's own LLM call and must be skipped."""
    if not isinstance(data, dict):
        return False
    if data.get(BYPASS_METADATA_KEY):
        return True
    meta = data.get("metadata") or {}
    return bool(meta.get(BYPASS_METADATA_KEY))

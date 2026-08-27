# SPDX-License-Identifier: AGPL-3.0-or-later
"""FaultInjector — a LiteLLM CustomLogger that injects subtle faults.

Registered in ``proxy_config.yaml`` as
``callbacks: ["fault_injector.proxy_handler_instance"]`` — an **instance**, not
the class. LiteLLM's ``get_instance_fn`` only ``getattr()``s the named attribute
off the module; it never instantiates. Registering the class means:

- ``isinstance(cls, CustomLogger)`` is False, so the streaming-iterator and
  response-headers hooks are filtered out and never run at all;
- the success hook is dispatched as an *unbound* function and raises
  ``TypeError: missing 1 required positional argument: 'self'`` — which LiteLLM
  re-raises, 500-ing every successful completion;
- LiteLLM >= 1.98.0 rejects a class-valued callback outright at config load.

``get_instance_fn`` also resolves the module file **relative to the config
YAML's own directory**, so ``proxy_config.yaml`` must sit beside this file.
``tests/test_registration.py`` pins both invariants.

The module-level instance locates its config from ``FAULT_INJECTION_CONFIG``
(default: ``proxy_config.yaml`` next to this file).

Three hooks (exact signatures verified against LiteLLM main):
- ``async_post_call_success_hook``            — non-streaming: mutate content in place
- ``async_post_call_streaming_iterator_hook`` — streaming: buffer -> inject -> re-emit
- ``async_post_call_response_headers_hook``   — set the ``x-fault-injected`` marker

Marker caveat: for streaming responses HTTP headers are flushed before the body
is produced, so the header hook generally runs *before* the iterator hook knows
the outcome. The marker is therefore reliable for non-streaming responses; for
streaming, the **audit log is the authoritative debrief record** (it is written
the moment injection happens during iteration).
"""

from __future__ import annotations

import logging
import os
import random
import sys
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, List, Optional

# LiteLLM loads this module by file path (``spec_from_file_location``), which
# does NOT put the module's own directory on ``sys.path``. Without this the
# sibling imports below fail at proxy boot with ImportError.
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

from litellm.integrations.custom_logger import CustomLogger

import audit_log
from config import FaultInjectionConfig
from injectors import InjectionResult, build_injectors
from state import (
    STORE,
    InjectionDecision,
    extract_call_id,
    is_bypass_call,
)

logger = logging.getLogger("fault_injection")

MARKER_HEADER = "x-fault-injected"
# Resolved against this file, not the CWD: the proxy's working directory is
# whatever the operator launched it from, but the YAML lives beside the module.
_DEFAULT_CONFIG_PATH = os.path.join(_PLUGIN_DIR, "proxy_config.yaml")


class FaultInjector(CustomLogger):
    def __init__(self, config: Optional[FaultInjectionConfig] = None) -> None:
        super().__init__()
        if config is None:
            path = os.getenv("FAULT_INJECTION_CONFIG", _DEFAULT_CONFIG_PATH)
            try:
                config = FaultInjectionConfig.from_yaml(path)
            except FileNotFoundError:
                logger.warning(
                    "fault-injection config not found at %s; injection disabled", path
                )
                config = FaultInjectionConfig(enabled=False)
            except Exception:
                # Any parse/validation error must fail closed (injection off)
                # rather than take down the proxy callback at load time.
                logger.exception(
                    "failed to load fault-injection config at %s; injection disabled",
                    path,
                )
                config = FaultInjectionConfig(enabled=False)
        self.cfg = config
        self.injectors = build_injectors(self.cfg)
        logger.info(
            "FaultInjector loaded: enabled=%s rate=%s deterministic=%s types=%s",
            self.cfg.enabled,
            self.cfg.inject_rate,
            self.cfg.deterministic,
            sorted(self.injectors),
        )

    # ------------------------------------------------------------------ hooks

    async def async_post_call_success_hook(
        self, data: dict, user_api_key_dict: Any, response: Any
    ) -> Any:
        """Non-streaming path. Never raises: LiteLLM re-raises whatever a
        post-call hook throws, which would turn an already-successful
        completion into a 500. An awareness tool must not be able to take the
        proxy down, so any internal failure degrades to pass-through."""
        try:
            return await self._success_hook(data, user_api_key_dict, response)
        except Exception:
            logger.exception("fault-injection success hook failed; passing response through")
            return response

    async def _success_hook(
        self, data: dict, user_api_key_dict: Any, response: Any
    ) -> Any:
        content = _get_response_content(response)
        call_id = extract_call_id(data, response)
        if content is None:
            # pass-through / non-ModelResponse shape (e.g. raw provider dict):
            # we cannot safely locate the content field -> no-op.
            logger.debug("shape-guard: unrecognized response shape, skipping")
            STORE.record(call_id, InjectionDecision(injected=False))
            return response

        result = await self._decide_and_inject(
            content, data, user_api_key_dict, call_id, streaming=False
        )
        if result is not None:
            _set_response_content(response, result.content)
            # Stamp the marker onto the response itself so it survives regardless
            # of whether the header hook runs before or after this hook (headers
            # are assembled at serialization, after all hooks). The public header
            # hook below is kept as a fallback.
            _stamp_marker(response)
            STORE.record(
                call_id,
                InjectionDecision(injected=True, error_type=result.record.error_type),
            )
        else:
            STORE.record(call_id, InjectionDecision(injected=False))
        return response

    async def async_post_call_streaming_iterator_hook(
        self, user_api_key_dict: Any, response: Any, request_data: dict
    ) -> AsyncGenerator[Any, None]:
        # Decide whether to sample BEFORE consuming the stream, using only the
        # request (no body/response id yet). If not selected, pass chunks
        # straight through to preserve streaming latency for most traffic.
        sample_id = extract_call_id(request_data)
        try:
            selected = self._selected_for_injection(
                request_data, user_api_key_dict, sample_id
            )
        except Exception:
            logger.exception("fault-injection sampling failed; streaming through")
            selected = False
        if not selected:
            STORE.record(sample_id, InjectionDecision(injected=False))
            async for chunk in response:
                yield chunk
            return

        # Selected: buffer the whole stream so we can inject on the full text.
        chunks: List[Any] = []
        async for chunk in response:
            chunks.append(chunk)
        full_text = _assemble_stream_text(chunks)
        # Key audit/store on the client-visible completion id (carried on the
        # chunks), matching what a client echoes to /feedback; fall back to the
        # request id only if the chunks carry none.
        client_id = _first_chunk_id(chunks) or sample_id

        result = None
        try:
            if full_text.strip():
                # Seed from ``sample_id`` — the SAME id the sampling draw above
                # used. Seeding the injector choice from the chunk-carried
                # completion id instead would put the two halves of one
                # decision on two different RNG streams, so `deterministic`
                # mode would not actually reproduce (the completion id is
                # provider-assigned and differs per run).
                result = await self._run_injector(
                    full_text,
                    request_data,
                    client_id,
                    streaming=True,
                    rng=self._rng(sample_id),
                )
        except Exception:
            # Buffered chunks are already captured; ship them unmodified rather
            # than breaking a stream the client is mid-way through reading.
            logger.exception("fault-injection streaming hook failed; passing stream through")
            result = None

        if result is None:
            STORE.record(client_id, InjectionDecision(injected=False))
            for chunk in chunks:
                yield chunk
            return

        STORE.record(
            client_id,
            InjectionDecision(injected=True, error_type=result.record.error_type),
        )
        for chunk in _rewrite_stream_chunks(chunks, result.content):
            yield chunk

    async def async_post_call_response_headers_hook(
        self,
        data: dict,
        user_api_key_dict: Any,
        response: Any,
        request_headers: Optional[dict] = None,
        litellm_call_info: Optional[dict] = None,
    ) -> Optional[dict]:
        # Reliable for non-streaming (success hook already recorded the outcome).
        # For streaming this usually runs before the body is produced and finds
        # nothing recorded yet -> no header; the audit log is authoritative there.
        try:
            call_id = extract_call_id(data, response)
            decision = STORE.pop(call_id)
            if decision and decision.injected:
                return {MARKER_HEADER: "true"}
        except Exception:
            logger.exception("fault-injection header hook failed; omitting marker")
        return None

    # ------------------------------------------------------------- internals

    def _selected_for_injection(
        self, data: dict, user_api_key_dict: Any, call_id: Optional[str]
    ) -> bool:
        """Sampling + target guard, independent of response content."""
        if not self.cfg.enabled or not self.injectors:
            return False
        if is_bypass_call(data):  # the injector's own nested LLM call
            return False
        key_alias = _get_key_alias(data, user_api_key_dict)
        request_text = _get_request_text(data)
        if not self.cfg.targets.is_targetable(key_alias, request_text):
            return False
        rng = self._rng(call_id)
        return rng.random() < self.cfg.inject_rate

    async def _decide_and_inject(
        self,
        content: str,
        data: dict,
        user_api_key_dict: Any,
        call_id: Optional[str],
        streaming: bool,
    ) -> Optional[InjectionResult]:
        if not self._selected_for_injection(data, user_api_key_dict, call_id):
            return None
        return await self._run_injector(
            content, data, call_id, streaming, rng=self._rng(call_id)
        )

    async def _run_injector(
        self,
        content: str,
        data: dict,
        call_id: Optional[str],
        streaming: bool,
        rng: random.Random,
    ) -> Optional[InjectionResult]:
        """Pick an applicable injector, run it, and audit the outcome.

        A durable audit record is a PRECONDITION for shipping an injection: if
        the ``injected`` audit write fails, we decline rather than send an
        un-debriefable deception. Skip/decline audits stay best-effort.
        """
        injector = self._choose_injector(content, rng)
        model = data.get("model") if isinstance(data, dict) else None

        if injector is None:
            self._audit_safe(
                call_id, model, "skipped", None,
                "no applicable injector", content, None, streaming,
            )
            return None

        result = await injector.inject(content, rng)
        if result is None:
            self._audit_safe(
                call_id, model, "skipped", injector.error_type,
                "injector declined", content, None, streaming,
            )
            return None

        try:
            self._audit_write(
                call_id, model, "injected", result.record.error_type,
                result.record.detail, content, result.content, streaming,
            )
        except Exception:
            logger.exception(
                "audit write failed; declining injection to preserve the "
                "debrief guarantee (request_id=%s)", call_id,
            )
            return None
        return result

    def _choose_injector(self, content: str, rng: random.Random):
        weights = self.cfg.active_error_types()
        applicable = [
            (name, inj)
            for name, inj in self.injectors.items()
            if weights.get(name, 0) > 0 and inj.applies(content)
        ]
        if not applicable:
            return None
        names = [n for n, _ in applicable]
        chosen = rng.choices(names, weights=[weights[n] for n in names], k=1)[0]
        return dict(applicable)[chosen]

    def _rng(self, call_id: Optional[str]) -> random.Random:
        # Deterministic mode: seed per-call so runs are reproducible and
        # order-independent under concurrency. Otherwise system entropy.
        if self.cfg.deterministic:
            return random.Random(f"{self.cfg.seed}:{call_id}")
        return random.Random()

    def _audit_write(
        self,
        call_id: Optional[str],
        model: Optional[str],
        event: str,
        error_type: Optional[str],
        detail: str,
        original: str,
        manipulated: Optional[str],
        streaming: bool,
    ) -> None:
        """Write one audit line. Propagates on failure (callers decide)."""
        audit_log.write_entry(
            self.cfg.audit_log_path,
            audit_log.AuditEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                request_id=call_id,
                model=model,
                event=event,
                error_type=error_type,
                detail=detail,
                original=original,
                manipulated=manipulated,
                streaming=streaming,
            ),
        )

    def _audit_safe(self, *args) -> None:
        """Best-effort audit for non-critical (skip/decline) events."""
        try:
            self._audit_write(*args)
        except Exception:
            logger.exception("failed to write (non-critical) audit entry")


# --------------------------------------------------------------- helpers (pure)


def _stamp_marker(response: Any) -> None:
    """Attach the marker header to the response's hidden params.

    LiteLLM merges ``_hidden_params['additional_headers']`` into the outgoing
    HTTP response at serialization time — after every hook has run — so this is
    ordering-independent, unlike the header hook which may run before the
    success hook. Best-effort: never break the response path.
    """
    try:
        hidden = getattr(response, "_hidden_params", None)
        if hidden is None:
            hidden = {}
            response._hidden_params = hidden
        headers = hidden.setdefault("additional_headers", {})
        headers[MARKER_HEADER] = "true"
    except Exception:
        pass


def _first_chunk_id(chunks: List[Any]) -> Optional[str]:
    for chunk in chunks:
        cid = getattr(chunk, "id", None)
        if cid:
            return str(cid)
    return None


def _get_response_content(response: Any) -> Optional[str]:
    """Return message content from a ModelResponse, or None if the shape is
    not the recognized ``choices[0].message.content`` string (e.g. a raw
    pass-through dict)."""
    try:
        choices = getattr(response, "choices", None)
        if not choices:
            return None
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        return content if isinstance(content, str) else None
    except Exception:
        return None


def _set_response_content(response: Any, new_content: str) -> None:
    response.choices[0].message.content = new_content


def _assemble_stream_text(chunks: List[Any]) -> str:
    parts: List[str] = []
    for chunk in chunks:
        try:
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if isinstance(piece, str):
                parts.append(piece)
        except Exception:
            continue
    return "".join(parts)


def _rewrite_stream_chunks(chunks: List[Any], new_content: str) -> List[Any]:
    """Re-emit buffered chunks with the injected content.

    The full new content is placed on the first chunk that carried delta text;
    every other content-bearing delta is blanked. Non-content chunks (role,
    finish_reason, usage) are preserved untouched so downstream clients still
    see a well-formed stream. Concatenating the deltas yields exactly
    ``new_content``.
    """
    placed = False
    for chunk in chunks:
        try:
            delta = chunk.choices[0].delta
        except Exception:
            continue
        if not hasattr(delta, "content") or not isinstance(
            getattr(delta, "content", None), str
        ):
            continue
        if not placed:
            delta.content = new_content
            placed = True
        else:
            delta.content = ""
    return chunks


def _get_request_text(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    parts: List[str] = []
    # Anthropic-style requests carry the system prompt as a TOP-LEVEL `system`
    # field rather than a messages entry. Reading only `messages` let a denied
    # topic that appears solely in the system prompt slip past the guard.
    system = data.get("system")
    if isinstance(system, str):
        parts.append(system)
    elif isinstance(system, list):
        parts.extend(b.get("text", "") for b in system if isinstance(b, dict))
    for msg in data.get("messages") or []:
        if isinstance(msg, dict):
            c = msg.get("content")
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, list):  # multimodal content blocks
                parts.extend(
                    b.get("text", "") for b in c if isinstance(b, dict)
                )
    return "\n".join(parts)


def _get_key_alias(data: Any, user_api_key_dict: Any) -> Optional[str]:
    alias = getattr(user_api_key_dict, "key_alias", None)
    if alias:
        return str(alias)
    if isinstance(data, dict):
        meta = data.get("metadata") or {}
        for key in ("user_api_key_alias", "user_api_key_team_alias"):
            if meta.get(key):
                return str(meta[key])
    return None


# ---------------------------------------------------------------- registration

#: The object LiteLLM must load. ``get_instance_fn`` performs a bare
#: ``getattr(module, "proxy_handler_instance")`` — it does not instantiate — so
#: the registered attribute has to be an INSTANCE of ``CustomLogger``. Register
#: this, never the class:
#:
#:     litellm_settings:
#:       callbacks: ["fault_injector.proxy_handler_instance"]
#:
#: Constructed at import time and fail-closed: if the config is missing or
#: unparseable the instance still loads with injection disabled, so a config
#: problem degrades to a no-op callback rather than a proxy that will not boot.
proxy_handler_instance = FaultInjector()

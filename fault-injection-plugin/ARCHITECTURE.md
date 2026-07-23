<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Architecture

This document explains how the fault-injection plugin is put together: the
components, the request lifecycle, the decision flow, and the non-obvious
constraints that shaped the design. For usage see [README.md](README.md).

## 1. Big picture

The system has **two halves** that meet at a shared key — the per-request id.

```
                        INJECTION HALF                         REACTION HALF
   ┌───────────────────────────────────────────┐     ┌─────────────────────────┐
   │  Client → LiteLLM Proxy → LLM Provider     │     │  client / reviewer UI   │
   │                    ↑                        │     │          │ POST         │
   │        FaultInjector (CustomLogger)         │     │          ▼              │
   │        ├─ success hook   (mutate + marker)  │     │   feedback_api.py       │
   │        ├─ streaming hook (streaming)        │     │          │              │
   │        └─ headers hook   (marker fallback)  │     │          ▼              │
   │                    │                        │     │   feedback.jsonl        │
   │                    ▼                        │     └─────────────────────────┘
   │            injections.jsonl  ───────────────┼────────────────┐
   └─────────────────────────────────────────────┘                ▼
                                                          report.py  (join on request_id)
                                                                    │
                                                                    ▼
                                                          noticed-rate per error type
```

The injector never learns whether a user noticed; the feedback service never
learns what was faked. `report.py` is the only place the two logs are joined,
keyed on `request_id`. This separation is deliberate: the measurement is a
post-hoc analysis, not something the hot path needs.

## 2. Components

| module | responsibility |
|--------|----------------|
| `fault_injector.py` | `FaultInjector(CustomLogger)` — the three hooks, sampling, target-guard, orchestration, audit writes. The only module LiteLLM loads. |
| `config.py` | `FaultInjectionConfig` dataclasses; YAML + env loading; `active_error_types()` gates LLM injectors out of deterministic mode. |
| `injectors/` | Strategy pattern. `base.py` defines the `Injector` protocol; `mechanical.py` and `llm.py` implement it; `__init__.build_injectors()` is the registry/factory. |
| `state.py` | `DecisionStore` (per-request injection decision, shared across hooks) + bypass-flag helpers. |
| `audit_log.py` | Atomic JSONL append: `write_entry` (used by the injector) wraps `append_json` (used directly by the feedback service). |
| `feedback_api.py` | FastAPI service capturing user reactions to `feedback.jsonl`. |
| `report.py` | Joins injections ⋈ feedback → noticed-rate metric. |

## 3. Request lifecycle

LiteLLM invokes the hooks as **separate callbacks** for one request. The order
that matters:

**Non-streaming (`stream=false`):**
```
provider returns → async_post_call_success_hook  (mutate content, STAMP marker
                                                   on response._hidden_params)
                 → async_post_call_response_headers_hook (fallback marker source)
                 → body + headers sent to client
```
The marker is **not** dependent on hook order. `async_post_call_success_hook`
stamps `x-fault-injected` onto `response._hidden_params["additional_headers"]`,
which LiteLLM merges into the HTTP response at serialization — after every hook
has run. The `async_post_call_response_headers_hook` (which reads the decision
store) is only a **fallback**: it would miss the marker if it happened to run
before the success hook, which is exactly why the stamp is the primary path.

**Streaming (`stream=true`):**
```
async_post_call_response_headers_hook   ← runs here (HTTP headers flush FIRST)
async_post_call_streaming_iterator_hook ← body produced/mutated AFTER headers
```
Because HTTP requires headers before the body, the header hook generally runs
**before** the iterator hook has decided anything. So for streaming the marker
header is best-effort (usually absent), and the **audit log is the authoritative
debrief record** — it is written the instant injection happens during iteration.
This ordering constraint is a property of HTTP, not of LiteLLM, and cannot be
designed away; the audit log is the answer to it.

## 4. Decision flow (per response)

```
enabled? ──no──► pass through
   │yes
bypass call? ──yes──► pass through          (the injector's own nested LLM call)
   │no
target-guard: key alias ∈ allowlist AND no denied topic? ──no──► pass through
   │yes
sample: rng.random() < inject_rate? ──no──► pass through
   │yes
choose injector: applicable(content) ∩ weighted(error_types) ──none──► skip (audit)
   │
run injector.inject() ──None (declined)──► skip (audit)
   │
audit "injected" (durably) ──write fails──► DECLINE (ship original, no marker)
   │ ok
mutate content + stamp marker + record decision
```

Two guards run **before** sampling — the kill-switch/bypass and the target-guard
— so a non-targetable request never even draws the RNG. `_selected_for_injection`
holds the content-independent part (used identically by the streaming path to
decide whether to buffer at all) and `_run_injector` holds the content-dependent
part (applicability + the actual injection).

## 5. Injectors

`Injector` is a two-method protocol:
- `applies(content) -> bool` — cheap gate (e.g. `bad_code` only fires when a
  fenced code block exists).
- `inject(content, rng) -> InjectionResult | None` — perform the change, or
  **decline** by returning `None` (the caller then passes the original through
  and logs a skip). Declining is a first-class outcome, not an error.

Two families:
- **Mechanical** (`bad_code`, `fake_source`) — pure, deterministic given the
  RNG. `bad_code` parses Python with `ast` and swaps exactly one operator
  (falling back to conservative token swaps for other languages, else declines).
- **LLM-in-the-loop** (`factual`, `logic_break`) — call a configured model to
  rewrite with one subtle flaw. Two safety properties, both enforced here:
  1. **No recursion.** The nested `litellm.acompletion` call is tagged with
     `BYPASS_METADATA_KEY`; `is_bypass_call()` makes the sampler skip it.
     Without this, an injected answer could trigger another injection.
  2. **Fail safe.** An unchanged, empty, or length-divergent rewrite (>
     `max_len_delta`) is rejected → decline, never emit a corrupted response.

`deterministic: true` removes the LLM family entirely (both from the weights via
`active_error_types()` and from construction in `build_injectors()`), leaving a
fully reproducible, seed-driven system — the mode the test suite runs in.

## 6. Shared state & determinism

- **`DecisionStore`** backs the *fallback* header hook (the primary marker path
  is the `_hidden_params` stamp, §3) so it still matches what was injected. It is
  a thread-safe, size-bounded LRU keyed on the **client-visible completion id**
  (`response.id` — the `id` field the caller receives, and the value it echoes to
  `/feedback`), falling back to the internal `litellm_call_id` only when that is
  unavailable. Entries are consumed (`pop`) by the header hook or evicted, so a
  request that never reaches it cannot leak memory.
- **RNG**: in deterministic mode the per-request RNG is seeded from
  `f"{seed}:{call_id}"`, making injection reproducible **and order-independent
  under concurrency** (no shared mutable RNG). In normal mode it uses system
  entropy.

## 7. Shape-guard (provider uniformity)

The content hook acts on `response.choices[0].message.content`. That shape holds
for the unified `/chat/completions` path (LiteLLM normalizes every provider into
a `ModelResponse`). **Pass-through endpoints** (e.g. `/v1/messages`) instead hand
the hook a **raw provider dict**. `_get_response_content()` returns `None` for
any unrecognized shape and the hook **no-ops** — it never guesses at a dict's
content field and risks corrupting an unrelated response.

## 8. Failure containment

- **Auditing an `injected` event is a precondition, not an afterthought.** If the
  audit write fails, `_run_injector` **declines** the injection — the original is
  shipped and no marker is stamped, so the tool never emits an un-debriefable
  deception. Skip/decline audits stay best-effort (a logging failure there never
  breaks the response).
- Every injector can decline; the orchestrator always has a defined
  pass-through path.
- The plugin fails **closed**: missing/unparseable config, empty allowlist, a
  non-numeric/out-of-range rate, or an unrecognized response shape all result in
  *no injection*, never an accidental one. The `FAULT_INJECTION_ENABLED` env var
  is one-directional — it can force injection off but never on.

## 9. Extending it

Add an error type by implementing the `Injector` protocol in `injectors/` and
registering it in `build_injectors()` with a config weight. Mechanical injectors
should be deterministic from `rng`; LLM injectors must tag their own calls with
`BYPASS_METADATA_KEY` and implement a reject condition. Add a matching unit test
(seeded) and, if it touches the response shape, extend the integration smoke
test.

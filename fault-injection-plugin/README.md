<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Fault-Injection Plugin for LiteLLM

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](LICENSE)
[![Writeup](https://img.shields.io/badge/Writeup-limacher--itsec.ch-0a7ea4.svg)](https://limacher-itsec.ch/blog/ai-fault-projection.html)

A LiteLLM proxy callback that **deliberately injects subtle faults** into LLM
responses to measure whether users notice — an *awareness* tool, analogous to
phishing simulations. It is **transparent by design**: every manipulation is
marked and audited so it can be reconstructed and debriefed.

> ⚠️ **Operate only in briefed environments.** During a session the user is
> genuinely misled (the marker is an HTTP header they never see); consent is
> established up front at the organization/training level and resolved by a
> debrief. Never point this at users making real high-stakes decisions — the
> `targets` allowlist/denylist exists to enforce that.

**Background:** [The AI fault-injection writeup](https://limacher-itsec.ch/blog/ai-fault-projection.html)
explains the threat model and why measuring the *noticed rate* matters.

## How it works

Injection is one half; **reaction capture is the other**. The injector records
*what* was faked; the feedback service records *how the user reacted*; the
report joins them to produce the metric the tool exists for — the **noticed
rate** per error type.

```
Client → LiteLLM Proxy → Provider
              ↑
      FaultInjector (CustomLogger)
      ├─ async_post_call_success_hook            # non-streaming: mutate content
      ├─ async_post_call_streaming_iterator_hook # streaming: buffer→inject→re-emit
      └─ async_post_call_response_headers_hook   # x-fault-injected marker (fallback)
              │
              ├─ sampler + target-guard (decided per request)
              ├─ injectors/ (mechanical | llm)
              └─ audit → injections.jsonl

feedback_api.py  → feedback.jsonl        # user reactions, keyed on request-id
report.py        → injections ⋈ feedback → noticed-rate per error type
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the hook lifecycle, decision flow,
shared-state model, and the design constraints behind them.

## Injectors

| type          | engine        | notes                                            |
|---------------|---------------|--------------------------------------------------|
| `bad_code`    | mechanical    | flips one operator in a fenced code block (Python via `ast`) |
| `fake_source` | mechanical    | appends a fabricated citation                    |
| `factual`     | LLM-in-the-loop | rewrites one number/date/name to a wrong value |
| `logic_break` | LLM-in-the-loop | introduces one subtle contradiction            |

The two LLM injectors call a configured model **directly via the SDK** (not back
through the proxy) and tag the call so it **bypasses injection** (no recursion).
Each fails safe: an unchanged, empty, or wildly-divergent rewrite is **declined**
and the original passes through (logged as a skip).

Set `deterministic: true` to restrict to the mechanical injectors with a fixed
seed — fully reproducible, and the mode used by the test suite.

## Setup

```bash
pip install -r requirements.txt        # pins litellm[proxy]==1.93.0 (hook surface moves weekly)
# ensure the plugin dir is importable, then start the proxy (from the plugin root):
litellm --config config/proxy_config.yaml
# in a SEPARATE terminal, the reaction-capture service (also from the plugin root):
uvicorn feedback_api:app --port 8181
```

Configuration lives under the `fault_injection:` key of `config/proxy_config.yaml`
(see that file for every option). The kill-switch is `enabled`.

**Environment variables:**

| var | effect |
|-----|--------|
| `FAULT_INJECTION_CONFIG` | path to the proxy config (default `config/proxy_config.yaml`); read by both the plugin and the feedback service |
| `FAULT_INJECTION_ENABLED` | one-directional kill-switch: a falsy value forces injection **off**; a truthy value cannot enable a disabled config |
| `FAULT_FEEDBACK_TOKEN` | bearer token required by `POST /feedback` (unset = open — bind to localhost then) |
| `FAULT_FEEDBACK_LOG` | overrides the `feedback_log_path` from the config |

## Debrief / reporting

```bash
python report.py --injections audit/injections.jsonl --feedback audit/feedback.jsonl
```

Every injected response is one line in `audit/injections.jsonl` (original +
manipulated content, error type, request-id, timestamp). Clients/reviewers post
reactions to `POST /feedback {request_id, signal}` where `signal ∈ {noticed,
corrected, reasked, thumbs_down, none}`.

**The `request_id` must be the `id` field from the response body the client
received** (the completion id, e.g. `chatcmpl-…`). The injector keys its audit
log on that same client-visible id, so this is what makes the join succeed.

Set `FAULT_FEEDBACK_TOKEN` to require `Authorization: Bearer <token>` on the
feedback endpoint; leave it unset only if the service is bound to localhost/a
private network. The feedback path follows `feedback_log_path` from the config
(override with `FAULT_FEEDBACK_LOG`).

## Scope & known limitations

- **Marker: non-streaming reliable, streaming best-effort.** For non-streaming
  responses the `x-fault-injected` marker is stamped onto the response's
  `_hidden_params` (merged into the HTTP headers at serialization, so it is
  independent of hook ordering), with the response-headers hook as a fallback.
  For streaming, HTTP headers flush before the body is produced, so the marker is
  best-effort and the **audit log is the authoritative debrief record**.
- **Sampled streamed responses lose streaming latency** (they are buffered so
  the injector can act on the full text); bounded to ~`inject_rate` of traffic.
- **Unified endpoint only.** The content hook acts on `choices[0].message.content`.
  Pass-through endpoints (e.g. `/v1/messages`) hand the hook a raw dict; a
  shape-guard detects this and **no-ops** rather than corrupting the response.
- **Audit/feedback logs are sensitive at rest** (full prompts/responses). All
  `*.jsonl` are gitignored regardless of the configured path; apply filesystem
  permissions and a retention policy.
- **`deny_topics` is best-effort, not a safety boundary.** It substring-scans the
  *prompt* only, so it misses high-stakes *answers* whose prompts contain no
  keyword. The real blast-radius control is the **key-alias allowlist** — point
  red-team keys only at low-stakes corpora.

## Testing

```bash
python -m pytest        # hermetic: stubs litellm, no network, no proxy needed
```

Unit tests cover each injector (seeded/deterministic), sampling, target-guard,
kill-switch, shape-guard, streaming buffer-and-inject, the no-recursion bypass,
the feedback API, and the report join.

The **integration smoke test** (`tests/test_integration_smoke.py`) builds a real
`ModelResponse` with the installed LiteLLM and drives the hooks against it —
proving the mutation, marker, and audit keying survive a version bump. Run it on
its own so the litellm stub is not installed:

```bash
RUN_INTEGRATION=1 python -m pytest tests/test_integration_smoke.py
```

A full end-to-end against a booted proxy is documented as a manual step in that
test's docstring.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE). Because the proxy is network
software, the AGPL network clause applies to anyone offering this plugin over a
network.

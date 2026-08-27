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
# start the proxy FROM THE PLUGIN ROOT (the YAML must sit beside fault_injector.py):
litellm --config proxy_config.yaml
# in a SEPARATE terminal, the reaction-capture service (also from the plugin root):
uvicorn feedback_api:app --host 127.0.0.1 --port 8181
```

Configuration lives under the `fault_injection:` key of `proxy_config.yaml`
(see that file for every option). The kill-switch is `enabled`, and the shipped
example is **off** — arm it deliberately, per deployment.

### Registration (the part that is easy to get wrong)

```yaml
litellm_settings:
  callbacks: ["fault_injector.proxy_handler_instance"]   # an INSTANCE, not the class
```

Two LiteLLM behaviours constrain this, and getting either wrong yields a plugin
that looks correct and never runs:

- **`get_instance_fn` does not instantiate.** It performs a bare
  `getattr(module, attr)`. Registering `fault_injector.FaultInjector` leaves
  LiteLLM holding a *class*: `isinstance(cb, CustomLogger)` is False, so the
  streaming-iterator and response-headers hooks are silently dropped, while the
  success hook is dispatched unbound and raises `TypeError: missing 1 required
  positional argument: 'self'` — a 500 on every completion. LiteLLM ≥ 1.98.0
  rejects a class-valued callback at config load instead.
- **The module path resolves relative to the YAML**, as
  `dirname(config) + "/fault_injector.py"` — not via `sys.path`. So
  `proxy_config.yaml` must live in the same directory as `fault_injector.py`.
  (The module puts its own directory on `sys.path` at import so its sibling
  imports work under LiteLLM's by-file-path loading.)

`tests/test_registration.py` pins both against the real shipped YAML.

**Environment variables:**

| var | effect |
|-----|--------|
| `FAULT_INJECTION_CONFIG` | path to the proxy config (default: `proxy_config.yaml` beside the module); read by both the plugin and the feedback service |
| `FAULT_INJECTION_ENABLED` | one-directional kill-switch: a falsy value forces injection **off**; a truthy value cannot enable a disabled config |
| `FAULT_FEEDBACK_TOKEN` | bearer token required by `POST /feedback` (unset = open, and logged as a warning — bind to localhost then) |
| `FAULT_FEEDBACK_LOG` | overrides the `feedback_log_path` from the config |
| `FAULT_FEEDBACK_RATE_LIMIT` | `POST /feedback` requests per minute per client IP (default `60`; `0` disables) |

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
- **Audit/feedback logs are sensitive at rest** — they hold verbatim model
  output (original *and* manipulated), though not request prompts. Files and
  directories are created `0600`/`0700` and all `*.jsonl` are gitignored
  regardless of the configured path; still apply a retention policy, and tighten
  any pre-existing `audit/` (creation modes do not touch existing files).
- **`deny_topics` is best-effort, not a safety boundary.** It substring-scans the
  *request* only (messages plus an Anthropic-style top-level `system`), so it
  misses high-stakes *answers* whose prompts contain no keyword, and it is
  brittle to synonyms and other languages. The real blast-radius control is the
  **key-alias allowlist** — point red-team keys only at low-stakes corpora.
  A request carrying no key alias is never targetable, so even `["*"]` excludes
  unkeyed traffic; that is not a reason to ship `["*"]`.
- **The LLM injectors are an egress path.** `factual` and `logic_break` send the
  **full original answer** to `llm_injector.model` over a direct SDK call, which
  skips whatever DLP or redaction the proxy applies to normal traffic. Point that
  model at a cleared deployment, or set `deterministic: true` to drop the LLM
  family entirely and keep every answer inside the proxy. The call is bounded by
  `llm_injector.timeout_s`.
- **`inject_rate` fails closed.** A value outside `[0,1]` (e.g. `10` from reading
  "10%") is **not** clamped up to `1.0` — it resolves to `0.0` and logs a
  warning. Clamping up would turn a typo into 100% injection.
- **A hook failure never breaks a response.** All three hooks contain their own
  exceptions and degrade to pass-through; LiteLLM re-raises what a post-call hook
  throws, so an uncontained error would 500 an already-successful completion.

## Testing

```bash
python -m pytest        # hermetic: stubs litellm, no network, no proxy needed
```

Unit tests cover each injector (seeded/deterministic), sampling, target-guard,
kill-switch, shape-guard, streaming buffer-and-inject, the no-recursion bypass,
the feedback API, and the report join.

`tests/test_registration.py` covers the other half: whether LiteLLM will ever
*call* the hooks. It re-implements `get_instance_fn`'s resolution against the
real shipped `proxy_config.yaml` and asserts the callback resolves to an
instance, the YAML sits beside the module, every hook is on the leaf class, and
each hook accepts LiteLLM's keyword call site. Hook-body tests construct a
`FaultInjector` by hand and so cannot catch any of that.

The **integration smoke test** (`tests/test_integration_smoke.py`) builds a real
`ModelResponse` with the installed LiteLLM and drives the hooks against it —
proving the mutation, marker, and audit keying survive a version bump. Run it on
its own so the litellm stub is not installed:

```bash
RUN_INTEGRATION=1 python -m pytest tests/test_integration_smoke.py
```

A full end-to-end against a booted proxy is documented as a manual step in that
test's docstring.

## Operating guidance

If you run this:

- **Dedicated `redteam-*` key aliases only.** Never `allow_key_aliases: ["*"]`.
- **Set `FAULT_FEEDBACK_TOKEN`** and bind the feedback service to `127.0.0.1`.
  Without a token anyone reachable can forge reaction signals and skew the
  noticed-rate the tool exists to measure.
- **Treat `audit/*.jsonl` as sensitive at rest.** New files/dirs are created
  `0600`/`0700`, but tighten any that predate this (`chmod -R go-rwx audit/`)
  and set a retention policy. The lines hold verbatim model output; the injector
  does not log request prompts.
- **Prefer `deterministic: true`** if answers must not leave the proxy.
- **Keep `inject_rate` in `[0,1]`** — anything else silently disables injection.
- **Hash-pin `litellm[proxy]==1.93.0` yourself** (`pip-compile --generate-hashes`)
  and re-run the suite, `tests/test_registration.py` included, on any bump: the
  hook surface and the callback-loading rules both move between releases.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE). Because the proxy is network
software, the AGPL network clause applies to anyone offering this plugin over a
network.

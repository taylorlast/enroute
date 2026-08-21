# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Gym-shaped environments: subclass `Environment`, decorate methods with
  `@tool`, and use `reset` / `step` / `close_episode` alongside `rollout()`.
  One rollout is one episode; the model is passed in, not baked into the
  environment. The default `step` dispatches tools and records a decision;
  override `step` when the action is not a tool call (OpenEnv-style).
- Episode traces: optional `Decision` steps (observation, model context, parsed
  action, tool results, reward events) plus `initial_state`, `final_state`,
  `model`, `metrics`, `terminated`, `truncated`, and `environment_fingerprint`.
  `Trace.transitions()` flattens an episode for a future trainer. Production
  `llm` / `tool` / `event` steps still round-trip.
- Simulated Twitter account example (`examples/environment/twitter`) with
  person-complete tools and pluggable goals.
- `Trace.credit()` (late reward on the episode or one decision) and
  `Trace.returns(gamma=…)` for discounted credit assignment. Reward is not
  assumed per decision — research-then-answer uses a terminal scorer; likes
  that arrive later use `credit`.
- Library example (`examples/environment/library`): search / read / answer,
  the start-to-finish RL handoff.
- Wordle example (`examples/environment/wordle`): `WordleEnv` owns the secret,
  official word lists, `guess`, board observations, and rewards. The model
  is only the policy. The example plays via `reset` / `step` (Gymnasium /
  OpenEnv); `rollout()` stays as convenience.
- `Observation` and `State` types (`Environment[ObsT, StateT]`). Nested
  `@tool` calls are recorded as `ToolCallStep.children` for hierarchical
  credit later. Library `research` is the demo.

### Changed

- `World` / `BaseWorld` / `world=` are gone. Write
  `class WordleEnv(Environment[WordleObservation, WordleState])` and put
  episode data on `self.state`. `Rollout.env` replaces `Rollout.world`.
  Concurrent episodes use `env.spawn()` (Benchmark does this automatically).
  Drive agents with `reset` / `step` / `rollout`; `observe` is an author hook.

## [0.3.4] - 2026-08-18

### Added

- `StreamDelta.reasoning_started` and `reasoning_finished` mark the bounds of a
  thinking block, including when Anthropic encrypts the tokens and sends empty
  `thinking_delta` events. A UI can show a thinking indicator the same way
  Cursor does, even when there is no readable text. Anthropic requests ask for
  adaptive thinking by default so those bounds actually arrive.

## [0.3.3] - 2026-08-18

### Added

- `Message.reasoning` and `StreamDelta.reasoning` surface thinking-model output,
  read from `/v1/responses` on OpenAI, `reasoning`/`reasoning_content` on other
  OpenAI-compatible hosts, `thinking` blocks on Anthropic, `thought` parts on
  Gemini, and `reasoningContent` on Bedrock. `reasoning_signature` carries the
  host's attestation so a thinking block can be replayed on the next turn.
- OpenAI requests now use `/v1/responses` by default. Chat Completions rejects
  function tools on every current model and never reports reasoning; Responses
  streams both. A request that needs `stop`, `seed`, or a `max_tokens` below 16
  stays on Chat Completions. Pin `transport="chat"` or `transport="responses"`
  to force one endpoint.
- `response_format` now works on every host. Gemini uses its native
  `responseSchema`; Anthropic and Bedrock force a single-tool call whose input
  schema is the requested schema and unwrap the result into message content.
- `tool_choice` is translated for Anthropic and Bedrock.
- A capabilities guide covering streaming, tool calling, structured output, and
  reasoning.

### Changed

- Gemini requests ask for thought summaries (`includeThoughts`) so a thinking
  model is not silent for seconds before the first text delta.

### Fixed

- Anthropic streams now emit tool calls. `content_block_start` and
  `input_json_delta` were dropped, so tool calling silently produced nothing
  when `stream=True`.
- Bedrock streams now emit tool calls and reasoning from `contentBlockStart` and
  `contentBlockDelta`.
- Gemini streams now emit tool calls; `functionCall` parts were dropped.
- Failed streaming requests report the host's error instead of an httpx
  "Attempted to access streaming response content" message, which masked real
  4xx bodies on Anthropic, Google, and Bedrock.
- OpenAI-compatible hosts that 400 on `max_tokens` are retried with
  `max_completion_tokens`, and reasoning models that reject `temperature` or
  `top_p` are retried without them. Both rejections are remembered per model.

## [0.3.1] - 2026-08-18

### Added

- `StreamChunk.to_openai()` so a hosted gateway can emit one OpenAI Chat Completions
  SSE shape from every host. `raw` stays on the chunk for debugging and is not
  part of the client contract.
- Live stream smoke tests that pin `provider.only` for each configured key.

### Fixed

- Anthropic streams now carry `input_tokens` from `message_start` onto the usage
  chunk, so a streamed prompt is billed at the real token count.
- OpenAI-compatible hosts that 400 on `stream_options` are retried without it.

## [0.1.0] - 2026-08-13

### Added

- Initial `enroute` package: unified LLM routing with traces, environments, and benchmarks.
- Native httpx providers for OpenAI-compatible APIs, Anthropic, and Google.
- `Enroute` client with hosted-gateway mode (`ENROUTE_API_KEY`) and BYOK `providers={...}`.
- Routing policies, fallbacks, retries, model catalog, and local cost estimation.
- Tracing with JSONL/SQLite/OTel sinks, redaction, sampling, and late labels.
- Environments (tasks, tools, scorers), versioned datasets, and benchmark reports.
- Docs site (MkDocs) and cookbook-style routing examples.

[Unreleased]: https://github.com/taylorlast/enroute/compare/v0.3.4...HEAD
[0.3.4]: https://github.com/taylorlast/enroute/releases/tag/v0.3.4
[0.3.3]: https://github.com/taylorlast/enroute/releases/tag/v0.3.3
[0.3.1]: https://github.com/taylorlast/enroute/releases/tag/v0.3.1
[0.1.0]: https://github.com/taylorlast/enroute/releases/tag/v0.1.0

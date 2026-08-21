# Trace

**Scenario:** You run a support triage bot. Last Tuesday refund tickets started failing more often. You need to see the exact prompts, which model served them, which fallbacks were tried, and whether the user ultimately resolved the issue. Or you ran a Twitter-account episode and need the full trajectory for training.

A **Trace** is that record.

## One sentence

A Trace is the ordered history of one interaction — for environments, **one episode** — plus an optional labeled outcome.

## Per episode, not per decision

Persist **one trace per episode**. Each step *inside* that trace is a **decision** (observation → model → action → tool result → reward).

A top-level document per decision would lose the trajectory: initial state, return, and which actions belonged together. Offline RL needs the episode. Flatten later with `trace.transitions()`.

```python
from enroute import Trace, Outcome

trace = Trace(trace_id="...", tags={"surface": "support"})
trace.add_llm(request=req, response=resp, attempts=attempts)
trace.label(scores={"resolved": 1.0}, reward=1.0, feedback="user closed ticket")
```

Environment rollouts produce the same object, filled as an episode:

```python
rollout = env.rollout(task, client, model="openai/gpt-4o-mini")
assert isinstance(rollout.trace, Trace)
assert rollout.trace.environment_version == env.version
for decision in rollout.trace.steps:
    if decision.type == "decision":
        print(decision.parsed_action, decision.reward_events)
```

That is the bridge from "what we observed" to "what we train and benchmark on."

A production chat is a degenerate episode: one decision (or a flat `llm` step), no environment state.

## Anatomy

| Field | Meaning |
| --- | --- |
| `trace_id` | Opaque unique id |
| `environment` / `environment_version` / `environment_fingerprint` | Compatibility key for the harness |
| `model` | Policy used for this episode |
| `initial_state` / `final_state` | Environment snapshots at reset and close |
| `steps` | `Decision` (environments) or `LLMCall` / `ToolCall` / `Event` (production) |
| `outcome` | Scores, **reward** (the episode return), labels, feedback |
| `metrics` | Turns, cost, latency, tool counts |
| `terminated` / `truncated` | Natural end vs `max_turns` |
| `schema_version` | Stability marker for partners |

A **Decision** stores `observation`, `model_context` (the request the policy saw), `model_output`, `parsed_action`, `tool_calls`, `reward_events`, and `timestamp`. One decision is one model turn and may include several tool calls.

`trace.transitions()` yields Gymnasium-style `(obs, action, reward, next_obs, terminated, truncated)` tuples.

`trace.returns(gamma=0.9)` is the training signal: discounted return per decision. Reward does not have to be per decision — see [reward injection](environment.md#reward-is-injected-not-assumed-per-decision). Late signals use `trace.credit(...)` on the episode or on one decision (the tweet that later got likes). The trainer, not the environment, gives earlier research actions their share.

## When *not* to capture content

By default enroute drops message content before persistence (`capture_content=False`) to reduce PII risk. Turn it on only when you have a retention policy and preferably a [Redactor](../guides/redact-pii.md). Decision observations and `model_context` are omitted when content is dropped.

## How this connects

Traces land in a [Sink](sink.md). Collections of traces become a [Dataset](dataset.md). Datasets feed [Benchmarks](benchmark.md).

# Trace

**Scenario:** You run a support triage bot. Last Tuesday refund tickets started failing more often. You need to see the exact prompts, which model served them, which fallbacks were tried, and whether the user ultimately resolved the issue.

A **Trace** is that record.

## One sentence

A Trace is the ordered history of LLM calls, tool calls, and events for a single interaction, plus an optional labeled outcome.

## Why production and environments share it

```python
from enroute import Trace, Outcome

trace = Trace(trace_id="...", tags={"surface": "support"})
trace.add_llm(request=req, response=resp, attempts=attempts)
trace.label(scores={"resolved": 1.0}, reward=1.0, feedback="user closed ticket")
```

Environment rollouts produce the same object:

```python
rollout = env.rollout(task, client, model="openai/gpt-4o-mini")
assert isinstance(rollout.trace, Trace)
```

That is the bridge from "what we observed" to "what we train and benchmark on."

## Anatomy

| Field | Meaning |
| --- | --- |
| `trace_id` | Opaque unique id |
| `steps` | Ordered `LLMCall` / `ToolCall` / `Event` |
| `attempts` (on LLMCall) | Every retry and fallback tried |
| `outcome` | Scores, reward, labels, feedback |
| `environment` / `task_id` | Set when produced by a rollout |
| `schema_version` | Stability marker for partners |

## When *not* to capture content

By default enroute drops message content before persistence (`capture_content=False`) to reduce PII risk. Turn it on only when you have a retention policy and preferably a [Redactor](../guides/redact-pii.md).

## How this connects

Traces land in a [Sink](sink.md). Collections of traces become a [Dataset](dataset.md). Datasets feed [Benchmarks](benchmark.md).

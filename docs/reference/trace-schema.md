# Trace JSON schema

Canonical schema version: **1.0.0**

Stability policy:

- Additive fields may appear in `1.x` without bumping the major version.
- Removing or renaming fields requires a major bump (`2.0.0`).
- Partners should ignore unknown fields.

The live schema artifact ships in the package and is mirrored here:

See [`schemas/trace.v1.json`](../schemas/trace.v1.json).

```json
{
  "trace_id": "string",
  "environment": "string|null",
  "environment_version": "string|null",
  "environment_fingerprint": "string|null",
  "task_id": "string|null",
  "model": "string|null",
  "initial_state": {},
  "steps": [
    {
      "type": "decision|llm|tool|event"
    }
  ],
  "final_state": {},
  "outcome": {
    "scores": {"scorer_name": 0.0},
    "reward": 0.0,
    "labels": {},
    "feedback": "string|null"
  },
  "metrics": {},
  "terminated": "boolean|null",
  "truncated": "boolean|null",
  "tags": {},
  "metadata": {},
  "created_at": "ISO-8601",
  "schema_version": "1.0.0"
}
```

A `decision` step is one model turn:

```json
{
  "type": "decision",
  "observation": "string|object|null",
  "model_context": {},
  "model_output": {},
  "parsed_action": [{"name": "tweet", "arguments": {}}],
  "tool_calls": [{"type": "tool", "name": "tweet", "arguments": {}, "result": {}}],
  "reward_events": [{"name": "tweet", "value": 0.05, "reason": null}],
  "timestamp": "ISO-8601"
}
```

`llm`, `tool`, and `event` remain valid for production traces.

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
  "task_id": "string|null",
  "steps": [
    {
      "type": "llm|tool|event"
    }
  ],
  "outcome": {
    "scores": {"scorer_name": 0.0},
    "reward": 0.0,
    "labels": {},
    "feedback": "string|null"
  },
  "tags": {},
  "metadata": {},
  "created_at": "ISO-8601",
  "schema_version": "1.0.0"
}
```

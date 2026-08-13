"""Export enroute traces to a verifiers-style structure.

Examples:
    >>> from enroute.tracing.schema import Trace, Outcome
    >>> from enroute.environments.export.verifiers import to_verifiers_trace
    >>> out = to_verifiers_trace(Trace(trace_id="t", outcome=Outcome(reward=1.0)))
    >>> out["reward"]
    1.0
"""

from __future__ import annotations

from typing import Any

from enroute.tracing.schema import Trace


def to_verifiers_trace(trace: Trace) -> dict[str, Any]:
    """Convert an enroute trace to a simplified verifiers-compatible dict.

    Args:
        trace: Source trace.

    Returns:
        A dictionary with ``messages``, ``reward``, ``task``, and ``extra``.
    """
    messages: list[dict[str, Any]] = []
    for step in trace.steps:
        if step.type == "llm" and step.response is not None:
            resp = step.response
            if hasattr(resp, "message"):
                messages.append(resp.message.model_dump(exclude_none=True))
            elif isinstance(resp, dict):
                choices = resp.get("choices") or []
                if choices:
                    messages.append(choices[0].get("message") or {})
        elif step.type == "tool":
            messages.append(
                {
                    "role": "tool",
                    "name": step.name,
                    "content": step.result,
                    "tool_call_id": step.tool_call_id,
                }
            )
    return {
        "id": trace.trace_id,
        "task": trace.task_id,
        "environment": trace.environment,
        "messages": messages,
        "reward": trace.outcome.reward if trace.outcome else None,
        "scores": trace.outcome.scores if trace.outcome else {},
        "extra": {
            "tags": trace.tags,
            "metadata": trace.metadata,
            "schema_version": trace.schema_version,
        },
    }

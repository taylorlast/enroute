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

from enroute.tracing.schema import Decision, LLMCall, ToolCallStep, Trace
from enroute.types import ChatResponse


def to_verifiers_trace(trace: Trace) -> dict[str, Any]:
    """Convert an enroute trace to a simplified verifiers-compatible dict.

    Prefers :class:`~enroute.tracing.schema.Decision` steps (observation,
    actions, tool results). Falls back to flat ``llm`` / ``tool`` steps.

    Args:
        trace: Source trace.

    Returns:
        A dictionary with ``messages``, ``reward``, ``task``, and ``extra``.
    """
    messages: list[dict[str, Any]] = []
    decisions = [s for s in trace.steps if isinstance(s, Decision)]
    if decisions:
        for decision in decisions:
            if decision.model_output is not None:
                messages.append(_message_from_output(decision.model_output))
            for tool_step in decision.tool_calls:
                messages.append(_tool_message(tool_step))
    else:
        for step in trace.steps:
            if isinstance(step, LLMCall) and step.response is not None:
                messages.append(_message_from_output(step.response))
            elif isinstance(step, ToolCallStep):
                messages.append(_tool_message(step))
    return {
        "id": trace.trace_id,
        "task": trace.task_id,
        "environment": trace.environment,
        "environment_version": trace.environment_version,
        "environment_fingerprint": trace.environment_fingerprint,
        "model": trace.model,
        "messages": messages,
        "reward": trace.outcome.reward if trace.outcome else None,
        "scores": trace.outcome.scores if trace.outcome else {},
        "transitions": [t.model_dump(mode="json") for t in trace.transitions()],
        "returns": trace.returns(),
        "extra": {
            "tags": trace.tags,
            "metadata": trace.metadata,
            "metrics": trace.metrics,
            "terminated": trace.terminated,
            "truncated": trace.truncated,
            "schema_version": trace.schema_version,
        },
    }


def _message_from_output(output: ChatResponse | dict[str, Any]) -> dict[str, Any]:
    if isinstance(output, ChatResponse):
        return output.message.model_dump(exclude_none=True)
    if hasattr(output, "message"):
        dumped = output.message.model_dump(exclude_none=True)
        return dumped if isinstance(dumped, dict) else {}
    choices = output.get("choices") or []
    if choices:
        return choices[0].get("message") or {}
    return {}


def _tool_message(step: ToolCallStep) -> dict[str, Any]:
    return {
        "role": "tool",
        "name": step.name,
        "content": step.result,
        "tool_call_id": step.tool_call_id,
    }

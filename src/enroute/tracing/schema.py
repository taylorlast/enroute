"""Canonical Trace schema shared by production traffic and environments.

Examples:
    >>> from enroute.tracing.schema import Trace, Outcome
    >>> t = Trace(trace_id="abc", outcome=Outcome(reward=1.0))
    >>> t.outcome.reward
    1.0
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from enroute.types import ChatRequest, ChatResponse, Usage


def new_trace_id() -> str:
    """Generate a new opaque trace id.

    Returns:
        A UUID4 hex string.

    Examples:
        >>> len(new_trace_id()) == 32
        True
    """
    return uuid.uuid4().hex


class Attempt(BaseModel):
    """A single provider attempt within an LLM call.

    Attributes:
        model: Model id attempted.
        provider: Provider slug.
        error: Error message if failed.
        latency_ms: Attempt latency.
        status_code: HTTP status code if available.
    """

    model: str
    provider: str
    error: str | None = None
    latency_ms: float | None = None
    status_code: int | None = None


class LLMCall(BaseModel):
    """An LLM invocation step.

    Attributes:
        type: Discriminator; always ``"llm"``.
        request: The chat request (may be redacted).
        response: The chat response (may be redacted).
        provider: Winning provider slug.
        model: Winning model id.
        usage: Token usage.
        cost: USD cost when known.
        latency_ms: End-to-end latency.
        attempts: All attempts including retries and fallbacks.
        error: Top-level error if the call ultimately failed.
    """

    type: Literal["llm"] = "llm"
    request: ChatRequest | dict[str, Any] | None = None
    response: ChatResponse | dict[str, Any] | None = None
    provider: str | None = None
    model: str | None = None
    usage: Usage | None = None
    cost: float | None = None
    latency_ms: float | None = None
    attempts: list[Attempt] = Field(default_factory=list)
    error: str | None = None


class ToolCallStep(BaseModel):
    """A tool invocation step within a rollout.

    Attributes:
        type: Discriminator; always ``"tool"``.
        tool_call_id: Correlates with the model's tool call id.
        name: Tool name.
        arguments: Parsed arguments.
        result: Tool result payload.
        error: Error message if the tool failed.
        latency_ms: Tool execution latency.
    """

    type: Literal["tool"] = "tool"
    tool_call_id: str | None = None
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: str | None = None
    latency_ms: float | None = None


class Event(BaseModel):
    """A free-form event step.

    Attributes:
        type: Discriminator; always ``"event"``.
        name: Event name.
        data: Arbitrary event payload.
    """

    type: Literal["event"] = "event"
    name: str
    data: dict[str, Any] = Field(default_factory=dict)


Step = LLMCall | ToolCallStep | Event


class Outcome(BaseModel):
    """Labeled outcome attached to a trace.

    Attributes:
        scores: Named scorer outputs.
        reward: Scalar reward used for RL / ranking.
        labels: Discrete labels (e.g. intent, success/fail).
        feedback: Free-form human or automated feedback.
    """

    scores: dict[str, float] = Field(default_factory=dict)
    reward: float | None = None
    labels: dict[str, Any] = Field(default_factory=dict)
    feedback: str | None = None


class Trace(BaseModel):
    """A complete record of an LLM interaction or environment rollout.

    Production traffic and environment rollouts emit the same shape so that
    datasets, benchmarks, and a future autorouter can consume either source.

    Attributes:
        trace_id: Opaque unique id.
        environment: Environment name when produced by a rollout.
        environment_version: Environment version string.
        task_id: Task id within an environment.
        steps: Ordered steps (LLM calls, tool calls, events).
        outcome: Optional labeled outcome.
        tags: Free-form tags for filtering.
        metadata: Arbitrary metadata.
        created_at: Creation timestamp (UTC).
        schema_version: Trace schema version for stability.

    Examples:
        >>> t = Trace(trace_id="t1", tags={"env": "prod"})
        >>> t.schema_version
        '1.0.0'
    """

    trace_id: str = Field(default_factory=new_trace_id)
    environment: str | None = None
    environment_version: str | None = None
    task_id: str | None = None
    steps: list[Step] = Field(default_factory=list)
    outcome: Outcome | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = "1.0.0"

    def add_llm(
        self,
        *,
        request: ChatRequest | dict[str, Any] | None,
        response: ChatResponse | None,
        attempts: list[Attempt] | None = None,
        error: str | None = None,
    ) -> LLMCall:
        """Append an LLM call step.

        Args:
            request: Chat request.
            response: Chat response if successful.
            attempts: Attempt records.
            error: Error message if failed.

        Returns:
            The appended :class:`LLMCall`.
        """
        step = LLMCall(
            request=request,
            response=response,
            provider=response.provider if response else None,
            model=response.model
            if response
            else (request.model if isinstance(request, ChatRequest) else None),
            usage=response.usage if response else None,
            cost=response.usage.cost if response else None,
            latency_ms=response.latency_ms if response else None,
            attempts=attempts or [],
            error=error,
        )
        self.steps.append(step)
        return step

    def add_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        tool_call_id: str | None = None,
        result: Any = None,
        error: str | None = None,
        latency_ms: float | None = None,
    ) -> ToolCallStep:
        """Append a tool call step.

        Args:
            name: Tool name.
            arguments: Tool arguments.
            tool_call_id: Optional correlation id.
            result: Tool result.
            error: Error message if failed.
            latency_ms: Execution latency.

        Returns:
            The appended :class:`ToolCallStep`.
        """
        step = ToolCallStep(
            name=name,
            arguments=arguments,
            tool_call_id=tool_call_id,
            result=result,
            error=error,
            latency_ms=latency_ms,
        )
        self.steps.append(step)
        return step

    def label(
        self,
        *,
        scores: dict[str, float] | None = None,
        reward: float | None = None,
        labels: dict[str, Any] | None = None,
        feedback: str | None = None,
    ) -> Outcome:
        """Attach or update the outcome.

        Args:
            scores: Named scorer outputs.
            reward: Scalar reward.
            labels: Discrete labels.
            feedback: Free-form feedback.

        Returns:
            The updated :class:`Outcome`.
        """
        current = self.outcome or Outcome()
        if scores:
            current.scores.update(scores)
        if reward is not None:
            current.reward = reward
        if labels:
            current.labels.update(labels)
        if feedback is not None:
            current.feedback = feedback
        self.outcome = current
        return current

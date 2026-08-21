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
        parent: Name of the enclosing tool when this call was nested.
        children: Tools invoked by this tool (hierarchical actions).
    """

    type: Literal["tool"] = "tool"
    tool_call_id: str | None = None
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: str | None = None
    latency_ms: float | None = None
    parent: str | None = None
    children: list[ToolCallStep] = Field(default_factory=list)


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


class ParsedAction(BaseModel):
    """A structured action the policy chose this turn.

    Attributes:
        name: Tool / action name. Empty or ``"respond"`` means no tool call.
        arguments: Parsed arguments.
    """

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class RewardEvent(BaseModel):
    """A dense reward attached to one decision.

    Attributes:
        name: Reward source (usually the tool name).
        value: Scalar reward.
        reason: Optional human-readable reason.
    """

    name: str
    value: float
    reason: str | None = None


class Decision(BaseModel):
    """One model turn inside an episode: observation → action → reward.

    A single decision may include several tool calls (the model liked a post
    and replied in the same turn).

    Attributes:
        type: Discriminator; always ``"decision"``.
        observation: Environment observation the policy saw this turn.
        model_context: Chat request actually sent (the policy input).
        model_output: Chat response from the model.
        parsed_action: Structured actions extracted from the model output.
        tool_calls: Tool invocations with results.
        reward_events: Per-tool step rewards.
        timestamp: When the decision was recorded (UTC).
    """

    type: Literal["decision"] = "decision"
    observation: Any = None
    model_context: ChatRequest | dict[str, Any] | None = None
    model_output: ChatResponse | dict[str, Any] | None = None
    parsed_action: list[ParsedAction] = Field(default_factory=list)
    tool_calls: list[ToolCallStep] = Field(default_factory=list)
    reward_events: list[RewardEvent] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Transition(BaseModel):
    """Gymnasium-style ``(obs, action, reward, next_obs, done)`` tuple.

    Produced by :meth:`Trace.transitions` for a future trainer.

    Attributes:
        observation: Observation at the start of the decision.
        action: Parsed actions taken.
        reward: Sum of this decision's ``reward_events``.
        next_observation: Observation after the decision (or final state).
        terminated: Whether the episode ended naturally on this step.
        truncated: Whether the episode was cut off on this step.
    """

    observation: Any = None
    action: list[ParsedAction] = Field(default_factory=list)
    reward: float = 0.0
    next_observation: Any = None
    terminated: bool = False
    truncated: bool = False


Step = LLMCall | ToolCallStep | Event | Decision


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
        environment_fingerprint: Hash of tools, instructions, and scorers.
        task_id: Task id within an environment.
        model: Policy / model id used for this episode.
        initial_state: Environment snapshot at reset.
        steps: Ordered steps (decisions, LLM calls, tool calls, events).
        final_state: Environment snapshot at episode close.
        outcome: Optional labeled outcome. ``outcome.reward`` is the return.
        metrics: Episode aggregates (turns, cost, latency, tool counts).
        terminated: Natural end (``env.done()``).
        truncated: Cut off by ``max_turns``.
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
    environment_fingerprint: str | None = None
    task_id: str | None = None
    model: str | None = None
    initial_state: dict[str, Any] | None = None
    steps: list[Step] = Field(default_factory=list)
    final_state: dict[str, Any] | None = None
    outcome: Outcome | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    terminated: bool | None = None
    truncated: bool | None = None
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

    def add_decision(
        self,
        *,
        observation: Any = None,
        model_context: ChatRequest | dict[str, Any] | None = None,
        model_output: ChatResponse | None = None,
        parsed_action: list[ParsedAction] | None = None,
        tool_calls: list[ToolCallStep] | None = None,
        reward_events: list[RewardEvent] | None = None,
    ) -> Decision:
        """Append a decision step for one model turn.

        Args:
            observation: Environment observation the policy saw.
            model_context: Chat request sent to the model.
            model_output: Chat response.
            parsed_action: Structured actions.
            tool_calls: Tool invocations with results.
            reward_events: Per-tool step rewards.

        Returns:
            The appended :class:`Decision`.
        """
        step = Decision(
            observation=observation,
            model_context=model_context,
            model_output=model_output,
            parsed_action=parsed_action or [],
            tool_calls=tool_calls or [],
            reward_events=reward_events or [],
        )
        self.steps.append(step)
        return step

    def transitions(self) -> list[Transition]:
        """Flatten this episode into Gymnasium-style transitions.

        Prefers :class:`Decision` steps. Falls back to grouping flat
        ``llm`` / ``tool`` steps so production traces still export.

        Returns:
            One :class:`Transition` per decision.
        """
        decisions = self.decisions()
        if decisions:
            return self._transitions_from_decisions(decisions)
        return self._transitions_from_flat_steps()

    def decisions(self) -> list[Decision]:
        """Return decision steps in order.

        Returns:
            The :class:`Decision` steps in this episode.
        """
        return [s for s in self.steps if isinstance(s, Decision)]

    def credit(
        self,
        value: float,
        *,
        name: str = "late",
        reason: str | None = None,
        decision_index: int | None = None,
    ) -> RewardEvent | Outcome:
        """Inject reward after the fact — episode-wide or onto one decision.

        Use this when the signal arrives late (likes, a human review, a
        downstream KPI). Credit assignment across earlier actions is
        :meth:`returns`, not the environment's job.

        Args:
            value: Reward to add.
            name: Reward source (``"likes"``, ``"review"``, …).
            reason: Optional human-readable reason.
            decision_index: Which decision to attribute to. ``None`` updates
                ``outcome.reward`` (and ``outcome.scores[name]``). Negative
                indices count from the end.

        Returns:
            The new :class:`RewardEvent` or the updated :class:`Outcome`.

        Raises:
            IndexError: If ``decision_index`` is out of range.
        """
        if decision_index is None:
            current = self.outcome or Outcome()
            current.scores[name] = current.scores.get(name, 0.0) + float(value)
            current.reward = (current.reward or 0.0) + float(value)
            self.outcome = current
            return current
        decisions = self.decisions()
        if not decisions:
            raise IndexError("trace has no decision steps to credit")
        index = decision_index if decision_index >= 0 else len(decisions) + decision_index
        if index < 0 or index >= len(decisions):
            raise IndexError(f"decision_index {decision_index} out of range")
        event = RewardEvent(name=name, value=float(value), reason=reason)
        decisions[index].reward_events.append(event)
        return event

    def decision_rewards(
        self,
        *,
        source: Literal["outcome", "events", "both"] = "outcome",
    ) -> list[float]:
        """Per-decision reward used as ``r_t`` before discounting.

        Args:
            source: ``"outcome"`` — sparse: zeros, then ``outcome.reward`` on
                the last decision (research-then-answer). ``"events"`` — only
                ``reward_events`` (dense shaping and late-attributed credit).
                ``"both"`` — events plus outcome on the last decision.

        Returns:
            One float per decision. Empty if there are no decisions.
        """
        decisions = self.decisions()
        if not decisions:
            return []
        rewards = [sum(event.value for event in d.reward_events) for d in decisions]
        terminal = self.outcome.reward if self.outcome and self.outcome.reward is not None else None
        if source == "events":
            return rewards
        if source == "outcome":
            out = [0.0] * len(decisions)
            if terminal is not None:
                out[-1] = float(terminal)
            return out
        if terminal is not None:
            rewards[-1] += float(terminal)
        return rewards

    def returns(
        self,
        gamma: float = 1.0,
        *,
        source: Literal["outcome", "events", "both"] = "outcome",
    ) -> list[float]:
        """Discounted return ``G_t = r_t + γ G_{t+1}`` for each decision.

        This is how a trainer gives the *research* action credit for a later
        post's likes: the environment does not rewrite history; the trainer
        walks the episode backward.

        Args:
            gamma: Discount factor in ``[0, 1]``.
            source: See :meth:`decision_rewards`.

        Returns:
            One return per decision, same order as :meth:`decisions`.

        Examples:
            >>> t = Trace(outcome=Outcome(reward=1.0))
            >>> t.add_decision(parsed_action=[ParsedAction(name="search")])
            >>> t.add_decision(parsed_action=[ParsedAction(name="answer")])
            >>> t.returns(gamma=0.9)
            [0.9, 1.0]
        """
        rewards = self.decision_rewards(source=source)
        out = [0.0] * len(rewards)
        running = 0.0
        for i in range(len(rewards) - 1, -1, -1):
            running = rewards[i] + gamma * running
            out[i] = running
        return out

    def _transitions_from_decisions(self, decisions: list[Decision]) -> list[Transition]:
        out: list[Transition] = []
        for i, decision in enumerate(decisions):
            is_last = i == len(decisions) - 1
            next_obs = decisions[i + 1].observation if not is_last else self.final_state
            reward = sum(event.value for event in decision.reward_events)
            out.append(
                Transition(
                    observation=decision.observation,
                    action=list(decision.parsed_action),
                    reward=reward,
                    next_observation=next_obs,
                    terminated=bool(self.terminated) if is_last else False,
                    truncated=bool(self.truncated) if is_last else False,
                )
            )
        return out

    def _transitions_from_flat_steps(self) -> list[Transition]:
        groups: list[tuple[Any, list[ParsedAction]]] = []
        current_obs: Any = self.initial_state
        pending_actions: list[ParsedAction] = []
        started = False

        def flush() -> None:
            if not started:
                return
            groups.append((current_obs, list(pending_actions)))

        for step in self.steps:
            if isinstance(step, LLMCall):
                if started:
                    flush()
                    pending_actions = []
                started = True
                if isinstance(step.request, ChatRequest) and step.request.messages:
                    current_obs = step.request.messages[-1].content
                elif isinstance(step.request, dict):
                    msgs = step.request.get("messages") or []
                    if msgs:
                        current_obs = msgs[-1].get("content")
            elif isinstance(step, ToolCallStep):
                started = True
                pending_actions.append(ParsedAction(name=step.name, arguments=step.arguments))
        if started:
            flush()

        out: list[Transition] = []
        for i, (obs, actions) in enumerate(groups):
            is_last = i == len(groups) - 1
            next_obs = groups[i + 1][0] if not is_last else self.final_state
            reward = 0.0
            if is_last and self.outcome and self.outcome.reward is not None:
                reward = float(self.outcome.reward)
            out.append(
                Transition(
                    observation=obs,
                    action=actions,
                    reward=reward,
                    next_observation=next_obs,
                    terminated=bool(self.terminated) if is_last else False,
                    truncated=bool(self.truncated) if is_last else False,
                )
            )
        return out

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

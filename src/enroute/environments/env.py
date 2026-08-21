"""Environment definitions, Gym-shaped episodes, and scoring.

Subclass :class:`Environment` to write a game or work env — name, version,
tools, observations, and rewards live on that class. The default
:meth:`Environment.step` dispatches ``@tool`` methods and records a
decision. Override ``step`` when the action is not a tool call.

Examples:
    >>> from enroute.environments.env import Environment, TaskData
    >>> env = Environment(name="support-triage", version="0.1.0")
    >>> @env.scorer(weight=1.0)
    ... def always_one(rollout):
    ...     return 1.0
    >>> list(env.scorers)[0][0]
    'always_one'
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import time
from collections.abc import Callable, Iterable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field

from enroute.client import Enroute
from enroute.environments.runtime import LocalRuntime, Runtime
from enroute.environments.types import Observation, State
from enroute.tracing.schema import (
    Decision,
    Outcome,
    ParsedAction,
    RewardEvent,
    ToolCallStep,
    Trace,
)
from enroute.types import (
    ChatRequest,
    ChatResponse,
    FunctionDefinition,
    Message,
    Tool,
    ToolCall,
)

_TOOL_ATTR = "__enroute_tool__"
_INSTRUMENTED = "_enroute_instrumented"
_tool_stack: ContextVar[list[ToolCallStep] | None] = ContextVar("enroute_tool_stack", default=None)
_tool_roots: ContextVar[list[ToolCallStep] | None] = ContextVar("enroute_tool_roots", default=None)

ObsT = TypeVar("ObsT", bound=Observation)
StateT = TypeVar("StateT", bound=State)


def tool(fn: Callable[..., Any] | None = None, *, name: str | None = None) -> Any:
    """Mark an :class:`Environment` method as a tool.

    Args:
        fn: Method to register (decorator usage).
        name: Optional explicit tool name. Defaults to the method name.

    Returns:
        The original method (decorator) or a decorator.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        setattr(func, _TOOL_ATTR, name or func.__name__)
        return func

    if fn is not None:
        return decorator(fn)
    return decorator


def _is_env_tool(fn: Callable[..., Any]) -> bool:
    return hasattr(fn, _TOOL_ATTR)


def _iter_env_tools(env_cls: type) -> Iterable[tuple[str, Callable[..., Any]]]:
    seen: set[str] = set()
    for cls in env_cls.__mro__:
        for attr, value in cls.__dict__.items():
            if attr in seen or not callable(value) or not _is_env_tool(value):
                continue
            seen.add(attr)
            yield str(getattr(value, _TOOL_ATTR, attr)), value


class StepResult(BaseModel):
    """Gymnasium-shaped result of :meth:`Environment.step`.

    Unpack as ``obs, reward, terminated, truncated, info``.

    Attributes:
        observation: Next observation from the environment.
        reward: Sum of this decision's step rewards (0 if none).
        terminated: ``True`` when :meth:`Environment.done` reports a natural end.
        truncated: ``True`` when ``max_turns`` was hit.
        info: Extra diagnostics (stop reason, tool errors, …).
    """

    observation: Any = None
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] = Field(default_factory=dict)

    def __iter__(self) -> Any:
        """Yield Gymnasium tuple fields in order."""
        yield self.observation
        yield self.reward
        yield self.terminated
        yield self.truncated
        yield self.info


class TaskData(BaseModel):
    """Seed data for a single task instance.

    Attributes:
        task_id: Stable task identifier.
        input: Primary input payload (often a user message or structured case).
        expected: Optional expected output / label used by scorers.
        metadata: Arbitrary task metadata (include ``seed`` for determinism).
    """

    task_id: str
    input: Any
    expected: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Rollout(BaseModel):
    """Result of running one task through an environment.

    Attributes:
        task: The task that was run.
        trace: Scored episode trace produced by the rollout.
        messages: Final conversation messages.
        response: Final model response, if any.
        env: Environment instance after the episode.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task: TaskData
    trace: Trace
    messages: list[Message] = Field(default_factory=list)
    response: ChatResponse | None = None
    env: Any = None


ScorerFn = Callable[[Rollout], float]
TaskFn = Callable[[], Iterable[TaskData]]


@dataclass
class _Episode:
    task: TaskData
    trace: Trace
    messages: list[Message]
    turn: int = 0
    observation: Any = None
    last_response: ChatResponse | None = None
    closed: bool = False
    stop_reason: str | None = None
    terminated: bool = False
    truncated: bool = False
    tool_errors: list[str] = field(default_factory=list)


class Environment(Generic[ObsT, StateT]):
    """Versioned simulator: instructions, tools, observations, and scorers.

    Subclass this as ``Environment[MyObservation, MyState]``. Put episode
    state on ``self.state`` and decorate actions with :func:`tool`. Drive an
    agent with :meth:`reset` / :meth:`step` or :meth:`rollout` — not
    :meth:`observe`.

    The model is not part of the environment. The default :meth:`step` runs
    ``@tool`` methods on ``self`` and records a decision. Override
    :meth:`step` when the action is not a tool call.

    Class attributes ``name``, ``version``, ``system_prompt``, and
    ``max_turns`` are the defaults; constructor kwargs override them.

    Args:
        name: Environment name.
        version: Semver version string. Bump when tools, observations, or
            reward semantics change.
        system_prompt: Optional instructions prepended to every episode.
        max_turns: Maximum model↔tool turns per episode.
        metadata: Arbitrary environment metadata.

    Examples:
        >>> from enroute.environments.env import Environment, tool
        >>> from enroute.environments.types import Observation, State
        >>> class CounterState(State):
        ...     n: int = 0
        >>> class CounterObservation(Observation):
        ...     n: int = 0
        ...
        ...     def render(self) -> str:
        ...         return f"count={self.n}"
        >>> class CounterEnv(Environment[CounterObservation, CounterState]):
        ...     name = "counter"
        ...     version = "0.1.0"
        ...
        ...     def setup(self, task):
        ...         super().setup(task)
        ...         self.state = CounterState(seed=self.seed, n=0)
        ...
        ...     def observe(self):
        ...         return CounterObservation(n=self.state.n)
        ...
        ...     @tool
        ...     def inc(self, by: int = 1) -> dict:
        ...         '''Increment the counter.'''
        ...         self.state.n += by
        ...         return {"n": self.state.n}
        >>> env = CounterEnv()
        >>> "inc" in env.tool_functions
        True
    """

    name: str = "environment"
    version: str = "0.1.0"
    system_prompt: str | None = None
    max_turns: int = 8

    def __init__(
        self,
        name: str | None = None,
        version: str | None = None,
        *,
        system_prompt: str | None = None,
        max_turns: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        cls = type(self)
        self.name = cls.name if name is None else name
        self.version = cls.version if version is None else version
        self.system_prompt = cls.system_prompt if system_prompt is None else system_prompt
        self.max_turns = cls.max_turns if max_turns is None else max_turns
        self.metadata = metadata or {}
        self.tool_functions: dict[str, Callable[..., Any]] = {}
        self.tool_defs: list[Tool] = []
        self.scorers: list[tuple[str, ScorerFn, float]] = []
        self._tasks_fn: TaskFn | None = None
        self.task: Any = None
        self.seed: int | None = None
        self._state: State = State()
        self._observation: Observation | None = None
        self._episode: _Episode | None = None
        self._register_class_tools()

    @property
    def state(self) -> StateT:
        """Current episode :class:`~enroute.environments.types.State`."""
        return self._state  # type: ignore[return-value]

    @state.setter
    def state(self, value: StateT) -> None:
        self._state = value
        if value.seed is not None:
            self.seed = value.seed

    @property
    def observation(self) -> ObsT:
        """Last observation produced by :meth:`reset` or :meth:`step`.

        Tools should read this cached view. Do not call :meth:`observe` from
        tools or from an agent loop.
        """
        if self._observation is None:
            return self.observe()
        return self._observation  # type: ignore[return-value]

    def setup(self, task: TaskData) -> None:
        """Seed ``self.state`` from ``task``. Override in subclasses.

        This is an author hook. Callers driving an agent use :meth:`reset`.

        Args:
            task: Task data with optional ``metadata["seed"]``.
        """
        self.task = task
        raw = None
        metadata = getattr(task, "metadata", None)
        if isinstance(metadata, dict):
            raw = metadata.get("seed")
        self.seed = int(raw) if raw is not None else None
        self.state = State(seed=self.seed)  # type: ignore[assignment]

    def observe(self) -> ObsT:
        """Build the current observation from ``self.state``.

        Author hook — :meth:`reset` and :meth:`step` call this. Drive an
        agent with those methods (or :meth:`rollout`), not ``observe``.

        Returns:
            An :class:`~enroute.environments.types.Observation`. Empty
            ``text`` on the first turn falls back to the task input.
        """
        return Observation()  # type: ignore[return-value]

    def done(self) -> bool:
        """Return whether the episode has reached a natural end.

        Returns:
            ``True`` to terminate. Default is ``False`` (run until truncated).
        """
        return False

    def snapshot(self) -> dict[str, Any] | None:
        """Return a JSON snapshot of :attr:`state` for traces.

        Returns:
            ``state.model_dump()``, or ``None`` when state is empty.
        """
        data = self.state.model_dump()
        if not data.get("metadata"):
            data.pop("metadata", None)
        if data == {"seed": None} or data == {}:
            return None
        return data

    def step_reward(self, tool_name: str, result: Any) -> float | None:
        """Optional dense reward after one tool call.

        Args:
            tool_name: Tool that just ran.
            result: Tool result payload.

        Returns:
            A float reward, or ``None`` to record nothing.
        """
        return None

    def spawn(self) -> Environment[ObsT, StateT]:
        """Return a fresh instance for a concurrent episode.

        Copies tools, scorers, and tasks. Game state starts empty — call
        :meth:`reset` on the result. One instance is one episode; Benchmark
        uses this so worker threads do not share ``self``.
        """
        other = type(self)(
            name=self.name,
            version=self.version,
            system_prompt=self.system_prompt,
            max_turns=self.max_turns,
            metadata=dict(self.metadata),
        )
        for tool_name, func in self.tool_functions.items():
            if tool_name not in other.tool_functions:
                other._add_tool(tool_name, func)
        other.scorers = list(self.scorers)
        other._tasks_fn = self._tasks_fn
        return other

    def tool(self, fn: Callable[..., Any] | None = None, *, name: str | None = None) -> Any:
        """Register a Python function as a tool.

        Args:
            fn: Function to register (decorator usage).
            name: Optional explicit tool name.

        Returns:
            The original function (decorator) or a decorator.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._add_tool(name or func.__name__, func)
            return func

        if fn is not None:
            return decorator(fn)
        return decorator

    def scorer(
        self,
        fn: ScorerFn | None = None,
        *,
        weight: float = 1.0,
        name: str | None = None,
    ) -> Any:
        """Register a scorer.

        Args:
            fn: Scorer callable ``(rollout) -> float``.
            weight: Weight in the aggregate reward.
            name: Optional scorer name.

        Returns:
            The original function (decorator) or a decorator.
        """

        def decorator(func: ScorerFn) -> ScorerFn:
            self.scorers.append((name or func.__name__, func, weight))
            return func

        if fn is not None:
            return decorator(fn)
        return decorator

    def tasks(self, fn: TaskFn) -> TaskFn:
        """Register a tasks provider.

        Args:
            fn: Callable returning an iterable of :class:`TaskData`.

        Returns:
            The original function.
        """
        self._tasks_fn = fn
        return fn

    def iter_tasks(self) -> Iterator[TaskData]:
        """Yield all tasks.

        Yields:
            Task data instances.

        Raises:
            RuntimeError: If no tasks provider was registered.
        """
        if self._tasks_fn is None:
            raise RuntimeError(f"environment '{self.name}' has no tasks provider")
        yield from self._tasks_fn()

    def fingerprint(self) -> str:
        """Hash the action surface and scoring contract.

        Two episodes with the same fingerprint saw the same tools,
        instructions, and scorer weights.

        Returns:
            Hex SHA256 digest.
        """
        obs_name, state_name = self._contract_names()
        payload = {
            "name": self.name,
            "version": self.version,
            "instructions": self.system_prompt,
            "observation": obs_name,
            "state": state_name,
            "tools": [t.model_dump(mode="json") for t in self.tool_defs],
            "scorers": [(name, weight) for name, _fn, weight in self.scorers],
        }
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def reset(
        self,
        task: TaskData,
        *,
        model: str | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Start an episode. Gymnasium-shaped: ``obs, info = env.reset(task)``.

        Args:
            task: Task that seeds this instance via :meth:`setup`.
            model: Optional policy id recorded on the episode trace.

        Returns:
            ``(observation, info)``.
        """
        self.setup(task)
        observation = self._initial_observation(task)
        self._observation = observation if isinstance(observation, Observation) else None
        messages: list[Message] = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))
        messages.append(Message(role="user", content=_as_text(observation)))

        trace = Trace(
            environment=self.name,
            environment_version=self.version,
            environment_fingerprint=self.fingerprint(),
            task_id=task.task_id,
            model=model,
            initial_state=self._safe_snapshot(),
            metadata={"task": task.model_dump(), **self.metadata},
            tags={"environment": self.name},
        )
        self._episode = _Episode(
            task=task,
            trace=trace,
            messages=messages,
            observation=observation,
        )
        info = {
            "task_id": task.task_id,
            "environment": self.name,
            "environment_version": self.version,
            "environment_fingerprint": trace.environment_fingerprint,
        }
        return observation, info

    def messages(self) -> list[Message]:
        """Return the current episode conversation for the policy.

        After :meth:`reset`, this is instructions plus the first observation.
        After :meth:`step`, tool results are appended so the next ``chat``
        sees the latest observation.

        Returns:
            A copy of the open episode's messages.

        Raises:
            RuntimeError: If :meth:`reset` has not been called.
        """
        return list(self._require_episode().messages)

    def step(
        self,
        action: Any = None,
        *,
        request: ChatRequest | dict[str, Any] | None = None,
        response: ChatResponse | None = None,
        runtime: Runtime | None = None,
    ) -> StepResult:
        """Apply one policy decision. Unpack as Gymnasium's 5-tuple.

        The default implementation treats ``action`` as tool calls on
        ``self``, records a :class:`~enroute.tracing.schema.Decision`, and
        returns the next observation / :meth:`done`. Override this method
        when the action is not a tool call. Call :meth:`record_decision`
        and :meth:`finish_turn` to keep tracing.

        Args:
            action: Parsed actions, tool calls, a :class:`ChatResponse`, or
                ``None`` to take tool calls from ``response``.
            request: Chat request the policy saw (stored on the decision).
            response: Chat response for this turn.
            runtime: Tool runtime; defaults to :class:`LocalRuntime`.

        Returns:
            :class:`StepResult`.

        Raises:
            RuntimeError: If :meth:`reset` has not been called.
        """
        episode = self._require_episode()
        runtime = runtime or LocalRuntime(self.tool_functions)
        self._append_assistant(response)
        parsed, tool_calls_in = _normalize_action(action, response)
        observation = episode.observation

        tool_steps: list[ToolCallStep] = []
        reward_events: list[RewardEvent] = []
        episode.tool_errors = []

        for parsed_action, tool_call_id in zip(parsed, tool_calls_in, strict=False):
            if not parsed_action.name or parsed_action.name == "respond":
                continue
            roots_token = _tool_roots.set([])
            try:
                result, latency_ms, error = self._invoke_tool(
                    runtime, parsed_action.name, parsed_action.arguments
                )
                roots = list(_tool_roots.get() or [])
            finally:
                _tool_roots.reset(roots_token)
            tool_step = roots[0] if roots else None
            if tool_step is None:
                tool_step = ToolCallStep(
                    name=parsed_action.name,
                    arguments=parsed_action.arguments,
                    result=result,
                    error=error,
                    latency_ms=latency_ms,
                )
            tool_step.tool_call_id = tool_call_id
            tool_steps.append(tool_step)
            if error:
                episode.tool_errors.append(error)
            episode.messages.append(
                Message(
                    role="tool",
                    tool_call_id=tool_call_id,
                    name=parsed_action.name,
                    content=json.dumps(result) if not isinstance(result, str) else result,
                )
            )
            value = self.step_reward(parsed_action.name, result)
            if value is not None:
                reward_events.append(RewardEvent(name=parsed_action.name, value=float(value)))

        self.record_decision(
            observation=observation,
            model_context=request,
            model_output=response,
            parsed_action=parsed,
            tool_calls=tool_steps,
            reward_events=reward_events,
        )
        stop_reason = None
        if not parsed or all(a.name in {"", "respond"} for a in parsed):
            stop_reason = "no_tool_calls"
        return self.finish_turn(reward_events, stop_reason=stop_reason)

    def record_decision(
        self,
        *,
        observation: Any = None,
        model_context: ChatRequest | dict[str, Any] | None = None,
        model_output: ChatResponse | None = None,
        parsed_action: list[ParsedAction] | None = None,
        tool_calls: list[ToolCallStep] | None = None,
        reward_events: list[RewardEvent] | None = None,
    ) -> None:
        """Append a decision to the open episode trace.

        Used by the default :meth:`step`. Call this from a custom ``step``
        so the episode still looks like every other environment.

        Args:
            observation: Observation the policy saw this turn.
            model_context: Chat request, if any.
            model_output: Chat response, if any.
            parsed_action: Actions applied this turn.
            tool_calls: Tool results this turn.
            reward_events: Step rewards this turn.
        """
        episode = self._require_episode()
        episode.trace.add_decision(
            observation=_serialize_observation(observation),
            model_context=model_context,
            model_output=model_output,
            parsed_action=parsed_action or [],
            tool_calls=tool_calls or [],
            reward_events=reward_events or [],
        )
        if model_output is not None:
            episode.last_response = model_output

    def finish_turn(
        self,
        reward_events: list[RewardEvent] | None = None,
        *,
        stop_reason: str | None = None,
    ) -> StepResult:
        """Advance the episode after an action and return a Gym 5-tuple.

        Increments the turn, rebuilds the observation, reads :meth:`done`,
        and marks truncation at ``max_turns``.

        Args:
            reward_events: Events whose values sum to this step's reward.
            stop_reason: Override (``no_tool_calls``, ``terminated``, …).

        Returns:
            :class:`StepResult`.
        """
        episode = self._require_episode()
        events = reward_events or []
        episode.turn += 1
        next_obs = self.observe()
        if _is_empty_observation(next_obs):
            next_obs = episode.observation
        self._observation = next_obs if isinstance(next_obs, Observation) else None
        episode.observation = next_obs
        terminated = self.done()
        truncated = episode.turn >= self.max_turns and not terminated
        if stop_reason is None:
            if terminated:
                stop_reason = "terminated"
            elif truncated:
                stop_reason = "truncated"
        elif terminated:
            stop_reason = "terminated"
        elif truncated and stop_reason != "no_tool_calls":
            stop_reason = "truncated"
        episode.terminated = terminated
        episode.truncated = truncated
        episode.stop_reason = stop_reason
        return StepResult(
            observation=next_obs,
            reward=sum(event.value for event in events),
            terminated=terminated,
            truncated=truncated,
            info={
                "turn": episode.turn,
                "stop_reason": stop_reason,
                "tool_errors": list(episode.tool_errors),
            },
        )

    def close_episode(
        self,
        *,
        response: ChatResponse | None = None,
        client: Enroute | None = None,
    ) -> Rollout:
        """Score the open episode.

        Args:
            response: Final model response, if not already stored.
            client: If given, persist the episode trace on its writer.

        Returns:
            A scored :class:`Rollout`.

        Raises:
            RuntimeError: If :meth:`reset` has not been called.
        """
        episode = self._require_episode()
        final_response = response or episode.last_response
        rollout = Rollout(
            task=episode.task,
            trace=episode.trace,
            messages=episode.messages,
            response=final_response,
            env=self,
        )
        scores, reward = self._score(rollout)
        episode.trace.outcome = Outcome(scores=scores, reward=reward if self.scorers else None)
        episode.trace.final_state = self._safe_snapshot()
        episode.trace.terminated = episode.terminated
        episode.trace.truncated = episode.truncated
        episode.trace.metrics = _episode_metrics(episode)
        episode.closed = True
        if client is not None:
            client.writer.record(episode.trace)
        return rollout

    def rollout(
        self,
        task: TaskData,
        client: Enroute,
        *,
        model: str,
        models: list[str] | None = None,
        runtime: Runtime | None = None,
        temperature: float | None = None,
    ) -> Rollout:
        """Convenience: reset, LLM policy loop, :meth:`step` until done, score.

        Prefer :meth:`reset` / :meth:`step` when writing a training loop.
        This helper is what :class:`~enroute.benchmarks.runner.Benchmark` uses.

        Args:
            task: Task to run.
            client: Enroute client used for model calls.
            model: Primary model id.
            models: Optional fallback chain.
            runtime: Tool runtime; defaults to an in-process :class:`LocalRuntime`.
            temperature: Optional sampling temperature.

        Returns:
            A :class:`Rollout` containing a scored :class:`~enroute.tracing.schema.Trace`.
        """
        runtime = runtime or LocalRuntime(self.tool_functions)
        self.reset(task, model=model)
        episode = self._require_episode()
        final_response: ChatResponse | None = None

        for _ in range(self.max_turns):
            request = ChatRequest(
                model=model,
                messages=list(episode.messages),
                models=models,
                tools=self.tool_defs or None,
                temperature=temperature,
            )
            response = client.chat(
                model=model,
                messages=episode.messages,
                models=models,
                tools=self.tool_defs or None,
                temperature=temperature,
                tags={"environment": self.name, "task_id": task.task_id},
            )
            final_response = response
            result = self.step(
                response.message.tool_calls,
                request=request,
                response=response,
                runtime=runtime,
            )
            stop = result.info.get("stop_reason")
            if result.terminated or result.truncated or stop == "no_tool_calls":
                break

        return self.close_episode(response=final_response, client=client)

    def _require_episode(self) -> _Episode:
        episode = self._episode
        if episode is None or episode.closed:
            raise RuntimeError("no episode in progress; call Environment.reset() first")
        return episode

    def _append_assistant(self, response: ChatResponse | None) -> None:
        if response is None:
            return
        episode = self._require_episode()
        assistant = response.message
        if episode.messages and episode.messages[-1] is assistant:
            return
        last = episode.messages[-1] if episode.messages else None
        already = (
            last is not None
            and last.role == "assistant"
            and last.tool_calls == assistant.tool_calls
            and last.content == assistant.content
        )
        if not already:
            episode.messages.append(assistant)

    def _score(self, rollout: Rollout) -> tuple[dict[str, float], float]:
        scores: dict[str, float] = {}
        reward = 0.0
        total_weight = 0.0
        for scorer_name, fn, weight in self.scorers:
            value = float(fn(rollout))
            scores[scorer_name] = value
            reward += value * weight
            total_weight += weight
        if total_weight > 0:
            reward /= total_weight
        return scores, reward

    def _register_class_tools(self) -> None:
        for tool_name, func in _iter_env_tools(type(self)):
            bound = getattr(self, func.__name__)
            self._add_tool(tool_name, bound, schema_from=func, bind_method=func.__name__)

    def _add_tool(
        self,
        tool_name: str,
        func: Callable[..., Any],
        *,
        schema_from: Callable[..., Any] | None = None,
        bind_method: str | None = None,
    ) -> None:
        source = schema_from or func
        wrapped = self._instrument_tool(tool_name, func)
        self.tool_functions[tool_name] = wrapped
        if bind_method:
            setattr(self, bind_method, wrapped)
        self.tool_defs.append(
            Tool(
                function=FunctionDefinition(
                    name=tool_name,
                    description=(inspect.getdoc(source) or "").strip() or None,
                    parameters=_function_schema(source),
                )
            )
        )

    def _instrument_tool(self, tool_name: str, func: Callable[..., Any]) -> Callable[..., Any]:
        if getattr(func, _INSTRUMENTED, False):
            return func

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            recorded = _call_arguments(func, args, kwargs)
            step = ToolCallStep(name=tool_name, arguments=recorded)
            stack = list(_tool_stack.get() or [])
            if stack:
                step.parent = stack[-1].name
                stack[-1].children.append(step)
            else:
                roots = list(_tool_roots.get() or [])
                roots.append(step)
                _tool_roots.set(roots)
            stack.append(step)
            token = _tool_stack.set(stack)
            started = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                step.result = result
                step.latency_ms = (time.perf_counter() - started) * 1000
                return result
            except Exception as exc:
                step.error = str(exc)
                step.result = {"error": str(exc)}
                step.latency_ms = (time.perf_counter() - started) * 1000
                raise
            finally:
                _tool_stack.reset(token)

        setattr(wrapper, _INSTRUMENTED, True)
        return wrapper

    def _contract_names(self) -> tuple[str, str]:
        for base in getattr(type(self), "__orig_bases__", ()):
            if get_origin(base) is Environment:
                args = get_args(base)
                if len(args) >= 2:
                    return args[0].__name__, args[1].__name__
        return Observation.__name__, State.__name__

    def _initial_observation(self, task: TaskData) -> Any:
        observed: Any = self.observe()
        if _is_empty_observation(observed):
            observed = Observation(text=_as_text(task.input))
        if isinstance(observed, Observation):
            self._observation = observed
        return observed

    def _safe_snapshot(self) -> dict[str, Any] | None:
        try:
            value = self.snapshot()
        except Exception:  # noqa: BLE001
            return None
        if value is None:
            return None
        return value if isinstance(value, dict) else {"value": value}

    def _invoke_tool(
        self,
        runtime: Runtime,
        name: str,
        arguments: dict[str, Any],
    ) -> tuple[Any, float, str | None]:
        started = time.perf_counter()
        try:
            if isinstance(runtime, LocalRuntime):
                result, latency_ms = runtime.call_timed(name, arguments)
            else:
                result = runtime.call(name, arguments)
                latency_ms = (time.perf_counter() - started) * 1000
            return result, latency_ms, None
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}, (time.perf_counter() - started) * 1000, str(exc)


def _normalize_action(
    action: Any,
    response: ChatResponse | None,
) -> tuple[list[ParsedAction], list[str | None]]:
    if action is None and response is not None:
        action = response.message.tool_calls
    if isinstance(action, ChatResponse):
        action = action.message.tool_calls
    if not action:
        return [ParsedAction(name="respond", arguments={})], [None]
    parsed: list[ParsedAction] = []
    ids: list[str | None] = []
    if isinstance(action, ParsedAction):
        return [action], [None]
    if isinstance(action, ToolCall):
        parsed_one = ParsedAction(
            name=action.function.name,
            arguments=_parse_args(action.function.arguments),
        )
        return [parsed_one], [action.id]
    for item in action:
        if isinstance(item, ParsedAction):
            parsed.append(item)
            ids.append(None)
        elif isinstance(item, ToolCall):
            parsed.append(
                ParsedAction(
                    name=item.function.name,
                    arguments=_parse_args(item.function.arguments),
                )
            )
            ids.append(item.id)
        elif isinstance(item, dict):
            parsed.append(
                ParsedAction(
                    name=str(item.get("name") or "respond"),
                    arguments=dict(item.get("arguments") or {}),
                )
            )
            ids.append(item.get("id"))
        else:
            parsed.append(ParsedAction(name=str(item), arguments={}))
            ids.append(None)
    return parsed, ids


def _episode_metrics(episode: _Episode) -> dict[str, Any]:
    decisions = [s for s in episode.trace.steps if isinstance(s, Decision)]
    tool_count = sum(len(d.tool_calls) for d in decisions)
    cost = 0.0
    latency = 0.0
    for decision in decisions:
        output = decision.model_output
        if isinstance(output, ChatResponse):
            if output.usage and output.usage.cost is not None:
                cost += output.usage.cost
            if output.latency_ms is not None:
                latency += output.latency_ms
        elif isinstance(output, dict):
            usage = output.get("usage") or {}
            if usage.get("cost") is not None:
                cost += float(usage["cost"])
            if output.get("latency_ms") is not None:
                latency += float(output["latency_ms"])
    return {
        "turns": episode.turn,
        "decisions": len(decisions),
        "tool_calls": tool_count,
        "cost": cost,
        "latency_ms": latency,
        "stop_reason": episode.stop_reason,
    }


def _call_arguments(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    recorded = dict(kwargs)
    if not args:
        return recorded
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return recorded
    names = [p.name for p in sig.parameters.values() if p.name not in {"self", "cls"}]
    for name, value in zip(names, args, strict=False):
        recorded.setdefault(name, value)
    return recorded


def _is_empty_observation(value: Any) -> bool:
    if value is None or value == "":
        return True
    if isinstance(value, Observation):
        return not value.render()
    return False


def _serialize_observation(value: Any) -> Any:
    if isinstance(value, Observation):
        return value.model_dump()
    return value


def _as_text(value: Any) -> str:
    if isinstance(value, Observation):
        return value.render()
    if isinstance(value, str):
        return value
    render = getattr(value, "render", None)
    if callable(render):
        return str(render())
    if hasattr(value, "model_dump"):
        return json.dumps(value.model_dump())
    return json.dumps(value)


def _parse_args(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
    if isinstance(value, dict):
        return value
    return {"value": value}


def _function_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []
    hints = getattr(fn, "__annotations__", {})
    for param in sig.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.name in {"self", "cls"}:
            continue
        ann = hints.get(param.name, str)
        properties[param.name] = _annotation_to_json_schema(ann)
        if param.default is inspect.Parameter.empty:
            required.append(param.name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _annotation_to_json_schema(ann: Any) -> dict[str, Any]:
    if ann is int:
        return {"type": "integer"}
    if ann is float:
        return {"type": "number"}
    if ann is bool:
        return {"type": "boolean"}
    if ann is str:
        return {"type": "string"}
    origin = getattr(ann, "__origin__", None)
    if origin is list:
        return {"type": "array"}
    if origin is dict:
        return {"type": "object"}
    return {"type": "string"}

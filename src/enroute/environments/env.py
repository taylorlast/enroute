"""Environment definitions, rollouts, and scoring.

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

import inspect
import json
import time
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from pydantic import BaseModel, Field

from enroute.client import Enroute
from enroute.environments.runtime import LocalRuntime, Runtime
from enroute.tracing.schema import Outcome, Trace
from enroute.types import (
    ChatResponse,
    FunctionDefinition,
    Message,
    Tool,
    ToolCall,
)


class TaskData(BaseModel):
    """Seed data for a single task instance.

    Attributes:
        task_id: Stable task identifier.
        input: Primary input payload (often a user message or structured case).
        expected: Optional expected output / label used by scorers.
        metadata: Arbitrary task metadata.
    """

    task_id: str
    input: Any
    expected: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Rollout(BaseModel):
    """Result of running one task through an environment.

    Attributes:
        task: The task that was run.
        trace: Scored trace produced by the rollout.
        messages: Final conversation messages.
        response: Final model response, if any.
    """

    task: TaskData
    trace: Trace
    messages: list[Message] = Field(default_factory=list)
    response: ChatResponse | None = None


ScorerFn = Callable[[Rollout], float]
TaskFn = Callable[[], Iterable[TaskData]]


class Environment:
    """An RL-style harness: tasks + tools + scorers that emit traces.

    Args:
        name: Environment name.
        version: Semver-ish version string.
        system_prompt: Optional system prompt prepended to every rollout.
        max_turns: Maximum model↔tool turns per rollout.
        metadata: Arbitrary environment metadata.

    Examples:
        >>> env = Environment(name="demo", version="0.1.0")
        >>> @env.tool
        ... def ping() -> str:
        ...     '''Return pong.'''
        ...     return "pong"
        >>> "ping" in env.tool_functions
        True
    """

    def __init__(
        self,
        name: str,
        version: str = "0.1.0",
        *,
        system_prompt: str | None = None,
        max_turns: int = 8,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.version = version
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.metadata = metadata or {}
        self.tool_functions: dict[str, Callable[..., Any]] = {}
        self.tool_defs: list[Tool] = []
        self.scorers: list[tuple[str, ScorerFn, float]] = []
        self._tasks_fn: TaskFn | None = None

    def tool(self, fn: Callable[..., Any] | None = None, *, name: str | None = None) -> Any:
        """Register a Python function as a tool.

        Args:
            fn: Function to register (decorator usage).
            name: Optional explicit tool name.

        Returns:
            The original function (decorator) or a decorator.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or func.__name__
            self.tool_functions[tool_name] = func
            schema = _function_schema(func)
            self.tool_defs.append(
                Tool(
                    function=FunctionDefinition(
                        name=tool_name,
                        description=(inspect.getdoc(func) or "").strip() or None,
                        parameters=schema,
                    )
                )
            )
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
        """Run one task and return a scored rollout.

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
        messages: list[Message] = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))
        user_content = task.input if isinstance(task.input, str) else json.dumps(task.input)
        messages.append(Message(role="user", content=user_content))

        trace = Trace(
            environment=self.name,
            environment_version=self.version,
            task_id=task.task_id,
            metadata={"task": task.model_dump(), **self.metadata},
            tags={"environment": self.name},
        )
        final_response: ChatResponse | None = None

        for _ in range(self.max_turns):
            response = client.chat(
                model=model,
                messages=messages,
                models=models,
                tools=self.tool_defs or None,
                temperature=temperature,
                tags={"environment": self.name, "task_id": task.task_id},
            )
            final_response = response
            # Pull the LLM step from the client's last write is awkward; record here too
            # for environment-scoped completeness. The client also records a production trace.
            trace.add_llm(request=None, response=response)
            assistant = response.message
            messages.append(assistant)

            if not assistant.tool_calls:
                break

            for tc in assistant.tool_calls:
                result, latency_ms, error = self._invoke_tool(runtime, tc)
                trace.add_tool(
                    tc.function.name,
                    _parse_args(tc.function.arguments),
                    tool_call_id=tc.id,
                    result=result,
                    error=error,
                    latency_ms=latency_ms,
                )
                messages.append(
                    Message(
                        role="tool",
                        tool_call_id=tc.id,
                        name=tc.function.name,
                        content=json.dumps(result) if not isinstance(result, str) else result,
                    )
                )

        rollout = Rollout(task=task, trace=trace, messages=messages, response=final_response)
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
        trace.outcome = Outcome(scores=scores, reward=reward if self.scorers else None)
        client.writer.record(trace)
        return rollout

    def _invoke_tool(
        self,
        runtime: Runtime,
        tc: ToolCall,
    ) -> tuple[Any, float, str | None]:
        args = _parse_args(tc.function.arguments)
        started = time.perf_counter()
        try:
            if isinstance(runtime, LocalRuntime):
                result, latency_ms = runtime.call_timed(tc.function.name, args)
            else:
                result = runtime.call(tc.function.name, args)
                latency_ms = (time.perf_counter() - started) * 1000
            return result, latency_ms, None
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}, (time.perf_counter() - started) * 1000, str(exc)


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

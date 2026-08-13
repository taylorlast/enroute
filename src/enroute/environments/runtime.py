"""Runtime protocol for executing tools during a rollout.

Examples:
    >>> from enroute.environments.runtime import LocalRuntime
    >>> rt = LocalRuntime({"add": lambda a, b: a + b})
    >>> rt.call("add", {"a": 1, "b": 2})
    3
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Runtime(Protocol):
    """Execution surface for environment tools."""

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        """Invoke a tool by name.

        Args:
            name: Tool name.
            arguments: Keyword arguments for the tool.

        Returns:
            Tool result.
        """
        ...


class LocalRuntime:
    """In-process runtime that calls registered Python callables.

    Args:
        tools: Mapping of tool name to callable.
    """

    def __init__(self, tools: dict[str, Callable[..., Any]] | None = None) -> None:
        self.tools = dict(tools or {})

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        """Register a tool.

        Args:
            name: Tool name.
            fn: Python callable.
        """
        self.tools[name] = fn

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call a registered tool.

        Args:
            name: Tool name.
            arguments: Keyword arguments.

        Returns:
            Tool result.

        Raises:
            KeyError: If the tool is unknown.
        """
        if name not in self.tools:
            raise KeyError(f"unknown tool: {name}")
        return self.tools[name](**arguments)

    def call_timed(self, name: str, arguments: dict[str, Any]) -> tuple[Any, float]:
        """Call a tool and return ``(result, latency_ms)``.

        Args:
            name: Tool name.
            arguments: Keyword arguments.

        Returns:
            Tuple of result and latency in milliseconds.
        """
        started = time.perf_counter()
        result = self.call(name, arguments)
        return result, (time.perf_counter() - started) * 1000

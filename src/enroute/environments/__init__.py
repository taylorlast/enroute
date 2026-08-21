"""Environments: RL-style harnesses that generate scored traces.

Subclass :class:`Environment` for a versioned simulator (tools, observations,
scorers). Its output is the same :class:`~enroute.tracing.schema.Trace` used
for production traffic.

Examples:
    >>> from enroute.environments import Environment, TaskData
    >>> env = Environment(name="demo", version="0.1.0")
    >>> env.name
    'demo'
"""

from __future__ import annotations

from enroute.environments.dataset import Dataset
from enroute.environments.env import Environment, Rollout, StepResult, TaskData, tool
from enroute.environments.runtime import LocalRuntime, Runtime
from enroute.environments.types import Observation, State

__all__ = [
    "Dataset",
    "Environment",
    "LocalRuntime",
    "Observation",
    "Rollout",
    "Runtime",
    "State",
    "StepResult",
    "TaskData",
    "tool",
]

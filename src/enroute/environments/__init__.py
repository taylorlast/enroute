"""Environments: RL-style harnesses that generate scored traces.

An environment owns tasks, the tool/action surface, and scorers. Its output is
the same :class:`~enroute.tracing.schema.Trace` used for production traffic.

Examples:
    >>> from enroute.environments import Environment, TaskData
    >>> env = Environment(name="demo", version="0.1.0")
    >>> env.name
    'demo'
"""

from __future__ import annotations

from enroute.environments.dataset import Dataset
from enroute.environments.env import Environment, Rollout, TaskData
from enroute.environments.runtime import LocalRuntime, Runtime

__all__ = [
    "Dataset",
    "Environment",
    "LocalRuntime",
    "Rollout",
    "Runtime",
    "TaskData",
]

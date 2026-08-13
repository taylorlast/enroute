"""enroute: unified LLM routing with first-class traces, environments, and benchmarks.

Examples:
    >>> from enroute import __version__
    >>> isinstance(__version__, str)
    True
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from enroute.benchmarks import Benchmark, Report
from enroute.catalog import ModelCatalog, ModelSpec, estimate_cost
from enroute.client import Enroute
from enroute.environments import Dataset, Environment, TaskData
from enroute.errors import (
    AuthenticationError,
    BudgetExceededError,
    ConfigurationError,
    ContentFilterError,
    ContextLengthError,
    EnrouteError,
    InvalidRequestError,
    NotFoundError,
    ProviderUnavailable,
    RateLimitError,
    TimeoutError,
    is_retryable,
)
from enroute.tracing import (
    JSONLSink,
    Outcome,
    Redactor,
    Sampler,
    SQLiteSink,
    Trace,
    TraceWriter,
)
from enroute.types import ChatRequest, ChatResponse, Message, Tool, Usage

try:
    __version__ = version("enroute")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    "AuthenticationError",
    "Benchmark",
    "BudgetExceededError",
    "ChatRequest",
    "ChatResponse",
    "ConfigurationError",
    "ContentFilterError",
    "ContextLengthError",
    "Dataset",
    "Enroute",
    "EnrouteError",
    "Environment",
    "InvalidRequestError",
    "JSONLSink",
    "Message",
    "ModelCatalog",
    "ModelSpec",
    "NotFoundError",
    "Outcome",
    "ProviderUnavailable",
    "RateLimitError",
    "Redactor",
    "Report",
    "SQLiteSink",
    "Sampler",
    "TaskData",
    "TimeoutError",
    "Tool",
    "Trace",
    "TraceWriter",
    "Usage",
    "__version__",
    "estimate_cost",
    "is_retryable",
]

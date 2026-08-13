"""Runtime configuration for enroute clients and sinks.

Examples:
    >>> from enroute.config import Settings
    >>> s = Settings()
    >>> s.gateway_base_url
    'https://api.enroute.dev/v1'
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Process-wide defaults for enroute.

    Attributes:
        gateway_base_url: Base URL of the hosted enroute gateway. Used when
            constructing a client with a single ``api_key``.
        default_timeout_s: Default HTTP timeout for provider calls, in seconds.
        max_retries: Default number of retries for retryable provider errors.
        trace_dir: Default directory for local JSONL/SQLite sinks.
        capture_content: Whether to record full prompt/response content in traces.
            Disabled by default to reduce PII exposure risk.
    """

    gateway_base_url: str = "https://api.enroute.dev/v1"
    default_timeout_s: float = 60.0
    max_retries: int = 2
    trace_dir: Path = Field(default_factory=lambda: Path(".enroute"))
    capture_content: bool = False


DEFAULT_SETTINGS = Settings()

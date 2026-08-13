"""Classified exception hierarchy for enroute.

Provider adapters map HTTP status codes and provider-specific error payloads
into these types so routers can make retry and fallback decisions without
parsing raw response bodies.

Examples:
    >>> from enroute.errors import RateLimitError, is_retryable
    >>> err = RateLimitError("too many requests", provider="openai", status_code=429)
    >>> is_retryable(err)
    True
"""

from __future__ import annotations

from typing import Any


class EnrouteError(Exception):
    """Base class for all enroute errors.

    Args:
        message: Human-readable description of the failure.
        provider: Provider slug associated with the failure, if any.
        status_code: HTTP status code, if the failure came from an HTTP response.
        body: Raw response body or structured error payload, if available.
        model: Model id that was being called, if known.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int | None = None,
        body: Any = None,
        model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.status_code = status_code
        self.body = body
        self.model = model

    def __str__(self) -> str:  # noqa: D105
        parts = [self.message]
        if self.provider:
            parts.append(f"provider={self.provider}")
        if self.model:
            parts.append(f"model={self.model}")
        if self.status_code is not None:
            parts.append(f"status={self.status_code}")
        return " | ".join(parts)


class AuthenticationError(EnrouteError):
    """Raised when credentials are missing or rejected (typically HTTP 401/403)."""


class RateLimitError(EnrouteError):
    """Raised when a provider rate-limits the request (typically HTTP 429)."""


class ContextLengthError(EnrouteError):
    """Raised when the request exceeds the model's context window."""


class ContentFilterError(EnrouteError):
    """Raised when a provider refuses the request due to a content filter."""


class ProviderUnavailable(EnrouteError):
    """Raised when a provider is down or returns a transient 5xx error."""


class InvalidRequestError(EnrouteError):
    """Raised when the request is malformed or uses unsupported parameters."""


class TimeoutError(EnrouteError):
    """Raised when a provider call exceeds the configured timeout."""


class BudgetExceededError(EnrouteError):
    """Raised when a request would exceed a configured cost or token budget."""


class ConfigurationError(EnrouteError):
    """Raised when enroute is misconfigured (missing keys, unknown models, etc.)."""


class NotFoundError(EnrouteError):
    """Raised when a requested resource (model, trace, dataset) cannot be found."""


RETRYABLE_ERRORS: tuple[type[EnrouteError], ...] = (
    RateLimitError,
    ProviderUnavailable,
    TimeoutError,
)


def is_retryable(error: BaseException) -> bool:
    """Return whether ``error`` should be retried by the router.

    Args:
        error: The exception raised by a provider call.

    Returns:
        ``True`` if the error is a known retryable enroute error.

    Examples:
        >>> is_retryable(TimeoutError("timed out"))
        True
        >>> is_retryable(AuthenticationError("bad key"))
        False
    """
    return isinstance(error, RETRYABLE_ERRORS)

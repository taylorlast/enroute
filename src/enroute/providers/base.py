"""Provider protocol and shared HTTP utilities.

Examples:
    >>> from enroute.providers.base import ProviderConfig
    >>> cfg = ProviderConfig(api_key="sk-test", base_url="https://api.openai.com/v1")
    >>> cfg.timeout_s
    60.0
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, Field

from enroute.config import DEFAULT_SETTINGS
from enroute.errors import (
    AuthenticationError,
    ContentFilterError,
    ContextLengthError,
    EnrouteError,
    InvalidRequestError,
    ProviderUnavailable,
    RateLimitError,
    TimeoutError,
)
from enroute.types import ChatRequest, ChatResponse, StreamChunk


class ProviderConfig(BaseModel):
    """Configuration for a provider adapter.

    Attributes:
        api_key: Provider API key.
        base_url: API base URL including version prefix when required.
        timeout_s: Request timeout in seconds.
        default_headers: Extra headers sent with every request.
        organization: Optional organization id (OpenAI-style).
    """

    api_key: str
    base_url: str
    timeout_s: float = DEFAULT_SETTINGS.default_timeout_s
    default_headers: dict[str, str] = Field(default_factory=dict)
    organization: str | None = None


@runtime_checkable
class Provider(Protocol):
    """Protocol implemented by all provider adapters.

    Adapters must provide sync and async chat/stream methods that accept
    normalized :class:`~enroute.types.ChatRequest` objects and return
    normalized responses.
    """

    name: str

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Execute a non-streaming chat completion."""
        ...

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        """Execute a streaming chat completion."""
        ...

    async def achat(self, request: ChatRequest) -> ChatResponse:
        """Async non-streaming chat completion."""
        ...

    def astream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Async streaming chat completion.

        Implementations are typically ``async def`` generators whose return
        type is :class:`~typing.AsyncIterator`.
        """
        ...

    def close(self) -> None:
        """Close underlying HTTP resources."""
        ...

    async def aclose(self) -> None:
        """Close underlying async HTTP resources."""
        ...


def classify_http_error(
    *,
    status_code: int,
    body: Any,
    provider: str,
    model: str | None = None,
) -> EnrouteError:
    """Map an HTTP error response to a classified :class:`EnrouteError`.

    Args:
        status_code: HTTP status code.
        body: Parsed or raw response body.
        provider: Provider slug.
        model: Model id if known.

    Returns:
        A specific :class:`EnrouteError` subclass instance.

    Examples:
        >>> err = classify_http_error(status_code=429, body={}, provider="openai")
        >>> isinstance(err, RateLimitError)
        True
    """
    message = _extract_message(body) or f"HTTP {status_code}"
    lowered = message.lower()
    kwargs: dict[str, Any] = {
        "provider": provider,
        "status_code": status_code,
        "body": body,
        "model": model,
    }

    if status_code in {401, 403}:
        return AuthenticationError(message, **kwargs)
    if status_code == 429:
        return RateLimitError(message, **kwargs)
    if status_code == 400 and any(
        token in lowered for token in ("context", "maximum context", "too many tokens", "token")
    ):
        return ContextLengthError(message, **kwargs)
    if status_code == 400 and any(
        token in lowered for token in ("content", "safety", "filtered", "moderation")
    ):
        return ContentFilterError(message, **kwargs)
    if status_code == 400:
        return InvalidRequestError(message, **kwargs)
    if status_code >= 500:
        return ProviderUnavailable(message, **kwargs)
    return EnrouteError(message, **kwargs)


def _extract_message(body: Any) -> str | None:
    if body is None:
        return None
    if isinstance(body, str):
        return body
    if not isinstance(body, Mapping):
        return str(body)
    error = body.get("error")
    if isinstance(error, Mapping):
        msg = error.get("message") or error.get("msg")
        if msg:
            return str(msg)
    if "message" in body:
        return str(body["message"])
    return None


def parse_json_or_text(response: httpx.Response) -> Any:
    """Parse a response body as JSON, falling back to text.

    Args:
        response: The HTTP response.

    Returns:
        Parsed JSON or the raw text body.
    """
    try:
        return response.json()
    except (json.JSONDecodeError, ValueError):
        return response.text


def raise_for_status(
    response: httpx.Response,
    *,
    provider: str,
    model: str | None = None,
) -> None:
    """Raise a classified error for non-success HTTP responses.

    Args:
        response: The HTTP response.
        provider: Provider slug.
        model: Model id if known.

    Raises:
        EnrouteError: When the status code indicates failure.
    """
    if response.is_success:
        return
    body = parse_json_or_text(response)
    raise classify_http_error(
        status_code=response.status_code,
        body=body,
        provider=provider,
        model=model,
    )


def map_transport_error(exc: Exception, *, provider: str, model: str | None = None) -> EnrouteError:
    """Map httpx transport errors to enroute errors.

    Args:
        exc: The transport exception.
        provider: Provider slug.
        model: Model id if known.

    Returns:
        A classified :class:`EnrouteError`.
    """
    if isinstance(exc, httpx.TimeoutException):
        return TimeoutError(str(exc) or "request timed out", provider=provider, model=model)
    if isinstance(exc, httpx.TransportError):
        return ProviderUnavailable(str(exc) or "transport error", provider=provider, model=model)
    return EnrouteError(str(exc), provider=provider, model=model)


def iter_sse_lines(response: httpx.Response) -> Iterator[str]:
    """Yield data payloads from an SSE response.

    Args:
        response: A streaming HTTP response.

    Yields:
        JSON (or plain) data strings from ``data:`` lines.
    """
    for line in response.iter_lines():
        if not line:
            continue
        if line.startswith("data:"):
            data = line[5:].strip()
            if data:
                yield data


async def aiter_sse_lines(response: httpx.Response) -> AsyncIterator[str]:
    """Async variant of :func:`iter_sse_lines`.

    Args:
        response: A streaming HTTP response.

    Yields:
        JSON (or plain) data strings from ``data:`` lines.
    """
    async for line in response.aiter_lines():
        if not line:
            continue
        if line.startswith("data:"):
            data = line[5:].strip()
            if data:
                yield data

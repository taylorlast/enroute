"""Request router with retries, fallbacks, and budget checks.

Examples:
    >>> from enroute.routing.router import Router
    >>> Router.__name__
    'Router'
"""

from __future__ import annotations

import random
import time
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from enroute.catalog.models import ModelCatalog, estimate_cost
from enroute.errors import (
    BudgetExceededError,
    ConfigurationError,
    EnrouteError,
    is_retryable,
)
from enroute.providers.base import Provider
from enroute.routing.policies import Explicit, ModelRoute, RoutingPolicy
from enroute.types import ChatRequest, ChatResponse, StreamChunk, Usage


class AttemptRecord:
    """Record of a single provider attempt.

    Args:
        model: Model id attempted.
        provider: Provider slug.
        error: Error message if the attempt failed.
        latency_ms: Attempt latency in milliseconds.
        status_code: HTTP status code if available.
    """

    def __init__(
        self,
        model: str,
        provider: str,
        *,
        error: str | None = None,
        latency_ms: float | None = None,
        status_code: int | None = None,
    ) -> None:
        self.model = model
        self.provider = provider
        self.error = error
        self.latency_ms = latency_ms
        self.status_code = status_code

    def to_dict(self) -> dict[str, Any]:
        """Serialize the attempt for inclusion in a trace.

        Returns:
            A JSON-compatible dictionary.
        """
        return {
            "model": self.model,
            "provider": self.provider,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "status_code": self.status_code,
        }


class Router:
    """Execute chat requests against providers with policy, retry, and fallback.

    Args:
        providers: Mapping of provider slug to :class:`~enroute.providers.base.Provider`.
        catalog: Model catalog for cost and provider lookup.
        policy: Routing policy. Defaults to :class:`~enroute.routing.policies.Explicit`.
        max_retries: Retries per route for retryable errors.
        max_cost_usd: Optional per-request cost budget.
        retry_backoff_s: Base backoff in seconds (jittered).
    """

    def __init__(
        self,
        providers: dict[str, Provider],
        catalog: ModelCatalog | None = None,
        policy: RoutingPolicy | None = None,
        *,
        max_retries: int = 2,
        max_cost_usd: float | None = None,
        retry_backoff_s: float = 0.5,
    ) -> None:
        self.providers = providers
        self.catalog = catalog or ModelCatalog()
        self.policy = policy or Explicit()
        self.max_retries = max_retries
        self.max_cost_usd = max_cost_usd
        self.retry_backoff_s = retry_backoff_s

    def _candidates(self, request: ChatRequest) -> list[str]:
        models = [request.model]
        if request.models:
            for m in request.models:
                if m not in models:
                    models.append(m)
        return models

    def _routes(self, request: ChatRequest) -> list[ModelRoute]:
        candidates = self._candidates(request)
        routes = self.policy.select(request, candidates, self.catalog)
        if not routes:
            raise ConfigurationError("no routes available for request", model=request.model)
        # Hosted gateway mode: one ``enroute`` provider serves every author/slug.
        if "enroute" in self.providers:
            routes = [
                ModelRoute(model=route.model, provider="enroute", priority=route.priority)
                for route in routes
            ]
        return routes

    def _get_provider(self, route: ModelRoute) -> Provider:
        provider = self.providers.get(route.provider)
        if provider is None and "enroute" in self.providers:
            provider = self.providers["enroute"]
        if provider is None:
            raise ConfigurationError(
                f"no provider configured for '{route.provider}'",
                provider=route.provider,
                model=route.model,
            )
        return provider

    def _check_budget(self, request: ChatRequest, route: ModelRoute) -> None:
        if self.max_cost_usd is None:
            return
        prefs = request.provider
        max_price = prefs.max_price if prefs else None
        # Soft check using catalog pricing with a nominal token estimate.
        spec = self.catalog.get(route.model)
        if spec is None or spec.pricing is None:
            return
        estimated = estimate_cost(Usage.from_counts(1000, 500), spec)
        if estimated is not None and estimated > self.max_cost_usd:
            raise BudgetExceededError(
                f"estimated cost {estimated:.6f} exceeds budget {self.max_cost_usd}",
                model=route.model,
                provider=route.provider,
            )
        if max_price:
            prompt_cap = max_price.get("prompt")
            completion_cap = max_price.get("completion")
            if prompt_cap is not None and spec.pricing.prompt > prompt_cap:
                raise BudgetExceededError(
                    "prompt price exceeds max_price.prompt",
                    model=route.model,
                    provider=route.provider,
                )
            if completion_cap is not None and spec.pricing.completion > completion_cap:
                raise BudgetExceededError(
                    "completion price exceeds max_price.completion",
                    model=route.model,
                    provider=route.provider,
                )

    def _sleep(self, attempt: int) -> None:
        delay = self.retry_backoff_s * (2**attempt) * (0.5 + random.random())
        time.sleep(min(delay, 8.0))

    async def _asleep(self, attempt: int) -> None:
        import asyncio

        delay = self.retry_backoff_s * (2**attempt) * (0.5 + random.random())
        await asyncio.sleep(min(delay, 8.0))

    def _annotate_cost(self, response: ChatResponse, model: str) -> ChatResponse:
        cost = estimate_cost(response.usage, self.catalog.get(model))
        if cost is not None:
            response.usage.cost = cost
        return response

    def chat(self, request: ChatRequest) -> tuple[ChatResponse, list[AttemptRecord]]:
        """Execute a non-streaming chat request.

        Args:
            request: Normalized chat request.

        Returns:
            A tuple of ``(response, attempts)``.

        Raises:
            EnrouteError: When all routes fail.
        """
        attempts: list[AttemptRecord] = []
        last_error: EnrouteError | None = None
        routes = self._routes(request)

        for route in routes:
            self._check_budget(request, route)
            provider = self._get_provider(route)
            route_request = request.model_copy(update={"model": route.model})
            for retry in range(self.max_retries + 1):
                started = time.perf_counter()
                try:
                    response = provider.chat(route_request)
                    response = self._annotate_cost(response, route.model)
                    response.provider = route.provider
                    response.attempts = len(attempts) + 1
                    attempts.append(
                        AttemptRecord(
                            route.model,
                            route.provider,
                            latency_ms=(time.perf_counter() - started) * 1000,
                        )
                    )
                    return response, attempts
                except EnrouteError as exc:
                    last_error = exc
                    attempts.append(
                        AttemptRecord(
                            route.model,
                            route.provider,
                            error=str(exc),
                            latency_ms=(time.perf_counter() - started) * 1000,
                            status_code=exc.status_code,
                        )
                    )
                    if is_retryable(exc) and retry < self.max_retries:
                        self._sleep(retry)
                        continue
                    break

        assert last_error is not None
        raise last_error

    async def achat(self, request: ChatRequest) -> tuple[ChatResponse, list[AttemptRecord]]:
        """Async non-streaming chat request.

        Args:
            request: Normalized chat request.

        Returns:
            A tuple of ``(response, attempts)``.
        """
        attempts: list[AttemptRecord] = []
        last_error: EnrouteError | None = None
        routes = self._routes(request)

        for route in routes:
            self._check_budget(request, route)
            provider = self._get_provider(route)
            route_request = request.model_copy(update={"model": route.model})
            for retry in range(self.max_retries + 1):
                started = time.perf_counter()
                try:
                    response = await provider.achat(route_request)
                    response = self._annotate_cost(response, route.model)
                    response.provider = route.provider
                    response.attempts = len(attempts) + 1
                    attempts.append(
                        AttemptRecord(
                            route.model,
                            route.provider,
                            latency_ms=(time.perf_counter() - started) * 1000,
                        )
                    )
                    return response, attempts
                except EnrouteError as exc:
                    last_error = exc
                    attempts.append(
                        AttemptRecord(
                            route.model,
                            route.provider,
                            error=str(exc),
                            latency_ms=(time.perf_counter() - started) * 1000,
                            status_code=exc.status_code,
                        )
                    )
                    if is_retryable(exc) and retry < self.max_retries:
                        await self._asleep(retry)
                        continue
                    break

        assert last_error is not None
        raise last_error

    def stream(
        self,
        request: ChatRequest,
        on_complete: Callable[[ChatResponse, list[AttemptRecord]], None] | None = None,
    ) -> Iterator[StreamChunk]:
        """Stream a chat request, trying routes until one succeeds.

        Args:
            request: Normalized chat request.
            on_complete: Optional callback invoked with the reconstructed
                response and attempt records when the stream finishes.

        Yields:
            Stream chunks from the successful provider.
        """
        attempts: list[AttemptRecord] = []
        last_error: EnrouteError | None = None
        routes = self._routes(request)

        for route in routes:
            self._check_budget(request, route)
            provider = self._get_provider(route)
            route_request = request.model_copy(update={"model": route.model, "stream": True})
            started = time.perf_counter()
            content_parts: list[str] = []
            usage = Usage()
            finish = None
            response_id = ""
            try:
                for chunk in provider.stream(route_request):
                    response_id = chunk.id or response_id
                    if chunk.delta.content:
                        content_parts.append(chunk.delta.content)
                    if chunk.usage:
                        usage = chunk.usage
                    if chunk.finish_reason:
                        finish = chunk.finish_reason
                    yield chunk
                latency_ms = (time.perf_counter() - started) * 1000
                attempts.append(AttemptRecord(route.model, route.provider, latency_ms=latency_ms))
                from enroute.types import Choice, Message

                response = ChatResponse(
                    id=response_id,
                    model=route.model,
                    choices=[
                        Choice(
                            message=Message(role="assistant", content="".join(content_parts)),
                            finish_reason=finish,
                        )
                    ],
                    usage=usage,
                    provider=route.provider,
                    latency_ms=latency_ms,
                    attempts=len(attempts),
                )
                response = self._annotate_cost(response, route.model)
                if on_complete:
                    on_complete(response, attempts)
                return
            except EnrouteError as exc:
                last_error = exc
                attempts.append(
                    AttemptRecord(
                        route.model,
                        route.provider,
                        error=str(exc),
                        latency_ms=(time.perf_counter() - started) * 1000,
                        status_code=exc.status_code,
                    )
                )
                continue

        assert last_error is not None
        raise last_error

    async def astream(
        self,
        request: ChatRequest,
        on_complete: Callable[[ChatResponse, list[AttemptRecord]], None] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Async streaming chat request.

        Args:
            request: Normalized chat request.
            on_complete: Optional completion callback.

        Yields:
            Stream chunks from the successful provider.
        """
        attempts: list[AttemptRecord] = []
        last_error: EnrouteError | None = None
        routes = self._routes(request)

        for route in routes:
            self._check_budget(request, route)
            provider = self._get_provider(route)
            route_request = request.model_copy(update={"model": route.model, "stream": True})
            started = time.perf_counter()
            content_parts: list[str] = []
            usage = Usage()
            finish = None
            response_id = ""
            try:
                async for chunk in provider.astream(route_request):
                    response_id = chunk.id or response_id
                    if chunk.delta.content:
                        content_parts.append(chunk.delta.content)
                    if chunk.usage:
                        usage = chunk.usage
                    if chunk.finish_reason:
                        finish = chunk.finish_reason
                    yield chunk
                latency_ms = (time.perf_counter() - started) * 1000
                attempts.append(AttemptRecord(route.model, route.provider, latency_ms=latency_ms))
                from enroute.types import Choice, Message

                response = ChatResponse(
                    id=response_id,
                    model=route.model,
                    choices=[
                        Choice(
                            message=Message(role="assistant", content="".join(content_parts)),
                            finish_reason=finish,
                        )
                    ],
                    usage=usage,
                    provider=route.provider,
                    latency_ms=latency_ms,
                    attempts=len(attempts),
                )
                response = self._annotate_cost(response, route.model)
                if on_complete:
                    on_complete(response, attempts)
                return
            except EnrouteError as exc:
                last_error = exc
                attempts.append(
                    AttemptRecord(
                        route.model,
                        route.provider,
                        error=str(exc),
                        latency_ms=(time.perf_counter() - started) * 1000,
                        status_code=exc.status_code,
                    )
                )
                continue

        assert last_error is not None
        raise last_error

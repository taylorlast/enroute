"""Routing policy protocol and built-in policies.

The :class:`RoutingPolicy` protocol is the seat a future learned autorouter
will occupy. Built-ins cover explicit selection, fallback chains, least cost,
and lowest latency heuristics.

Examples:
    >>> from enroute.catalog import ModelCatalog
    >>> from enroute.routing.policies import LeastCost
    >>> from enroute.types import ChatRequest, Message
    >>> catalog = ModelCatalog()
    >>> req = ChatRequest(
    ...     model="openai/gpt-4o",
    ...     messages=[Message(role="user", content="hi")],
    ...     models=["openai/gpt-4o-mini", "google/gemini-2.5-flash"],
    ... )
    >>> routes = LeastCost().select(req, req.models or [], catalog)
    >>> routes[0].model
    'openai/gpt-4o'
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from enroute.catalog.models import ModelCatalog
from enroute.types import ChatRequest, ProviderPreferences


class ModelRoute(BaseModel):
    """A single candidate route selected by a policy.

    Attributes:
        model: Model id in ``author/slug`` form.
        provider: Provider slug that should serve the request.
        priority: Lower values are tried first.
    """

    model: str
    provider: str
    priority: int = 0


@runtime_checkable
class RoutingPolicy(Protocol):
    """Select an ordered list of model routes for a request."""

    def select(
        self,
        request: ChatRequest,
        candidates: list[str],
        catalog: ModelCatalog,
    ) -> list[ModelRoute]:
        """Return ordered routes to try.

        Args:
            request: The chat request.
            candidates: Candidate model ids (primary + fallbacks).
            catalog: Model catalog used for cost/latency metadata.

        Returns:
            Ordered list of :class:`ModelRoute` values.
        """
        ...


def _provider_for(model_id: str, catalog: ModelCatalog) -> str:
    spec = catalog.get(model_id)
    if spec is not None:
        return spec.provider
    if "/" in model_id:
        return model_id.split("/", 1)[0]
    return model_id


def _apply_provider_prefs(
    routes: list[ModelRoute],
    prefs: ProviderPreferences | None,
) -> list[ModelRoute]:
    if prefs is None:
        return routes
    result = routes
    if prefs.only:
        allowed = set(prefs.only)
        result = [r for r in result if r.provider in allowed]
    if prefs.ignore:
        ignored = set(prefs.ignore)
        result = [r for r in result if r.provider not in ignored]
    if prefs.order:
        order = {name: i for i, name in enumerate(prefs.order)}
        preferred = [r for r in result if r.provider in order]
        preferred.sort(key=lambda r: order[r.provider])
        rest = [r for r in result if r.provider not in order]
        result = preferred + rest if prefs.allow_fallbacks else preferred
    return result


class Explicit:
    """Use the request's model (and optional ``models`` fallbacks) as-is.

    Examples:
        >>> from enroute.catalog import ModelCatalog
        >>> from enroute.types import ChatRequest, Message
        >>> policy = Explicit()
        >>> req = ChatRequest(
        ...     model="openai/gpt-4o-mini", messages=[Message(role="user", content="x")]
        ... )
        >>> policy.select(req, [req.model], ModelCatalog())[0].model
        'openai/gpt-4o-mini'
    """

    def select(
        self,
        request: ChatRequest,
        candidates: list[str],
        catalog: ModelCatalog,
    ) -> list[ModelRoute]:
        """Return routes in candidate order.

        Args:
            request: The chat request.
            candidates: Candidate model ids.
            catalog: Model catalog.

        Returns:
            Ordered routes.
        """
        routes = [
            ModelRoute(model=m, provider=_provider_for(m, catalog), priority=i)
            for i, m in enumerate(candidates)
        ]
        return _apply_provider_prefs(routes, request.provider)


class Fallback(Explicit):
    """Alias for :class:`Explicit` emphasizing fallback-chain semantics."""


class LeastCost:
    """Prefer cheaper models among candidates after the primary.

    The primary model remains first; remaining candidates are sorted by
    estimated prompt+completion price (using equal token weights).
    """

    def select(
        self,
        request: ChatRequest,
        candidates: list[str],
        catalog: ModelCatalog,
    ) -> list[ModelRoute]:
        """Return the primary route first, then cheapest fallbacks.

        Args:
            request: The chat request.
            candidates: Candidate model ids.
            catalog: Model catalog.

        Returns:
            Ordered routes.
        """
        if not candidates:
            return []
        primary = candidates[0]
        rest = list(candidates[1:])

        def price_key(model_id: str) -> float:
            spec = catalog.get(model_id)
            if spec is None or spec.pricing is None:
                return float("inf")
            return spec.pricing.prompt + spec.pricing.completion

        rest.sort(key=price_key)
        ordered = [primary, *rest]
        routes = [
            ModelRoute(model=m, provider=_provider_for(m, catalog), priority=i)
            for i, m in enumerate(ordered)
        ]
        if request.provider and request.provider.sort == "price":
            all_sorted = sorted(candidates, key=price_key)
            routes = [
                ModelRoute(model=m, provider=_provider_for(m, catalog), priority=i)
                for i, m in enumerate(all_sorted)
            ]
        return _apply_provider_prefs(routes, request.provider)


class LowestLatency:
    """Prefer providers historically associated with low latency.

    Uses a static heuristic table for v1; a learned policy can replace this
    later via the same :class:`RoutingPolicy` protocol.
    """

    _LATENCY_RANK: dict[str, int] = {
        "groq": 0,
        "fireworks": 1,
        "together": 2,
        "deepseek": 3,
        "google": 4,
        "openai": 5,
        "xai": 6,
        "mistral": 7,
        "anthropic": 8,
    }

    def select(
        self,
        request: ChatRequest,
        candidates: list[str],
        catalog: ModelCatalog,
    ) -> list[ModelRoute]:
        """Return routes ordered by heuristic provider latency rank.

        Args:
            request: The chat request.
            candidates: Candidate model ids.
            catalog: Model catalog.

        Returns:
            Ordered routes.
        """
        ranked = sorted(
            candidates,
            key=lambda m: self._LATENCY_RANK.get(_provider_for(m, catalog), 100),
        )
        # Keep primary first unless sort=latency is requested.
        if not (request.provider and request.provider.sort == "latency") and candidates:
            primary = candidates[0]
            ranked = [primary] + [m for m in ranked if m != primary]
        routes = [
            ModelRoute(model=m, provider=_provider_for(m, catalog), priority=i)
            for i, m in enumerate(ranked)
        ]
        return _apply_provider_prefs(routes, request.provider)

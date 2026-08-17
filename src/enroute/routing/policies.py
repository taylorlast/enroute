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
    ...     model="openai/gpt-5.6-sol",
    ...     messages=[Message(role="user", content="hi")],
    ...     models=["openai/gpt-5.6-luna", "google/gemini-3.7-flash"],
    ... )
    >>> routes = LeastCost().select(req, req.models or [], catalog)
    >>> routes[0].model
    'openai/gpt-5.6-sol'
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
        provider: Host slug that should serve the request.
        upstream_id: Model id the host expects, when different from ``model``.
        priority: Lower values are tried first.
    """

    model: str
    provider: str
    upstream_id: str | None = None
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


def _author_for(model_id: str, catalog: ModelCatalog) -> str:
    spec = catalog.get(model_id)
    if spec is not None:
        return spec.provider
    if "/" in model_id:
        return model_id.split("/", 1)[0]
    return model_id


def routes_for_model(
    model_id: str,
    catalog: ModelCatalog,
    *,
    priority_base: int = 0,
) -> list[ModelRoute]:
    """Expand a catalog model into host routes (US hosts first).

    Args:
        model_id: Canonical ``author/slug`` id.
        catalog: Model catalog.
        priority_base: Starting priority for this model's endpoints.

    Returns:
        Ordered host routes for the model.
    """
    spec = catalog.get(model_id)
    if spec is None:
        return [
            ModelRoute(
                model=model_id,
                provider=_author_for(model_id, catalog),
                upstream_id=None,
                priority=priority_base,
            )
        ]
    return [
        ModelRoute(
            model=model_id,
            provider=endpoint.provider,
            upstream_id=endpoint.upstream_id,
            priority=priority_base + index,
        )
        for index, endpoint in enumerate(spec.ordered_endpoints())
    ]


def expand_models(model_ids: list[str], catalog: ModelCatalog) -> list[ModelRoute]:
    """Expand each candidate model into its host routes, keeping model order.

    Returns:
        Every host route for the candidates, ordered by model then endpoint priority.
    """
    routes: list[ModelRoute] = []
    for index, model_id in enumerate(model_ids):
        routes.extend(routes_for_model(model_id, catalog, priority_base=index * 10))
    return routes


def _apply_provider_prefs(
    routes: list[ModelRoute],
    prefs: ProviderPreferences | None,
) -> list[ModelRoute]:
    if prefs is None:
        return routes
    result = routes
    if prefs.only:
        allowed = set(prefs.only)
        result = [route for route in result if route.provider in allowed]
    if prefs.ignore:
        ignored = set(prefs.ignore)
        result = [route for route in result if route.provider not in ignored]
    if prefs.order:
        order = {name: i for i, name in enumerate(prefs.order)}
        preferred = [route for route in result if route.provider in order]
        preferred.sort(key=lambda route: order[route.provider])
        rest = [route for route in result if route.provider not in order]
        result = preferred + rest if prefs.allow_fallbacks else preferred
    return result


class Explicit:
    """Use the request's model (and optional ``models`` fallbacks) as-is.

    Multi-host models expand to US endpoints first, then the lab host.

    Examples:
        >>> from enroute.catalog import ModelCatalog
        >>> from enroute.types import ChatRequest, Message
        >>> policy = Explicit()
        >>> req = ChatRequest(
        ...     model="openai/gpt-5.6-luna", messages=[Message(role="user", content="x")]
        ... )
        >>> policy.select(req, [req.model], ModelCatalog())[0].model
        'openai/gpt-5.6-luna'
    """

    def select(
        self,
        request: ChatRequest,
        candidates: list[str],
        catalog: ModelCatalog,
    ) -> list[ModelRoute]:
        """Return routes in candidate order, expanding multi-host models.

        Args:
            request: The chat request.
            candidates: Candidate model ids.
            catalog: Model catalog.

        Returns:
            Ordered routes.
        """
        return _apply_provider_prefs(expand_models(candidates, catalog), request.provider)


class Fallback(Explicit):
    """Alias for :class:`Explicit` emphasizing fallback-chain semantics."""


class LeastCost:
    """Prefer cheaper models among candidates after the primary.

    The primary model remains first; remaining candidates are sorted by
    estimated prompt+completion price (using equal token weights). Each
    model then expands to its host endpoints (US first).
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

        def price_key(model_id: str) -> float:
            spec = catalog.get(model_id)
            if spec is None:
                return float("inf")
            pricing = spec.pricing_for()
            if pricing is None:
                return float("inf")
            return pricing.prompt + pricing.completion

        if request.provider and request.provider.sort == "price":
            ordered = sorted(candidates, key=price_key)
        else:
            primary = candidates[0]
            rest = list(candidates[1:])
            rest.sort(key=price_key)
            ordered = [primary, *rest]
        return _apply_provider_prefs(expand_models(ordered, catalog), request.provider)


class LowestLatency:
    """Prefer providers historically associated with low latency.

    Uses a static heuristic table for v1; a learned policy can replace this
    later via the same :class:`RoutingPolicy` protocol.
    """

    _LATENCY_RANK: dict[str, int] = {
        "groq": 0,
        "fireworks": 1,
        "baseten": 2,
        "together": 3,
        "deepseek": 4,
        "google": 5,
        "openai": 6,
        "xai": 7,
        "meta": 8,
        "mistral": 9,
        "anthropic": 10,
        "moonshot": 20,
        "qwen": 21,
        "zhipu": 22,
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
        routes = expand_models(candidates, catalog)
        if request.provider and request.provider.sort == "latency":
            routes.sort(key=lambda route: self._LATENCY_RANK.get(route.provider, 100))
        elif candidates:
            primary = [route for route in routes if route.model == candidates[0]]
            rest = [route for route in routes if route.model != candidates[0]]
            rest.sort(key=lambda route: self._LATENCY_RANK.get(route.provider, 100))
            routes = primary + rest
        return _apply_provider_prefs(routes, request.provider)

"""Pick live-test models from the catalog instead of hardcoding ids.

Hardcoded ids rot: hosts retire model names, and the live suite then fails with
a 404 that says nothing about the code under test. Reading the catalog means the
live tests follow the same list the router serves.
"""

from __future__ import annotations

import os
from functools import cache

from enroute.catalog.models import ModelCatalog

HOST_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}


@cache
def _catalog() -> ModelCatalog:
    catalog = ModelCatalog()
    catalog.load_bundled()
    return catalog


@cache
def cheapest_model(host: str, *requires: str) -> str | None:
    """Cheapest catalog model the host serves that accepts the given parameters.

    Cheapest keeps the live suite affordable and tends to select the small, fast
    model in a family, which is enough to exercise a wire format. Filtering on
    ``supported_parameters`` avoids failures that say nothing about our code, such
    as a model that only accepts tools on a different endpoint.

    Args:
        host: Provider slug such as ``anthropic``.
        requires: Request parameters the model must accept, e.g. ``"tools"``.

    Returns:
        A catalog model id, or ``None`` when the host serves no such model.
    """
    candidates: list[tuple[float, str]] = []
    for spec in _catalog().models():
        supported = set(spec.supported_parameters)
        if not supported.issuperset(requires):
            continue
        for endpoint in spec.endpoints:
            if endpoint.provider != host:
                continue
            pricing = endpoint.pricing or spec.pricing
            rate = pricing.completion if pricing else None
            candidates.append((rate if rate is not None else float("inf"), spec.id))
    if not candidates:
        return None
    return min(candidates)[1]


def live_cases(*requires: str) -> list[tuple[str, str]]:
    """Hosts that have both a key in the environment and a suitable model.

    Args:
        requires: Request parameters the model must accept.

    Returns:
        ``(host, model)`` pairs for parametrizing live tests.
    """
    cases: list[tuple[str, str]] = []
    for host, env_key in HOST_ENV_KEYS.items():
        if not os.environ.get(env_key):
            continue
        model = cheapest_model(host, *requires)
        if model is not None:
            cases.append((host, model))
    return cases

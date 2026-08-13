"""Routing policies and the request router.

Examples:
    >>> from enroute.routing import Explicit, Fallback, LeastCost
    >>> Explicit().__class__.__name__
    'Explicit'
"""

from __future__ import annotations

from enroute.routing.policies import (
    Explicit,
    Fallback,
    LeastCost,
    LowestLatency,
    ModelRoute,
    RoutingPolicy,
)
from enroute.routing.router import Router

__all__ = [
    "Explicit",
    "Fallback",
    "LeastCost",
    "LowestLatency",
    "ModelRoute",
    "Router",
    "RoutingPolicy",
]

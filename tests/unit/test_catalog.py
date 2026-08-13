from enroute.catalog import ModelCatalog, estimate_cost
from enroute.types import Usage


def test_bundled_catalog() -> None:
    catalog = ModelCatalog()
    assert catalog.get("openai/gpt-4o-mini") is not None
    assert catalog.require("openai/gpt-4o-mini").provider == "openai"


def test_estimate_cost() -> None:
    catalog = ModelCatalog()
    spec = catalog.require("openai/gpt-4o-mini")
    cost = estimate_cost(Usage.from_counts(1_000_000, 0), spec)
    assert cost is not None
    assert abs(cost - 0.15) < 1e-9

from enroute.catalog import ModelCatalog, estimate_cost
from enroute.types import Usage


def test_bundled_catalog() -> None:
    catalog = ModelCatalog()
    assert catalog.get("openai/gpt-5.6-luna") is not None
    assert catalog.require("openai/gpt-5.6-luna").provider == "openai"


def test_estimate_cost() -> None:
    catalog = ModelCatalog()
    spec = catalog.require("openai/gpt-5.6-luna")
    cost = estimate_cost(Usage.from_counts(1_000_000, 0), spec)
    assert cost is not None
    assert abs(cost - 0.20) < 1e-9


def test_multi_host_us_first() -> None:
    catalog = ModelCatalog()
    spec = catalog.require("moonshot/kimi-k3")
    hosts = [endpoint.provider for endpoint in spec.ordered_endpoints()]
    assert hosts[0] == "fireworks"
    assert hosts[1] == "baseten"
    assert hosts[-1] == "moonshot"
    assert spec.pricing_for("moonshot").prompt > spec.pricing_for("fireworks").prompt

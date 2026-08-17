import json

from enroute.catalog import ModelCatalog, estimate_cost
from enroute.catalog.models import ModelPricing, ModelSpec, PriceTier
from enroute.types import Usage

LONG_CONTEXT = ModelPricing(
    prompt=5e-6,
    completion=3e-5,
    tiers=[PriceTier(min_prompt_tokens=272_000, prompt=1e-5, completion=6e-5)],
)


def test_bundled_catalog() -> None:
    catalog = ModelCatalog()
    assert catalog.get("openai/gpt-5.6-luna") is not None
    assert catalog.require("openai/gpt-5.6-luna").provider == "openai"


def test_estimate_cost() -> None:
    catalog = ModelCatalog()
    spec = catalog.require("openai/gpt-5.6-luna")
    # Below the long-context threshold, so the base rate applies.
    cost = estimate_cost(Usage.from_counts(200_000, 0), spec)
    assert cost is not None
    assert abs(cost - 0.04) < 1e-9


def test_region_bills_at_its_own_rate() -> None:
    catalog = ModelCatalog()
    spec = catalog.require("openai/gpt-5.6-sol")
    us = spec.pricing_for("azure", "us")
    eu = spec.pricing_for("azure", "eu")
    # EU capacity lists above US, so resolving without the region would undercharge.
    assert eu.prompt > us.prompt
    usage = Usage.from_counts(1_000_000, 0)
    assert estimate_cost(usage, spec, provider="azure", region="eu") > estimate_cost(
        usage, spec, provider="azure", region="us"
    )


def test_long_prompt_tier_replaces_the_base_rate() -> None:
    # Providers that price long context charge the higher rate on the whole
    # request, not only on tokens past the threshold.
    assert LONG_CONTEXT.rates_for_prompt(271_999) == (5e-6, 3e-5)
    assert LONG_CONTEXT.rates_for_prompt(272_000) == (1e-5, 6e-5)

    spec = ModelSpec(id="openai/gpt-x", pricing=LONG_CONTEXT)
    short = estimate_cost(Usage.from_counts(100_000, 1_000), spec)
    long = estimate_cost(Usage.from_counts(300_000, 1_000), spec)
    assert short is not None and long is not None
    assert short == 100_000 * 5e-6 + 1_000 * 3e-5
    assert long == 300_000 * 1e-5 + 1_000 * 6e-5
    # Tripling the prompt across the threshold more than triples the cost.
    assert long / short > 3


def test_highest_matching_tier_wins() -> None:
    pricing = ModelPricing(
        prompt=1e-6,
        completion=2e-6,
        tiers=[
            PriceTier(min_prompt_tokens=128_000, prompt=2e-6, completion=4e-6),
            PriceTier(min_prompt_tokens=512_000, prompt=4e-6, completion=8e-6),
        ],
    )
    assert pricing.rates_for_prompt(200_000) == (2e-6, 4e-6)
    assert pricing.rates_for_prompt(600_000) == (4e-6, 8e-6)


def test_tiers_survive_a_snapshot_round_trip(tmp_path) -> None:
    catalog = ModelCatalog()
    spec = catalog.require("openai/gpt-5.6-sol")
    assert spec.pricing is not None and spec.pricing.tiers, "expected long-context rates"

    destination = tmp_path / "models.json"
    catalog.write_snapshot(destination)
    written = json.loads(destination.read_text())
    entry = next(m for m in written["models"] if m["id"] == "openai/gpt-5.6-sol")
    # A tier lost in serialization would silently bill long prompts at the base rate.
    assert entry["pricing"]["tiers"]

    reloaded = ModelCatalog(destination).require("openai/gpt-5.6-sol")
    assert reloaded.pricing is not None
    assert reloaded.pricing.tiers == spec.pricing.tiers
    assert reloaded.pricing_for("azure").tiers == spec.pricing_for("azure").tiers


def test_multi_host_us_first() -> None:
    catalog = ModelCatalog()
    spec = catalog.require("moonshot/kimi-k3")
    hosts = [endpoint.provider for endpoint in spec.ordered_endpoints()]
    assert hosts[0] == "fireworks"
    assert hosts[1] == "baseten"
    assert hosts[-1] == "moonshot"
    assert spec.pricing_for("moonshot").prompt > spec.pricing_for("fireworks").prompt

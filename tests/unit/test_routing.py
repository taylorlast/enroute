import pytest

from enroute.catalog import ModelCatalog
from enroute.errors import ConfigurationError
from enroute.routing import Explicit, LeastCost, LowestLatency
from enroute.routing.router import Router
from enroute.types import ChatRequest, ChatResponse, Choice, Message, ProviderPreferences, Usage


class _StubProvider:
    name = "stub"

    def chat(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(
            id="1",
            model=request.model,
            choices=[Choice(message=Message(role="assistant", content="ok"))],
            usage=Usage.from_counts(1, 1),
            provider=self.name,
        )

    def stream(self, request: ChatRequest):
        raise NotImplementedError

    async def achat(self, request: ChatRequest) -> ChatResponse:
        return self.chat(request)

    async def astream(self, request: ChatRequest):
        raise NotImplementedError
        yield

    def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


def model_order(routes: list) -> list[str]:
    """Candidate order with host expansion collapsed.

    Models list a varying number of hosts, so asserting on positional index
    breaks whenever a cloud is added to the catalog.
    """
    return list(dict.fromkeys(route.model for route in routes))


def test_explicit_order() -> None:
    catalog = ModelCatalog()
    req = ChatRequest(model="openai/gpt-5.6-sol", messages=[Message(role="user", content="hi")])
    routes = Explicit().select(req, ["openai/gpt-5.6-sol", "openai/gpt-5.6-luna"], catalog)
    assert model_order(routes) == ["openai/gpt-5.6-sol", "openai/gpt-5.6-luna"]
    # The lab's own endpoint leads; clouds follow as fallbacks.
    assert routes[0].provider == "openai"


def test_least_cost_sorts_fallbacks() -> None:
    catalog = ModelCatalog()
    req = ChatRequest(
        model="openai/gpt-5.6-sol",
        messages=[Message(role="user", content="hi")],
        models=["openai/gpt-5.6-luna", "anthropic/claude-sonnet-5"],
    )
    routes = LeastCost().select(
        req,
        ["openai/gpt-5.6-sol", "openai/gpt-5.6-luna", "anthropic/claude-sonnet-5"],
        catalog,
    )
    # The primary stays first; the cheaper fallback outranks the pricier one.
    assert model_order(routes) == [
        "openai/gpt-5.6-sol",
        "openai/gpt-5.6-luna",
        "anthropic/claude-sonnet-5",
    ]


def test_provider_only_filter() -> None:
    catalog = ModelCatalog()
    req = ChatRequest(
        model="openai/gpt-5.6-luna",
        messages=[Message(role="user", content="hi")],
        provider=ProviderPreferences(only=["anthropic"]),
    )
    routes = Explicit().select(req, ["openai/gpt-5.6-luna", "anthropic/claude-sonnet-5"], catalog)
    assert [r.provider for r in routes] == ["anthropic"]


def test_lowest_latency() -> None:
    catalog = ModelCatalog()
    req = ChatRequest(
        model="anthropic/claude-sonnet-5",
        messages=[Message(role="user", content="hi")],
        provider=ProviderPreferences(sort="latency"),
    )
    routes = LowestLatency().select(
        req, ["anthropic/claude-sonnet-5", "openai/gpt-5.6-luna"], catalog
    )
    assert routes[0].provider == "openai"


def test_multi_host_defaults_to_us() -> None:
    catalog = ModelCatalog()
    req = ChatRequest(model="moonshot/kimi-k3", messages=[Message(role="user", content="hi")])
    routes = Explicit().select(req, ["moonshot/kimi-k3"], catalog)
    assert [r.provider for r in routes] == ["fireworks", "baseten", "moonshot"]
    assert routes[0].upstream_id == "accounts/fireworks/models/kimi-k3"


def test_multi_host_lab_opt_in() -> None:
    catalog = ModelCatalog()
    req = ChatRequest(
        model="moonshot/kimi-k3",
        messages=[Message(role="user", content="hi")],
        provider=ProviderPreferences(only=["moonshot"]),
    )
    routes = Explicit().select(req, ["moonshot/kimi-k3"], catalog)
    assert [r.provider for r in routes] == ["moonshot"]
    assert routes[0].upstream_id == "kimi-k3"


def test_routes_skip_hosts_we_have_no_key_for() -> None:
    # The catalog lists clouds like Azure and Bedrock for flagship models. A
    # deployment without those credentials must still reach the hosts it has.
    catalog = ModelCatalog()
    spec = catalog.require("moonshot/kimi-k3")
    hosts = [endpoint.provider for endpoint in spec.ordered_endpoints()]
    assert "moonshot" in hosts and "fireworks" in hosts

    router = Router({"moonshot": _StubProvider()}, catalog=catalog)
    req = ChatRequest(model="moonshot/kimi-k3", messages=[Message(role="user", content="hi")])
    routes = router._routes(req)

    assert [route.provider for route in routes] == ["moonshot"]
    assert router.chat(req)[0].text == "ok"


def test_unroutable_model_names_the_hosts_it_needs() -> None:
    catalog = ModelCatalog()
    router = Router({"moonshot": _StubProvider()}, catalog=catalog)
    req = ChatRequest(model="openai/gpt-5.6-sol", messages=[Message(role="user", content="hi")])
    with pytest.raises(ConfigurationError) as excinfo:
        router._routes(req)
    assert "openai" in str(excinfo.value)


def test_routes_carry_the_region_they_will_be_billed_at() -> None:
    catalog = ModelCatalog()
    req = ChatRequest(model="openai/gpt-5.6-sol", messages=[Message(role="user", content="hi")])
    routes = Explicit().select(req, ["openai/gpt-5.6-sol"], catalog)
    regions = {(r.provider, r.region) for r in routes}
    assert ("azure", "eu") in regions and ("azure", "us") in regions


def test_a_regional_provider_is_not_offered_other_regions() -> None:
    # An EU-bound Azure resource must not be handed a US route, or we would bill
    # the US rate for capacity we never called.
    catalog = ModelCatalog()
    eu_only = _StubProvider()
    eu_only.name = "azure"
    eu_only.region = "eu"
    router = Router({"azure": eu_only}, catalog=catalog)
    req = ChatRequest(model="openai/gpt-5.6-sol", messages=[Message(role="user", content="hi")])

    routes = router._routes(req)
    assert [(r.provider, r.region) for r in routes] == [("azure", "eu")]

    us_only = _StubProvider()
    us_only.name = "azure"
    us_only.region = "us"
    assert [r.region for r in Router({"azure": us_only}, catalog=catalog)._routes(req)] == ["us"]


def test_gateway_mode_routes_all_models_through_enroute() -> None:
    gateway = _StubProvider()
    gateway.name = "enroute"
    router = Router({"enroute": gateway})
    req = ChatRequest(
        model="openai/gpt-5.6-luna",
        messages=[Message(role="user", content="hi")],
        models=["anthropic/claude-sonnet-5"],
    )
    routes = router._routes(req)
    assert all(route.provider == "enroute" for route in routes)
    response, attempts = router.chat(req)
    assert response.text == "ok"
    assert attempts[0].provider == "enroute"

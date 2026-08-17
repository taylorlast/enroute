from enroute.catalog import ModelCatalog
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


def test_explicit_order() -> None:
    catalog = ModelCatalog()
    req = ChatRequest(model="openai/gpt-5.6-sol", messages=[Message(role="user", content="hi")])
    routes = Explicit().select(req, ["openai/gpt-5.6-sol", "openai/gpt-5.6-luna"], catalog)
    assert [r.model for r in routes] == ["openai/gpt-5.6-sol", "openai/gpt-5.6-luna"]


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
    assert routes[0].model == "openai/gpt-5.6-sol"
    assert routes[1].model == "openai/gpt-5.6-luna"


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

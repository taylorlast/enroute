"""End-to-end streaming: chunks reach the caller and the stream is billed correctly."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from enroute import Enroute
from enroute.catalog import ModelCatalog
from enroute.errors import ProviderUnavailable
from enroute.tracing import JSONLSink
from enroute.types import (
    ChatRequest,
    ChatResponse,
    Choice,
    Message,
    StreamChunk,
    StreamDelta,
    Usage,
)

MODEL = "openai/gpt-5.6-sol"
LONG_PROMPT_TOKENS = 300_000


class StreamingProvider:
    """Emits token deltas, then usage on the final chunk, the way real APIs do."""

    name = "openai"

    def __init__(self, *, prompt_tokens: int = 10, fail: bool = False) -> None:
        self.prompt_tokens = prompt_tokens
        self.fail = fail
        self.streams = 0

    def chat(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(
            id="c1",
            model=request.model,
            choices=[Choice(message=Message(role="assistant", content="Hello"))],
            usage=Usage.from_counts(self.prompt_tokens, 5),
            provider=self.name,
        )

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        self.streams += 1
        if self.fail:
            raise ProviderUnavailable("upstream down", provider=self.name)
        for piece in ("Hel", "lo"):
            yield StreamChunk(
                id="s1", model=request.model, delta=StreamDelta(content=piece), provider=self.name
            )
        yield StreamChunk(
            id="s1",
            model=request.model,
            finish_reason="stop",
            usage=Usage.from_counts(self.prompt_tokens, 5),
            provider=self.name,
        )

    async def achat(self, request: ChatRequest) -> ChatResponse:
        return self.chat(request)

    async def astream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        for chunk in self.stream(request):
            yield chunk

    def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


def client(provider: StreamingProvider, tmp_path: Path) -> Enroute:
    return Enroute(
        providers={"openai": provider},
        sink=JSONLSink(tmp_path / "traces.jsonl"),
        capture_content=True,
    )


def test_stream_yields_deltas_and_traces_once(tmp_path: Path) -> None:
    sink = JSONLSink(tmp_path / "traces.jsonl")
    enroute = Enroute(providers={"openai": StreamingProvider()}, sink=sink, capture_content=True)
    chunks = list(enroute.stream(model=MODEL, messages=[Message(role="user", content="hi")]))

    assert "".join(c.delta.content or "" for c in chunks) == "Hello"
    assert chunks[-1].finish_reason == "stop"
    # Every chunk is attributed to the host that served it, for billing.
    assert {c.provider for c in chunks} == {"openai"}
    assert {c.region for c in chunks} == {"us"}

    enroute.flush()
    traces = sink.read_all()
    assert len(traces) == 1
    assert traces[0].steps[0].response.usage.completion_tokens == 5
    enroute.close()


@pytest.mark.asyncio
async def test_astream_matches_sync(tmp_path: Path) -> None:
    enroute = client(StreamingProvider(), tmp_path)
    chunks = [
        chunk
        async for chunk in enroute.astream(
            model=MODEL, messages=[Message(role="user", content="hi")]
        )
    ]
    assert "".join(c.delta.content or "" for c in chunks) == "Hello"
    await enroute.aclose()


def test_streams_are_billed_from_the_usage_chunk(tmp_path: Path) -> None:
    provider = StreamingProvider()
    sink = JSONLSink(tmp_path / "traces.jsonl")
    enroute = Enroute(providers={"openai": provider}, sink=sink, capture_content=True)
    list(enroute.stream(model=MODEL, messages=[Message(role="user", content="hi")]))
    enroute.flush()

    spec = ModelCatalog().require(MODEL)
    cost = sink.read_all()[0].steps[0].response.usage.cost
    assert cost == pytest.approx(10 * spec.pricing.prompt + 5 * spec.pricing.completion)
    enroute.close()


def test_a_long_streamed_prompt_is_billed_at_the_tier_rate(tmp_path: Path) -> None:
    spec = ModelCatalog().require(MODEL)
    assert spec.pricing is not None and spec.pricing.tiers
    tier = spec.pricing.tiers[0]
    assert tier.min_prompt_tokens < LONG_PROMPT_TOKENS

    sink = JSONLSink(tmp_path / "traces.jsonl")
    enroute = Enroute(
        providers={"openai": StreamingProvider(prompt_tokens=LONG_PROMPT_TOKENS)},
        sink=sink,
        capture_content=True,
    )
    list(enroute.stream(model=MODEL, messages=[Message(role="user", content="hi")]))
    enroute.flush()

    cost = sink.read_all()[0].steps[0].response.usage.cost
    assert cost == pytest.approx(LONG_PROMPT_TOKENS * tier.prompt + 5 * tier.completion)
    # Billing the base rate here is the leak this guards against.
    base = LONG_PROMPT_TOKENS * spec.pricing.prompt + 5 * spec.pricing.completion
    assert cost > base
    enroute.close()


def test_stream_falls_back_to_the_next_host(tmp_path: Path) -> None:
    broken = StreamingProvider(fail=True)
    healthy = StreamingProvider()
    healthy.name = "azure"
    enroute = Enroute(
        providers={"openai": broken, "azure": healthy},
        sink=JSONLSink(tmp_path / "traces.jsonl"),
        capture_content=True,
    )
    chunks = list(enroute.stream(model=MODEL, messages=[Message(role="user", content="hi")]))

    assert broken.streams == 1
    assert healthy.streams == 1
    assert "".join(c.delta.content or "" for c in chunks) == "Hello"
    assert {c.provider for c in chunks} == {"azure"}
    enroute.close()

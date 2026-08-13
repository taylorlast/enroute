"""Shared offline stand-ins for tracing / environment / benchmarking examples."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

from enroute.types import (
    ChatRequest,
    ChatResponse,
    Choice,
    Message,
    StreamChunk,
    StreamDelta,
    Usage,
)

EXAMPLES_DIR = Path(".enroute/examples")


class ScriptedProvider:
    """Deterministic provider for offline examples."""

    def __init__(self, name: str, text: str) -> None:
        self.name = name
        self.text = text
        self.calls = 0

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        return ChatResponse(
            id=f"{self.name}-{self.calls}",
            model=request.model,
            choices=[Choice(message=Message(role="assistant", content=self.text))],
            usage=Usage.from_counts(8, 8, cost=0.0),
            provider=self.name,
            latency_ms=1.5,
        )

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        yield StreamChunk(
            id=f"{self.name}-stream",
            model=request.model,
            delta=StreamDelta(content=self.text),
            finish_reason="stop",
            usage=Usage.from_counts(8, 8),
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


def ensure_out_dir() -> Path:
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    return EXAMPLES_DIR

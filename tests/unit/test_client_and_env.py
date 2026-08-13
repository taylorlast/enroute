from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

from enroute import Benchmark, Dataset, Enroute, Environment, TaskData, Trace
from enroute.types import (
    ChatRequest,
    ChatResponse,
    Choice,
    Message,
    StreamChunk,
    StreamDelta,
    Usage,
)


class FakeProvider:
    name = "openai"

    def __init__(self, text: str = "ok", tool_calls: list[Any] | None = None) -> None:
        self.text = text
        self.tool_calls = tool_calls
        self.calls = 0

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        msg = Message(role="assistant", content=self.text, tool_calls=self.tool_calls)
        return ChatResponse(
            id=f"id-{self.calls}",
            model=request.model,
            choices=[Choice(message=msg, finish_reason="stop")],
            usage=Usage.from_counts(10, 5, cost=0.001),
            provider=self.name,
            latency_ms=12.0,
        )

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        yield StreamChunk(
            id="s1",
            model=request.model,
            delta=StreamDelta(content=self.text),
            finish_reason="stop",
            usage=Usage.from_counts(1, 1),
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


def test_client_chat_records_trace(tmp_path: Path) -> None:
    provider = FakeProvider("hello")
    client = Enroute(
        providers={"openai": provider},
        sink=None,
        trace_dir=tmp_path,
        capture_content=True,
    )
    # Replace default sink path created inside — recreate with explicit sink
    from enroute.tracing import JSONLSink

    client.close()
    sink = JSONLSink(tmp_path / "traces.jsonl")
    client = Enroute(
        providers={"openai": provider},
        sink=sink,
        capture_content=True,
    )
    resp = client.chat(
        model="openai/gpt-4o-mini",
        messages=[Message(role="user", content="Hi")],
    )
    assert resp.text == "hello"
    client.flush()
    traces = sink.read_all()
    assert len(traces) == 1
    assert traces[0].steps[0].type == "llm"
    client.close()


def test_environment_rollout_and_benchmark(tmp_path: Path) -> None:
    from enroute.tracing import JSONLSink

    sink = JSONLSink(tmp_path / "traces.jsonl")
    client = Enroute(providers={"openai": FakeProvider("refund")}, sink=sink, capture_content=True)

    env = Environment(name="support-triage", version="0.1.0", system_prompt="Triage tickets.")

    @env.tool
    def lookup_order(order_id: str) -> dict[str, str]:
        """Look up an order."""
        return {"order_id": order_id, "status": "shipped"}

    @env.scorer(weight=1.0)
    def mentions_refund(rollout: Any) -> float:
        text = rollout.response.text or ""
        return 1.0 if "refund" in text.lower() else 0.0

    @env.tasks
    def tasks() -> list[TaskData]:
        return [TaskData(task_id="t1", input="I want a refund for order 1")]

    rollout = env.rollout(next(env.iter_tasks()), client, model="openai/gpt-4o-mini")
    assert rollout.trace.outcome is not None
    assert rollout.trace.outcome.reward == 1.0

    report = Benchmark(env, models=["openai/gpt-4o-mini"], client=client, concurrency=1).run()
    assert report.models["openai/gpt-4o-mini"].mean_reward == 1.0
    assert "openai/gpt-4o-mini" in report.to_markdown()

    ds = Dataset.from_traces("support", [rollout.trace])
    assert len(ds) == 1
    ds.save(tmp_path / "ds.jsonl")
    loaded = Dataset.load(tmp_path / "ds.jsonl")
    assert loaded.content_hash == ds.content_hash
    client.close()


def test_dataset_from_sink(tmp_path: Path) -> None:
    from enroute.tracing import JSONLSink

    sink = JSONLSink(tmp_path / "t.jsonl")
    sink.write(Trace(trace_id="1", environment="a"))
    sink.write(Trace(trace_id="2", environment="b"))
    sink.close()
    ds = Dataset.from_sink(tmp_path / "t.jsonl", "prod", where=lambda t: t.environment == "a")
    assert len(ds) == 1

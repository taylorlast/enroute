"""Collect library episodes and show the RL training handoff.

Two scripted policies on the same task:

* researcher — search, read, answer correctly
* guesser — answer immediately, wrong

Research has no per-decision reward. ``trace.returns(gamma=0.9)`` is how the
search/read actions get credit for the later correct answer — the same
mechanism as researching, posting, then receiving likes after the fact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _shared import ensure_out_dir
from enroute import Dataset, Enroute
from enroute.environments.export.verifiers import to_verifiers_trace
from enroute.tracing import JSONLSink, Trace
from enroute.types import (
    ChatRequest,
    ChatResponse,
    Choice,
    FunctionCall,
    Message,
    ToolCall,
    Usage,
)
from examples.environment.library.env import make_env


class ScriptedLibraryProvider:
    """Deterministic offline policy."""

    name = "openai"

    def __init__(self, script: list[tuple[str, dict[str, object]] | str]) -> None:
        self.script = script
        self.calls = 0

    def chat(self, request: ChatRequest) -> ChatResponse:
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(item, str):
            msg = Message(role="assistant", content=item)
        else:
            name, arguments = item
            msg = Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id=f"lib-{self.calls}",
                        function=FunctionCall(name=name, arguments=json.dumps(arguments)),
                    )
                ],
            )
        return ChatResponse(
            id=f"lib-{self.calls}",
            model=request.model,
            choices=[Choice(message=msg)],
            usage=Usage.from_counts(12, 8, cost=0.0),
            provider=self.name,
            latency_ms=1.0,
        )

    def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


def _print_episode(title: str, trace: Trace) -> None:
    print(f"\n== {title} ==")
    reward = trace.outcome.reward if trace.outcome else None
    print(f"reward={reward}  terminated={trace.terminated}")
    for i, decision in enumerate(trace.decisions()):
        action = ", ".join(a.name for a in decision.parsed_action) or "respond"
        print(f"  t={i}  {action}")
    print(f"  r_t  (outcome) {trace.decision_rewards(source='outcome')}")
    print(f"  G_t  γ=0.9    {trace.returns(gamma=0.9)}")


def main() -> None:
    out = ensure_out_dir()
    env = make_env()
    task = next(env.iter_tasks())

    researcher = ScriptedLibraryProvider(
        [
            ("search", {"query": "voting"}),
            ("read", {"doc_id": "d1"}),
            ("answer", {"text": "the river path"}),
        ]
    )
    guesser = ScriptedLibraryProvider(
        [
            ("answer", {"text": "I don't know"}),
        ]
    )

    traces = []
    for name, provider in (("researcher", researcher), ("guesser", guesser)):
        client = Enroute(
            providers={"openai": provider},
            sink=JSONLSink(out / f"library-{name}.jsonl"),
            capture_content=True,
        )
        rollout = env.rollout(task, client, model="openai/gpt-4o-mini")
        traces.append(rollout.trace)
        _print_episode(name, rollout.trace)
        client.close()

    good = traces[0]
    # Likes / a reviewer arrive later — attribute them to the answer decision.
    good.credit(0.4, name="reviewer", reason="readers found it useful", decision_index=-1)
    print("\n== researcher after late reviewer credit on the answer ==")
    print(f"  r_t  (events) {good.decision_rewards(source='events')}")
    print(f"  r_t  (both)   {good.decision_rewards(source='both')}")
    print(f"  G_t  γ=0.9 both {good.returns(gamma=0.9, source='both')}")
    print("  search/read now share credit for the later review — no env rewrite.")

    ds = Dataset.from_traces("library", traces, version="0.1.0")
    ds.save(out / "library-dataset.jsonl")
    exported = to_verifiers_trace(good)
    print(f"\ndataset={ds.content_hash[:12]}…  verifiers_returns={exported['returns']}")


if __name__ == "__main__":
    main()

"""Run one twitter-account episode and print the decision trace."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _shared import ensure_out_dir
from enroute import Enroute
from enroute.tracing import JSONLSink
from enroute.types import (
    ChatRequest,
    ChatResponse,
    Choice,
    FunctionCall,
    Message,
    ToolCall,
    Usage,
)
from examples.environment.twitter.env import make_env


class ScriptedTwitterProvider:
    """Offline policy: check notifications, reply, like, then stop."""

    name = "openai"

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        if self.calls == 1:
            return self._tool("view_notifications", {})
        if self.calls == 2:
            mention_id = _first_mention_id(request) or "p4"
            return self._tool(
                "reply",
                {
                    "post_id": mention_id,
                    "text": "Great question — start with the river-path piece.",
                },
            )
        if self.calls == 3:
            mention_id = _second_mention_id(request) or "p6"
            return self._tool("reply", {"post_id": mention_id, "text": "I'll be there."})
        return ChatResponse(
            id=f"tw-{self.calls}",
            model=request.model,
            choices=[Choice(message=Message(role="assistant", content="Mentions answered."))],
            usage=Usage.from_counts(20, 12, cost=0.0),
            provider=self.name,
            latency_ms=2.0,
        )

    def _tool(self, name: str, arguments: dict[str, object]) -> ChatResponse:
        return ChatResponse(
            id=f"tw-{self.calls}",
            model="openai/gpt-4o-mini",
            choices=[
                Choice(
                    message=Message(
                        role="assistant",
                        tool_calls=[
                            ToolCall(
                                id=f"call-{self.calls}",
                                function=FunctionCall(name=name, arguments=json.dumps(arguments)),
                            )
                        ],
                    )
                )
            ],
            usage=Usage.from_counts(20, 12, cost=0.0),
            provider=self.name,
            latency_ms=2.0,
        )

    def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


def _first_mention_id(request: ChatRequest) -> str | None:
    for message in reversed(request.messages):
        if message.role != "tool" or not message.content:
            continue
        try:
            payload = json.loads(message.content)
        except json.JSONDecodeError:
            continue
        mentions = payload.get("mentions") or []
        if mentions:
            return str(mentions[0]["post_id"])
    return None


def _second_mention_id(request: ChatRequest) -> str | None:
    for message in reversed(request.messages):
        if message.role != "tool" or not message.content:
            continue
        try:
            payload = json.loads(message.content)
        except json.JSONDecodeError:
            continue
        mentions = payload.get("mentions") or []
        if len(mentions) > 1:
            return str(mentions[1]["post_id"])
    return None


def main() -> None:
    out = ensure_out_dir()
    client = Enroute(
        providers={"openai": ScriptedTwitterProvider()},
        sink=JSONLSink(out / "twitter.jsonl"),
        capture_content=True,
    )
    env = make_env()
    task = next(env.iter_tasks())
    rollout = env.rollout(task, client, model="openai/gpt-4o-mini")
    trace = rollout.trace
    print(f"environment={trace.environment}@{trace.environment_version}")
    print(f"fingerprint={trace.environment_fingerprint}")
    print(f"reward={trace.outcome.reward if trace.outcome else None}")
    print(f"terminated={trace.terminated} truncated={trace.truncated}")
    print(f"metrics={trace.metrics}")
    for i, step in enumerate(trace.steps, start=1):
        if step.type != "decision":
            continue
        actions = ", ".join(a.name for a in step.parsed_action) or "respond"
        reward = sum(e.value for e in step.reward_events)
        print(f"  decision {i}: {actions}  step_reward={reward}")
    print(f"transitions={len(trace.transitions())}")
    client.close()


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from enroute import Enroute, TaskData
from enroute.tracing import JSONLSink
from enroute.tracing.schema import ParsedAction
from enroute.types import (
    ChatRequest,
    ChatResponse,
    Choice,
    FunctionCall,
    Message,
    ToolCall,
    Usage,
)

_LIB = Path(__file__).resolve().parents[2] / "examples" / "environment" / "library"
sys.path.insert(0, str(_LIB))
sys.modules.pop("env", None)
from env import LibraryEnv, make_env  # noqa: E402

sys.modules.pop("env", None)
sys.path.remove(str(_LIB))


def _env() -> LibraryEnv:
    env = LibraryEnv()
    env.setup(
        TaskData(
            task_id="river-path",
            input="What is the city voting on tonight?",
            expected="river path",
        )
    )
    return env


def test_search_read_answer_score() -> None:
    env = _env()
    hits = env.search("voting")
    assert hits["hits"][0]["doc_id"] == "d1"
    body = env.read("d1")
    assert "river path" in body["body"]
    env.answer("They vote on the river path")
    assert env.done() is True
    assert env.score() == 1.0
    assert env.state.submitted is not None
    assert env.step_reward("search", hits) is None


def test_wrong_answer_scores_zero() -> None:
    env = _env()
    env.answer("I don't know")
    assert env.score() == 0.0


def test_make_env_and_rollout(tmp_path: Path) -> None:
    class Provider:
        name = "openai"
        calls = 0

        def chat(self, request: ChatRequest) -> ChatResponse:
            self.calls += 1
            script = [
                ("search", {"query": "voting"}),
                ("read", {"doc_id": "d1"}),
                ("answer", {"text": "the river path"}),
            ]
            name, arguments = script[min(self.calls - 1, len(script) - 1)]
            return ChatResponse(
                id=f"id-{self.calls}",
                model=request.model,
                choices=[
                    Choice(
                        message=Message(
                            role="assistant",
                            tool_calls=[
                                ToolCall(
                                    id=f"c{self.calls}",
                                    function=FunctionCall(
                                        name=name, arguments=json.dumps(arguments)
                                    ),
                                )
                            ],
                        )
                    )
                ],
                usage=Usage.from_counts(4, 2, cost=0.0),
                provider=self.name,
                latency_ms=1.0,
            )

        def close(self) -> None:
            return None

        async def aclose(self) -> None:
            return None

    env = make_env()
    assert env.name == "library"
    names = {t.function.name for t in env.tool_defs}
    assert "research" in names
    client = Enroute(
        providers={"openai": Provider()},
        sink=JSONLSink(tmp_path / "t.jsonl"),
        capture_content=True,
    )
    rollout = env.rollout(next(env.iter_tasks()), client, model="openai/gpt-4o-mini")
    assert rollout.trace.outcome is not None
    assert rollout.trace.outcome.reward == 1.0
    names = [a.name for d in rollout.trace.decisions() for a in d.parsed_action]
    assert names == ["search", "read", "answer"]
    assert rollout.trace.returns(gamma=0.9) == pytest.approx([0.81, 0.9, 1.0])
    client.close()


def test_research_records_nested_tools() -> None:
    env = make_env()
    task = TaskData(
        task_id="river-path",
        input="What is the city voting on tonight?",
        expected="river path",
    )
    env.reset(task)
    env.step([ParsedAction(name="research", arguments={"query": "voting"})])
    rollout = env.close_episode()
    decision = rollout.trace.decisions()[0]
    assert decision.tool_calls[0].name == "research"
    assert [child.name for child in decision.tool_calls[0].children] == ["search", "read"]
    assert decision.tool_calls[0].children[0].parent == "research"

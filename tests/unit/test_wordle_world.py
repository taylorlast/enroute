from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from enroute import Enroute, TaskData
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

_WORDLE = Path(__file__).resolve().parents[2] / "examples" / "environment" / "wordle"
sys.path.insert(0, str(_WORDLE))
for _name in ("env", "words"):
    sys.modules.pop(_name, None)
from env import WordleEnv, make_env  # noqa: E402
from words import is_allowed, pattern, pick_answer  # noqa: E402

sys.modules.pop("env", None)
sys.modules.pop("words", None)
sys.path.remove(str(_WORDLE))


def _env(secret: str = "crane") -> WordleEnv:
    env = WordleEnv()
    env.setup(TaskData(task_id="t", input="play", expected=secret, metadata={"seed": 0}))
    return env


def test_pattern_duplicate_letters() -> None:
    assert pattern("crane", "crane") == "GGGGG"
    assert pattern("abide", "speed") == "...YY"
    assert pattern("erase", "speed") == "Y..YY"


def test_invalid_guess_does_not_consume_turn() -> None:
    env = _env()
    out = env.guess("xyzzy")
    assert "error" in out
    assert env.rows == []
    assert env.guesses_left == 6
    env.guess("hi")
    assert env.guesses_left == 6


def test_reset_seed_is_deterministic() -> None:
    a = WordleEnv()
    b = WordleEnv()
    task = TaskData(task_id="s", input="play", metadata={"seed": 42})
    a.setup(task)
    b.setup(task)
    assert a.secret == b.secret
    assert a.secret == pick_answer(42)
    assert is_allowed(a.secret)


def test_solve_sets_done_and_score() -> None:
    env = _env("crane")
    env.guess("slate")
    assert env.done() is False
    assert env.score() == 0.0
    env.guess("crane")
    assert env.solved is True
    assert env.done() is True
    assert env.score() == pytest.approx((7 - 2) / 6)


def test_make_env_only_guess() -> None:
    env = make_env()
    assert env.name == "wordle"
    assert env.version == "0.1.0"
    names = {t.function.name for t in env.tool_defs}
    assert names == {"guess"}


def test_rollout_solver(tmp_path: Path) -> None:
    class Provider:
        name = "openai"
        calls = 0

        def chat(self, request: ChatRequest) -> ChatResponse:
            self.calls += 1
            word = "crane" if self.calls > 1 else "slate"
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
                                        name="guess",
                                        arguments=json.dumps({"word": word}),
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
    client = Enroute(
        providers={"openai": Provider()},
        sink=JSONLSink(tmp_path / "t.jsonl"),
        capture_content=True,
    )
    task = TaskData(task_id="crane", input="play", expected="crane", metadata={"seed": 0})
    rollout = env.rollout(task, client, model="openai/gpt-4o-mini")
    assert rollout.trace.outcome is not None
    assert rollout.trace.outcome.reward == pytest.approx((7 - 2) / 6)
    assert rollout.env.solved is True
    client.close()


def test_play_via_reset_and_step(tmp_path: Path) -> None:
    from enroute.types import ChatRequest as Req

    class Provider:
        name = "openai"
        calls = 0

        def chat(self, request: ChatRequest) -> ChatResponse:
            self.calls += 1
            word = "crane"
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
                                        name="guess",
                                        arguments=json.dumps({"word": word}),
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
    client = Enroute(
        providers={"openai": Provider()},
        sink=JSONLSink(tmp_path / "t.jsonl"),
        capture_content=True,
    )
    task = TaskData(task_id="crane", input="play", expected="crane")
    obs, info = env.reset(task, model="openai/gpt-4o-mini")
    assert "Guesses left: 6" in str(obs)
    assert "environment" in info
    assert env.observation.guesses_left == 6
    while True:
        msgs = env.messages()
        request = Req(model="openai/gpt-4o-mini", messages=msgs, tools=env.tool_defs)
        response = client.chat(
            model="openai/gpt-4o-mini",
            messages=msgs,
            tools=env.tool_defs,
        )
        obs, _reward, terminated, truncated, extra = env.step(
            response.message.tool_calls,
            request=request,
            response=response,
        )
        if terminated or truncated:
            break
    rollout = env.close_episode(client=client)
    assert rollout.env.solved is True
    assert rollout.trace.outcome is not None
    assert rollout.trace.outcome.reward == pytest.approx(1.0)
    client.close()

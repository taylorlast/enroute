from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from enroute import Dataset, Enroute, Environment, TaskData, Trace
from enroute.environments import Observation, State, StepResult, tool
from enroute.environments.export.hf import to_huggingface_records
from enroute.environments.export.verifiers import to_verifiers_trace
from enroute.tracing import JSONLSink
from enroute.tracing.schema import Decision, Outcome, ParsedAction
from enroute.types import (
    ChatRequest,
    ChatResponse,
    Choice,
    FunctionCall,
    Message,
    ToolCall,
    Usage,
)


class SequentialProvider:
    name = "openai"

    def __init__(self, script: list[Any]) -> None:
        self.script = script
        self.calls = 0

    def chat(self, request: ChatRequest) -> ChatResponse:
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(item, tuple):
            name, arguments = item
            msg = Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id=f"c{self.calls}",
                        function=FunctionCall(name=name, arguments=str(arguments)),
                    )
                ],
            )
        else:
            msg = Message(role="assistant", content=str(item))
        return ChatResponse(
            id=f"id-{self.calls}",
            model=request.model,
            choices=[Choice(message=msg, finish_reason="stop")],
            usage=Usage.from_counts(4, 2, cost=0.01),
            provider=self.name,
            latency_ms=5.0,
        )

    def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


class CounterState(State):
    n: int = 0
    finished: bool = False


class CounterObservation(Observation):
    n: int = 0
    seed: int | None = None

    def render(self) -> str:
        return f"count={self.n} seed={self.seed}"


class CounterEnv(Environment[CounterObservation, CounterState]):
    name = "counter"
    version = "0.1.0"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.state = CounterState()

    def setup(self, task: TaskData) -> None:
        super().setup(task)
        self.state = CounterState(seed=self.seed, n=0)

    def observe(self) -> CounterObservation:
        return CounterObservation(n=self.state.n, seed=self.seed)

    def done(self) -> bool:
        return self.state.finished

    def step_reward(self, tool_name: str, result: Any) -> float | None:
        if tool_name == "inc":
            return 0.25
        return None

    @tool
    def inc(self, by: int = 1) -> dict[str, int]:
        """Increment the counter."""
        self.state.n += by
        if self.state.n >= 2:
            self.state.finished = True
        return {"n": self.state.n}


def _client(provider: SequentialProvider, tmp_path: Path) -> Enroute:
    return Enroute(
        providers={"openai": provider},
        sink=JSONLSink(tmp_path / "t.jsonl"),
        capture_content=True,
    )


def test_reset_step_gym_tuple() -> None:
    env = CounterEnv(max_turns=4)
    task = TaskData(task_id="t1", input="go", metadata={"seed": 7})
    obs, info = env.reset(task, model="openai/gpt-4o-mini")
    assert "count=0" in str(obs)
    assert info["environment_version"] == "0.1.0"
    assert info["environment_fingerprint"] == env.fingerprint()

    result = env.step([ParsedAction(name="inc", arguments={"by": 1})])
    obs2, reward, terminated, truncated, extra = result
    assert isinstance(result, StepResult)
    assert reward == 0.25
    assert terminated is False
    assert truncated is False
    assert extra["turn"] == 1
    assert "count=1" in str(obs2)

    env.step([ParsedAction(name="inc", arguments={"by": 1})])
    rollout = env.close_episode()
    assert rollout.env.state.n == 2
    assert rollout.trace.terminated is True
    assert rollout.trace.initial_state == {"n": 0, "seed": 7, "finished": False}
    assert rollout.trace.final_state == {"n": 2, "seed": 7, "finished": True}
    assert all(isinstance(s, Decision) for s in rollout.trace.steps)
    assert rollout.trace.steps[0].model_context is None
    assert rollout.trace.steps[0].parsed_action[0].name == "inc"


def test_truncated_at_max_turns() -> None:
    class NoDone(CounterEnv):
        def done(self) -> bool:
            return False

        @tool
        def inc(self, by: int = 1) -> dict[str, int]:
            """Increment the counter."""
            self.state.n += by
            return {"n": self.state.n}

    env = NoDone(max_turns=1)
    env.reset(TaskData(task_id="t", input="go", metadata={"seed": 1}))
    result = env.step([ParsedAction(name="inc", arguments={"by": 1})])
    assert result.truncated is True
    assert result.terminated is False
    env.close_episode()


def test_rollout_records_request_on_decision(tmp_path: Path) -> None:
    import json as json_mod

    env = CounterEnv(system_prompt="Count.", max_turns=4)

    @env.scorer(weight=1.0)
    def reached_two(rollout: Any) -> float:
        return 1.0 if rollout.env.state.n >= 2 else 0.0

    provider = SequentialProvider(
        [("inc", json_mod.dumps({"by": 1})), ("inc", json_mod.dumps({"by": 1})), "done"]
    )
    client = _client(provider, tmp_path)
    task = TaskData(task_id="t1", input="go", metadata={"seed": 3})
    rollout = env.rollout(task, client, model="openai/gpt-4o-mini")
    assert rollout.trace.outcome is not None
    assert rollout.trace.outcome.reward == 1.0
    assert rollout.trace.model == "openai/gpt-4o-mini"
    decisions = [s for s in rollout.trace.steps if s.type == "decision"]
    assert decisions
    assert decisions[0].model_context is not None
    request = decisions[0].model_context
    assert isinstance(request, ChatRequest)
    assert request.messages[0].role == "system"
    assert request.tools
    trans = rollout.trace.transitions()
    assert trans[0].reward == 0.25
    assert trans[-1].terminated is True
    exported = to_verifiers_trace(rollout.trace)
    assert exported["reward"] == 1.0
    assert exported["transitions"]
    assert any(m.get("role") == "tool" for m in exported["messages"])
    recs = to_huggingface_records(Dataset.from_traces("c", [rollout.trace]))
    assert recs[0]["environment_fingerprint"] == env.fingerprint()
    client.close()


def test_custom_step_override() -> None:
    class ManualEnv(Environment):
        name = "manual"
        version = "0.1.0"

        def setup(self, task: TaskData) -> None:
            super().setup(task)
            self.state = CounterState(seed=self.seed, n=0)

        def observe(self) -> CounterObservation:
            return CounterObservation(n=self.state.n, seed=self.seed)

        def done(self) -> bool:
            return self.state.n >= 1

        def step(self, action: Any = None, **kwargs: Any) -> StepResult:
            self.state.n = int(action)
            self.record_decision(
                parsed_action=[ParsedAction(name="set", arguments={"n": self.state.n})],
            )
            return self.finish_turn()

    env = ManualEnv()
    env.reset(TaskData(task_id="t", input="go"))
    result = env.step(1)
    assert result.terminated is True
    assert result.observation.n == 1
    rollout = env.close_episode()
    assert rollout.trace.steps[0].parsed_action[0].name == "set"


def test_fingerprint_changes_with_tools_and_scorers() -> None:
    env = Environment(name="x", version="0.1.0", system_prompt="A")

    @env.tool
    def ping() -> str:
        """Ping."""
        return "pong"

    first = env.fingerprint()
    assert first == env.fingerprint()

    @env.tool
    def pong() -> str:
        """Pong."""
        return "ping"

    assert env.fingerprint() != first

    env2 = Environment(name="x", version="0.1.0", system_prompt="A")

    @env2.tool(name="ping")
    def ping_env2() -> str:
        """Ping."""
        return "pong"

    @env2.scorer(weight=0.5)
    def s(rollout: Any) -> float:
        return 0.0

    assert env2.fingerprint() != first


def test_support_triage_still_works(tmp_path: Path) -> None:
    client = _client(SequentialProvider(["refund"]), tmp_path)
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
    assert rollout.trace.environment_fingerprint
    assert rollout.env is env
    client.close()


def test_spawn_isolation_under_concurrency() -> None:
    template = CounterEnv(max_turns=2)

    def _run(seed: int) -> int:
        env = template.spawn()
        obs, _info = env.reset(TaskData(task_id=str(seed), input="go", metadata={"seed": seed}))
        env.step([ParsedAction(name="inc", arguments={"by": 1})])
        rollout = env.close_episode()
        assert f"seed={seed}" in str(obs)
        return rollout.env.state.n

    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(_run, range(8)))
    assert values == [1] * 8


def test_decision_round_trip() -> None:
    trace = Trace(
        trace_id="e1",
        environment="counter",
        environment_version="0.1.0",
        environment_fingerprint="abc",
        model="openai/gpt-4o-mini",
        initial_state={"n": 0},
        final_state={"n": 1},
        terminated=True,
        truncated=False,
        outcome=Outcome(reward=1.0, scores={"ok": 1.0}),
    )
    trace.add_decision(
        observation="count=0",
        parsed_action=[ParsedAction(name="inc", arguments={"by": 1})],
    )
    loaded = Trace.model_validate_json(trace.model_dump_json())
    assert loaded.steps[0].type == "decision"
    assert loaded.environment_fingerprint == "abc"
    assert loaded.transitions()[0].action[0].name == "inc"


def test_flat_trace_transitions_fallback() -> None:
    from enroute.tracing.schema import LLMCall, ToolCallStep

    trace = Trace(trace_id="p", outcome=Outcome(reward=0.5), terminated=True)
    trace.steps.append(
        LLMCall(
            request=ChatRequest(model="m", messages=[Message(role="user", content="hi")]),
            response=None,
        )
    )
    trace.steps.append(ToolCallStep(name="lookup", arguments={"id": "1"}))
    trans = trace.transitions()
    assert len(trans) == 1
    assert trans[0].action[0].name == "lookup"
    assert trans[0].reward == 0.5

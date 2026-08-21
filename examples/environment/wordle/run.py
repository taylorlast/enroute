"""Play Wordle by stepping the environment. The LLM is only the policy.

Primary loop (Gymnasium / OpenEnv):

    obs, info = env.reset(task)
    while True:
        response = client.chat(...)   # policy — not part of the env
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _shared import ensure_out_dir
from enroute import Dataset, Enroute, Environment, TaskData
from enroute.environments import Rollout
from enroute.providers import OpenAICompatible
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
from examples.environment.wordle.env import make_env
from examples.environment.wordle.words import is_allowed


class ScriptedWordleProvider:
    """Deterministic offline policy: a list of guess words."""

    name = "openai"

    def __init__(self, guesses: list[str]) -> None:
        self.guesses = guesses
        self.calls = 0

    def chat(self, request: ChatRequest) -> ChatResponse:
        word = self.guesses[min(self.calls, len(self.guesses) - 1)]
        self.calls += 1
        return ChatResponse(
            id=f"wd-{self.calls}",
            model=request.model,
            choices=[
                Choice(
                    message=Message(
                        role="assistant",
                        tool_calls=[
                            ToolCall(
                                id=f"g{self.calls}",
                                function=FunctionCall(
                                    name="guess",
                                    arguments=json.dumps({"word": word}),
                                ),
                            )
                        ],
                    )
                )
            ],
            usage=Usage.from_counts(16, 8, cost=0.0),
            provider=self.name,
            latency_ms=1.0,
        )

    def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


def play(
    env: Environment,
    task: TaskData,
    client: Enroute,
    *,
    model: str,
) -> Rollout:
    """One episode: reset, policy, step until the env says stop."""
    obs, _info = env.reset(task, model=model)
    while True:
        request = ChatRequest(
            model=model,
            messages=env.messages(),
            tools=env.tool_defs or None,
        )
        response = client.chat(
            model=model,
            messages=env.messages(),
            tools=env.tool_defs or None,
            tags={"environment": env.name, "task_id": task.task_id},
        )
        action = response.message.tool_calls
        obs, _reward, terminated, truncated, info = env.step(
            action,
            request=request,
            response=response,
        )
        if terminated or truncated or info.get("stop_reason") == "no_tool_calls":
            break
    return env.close_episode(client=client)


def _print_episode(title: str, trace: Trace, board: str) -> None:
    print(f"\n== {title} ==")
    reward = trace.outcome.reward if trace.outcome else None
    print(f"reward={reward}  terminated={trace.terminated}")
    print(board)
    for i, decision in enumerate(trace.decisions()):
        action = ", ".join(f"{a.name}({a.arguments})" for a in decision.parsed_action)
        step_r = sum(e.value for e in decision.reward_events)
        print(f"  t={i}  {action}  step_reward={step_r:.2f}")
    print(f"  r_t  (outcome) {trace.decision_rewards(source='outcome')}")
    print(f"  r_t  (both)    {trace.decision_rewards(source='both')}")
    print(f"  G_t  γ=1 both  {trace.returns(gamma=1.0, source='both')}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play Wordle via env.reset / env.step. The LLM is only the policy."
    )
    parser.add_argument(
        "--secret",
        default="crane",
        help="Hidden 5-letter answer (must be in the Wordle word list). Default: crane.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Policy model id. With --base-url this is the name your local server "
            "exposes (e.g. llama3.2). Without --base-url it is an enroute model "
            "id (e.g. openai/gpt-4o-mini). Omit to run the offline scripted demo."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "OpenAI-compatible base URL for a local server "
            "(Ollama http://127.0.0.1:11434/v1, LM Studio http://127.0.0.1:1234/v1, "
            "vLLM http://127.0.0.1:8000/v1)."
        ),
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key for --base-url. Local servers usually accept EMPTY. Default: EMPTY.",
    )
    return parser.parse_args()


def _task(secret: str) -> TaskData:
    return TaskData(
        task_id=secret,
        input="Play Wordle. Use guess.",
        expected=secret,
        metadata={"seed": 0},
    )


def _local_client(model: str, base_url: str, api_key: str, sink: Path) -> tuple[Enroute, str]:
    provider = "local"
    server_model = model
    if "/" in model:
        provider, server_model = model.split("/", 1)
    client = Enroute(
        providers={
            provider: OpenAICompatible(
                api_key=api_key,
                base_url=base_url,
                name=provider,
            )
        },
        sink=JSONLSink(sink),
        capture_content=True,
    )
    return client, f"{provider}/{server_model}"


def _run_scripted(env: Environment, task: TaskData, secret: str, out: Path) -> list[Trace]:
    traces = []
    policies = (
        ("solver", ScriptedWordleProvider(["slate", secret])),
        ("misses", ScriptedWordleProvider(["audio", "wordy", "aback", "abase", "abate", "abbey"])),
    )
    for name, provider in policies:
        client = Enroute(
            providers={"openai": provider},
            sink=JSONLSink(out / f"wordle-{name}.jsonl"),
            capture_content=True,
        )
        rollout = play(env, task, client, model="openai/gpt-4o-mini")
        traces.append(rollout.trace)
        board = str(rollout.env.observation) if rollout.env is not None else ""
        _print_episode(name, rollout.trace, board)
        client.close()
    return traces


def main() -> None:
    args = _parse_args()
    secret = str(args.secret).strip().lower()
    if len(secret) != 5 or not is_allowed(secret):
        raise SystemExit(f"--secret must be a valid 5-letter Wordle word, got {secret!r}")
    if args.base_url and not args.model:
        raise SystemExit("--base-url requires --model (the name your local server exposes)")

    out = ensure_out_dir()
    env = make_env()
    task = _task(secret)

    if args.model is None:
        traces = _run_scripted(env, task, secret, out)
    elif args.base_url:
        client, model = _local_client(
            args.model, args.base_url, args.api_key, out / "wordle-local.jsonl"
        )
        try:
            rollout = play(env, task, client, model=model)
        finally:
            client.close()
        traces = [rollout.trace]
        board = str(rollout.env.observation) if rollout.env is not None else ""
        _print_episode(model, rollout.trace, board)
    else:
        client = Enroute(sink=JSONLSink(out / "wordle-model.jsonl"), capture_content=True)
        try:
            rollout = play(env, task, client, model=args.model)
        finally:
            client.close()
        traces = [rollout.trace]
        board = str(rollout.env.observation) if rollout.env is not None else ""
        _print_episode(args.model, rollout.trace, board)

    ds = Dataset.from_traces("wordle", traces, version="0.1.0")
    ds.save(out / "wordle-dataset.jsonl")
    print(f"\ndataset={ds.content_hash[:12]}…")


if __name__ == "__main__":
    main()

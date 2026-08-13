"""Benchmark two models against the same environment."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _shared import ScriptedProvider, ensure_out_dir
from enroute import Benchmark, Enroute, Environment, TaskData
from enroute.tracing import JSONLSink


def main() -> None:
    out = ensure_out_dir()
    good = ScriptedProvider("openai", "approved refund")
    bad = ScriptedProvider("anthropic", "please wait")

    client = Enroute(
        providers={"openai": good, "anthropic": bad},
        sink=JSONLSink(out / "bench.jsonl"),
        capture_content=True,
    )

    env = Environment(name="refund-quality", version="0.1.0")

    @env.scorer(weight=1.0)
    def says_refund(rollout) -> float:
        return 1.0 if "refund" in (rollout.response.text or "").lower() else 0.0

    @env.tasks
    def tasks():
        yield TaskData(task_id="r1", input="I need a refund")

    report = Benchmark(
        env,
        models=["openai/gpt-4o-mini", "anthropic/claude-sonnet-4"],
        client=client,
        concurrency=2,
    ).run()
    print(report.to_markdown())
    client.close()


if __name__ == "__main__":
    main()

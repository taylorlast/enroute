"""Define an environment, run a rollout, and build a dataset."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _shared import ScriptedProvider, ensure_out_dir
from enroute import Dataset, Enroute, Environment, TaskData
from enroute.tracing import JSONLSink


def main() -> None:
    out = ensure_out_dir()
    client = Enroute(
        providers={"openai": ScriptedProvider("openai", "Order shipped")},
        sink=JSONLSink(out / "env.jsonl"),
        capture_content=True,
    )

    env = Environment(name="support-triage", version="0.1.0")

    @env.tool
    def lookup_order(order_id: str) -> dict[str, str]:
        """Look up an order by id."""
        return {"order_id": order_id, "status": "shipped"}

    @env.scorer(weight=1.0)
    def mentions_shipped(rollout) -> float:
        text = (rollout.response.text or "").lower()
        return 1.0 if "shipped" in text else 0.0

    @env.tasks
    def tasks():
        yield TaskData(task_id="ord-1", input="Where is order A123?")

    rollout = env.rollout(next(env.iter_tasks()), client, model="openai/gpt-4o-mini")
    ds = Dataset.from_traces("support-triage", [rollout.trace], version="0.1.0")
    ds.save(out / "support-dataset.jsonl")
    print("reward=", rollout.trace.outcome.reward if rollout.trace.outcome else None)
    print("dataset_hash=", ds.content_hash)
    client.close()


if __name__ == "__main__":
    main()

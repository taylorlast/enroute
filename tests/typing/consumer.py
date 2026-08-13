"""mypy --strict consumer smoke test for public exports."""

from __future__ import annotations

from enroute import (
    Benchmark,
    Dataset,
    Enroute,
    Environment,
    Message,
    ModelCatalog,
    TaskData,
    Trace,
    Usage,
)


def main() -> None:
    catalog: ModelCatalog = ModelCatalog()
    _ = catalog.get("openai/gpt-4o-mini")
    usage: Usage = Usage.from_counts(1, 1)
    msg: Message = Message(role="user", content="hi")
    trace: Trace = Trace(trace_id="t")
    env: Environment = Environment(name="x", version="0.1.0")
    task: TaskData = TaskData(task_id="1", input="hi")
    ds: Dataset = Dataset.from_traces("d", [trace])
    assert usage.total_tokens == 2
    assert msg.role == "user"
    assert env.name == "x"
    assert task.task_id == "1"
    assert len(ds) == 1
    _ = Enroute
    _ = Benchmark


if __name__ == "__main__":
    main()

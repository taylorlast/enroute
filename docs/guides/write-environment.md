# Write your first environment

1. Name and version it — versions make benchmark history comparable.
2. Register tools that mirror production.
3. Write scorers that match your definition of done.
4. Provide tasks (synthetic or sampled from production).

```python
from enroute import Environment, TaskData

env = Environment(
    name="support-triage",
    version="0.2.0",
    system_prompt="You are a support agent. Use tools before guessing.",
)

@env.tool
def lookup_order(order_id: str) -> dict:
    """Return order status."""
    return db.get_order(order_id)

@env.scorer(weight=1.0)
def used_tool(rollout) -> float:
    return 1.0 if any(s.type == "tool" for s in rollout.trace.steps) else 0.0

@env.tasks
def tasks():
    for row in load_cases("cases.jsonl"):
        yield TaskData(task_id=row["id"], input=row["message"], expected=row.get("label"))
```

Run one task:

```python
rollout = env.rollout(task, client, model="openai/gpt-4o-mini")
print(rollout.trace.outcome.scores)
```

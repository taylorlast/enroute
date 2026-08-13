# Environment

**Scenario:** You want to know which model handles refund tickets best — not on a public leaderboard, but on *your* tools, *your* policies, and *your* definition of success.

An **Environment** is that test bed.

## One sentence

An Environment is an RL-style harness: a set of tasks, the tools/actions the agent may use, and scorers that grade the result. Its output is a scored [Trace](trace.md).

You do not need a machine-learning background. Think of it as a repeatable rehearsal of your product:

- **Tasks** — the tickets / prompts / cases to run
- **Tools** — the same APIs your agent can call in production
- **Scorers** — functions that return how well the run went (0–1, or any float)

## Minimal example

```python
from enroute import Environment, TaskData

env = Environment(name="support-triage", version="0.1.0")

@env.tool
def lookup_order(order_id: str) -> dict:
    """Look up an order by id."""
    return {"order_id": order_id, "status": "shipped"}

@env.scorer(weight=1.0)
def resolved_correctly(rollout) -> float:
    text = (rollout.response.text or "").lower()
    return 1.0 if "shipped" in text else 0.0

@env.tasks
def tasks():
    yield TaskData(task_id="t1", input="Where is order A123?")

trace = env.rollout(next(env.iter_tasks()), client, model="openai/gpt-4o-mini").trace
```

## How this differs from an "eval script"

Eval scripts usually hard-code a prompt list and a string match. Environments:

1. Expose the **same tool surface** the agent had when deciding
2. Emit the **same Trace** as production
3. Are **versioned**, so benchmark numbers stay comparable over time

## How this connects

Rollouts fill a [Dataset](dataset.md). Running the environment across models is a [Benchmark](benchmark.md).

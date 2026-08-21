# Write your first environment

1. Subclass `Environment[MyObservation, MyState]` — name, version, and `max_turns` live on the class.
2. Seed `self.state` in `setup(task)`. Implement `observe()` so reset/step know what the agent can see.
3. Decorate actions with `@tool` (or use `@env.tool` on a plain `Environment()`). A tool may call other tools; those nests are recorded.
4. Write scorers that match your definition of done. Read `rollout.env`.
5. Provide tasks (synthetic or sampled from production). Tasks carry the **goal** and a **seed**.

Drive an agent with `reset` / `step` / `rollout`. Do not call `observe` in that loop.

## Stateless eval (support triage)

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
    return 1.0 if any(s.type == "decision" and s.tool_calls for s in rollout.trace.steps) else 0.0

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

## Basic RL (library)

The smallest sequential environment we train against: a closed corpus, tools `search`, `read`, `answer`, and hierarchical `research` (calls search then read). Research has **no** per-decision reward. The scorer is 1 if the submitted answer contains the hidden fact.

```python
from enroute import Environment
from enroute.environments import Observation, State, tool

class LibraryState(State):
    submitted: str | None = None

class LibraryObservation(Observation):
    question: str = ""

class LibraryEnv(Environment[LibraryObservation, LibraryState]):
    name = "library"
    version = "0.1.0"

    @tool
    def search(self, query: str) -> dict: ...

    @tool
    def read(self, doc_id: str) -> dict: ...

    @tool
    def research(self, query: str) -> dict:
        """Search, then read the first hit."""
        found = self.search(query)
        return {"doc": self.read(found["hits"][0]["doc_id"])}

    @tool
    def answer(self, text: str) -> dict:
        """Submit the final answer and end the episode."""
        self.state.submitted = text
        return {"ok": True}
```

```bash
uv run python examples/environment/library/run.py
```

That script rolls out a researcher and a guesser, then shows `trace.returns(gamma=0.9)` so search/read inherit credit from the later correct answer. It also late-credits a reviewer score onto the answer decision — the same API you would use when a tweet's likes arrive tomorrow.

Full code: [`examples/environment/library/`](https://github.com/enroute-ai/enroute/tree/main/examples/environment/library).

## Game (Wordle)

The environment *is* the game: word lists, board, `guess`, observations, and rewards. You play it with `step`. The LLM is only the policy that picks a word.

```python
obs, info = env.reset(task)
while True:
    response = client.chat(model=model, messages=env.messages(), tools=env.tool_defs)
    obs, reward, terminated, truncated, info = env.step(
        response.message.tool_calls, request=..., response=response
    )
    if terminated or truncated:
        break
rollout = env.close_episode(client=client)
```

```bash
uv run python examples/wordle/run.py --secret crane
```

Secrets come from [`data/answers.txt`](https://github.com/enroute-ai/enroute/tree/main/examples/environment/wordle/data/answers.txt); legal guesses from [`data/allowed.txt`](https://github.com/enroute-ai/enroute/tree/main/examples/environment/wordle/data/allowed.txt). Full code: [`examples/environment/wordle/`](https://github.com/enroute-ai/enroute/tree/main/examples/environment/wordle).

## Work environment (Twitter)

A work environment *is* the simulator the tools act on. The model is passed at rollout time so you can swap it.

```python
from enroute import Environment, TaskData
from enroute.environments import Observation, State, tool

class TwitterState(State):
    account: str = "agent"

class TwitterObservation(Observation):
    briefing: str = ""
    def render(self) -> str:
        return self.briefing

class TwitterEnv(Environment[TwitterObservation, TwitterState]):
    name = "twitter-account"
    version = "0.1.0"
    system_prompt = "You operate this account. Use tools to look around and act."
    max_turns = 12

    def setup(self, task):
        super().setup(task)
        self.state = TwitterState(seed=self.seed, account=task.metadata.get("account", "agent"))

    def observe(self) -> TwitterObservation:
        return TwitterObservation(briefing=self.briefing())

    @tool
    def tweet(self, text: str) -> dict:
        """Publish a tweet from this account."""
        return self.publish(text)

    @tool
    def reply(self, post_id: str, text: str) -> dict:
        """Reply to a post."""
        return self.publish_reply(post_id, text)

env = TwitterEnv()

@env.scorer(weight=1.0)
def goal_progress(rollout) -> float:
    return rollout.env.score(rollout.task.metadata.get("goal"))

@env.tasks
def tasks():
    yield TaskData(
        task_id="reply-mentions-1",
        input="Reply to every mention.",
        metadata={"seed": 42, "account": "agent", "goal": "reply_to_mentions"},
    )
```

A full simulated Twitter account (timeline, follow, like, quote, media placeholders, two example goals) lives in [`examples/environment/twitter/`](https://github.com/enroute-ai/enroute/tree/main/examples/environment/twitter).

```bash
uv run python examples/environment/twitter/run.py
```

## Gym loop

`rollout()` is enough for most callers. A future trainer (or you) can drive the episode with `reset` / `step`. Do not call `observe` here:

```python
obs, info = env.reset(task, model="openai/gpt-4o-mini")
# ... policy calls client.chat, then:
obs, reward, terminated, truncated, info = env.step(response.message.tool_calls, request=req, response=resp)
rollout = env.close_episode(client=client)
```

Each `step` appends one **decision** to the episode [Trace](../concepts/trace.md).

## When to bump the version

See [Environment versioning](../concepts/environment.md#versioning). Short version: patch for bugfixes, minor when you add tools, major when observation shape or scorer meaning changes.

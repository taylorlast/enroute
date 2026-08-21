# Environment

**Scenario:** You want to know which model handles refund tickets best — not on a public leaderboard, but on *your* tools, *your* policies, and *your* definition of success. Or you want an agent that *runs a Twitter account* and you need a repeatable environment to train and benchmark against.

An **Environment** is that test bed.

## One sentence

An Environment is a versioned RL-style harness: `class WordleEnv(Environment[WordleObservation, WordleState])` with instructions, tools, and scorers. Its output is a scored [Trace](trace.md) — one document per episode.

You do not need a machine-learning background. Think of it as a repeatable rehearsal of your product:

- **Tasks** — the tickets / prompts / cases to run (they seed `state`)
- **State** — internal episode data, including hidden fields (the Wordle secret)
- **Observation** — what the agent is allowed to see (the board, not the secret)
- **Tools** — `@tool` methods on the env, the same APIs your agent can call in production
- **Scorers** — functions that return how well the run went (0–1, or any float)

The **model is not part of the environment**. Drive an agent with `reset` / `step` or `rollout`. `observe` is an author hook that those methods call — do not call it to test an agent.

## Gymnasium-shaped loop

`step` is the action. Same contract as Gymnasium and Hugging Face OpenEnv: `reset`, then `step` until the env says stop. `rollout()` is sugar that runs an LLM through that loop (used by Benchmark).

```python
obs, info = env.reset(task)
while True:
    action = policy(obs)          # client.chat → tool calls — not the env
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
rollout = env.close_episode(client=client)
```

| Gymnasium | enroute |
| --- | --- |
| `reset(seed)` | `reset(task)` — calls `setup(task)`, opens an episode trace |
| `step(action)` | Same 5-tuple. Default: dispatch `@tool` methods. Override if needed. |
| `action_space` | `env.tool_defs` |
| `observation_space` | `Observation` (returned by `reset` / `step`) |
| `terminated` | `env.done()` |
| `truncated` | `max_turns` hit |
| Episode return | `trace.outcome.reward` |

A **rollout is an episode**. The code keeps the `rollout` name.

## Write an environment

The environment *is* the simulator. Subclass `Environment[Obs, State]`, put episode data on `self.state`, decorate actions with `@tool`. Name and version come with the class.

```python
from enroute import Environment
from enroute.environments import Observation, State, tool

class CounterState(State):
    n: int = 0

class CounterObservation(Observation):
    n: int = 0
    def render(self) -> str:
        return f"count={self.n}"

class CounterEnv(Environment[CounterObservation, CounterState]):
    name = "counter"
    version = "0.1.0"

    def setup(self, task):
        super().setup(task)
        self.state = CounterState(seed=self.seed, n=0)

    def observe(self) -> CounterObservation:
        return CounterObservation(n=self.state.n)

    @tool
    def inc(self, by: int = 1) -> dict:
        """Increment the counter."""
        self.state.n += by
        return {"n": self.state.n}

env = CounterEnv()
obs, info = env.reset(task)
obs, reward, terminated, truncated, info = env.step(action)
```

`observe` defines the observation type. `reset` and `step` call it. Tools read `self.state` and the cached `self.observation`.

The default `step` runs those tools and records a decision. Override `step` when the action is not a tool call; use `record_decision` and `finish_turn` so the trace still matches. A `@tool` may call other `@tool` methods; those inner calls are recorded as children on the `ToolCallStep` (still one Decision).

Stateless tools still work: `@env.tool` on a plain `Environment()` (the support-triage path).

## Minimal example (no subclass)

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

## Traces from any environment

Every environment is different (library, Twitter, your CRM). The trace is not. `rollout()` always writes **one [Trace](trace.md) per episode**:

1. `reset` snapshots `initial_state` and the first observation
2. Each `step` appends a **decision** (observation, `model_context`, action, tool results)
3. `close_episode` runs scorers → `outcome.reward`, plus `final_state` and `metrics`

That is the only shape observability and a future trainer need. Environments vary in their **state and tools**; they do not invent a new log format.

```python
rollout = env.rollout(task, client, model="openai/gpt-4o-mini")
trace = rollout.trace
trace.decisions()          # what happened
trace.transitions()        # (obs, action, reward, next_obs, done)
trace.returns(gamma=0.9)   # G_t for each decision — the RL training signal
```

## Reward is injected, not assumed per decision

Most real work is **sparse or delayed**. Researching, then posting, then getting likes tomorrow is one trajectory with one outcome — not three independently scored turns.

Three injection points, none of them environment-specific:

| When | Where | Use |
| --- | --- | --- |
| During a step (optional) | `env.step_reward` → `Decision.reward_events` | Shaping, costs |
| End of episode | scorers → `outcome.reward` | The usual return (correct answer, goal hit) |
| Later (hours, humans, KPIs) | `trace.credit(value, decision_index=…)` or `client.label(trace_id, reward=…)` | Likes, reviews, downstream success |

**Do not** have the environment assign credit backward onto `search` because a later `answer` was good. Persist the episode; the trainer walks it:

`G_t = r_t + γ G_{t+1}` via `trace.returns(gamma=…)`.

- `source="outcome"` (default) — zeros, then the scorer on the last decision. Research-then-answer.
- `source="events"` — only `reward_events` (including late `credit` on a specific decision, e.g. likes on the tweet).
- `source="both"` — events plus the terminal scorer.

The start-to-finish example is the [library environment](../guides/write-environment.md#basic-rl-library): search / read / answer, no step rewards, `returns(γ=0.9)` credits the research.

## Versioning

`version` is a **compatibility key**, not a label. Traces from `twitter-account@0.2.0` are not comparable to `0.1.0` if tools, observations, or reward semantics changed.

Each episode also stores `environment_fingerprint` — a hash of name, version, observation/state type names, tool JSON schemas, instructions, and scorer names + weights. Two episodes with the same fingerprint saw the same action surface and scoring contract.

Bump rules:

- **patch** — bugfix, docs, deterministic seed fix; traces stay comparable
- **minor** — add tools or tasks; old traces still valid
- **major** — remove/rename tools, change `observe()` shape, or change what a scorer means

Datasets record fingerprints in metadata. Benchmark reports include the fingerprint.

## How this differs from an "eval script"

Eval scripts usually hard-code a prompt list and a string match. Environments:

1. Expose the **same tool surface** the agent had when deciding
2. Emit the **same Trace** as production (one episode, steps are decisions)
3. Are **versioned**, so benchmark numbers stay comparable over time

## How this connects

Rollouts fill a [Dataset](dataset.md). Running the environment across models is a [Benchmark](benchmark.md).

# Quickstart

Get a multi-provider chat call in under a minute.

## Install

```bash
pip install enroute
```

## Set your API key

```bash
export ENROUTE_API_KEY=enroute-...
```

## Primary: one enroute API key

Looks like the OpenAI SDK — construct a client, call `chat`, close when done.
`Enroute()` reads `ENROUTE_API_KEY` from the environment automatically:

```python
from enroute import Enroute, Message

client = Enroute()

response = client.chat(
    model="openai/gpt-4o-mini",
    messages=[Message(role="user", content="Summarize enroute in one sentence.")],
    models=["anthropic/claude-sonnet-4"],  # optional fallback chain
)
print(response.text)
print(response.usage.cost)
client.close()
```

Traces append to `.enroute/traces.jsonl` by default.

## Tracing style

Use a context manager when you want explicit lifecycle + content capture:

```python
from enroute import Enroute, Message
from enroute.tracing import JSONLSink

with Enroute(sink=JSONLSink(".enroute/traces.jsonl"), capture_content=True) as client:
    response = client.chat(
        model="anthropic/claude-sonnet-4",
        messages=[Message(role="user", content="Hello")],
    )
    print(response.text)
```

## Secondary: bring your own keys (BYOK)

Pass upstream keys explicitly — they are not loaded unless you ask:

```python
import os
from enroute import Enroute

client = Enroute(
    providers={
        "openai": os.environ["OPENAI_API_KEY"],
        "anthropic": os.environ["ANTHROPIC_API_KEY"],
    },
)
```

Prefer the enroute API key for product traffic. See [Routing examples](guides/routing-examples.md).

## Fallback and cost-aware routing

```python
from enroute import Enroute
from enroute.routing import LeastCost

client = Enroute(policy=LeastCost())
```

## Next concepts

1. [Trace](concepts/trace.md) — what got recorded
2. [Environment](concepts/environment.md) — turn tasks into scored traces
3. [Benchmark](concepts/benchmark.md) — compare models on your environment

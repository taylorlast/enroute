# Routing policy

**Scenario:** `gpt-4o` is down or rate-limited. You want the request to fall back to Claude, prefer cheaper models when quality is equal, and leave a seat for a learned autorouter later.

A **RoutingPolicy** decides the ordered list of model routes to try.

## Built-ins

| Policy | Behavior |
| --- | --- |
| `Explicit` / `Fallback` | Try `model`, then `models=[...]` in order |
| `LeastCost` | Keep primary first; sort fallbacks by catalog price |
| `LowestLatency` | Heuristic provider latency ranking |

```python
from enroute import Enroute
from enroute.routing import LeastCost
from enroute.types import ProviderPreferences

client = Enroute(providers={...}, policy=LeastCost())
client.chat(
    model="openai/gpt-4o",
    messages=[...],
    models=["openai/gpt-4o-mini", "google/gemini-2.5-flash"],
    provider=ProviderPreferences(ignore=["together"], sort="price"),
)
```

## The autorouter seat

`RoutingPolicy.select(request, candidates, catalog) -> list[ModelRoute]` is intentionally tiny. A future learned policy implements the same protocol — no client rewrite.

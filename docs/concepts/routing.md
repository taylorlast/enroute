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

## Hosts and regions

A model can be served by several hosts, and the catalog lists each one with its
own price. A route therefore names a provider *and* a region, because the same
provider charges differently by region — Azure EU lists above Azure US for
identical models.

Routes whose host is not configured are dropped before the policy runs, so
listing a cloud endpoint in the catalog does not break requests for callers who
have no credentials for it. A provider bound to one region is only offered routes
in that region, so a request is never billed at a region's rate it did not use.

## Prompt-length pricing

Some models charge more once a prompt crosses a threshold. A tier **replaces** the
base rate for the whole request rather than applying only to tokens past the
threshold, which is how the providers themselves bill it:

```python
pricing = catalog.require("openai/gpt-5.6-sol").pricing
pricing.rates_for_prompt(100_000)   # base rate
pricing.rates_for_prompt(300_000)   # tier rate, applied to every token
```

`usage.cost` accounts for this on both `chat` and `stream`, using the real prompt
token count reported by the host.

## The autorouter seat

`RoutingPolicy.select(request, candidates, catalog) -> list[ModelRoute]` is intentionally tiny. A future learned policy implements the same protocol — no client rewrite.

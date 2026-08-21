# Dataset

**Scenario:** After a week of production traffic (and a few environment rollouts), you want a frozen, versioned collection to benchmark new models against.

A **Dataset** is that collection.

## One sentence

A Dataset is a named, content-hashed set of traces (plus labels) used for benchmarks and future training.

```python
from enroute import Dataset

# From environment rollouts
ds = Dataset.from_traces("support-week-12", [r.trace for r in rollouts], version="2026.08.12")

# From production sinks
ds = Dataset.from_sink(
    ".enroute/traces.jsonl",
    "prod-refunds",
    where=lambda t: t.tags.get("intent") == "refund",
)
ds.save("data/prod-refunds.jsonl")
```

## Content hash

`content_hash` fingerprints trace ids and outcomes so you can tell whether two dataset builds are identical — important for CI and paper-trail reproducibility. `Dataset.from_traces` also records `environment_fingerprints` in metadata when traces carry them.

## How this connects

Datasets are what [Benchmarks](benchmark.md) evaluate against, and what exporters send to Hugging Face or verifiers-style trainers.

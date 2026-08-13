# Turn production traffic into a dataset

```python
from enroute import Dataset

ds = Dataset.from_sink(
    ".enroute/traces.jsonl",
    name="prod-refunds-2026-w32",
    version="2026.08.12",
    where=lambda t: t.tags.get("intent") == "refund" and t.outcome is not None,
)
ds.save("data/prod-refunds-2026-w32.jsonl")
print(ds.content_hash, len(ds))
```

Only include traces that already have outcomes (human labels or online rewards). Unlabeled traffic is useful for inspection but noisy for benchmarks.

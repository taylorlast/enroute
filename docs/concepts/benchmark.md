# Benchmark

**Scenario:** You have an environment for refund quality. You want a table: model × mean reward × cost × latency, plus pairwise win rates — something you can paste into a PR.

A **Benchmark** produces that report.

```python
from enroute import Benchmark

report = Benchmark(
    env,
    models=["openai/gpt-4o-mini", "anthropic/claude-sonnet-4", "google/gemini-2.5-flash"],
    client=client,
    repeats=3,
    concurrency=8,
).run()

print(report.to_markdown())
print(report.compare(baseline))  # CI regression check
```

## One sentence

A Benchmark runs an environment across models (with optional repeats) and aggregates scores, cost, latency, win rates, and failure buckets.

## How this connects

Tomorrow's autorouter will be trained on the same traces these benchmarks score. Getting the measurement loop right is the product.

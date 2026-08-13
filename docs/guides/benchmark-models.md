# Benchmark models

```python
from enroute import Benchmark

report = Benchmark(
    env,
    models=[
        "openai/gpt-4o-mini",
        "anthropic/claude-sonnet-4",
        "google/gemini-2.5-flash",
    ],
    client=client,
    repeats=3,
    concurrency=8,
).run()

Path("report.md").write_text(report.to_markdown())
Path("report.json").write_text(report.to_json())
```

In CI, fail on regressions:

```python
baseline = Report.model_validate_json(Path("baseline.json").read_text())
comparison = report.compare(baseline)
assert not comparison["regressions"], comparison
```

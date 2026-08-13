# Redact PII before anything hits disk

```python
from enroute import Enroute, Redactor
from enroute.tracing import JSONLSink

redactor = Redactor(
    fields={"metadata.email", "metadata.phone"},
    patterns=[
        r"\b\d{3}-\d{2}-\d{4}\b",          # SSN-like
        r"\b[\w.-]+@[\w.-]+\.\w+\b",     # emails in content
    ],
    drop_content=False,  # keep content but scrub patterns
)

client = Enroute(
    providers={"openai": "..."},
    sink=JSONLSink(".enroute/traces.jsonl"),
    redactor=redactor,
    capture_content=True,
)
```

Defaults: if you leave `capture_content=False` (the default), message bodies are omitted entirely before persistence.

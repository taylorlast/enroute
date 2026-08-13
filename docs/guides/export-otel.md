# Export to OpenTelemetry

```bash
pip install "enroute[otel]"
```

```python
from enroute import Enroute
from enroute.tracing import MultiSink, JSONLSink, OTelSink

# Configure your TracerProvider elsewhere (ODLP, Datadog, Honeycomb, ...).
client = Enroute(
    providers={"openai": "..."},
    sink=MultiSink([JSONLSink(".enroute/traces.jsonl"), OTelSink()]),
)
```

enroute's `Trace` object remains canonical. `OTelSink` maps LLM steps onto Development-status `gen_ai.*` attributes. Prefer the enroute schema for datasets and training; use OTel for live ops dashboards.

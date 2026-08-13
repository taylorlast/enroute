"""Google Gemini model via enroute, with local tracing.

Requires ``ENROUTE_API_KEY`` in your environment::

    export ENROUTE_API_KEY=enroute-...
    uv run python examples/routing/models/google/with_tracing.py
"""

from pathlib import Path

from enroute import Enroute, Message
from enroute.tracing import JSONLSink

Path(".enroute/examples").mkdir(parents=True, exist_ok=True)

with Enroute(
    sink=JSONLSink(".enroute/examples/routing-google.jsonl"),
    capture_content=True,
) as client:
    response = client.chat(
        model="google/gemini-2.5-flash",
        messages=[Message(role="user", content="In one sentence, what is multimodal AI?")],
        temperature=0.2,
        max_tokens=128,
    )
    print(response.text)
    print("trace_id:", (response.raw or {}).get("enroute_trace_id"))

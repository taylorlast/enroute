"""Same quickstart, with an explicit local trace sink.

Requires ``ENROUTE_API_KEY`` in your environment::

    export ENROUTE_API_KEY=enroute-...
    uv run python examples/routing/quickstart/with_tracing.py
"""

from pathlib import Path

from enroute import Enroute, Message
from enroute.tracing import JSONLSink

Path(".enroute/examples").mkdir(parents=True, exist_ok=True)

with Enroute(
    sink=JSONLSink(".enroute/examples/routing-quickstart.jsonl"),
    capture_content=True,
) as client:
    response = client.chat(
        model="openai/gpt-4o-mini",
        messages=[
            Message(role="user", content="Reply with one short sentence introducing yourself.")
        ],
        models=["anthropic/claude-sonnet-4"],
    )
    print(response.text)
    print("trace_id:", (response.raw or {}).get("enroute_trace_id"))

print("traces → .enroute/examples/routing-quickstart.jsonl")

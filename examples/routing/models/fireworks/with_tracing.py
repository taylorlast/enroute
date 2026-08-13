"""Fireworks model via enroute, with local tracing.

Requires ``ENROUTE_API_KEY`` in your environment::

    export ENROUTE_API_KEY=enroute-...
    uv run python examples/routing/models/fireworks/with_tracing.py
"""

from pathlib import Path

from enroute import Enroute, Message
from enroute.tracing import JSONLSink

Path(".enroute/examples").mkdir(parents=True, exist_ok=True)

with Enroute(
    sink=JSONLSink(".enroute/examples/routing-fireworks.jsonl"),
    capture_content=True,
) as client:
    response = client.chat(
        model="fireworks/accounts/fireworks/models/llama-v3p3-70b-instruct",
        messages=[
            Message(
                role="user",
                content="In one sentence, why do teams use inference specialists like Fireworks?",
            )
        ],
        temperature=0.2,
        max_tokens=128,
    )
    print(response.text)
    print("trace_id:", (response.raw or {}).get("enroute_trace_id"))

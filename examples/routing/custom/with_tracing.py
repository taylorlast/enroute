"""Your own model, with local tracing.

export CUSTOM_BASE_URL=http://127.0.0.1:8000/v1
export CUSTOM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
uv run python examples/routing/custom/with_tracing.py
"""

import os
from pathlib import Path

from enroute import Enroute, Message
from enroute.providers import OpenAICompatible
from enroute.tracing import JSONLSink

base_url = os.environ.get("CUSTOM_BASE_URL", "http://127.0.0.1:8000/v1")
api_key = os.environ.get("CUSTOM_API_KEY", "EMPTY")
server_model = os.environ.get("CUSTOM_MODEL", "my-model")

Path(".enroute/examples").mkdir(parents=True, exist_ok=True)

with Enroute(
    providers={
        "custom": OpenAICompatible(
            api_key=api_key,
            base_url=base_url,
            name="custom",
        )
    },
    sink=JSONLSink(".enroute/examples/routing-custom.jsonl"),
    capture_content=True,
) as client:
    response = client.chat(
        model=f"custom/{server_model}",
        messages=[Message(role="user", content="In one sentence, introduce yourself.")],
        temperature=0.2,
        max_tokens=128,
    )
    print(response.text)
    print("trace_id:", (response.raw or {}).get("enroute_trace_id"))

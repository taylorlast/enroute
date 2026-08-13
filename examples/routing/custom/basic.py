"""Chat with your own OpenAI-compatible model (vLLM, Ollama, etc.).

export CUSTOM_BASE_URL=http://127.0.0.1:8000/v1
export CUSTOM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
export CUSTOM_API_KEY=EMPTY   # optional
uv run python examples/routing/custom/basic.py
"""

import os

from enroute import Enroute, Message
from enroute.providers import OpenAICompatible

base_url = os.environ.get("CUSTOM_BASE_URL", "http://127.0.0.1:8000/v1")
api_key = os.environ.get("CUSTOM_API_KEY", "EMPTY")
server_model = os.environ.get("CUSTOM_MODEL", "my-model")

client = Enroute(
    providers={
        "custom": OpenAICompatible(
            api_key=api_key,
            base_url=base_url,
            name="custom",
        )
    }
)

response = client.chat(
    model=f"custom/{server_model}",
    messages=[Message(role="user", content="In one sentence, introduce yourself.")],
    temperature=0.2,
    max_tokens=128,
)

print(response.model)
print(response.text)
client.close()

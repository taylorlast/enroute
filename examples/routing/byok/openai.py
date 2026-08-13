"""Bring-your-own-key: call an upstream provider with your own credentials.

Prefer ``ENROUTE_API_KEY`` for product traffic. Use BYOK when you must talk to
an upstream directly.

    export OPENAI_API_KEY=sk-...
    uv run python examples/routing/byok/openai.py
"""

import os

from enroute import Enroute, Message

client = Enroute(providers={"openai": os.environ["OPENAI_API_KEY"]})

response = client.chat(
    model="openai/gpt-4o-mini",
    messages=[Message(role="user", content="Say hello in one short sentence.")],
)

print(response.model)
print(response.text)
client.close()

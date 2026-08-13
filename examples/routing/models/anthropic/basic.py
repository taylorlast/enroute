"""Chat with an Anthropic model via your enroute API key.

Requires ``ENROUTE_API_KEY`` in your environment::

    export ENROUTE_API_KEY=enroute-...
    uv run python examples/routing/models/anthropic/basic.py
"""

from enroute import Enroute, Message

client = Enroute()

response = client.chat(
    model="anthropic/claude-sonnet-4",
    messages=[Message(role="user", content="In one sentence, what is constitutional AI?")],
    temperature=0.2,
    max_tokens=128,
)

print(response.model)
print(response.text)
client.close()

"""Quickstart: one enroute API key, one chat call.

Requires ``ENROUTE_API_KEY`` in your environment::

    export ENROUTE_API_KEY=enroute-...
    uv run python examples/routing/quickstart/basic.py
"""

from enroute import Enroute, Message

client = Enroute()

response = client.chat(
    model="openai/gpt-4o-mini",
    messages=[Message(role="user", content="Reply with one short sentence introducing yourself.")],
    models=["anthropic/claude-sonnet-4"],  # optional fallbacks
)

print(response.text)
client.close()

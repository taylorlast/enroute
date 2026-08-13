"""Chat with a Fireworks model via your enroute API key.

Requires ``ENROUTE_API_KEY`` in your environment::

    export ENROUTE_API_KEY=enroute-...
    uv run python examples/routing/models/fireworks/basic.py
"""

from enroute import Enroute, Message

client = Enroute()

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

print(response.model)
print(response.text)
client.close()

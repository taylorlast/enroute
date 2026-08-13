"""Chat with an OpenAI model via your enroute API key.

Requires ``ENROUTE_API_KEY`` in your environment::

    export ENROUTE_API_KEY=enroute-...
    uv run python examples/routing/models/openai/basic.py
"""

from enroute import Enroute, Message

client = Enroute()

response = client.chat(
    model="openai/gpt-4o-mini",
    messages=[
        Message(role="user", content="In one sentence, what is a transformer neural network?")
    ],
    temperature=0.2,
    max_tokens=128,
)

print(response.model)
print(response.text)
client.close()

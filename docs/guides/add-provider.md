# Add your own provider

Implement the four methods on the :class:`~enroute.providers.base.Provider` protocol and translate to enroute's normalized types.

```python
from enroute.providers.base import Provider
from enroute.types import ChatRequest, ChatResponse, StreamChunk

class MyProvider:
    name = "myvendor"

    def chat(self, request: ChatRequest) -> ChatResponse: ...
    def stream(self, request: ChatRequest): ...
    async def achat(self, request: ChatRequest) -> ChatResponse: ...
    async def astream(self, request: ChatRequest): ...
    def close(self) -> None: ...
    async def aclose(self) -> None: ...
```

Wire it in:

```python
from enroute import Enroute

client = Enroute(providers={"myvendor": MyProvider(...)})
```

Prefer subclassing :class:`~enroute.providers.openai_compatible.OpenAICompatible` when the vendor speaks the OpenAI Chat Completions shape — you usually only need a different `base_url` and `name`.

Streaming is part of the same contract. `stream` and `astream` must yield
:class:`~enroute.types.StreamChunk` as tokens arrive: `delta.content` for text,
`finish_reason` and `usage` on the last chunk. Keep the vendor event on
`raw` for debugging. Never expect a caller to read `raw` — hosted gateways
serialize with :meth:`~enroute.types.StreamChunk.to_openai` so every host
looks like OpenAI Chat Completions SSE.

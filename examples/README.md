# enroute examples

## Routing (start here)

See [`routing/README.md`](routing/README.md) for the full walkthrough.

```bash
export ENROUTE_API_KEY=enroute-...

uv run python examples/routing/quickstart/basic.py
uv run python examples/routing/catalog/list_models.py
uv run python examples/routing/models/openai/basic.py
uv run python examples/routing/models/anthropic/basic.py
uv run python examples/routing/models/google/basic.py
uv run python examples/routing/models/fireworks/basic.py
```

```python
from enroute import Enroute, Message

client = Enroute()
response = client.chat(
    model="openai/gpt-4o-mini",
    messages=[Message(role="user", content="Hello")],
)
print(response.text)
client.close()
```

| Path | Purpose |
| --- | --- |
| [`routing/quickstart/`](routing/quickstart/) | First call (basic + tracing) |
| [`routing/catalog/`](routing/catalog/) | List model ids |
| [`routing/models/`](routing/models/) | OpenAI, Anthropic, Google, Fireworks |
| [`routing/custom/`](routing/custom/) | Your own OpenAI-compatible server |
| [`routing/byok/`](routing/byok/) | Bring-your-own upstream keys |

## Other folders

| Folder | What it covers |
| --- | --- |
| [`tracing/`](tracing/) | Redaction, sinks, late labels |
| [`environment/`](environment/) | Tasks, tools, scorers → datasets |
| [`environment/library/`](environment/library/) | Basic RL: search / read / answer, then `returns(γ)` |
| [`environment/wordle/`](environment/wordle/) | Wordle: `uv run python examples/wordle/run.py --secret crane` |
| [`environment/twitter/`](environment/twitter/) | `TwitterEnv`: run a simulated Twitter account |
| [`benchmarking/`](benchmarking/) | Compare models on an environment |

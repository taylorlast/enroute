# Routing examples

Set your key once:

```bash
export ENROUTE_API_KEY=enroute-...
```

## Layout

```
routing/
  quickstart/     # start here
  catalog/        # list available model ids
  models/         # openai, anthropic, google, fireworks
  custom/         # your own OpenAI-compatible endpoint
  byok/           # bring-your-own upstream keys (secondary)
```

Each topic has a **basic** script (`Enroute()` + `client.chat`) and, where useful, a **with_tracing** variant (`with Enroute(...) as client` + local sink).

## Walkthrough

```bash
# 1. Quickstart
uv run python examples/routing/quickstart/basic.py
uv run python examples/routing/quickstart/with_tracing.py

# 2. See which models enroute knows about
uv run python examples/routing/catalog/list_models.py
uv run python examples/routing/catalog/list_models.py --provider openai

# 3. Provider models (via your enroute key)
uv run python examples/routing/models/openai/basic.py
uv run python examples/routing/models/anthropic/basic.py
uv run python examples/routing/models/google/basic.py
uv run python examples/routing/models/fireworks/basic.py

# 4. Your own model
export CUSTOM_BASE_URL=http://127.0.0.1:8000/v1
export CUSTOM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
uv run python examples/routing/custom/basic.py

# 5. Optional BYOK
export OPENAI_API_KEY=sk-...
uv run python examples/routing/byok/openai.py
```

## Shape

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

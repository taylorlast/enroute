# Wordle environment

The environment *is* the game: secret, board, `guess`, observations, and rewards. You play it with `reset` / `step`. The model is only the policy that picks a word.

## Run

From the repo root:

```bash
uv run python examples/wordle/run.py --secret crane
```

That is a shortcut for [`examples/environment/wordle/run.py`](run.py). `--secret` is the hidden 5-letter answer. It must be in the Wordle word list (`data/allowed.txt` / `data/answers.txt`). If you omit it, the secret is `crane`.

```bash
uv run python examples/wordle/run.py
uv run python examples/wordle/run.py --secret slate
```

Without `--model`, the script runs two offline policies (no API key):

- **solver** — guesses `slate`, then the secret
- **misses** — six valid words that are not the secret

Each episode prints the board, per-step rewards, and discounted returns. Traces land in the example output directory as a small dataset.

## Local model (laptop)

Point `--model` at the name your server exposes and `--base-url` at its OpenAI-compatible `/v1` endpoint. The model is the policy; it must support tool calls (`guess`).

```bash
# Ollama
uv run python examples/wordle/run.py --secret crane \
  --model llama3.2 --base-url http://127.0.0.1:11434/v1

# LM Studio
uv run python examples/wordle/run.py --secret crane \
  --model local-model --base-url http://127.0.0.1:1234/v1

# vLLM / llama.cpp
uv run python examples/wordle/run.py --secret crane \
  --model meta-llama/Llama-3.1-8B-Instruct --base-url http://127.0.0.1:8000/v1
```

`--api-key` defaults to `EMPTY`. Pass a real key only if your server requires one.

## Hosted model

If the model is already configured on your Enroute client (env keys, etc.):

```bash
uv run python examples/wordle/run.py --secret crane --model openai/gpt-4o-mini
```

## Play loop

```python
obs, info = env.reset(task)
while True:
    response = client.chat(model=model, messages=env.messages(), tools=env.tool_defs)
    obs, reward, terminated, truncated, info = env.step(
        response.message.tool_calls, request=..., response=response
    )
    if terminated or truncated:
        break
rollout = env.close_episode(client=client)
```

Do not call `observe` to drive the agent. `reset` and `step` already return the observation.

## Files

| Path | Role |
| --- | --- |
| [`env.py`](env.py) | `WordleEnv`, `WordleObservation`, `WordleState` |
| [`words.py`](words.py) | Load lists, `pattern()`, `is_allowed()` |
| [`data/answers.txt`](data/answers.txt) | Official secrets |
| [`data/allowed.txt`](data/allowed.txt) | Legal guesses |
| [`run.py`](run.py) | CLI + scripted `reset` / `step` loop |

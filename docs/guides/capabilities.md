# Streaming, tool calling, structured output, and reasoning

Every host expresses these four features differently. enroute normalizes them so
one piece of client code works across all of them, and the normalized shape is
always OpenAI's.

## Streaming

`client.stream(...)` yields `StreamChunk` objects as tokens arrive. Text is on
`chunk.delta.content`, and usage lands on the final chunk so the request can be
billed.

```python
for chunk in client.stream(model="anthropic/claude-sonnet-4", messages=messages):
    if chunk.delta.content:
        print(chunk.delta.content, end="", flush=True)
```

Serving this over HTTP means calling `chunk.to_openai()`, which produces a
`chat.completion.chunk` object. Do not forward `chunk.raw`: that is the
host-native event, and an Anthropic or Bedrock event has no
`choices[0].delta.content` for a client to read.

!!! note "A single burst of text is usually the model, not the transport"
    Hosts batch text into deltas of roughly 60–150 characters, so a short answer
    can arrive in one delta. A thinking model also produces no `content` at all
    until its reasoning finishes. Both look like buffering. To confirm a stream
    is incremental, ask for a few hundred words and watch the gaps between
    chunks.

## Tool calling

Tools are declared in OpenAI's shape and translated per host — Anthropic's
`tools`, Gemini's `functionDeclarations`, Bedrock's `toolConfig`.

Streaming a tool call yields OpenAI tool-call fragments. Hosts that send
arguments in pieces (Anthropic's `input_json_delta`, Bedrock's `toolUse.input`)
are reassembled by index, so accumulate `function.arguments` per `index` and
parse once the stream ends:

```python
calls: dict[int, str] = {}
for chunk in client.stream(model=model, messages=messages, tools=tools):
    for fragment in chunk.delta.tool_calls or []:
        index = fragment["index"]
        calls[index] = calls.get(index, "") + (
            fragment.get("function", {}).get("arguments") or ""
        )

arguments = {index: json.loads(raw) for index, raw in calls.items()}
```

The opening fragment for each call carries `id` and `function.name`; later
fragments carry only argument text.

## Structured output

`response_format` works on every host.

```python
resp = client.chat(
    model=model,
    messages=messages,
    response_format=ResponseFormat(
        type="json_schema",
        json_schema={"name": "city", "schema": schema},
    ),
)
data = json.loads(resp.text)
```

How it is delivered depends on what the host supports:

| Host | Mechanism |
| --- | --- |
| OpenAI | `/v1/responses` `text.format` (Chat Completions for `stop`/`seed` only) |
| Other OpenAI-compatible | Native `response_format` |
| Gemini | Native `responseMimeType` and `responseSchema` |
| Anthropic, Bedrock | A forced single-tool call whose input schema is your schema |

The emulation is invisible to callers. The JSON arrives as message content (and
streams as `delta.content`), never as a tool call, and `finish_reason` is `stop`
rather than `tool_calls`. Emulation is skipped when you pass your own `tools`,
since forcing the shim would make your tools unreachable — pass a schema *or*
tools, not both.

Gemini's `responseSchema` takes an OpenAPI 3.0 subset, so JSON Schema
bookkeeping keys such as `additionalProperties` and `$schema` are pruned before
the request is sent.

`{"type": "json_object"}` has no schema to enforce. Gemini gets a JSON mime
type; Anthropic and Bedrock get a system instruction to reply with a single JSON
object.

## Reasoning

Thinking models emit reasoning before any answer. It arrives on
`delta.reasoning` while streaming and on `message.reasoning` otherwise, kept
separate from `content` because callers usually render it differently.

```python
thinking = False
for chunk in client.stream(model=model, messages=messages):
    if chunk.delta.reasoning_started:
        thinking = True
        render_thinking_started()
    if chunk.delta.reasoning:
        render_thinking(chunk.delta.reasoning)
    if chunk.delta.reasoning_finished:
        thinking = False
        render_thinking_finished()
    if chunk.delta.content:
        render_answer(chunk.delta.content)
```

Reasoning is read from `/v1/responses` summary deltas on OpenAI, from
`reasoning` or `reasoning_content` on other OpenAI-compatible hosts, from
`thinking` blocks on Anthropic, from `thought` parts on Gemini, and from
`reasoningContent` on Bedrock. OpenAI Chat Completions never reports it, which
is why tool-and-reasoning requests go to Responses.

`reasoning_signature` is the host's attestation of a reasoning block. Anthropic
and Bedrock reject a thinking block that is replayed without it, so pass both
`reasoning` and `reasoning_signature` back on the assistant message to continue
a thinking conversation. Reasoning is never sent to OpenAI-compatible hosts,
which reject unknown message keys.

!!! note "Some models encrypt their reasoning"
    Anthropic's current models (Fable, Opus 5, Sonnet 5) often return a thinking
    block with a signature but no readable text. `delta.reasoning` is empty;
    `delta.reasoning_started` and `delta.reasoning_finished` still fire, which
    is what a UI uses to show a thinking indicator the way Cursor does.
    Anthropic requests ask for `thinking.type=adaptive` automatically. Override
    or disable it with `extra={"thinking": ...}`.

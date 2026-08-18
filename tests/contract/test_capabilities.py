"""Cross-provider contract for reasoning, streamed tool calls, and structured output.

These are the features callers assume a routing layer normalizes. Each host
expresses them differently — Anthropic indexes content blocks, Gemini flags
thought parts, Bedrock frames binary events — so the assertions here are on the
normalized shape rather than on any host's wire format.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import respx

from enroute.providers import (
    AnthropicProvider,
    BedrockProvider,
    GoogleProvider,
    OpenAIProvider,
)
from enroute.providers.bedrock import encode_event_frame
from enroute.providers.structured import STRUCTURED_TOOL_NAME, gemini_schema
from enroute.types import (
    ChatRequest,
    FunctionDefinition,
    Message,
    ResponseFormat,
    Tool,
)

SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}
WEATHER_TOOL = Tool(
    function=FunctionDefinition(
        name="get_weather",
        description="Look up the weather",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
    )
)


def _request(**kwargs: Any) -> ChatRequest:
    kwargs.setdefault("model", "anthropic/claude-sonnet-4")
    kwargs.setdefault("messages", [Message(role="user", content="Hi")])
    return ChatRequest(**kwargs)


def _sse(*events: dict[str, Any]) -> bytes:
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode()


def _sent(route: respx.Route) -> dict[str, Any]:
    return json.loads(route.calls.last.request.content)


def _joined_tool_arguments(chunks: list[Any]) -> dict[int, str]:
    """Reassemble streamed tool arguments per tool-call index."""
    merged: dict[int, str] = {}
    for chunk in chunks:
        for fragment in chunk.delta.tool_calls or []:
            index = fragment["index"]
            merged[index] = merged.get(index, "") + (
                (fragment.get("function") or {}).get("arguments") or ""
            )
    return merged


# --------------------------------------------------------------------------
# Reasoning
# --------------------------------------------------------------------------


@respx.mock
def test_anthropic_streams_thinking_before_the_answer() -> None:
    """The case that looked like a buffering bug: reasoning first, then text.

    A thinking model produces no ``content`` for seconds. Dropping the thinking
    deltas leaves the caller with nothing to render until the answer lands.
    """
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            content=_sse(
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1",
                        "model": "claude-fable-5",
                        "usage": {"input_tokens": 12, "output_tokens": 0},
                    },
                },
                {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}},
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "Counting "},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "to twenty."},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "signature_delta", "signature": "sig-abc"},
                },
                {"type": "content_block_stop", "index": 0},
                {"type": "content_block_start", "index": 1, "content_block": {"type": "text"}},
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "text_delta", "text": "1\n2\n3"},
                },
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 9},
                },
            ),
            headers={"Content-Type": "text/event-stream"},
        )
    )
    provider = AnthropicProvider("sk-ant", base_url="https://api.anthropic.com")
    chunks = list(provider.stream(_request(stream=True)))
    provider.close()

    assert "".join(c.delta.reasoning or "" for c in chunks) == "Counting to twenty."
    assert "".join(c.delta.content or "" for c in chunks) == "1\n2\n3"
    assert any(c.delta.reasoning_signature == "sig-abc" for c in chunks)
    # Reasoning must precede content, so a UI can show progress immediately.
    first_reasoning = next(i for i, c in enumerate(chunks) if c.delta.reasoning)
    first_content = next(i for i, c in enumerate(chunks) if c.delta.content)
    assert first_reasoning < first_content
    # And it must survive the OpenAI serialization the gateway emits.
    payload = chunks[first_reasoning].to_openai()
    assert payload["choices"][0]["delta"]["reasoning"] == "Counting "
    assert "content" not in payload["choices"][0]["delta"]


@respx.mock
def test_anthropic_returns_reasoning_on_a_non_streamed_call() -> None:
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-fable-5",
                "content": [
                    {"type": "thinking", "thinking": "Let me think.", "signature": "sig-1"},
                    {"type": "text", "text": "42"},
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 4, "output_tokens": 6},
            },
        )
    )
    provider = AnthropicProvider("sk-ant", base_url="https://api.anthropic.com")
    resp = provider.chat(_request())
    provider.close()
    assert resp.text == "42"
    assert resp.message.reasoning == "Let me think."
    assert resp.message.reasoning_signature == "sig-1"


@respx.mock
def test_anthropic_replays_a_thinking_block_with_its_signature() -> None:
    """Anthropic rejects a thinking block replayed without its attestation."""
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_2",
                "model": "claude-fable-5",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 9, "output_tokens": 1},
            },
        )
    )
    provider = AnthropicProvider("sk-ant", base_url="https://api.anthropic.com")
    provider.chat(
        _request(
            messages=[
                Message(role="user", content="Think"),
                Message(
                    role="assistant",
                    content="42",
                    reasoning="Because.",
                    reasoning_signature="sig-1",
                ),
                Message(role="user", content="Again"),
            ]
        )
    )
    provider.close()
    blocks = _sent(route)["messages"][1]["content"]
    assert blocks[0] == {
        "type": "thinking",
        "thinking": "Because.",
        "signature": "sig-1",
    }
    assert blocks[1]["text"] == "42"


@respx.mock
def test_openai_compatible_reads_reasoning_under_either_key() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=_sse(
                {
                    "id": "c1",
                    "model": "deepseek-reasoner",
                    "choices": [{"index": 0, "delta": {"reasoning_content": "hmm"}}],
                },
                {
                    "id": "c1",
                    "model": "deepseek-reasoner",
                    "choices": [{"index": 0, "delta": {"reasoning": "more"}}],
                },
                {
                    "id": "c1",
                    "model": "deepseek-reasoner",
                    "choices": [
                        {"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}
                    ],
                },
            ),
            headers={"Content-Type": "text/event-stream"},
        )
    )
    provider = OpenAIProvider("sk-test", transport="chat")
    chunks = list(provider.stream(_request(model="openai/gpt-5", stream=True)))
    provider.close()
    assert "".join(c.delta.reasoning or "" for c in chunks) == "hmmmore"
    assert "".join(c.delta.content or "" for c in chunks) == "done"


@respx.mock
def test_reasoning_is_never_sent_back_to_an_openai_host() -> None:
    """OpenAI 400s on unknown message keys, so reasoning is read-only there."""
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "c1",
                "model": "gpt-5",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )
    )
    provider = OpenAIProvider("sk-test", transport="chat")
    provider.chat(
        _request(
            model="openai/gpt-5",
            messages=[
                Message(
                    role="assistant",
                    content="hi",
                    reasoning="secret",
                    reasoning_signature="sig",
                )
            ],
        )
    )
    provider.close()
    sent = _sent(route)["messages"][0]
    assert "reasoning" not in sent
    assert "reasoning_signature" not in sent


@respx.mock
def test_google_maps_thought_parts_to_reasoning() -> None:
    model = "gemini-3-pro"
    respx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"
    ).mock(
        return_value=httpx.Response(
            200,
            content=_sse(
                {
                    "responseId": "r1",
                    "candidates": [{"content": {"parts": [{"text": "planning", "thought": True}]}}],
                },
                {
                    "responseId": "r1",
                    "candidates": [
                        {"content": {"parts": [{"text": "answer"}]}, "finishReason": "STOP"}
                    ],
                    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
                },
            ),
            headers={"Content-Type": "text/event-stream"},
        )
    )
    provider = GoogleProvider("key")
    chunks = list(provider.stream(_request(model=f"google/{model}", stream=True)))
    provider.close()
    assert "".join(c.delta.reasoning or "" for c in chunks) == "planning"
    assert "".join(c.delta.content or "" for c in chunks) == "answer"


# --------------------------------------------------------------------------
# Streamed tool calls
# --------------------------------------------------------------------------


@respx.mock
def test_anthropic_streams_tool_arguments_as_openai_fragments() -> None:
    """Anthropic sends argument fragments; callers expect OpenAI tool deltas."""
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            content=_sse(
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1",
                        "model": "claude-sonnet-4",
                        "usage": {"input_tokens": 20, "output_tokens": 0},
                    },
                },
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '{"ci'},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": 'ty": "Oslo"}'},
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use"},
                    "usage": {"output_tokens": 8},
                },
            ),
            headers={"Content-Type": "text/event-stream"},
        )
    )
    provider = AnthropicProvider("sk-ant", base_url="https://api.anthropic.com")
    chunks = list(provider.stream(_request(stream=True, tools=[WEATHER_TOOL])))
    provider.close()

    opening = next(c for c in chunks if (c.delta.tool_calls or [{}])[0].get("id"))
    fragment = opening.delta.tool_calls[0]
    assert fragment["index"] == 0
    assert fragment["id"] == "toolu_1"
    assert fragment["function"]["name"] == "get_weather"
    assert json.loads(_joined_tool_arguments(chunks)[0]) == {"city": "Oslo"}
    assert chunks[-1].finish_reason == "tool_calls"


@respx.mock
def test_anthropic_keeps_two_streamed_tool_calls_apart() -> None:
    """Fragments are only meaningful relative to the block that opened them."""
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            content=_sse(
                {
                    "type": "message_start",
                    "message": {"id": "m", "model": "claude-sonnet-4", "usage": {}},
                },
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "tool_use", "id": "t0", "name": "get_weather"},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '{"city":"Oslo"}'},
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "tool_use", "id": "t1", "name": "get_weather"},
                },
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": '{"city":"Lima"}'},
                },
                {"type": "content_block_stop", "index": 1},
                {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {}},
            ),
            headers={"Content-Type": "text/event-stream"},
        )
    )
    provider = AnthropicProvider("sk-ant", base_url="https://api.anthropic.com")
    chunks = list(provider.stream(_request(stream=True, tools=[WEATHER_TOOL])))
    provider.close()
    merged = _joined_tool_arguments(chunks)
    assert json.loads(merged[0]) == {"city": "Oslo"}
    assert json.loads(merged[1]) == {"city": "Lima"}


@respx.mock
def test_google_streams_a_function_call() -> None:
    model = "gemini-3-pro"
    respx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"
    ).mock(
        return_value=httpx.Response(
            200,
            content=_sse(
                {
                    "responseId": "r1",
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "functionCall": {
                                            "name": "get_weather",
                                            "args": {"city": "Oslo"},
                                        }
                                    }
                                ]
                            },
                            "finishReason": "STOP",
                        }
                    ],
                }
            ),
            headers={"Content-Type": "text/event-stream"},
        )
    )
    provider = GoogleProvider("key")
    chunks = list(
        provider.stream(_request(model=f"google/{model}", stream=True, tools=[WEATHER_TOOL]))
    )
    provider.close()
    fragment = next(c.delta.tool_calls[0] for c in chunks if c.delta.tool_calls)
    assert fragment["function"]["name"] == "get_weather"
    assert json.loads(fragment["function"]["arguments"]) == {"city": "Oslo"}


@respx.mock
def test_bedrock_streams_tool_arguments() -> None:
    frames = b"".join(
        encode_event_frame({":event-type": name}, json.dumps(body).encode())
        for name, body in [
            (
                "contentBlockStart",
                {
                    "contentBlockIndex": 0,
                    "start": {"toolUse": {"toolUseId": "tu_1", "name": "get_weather"}},
                },
            ),
            (
                "contentBlockDelta",
                {"contentBlockIndex": 0, "delta": {"toolUse": {"input": '{"city":'}}},
            ),
            (
                "contentBlockDelta",
                {"contentBlockIndex": 0, "delta": {"toolUse": {"input": '"Oslo"}'}}},
            ),
            ("messageStop", {"stopReason": "tool_use"}),
            ("metadata", {"usage": {"inputTokens": 7, "outputTokens": 5}}),
        ]
    )
    respx.post(
        "https://bedrock-runtime.us-east-1.amazonaws.com"
        "/model/us.anthropic.claude-sonnet-4/converse-stream"
    ).mock(return_value=httpx.Response(200, content=frames))
    provider = BedrockProvider(
        "bedrock-key",
        region="us-east-1",
        models={"anthropic/claude-sonnet-4": "us.anthropic.claude-sonnet-4"},
    )
    chunks = list(provider.stream(_request(stream=True, tools=[WEATHER_TOOL])))
    provider.close()
    opening = next(c for c in chunks if (c.delta.tool_calls or [{}])[0].get("id"))
    assert opening.delta.tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(_joined_tool_arguments(chunks)[0]) == {"city": "Oslo"}
    assert next(c.usage for c in reversed(chunks) if c.usage).completion_tokens == 5


@respx.mock
def test_bedrock_streams_reasoning() -> None:
    frames = b"".join(
        encode_event_frame({":event-type": name}, json.dumps(body).encode())
        for name, body in [
            (
                "contentBlockDelta",
                {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "thinking"}}},
            ),
            (
                "contentBlockDelta",
                {"contentBlockIndex": 0, "delta": {"reasoningContent": {"signature": "sig"}}},
            ),
            ("contentBlockDelta", {"contentBlockIndex": 1, "delta": {"text": "answer"}}),
            ("messageStop", {"stopReason": "end_turn"}),
        ]
    )
    respx.post(
        "https://bedrock-runtime.us-east-1.amazonaws.com"
        "/model/us.anthropic.claude-sonnet-4/converse-stream"
    ).mock(return_value=httpx.Response(200, content=frames))
    provider = BedrockProvider(
        "bedrock-key",
        region="us-east-1",
        models={"anthropic/claude-sonnet-4": "us.anthropic.claude-sonnet-4"},
    )
    chunks = list(provider.stream(_request(stream=True)))
    provider.close()
    assert "".join(c.delta.reasoning or "" for c in chunks) == "thinking"
    assert any(c.delta.reasoning_signature == "sig" for c in chunks)
    assert "".join(c.delta.content or "" for c in chunks) == "answer"


# --------------------------------------------------------------------------
# Structured output
# --------------------------------------------------------------------------


@respx.mock
def test_anthropic_forces_a_tool_for_a_json_schema() -> None:
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-sonnet-4",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": STRUCTURED_TOOL_NAME,
                        "input": {"city": "Oslo"},
                    }
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 5, "output_tokens": 4},
            },
        )
    )
    provider = AnthropicProvider("sk-ant", base_url="https://api.anthropic.com")
    resp = provider.chat(
        _request(
            response_format=ResponseFormat(
                type="json_schema", json_schema={"name": "out", "schema": SCHEMA}
            )
        )
    )
    provider.close()
    sent = _sent(route)
    assert sent["tool_choice"] == {"type": "tool", "name": STRUCTURED_TOOL_NAME}
    assert sent["tools"][0]["input_schema"] == SCHEMA
    # The shim is invisible: JSON arrives as content, not as a tool call.
    assert json.loads(resp.text or "") == {"city": "Oslo"}
    assert resp.message.tool_calls is None
    assert resp.choices[0].finish_reason == "stop"


@respx.mock
def test_anthropic_streams_structured_json_as_content() -> None:
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            content=_sse(
                {
                    "type": "message_start",
                    "message": {"id": "m", "model": "claude-sonnet-4", "usage": {}},
                },
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "t0",
                        "name": STRUCTURED_TOOL_NAME,
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '{"city"'},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": ':"Oslo"}'},
                },
                {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {}},
            ),
            headers={"Content-Type": "text/event-stream"},
        )
    )
    provider = AnthropicProvider("sk-ant", base_url="https://api.anthropic.com")
    chunks = list(
        provider.stream(
            _request(
                stream=True,
                response_format=ResponseFormat(
                    type="json_schema", json_schema={"name": "out", "schema": SCHEMA}
                ),
            )
        )
    )
    provider.close()
    assert json.loads("".join(c.delta.content or "" for c in chunks)) == {"city": "Oslo"}
    assert all(not c.delta.tool_calls for c in chunks)
    assert chunks[-1].finish_reason == "stop"


@respx.mock
def test_real_tools_win_over_the_structured_output_shim() -> None:
    """Forcing our tool would make the caller's own tools unreachable."""
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "m",
                "model": "claude-sonnet-4",
                "content": [{"type": "text", "text": "{}"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )
    )
    provider = AnthropicProvider("sk-ant", base_url="https://api.anthropic.com")
    provider.chat(
        _request(
            tools=[WEATHER_TOOL],
            response_format=ResponseFormat(
                type="json_schema", json_schema={"name": "out", "schema": SCHEMA}
            ),
        )
    )
    provider.close()
    names = [t["name"] for t in _sent(route)["tools"]]
    assert names == ["get_weather"]


@respx.mock
def test_anthropic_asks_for_json_when_no_schema_is_given() -> None:
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "m",
                "model": "claude-sonnet-4",
                "content": [{"type": "text", "text": "{}"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )
    )
    provider = AnthropicProvider("sk-ant", base_url="https://api.anthropic.com")
    provider.chat(_request(response_format=ResponseFormat(type="json_object")))
    provider.close()
    sent = _sent(route)
    assert "JSON" in sent["system"]
    assert "tools" not in sent


@respx.mock
def test_google_sends_a_native_response_schema() -> None:
    model = "gemini-3-pro"
    route = respx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "responseId": "r1",
                "candidates": [
                    {"content": {"parts": [{"text": '{"city":"Oslo"}'}]}, "finishReason": "STOP"}
                ],
                "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 4},
            },
        )
    )
    provider = GoogleProvider("key")
    resp = provider.chat(
        _request(
            model=f"google/{model}",
            response_format=ResponseFormat(
                type="json_schema", json_schema={"name": "out", "schema": SCHEMA}
            ),
        )
    )
    provider.close()
    generation = _sent(route)["generationConfig"]
    assert generation["responseMimeType"] == "application/json"
    # Gemini 400s on JSON Schema bookkeeping keys.
    assert "additionalProperties" not in generation["responseSchema"]
    assert generation["responseSchema"]["required"] == ["city"]
    assert json.loads(resp.text or "") == {"city": "Oslo"}


@respx.mock
def test_bedrock_forces_a_tool_for_a_json_schema() -> None:
    route = respx.post(
        "https://bedrock-runtime.us-east-1.amazonaws.com"
        "/model/us.anthropic.claude-sonnet-4/converse"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "output": {
                    "message": {
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "tu_1",
                                    "name": STRUCTURED_TOOL_NAME,
                                    "input": {"city": "Oslo"},
                                }
                            }
                        ]
                    }
                },
                "stopReason": "tool_use",
                "usage": {"inputTokens": 3, "outputTokens": 4},
            },
        )
    )
    provider = BedrockProvider(
        "bedrock-key",
        region="us-east-1",
        models={"anthropic/claude-sonnet-4": "us.anthropic.claude-sonnet-4"},
    )
    resp = provider.chat(
        _request(
            response_format=ResponseFormat(
                type="json_schema", json_schema={"name": "out", "schema": SCHEMA}
            )
        )
    )
    provider.close()
    config = _sent(route)["toolConfig"]
    assert config["toolChoice"] == {"tool": {"name": STRUCTURED_TOOL_NAME}}
    assert json.loads(resp.text or "") == {"city": "Oslo"}
    assert resp.message.tool_calls is None


def test_gemini_schema_prunes_nested_bookkeeping_keys() -> None:
    pruned = gemini_schema(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "object", "additionalProperties": False},
                }
            },
        }
    )
    assert "$schema" not in pruned
    assert "additionalProperties" not in pruned
    assert "additionalProperties" not in pruned["properties"]["items"]["items"]
    assert pruned["properties"]["items"]["type"] == "array"


@respx.mock
def test_openai_still_uses_its_native_response_format() -> None:
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "c1",
                "model": "gpt-5",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"city":"Oslo"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            },
        )
    )
    provider = OpenAIProvider("sk-test", transport="chat")
    provider.chat(
        _request(
            model="openai/gpt-5",
            response_format=ResponseFormat(
                type="json_schema", json_schema={"name": "out", "schema": SCHEMA}
            ),
        )
    )
    provider.close()
    sent = _sent(route)
    assert sent["response_format"]["type"] == "json_schema"
    assert "tools" not in sent

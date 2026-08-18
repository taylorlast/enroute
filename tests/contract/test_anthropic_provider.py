import httpx
import respx

from enroute.providers import AnthropicProvider
from enroute.types import ChatRequest, Message


def _provider() -> AnthropicProvider:
    return AnthropicProvider("sk-ant", base_url="https://api.anthropic.com")


@respx.mock
def test_anthropic_chat() -> None:
    provider = _provider()
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-sonnet-4",
                "content": [{"type": "text", "text": "Hi there"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 2},
            },
        )
    )
    resp = provider.chat(
        ChatRequest(
            model="anthropic/claude-sonnet-4",
            messages=[
                Message(role="system", content="Be brief"),
                Message(role="user", content="Hello"),
            ],
        )
    )
    assert resp.text == "Hi there"
    assert resp.usage.prompt_tokens == 3


@respx.mock
def test_anthropic_stream_is_openai_shaped_and_bills_input_tokens() -> None:
    sse = (
        'data: {"type":"message_start","message":{"id":"msg_1","model":"claude-sonnet-4",'
        '"usage":{"input_tokens":11,"output_tokens":0}}}\n\n'
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hel"}}\n\n'
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"lo"}}\n\n'
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
        '"usage":{"output_tokens":2}}\n\n'
    )
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200, content=sse.encode(), headers={"Content-Type": "text/event-stream"}
        )
    )
    provider = _provider()
    chunks = list(
        provider.stream(
            ChatRequest(
                model="anthropic/claude-sonnet-4",
                messages=[Message(role="user", content="Hi")],
                stream=True,
            )
        )
    )
    assert "".join(c.delta.content or "" for c in chunks) == "Hello"
    usage = next(c.usage for c in reversed(chunks) if c.usage)
    assert usage.prompt_tokens == 11
    assert usage.completion_tokens == 2
    payload = chunks[0].to_openai()
    assert payload["object"] == "chat.completion.chunk"
    assert payload["choices"][0]["delta"]["content"] == "Hel"
    assert payload["choices"][0]["delta"]["content"] == chunks[0].delta.content
    assert "type" not in payload
    provider.close()

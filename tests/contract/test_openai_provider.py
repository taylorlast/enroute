import json

import httpx
import pytest
import respx

from enroute.providers import OpenAIProvider
from enroute.types import ChatRequest, Message


@pytest.fixture
def provider() -> OpenAIProvider:
    return OpenAIProvider("sk-test", base_url="https://api.openai.com/v1")


@respx.mock
def test_openai_chat(provider: OpenAIProvider) -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )
    )
    resp = provider.chat(
        ChatRequest(model="openai/gpt-4o-mini", messages=[Message(role="user", content="Hi")])
    )
    assert resp.text == "Hello"
    assert resp.usage.total_tokens == 7
    assert resp.provider == "openai"
    request = respx.calls.last.request
    body = json.loads(request.content)
    assert body["model"] == "gpt-4o-mini"


@respx.mock
def test_openai_rate_limit(provider: OpenAIProvider) -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(429, json={"error": {"message": "rate limited"}})
    )
    from enroute.errors import RateLimitError

    with pytest.raises(RateLimitError):
        provider.chat(
            ChatRequest(model="openai/gpt-4o-mini", messages=[Message(role="user", content="Hi")])
        )


@respx.mock
def test_openai_stream(provider: OpenAIProvider) -> None:
    sse = (
        'data: {"id":"1","model":"gpt-4o-mini","choices":[{"delta":{"content":"He"}}]}\n\n'
        'data: {"id":"1","model":"gpt-4o-mini","choices":[{"delta":{"content":"llo"},'
        '"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":1,"completion_tokens":2}}\n\n'
        "data: [DONE]\n\n"
    )
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=sse.encode(), headers={"Content-Type": "text/event-stream"}
        )
    )
    chunks = list(
        provider.stream(
            ChatRequest(
                model="openai/gpt-4o-mini",
                messages=[Message(role="user", content="Hi")],
                stream=True,
            )
        )
    )
    assert "".join(c.delta.content or "" for c in chunks) == "Hello"

import httpx
import respx

from enroute.providers import GoogleProvider
from enroute.types import ChatRequest, Message


@respx.mock
def test_google_chat() -> None:
    provider = GoogleProvider("key", base_url="https://generativelanguage.googleapis.com/v1beta")
    respx.post(url__regex=r".*/models/gemini-2.5-flash:generateContent.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": "Hey"}], "role": "model"},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1},
            },
        )
    )
    resp = provider.chat(
        ChatRequest(
            model="google/gemini-2.5-flash",
            messages=[Message(role="user", content="Hi")],
        )
    )
    assert resp.text == "Hey"


@respx.mock
def test_google_stream_is_openai_shaped() -> None:
    sse = (
        'data: {"responseId":"r1","candidates":[{"content":{"parts":[{"text":"He"}]}}]}\n\n'
        'data: {"responseId":"r1","candidates":[{"content":{"parts":[{"text":"y"}],'
        '"role":"model"},"finishReason":"STOP"}],'
        '"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":1}}\n\n'
    )
    respx.post(url__regex=r".*/models/gemini-2.5-flash:streamGenerateContent.*").mock(
        return_value=httpx.Response(
            200, content=sse.encode(), headers={"Content-Type": "text/event-stream"}
        )
    )
    provider = GoogleProvider("key", base_url="https://generativelanguage.googleapis.com/v1beta")
    chunks = list(
        provider.stream(
            ChatRequest(
                model="google/gemini-2.5-flash",
                messages=[Message(role="user", content="Hi")],
                stream=True,
            )
        )
    )
    assert "".join(c.delta.content or "" for c in chunks) == "Hey"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.prompt_tokens == 2
    payload = chunks[0].to_openai()
    assert payload["object"] == "chat.completion.chunk"
    assert payload["choices"][0]["delta"]["content"] == "He"
    assert "candidates" not in payload
    provider.close()

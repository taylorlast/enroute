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

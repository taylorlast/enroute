import httpx
import respx

from enroute.providers import AnthropicProvider
from enroute.types import ChatRequest, Message


@respx.mock
def test_anthropic_chat() -> None:
    provider = AnthropicProvider("sk-ant", base_url="https://api.anthropic.com")
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

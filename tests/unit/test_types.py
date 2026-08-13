from enroute.types import ChatRequest, ChatResponse, Choice, Message, Usage, text_content


def test_usage_from_counts() -> None:
    usage = Usage.from_counts(10, 5, cost=0.01)
    assert usage.total_tokens == 15
    assert usage.cost == 0.01


def test_chat_response_text() -> None:
    resp = ChatResponse(
        id="1",
        model="openai/gpt-4o-mini",
        choices=[Choice(message=Message(role="assistant", content="Hello"))],
    )
    assert resp.text == "Hello"
    assert resp.message.role == "assistant"


def test_text_content_multipart() -> None:
    msg = Message(
        role="user", content=[{"type": "text", "text": "Hi"}, {"type": "text", "text": "!"}]
    )
    assert text_content(msg) == "Hi!"


def test_chat_request_roundtrip() -> None:
    req = ChatRequest(model="openai/gpt-4o-mini", messages=[Message(role="user", content="x")])
    assert req.model_dump()["model"] == "openai/gpt-4o-mini"

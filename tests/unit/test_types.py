from enroute.types import (
    ChatRequest,
    ChatResponse,
    Choice,
    FinishReason,
    Message,
    StreamChunk,
    StreamDelta,
    Usage,
    text_content,
)


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


def test_stream_chunk_to_openai_hides_the_native_event() -> None:
    chunk = StreamChunk(
        id="s1",
        model="anthropic/claude-sonnet-4",
        delta=StreamDelta(content="Hi"),
        finish_reason=FinishReason.STOP,
        usage=Usage.from_counts(3, 1),
        provider="anthropic",
        region="us",
        raw={"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}},
    )
    payload = chunk.to_openai()
    assert payload["object"] == "chat.completion.chunk"
    assert payload["choices"][0]["delta"]["content"] == "Hi"
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "total_tokens": 4,
    }
    assert payload["provider"] == "anthropic"
    assert payload["region"] == "us"
    assert "type" not in payload
    assert "raw" not in payload

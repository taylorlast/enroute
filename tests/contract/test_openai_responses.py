"""Contract tests for the OpenAI Responses endpoint.

OpenAI rejects function tools on ``/chat/completions`` for every current model,
so tool calling and reasoning ride the Responses API instead. These tests pin
the translation in both directions: what enroute sends, and that what comes back
is indistinguishable from any other provider's stream.
"""

import json

import httpx
import pytest
import respx

from enroute.errors import ProviderUnavailable
from enroute.providers import OpenAIProvider, OpenAIResponsesProvider
from enroute.types import (
    ChatRequest,
    FinishReason,
    FunctionCall,
    FunctionDefinition,
    Message,
    ResponseFormat,
    Tool,
)

CHAT_URL = "https://api.openai.com/v1/chat/completions"
RESPONSES_URL = "https://api.openai.com/v1/responses"

WEATHER = Tool(
    function=FunctionDefinition(
        name="get_weather",
        description="Look up the weather.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
)


@pytest.fixture
def provider() -> OpenAIProvider:
    return OpenAIProvider("sk-test", base_url="https://api.openai.com/v1")


@pytest.fixture
def responses() -> OpenAIResponsesProvider:
    return OpenAIResponsesProvider("sk-test", base_url="https://api.openai.com/v1")


def ask(**kwargs: object) -> ChatRequest:
    kwargs.setdefault("model", "openai/gpt-5.6")
    kwargs.setdefault("messages", [Message(role="user", content="Weather in Oslo?")])
    return ChatRequest(**kwargs)  # type: ignore[arg-type]


def sse(*events: dict) -> httpx.Response:
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    return httpx.Response(
        200, text=body + "data: [DONE]\n\n", headers={"content-type": "text/event-stream"}
    )


def completed(**overrides: object) -> dict:
    response = {
        "id": "resp_1",
        "model": "gpt-5.6",
        "status": "completed",
        "output": [],
        "usage": {"input_tokens": 9, "output_tokens": 5},
    }
    response.update(overrides)
    return {"type": "response.completed", "response": response}


def test_requests_prefer_responses(provider: OpenAIProvider) -> None:
    delegate = provider._delegate(ask(tools=[WEATHER]))
    assert isinstance(delegate, OpenAIResponsesProvider)
    # The slug is what shows up in traces and billing, so it must not change.
    assert delegate.name == "openai"
    assert provider._delegate(ask()) is not None


def test_a_request_using_stop_or_seed_stays_on_chat_completions(
    provider: OpenAIProvider,
) -> None:
    # Responses rejects both as unknown parameters, and quietly dropping a stop
    # sequence can let a completion run past where the caller wanted it cut.
    assert provider._delegate(ask(stop=["\n"])) is None
    assert provider._delegate(ask(seed=7)) is None
    # Unless tools are also in play, which chat completions refuses outright.
    assert provider._delegate(ask(stop=["\n"], tools=[WEATHER])) is not None


def test_pinning_the_transport_overrides_the_choice() -> None:
    pinned = OpenAIProvider("sk-test", transport="responses")
    assert pinned._delegate(ask(stop=["\n"])) is not None
    chat_only = OpenAIProvider("sk-test", transport="chat")
    assert chat_only._delegate(ask(tools=[WEATHER])) is None


@respx.mock
def test_tools_are_sent_flattened_and_nothing_is_retained(provider: OpenAIProvider) -> None:
    route = respx.post(RESPONSES_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "resp_1",
                "model": "gpt-5.6",
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )
    )
    provider.chat(ask(tools=[WEATHER], tool_choice="required"))

    body = json.loads(route.calls.last.request.content)
    # Responses drops the ``function`` wrapper Chat Completions requires.
    assert body["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "parameters": WEATHER.function.parameters,
            "description": "Look up the weather.",
        }
    ]
    assert body["tool_choice"] == "required"
    # Responses retains by default; Chat Completions does not, so neither do we.
    assert body["store"] is False
    assert body["reasoning"] == {"summary": "auto"}


@respx.mock
def test_a_forced_tool_is_flattened_too(responses: OpenAIResponsesProvider) -> None:
    route = respx.post(RESPONSES_URL).mock(
        return_value=httpx.Response(
            200, json={"id": "resp_1", "model": "gpt-5.6", "status": "completed", "output": []}
        )
    )
    responses.chat(
        ask(tools=[WEATHER], tool_choice={"type": "function", "function": {"name": "get_weather"}})
    )
    body = json.loads(route.calls.last.request.content)
    assert body["tool_choice"] == {"type": "function", "name": "get_weather"}


@respx.mock
def test_a_tool_result_becomes_a_sibling_input_item(responses: OpenAIResponsesProvider) -> None:
    route = respx.post(RESPONSES_URL).mock(
        return_value=httpx.Response(
            200, json={"id": "resp_1", "model": "gpt-5.6", "status": "completed", "output": []}
        )
    )
    responses.chat(
        ask(
            messages=[
                Message(role="user", content="Weather in Oslo?"),
                Message(
                    role="assistant",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": FunctionCall(
                                name="get_weather", arguments='{"city":"Oslo"}'
                            ),
                        }
                    ],
                ),
                Message(role="tool", tool_call_id="call_1", content="12C"),
            ],
            tools=[WEATHER],
        )
    )

    body = json.loads(route.calls.last.request.content)
    assert body["input"] == [
        {"role": "user", "content": "Weather in Oslo?"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": '{"city":"Oslo"}',
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "12C"},
    ]


@respx.mock
def test_structured_output_becomes_a_text_format(responses: OpenAIResponsesProvider) -> None:
    route = respx.post(RESPONSES_URL).mock(
        return_value=httpx.Response(
            200, json={"id": "resp_1", "model": "gpt-5.6", "status": "completed", "output": []}
        )
    )
    schema = {"type": "object", "properties": {"city": {"type": "string"}}}
    responses.chat(
        ask(
            response_format=ResponseFormat(
                type="json_schema",
                json_schema={"name": "place", "schema": schema, "strict": True},
            )
        )
    )
    body = json.loads(route.calls.last.request.content)
    assert body["text"] == {
        "format": {"type": "json_schema", "name": "place", "schema": schema, "strict": True}
    }


@respx.mock
def test_stop_and_seed_are_dropped_rather_than_rejected(
    responses: OpenAIResponsesProvider,
) -> None:
    route = respx.post(RESPONSES_URL).mock(
        return_value=httpx.Response(
            200, json={"id": "resp_1", "model": "gpt-5.6", "status": "completed", "output": []}
        )
    )
    responses.chat(ask(stop=["\n"], seed=7, max_tokens=64))

    body = json.loads(route.calls.last.request.content)
    # The endpoint 400s on both as unknown parameters, and failing the whole
    # request over an unsupported nicety would push callers to a lesser host.
    assert "stop" not in body
    assert "seed" not in body
    assert body["max_output_tokens"] == 64


@respx.mock
def test_a_tiny_token_cap_is_raised_to_the_endpoints_floor(
    responses: OpenAIResponsesProvider,
) -> None:
    route = respx.post(RESPONSES_URL).mock(
        return_value=httpx.Response(
            200, json={"id": "resp_1", "model": "gpt-5.6", "status": "completed", "output": []}
        )
    )
    # The endpoint 400s below 16 where chat completions accepts any cap. The
    # dispatcher normally keeps such a request on chat completions; when tools
    # drag it here anyway, a few extra tokens beat a hard failure.
    responses.chat(ask(max_tokens=8, tools=[WEATHER]))
    assert json.loads(route.calls.last.request.content)["max_output_tokens"] == 16


@respx.mock
def test_a_reasoning_model_that_rejects_sampling_is_retried_without_it(
    responses: OpenAIResponsesProvider,
) -> None:
    ok = {"id": "resp_1", "model": "gpt-5.6", "status": "completed", "output": []}
    route = respx.post(RESPONSES_URL).mock(
        side_effect=[
            httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Unsupported parameter: 'temperature' is not supported "
                        "with this model.",
                        "type": "invalid_request_error",
                    }
                },
            ),
            httpx.Response(200, json=ok),
        ]
    )
    responses.chat(ask(temperature=0.5))

    assert json.loads(route.calls[0].request.content)["temperature"] == 0.5
    assert "temperature" not in json.loads(route.calls[1].request.content)


@respx.mock
def test_a_completed_response_is_parsed_into_content_reasoning_and_tools(
    responses: OpenAIResponsesProvider,
) -> None:
    respx.post(RESPONSES_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "resp_1",
                "model": "gpt-5.6",
                "status": "completed",
                "created_at": 1787066084,
                "output": [
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": "Checking Oslo."}],
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Calling the tool."}],
                    },
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "get_weather",
                        "arguments": '{"city":"Oslo"}',
                    },
                ],
                "usage": {"input_tokens": 9, "output_tokens": 5},
            },
        )
    )
    response = responses.chat(ask(tools=[WEATHER]))

    assert response.text == "Calling the tool."
    assert response.message.reasoning == "Checking Oslo."
    assert response.message.tool_calls is not None
    call = response.message.tool_calls[0]
    # The call_id is the handle the endpoint expects back, not the item id.
    assert call.id == "call_1"
    assert call.function.name == "get_weather"
    assert response.choices[0].finish_reason is FinishReason.TOOL_CALLS
    assert response.usage.prompt_tokens == 9
    assert response.usage.completion_tokens == 5


@respx.mock
def test_a_stream_is_normalized_to_openai_deltas(responses: OpenAIResponsesProvider) -> None:
    respx.post(RESPONSES_URL).mock(
        return_value=sse(
            {"type": "response.created", "response": {"id": "resp_1", "model": "gpt-5.6"}},
            {"type": "response.reasoning_summary_text.delta", "delta": "Thinking"},
            {"type": "response.reasoning_summary_text.delta", "delta": " hard."},
            {"type": "response.output_text.delta", "delta": "It is "},
            {"type": "response.output_text.delta", "delta": "12C."},
            completed(output=[{"type": "message"}]),
        )
    )
    chunks = list(responses.stream(ask()))

    assert chunks[0].delta.role == "assistant"
    assert chunks[0].id == "resp_1"
    assert "".join(c.delta.reasoning or "" for c in chunks) == "Thinking hard."
    assert "".join(c.delta.content or "" for c in chunks) == "It is 12C."
    assert chunks[-1].finish_reason is FinishReason.STOP
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.total_tokens == 14

    # Reasoning and content must stay in separate fields all the way to the wire.
    reasoning_chunk = next(c for c in chunks if c.delta.reasoning)
    payload = reasoning_chunk.to_openai()
    assert payload["choices"][0]["delta"] == {"reasoning": "Thinking"}
    assert payload["object"] == "chat.completion.chunk"


@respx.mock
def test_streamed_tool_calls_are_indexed_and_reassembled(
    responses: OpenAIResponsesProvider,
) -> None:
    respx.post(RESPONSES_URL).mock(
        return_value=sse(
            {"type": "response.created", "response": {"id": "resp_1", "model": "gpt-5.6"}},
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "get_weather",
                },
            },
            {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": '{"ci'},
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "id": "fc_2",
                    "call_id": "call_2",
                    "name": "get_weather",
                },
            },
            {"type": "response.function_call_arguments.delta", "item_id": "fc_2", "delta": '{"ci'},
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "delta": 'ty":"Oslo"}',
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_2",
                "delta": 'ty":"Bergen"}',
            },
            completed(output=[{"type": "function_call"}]),
        )
    )
    chunks = list(responses.stream(ask(tools=[WEATHER])))

    arguments: dict[int, str] = {}
    names: dict[int, str] = {}
    ids: dict[int, str] = {}
    for chunk in chunks:
        for fragment in chunk.delta.tool_calls or []:
            index = fragment["index"]
            arguments[index] = arguments.get(index, "") + (
                fragment["function"].get("arguments") or ""
            )
            if fragment["function"].get("name"):
                names[index] = fragment["function"]["name"]
            if fragment.get("id"):
                ids[index] = fragment["id"]

    # The endpoint interleaves two calls and addresses them by opaque item id;
    # a client reassembling by index must still get two whole argument objects.
    assert json.loads(arguments[0]) == {"city": "Oslo"}
    assert json.loads(arguments[1]) == {"city": "Bergen"}
    assert names == {0: "get_weather", 1: "get_weather"}
    assert ids == {0: "call_1", 1: "call_2"}
    assert chunks[-1].finish_reason is FinishReason.TOOL_CALLS


@respx.mock
def test_a_truncated_stream_finishes_on_length(responses: OpenAIResponsesProvider) -> None:
    respx.post(RESPONSES_URL).mock(
        return_value=sse(
            {"type": "response.created", "response": {"id": "resp_1", "model": "gpt-5.6"}},
            {
                "type": "response.incomplete",
                "response": {
                    "id": "resp_1",
                    "model": "gpt-5.6",
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "output": [],
                },
            },
        )
    )
    chunks = list(responses.stream(ask(max_tokens=16)))
    assert chunks[-1].finish_reason is FinishReason.LENGTH


@respx.mock
def test_a_failed_stream_raises_the_hosts_message(responses: OpenAIResponsesProvider) -> None:
    respx.post(RESPONSES_URL).mock(
        return_value=sse(
            {"type": "response.created", "response": {"id": "resp_1", "model": "gpt-5.6"}},
            {
                "type": "response.failed",
                "response": {"status": "failed", "error": {"message": "model exploded"}},
            },
        )
    )
    with pytest.raises(ProviderUnavailable, match="model exploded"):
        list(responses.stream(ask()))


@respx.mock
@pytest.mark.asyncio
async def test_async_streaming_matches_the_sync_path(
    responses: OpenAIResponsesProvider,
) -> None:
    respx.post(RESPONSES_URL).mock(
        return_value=sse(
            {"type": "response.created", "response": {"id": "resp_1", "model": "gpt-5.6"}},
            {"type": "response.output_text.delta", "delta": "pong"},
            completed(output=[{"type": "message"}]),
        )
    )
    chunks = [chunk async for chunk in responses.astream(ask())]
    assert "".join(c.delta.content or "" for c in chunks) == "pong"
    assert chunks[-1].usage is not None
    await responses.aclose()


@respx.mock
def test_the_dispatcher_reaches_both_endpoints(provider: OpenAIProvider) -> None:
    chat = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": "gpt-5.6",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "pong"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )
    )
    tools = respx.post(RESPONSES_URL).mock(
        return_value=httpx.Response(
            200, json={"id": "resp_1", "model": "gpt-5.6", "status": "completed", "output": []}
        )
    )
    assert provider.chat(ask(stop=["\n"])).text == "pong"
    provider.chat(ask(tools=[WEATHER]))
    provider.close()

    assert chat.call_count == 1
    assert tools.call_count == 1

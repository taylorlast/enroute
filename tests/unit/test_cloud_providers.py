"""Azure and Bedrock adapters, including the framing Bedrock streams in."""

from __future__ import annotations

import datetime as dt
import json

import httpx
import pytest

from enroute.errors import ConfigurationError
from enroute.providers.azure import AzureOpenAIProvider, normalize_azure_base_url
from enroute.providers.bedrock import (
    BedrockProvider,
    EventStreamDecoder,
    bedrock_endpoint,
    encode_event_frame,
    parse_event_headers,
    sigv4_headers,
)
from enroute.types import ChatRequest, FunctionDefinition, Message, Tool

PROMPT = [Message(role="user", content="hi")]


def request(**kwargs) -> ChatRequest:
    kwargs.setdefault("messages", PROMPT)
    return ChatRequest(model="openai/gpt-5.6-sol", **kwargs)


def mount(provider, handler) -> None:
    """Point both clients at a mock transport."""
    transport = httpx.MockTransport(handler)
    provider._client = httpx.Client(
        base_url=provider.config.base_url, headers=provider._client.headers, transport=transport
    )
    provider._aclient = httpx.AsyncClient(
        base_url=provider.config.base_url, headers=provider._aclient.headers, transport=transport
    )


# --- Azure ---------------------------------------------------------------


def test_azure_requires_an_endpoint() -> None:
    with pytest.raises(ConfigurationError):
        AzureOpenAIProvider("key")


def test_azure_normalizes_endpoint_forms() -> None:
    assert normalize_azure_base_url("https://a.openai.azure.com") == (
        "https://a.openai.azure.com/openai/v1"
    )
    assert normalize_azure_base_url("https://a.openai.azure.com/openai/v1/") == (
        "https://a.openai.azure.com/openai/v1"
    )


def test_azure_uses_api_key_header_not_bearer() -> None:
    provider = AzureOpenAIProvider("secret", endpoint="https://a.openai.azure.com")
    # Azure reserves Authorization for Entra ID tokens.
    assert provider._client.headers["api-key"] == "secret"
    assert "authorization" not in provider._client.headers
    provider.close()

    entra = AzureOpenAIProvider(
        "token", endpoint="https://a.openai.azure.com", use_bearer_auth=True
    )
    assert entra._client.headers["authorization"] == "Bearer token"
    entra.close()


def test_azure_sends_the_deployment_name_as_model() -> None:
    seen: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        seen["path"] = req.url.path
        return httpx.Response(
            200,
            json={
                "id": "c1",
                "model": "sol-prod",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4},
            },
        )

    provider = AzureOpenAIProvider(
        "key",
        endpoint="https://a.openai.azure.com",
        deployments={"openai/gpt-5.6-sol": "sol-prod"},
    )
    mount(provider, handler)
    response = provider.chat(request())

    assert seen["model"] == "sol-prod"
    assert seen["path"] == "/openai/v1/chat/completions"
    assert response.text == "ok"
    assert response.usage.prompt_tokens == 3
    provider.close()


def test_azure_falls_back_to_the_bare_slug() -> None:
    provider = AzureOpenAIProvider("key", endpoint="https://a.openai.azure.com")
    assert provider._model_id("openai/gpt-5.6-sol") == "gpt-5.6-sol"
    provider.close()


def test_azure_streams_sse() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        body = (
            'data: {"id":"c1","model":"d","choices":[{"delta":{"content":"He"}}]}\n\n'
            'data: {"id":"c1","model":"d","choices":[{"delta":{"content":"llo"}}]}\n\n'
            'data: {"id":"c1","model":"d","choices":[{"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":2,"completion_tokens":5}}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=body)

    provider = AzureOpenAIProvider("key", endpoint="https://a.openai.azure.com")
    mount(provider, handler)
    chunks = list(provider.stream(request()))

    assert "".join(c.delta.content or "" for c in chunks) == "Hello"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.completion_tokens == 5
    provider.close()


# --- Bedrock framing -----------------------------------------------------


def test_bedrock_endpoint_is_regional() -> None:
    assert bedrock_endpoint("eu-west-1") == "https://bedrock-runtime.eu-west-1.amazonaws.com"


def test_decoder_reassembles_frames_split_across_reads() -> None:
    frames = encode_event_frame(
        {":event-type": "contentBlockDelta"}, json.dumps({"delta": {"text": "a"}}).encode()
    ) + encode_event_frame(
        {":event-type": "messageStop"}, json.dumps({"stopReason": "end_turn"}).encode()
    )

    # One byte at a time is the worst case a network read can produce.
    decoder = EventStreamDecoder()
    events = [event for i in range(len(frames)) for event in decoder.feed(frames[i : i + 1])]
    assert [headers[":event-type"] for headers, _ in events] == [
        "contentBlockDelta",
        "messageStop",
    ]

    # Both frames arriving together must yield both, not just the first.
    assert len(list(EventStreamDecoder().feed(frames))) == 2


def test_decoder_skips_non_string_headers_without_losing_alignment() -> None:
    # A bool header carries no value bytes; miscounting it corrupts the rest.
    raw = bytes([5]) + b"flagx"[:5] + bytes([0])
    raw += bytes([11]) + b":event-type" + bytes([7]) + b"\x00\x02" + b"ab"
    assert parse_event_headers(raw) == {":event-type": "ab"}


def test_decoder_rejects_an_implausible_frame_length() -> None:
    from enroute.errors import EnrouteError

    bogus = (99_999_999).to_bytes(4, "big") + (0).to_bytes(4, "big") + b"\x00" * 4
    with pytest.raises(EnrouteError):
        list(EventStreamDecoder().feed(bogus))


# --- Bedrock requests ----------------------------------------------------


def test_bedrock_needs_credentials() -> None:
    with pytest.raises(ConfigurationError):
        BedrockProvider()


def test_bedrock_reports_the_coarse_region_for_billing() -> None:
    provider = BedrockProvider("key", region="eu-central-1")
    assert (provider.aws_region, provider.region) == ("eu-central-1", "eu")
    provider.close()


def test_bedrock_encodes_converse_shape() -> None:
    seen: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        seen["path"] = req.url.path
        seen["auth"] = req.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 7, "outputTokens": 2},
            },
        )

    provider = BedrockProvider("key", models={"openai/gpt-5.6-sol": "openai.gpt-5.6-sol"})
    mount(provider, handler)
    response = provider.chat(
        request(
            messages=[
                Message(role="system", content="be brief"),
                Message(role="user", content="hi"),
            ],
            max_tokens=64,
            temperature=0.2,
            tools=[Tool(function=FunctionDefinition(name="lookup", parameters={"type": "object"}))],
        )
    )

    assert seen["path"] == "/model/openai.gpt-5.6-sol/converse"
    assert seen["auth"] == "Bearer key"
    # System prompts are a separate field, not a message.
    assert seen["system"] == [{"text": "be brief"}]
    assert seen["messages"] == [{"role": "user", "content": [{"text": "hi"}]}]
    assert seen["inferenceConfig"] == {"maxTokens": 64, "temperature": 0.2}
    assert seen["toolConfig"]["tools"][0]["toolSpec"]["name"] == "lookup"
    assert response.text == "ok"
    assert response.usage.prompt_tokens == 7
    assert response.region == "us"
    provider.close()


def test_bedrock_parses_tool_use() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "t1",
                                    "name": "lookup",
                                    "input": {"q": "x"},
                                }
                            }
                        ],
                    }
                },
                "stopReason": "tool_use",
                "usage": {"inputTokens": 1, "outputTokens": 1},
            },
        )

    provider = BedrockProvider("key")
    mount(provider, handler)
    message = provider.chat(request()).choices[0].message

    assert message.tool_calls is not None
    assert message.tool_calls[0].function.name == "lookup"
    assert json.loads(message.tool_calls[0].function.arguments) == {"q": "x"}
    provider.close()


def test_bedrock_streams_and_reports_usage() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        payload = b"".join(
            [
                encode_event_frame({":event-type": "messageStart"}, b'{"role":"assistant"}'),
                encode_event_frame(
                    {":event-type": "contentBlockDelta"}, b'{"delta":{"text":"Hel"}}'
                ),
                encode_event_frame(
                    {":event-type": "contentBlockDelta"}, b'{"delta":{"text":"lo"}}'
                ),
                encode_event_frame({":event-type": "messageStop"}, b'{"stopReason":"end_turn"}'),
                encode_event_frame(
                    {":event-type": "metadata"},
                    b'{"usage":{"inputTokens":11,"outputTokens":3}}',
                ),
            ]
        )
        return httpx.Response(200, content=payload)

    provider = BedrockProvider("key")
    mount(provider, handler)
    chunks = list(provider.stream(request()))

    assert "".join(c.delta.content or "" for c in chunks) == "Hello"
    assert any(c.finish_reason == "stop" for c in chunks)
    # Usage arrives only in the metadata event, and billing depends on it.
    usage = [c.usage for c in chunks if c.usage]
    assert usage and usage[-1].prompt_tokens == 11
    assert all(c.region == "us" for c in chunks)
    provider.close()


@pytest.mark.asyncio
async def test_bedrock_astream_matches_sync() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=encode_event_frame(
                {":event-type": "contentBlockDelta"}, b'{"delta":{"text":"hey"}}'
            ),
        )

    provider = BedrockProvider("key")
    mount(provider, handler)
    chunks = [chunk async for chunk in provider.astream(request())]
    assert "".join(c.delta.content or "" for c in chunks) == "hey"
    await provider.aclose()


# --- SigV4 ---------------------------------------------------------------


def test_sigv4_signs_deterministically() -> None:
    kwargs = dict(
        method="POST",
        url="https://bedrock-runtime.us-east-1.amazonaws.com/model/m/converse",
        region="us-east-1",
        payload=b'{"a":1}',
        access_key_id="AKIDEXAMPLE",
        secret_access_key="secret",
        now=dt.datetime(2026, 1, 2, 3, 4, 5, tzinfo=dt.timezone.utc),
    )
    first = sigv4_headers(**kwargs)
    assert first == sigv4_headers(**kwargs)
    assert first["x-amz-date"] == "20260102T030405Z"
    assert (
        "Credential=AKIDEXAMPLE/20260102/us-east-1/bedrock/aws4_request" in (first["Authorization"])
    )
    # A different body must produce a different signature.
    changed = sigv4_headers(**{**kwargs, "payload": b'{"a":2}'})
    assert changed["Authorization"] != first["Authorization"]


def test_sigv4_includes_session_token_when_present() -> None:
    headers = sigv4_headers(
        method="POST",
        url="https://bedrock-runtime.us-east-1.amazonaws.com/model/m/converse",
        region="us-east-1",
        payload=b"{}",
        access_key_id="AKID",
        secret_access_key="secret",
        session_token="tok",
    )
    assert headers["x-amz-security-token"] == "tok"
    assert "x-amz-security-token" in headers["Authorization"]


def test_bedrock_signs_when_given_iam_credentials() -> None:
    seen: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers.get("authorization", "")
        return httpx.Response(
            200,
            json={
                "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 1, "outputTokens": 1},
            },
        )

    provider = BedrockProvider(access_key_id="AKID", secret_access_key="secret")
    mount(provider, handler)
    provider.chat(request())
    assert str(seen["auth"]).startswith("AWS4-HMAC-SHA256 ")
    provider.close()


def test_explicit_key_wins_over_ambient_iam_credentials() -> None:
    provider = BedrockProvider("bearer-key", access_key_id="AKID", secret_access_key="secret")
    assert provider._auth_headers("/model/m/converse", b"{}") == {
        "Authorization": "Bearer bearer-key"
    }
    provider.close()

"""OpenAI-compatible provider adapters.

Covers OpenAI and any vendor that speaks the OpenAI Chat Completions API,
including Groq, Together, Fireworks, xAI, DeepSeek, Mistral, vLLM, and the
enroute gateway.

Examples:
    >>> from enroute.providers.openai_compatible import OpenAIProvider
    >>> OpenAIProvider.default_base_url
    'https://api.openai.com/v1'
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

from enroute.errors import EnrouteError
from enroute.providers.base import (
    ProviderConfig,
    aiter_sse_lines,
    iter_sse_lines,
    map_transport_error,
    raise_for_status,
)
from enroute.types import (
    ChatRequest,
    ChatResponse,
    Choice,
    FinishReason,
    FunctionCall,
    Message,
    StreamChunk,
    StreamDelta,
    ToolCall,
    Usage,
)


def _strip_author_prefix(model: str) -> str:
    """Strip ``author/`` prefix for providers that expect bare model ids.

    Host paths such as ``accounts/fireworks/models/...`` are left intact.

    Args:
        model: Model id, optionally prefixed with ``author/``.

    Returns:
        Bare model id without the author prefix.
    """
    if model.count("/") == 1:
        return model.split("/", 1)[1]
    return model


class OpenAICompatible:
    """Base adapter for OpenAI Chat Completions-compatible APIs.

    Args:
        api_key: Provider API key.
        base_url: API base URL.
        name: Provider slug used in traces and errors.
        timeout_s: Request timeout in seconds.
        default_headers: Extra headers.
        organization: Optional OpenAI organization header.
        strip_model_prefix: Whether to strip ``author/`` from model ids.
    """

    name: str = "openai"
    default_base_url: str = "https://api.openai.com/v1"
    strip_model_prefix: bool = True

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        name: str | None = None,
        timeout_s: float | None = None,
        default_headers: dict[str, str] | None = None,
        organization: str | None = None,
        strip_model_prefix: bool | None = None,
    ) -> None:
        self.name = name or self.name
        if strip_model_prefix is not None:
            self.strip_model_prefix = strip_model_prefix
        self.config = ProviderConfig(
            api_key=api_key,
            base_url=(base_url or self.default_base_url).rstrip("/"),
            timeout_s=timeout_s if timeout_s is not None else 60.0,
            default_headers=default_headers or {},
            organization=organization,
        )
        headers = self._build_headers()
        self._client = httpx.Client(
            base_url=self.config.base_url,
            headers=headers,
            timeout=self.config.timeout_s,
        )
        self._aclient = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers=headers,
            timeout=self.config.timeout_s,
        )

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            **self.config.default_headers,
        }
        if self.config.organization:
            headers["OpenAI-Organization"] = self.config.organization
        return headers

    def _model_id(self, model: str) -> str:
        return _strip_author_prefix(model) if self.strip_model_prefix else model

    def _encode_request(self, request: ChatRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model_id(request.model),
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
            "stream": stream,
        }
        for key in ("temperature", "top_p", "max_tokens", "stop", "seed", "user", "tool_choice"):
            value = getattr(request, key)
            if value is not None:
                payload[key] = value
        if request.tools:
            payload["tools"] = [t.model_dump(exclude_none=True) for t in request.tools]
        if request.response_format is not None:
            payload["response_format"] = request.response_format.model_dump(exclude_none=True)
        if stream:
            payload["stream_options"] = {"include_usage": True}
        payload.update(request.extra)
        return payload

    def _parse_message(self, raw: dict[str, Any]) -> Message:
        tool_calls = None
        if raw.get("tool_calls"):
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    type="function",
                    function=FunctionCall(
                        name=tc["function"]["name"],
                        arguments=tc["function"].get("arguments") or "{}",
                    ),
                )
                for tc in raw["tool_calls"]
            ]
        return Message(
            role=raw.get("role", "assistant"),
            content=raw.get("content"),
            name=raw.get("name"),
            tool_calls=tool_calls,
            tool_call_id=raw.get("tool_call_id"),
        )

    def _parse_finish_reason(self, value: str | None) -> FinishReason | str | None:
        if value is None:
            return None
        try:
            return FinishReason(value)
        except ValueError:
            return value

    def _parse_response(self, data: dict[str, Any], *, latency_ms: float) -> ChatResponse:
        usage_raw = data.get("usage") or {}
        usage = Usage.from_counts(
            int(usage_raw.get("prompt_tokens") or 0),
            int(usage_raw.get("completion_tokens") or 0),
        )
        choices = [
            Choice(
                index=int(c.get("index") or 0),
                message=self._parse_message(c.get("message") or {}),
                finish_reason=self._parse_finish_reason(c.get("finish_reason")),
            )
            for c in data.get("choices") or []
        ]
        return ChatResponse(
            id=str(data.get("id") or ""),
            model=str(data.get("model") or ""),
            choices=choices,
            usage=usage,
            provider=self.name,
            created=data.get("created"),
            raw=data,
            latency_ms=latency_ms,
        )

    def _parse_stream_chunk(self, data: dict[str, Any]) -> StreamChunk:
        choice = (data.get("choices") or [{}])[0]
        delta_raw = choice.get("delta") or {}
        usage = None
        if data.get("usage"):
            usage = Usage.from_counts(
                int(data["usage"].get("prompt_tokens") or 0),
                int(data["usage"].get("completion_tokens") or 0),
            )
        return StreamChunk(
            id=str(data.get("id") or ""),
            model=str(data.get("model") or ""),
            delta=StreamDelta(
                role=delta_raw.get("role"),
                content=delta_raw.get("content"),
                tool_calls=delta_raw.get("tool_calls"),
            ),
            finish_reason=self._parse_finish_reason(choice.get("finish_reason")),
            usage=usage,
            provider=self.name,
            raw=data,
        )

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Execute a non-streaming chat completion.

        Args:
            request: Normalized chat request.

        Returns:
            Normalized chat response.
        """
        payload = self._encode_request(request, stream=False)
        started = time.perf_counter()
        try:
            response = self._client.post("/chat/completions", json=payload)
        except Exception as exc:  # noqa: BLE001
            raise map_transport_error(exc, provider=self.name, model=request.model) from exc
        raise_for_status(response, provider=self.name, model=request.model)
        latency_ms = (time.perf_counter() - started) * 1000
        return self._parse_response(response.json(), latency_ms=latency_ms)

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        """Execute a streaming chat completion.

        Args:
            request: Normalized chat request.

        Yields:
            Normalized stream chunks.
        """
        payload = self._encode_request(request, stream=True)
        try:
            with self._client.stream("POST", "/chat/completions", json=payload) as response:
                raise_for_status(response, provider=self.name, model=request.model)
                for data in iter_sse_lines(response):
                    if data == "[DONE]":
                        break
                    yield self._parse_stream_chunk(json.loads(data))
        except EnrouteError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise map_transport_error(exc, provider=self.name, model=request.model) from exc

    async def achat(self, request: ChatRequest) -> ChatResponse:
        """Async non-streaming chat completion.

        Args:
            request: Normalized chat request.

        Returns:
            Normalized chat response.
        """
        payload = self._encode_request(request, stream=False)
        started = time.perf_counter()
        try:
            response = await self._aclient.post("/chat/completions", json=payload)
        except Exception as exc:  # noqa: BLE001
            raise map_transport_error(exc, provider=self.name, model=request.model) from exc
        raise_for_status(response, provider=self.name, model=request.model)
        latency_ms = (time.perf_counter() - started) * 1000
        return self._parse_response(response.json(), latency_ms=latency_ms)

    async def astream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Async streaming chat completion.

        Args:
            request: Normalized chat request.

        Yields:
            Normalized stream chunks.
        """
        payload = self._encode_request(request, stream=True)
        try:
            async with self._aclient.stream("POST", "/chat/completions", json=payload) as response:
                raise_for_status(response, provider=self.name, model=request.model)
                async for data in aiter_sse_lines(response):
                    if data == "[DONE]":
                        break
                    yield self._parse_stream_chunk(json.loads(data))
        except EnrouteError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise map_transport_error(exc, provider=self.name, model=request.model) from exc

    def close(self) -> None:
        """Close the sync HTTP client."""
        self._client.close()

    async def aclose(self) -> None:
        """Close the async HTTP client."""
        await self._aclient.aclose()


class OpenAIProvider(OpenAICompatible):
    """Official OpenAI Chat Completions API."""

    name = "openai"
    default_base_url = "https://api.openai.com/v1"


class GroqProvider(OpenAICompatible):
    """Groq OpenAI-compatible API."""

    name = "groq"
    default_base_url = "https://api.groq.com/openai/v1"


class TogetherProvider(OpenAICompatible):
    """Together AI OpenAI-compatible API."""

    name = "together"
    default_base_url = "https://api.together.xyz/v1"


class FireworksProvider(OpenAICompatible):
    """Fireworks AI OpenAI-compatible API."""

    name = "fireworks"
    default_base_url = "https://api.fireworks.ai/inference/v1"


class XAIProvider(OpenAICompatible):
    """xAI OpenAI-compatible API."""

    name = "xai"
    default_base_url = "https://api.x.ai/v1"


class DeepSeekProvider(OpenAICompatible):
    """DeepSeek OpenAI-compatible API."""

    name = "deepseek"
    default_base_url = "https://api.deepseek.com/v1"


class MistralProvider(OpenAICompatible):
    """Mistral OpenAI-compatible API."""

    name = "mistral"
    default_base_url = "https://api.mistral.ai/v1"


class BasetenProvider(OpenAICompatible):
    """Baseten Model APIs (OpenAI-compatible)."""

    name = "baseten"
    default_base_url = "https://inference.baseten.co/v1"
    strip_model_prefix = False


class MetaProvider(OpenAICompatible):
    """Meta Model API (OpenAI-compatible)."""

    name = "meta"
    default_base_url = "https://api.meta.ai/v1"
    strip_model_prefix = False


class MoonshotProvider(OpenAICompatible):
    """Moonshot / Kimi lab API."""

    name = "moonshot"
    default_base_url = "https://api.moonshot.ai/v1"


class QwenProvider(OpenAICompatible):
    """Qwen via DashScope compatible-mode."""

    name = "qwen"
    default_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class ZhipuProvider(OpenAICompatible):
    """Zhipu / Z.ai lab API."""

    name = "zhipu"
    default_base_url = "https://api.z.ai/api/paas/v4"

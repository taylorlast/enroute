"""Anthropic Messages API adapter.

Examples:
    >>> from enroute.providers.anthropic import AnthropicProvider
    >>> AnthropicProvider.default_base_url
    'https://api.anthropic.com'
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
    text_content,
)


class AnthropicProvider:
    """Adapter for Anthropic's Messages API.

    Args:
        api_key: Anthropic API key.
        base_url: API base URL.
        timeout_s: Request timeout in seconds.
        default_headers: Extra headers.
        anthropic_version: Value for the ``anthropic-version`` header.
    """

    name: str = "anthropic"
    default_base_url: str = "https://api.anthropic.com"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        timeout_s: float = 60.0,
        default_headers: dict[str, str] | None = None,
        anthropic_version: str = "2023-06-01",
    ) -> None:
        self.config = ProviderConfig(
            api_key=api_key,
            base_url=(base_url or self.default_base_url).rstrip("/"),
            timeout_s=timeout_s,
            default_headers=default_headers or {},
        )
        headers = {
            "x-api-key": api_key,
            "anthropic-version": anthropic_version,
            "content-type": "application/json",
            **self.config.default_headers,
        }
        self._client = httpx.Client(
            base_url=self.config.base_url,
            headers=headers,
            timeout=timeout_s,
        )
        self._aclient = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers=headers,
            timeout=timeout_s,
        )

    def _model_id(self, model: str) -> str:
        if model.startswith("anthropic/"):
            return model.split("/", 1)[1]
        return model

    def _encode_request(self, request: ChatRequest, *, stream: bool) -> dict[str, Any]:
        system_parts: list[str] = []
        messages: list[dict[str, Any]] = []
        for msg in request.messages:
            if msg.role == "system":
                system_parts.append(text_content(msg))
                continue
            if msg.role == "tool":
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": text_content(msg),
                            }
                        ],
                    }
                )
                continue
            if msg.role == "assistant" and msg.tool_calls:
                content: list[dict[str, Any]] = []
                text = text_content(msg)
                if text:
                    content.append({"type": "text", "text": text})
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {"raw": tc.function.arguments}
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.function.name,
                            "input": args,
                        }
                    )
                messages.append({"role": "assistant", "content": content})
                continue
            messages.append({"role": msg.role, "content": text_content(msg)})

        payload: dict[str, Any] = {
            "model": self._model_id(request.model),
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
            "stream": stream,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop is not None:
            payload["stop_sequences"] = (
                [request.stop] if isinstance(request.stop, str) else list(request.stop)
            )
        if request.tools:
            payload["tools"] = [
                {
                    "name": t.function.name,
                    "description": t.function.description or "",
                    "input_schema": t.function.parameters or {"type": "object", "properties": {}},
                }
                for t in request.tools
            ]
        payload.update(request.extra)
        return payload

    def _parse_response(self, data: dict[str, Any], *, latency_ms: float) -> ChatResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in data.get("content") or []:
            if block.get("type") == "text":
                text_parts.append(block.get("text") or "")
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block["id"],
                        function=FunctionCall(
                            name=block["name"],
                            arguments=json.dumps(block.get("input") or {}),
                        ),
                    )
                )
        stop = data.get("stop_reason")
        finish: FinishReason | str | None
        if stop == "end_turn":
            finish = FinishReason.STOP
        elif stop == "max_tokens":
            finish = FinishReason.LENGTH
        elif stop == "tool_use":
            finish = FinishReason.TOOL_CALLS
        else:
            finish = stop
        usage_raw = data.get("usage") or {}
        usage = Usage.from_counts(
            int(usage_raw.get("input_tokens") or 0),
            int(usage_raw.get("output_tokens") or 0),
        )
        message = Message(
            role="assistant",
            content="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls or None,
        )
        return ChatResponse(
            id=str(data.get("id") or ""),
            model=str(data.get("model") or ""),
            choices=[Choice(message=message, finish_reason=finish)],
            usage=usage,
            provider=self.name,
            raw=data,
            latency_ms=latency_ms,
        )

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Execute a non-streaming Messages API call.

        Args:
            request: Normalized chat request.

        Returns:
            Normalized chat response.
        """
        payload = self._encode_request(request, stream=False)
        started = time.perf_counter()
        try:
            response = self._client.post("/v1/messages", json=payload)
        except Exception as exc:  # noqa: BLE001
            raise map_transport_error(exc, provider=self.name, model=request.model) from exc
        raise_for_status(response, provider=self.name, model=request.model)
        return self._parse_response(
            response.json(), latency_ms=(time.perf_counter() - started) * 1000
        )

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        """Execute a streaming Messages API call.

        Args:
            request: Normalized chat request.

        Yields:
            Normalized stream chunks.
        """
        payload = self._encode_request(request, stream=True)
        message_id = ""
        model = self._model_id(request.model)
        try:
            with self._client.stream("POST", "/v1/messages", json=payload) as response:
                raise_for_status(response, provider=self.name, model=request.model)
                for data in iter_sse_lines(response):
                    event = json.loads(data)
                    etype = event.get("type")
                    if etype == "message_start":
                        msg = event.get("message") or {}
                        message_id = str(msg.get("id") or "")
                        model = str(msg.get("model") or model)
                    elif etype == "content_block_delta":
                        delta = event.get("delta") or {}
                        if delta.get("type") == "text_delta":
                            yield StreamChunk(
                                id=message_id,
                                model=model,
                                delta=StreamDelta(content=delta.get("text")),
                                provider=self.name,
                                raw=event,
                            )
                    elif etype == "message_delta":
                        usage_raw = event.get("usage") or {}
                        stop = (event.get("delta") or {}).get("stop_reason")
                        finish = (
                            FinishReason.STOP
                            if stop == "end_turn"
                            else FinishReason.LENGTH
                            if stop == "max_tokens"
                            else FinishReason.TOOL_CALLS
                            if stop == "tool_use"
                            else stop
                        )
                        yield StreamChunk(
                            id=message_id,
                            model=model,
                            finish_reason=finish,
                            usage=Usage.from_counts(
                                0,
                                int(usage_raw.get("output_tokens") or 0),
                            ),
                            provider=self.name,
                            raw=event,
                        )
        except EnrouteError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise map_transport_error(exc, provider=self.name, model=request.model) from exc

    async def achat(self, request: ChatRequest) -> ChatResponse:
        """Async non-streaming Messages API call.

        Args:
            request: Normalized chat request.

        Returns:
            Normalized chat response.
        """
        payload = self._encode_request(request, stream=False)
        started = time.perf_counter()
        try:
            response = await self._aclient.post("/v1/messages", json=payload)
        except Exception as exc:  # noqa: BLE001
            raise map_transport_error(exc, provider=self.name, model=request.model) from exc
        raise_for_status(response, provider=self.name, model=request.model)
        return self._parse_response(
            response.json(), latency_ms=(time.perf_counter() - started) * 1000
        )

    async def astream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Async streaming Messages API call.

        Args:
            request: Normalized chat request.

        Yields:
            Normalized stream chunks.
        """
        payload = self._encode_request(request, stream=True)
        message_id = ""
        model = self._model_id(request.model)
        try:
            async with self._aclient.stream("POST", "/v1/messages", json=payload) as response:
                raise_for_status(response, provider=self.name, model=request.model)
                async for data in aiter_sse_lines(response):
                    event = json.loads(data)
                    etype = event.get("type")
                    if etype == "message_start":
                        msg = event.get("message") or {}
                        message_id = str(msg.get("id") or "")
                        model = str(msg.get("model") or model)
                    elif etype == "content_block_delta":
                        delta = event.get("delta") or {}
                        if delta.get("type") == "text_delta":
                            yield StreamChunk(
                                id=message_id,
                                model=model,
                                delta=StreamDelta(content=delta.get("text")),
                                provider=self.name,
                                raw=event,
                            )
                    elif etype == "message_delta":
                        usage_raw = event.get("usage") or {}
                        stop = (event.get("delta") or {}).get("stop_reason")
                        finish = (
                            FinishReason.STOP
                            if stop == "end_turn"
                            else FinishReason.LENGTH
                            if stop == "max_tokens"
                            else FinishReason.TOOL_CALLS
                            if stop == "tool_use"
                            else stop
                        )
                        yield StreamChunk(
                            id=message_id,
                            model=model,
                            finish_reason=finish,
                            usage=Usage.from_counts(
                                0,
                                int(usage_raw.get("output_tokens") or 0),
                            ),
                            provider=self.name,
                            raw=event,
                        )
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

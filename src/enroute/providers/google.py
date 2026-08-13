"""Google Gemini generateContent adapter.

Examples:
    >>> from enroute.providers.google import GoogleProvider
    >>> GoogleProvider.default_base_url
    'https://generativelanguage.googleapis.com/v1beta'
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


class GoogleProvider:
    """Adapter for Google Gemini's generateContent API.

    Args:
        api_key: Google AI Studio API key.
        base_url: API base URL.
        timeout_s: Request timeout in seconds.
        default_headers: Extra headers.
    """

    name: str = "google"
    default_base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        timeout_s: float = 60.0,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.config = ProviderConfig(
            api_key=api_key,
            base_url=(base_url or self.default_base_url).rstrip("/"),
            timeout_s=timeout_s,
            default_headers=default_headers or {},
        )
        headers = {"Content-Type": "application/json", **self.config.default_headers}
        self._client = httpx.Client(
            base_url=self.config.base_url,
            headers=headers,
            timeout=timeout_s,
            params={"key": api_key},
        )
        self._aclient = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers=headers,
            timeout=timeout_s,
            params={"key": api_key},
        )

    def _model_id(self, model: str) -> str:
        if model.startswith("google/") or model.startswith("gemini/"):
            return model.split("/", 1)[1]
        return model

    def _encode_request(self, request: ChatRequest) -> dict[str, Any]:
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []
        for msg in request.messages:
            if msg.role == "system":
                system_parts.append(text_content(msg))
                continue
            role = "model" if msg.role == "assistant" else "user"
            parts: list[dict[str, Any]] = [{"text": text_content(msg)}] if text_content(msg) else []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {"raw": tc.function.arguments}
                    parts.append(
                        {
                            "functionCall": {
                                "name": tc.function.name,
                                "args": args,
                            }
                        }
                    )
            if msg.role == "tool":
                parts = [
                    {
                        "functionResponse": {
                            "name": msg.name or "tool",
                            "response": {"content": text_content(msg)},
                        }
                    }
                ]
                role = "user"
            contents.append({"role": role, "parts": parts})

        payload: dict[str, Any] = {"contents": contents}
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        generation: dict[str, Any] = {}
        if request.temperature is not None:
            generation["temperature"] = request.temperature
        if request.top_p is not None:
            generation["topP"] = request.top_p
        if request.max_tokens is not None:
            generation["maxOutputTokens"] = request.max_tokens
        if request.stop is not None:
            generation["stopSequences"] = (
                [request.stop] if isinstance(request.stop, str) else list(request.stop)
            )
        if generation:
            payload["generationConfig"] = generation
        if request.tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": t.function.name,
                            "description": t.function.description or "",
                            "parameters": t.function.parameters or {"type": "object"},
                        }
                        for t in request.tools
                    ]
                }
            ]
        payload.update(request.extra)
        return payload

    def _parse_response(
        self,
        data: dict[str, Any],
        *,
        model: str,
        latency_ms: float,
    ) -> ChatResponse:
        candidate = (data.get("candidates") or [{}])[0]
        parts = ((candidate.get("content") or {}).get("parts")) or []
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for i, part in enumerate(parts):
            if "text" in part:
                text_parts.append(part["text"])
            if "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append(
                    ToolCall(
                        id=f"call_{i}",
                        function=FunctionCall(
                            name=fc.get("name") or "",
                            arguments=json.dumps(fc.get("args") or {}),
                        ),
                    )
                )
        finish_map = {
            "STOP": FinishReason.STOP,
            "MAX_TOKENS": FinishReason.LENGTH,
            "SAFETY": FinishReason.CONTENT_FILTER,
        }
        finish = finish_map.get(candidate.get("finishReason"), candidate.get("finishReason"))
        usage_meta = data.get("usageMetadata") or {}
        usage = Usage.from_counts(
            int(usage_meta.get("promptTokenCount") or 0),
            int(usage_meta.get("candidatesTokenCount") or 0),
        )
        return ChatResponse(
            id=str(data.get("responseId") or data.get("id") or ""),
            model=model,
            choices=[
                Choice(
                    message=Message(
                        role="assistant",
                        content="".join(text_parts) if text_parts else None,
                        tool_calls=tool_calls or None,
                    ),
                    finish_reason=finish,
                )
            ],
            usage=usage,
            provider=self.name,
            raw=data,
            latency_ms=latency_ms,
        )

    def _parse_stream_chunk(self, data: dict[str, Any], *, model: str) -> StreamChunk:
        candidate = (data.get("candidates") or [{}])[0]
        parts = ((candidate.get("content") or {}).get("parts")) or []
        text = "".join(p.get("text") or "" for p in parts if "text" in p)
        finish_map = {
            "STOP": FinishReason.STOP,
            "MAX_TOKENS": FinishReason.LENGTH,
            "SAFETY": FinishReason.CONTENT_FILTER,
        }
        usage = None
        if data.get("usageMetadata"):
            um = data["usageMetadata"]
            usage = Usage.from_counts(
                int(um.get("promptTokenCount") or 0),
                int(um.get("candidatesTokenCount") or 0),
            )
        return StreamChunk(
            id=str(data.get("responseId") or ""),
            model=model,
            delta=StreamDelta(content=text or None),
            finish_reason=finish_map.get(
                candidate.get("finishReason"), candidate.get("finishReason")
            ),
            usage=usage,
            provider=self.name,
            raw=data,
        )

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Execute a non-streaming generateContent call.

        Args:
            request: Normalized chat request.

        Returns:
            Normalized chat response.
        """
        model = self._model_id(request.model)
        payload = self._encode_request(request)
        started = time.perf_counter()
        try:
            response = self._client.post(f"/models/{model}:generateContent", json=payload)
        except Exception as exc:  # noqa: BLE001
            raise map_transport_error(exc, provider=self.name, model=request.model) from exc
        raise_for_status(response, provider=self.name, model=request.model)
        return self._parse_response(
            response.json(),
            model=model,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        """Execute a streaming generateContent call.

        Args:
            request: Normalized chat request.

        Yields:
            Normalized stream chunks.
        """
        model = self._model_id(request.model)
        payload = self._encode_request(request)
        try:
            with self._client.stream(
                "POST",
                f"/models/{model}:streamGenerateContent",
                params={"alt": "sse"},
                json=payload,
            ) as response:
                raise_for_status(response, provider=self.name, model=request.model)
                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    yield self._parse_stream_chunk(json.loads(data), model=model)
        except EnrouteError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise map_transport_error(exc, provider=self.name, model=request.model) from exc

    async def achat(self, request: ChatRequest) -> ChatResponse:
        """Async non-streaming generateContent call.

        Args:
            request: Normalized chat request.

        Returns:
            Normalized chat response.
        """
        model = self._model_id(request.model)
        payload = self._encode_request(request)
        started = time.perf_counter()
        try:
            response = await self._aclient.post(f"/models/{model}:generateContent", json=payload)
        except Exception as exc:  # noqa: BLE001
            raise map_transport_error(exc, provider=self.name, model=request.model) from exc
        raise_for_status(response, provider=self.name, model=request.model)
        return self._parse_response(
            response.json(),
            model=model,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    async def astream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Async streaming generateContent call.

        Args:
            request: Normalized chat request.

        Yields:
            Normalized stream chunks.
        """
        model = self._model_id(request.model)
        payload = self._encode_request(request)
        try:
            async with self._aclient.stream(
                "POST",
                f"/models/{model}:streamGenerateContent",
                params={"alt": "sse"},
                json=payload,
            ) as response:
                raise_for_status(response, provider=self.name, model=request.model)
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    yield self._parse_stream_chunk(json.loads(data), model=model)
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

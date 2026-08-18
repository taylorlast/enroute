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
from dataclasses import dataclass, field
from typing import Any

import httpx

from enroute.errors import EnrouteError
from enroute.providers.base import (
    ProviderConfig,
    aiter_sse_lines,
    araise_for_stream_status,
    iter_sse_lines,
    map_transport_error,
    raise_for_status,
    raise_for_stream_status,
)
from enroute.providers.structured import (
    JSON_ONLY_INSTRUCTION,
    STRUCTURED_TOOL_DESCRIPTION,
    schema_tool_name,
    structured_schema,
    wants_json_object,
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


@dataclass
class _StreamState:
    """Cross-event state for one Anthropic stream.

    Anthropic describes a response as indexed content blocks, so a delta only
    means something in the context of the block it belongs to: identical
    ``input_json_delta`` events belong to different tool calls. Usage is split
    across the first and last event, so both are accumulated here.
    """

    message_id: str = ""
    model: str = ""
    prompt_tokens: int = 0
    block_types: dict[int, str] = field(default_factory=dict)
    tool_slots: dict[int, int] = field(default_factory=dict)
    structured_tool: str | None = None

    def tool_slot(self, block_index: int) -> int:
        """Assign this block the next position in OpenAI's ``tool_calls`` array.

        Args:
            block_index: Anthropic content block index.

        Returns:
            The stable OpenAI array position for the block.
        """
        if block_index not in self.tool_slots:
            self.tool_slots[block_index] = len(self.tool_slots)
        return self.tool_slots[block_index]


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
        # Current Claude models reject ``thinking.type.enabled``. Adaptive is
        # what they accept, and without it a thinking model sits silent.
        self._thinking: dict[str, str | None] = {}

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
            if msg.role == "assistant" and (msg.tool_calls or msg.reasoning):
                content: list[dict[str, Any]] = []
                # A replayed thinking block must lead the message and carry its
                # signature, or Anthropic rejects the turn.
                if msg.reasoning and msg.reasoning_signature:
                    content.append(
                        {
                            "type": "thinking",
                            "thinking": msg.reasoning,
                            "signature": msg.reasoning_signature,
                        }
                    )
                text = text_content(msg)
                if text:
                    content.append({"type": "text", "text": text})
                for tc in msg.tool_calls or []:
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
        if wants_json_object(request):
            system_parts.append(JSON_ONLY_INSTRUCTION)
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
            if request.tool_choice is not None:
                payload["tool_choice"] = self._tool_choice(request.tool_choice)
        elif (tool_name := schema_tool_name(request)) is not None:
            payload["tools"] = [
                {
                    "name": tool_name,
                    "description": STRUCTURED_TOOL_DESCRIPTION,
                    "input_schema": structured_schema(request),
                }
            ]
            payload["tool_choice"] = {"type": "tool", "name": tool_name}
        thinking = self._thinking_for(request)
        if thinking is not None:
            payload["thinking"] = {"type": thinking}
            # Adaptive thinking rejects a custom temperature / top_p.
            payload.pop("temperature", None)
            payload.pop("top_p", None)
        payload.update(request.extra)
        return payload

    def _thinking_for(self, request: ChatRequest) -> str | None:
        """Thinking mode to request, unless the caller already set one.

        Args:
            request: Normalized chat request.

        Returns:
            ``"adaptive"`` by default, or ``None`` after the host rejected it.
        """
        if "thinking" in request.extra:
            return None
        return self._thinking.setdefault(self._model_id(request.model), "adaptive")

    def _adapt_thinking(self, exc: EnrouteError, request: ChatRequest) -> bool:
        """Drop automatic thinking when this model rejects it.

        Args:
            exc: Classified error from the previous attempt.
            request: The request that failed.

        Returns:
            ``True`` when the request is worth retrying without thinking.
        """
        if exc.status_code != 400 or "thinking" in request.extra:
            return False
        blob = f"{exc.message} {exc.body}".lower()
        if "thinking" not in blob:
            return False
        model = self._model_id(request.model)
        if self._thinking.get(model) is None:
            return False
        self._thinking[model] = None
        return True

    def _tool_choice(self, choice: str | dict[str, Any]) -> dict[str, Any]:
        """Translate an OpenAI ``tool_choice`` into Anthropic's shape.

        Args:
            choice: OpenAI-style tool choice, either a keyword or a forced tool.

        Returns:
            Anthropic's ``tool_choice`` object.
        """
        if isinstance(choice, str):
            if choice == "required":
                return {"type": "any"}
            if choice == "none":
                return {"type": "none"}
            return {"type": "auto"}
        name = (choice.get("function") or {}).get("name")
        if name:
            return {"type": "tool", "name": name}
        return {"type": "auto"}

    def _parse_response(
        self,
        data: dict[str, Any],
        *,
        latency_ms: float,
        structured_tool: str | None = None,
    ) -> ChatResponse:
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        signature: str | None = None
        tool_calls: list[ToolCall] = []
        for block in data.get("content") or []:
            kind = block.get("type")
            if kind == "text":
                text_parts.append(block.get("text") or "")
            elif kind == "thinking":
                reasoning_parts.append(block.get("thinking") or "")
                signature = block.get("signature") or signature
            elif kind == "redacted_thinking":
                signature = block.get("data") or signature
            elif kind == "tool_use":
                if block.get("name") == structured_tool:
                    # The forced tool is our own structured-output shim, so its
                    # input is the answer, not a call the caller should see.
                    text_parts.append(json.dumps(block.get("input") or {}))
                    continue
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
            finish = FinishReason.STOP if structured_tool else FinishReason.TOOL_CALLS
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
            reasoning="".join(reasoning_parts) or None,
            reasoning_signature=signature,
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
        while True:
            payload = self._encode_request(request, stream=False)
            started = time.perf_counter()
            try:
                response = self._client.post("/v1/messages", json=payload)
            except Exception as exc:  # noqa: BLE001
                raise map_transport_error(exc, provider=self.name, model=request.model) from exc
            try:
                raise_for_status(response, provider=self.name, model=request.model)
            except EnrouteError as exc:
                if self._adapt_thinking(exc, request):
                    continue
                raise
            return self._parse_response(
                response.json(),
                latency_ms=(time.perf_counter() - started) * 1000,
                structured_tool=schema_tool_name(request),
            )

    def _finish_reason(self, stop: str | None) -> FinishReason | str | None:
        """Map an Anthropic stop reason onto the shared finish enum.

        Args:
            stop: Anthropic ``stop_reason`` value.

        Returns:
            A :class:`~enroute.types.FinishReason`, the raw string, or ``None``.
        """
        if stop == "end_turn":
            return FinishReason.STOP
        if stop == "max_tokens":
            return FinishReason.LENGTH
        if stop == "tool_use":
            return FinishReason.TOOL_CALLS
        return stop

    def _chunk(self, state: _StreamState, delta: StreamDelta, event: dict[str, Any]) -> StreamChunk:
        """Wrap a delta in a chunk carrying this stream's accumulated identity.

        Args:
            state: Stream state holding the completion id and model.
            delta: The normalized delta to emit.
            event: Native Anthropic event, attached for debugging.

        Returns:
            A normalized stream chunk.
        """
        return StreamChunk(
            id=state.message_id,
            model=state.model,
            delta=delta,
            provider=self.name,
            raw=event,
        )

    def _block_start(self, event: dict[str, Any], state: _StreamState) -> StreamChunk | None:
        """Handle ``content_block_start``, which opens a text, thinking, or tool block.

        Args:
            event: Parsed Anthropic SSE JSON object.
            state: Mutable stream state.

        Returns:
            The opening ``tool_calls`` fragment for a tool block, else ``None``.
        """
        index = int(event.get("index") or 0)
        block = event.get("content_block") or {}
        kind = str(block.get("type") or "")
        state.block_types[index] = kind
        if kind in {"thinking", "redacted_thinking"}:
            return self._chunk(state, StreamDelta(reasoning_started=True), event)
        if kind != "tool_use":
            return None
        if block.get("name") == state.structured_tool:
            return None
        return self._chunk(
            state,
            StreamDelta(
                tool_calls=[
                    {
                        "index": state.tool_slot(index),
                        "id": block.get("id") or "",
                        "type": "function",
                        "function": {"name": block.get("name") or "", "arguments": ""},
                    }
                ]
            ),
            event,
        )

    def _block_delta(self, event: dict[str, Any], state: _StreamState) -> StreamChunk | None:
        """Handle ``content_block_delta`` for text, thinking, and tool arguments.

        Args:
            event: Parsed Anthropic SSE JSON object.
            state: Mutable stream state.

        Returns:
            A normalized chunk, or ``None`` for an empty delta.
        """
        index = int(event.get("index") or 0)
        delta = event.get("delta") or {}
        kind = delta.get("type")
        if kind == "text_delta":
            text = delta.get("text")
            return self._chunk(state, StreamDelta(content=text), event) if text else None
        if kind == "thinking_delta":
            thinking = delta.get("thinking")
            return self._chunk(state, StreamDelta(reasoning=thinking), event) if thinking else None
        if kind == "signature_delta":
            signature = delta.get("signature")
            return (
                self._chunk(state, StreamDelta(reasoning_signature=signature), event)
                if signature
                else None
            )
        if kind == "input_json_delta":
            fragment = delta.get("partial_json")
            if not fragment:
                return None
            if state.block_types.get(index) == "tool_use" and state.structured_tool:
                # Structured output is a forced tool underneath, so its argument
                # fragments are the JSON answer and stream as content.
                return self._chunk(state, StreamDelta(content=fragment), event)
            return self._chunk(
                state,
                StreamDelta(
                    tool_calls=[
                        {
                            "index": state.tool_slot(index),
                            "function": {"arguments": fragment},
                        }
                    ]
                ),
                event,
            )
        return None

    def _chunk_from_event(self, event: dict[str, Any], state: _StreamState) -> StreamChunk | None:
        """Translate one Anthropic SSE event into a normalized chunk.

        ``message_start`` carries ``input_tokens`` while ``message_delta``
        usually carries only ``output_tokens``, and both are needed to bill a
        stream, so token counts accumulate on ``state``.

        Args:
            event: Parsed Anthropic SSE JSON object.
            state: Mutable stream state, updated in place.

        Returns:
            A normalized chunk, or ``None`` for events with no client-visible
            delta such as ``ping`` and ``content_block_stop``.
        """
        etype = event.get("type")
        if etype == "message_start":
            msg = event.get("message") or {}
            state.message_id = str(msg.get("id") or state.message_id)
            state.model = str(msg.get("model") or state.model)
            usage_raw = msg.get("usage") or {}
            state.prompt_tokens = int(usage_raw.get("input_tokens") or state.prompt_tokens)
            return None
        if etype == "content_block_start":
            return self._block_start(event, state)
        if etype == "content_block_delta":
            return self._block_delta(event, state)
        if etype == "content_block_stop":
            index = int(event.get("index") or 0)
            if state.block_types.get(index) in {"thinking", "redacted_thinking"}:
                return self._chunk(state, StreamDelta(reasoning_finished=True), event)
            return None
        if etype == "message_delta":
            usage_raw = event.get("usage") or {}
            if usage_raw.get("input_tokens") is not None:
                state.prompt_tokens = int(usage_raw["input_tokens"])
            stop = (event.get("delta") or {}).get("stop_reason")
            finish = self._finish_reason(stop)
            if stop == "tool_use" and state.structured_tool:
                finish = FinishReason.STOP
            return StreamChunk(
                id=state.message_id,
                model=state.model,
                finish_reason=finish,
                usage=Usage.from_counts(
                    state.prompt_tokens,
                    int(usage_raw.get("output_tokens") or 0),
                ),
                provider=self.name,
                raw=event,
            )
        return None

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        """Execute a streaming Messages API call.

        Args:
            request: Normalized chat request.

        Yields:
            Normalized stream chunks.
        """
        while True:
            payload = self._encode_request(request, stream=True)
            state = _StreamState(
                model=self._model_id(request.model),
                structured_tool=schema_tool_name(request),
            )
            try:
                with self._client.stream("POST", "/v1/messages", json=payload) as response:
                    raise_for_stream_status(response, provider=self.name, model=request.model)
                    for data in iter_sse_lines(response):
                        chunk = self._chunk_from_event(json.loads(data), state)
                        if chunk is not None:
                            yield chunk
                return
            except EnrouteError as exc:
                if self._adapt_thinking(exc, request):
                    continue
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
        while True:
            payload = self._encode_request(request, stream=False)
            started = time.perf_counter()
            try:
                response = await self._aclient.post("/v1/messages", json=payload)
            except Exception as exc:  # noqa: BLE001
                raise map_transport_error(exc, provider=self.name, model=request.model) from exc
            try:
                raise_for_status(response, provider=self.name, model=request.model)
            except EnrouteError as exc:
                if self._adapt_thinking(exc, request):
                    continue
                raise
            return self._parse_response(
                response.json(),
                latency_ms=(time.perf_counter() - started) * 1000,
                structured_tool=schema_tool_name(request),
            )

    async def astream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Async streaming Messages API call.

        Args:
            request: Normalized chat request.

        Yields:
            Normalized stream chunks.
        """
        while True:
            payload = self._encode_request(request, stream=True)
            state = _StreamState(
                model=self._model_id(request.model),
                structured_tool=schema_tool_name(request),
            )
            try:
                async with self._aclient.stream("POST", "/v1/messages", json=payload) as response:
                    await araise_for_stream_status(
                        response, provider=self.name, model=request.model
                    )
                    async for data in aiter_sse_lines(response):
                        chunk = self._chunk_from_event(json.loads(data), state)
                        if chunk is not None:
                            yield chunk
                return
            except EnrouteError as exc:
                if self._adapt_thinking(exc, request):
                    continue
                raise
            except Exception as exc:  # noqa: BLE001
                raise map_transport_error(exc, provider=self.name, model=request.model) from exc

    def close(self) -> None:
        """Close the sync HTTP client."""
        self._client.close()

    async def aclose(self) -> None:
        """Close the async HTTP client."""
        await self._aclient.aclose()

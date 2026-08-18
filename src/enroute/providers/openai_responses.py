"""OpenAI Responses API adapter.

OpenAI has split its own surface in two. ``/chat/completions`` still serves
plain text, but it rejects function tools for every current model:

    Function tools with reasoning_effort are not supported for gpt-5.6 in
    /v1/chat/completions. To use function tools, use /v1/responses or set
    reasoning_effort to 'none'.

Turning reasoning off to keep tools is the wrong trade, so tool calling moves
here instead. ``/responses`` is also the only OpenAI endpoint that streams
reasoning summaries, which is what stops a thinking model from looking frozen.

The wire format is unrelated to Chat Completions: messages become a flat
``input`` list where tool calls and their results are items rather than message
fields, output arrives as typed SSE events rather than choice deltas, and usage
is reported as ``input_tokens``/``output_tokens``. This adapter translates all
of it back to the same :class:`~enroute.types.StreamChunk` every other provider
yields, so callers never branch on which OpenAI endpoint served them.

Two parameters have no equivalent here and are dropped: ``stop`` and ``seed``.
The endpoint rejects both outright, so a request carrying them still succeeds
rather than failing over to a lesser host.

Examples:
    >>> from enroute.providers.openai_responses import OpenAIResponsesProvider
    >>> OpenAIResponsesProvider.endpoint_path
    '/responses'
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

from enroute.providers.base import (
    aiter_sse_lines,
    araise_for_stream_status,
    classify_http_error,
    iter_sse_lines,
    raise_for_stream_status,
)
from enroute.providers.openai_compatible import HostQuirks, OpenAICompatible
from enroute.providers.structured import structured_schema, wants_json_object
from enroute.types import (
    AudioContent,
    ChatRequest,
    ChatResponse,
    Choice,
    FinishReason,
    FunctionCall,
    ImageContent,
    Message,
    StreamChunk,
    StreamDelta,
    TextContent,
    ToolCall,
    Usage,
    text_content,
)

# The endpoint rejects a smaller cap outright, where Chat Completions accepts
# any positive value. Raising a tiny cap costs a few tokens; refusing the request
# costs the caller their answer.
MIN_OUTPUT_TOKENS = 16

# Why a response stopped, in the endpoint's own vocabulary.
_INCOMPLETE_REASONS = {
    "max_output_tokens": FinishReason.LENGTH,
    "content_filter": FinishReason.CONTENT_FILTER,
}


@dataclass
class _StreamState:
    """Bookkeeping carried across the events of one streamed response.

    The endpoint identifies a tool call by an opaque item id and never by an
    index, but OpenAI's streaming tool-call shape is index-addressed, so the
    order of first appearance has to be remembered to assign indices.

    Attributes:
        id: Response id, learned from the first event.
        model: Model that served the response.
        slots: Tool call item id to its index in the output.
    """

    id: str = ""
    model: str = ""
    slots: dict[str, int] = field(default_factory=dict)

    def slot(self, item_id: str) -> int:
        """Index for a tool call item, assigned on first sight.

        Args:
            item_id: The endpoint's item id for the call.

        Returns:
            A stable zero-based index for this call.
        """
        if item_id not in self.slots:
            self.slots[item_id] = len(self.slots)
        return self.slots[item_id]


def _tool_fragment(index: int, **fields: Any) -> list[dict[str, Any]]:
    """Build an OpenAI-shaped streaming tool call fragment.

    Args:
        index: Tool call index within the response.
        **fields: Either ``id`` and ``name``, or ``arguments``.

    Returns:
        A single-element list suitable for :attr:`StreamDelta.tool_calls`.
    """
    fragment: dict[str, Any] = {"index": index, "type": "function"}
    if "id" in fields:
        fragment["id"] = fields["id"]
    function: dict[str, Any] = {}
    if "name" in fields:
        function["name"] = fields["name"]
    if "arguments" in fields:
        function["arguments"] = fields["arguments"]
    fragment["function"] = function
    return [fragment]


def _usage(raw: dict[str, Any] | None) -> Usage | None:
    """Convert Responses token counts to normalized usage.

    Reasoning tokens are already inside ``output_tokens``, which is also how they
    are billed, so they need no separate handling.

    Args:
        raw: The ``usage`` object, when present.

    Returns:
        Normalized usage, or ``None`` when the endpoint reported none.

    Examples:
        >>> _usage({"input_tokens": 9, "output_tokens": 5}).total_tokens
        14
    """
    if not raw:
        return None
    return Usage.from_counts(int(raw.get("input_tokens") or 0), int(raw.get("output_tokens") or 0))


class OpenAIResponsesProvider(OpenAICompatible):
    """OpenAI Responses API, normalized to the Chat Completions shape.

    Args:
        api_key: OpenAI API key.
        reasoning_summaries: Whether to ask for streamed reasoning summaries.
            On by default, because a thinking model that emits nothing for ten
            seconds is indistinguishable from a hung connection.
        store: Whether to let OpenAI retain the response. Off by default to
            match Chat Completions, which retains nothing.
        **kwargs: Forwarded to :class:`OpenAICompatible`.

    Examples:
        >>> provider = OpenAIResponsesProvider("sk-test")
        >>> provider.endpoint_path
        '/responses'
    """

    name = "openai"
    default_base_url = "https://api.openai.com/v1"
    endpoint_path = "/responses"

    def __init__(
        self,
        api_key: str,
        *,
        reasoning_summaries: bool = True,
        store: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(api_key, **kwargs)
        self._reasoning_summaries = reasoning_summaries
        self._store = store

    @staticmethod
    def serves_exactly(request: ChatRequest) -> bool:
        """Whether this endpoint can honour every part of a request.

        Args:
            request: Normalized chat request.

        Returns:
            ``False`` when something would have to be dropped or adjusted, which
            is what tells the OpenAI dispatcher to prefer Chat Completions.

        Examples:
            >>> from enroute.types import Message
            >>> hello = [Message(role="user", content="hi")]
            >>> req = ChatRequest(model="openai/gpt-5.6", messages=hello, seed=7)
            >>> OpenAIResponsesProvider.serves_exactly(req)
            False
        """
        if request.stop is not None or request.seed is not None:
            return False
        return request.max_tokens is None or request.max_tokens >= MIN_OUTPUT_TOKENS

    def _content(self, message: Message) -> str | list[dict[str, Any]]:
        """Encode message content as ``input`` content.

        Plain strings are sent as-is: the endpoint distinguishes ``input_text``
        from ``output_text`` inside parts arrays, and passing a string sidesteps
        having to pick correctly per role.

        Args:
            message: Message to encode.

        Returns:
            A string, or a list of typed content parts for multimodal input.
        """
        if message.content is None or isinstance(message.content, str):
            return message.content or ""
        if message.role == "assistant":
            return text_content(message)
        parts: list[dict[str, Any]] = []
        for part in message.content:
            if isinstance(part, TextContent):
                parts.append({"type": "input_text", "text": part.text})
            elif isinstance(part, ImageContent):
                image: dict[str, Any] = {"type": "input_image", "image_url": part.url}
                if part.detail:
                    image["detail"] = part.detail
                parts.append(image)
            elif isinstance(part, AudioContent):
                parts.append(
                    {
                        "type": "input_audio",
                        "input_audio": {"data": part.data, "format": part.format},
                    }
                )
        return parts

    def _encode_input(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Flatten messages into the endpoint's ``input`` list.

        Tool calls and tool results are sibling items here rather than fields on
        a message, so one assistant message can expand into several items.

        Args:
            messages: Conversation messages.

        Returns:
            The ``input`` list.
        """
        items: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "tool":
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id or "",
                        "output": text_content(message),
                    }
                )
                continue
            if message.tool_calls:
                text = text_content(message)
                if text:
                    items.append({"role": "assistant", "content": text})
                items.extend(
                    {
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    }
                    for call in message.tool_calls
                )
                continue
            items.append({"role": str(message.role), "content": self._content(message)})
        return items

    def _encode_tools(self, request: ChatRequest) -> list[dict[str, Any]]:
        """Encode tools in the endpoint's flattened form.

        Args:
            request: Normalized chat request.

        Returns:
            Tool definitions without Chat Completions' ``function`` wrapper.
        """
        tools: list[dict[str, Any]] = []
        for tool in request.tools or []:
            fn = tool.function
            spec: dict[str, Any] = {
                "type": "function",
                "name": fn.name,
                "parameters": fn.parameters,
            }
            if fn.description:
                spec["description"] = fn.description
            if fn.strict is not None:
                spec["strict"] = fn.strict
            tools.append(spec)
        return tools

    def _encode_tool_choice(self, choice: str | dict[str, Any]) -> str | dict[str, Any] | None:
        """Translate a Chat Completions ``tool_choice`` for this endpoint.

        Args:
            choice: Keyword, or a forced tool in either endpoint's shape.

        Returns:
            The translated choice, or ``None`` when it cannot be interpreted.

        Examples:
            >>> p = OpenAIResponsesProvider("sk-test")
            >>> p._encode_tool_choice({"type": "function", "function": {"name": "f"}})
            {'type': 'function', 'name': 'f'}
        """
        if isinstance(choice, str):
            return choice
        name = (choice.get("function") or {}).get("name") or choice.get("name")
        return {"type": "function", "name": name} if name else None

    def _encode_format(self, request: ChatRequest) -> dict[str, Any] | None:
        """Encode ``response_format`` as the endpoint's ``text.format``.

        Args:
            request: Normalized chat request.

        Returns:
            The ``text`` object, or ``None`` when no format was requested.
        """
        schema = structured_schema(request)
        if schema is not None:
            payload = (request.response_format and request.response_format.json_schema) or {}
            spec: dict[str, Any] = {
                "type": "json_schema",
                "name": payload.get("name") or "response",
                "schema": schema,
            }
            if payload.get("strict") is not None:
                spec["strict"] = payload["strict"]
            return {"format": spec}
        if wants_json_object(request):
            return {"format": {"type": "json_object"}}
        return None

    def _encode_request(
        self, request: ChatRequest, *, stream: bool, quirks: HostQuirks | None = None
    ) -> dict[str, Any]:
        quirks = quirks or HostQuirks()
        payload: dict[str, Any] = {
            "model": self._model_id(request.model),
            "input": self._encode_input(request.messages),
            "stream": stream,
            "store": self._store,
        }
        if quirks.sampling:
            for key in ("temperature", "top_p"):
                value = getattr(request, key)
                if value is not None:
                    payload[key] = value
        if request.max_tokens is not None:
            payload["max_output_tokens"] = max(request.max_tokens, MIN_OUTPUT_TOKENS)
        if request.user is not None:
            payload["user"] = request.user
        if request.tools:
            payload["tools"] = self._encode_tools(request)
        if request.tool_choice is not None:
            choice = self._encode_tool_choice(request.tool_choice)
            if choice is not None:
                payload["tool_choice"] = choice
        text = self._encode_format(request)
        if text is not None:
            payload["text"] = text
        if self._reasoning_summaries:
            payload["reasoning"] = {"summary": "auto"}
        payload.update(request.extra)
        return payload

    def _finish_reason(self, response: dict[str, Any]) -> FinishReason:
        """Derive a normalized finish reason from a terminal response object.

        Args:
            response: The ``response`` object from a terminal event or POST.

        Returns:
            The normalized finish reason.
        """
        if response.get("status") == "incomplete":
            reason = (response.get("incomplete_details") or {}).get("reason")
            return _INCOMPLETE_REASONS.get(str(reason), FinishReason.UNKNOWN)
        for item in response.get("output") or []:
            if item.get("type") == "function_call":
                return FinishReason.TOOL_CALLS
        return FinishReason.STOP

    def _parse_response(self, data: dict[str, Any], *, latency_ms: float) -> ChatResponse:
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for item in data.get("output") or []:
            kind = item.get("type")
            if kind == "message":
                text_parts.extend(
                    part.get("text") or ""
                    for part in item.get("content") or []
                    if part.get("type") == "output_text"
                )
            elif kind == "reasoning":
                reasoning_parts.extend(
                    part.get("text") or ""
                    for part in item.get("summary") or []
                    if part.get("type") == "summary_text"
                )
            elif kind == "function_call":
                tool_calls.append(
                    ToolCall(
                        id=str(item.get("call_id") or item.get("id") or ""),
                        function=FunctionCall(
                            name=str(item.get("name") or ""),
                            arguments=str(item.get("arguments") or "{}"),
                        ),
                    )
                )
        message = Message(
            role="assistant",
            content="".join(text_parts) or None,
            tool_calls=tool_calls or None,
            reasoning="".join(reasoning_parts) or None,
        )
        return ChatResponse(
            id=str(data.get("id") or ""),
            model=str(data.get("model") or ""),
            choices=[Choice(message=message, finish_reason=self._finish_reason(data))],
            usage=_usage(data.get("usage")) or Usage(),
            provider=self.name,
            created=data.get("created_at"),
            raw=data,
            latency_ms=latency_ms,
        )

    def _chunk(
        self,
        state: _StreamState,
        *,
        delta: StreamDelta,
        finish_reason: FinishReason | None = None,
        usage: Usage | None = None,
        raw: dict[str, Any],
    ) -> StreamChunk:
        return StreamChunk(
            id=state.id,
            model=state.model,
            delta=delta,
            finish_reason=finish_reason,
            usage=usage,
            provider=self.name,
            raw=raw,
        )

    def _chunk_from_event(
        self, event: dict[str, Any], state: _StreamState, *, model: str
    ) -> StreamChunk | None:
        """Translate one Responses SSE event into a stream chunk.

        Args:
            event: The decoded event object.
            state: Mutable stream state, updated in place.
            model: Requested model id, used for errors.

        Returns:
            A chunk to yield, or ``None`` for events that carry no output.

        Raises:
            EnrouteError: When the endpoint reports the response failed.
        """
        kind = event.get("type") or ""
        if kind == "response.created":
            response = event.get("response") or {}
            state.id = str(response.get("id") or "")
            state.model = str(response.get("model") or "")
            return self._chunk(state, delta=StreamDelta(role="assistant"), raw=event)
        if kind == "response.output_text.delta":
            return self._chunk(
                state, delta=StreamDelta(content=event.get("delta") or ""), raw=event
            )
        if kind == "response.reasoning_summary_text.delta":
            return self._chunk(
                state, delta=StreamDelta(reasoning=event.get("delta") or ""), raw=event
            )
        if kind == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") != "function_call":
                return None
            index = state.slot(str(item.get("id") or ""))
            fragment = _tool_fragment(
                index,
                id=str(item.get("call_id") or item.get("id") or ""),
                name=str(item.get("name") or ""),
                arguments="",
            )
            return self._chunk(state, delta=StreamDelta(tool_calls=fragment), raw=event)
        if kind == "response.function_call_arguments.delta":
            index = state.slot(str(event.get("item_id") or ""))
            fragment = _tool_fragment(index, arguments=event.get("delta") or "")
            return self._chunk(state, delta=StreamDelta(tool_calls=fragment), raw=event)
        if kind in {"response.completed", "response.incomplete"}:
            response = event.get("response") or {}
            return self._chunk(
                state,
                delta=StreamDelta(),
                finish_reason=self._finish_reason(response),
                usage=_usage(response.get("usage")),
                raw=event,
            )
        if kind in {"response.failed", "error"}:
            error = (event.get("response") or event).get("error") or {}
            raise classify_http_error(
                status_code=500,
                body={"error": error},
                provider=self.name,
                model=model,
            )
        return None

    def _iter_sse(self, request: ChatRequest, quirks: HostQuirks) -> Iterator[StreamChunk]:
        payload = self._encode_request(request, stream=True, quirks=quirks)
        state = _StreamState(model=self._model_id(request.model))
        with self._client.stream("POST", self.endpoint_path, json=payload) as response:
            raise_for_stream_status(response, provider=self.name, model=request.model)
            for data in iter_sse_lines(response):
                if data == "[DONE]":
                    break
                chunk = self._chunk_from_event(json.loads(data), state, model=request.model)
                if chunk is not None:
                    yield chunk

    async def _aiter_sse(
        self, request: ChatRequest, quirks: HostQuirks
    ) -> AsyncIterator[StreamChunk]:
        payload = self._encode_request(request, stream=True, quirks=quirks)
        state = _StreamState(model=self._model_id(request.model))
        async with self._aclient.stream("POST", self.endpoint_path, json=payload) as response:
            await araise_for_stream_status(response, provider=self.name, model=request.model)
            async for data in aiter_sse_lines(response):
                if data == "[DONE]":
                    break
                chunk = self._chunk_from_event(json.loads(data), state, model=request.model)
                if chunk is not None:
                    yield chunk

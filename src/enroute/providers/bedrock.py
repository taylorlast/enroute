"""Amazon Bedrock adapter built on the Converse API.

Converse is the unified surface across Bedrock's model families, so one encoder
covers Anthropic, OpenAI, Meta, and the rest instead of a per-family adapter.

Two pieces are specific to Bedrock:

* **Auth.** A Bedrock API key is sent as a bearer token. When IAM credentials are
  supplied instead, requests are signed with SigV4, because enterprises usually
  mandate role-based credentials over long-lived keys.
* **Streaming framing.** ``ConverseStream`` replies in the binary
  ``vnd.amazon.eventstream`` protocol rather than SSE, so chunks are decoded from
  length-prefixed frames. Frames split across network reads are buffered until
  complete; treating a partial frame as a whole one silently truncates output.

Model ids are Bedrock's own (``us.anthropic.claude-sonnet-4-6``) and do not match
catalog ids, so a mapping is accepted rather than guessed at.

Examples:
    >>> from enroute.providers.bedrock import bedrock_endpoint
    >>> bedrock_endpoint("us-east-1")
    'https://bedrock-runtime.us-east-1.amazonaws.com'
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import struct
import time
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlsplit

import httpx

from enroute.catalog.models import normalize_region
from enroute.errors import ConfigurationError, EnrouteError
from enroute.providers.base import (
    ProviderConfig,
    araise_for_stream_status,
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

SERVICE = "bedrock"
# A frame larger than this means the length prefix was misread; refuse to buffer.
MAX_FRAME_BYTES = 8 * 1024 * 1024
PRELUDE_BYTES = 12
_HEADER_STRING_TYPE = 7
# Widths of the AWS header value types we skip past, keyed by type code.
_FIXED_HEADER_WIDTHS = {0: 0, 1: 0, 2: 1, 3: 2, 4: 4, 5: 8, 8: 8, 9: 16}
_VARIABLE_HEADER_TYPES = {6, _HEADER_STRING_TYPE}


def bedrock_endpoint(region: str) -> str:
    """Build the runtime endpoint for a region.

    Args:
        region: AWS region such as ``eu-central-1``.

    Returns:
        The Bedrock runtime base URL.

    Examples:
        >>> bedrock_endpoint("eu-central-1")
        'https://bedrock-runtime.eu-central-1.amazonaws.com'
    """
    return f"https://bedrock-runtime.{region}.amazonaws.com"


def parse_event_headers(raw: bytes) -> dict[str, str]:
    """Read an event stream frame's headers.

    Only string values are returned, since that covers ``:event-type`` and
    ``:content-type``. Other types are skipped at their declared widths so the
    parser stays aligned rather than misreading the rest of the block.

    Args:
        raw: The frame's header bytes.

    Returns:
        String-valued headers.

    Examples:
        >>> block = bytes([11]) + b":event-type" + bytes([7]) + struct.pack(">H", 5) + b"hello"
        >>> parse_event_headers(block)
        {':event-type': 'hello'}
    """
    headers: dict[str, str] = {}
    offset = 0
    size = len(raw)
    while offset < size:
        name_len = raw[offset]
        offset += 1
        if offset + name_len > size:
            break
        name = raw[offset : offset + name_len].decode("utf-8", "replace")
        offset += name_len
        if offset >= size:
            break
        value_type = raw[offset]
        offset += 1
        if value_type in _VARIABLE_HEADER_TYPES:
            if offset + 2 > size:
                break
            (value_len,) = struct.unpack_from(">H", raw, offset)
            offset += 2
            value = raw[offset : offset + value_len]
            offset += value_len
            if value_type == _HEADER_STRING_TYPE:
                headers[name] = value.decode("utf-8", "replace")
        else:
            offset += _FIXED_HEADER_WIDTHS.get(value_type, 0)
    return headers


class EventStreamDecoder:
    """Incremental decoder for ``vnd.amazon.eventstream`` frames.

    Network reads do not align with frame boundaries, so bytes are buffered until
    a frame's declared length has fully arrived.

    Examples:
        >>> decoder = EventStreamDecoder()
        >>> frame = encode_event_frame({":event-type": "messageStop"}, b"{}")
        >>> list(decoder.feed(frame[:5])), list(decoder.feed(frame[5:]))
        ([], [({':event-type': 'messageStop'}, b'{}')])
    """

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> Iterator[tuple[dict[str, str], bytes]]:
        """Add bytes and yield every frame that is now complete.

        Args:
            data: Newly received bytes.

        Yields:
            A (headers, payload) pair per complete frame.

        Raises:
            EnrouteError: If a frame declares an implausible length.
        """
        self._buffer.extend(data)
        while True:
            if len(self._buffer) < PRELUDE_BYTES:
                return
            total_len, headers_len = struct.unpack_from(">II", self._buffer, 0)
            if total_len < PRELUDE_BYTES + 4 or total_len > MAX_FRAME_BYTES:
                raise EnrouteError(
                    f"bedrock event stream frame declared {total_len} bytes",
                    provider=SERVICE,
                )
            if len(self._buffer) < total_len:
                return
            headers_start = PRELUDE_BYTES
            payload_start = headers_start + headers_len
            # The trailing 4 bytes are the message CRC, not payload.
            payload_end = total_len - 4
            headers = parse_event_headers(bytes(self._buffer[headers_start:payload_start]))
            payload = bytes(self._buffer[payload_start:payload_end])
            del self._buffer[:total_len]
            yield headers, payload


def encode_event_frame(headers: Mapping[str, str], payload: bytes) -> bytes:
    """Encode a frame the way Bedrock does. Used by tests and doctests.

    Args:
        headers: String-valued headers.
        payload: Frame payload.

    Returns:
        A complete frame, with zeroed CRCs since the decoder does not verify them.

    Examples:
        >>> len(encode_event_frame({":event-type": "x"}, b"{}")) > 12
        True
    """
    header_bytes = bytearray()
    for name, value in headers.items():
        encoded_name = name.encode("utf-8")
        encoded_value = value.encode("utf-8")
        header_bytes.append(len(encoded_name))
        header_bytes.extend(encoded_name)
        header_bytes.append(_HEADER_STRING_TYPE)
        header_bytes.extend(struct.pack(">H", len(encoded_value)))
        header_bytes.extend(encoded_value)
    total = PRELUDE_BYTES + len(header_bytes) + len(payload) + 4
    return (
        struct.pack(">II", total, len(header_bytes))
        + b"\x00\x00\x00\x00"
        + bytes(header_bytes)
        + payload
        + b"\x00\x00\x00\x00"
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hmac(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def sigv4_headers(
    *,
    method: str,
    url: str,
    region: str,
    payload: bytes,
    access_key_id: str,
    secret_access_key: str,
    session_token: str | None = None,
    now: dt.datetime | None = None,
) -> dict[str, str]:
    """Sign a request with AWS Signature Version 4.

    Args:
        method: HTTP method.
        url: Full request URL.
        region: AWS region.
        payload: Exact request body bytes.
        access_key_id: AWS access key id.
        secret_access_key: AWS secret access key.
        session_token: Session token for temporary credentials.
        now: Signing time, for reproducible tests.

    Returns:
        Headers carrying the signature.

    Examples:
        >>> headers = sigv4_headers(
        ...     method="POST",
        ...     url="https://bedrock-runtime.us-east-1.amazonaws.com/model/m/converse",
        ...     region="us-east-1",
        ...     payload=b"{}",
        ...     access_key_id="AKIDEXAMPLE",
        ...     secret_access_key="secret",
        ...     now=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
        ... )
        >>> headers["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260101")
        True
    """
    moment = now or dt.datetime.now(dt.timezone.utc)
    amz_date = moment.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = moment.strftime("%Y%m%d")
    parts = urlsplit(url)
    canonical_uri = quote(parts.path or "/", safe="/-_.~")
    payload_hash = _sha256(payload)

    signed_headers = "host;x-amz-content-sha256;x-amz-date"
    canonical_headers = (
        f"host:{parts.netloc}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n"
    )
    if session_token:
        canonical_headers += f"x-amz-security-token:{session_token}\n"
        signed_headers += ";x-amz-security-token"

    canonical_request = "\n".join(
        [
            method,
            canonical_uri,
            parts.query,
            canonical_headers,
            signed_headers,
            payload_hash,
        ]
    )
    scope = f"{date_stamp}/{region}/{SERVICE}/aws4_request"
    to_sign = "\n".join(
        ["AWS4-HMAC-SHA256", amz_date, scope, _sha256(canonical_request.encode("utf-8"))]
    )
    key = _hmac(f"AWS4{secret_access_key}".encode(), date_stamp)
    key = _hmac(key, region)
    key = _hmac(key, SERVICE)
    key = _hmac(key, "aws4_request")
    signature = hmac.new(key, to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    headers = {
        "x-amz-date": amz_date,
        "x-amz-content-sha256": payload_hash,
        "Authorization": (
            f"AWS4-HMAC-SHA256 Credential={access_key_id}/{scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        ),
    }
    if session_token:
        headers["x-amz-security-token"] = session_token
    return headers


_STOP_REASONS: dict[str, FinishReason] = {
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "max_tokens": FinishReason.LENGTH,
    "tool_use": FinishReason.TOOL_CALLS,
    "content_filtered": FinishReason.CONTENT_FILTER,
}


def _finish_reason(stop: Any) -> FinishReason | str | None:
    """Map a Converse stop reason onto the normalized enum.

    Args:
        stop: Bedrock's ``stopReason``, if present.

    Returns:
        A known finish reason, the raw string when unrecognized, or ``None``.

    Examples:
        >>> _finish_reason("max_tokens"), _finish_reason(None)
        (<FinishReason.LENGTH: 'length'>, None)
    """
    if stop is None:
        return None
    return _STOP_REASONS.get(str(stop), str(stop))


@dataclass
class _BedrockStreamState:
    """Cross-event state for one ConverseStream response.

    Converse indexes content blocks, and argument fragments only make sense
    relative to the block that opened them, so the mapping from block index to
    OpenAI tool-call position is kept for the life of the stream.
    """

    tool_slots: dict[int, int] = field(default_factory=dict)
    structured_blocks: dict[int, bool] = field(default_factory=dict)
    structured_tool: str | None = None

    def tool_slot(self, block_index: int) -> int:
        """Assign this block the next position in OpenAI's ``tool_calls`` array.

        Args:
            block_index: Converse content block index.

        Returns:
            The stable OpenAI array position for the block.
        """
        if block_index not in self.tool_slots:
            self.tool_slots[block_index] = len(self.tool_slots)
        return self.tool_slots[block_index]


class BedrockProvider:
    """Amazon Bedrock via the Converse and ConverseStream APIs.

    Args:
        api_key: Bedrock API key sent as a bearer token. Omit when using IAM.
        region: AWS region such as ``us-east-1``. Exposed as the coarse catalog
            region via ``region``, since pricing varies by continent not by zone.
        models: Maps catalog model ids to Bedrock model ids. Both the full
            ``author/slug`` and the bare slug are accepted as keys.
        access_key_id: AWS access key id, to sign with SigV4 instead.
        secret_access_key: AWS secret access key.
        session_token: Session token for temporary credentials.
        base_url: Overrides the regional endpoint.
        name: Provider slug.
        timeout_s: Request timeout in seconds.
        default_headers: Extra headers.

    Raises:
        ConfigurationError: If no usable credentials are supplied.

    Examples:
        >>> provider = BedrockProvider("key", region="us-east-1")
        >>> provider.name, provider.aws_region, provider.region
        ('bedrock', 'us-east-1', 'us')
        >>> provider.close()
    """

    name: str = "bedrock"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        region: str = "us-east-1",
        models: dict[str, str] | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        session_token: str | None = None,
        base_url: str | None = None,
        name: str | None = None,
        timeout_s: float | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.name = name or self.name
        self.aws_region = region
        self.region = normalize_region(region)
        self.models = models or {}
        self._signing = None
        if not api_key and not (access_key_id and secret_access_key):
            raise ConfigurationError(
                "bedrock requires an api key or access_key_id/secret_access_key",
                provider=self.name,
            )
        # An explicit key wins: instance credentials are usually ambient, so
        # signing with them would quietly ignore what the caller asked for.
        if not api_key and access_key_id and secret_access_key:
            self._signing = (access_key_id, secret_access_key, session_token)
        self.config = ProviderConfig(
            api_key=api_key or "",
            base_url=(base_url or bedrock_endpoint(self.aws_region)).rstrip("/"),
            timeout_s=timeout_s if timeout_s is not None else 60.0,
            default_headers=default_headers or {},
        )
        base_headers = {"Content-Type": "application/json", **self.config.default_headers}
        self._client = httpx.Client(
            base_url=self.config.base_url,
            headers=base_headers,
            timeout=self.config.timeout_s,
        )
        self._aclient = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers=base_headers,
            timeout=self.config.timeout_s,
        )

    def _model_id(self, model: str) -> str:
        if model in self.models:
            return self.models[model]
        bare = model.split("/", 1)[1] if model.count("/") == 1 else model
        return self.models.get(bare, bare)

    def _path(self, model: str, *, stream: bool) -> str:
        action = "converse-stream" if stream else "converse"
        return f"/model/{quote(self._model_id(model), safe='')}/{action}"

    def _auth_headers(self, path: str, payload: bytes) -> dict[str, str]:
        if self._signing is None:
            return {"Authorization": f"Bearer {self.config.api_key}"}
        access_key_id, secret_access_key, session_token = self._signing
        return sigv4_headers(
            method="POST",
            url=f"{self.config.base_url}{path}",
            region=self.aws_region,
            payload=payload,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
        )

    def _encode_request(self, request: ChatRequest) -> dict[str, Any]:
        system: list[dict[str, Any]] = []
        messages: list[dict[str, Any]] = []
        for msg in request.messages:
            if msg.role == "system":
                system.append({"text": text_content(msg)})
                continue
            if msg.role == "tool":
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": msg.tool_call_id,
                                    "content": [{"text": text_content(msg)}],
                                }
                            }
                        ],
                    }
                )
                continue
            if msg.role == "assistant" and (msg.tool_calls or msg.reasoning):
                content: list[dict[str, Any]] = []
                if msg.reasoning and msg.reasoning_signature:
                    content.append(
                        {
                            "reasoningContent": {
                                "reasoningText": {
                                    "text": msg.reasoning,
                                    "signature": msg.reasoning_signature,
                                }
                            }
                        }
                    )
                text = text_content(msg)
                if text:
                    content.append({"text": text})
                for call in msg.tool_calls or []:
                    try:
                        arguments = json.loads(call.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": call.function.arguments}
                    content.append(
                        {
                            "toolUse": {
                                "toolUseId": call.id,
                                "name": call.function.name,
                                "input": arguments,
                            }
                        }
                    )
                messages.append({"role": "assistant", "content": content})
                continue
            messages.append({"role": msg.role, "content": [{"text": text_content(msg)}]})

        payload: dict[str, Any] = {"messages": messages}
        if wants_json_object(request):
            system.append({"text": JSON_ONLY_INSTRUCTION})
        if system:
            payload["system"] = system
        inference: dict[str, Any] = {}
        if request.max_tokens is not None:
            inference["maxTokens"] = request.max_tokens
        if request.temperature is not None:
            inference["temperature"] = request.temperature
        if request.top_p is not None:
            inference["topP"] = request.top_p
        if request.stop is not None:
            inference["stopSequences"] = (
                [request.stop] if isinstance(request.stop, str) else list(request.stop)
            )
        if inference:
            payload["inferenceConfig"] = inference
        if request.tools:
            payload["toolConfig"] = {
                "tools": [
                    {
                        "toolSpec": {
                            "name": tool.function.name,
                            "description": tool.function.description or "",
                            "inputSchema": {
                                "json": tool.function.parameters
                                or {"type": "object", "properties": {}}
                            },
                        }
                    }
                    for tool in request.tools
                ]
            }
            if request.tool_choice is not None:
                choice = self._tool_choice(request.tool_choice)
                if choice is not None:
                    payload["toolConfig"]["toolChoice"] = choice
        elif (tool_name := schema_tool_name(request)) is not None:
            payload["toolConfig"] = {
                "tools": [
                    {
                        "toolSpec": {
                            "name": tool_name,
                            "description": STRUCTURED_TOOL_DESCRIPTION,
                            "inputSchema": {"json": structured_schema(request)},
                        }
                    }
                ],
                "toolChoice": {"tool": {"name": tool_name}},
            }
        payload.update(request.extra)
        return payload

    def _tool_choice(self, choice: str | dict[str, Any]) -> dict[str, Any] | None:
        """Translate an OpenAI ``tool_choice`` into Converse's shape.

        Args:
            choice: OpenAI-style tool choice, either a keyword or a forced tool.

        Returns:
            A Converse ``toolChoice`` object, or ``None`` when Converse has no
            equivalent and the default behaviour should stand.
        """
        if isinstance(choice, str):
            if choice == "required":
                return {"any": {}}
            if choice == "auto":
                return {"auto": {}}
            return None
        name = (choice.get("function") or {}).get("name")
        return {"tool": {"name": name}} if name else None

    def _usage(self, raw: Mapping[str, Any] | None) -> Usage:
        data = raw or {}
        return Usage.from_counts(
            int(data.get("inputTokens") or 0), int(data.get("outputTokens") or 0)
        )

    def _parse_response(
        self,
        data: dict[str, Any],
        *,
        model: str,
        latency_ms: float,
        structured_tool: str | None = None,
    ) -> ChatResponse:
        message_raw = (data.get("output") or {}).get("message") or {}
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        signature: str | None = None
        tool_calls: list[ToolCall] = []
        for block in message_raw.get("content") or []:
            if "text" in block:
                text_parts.append(block["text"] or "")
            elif "reasoningContent" in block:
                reasoning_text = (block["reasoningContent"] or {}).get("reasoningText") or {}
                reasoning_parts.append(reasoning_text.get("text") or "")
                signature = reasoning_text.get("signature") or signature
            elif "toolUse" in block:
                use = block["toolUse"]
                if use.get("name") == structured_tool:
                    text_parts.append(json.dumps(use.get("input") or {}))
                    continue
                tool_calls.append(
                    ToolCall(
                        id=str(use.get("toolUseId") or ""),
                        function=FunctionCall(
                            name=str(use.get("name") or ""),
                            arguments=json.dumps(use.get("input") or {}),
                        ),
                    )
                )
        stop = data.get("stopReason")
        finish = _finish_reason(stop)
        if stop == "tool_use" and structured_tool:
            finish = FinishReason.STOP
        message = Message(
            role="assistant",
            content="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls or None,
            reasoning="".join(reasoning_parts) or None,
            reasoning_signature=signature,
        )
        return ChatResponse(
            id=str(data.get("id") or ""),
            model=model,
            choices=[Choice(message=message, finish_reason=finish)],
            usage=self._usage(data.get("usage")),
            provider=self.name,
            region=self.region,
            raw=data,
            latency_ms=latency_ms,
        )

    def _chunk(self, delta: StreamDelta, body: dict[str, Any], *, model: str) -> StreamChunk:
        """Wrap a delta in a chunk tagged with this provider's host and region.

        Args:
            delta: The normalized delta to emit.
            body: Native Converse event, attached for debugging.
            model: Upstream model id.

        Returns:
            A normalized stream chunk.
        """
        return StreamChunk(
            id="",
            model=model,
            delta=delta,
            provider=self.name,
            region=self.region,
            raw=body,
        )

    def _stream_delta(
        self, body: dict[str, Any], state: _BedrockStreamState, *, model: str
    ) -> StreamChunk | None:
        """Handle ``contentBlockDelta`` for text, reasoning, and tool arguments.

        Args:
            body: Parsed Converse event body.
            state: Mutable stream state.
            model: Upstream model id.

        Returns:
            A normalized chunk, or ``None`` for a delta with no payload.
        """
        index = int(body.get("contentBlockIndex") or 0)
        delta = body.get("delta") or {}
        text = delta.get("text")
        if text is not None:
            return self._chunk(StreamDelta(content=text), body, model=model)
        reasoning = delta.get("reasoningContent") or {}
        if reasoning:
            if reasoning.get("text") is not None:
                return self._chunk(StreamDelta(reasoning=reasoning["text"]), body, model=model)
            if reasoning.get("signature") is not None:
                return self._chunk(
                    StreamDelta(reasoning_signature=reasoning["signature"]), body, model=model
                )
            return None
        tool_use = delta.get("toolUse") or {}
        fragment = tool_use.get("input")
        if fragment is None:
            return None
        if state.structured_blocks.get(index):
            return self._chunk(StreamDelta(content=fragment), body, model=model)
        return self._chunk(
            StreamDelta(
                tool_calls=[
                    {
                        "index": state.tool_slot(index),
                        "function": {"arguments": fragment},
                    }
                ]
            ),
            body,
            model=model,
        )

    def _chunk_from_event(
        self,
        event_type: str,
        payload: bytes,
        *,
        model: str,
        state: _BedrockStreamState,
    ) -> StreamChunk | None:
        """Translate one Converse stream event into a normalized chunk.

        Args:
            event_type: Value of the frame's ``:event-type`` header.
            payload: Raw JSON body of the frame.
            model: Upstream model id.
            state: Mutable stream state, updated in place.

        Returns:
            A normalized chunk, or ``None`` for events with no client-visible
            delta.
        """
        try:
            body = json.loads(payload) if payload else {}
        except json.JSONDecodeError:
            return None
        if event_type == "contentBlockStart":
            index = int(body.get("contentBlockIndex") or 0)
            tool_use = (body.get("start") or {}).get("toolUse") or {}
            if not tool_use:
                return None
            if tool_use.get("name") == state.structured_tool:
                state.structured_blocks[index] = True
                return None
            return self._chunk(
                StreamDelta(
                    tool_calls=[
                        {
                            "index": state.tool_slot(index),
                            "id": tool_use.get("toolUseId") or "",
                            "type": "function",
                            "function": {"name": tool_use.get("name") or "", "arguments": ""},
                        }
                    ]
                ),
                body,
                model=model,
            )
        if event_type == "contentBlockDelta":
            return self._stream_delta(body, state, model=model)
        if event_type == "messageStop":
            stop = body.get("stopReason")
            finish = _finish_reason(stop)
            if stop == "tool_use" and state.structured_tool:
                finish = FinishReason.STOP
            return StreamChunk(
                id="",
                model=model,
                finish_reason=finish,
                provider=self.name,
                region=self.region,
                raw=body,
            )
        if event_type == "metadata":
            # Usage arrives only here, and billing depends on it.
            return StreamChunk(
                id="",
                model=model,
                usage=self._usage(body.get("usage")),
                provider=self.name,
                region=self.region,
                raw=body,
            )
        return None

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Execute a non-streaming Converse call.

        Args:
            request: Normalized chat request.

        Returns:
            Normalized chat response.
        """
        path = self._path(request.model, stream=False)
        body = json.dumps(self._encode_request(request)).encode("utf-8")
        started = time.perf_counter()
        try:
            response = self._client.post(path, content=body, headers=self._auth_headers(path, body))
        except Exception as exc:  # noqa: BLE001
            raise map_transport_error(exc, provider=self.name, model=request.model) from exc
        raise_for_status(response, provider=self.name, model=request.model)
        return self._parse_response(
            response.json(),
            model=request.model,
            latency_ms=(time.perf_counter() - started) * 1000,
            structured_tool=schema_tool_name(request),
        )

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        """Execute a streaming Converse call.

        Args:
            request: Normalized chat request.

        Yields:
            Normalized stream chunks.
        """
        path = self._path(request.model, stream=True)
        body = json.dumps(self._encode_request(request)).encode("utf-8")
        decoder = EventStreamDecoder()
        state = _BedrockStreamState(structured_tool=schema_tool_name(request))
        try:
            with self._client.stream(
                "POST", path, content=body, headers=self._auth_headers(path, body)
            ) as response:
                raise_for_stream_status(response, provider=self.name, model=request.model)
                for data in response.iter_bytes():
                    for headers, payload in decoder.feed(data):
                        chunk = self._chunk_from_event(
                            headers.get(":event-type", ""),
                            payload,
                            model=request.model,
                            state=state,
                        )
                        if chunk is not None:
                            yield chunk
        except EnrouteError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise map_transport_error(exc, provider=self.name, model=request.model) from exc

    async def achat(self, request: ChatRequest) -> ChatResponse:
        """Async non-streaming Converse call.

        Args:
            request: Normalized chat request.

        Returns:
            Normalized chat response.
        """
        path = self._path(request.model, stream=False)
        body = json.dumps(self._encode_request(request)).encode("utf-8")
        started = time.perf_counter()
        try:
            response = await self._aclient.post(
                path, content=body, headers=self._auth_headers(path, body)
            )
        except Exception as exc:  # noqa: BLE001
            raise map_transport_error(exc, provider=self.name, model=request.model) from exc
        raise_for_status(response, provider=self.name, model=request.model)
        return self._parse_response(
            response.json(),
            model=request.model,
            latency_ms=(time.perf_counter() - started) * 1000,
            structured_tool=schema_tool_name(request),
        )

    async def astream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Async streaming Converse call.

        Args:
            request: Normalized chat request.

        Yields:
            Normalized stream chunks.
        """
        path = self._path(request.model, stream=True)
        body = json.dumps(self._encode_request(request)).encode("utf-8")
        decoder = EventStreamDecoder()
        state = _BedrockStreamState(structured_tool=schema_tool_name(request))
        try:
            async with self._aclient.stream(
                "POST", path, content=body, headers=self._auth_headers(path, body)
            ) as response:
                await araise_for_stream_status(response, provider=self.name, model=request.model)
                async for data in response.aiter_bytes():
                    for headers, payload in decoder.feed(data):
                        chunk = self._chunk_from_event(
                            headers.get(":event-type", ""),
                            payload,
                            model=request.model,
                            state=state,
                        )
                        if chunk is not None:
                            yield chunk
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

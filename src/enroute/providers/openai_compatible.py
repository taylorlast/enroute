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
from dataclasses import dataclass
from typing import Any, Literal

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


@dataclass
class HostQuirks:
    """Parameters this host or model has already rejected.

    "OpenAI-compatible" is a family, not a spec: clones 400 on
    ``stream_options``, newer OpenAI models 400 on ``max_tokens`` in favour of
    ``max_completion_tokens``, and reasoning models 400 on ``temperature`` and
    ``top_p`` outright. None of it is discoverable up front, so the adapter
    retries once and remembers, which keeps the cost to one extra round trip per
    model rather than one per request.

    Attributes:
        stream_options: Whether the host accepts ``stream_options``.
        max_tokens_key: Which output-limit parameter name the model accepts.
        sampling: Whether the model accepts ``temperature`` and ``top_p``.
    """

    stream_options: bool = True
    max_tokens_key: str = "max_tokens"
    sampling: bool = True


def _reasoning_text(raw: dict[str, Any]) -> str | None:
    """Read reasoning text from whichever key the host used.

    OpenRouter emits ``reasoning``; DeepSeek and vLLM emit ``reasoning_content``.

    Args:
        raw: A message or delta object from the host.

    Returns:
        The reasoning text, or ``None`` when the host sent none.

    Examples:
        >>> _reasoning_text({"reasoning_content": "hmm"})
        'hmm'
    """
    value = raw.get("reasoning")
    if value is None:
        value = raw.get("reasoning_content")
    return value if isinstance(value, str) and value else None


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
    endpoint_path: str = "/chat/completions"

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
        self._quirks: dict[str, HostQuirks] = {}
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

    def _quirks_for(self, request: ChatRequest) -> HostQuirks:
        """Parameter spellings known to work for this request's model.

        Args:
            request: Normalized chat request.

        Returns:
            The remembered quirks, created on first use.
        """
        return self._quirks.setdefault(self._model_id(request.model), HostQuirks())

    def _encode_request(
        self, request: ChatRequest, *, stream: bool, quirks: HostQuirks | None = None
    ) -> dict[str, Any]:
        quirks = quirks or HostQuirks()
        payload: dict[str, Any] = {
            "model": self._model_id(request.model),
            # Reasoning is read back from these hosts but never sent to them:
            # OpenAI rejects unknown message keys, and no host accepts its own
            # reasoning output replayed as input.
            "messages": [
                m.model_dump(exclude_none=True, exclude={"reasoning", "reasoning_signature"})
                for m in request.messages
            ],
            "stream": stream,
        }
        skip = () if quirks.sampling else ("temperature", "top_p")
        for key in ("temperature", "top_p", "stop", "seed", "user", "tool_choice"):
            value = getattr(request, key)
            if value is not None and key not in skip:
                payload[key] = value
        if request.max_tokens is not None:
            payload[quirks.max_tokens_key] = request.max_tokens
        if request.tools:
            payload["tools"] = [t.model_dump(exclude_none=True) for t in request.tools]
        if request.response_format is not None:
            payload["response_format"] = request.response_format.model_dump(exclude_none=True)
        if stream and quirks.stream_options:
            payload["stream_options"] = {"include_usage": True}
        payload.update(request.extra)
        return payload

    def _adapt(self, exc: EnrouteError, quirks: HostQuirks) -> bool:
        """Record a rejected parameter spelling and report whether to retry.

        Args:
            exc: Classified error from the previous attempt.
            quirks: Quirks to update in place.

        Returns:
            ``True`` when the request is worth retrying with the adjustment.
        """
        if exc.status_code != 400:
            return False
        blob = f"{exc.message} {exc.body}".lower()
        if quirks.stream_options and "stream_options" in blob:
            # The stream works without it; usage just arrives later or not at all.
            quirks.stream_options = False
            return True
        if quirks.max_tokens_key == "max_tokens" and "max_completion_tokens" in blob:
            quirks.max_tokens_key = "max_completion_tokens"
            return True
        if quirks.sampling and ("'temperature'" in blob or "'top_p'" in blob):
            # Reasoning models pick their own sampling and reject both outright.
            quirks.sampling = False
            return True
        return False

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
            reasoning=_reasoning_text(raw),
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
                reasoning=_reasoning_text(delta_raw),
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
        quirks = self._quirks_for(request)
        while True:
            payload = self._encode_request(request, stream=False, quirks=quirks)
            started = time.perf_counter()
            try:
                response = self._client.post(self.endpoint_path, json=payload)
            except Exception as exc:  # noqa: BLE001
                raise map_transport_error(exc, provider=self.name, model=request.model) from exc
            try:
                raise_for_status(response, provider=self.name, model=request.model)
            except EnrouteError as exc:
                if self._adapt(exc, quirks):
                    continue
                raise
            latency_ms = (time.perf_counter() - started) * 1000
            return self._parse_response(response.json(), latency_ms=latency_ms)

    def _iter_sse(self, request: ChatRequest, quirks: HostQuirks) -> Iterator[StreamChunk]:
        """Read one OpenAI-compatible SSE response into stream chunks.

        Args:
            request: Normalized chat request.
            quirks: Parameter spellings to use for this attempt.

        Yields:
            Normalized stream chunks from the host.
        """
        payload = self._encode_request(request, stream=True, quirks=quirks)
        with self._client.stream("POST", self.endpoint_path, json=payload) as response:
            raise_for_stream_status(response, provider=self.name, model=request.model)
            for data in iter_sse_lines(response):
                if data == "[DONE]":
                    break
                yield self._parse_stream_chunk(json.loads(data))

    async def _aiter_sse(
        self, request: ChatRequest, quirks: HostQuirks
    ) -> AsyncIterator[StreamChunk]:
        """Async variant of :meth:`_iter_sse`.

        Args:
            request: Normalized chat request.
            quirks: Parameter spellings to use for this attempt.

        Yields:
            Normalized stream chunks from the host.
        """
        payload = self._encode_request(request, stream=True, quirks=quirks)
        async with self._aclient.stream("POST", self.endpoint_path, json=payload) as response:
            await araise_for_stream_status(response, provider=self.name, model=request.model)
            async for data in aiter_sse_lines(response):
                if data == "[DONE]":
                    break
                yield self._parse_stream_chunk(json.loads(data))

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        """Execute a streaming chat completion.

        Args:
            request: Normalized chat request.

        Yields:
            Normalized stream chunks.
        """
        quirks = self._quirks_for(request)
        while True:
            started_output = False
            try:
                for chunk in self._iter_sse(request, quirks):
                    started_output = True
                    yield chunk
                return
            except EnrouteError as exc:
                # Retrying after the first chunk would replay text to the caller.
                if not started_output and self._adapt(exc, quirks):
                    continue
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
        quirks = self._quirks_for(request)
        while True:
            payload = self._encode_request(request, stream=False, quirks=quirks)
            started = time.perf_counter()
            try:
                response = await self._aclient.post(self.endpoint_path, json=payload)
            except Exception as exc:  # noqa: BLE001
                raise map_transport_error(exc, provider=self.name, model=request.model) from exc
            try:
                raise_for_status(response, provider=self.name, model=request.model)
            except EnrouteError as exc:
                if self._adapt(exc, quirks):
                    continue
                raise
            latency_ms = (time.perf_counter() - started) * 1000
            return self._parse_response(response.json(), latency_ms=latency_ms)

    async def astream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Async streaming chat completion.

        Args:
            request: Normalized chat request.

        Yields:
            Normalized stream chunks.
        """
        quirks = self._quirks_for(request)
        while True:
            started_output = False
            try:
                async for chunk in self._aiter_sse(request, quirks):
                    started_output = True
                    yield chunk
                return
            except EnrouteError as exc:
                if not started_output and self._adapt(exc, quirks):
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


class OpenAIProvider(OpenAICompatible):
    """Official OpenAI API, over whichever of its two endpoints fits the request.

    OpenAI serves the same models through two incompatible wire formats, and
    ``/responses`` is now the larger one: ``/chat/completions`` rejects function
    tools for every current model and never reports reasoning. So requests go to
    :class:`~enroute.providers.openai_responses.OpenAIResponsesProvider` by
    default. The one thing it cannot do is ``stop`` and ``seed``, which it
    rejects as unknown, so a request using either and needing nothing
    Responses-only stays on chat completions instead. Every request therefore
    lands on the endpoint that can serve all of it.

    Args:
        api_key: OpenAI API key.
        transport: ``"auto"`` picks per request as described above. ``"chat"``
            and ``"responses"`` pin one endpoint, which is what to reach for when
            fronting a proxy that speaks only one of them.
        **kwargs: Forwarded to :class:`OpenAICompatible`.

    Examples:
        >>> from enroute.types import Message
        >>> provider = OpenAIProvider("sk-test")
        >>> hello = [Message(role="user", content="hi")]
        >>> provider._delegate(ChatRequest(model="openai/gpt-5.6", messages=hello)) is not None
        True
    """

    name = "openai"
    default_base_url = "https://api.openai.com/v1"

    def __init__(
        self,
        api_key: str,
        *,
        transport: Literal["auto", "chat", "responses"] = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init__(api_key, **kwargs)
        self._transport = transport
        self._responses_args = (api_key, kwargs)
        self._responses: OpenAICompatible | None = None

    def _delegate(self, request: ChatRequest) -> OpenAICompatible | None:
        """The Responses adapter, unless this request is better served elsewhere.

        Args:
            request: Normalized chat request.

        Returns:
            A Responses adapter, or ``None`` to stay on chat completions.
        """
        if self._transport == "chat":
            return None
        # Deferred: the Responses adapter subclasses this module.
        from enroute.providers.openai_responses import OpenAIResponsesProvider

        # Tools force the issue, since chat completions refuses them outright.
        # Absent tools, prefer the endpoint that honours the request exactly.
        if (
            self._transport == "auto"
            and not request.tools
            and not OpenAIResponsesProvider.serves_exactly(request)
        ):
            return None
        if self._responses is None:
            api_key, kwargs = self._responses_args
            self._responses = OpenAIResponsesProvider(api_key, **{**kwargs, "name": self.name})
        return self._responses

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Execute a non-streaming chat completion on the fitting endpoint.

        Args:
            request: Normalized chat request.

        Returns:
            Normalized chat response.
        """
        target = self._delegate(request)
        return target.chat(request) if target else super().chat(request)

    async def achat(self, request: ChatRequest) -> ChatResponse:
        """Async variant of :meth:`chat`.

        Args:
            request: Normalized chat request.

        Returns:
            Normalized chat response.
        """
        target = self._delegate(request)
        return await (target.achat(request) if target else super().achat(request))

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        """Stream a chat completion from the fitting endpoint.

        Args:
            request: Normalized chat request.

        Yields:
            Normalized stream chunks.
        """
        target = self._delegate(request)
        yield from (target.stream(request) if target else super().stream(request))

    async def astream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Async variant of :meth:`stream`.

        Args:
            request: Normalized chat request.

        Yields:
            Normalized stream chunks.
        """
        target = self._delegate(request)
        source = target.astream(request) if target else super().astream(request)
        async for chunk in source:
            yield chunk

    def close(self) -> None:
        """Close both endpoints' HTTP clients."""
        super().close()
        if self._responses is not None:
            self._responses.close()

    async def aclose(self) -> None:
        """Close both endpoints' async HTTP clients."""
        await super().aclose()
        if self._responses is not None:
            await self._responses.aclose()


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

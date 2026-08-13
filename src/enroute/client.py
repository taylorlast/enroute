"""High-level enroute client.

``Enroute(api_key=...)`` talks to the hosted gateway. ``Enroute(providers={...})``
calls providers directly with your own keys. Same request types and same traces.

Examples:
    >>> from enroute.client import Enroute
    >>> Enroute.__name__
    'Enroute'
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from enroute.catalog.models import ModelCatalog
from enroute.config import DEFAULT_SETTINGS
from enroute.errors import ConfigurationError
from enroute.providers.anthropic import AnthropicProvider
from enroute.providers.base import Provider
from enroute.providers.google import GoogleProvider
from enroute.providers.openai_compatible import (
    DeepSeekProvider,
    FireworksProvider,
    GroqProvider,
    MistralProvider,
    OpenAICompatible,
    OpenAIProvider,
    TogetherProvider,
    XAIProvider,
)
from enroute.routing.policies import RoutingPolicy
from enroute.routing.router import AttemptRecord, Router
from enroute.tracing.redaction import Redactor, Sampler
from enroute.tracing.schema import Attempt, Trace, new_trace_id
from enroute.tracing.sinks import JSONLSink, Sink
from enroute.tracing.writer import TraceWriter
from enroute.types import ChatRequest, ChatResponse, Message, StreamChunk, Tool

_PROVIDER_CTORS: dict[str, type] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "groq": GroqProvider,
    "together": TogetherProvider,
    "fireworks": FireworksProvider,
    "xai": XAIProvider,
    "deepseek": DeepSeekProvider,
    "mistral": MistralProvider,
}


class Enroute:
    """Unified client for routing, tracing, and (via other modules) environments.

    Args:
        api_key: Hosted enroute gateway API key. Mutually exclusive with building
            providers solely from ``providers`` / env vars when you want direct mode.
        providers: Mapping of provider slug to API key string or Provider instance.
        base_url: Gateway base URL when using ``api_key``.
        catalog: Optional model catalog.
        policy: Optional routing policy.
        sink: Trace sink. Defaults to ``.enroute/traces.jsonl``.
        redactor: Optional redactor applied before persistence.
        sampler: Optional sampler.
        capture_content: Whether to keep full prompt/response content in traces.
        max_retries: Retries per route.
        max_cost_usd: Optional per-request budget.
        default_headers: Extra headers for the gateway client.
        tags: Default tags applied to every trace.

    Examples:
        Hosted gateway (reads ``ENROUTE_API_KEY``)::

            client = Enroute()

        Bring-your-own upstream keys::

            client = Enroute(providers={"openai": "sk-..."})
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        providers: Mapping[str, str | Provider] | None = None,
        base_url: str | None = None,
        catalog: ModelCatalog | None = None,
        policy: RoutingPolicy | None = None,
        sink: Sink | None = None,
        redactor: Redactor | None = None,
        sampler: Sampler | None = None,
        capture_content: bool | None = None,
        max_retries: int = 2,
        max_cost_usd: float | None = None,
        default_headers: dict[str, str] | None = None,
        tags: dict[str, str] | None = None,
        trace_dir: str | Path | None = None,
    ) -> None:
        self.catalog = catalog or ModelCatalog()
        self.tags = tags or {}
        self.capture_content = (
            DEFAULT_SETTINGS.capture_content if capture_content is None else capture_content
        )
        if api_key is None and providers is None:
            api_key = os.environ.get("ENROUTE_API_KEY")
        provider_map = self._build_providers(
            api_key=api_key,
            providers=providers,
            base_url=base_url,
            default_headers=default_headers,
        )
        if not provider_map:
            raise ConfigurationError(
                "no providers configured; pass api_key=... / ENROUTE_API_KEY or providers={...}"
            )
        self._providers = provider_map
        self.router = Router(
            provider_map,
            catalog=self.catalog,
            policy=policy,
            max_retries=max_retries,
            max_cost_usd=max_cost_usd,
        )
        if sink is None:
            directory = Path(trace_dir) if trace_dir else DEFAULT_SETTINGS.trace_dir
            sink = JSONLSink(directory / "traces.jsonl")
        if redactor is None and not self.capture_content:
            redactor = Redactor(drop_content=True)
        self.writer = TraceWriter(sink, redactor=redactor, sampler=sampler)

    def _build_providers(
        self,
        *,
        api_key: str | None,
        providers: Mapping[str, str | Provider] | None,
        base_url: str | None,
        default_headers: dict[str, str] | None,
    ) -> dict[str, Provider]:
        result: dict[str, Provider] = {}
        if api_key:
            result["enroute"] = OpenAICompatible(
                api_key=api_key,
                base_url=base_url or DEFAULT_SETTINGS.gateway_base_url,
                name="enroute",
                default_headers=default_headers,
                strip_model_prefix=False,
            )
        if providers:
            for name, value in providers.items():
                if isinstance(value, str):
                    ctor = _PROVIDER_CTORS.get(name)
                    if ctor is None:
                        result[name] = OpenAICompatible(
                            api_key=value,
                            base_url=base_url or DEFAULT_SETTINGS.gateway_base_url,
                            name=name,
                        )
                    else:
                        result[name] = ctor(value)
                else:
                    result[name] = value
        # Env-var convenience for common providers.
        env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "groq": "GROQ_API_KEY",
            "together": "TOGETHER_API_KEY",
            "fireworks": "FIREWORKS_API_KEY",
            "xai": "XAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "mistral": "MISTRAL_API_KEY",
        }
        if not providers and not api_key:
            for slug, env_name in env_map.items():
                key = os.environ.get(env_name)
                if key:
                    result[slug] = _PROVIDER_CTORS[slug](key)
        return result

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Message | dict[str, Any]],
        models: list[str] | None = None,
        tools: Sequence[Tool | dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Create a chat completion.

        Args:
            model: Primary model id (``author/slug``).
            messages: Conversation messages.
            models: Optional fallback model chain.
            tools: Optional tools.
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.
            metadata: Request metadata stored on the trace.
            tags: Extra trace tags.
            **kwargs: Additional :class:`~enroute.types.ChatRequest` fields.

        Returns:
            Normalized :class:`~enroute.types.ChatResponse`.
        """
        request = self._make_request(
            model=model,
            messages=messages,
            models=models,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=metadata,
            **kwargs,
        )
        trace = Trace(tags={**self.tags, **(tags or {})}, metadata=dict(request.metadata))
        try:
            response, attempts = self.router.chat(request)
            self._record_success(trace, request, response, attempts)
            response.raw = {**(response.raw or {}), "enroute_trace_id": trace.trace_id}
            return response
        except Exception as exc:
            trace.add_llm(request=request, response=None, error=str(exc))
            self.writer.record(trace)
            raise

    async def achat(
        self,
        *,
        model: str,
        messages: Sequence[Message | dict[str, Any]],
        models: list[str] | None = None,
        tools: Sequence[Tool | dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Async chat completion.

        Args:
            model: Primary model id.
            messages: Conversation messages.
            models: Optional fallback chain.
            tools: Optional tools.
            temperature: Sampling temperature.
            max_tokens: Max tokens.
            metadata: Request metadata.
            tags: Extra trace tags.
            **kwargs: Additional request fields.

        Returns:
            Normalized chat response.
        """
        request = self._make_request(
            model=model,
            messages=messages,
            models=models,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=metadata,
            **kwargs,
        )
        trace = Trace(tags={**self.tags, **(tags or {})}, metadata=dict(request.metadata))
        try:
            response, attempts = await self.router.achat(request)
            self._record_success(trace, request, response, attempts)
            response.raw = {**(response.raw or {}), "enroute_trace_id": trace.trace_id}
            return response
        except Exception as exc:
            trace.add_llm(request=request, response=None, error=str(exc))
            self.writer.record(trace)
            raise

    def stream(
        self,
        *,
        model: str,
        messages: Sequence[Message | dict[str, Any]],
        models: list[str] | None = None,
        tags: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """Stream a chat completion and record a trace on completion.

        Args:
            model: Primary model id.
            messages: Conversation messages.
            models: Optional fallback chain.
            tags: Extra trace tags.
            **kwargs: Additional request fields.

        Yields:
            Stream chunks.
        """
        request = self._make_request(
            model=model, messages=messages, models=models, stream=True, **kwargs
        )
        trace = Trace(tags={**self.tags, **(tags or {})}, metadata=dict(request.metadata))

        def on_complete(response: ChatResponse, attempts: list[AttemptRecord]) -> None:
            self._record_success(trace, request, response, attempts)

        yield from self.router.stream(request, on_complete=on_complete)

    async def astream(
        self,
        *,
        model: str,
        messages: Sequence[Message | dict[str, Any]],
        models: list[str] | None = None,
        tags: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Async streaming chat completion.

        Args:
            model: Primary model id.
            messages: Conversation messages.
            models: Optional fallback chain.
            tags: Extra trace tags.
            **kwargs: Additional request fields.

        Yields:
            Stream chunks.
        """
        request = self._make_request(
            model=model, messages=messages, models=models, stream=True, **kwargs
        )
        trace = Trace(tags={**self.tags, **(tags or {})}, metadata=dict(request.metadata))

        def on_complete(response: ChatResponse, attempts: list[AttemptRecord]) -> None:
            self._record_success(trace, request, response, attempts)

        async for chunk in self.router.astream(request, on_complete=on_complete):
            yield chunk

    def label(
        self,
        trace_id: str,
        *,
        scores: dict[str, float] | None = None,
        reward: float | None = None,
        labels: dict[str, Any] | None = None,
        feedback: str | None = None,
    ) -> None:
        """Attach a late outcome label to a trace.

        Args:
            trace_id: Trace id returned via ``response.raw['enroute_trace_id']``.
            scores: Named scores.
            reward: Scalar reward.
            labels: Discrete labels.
            feedback: Free-form feedback.
        """
        self.writer.label(
            trace_id,
            scores=scores,
            reward=reward,
            labels=labels,
            feedback=feedback,
        )

    def flush(self) -> None:
        """Flush pending traces."""
        self.writer.flush()

    def close(self) -> None:
        """Flush traces and close providers."""
        self.writer.close()
        for provider in self._providers.values():
            provider.close()

    async def aclose(self) -> None:
        """Async close."""
        self.writer.close()
        for provider in self._providers.values():
            await provider.aclose()

    def __enter__(self) -> Enroute:
        """Enter context manager.

        Returns:
            This client instance.
        """
        return self

    def __exit__(self, *args: object) -> None:
        """Exit context manager and close resources."""
        self.close()

    def _make_request(
        self,
        *,
        model: str,
        messages: Sequence[Message | dict[str, Any]],
        models: list[str] | None = None,
        tools: Sequence[Tool | dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatRequest:
        parsed_messages = [
            m if isinstance(m, Message) else Message.model_validate(m) for m in messages
        ]
        parsed_tools = None
        if tools is not None:
            parsed_tools = [t if isinstance(t, Tool) else Tool.model_validate(t) for t in tools]
        return ChatRequest(
            model=model,
            messages=parsed_messages,
            models=models,
            tools=parsed_tools,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=metadata or {},
            stream=stream,
            **kwargs,
        )

    def _record_success(
        self,
        trace: Trace,
        request: ChatRequest,
        response: ChatResponse,
        attempts: list[AttemptRecord],
    ) -> None:
        req_for_trace: ChatRequest | dict[str, Any]
        resp_for_trace: ChatResponse | None
        if self.capture_content:
            req_for_trace = request
            resp_for_trace = response
        else:
            req_for_trace = request.model_dump(exclude={"messages"})
            resp_for_trace = response.model_copy(
                update={
                    "choices": [
                        c.model_copy(
                            update={"message": c.message.model_copy(update={"content": None})}
                        )
                        for c in response.choices
                    ]
                }
            )
        trace.add_llm(
            request=req_for_trace,
            response=resp_for_trace,
            attempts=[Attempt(**a.to_dict()) for a in attempts],
        )
        self.writer.record(trace)


def start_trace(**kwargs: Any) -> Trace:
    """Create an empty trace for manual / environment use.

    Args:
        **kwargs: Fields forwarded to :class:`~enroute.tracing.schema.Trace`.

    Returns:
        A new :class:`~enroute.tracing.schema.Trace`.
    """
    if "trace_id" not in kwargs:
        kwargs["trace_id"] = new_trace_id()
    return Trace(**kwargs)

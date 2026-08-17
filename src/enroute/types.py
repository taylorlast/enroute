"""Normalized, provider-agnostic types for chat completions.

Every provider adapter translates to and from these models. Nothing
provider-shaped leaks into the router, tracing, or environments.

Examples:
    >>> from enroute.types import Message, ChatRequest
    >>> req = ChatRequest(
    ...     model="openai/gpt-4o-mini",
    ...     messages=[Message(role="user", content="Hello")],
    ... )
    >>> req.model
    'openai/gpt-4o-mini'
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Role(str, Enum):
    """Chat message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(str, Enum):
    """Why generation stopped."""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    UNKNOWN = "unknown"


class TextContent(BaseModel):
    """A text content part.

    Attributes:
        type: Discriminator; always ``"text"``.
        text: The text content.
    """

    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    """An image content part (URL or base64 data URL).

    Attributes:
        type: Discriminator; always ``"image_url"``.
        url: HTTP(S) URL or ``data:`` URL for the image.
        detail: Optional detail hint for vision models.
    """

    type: Literal["image_url"] = "image_url"
    url: str
    detail: Literal["auto", "low", "high"] | None = None


class AudioContent(BaseModel):
    """An audio content part.

    Attributes:
        type: Discriminator; always ``"input_audio"``.
        data: Base64-encoded audio bytes.
        format: Audio format such as ``"wav"`` or ``"mp3"``.
    """

    type: Literal["input_audio"] = "input_audio"
    data: str
    format: str = "wav"


ContentPart = TextContent | ImageContent | AudioContent


class FunctionDefinition(BaseModel):
    """JSON-schema function definition for tool calling.

    Attributes:
        name: Function name exposed to the model.
        description: Human-readable description of the function.
        parameters: JSON Schema object describing the parameters.
        strict: Whether to request strict schema adherence when supported.
    """

    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    strict: bool | None = None


class Tool(BaseModel):
    """A tool the model may call.

    Attributes:
        type: Tool type; currently only ``"function"``.
        function: The function definition.

    Examples:
        >>> tool = Tool(
        ...     function=FunctionDefinition(
        ...         name="lookup_order",
        ...         description="Look up an order by id",
        ...         parameters={
        ...             "type": "object",
        ...             "properties": {"order_id": {"type": "string"}},
        ...             "required": ["order_id"],
        ...         },
        ...     )
        ... )
        >>> tool.function.name
        'lookup_order'
    """

    type: Literal["function"] = "function"
    function: FunctionDefinition


class FunctionCall(BaseModel):
    """A function invocation requested by the model.

    Attributes:
        name: Function name to call.
        arguments: JSON-encoded argument object as a string.
    """

    name: str
    arguments: str


class ToolCall(BaseModel):
    """A tool call in an assistant message.

    Attributes:
        id: Provider-assigned call id used to correlate tool responses.
        type: Tool type; currently only ``"function"``.
        function: The function call payload.
    """

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class Message(BaseModel):
    """A single chat message.

    Attributes:
        role: Message role.
        content: Text content, a list of content parts, or ``None`` when the
            message only carries tool calls.
        name: Optional participant name.
        tool_calls: Tool calls requested by the assistant, if any.
        tool_call_id: Id of the tool call this message responds to (tool role).

    Examples:
        >>> Message(role="user", content="Hello").role
        'user'
    """

    model_config = ConfigDict(use_enum_values=True)

    role: Role | Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentPart] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class ResponseFormat(BaseModel):
    """Structured output / response format request.

    Attributes:
        type: Format type such as ``"text"``, ``"json_object"``, or ``"json_schema"``.
        json_schema: Schema payload when ``type`` is ``"json_schema"``.
    """

    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: dict[str, Any] | None = None


class ProviderPreferences(BaseModel):
    """OpenRouter-compatible provider routing preferences.

    Attributes:
        order: Provider slugs to try in order.
        allow_fallbacks: Whether to allow backup providers.
        require_parameters: Only use providers that support all request params.
        data_collection: Whether to allow providers that may store data.
        only: Allow-list of provider slugs.
        ignore: Deny-list of provider slugs.
        sort: Sort key such as ``"price"``, ``"latency"``, or ``"throughput"``.
        max_price: Maximum price constraints for prompt/completion tokens.
    """

    order: list[str] | None = None
    allow_fallbacks: bool = True
    require_parameters: bool = False
    data_collection: Literal["allow", "deny"] | None = None
    only: list[str] | None = None
    ignore: list[str] | None = None
    sort: str | dict[str, Any] | None = None
    max_price: dict[str, float] | None = None


class ChatRequest(BaseModel):
    """A normalized chat completion request.

    Attributes:
        model: Primary model id in ``author/slug`` form.
        messages: Conversation messages.
        models: Optional fallback model chain (tried in order after ``model``).
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        max_tokens: Maximum tokens to generate.
        stop: Stop sequence(s).
        tools: Tools available to the model.
        tool_choice: Tool selection strategy or a forced tool.
        response_format: Structured output request.
        stream: Whether to stream the response.
        seed: Deterministic seed when supported.
        user: End-user identifier for abuse detection / analytics.
        provider: Provider routing preferences.
        metadata: Arbitrary request metadata propagated into traces.
        extra: Provider-specific passthrough fields.

    Examples:
        >>> ChatRequest(
        ...     model="anthropic/claude-sonnet-4",
        ...     messages=[Message(role="user", content="Hi")],
        ... ).temperature is None
        True
    """

    model: str
    messages: list[Message]
    models: list[str] | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    tools: list[Tool] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: ResponseFormat | None = None
    stream: bool = False
    seed: int | None = None
    user: str | None = None
    provider: ProviderPreferences | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    """Token usage for a completion.

    Attributes:
        prompt_tokens: Input token count.
        completion_tokens: Output token count.
        total_tokens: Sum of prompt and completion tokens.
        cost: Estimated USD cost when known.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float | None = None

    @classmethod
    def from_counts(cls, prompt: int, completion: int, cost: float | None = None) -> Usage:
        """Build a :class:`Usage` from prompt/completion counts.

        Args:
            prompt: Prompt token count.
            completion: Completion token count.
            cost: Optional USD cost.

        Returns:
            A populated :class:`Usage` instance.

        Examples:
            >>> Usage.from_counts(10, 5).total_tokens
            15
        """
        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            cost=cost,
        )


class Choice(BaseModel):
    """A single completion choice.

    Attributes:
        index: Choice index.
        message: The assistant message.
        finish_reason: Why generation stopped.
    """

    index: int = 0
    message: Message
    finish_reason: FinishReason | str | None = None


class ChatResponse(BaseModel):
    """A normalized chat completion response.

    Attributes:
        id: Provider-assigned completion id.
        model: Model that produced the response (may differ from the request).
        choices: Completion choices.
        usage: Token usage and optional cost.
        provider: Provider slug that served the request.
        region: Region of the host that served it. The same provider charges
            different rates per region, so billing needs both.
        created: Unix timestamp when the completion was created.
        raw: Original provider payload for debugging.
        latency_ms: End-to-end latency in milliseconds.
        attempts: Number of attempts (including retries/fallbacks) used.

    Examples:
        >>> resp = ChatResponse(
        ...     id="chatcmpl-1",
        ...     model="openai/gpt-4o-mini",
        ...     choices=[Choice(message=Message(role="assistant", content="Hi"))],
        ... )
        >>> resp.text
        'Hi'
    """

    id: str
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)
    provider: str | None = None
    region: str | None = None
    created: int | None = None
    raw: dict[str, Any] | None = None
    latency_ms: float | None = None
    attempts: int = 1

    @property
    def message(self) -> Message:
        """First choice's assistant message.

        Returns:
            The assistant :class:`Message` from choice 0.

        Raises:
            IndexError: If there are no choices.
        """
        return self.choices[0].message

    @property
    def text(self) -> str | None:
        """First choice's text content when it is a string.

        Returns:
            The text content, or ``None`` if missing or multipart.
        """
        content = self.message.content
        if isinstance(content, str):
            return content
        return None


class StreamDelta(BaseModel):
    """A partial delta within a stream chunk.

    Attributes:
        role: Role, typically present only on the first delta.
        content: Incremental text content.
        tool_calls: Incremental tool call fragments.
    """

    role: str | None = None
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class StreamChunk(BaseModel):
    """A single streaming chunk.

    Attributes:
        id: Completion id.
        model: Model id.
        delta: Incremental content.
        finish_reason: Present on the final chunk when known.
        usage: Usage, typically only on the final chunk.
        provider: Provider slug.
        region: Region of the host serving the stream.
        raw: Original provider chunk payload.
    """

    id: str
    model: str
    delta: StreamDelta = Field(default_factory=StreamDelta)
    finish_reason: FinishReason | str | None = None
    usage: Usage | None = None
    provider: str | None = None
    region: str | None = None
    raw: dict[str, Any] | None = None


def text_content(message: Message) -> str:
    """Extract plain text from a message.

    Args:
        message: The message to extract text from.

    Returns:
        Concatenated text content, or an empty string.
    """
    if message.content is None:
        return ""
    if isinstance(message.content, str):
        return message.content
    parts: list[str] = []
    for part in message.content:
        if isinstance(part, TextContent):
            parts.append(part.text)
    return "".join(parts)

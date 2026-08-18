"""Structured output support for hosts with no native ``response_format``.

OpenAI-compatible hosts take ``response_format`` directly. Anthropic and
Bedrock have no equivalent field, but both can force a named tool call whose
input schema *is* the requested schema. That constrains decoding the same way
and hands back a validated object, so the adapters force a tool and then unwrap
its input into message content. Callers asking for JSON get JSON and never
learn a tool was involved.

Gemini does have a native field, but it takes an OpenAPI 3.0 subset rather than
full JSON Schema, so its schema needs pruning first.

Examples:
    >>> from enroute.providers.structured import STRUCTURED_TOOL_NAME
    >>> STRUCTURED_TOOL_NAME
    'json_response'
"""

from __future__ import annotations

from typing import Any

from enroute.types import ChatRequest

STRUCTURED_TOOL_NAME = "json_response"
STRUCTURED_TOOL_DESCRIPTION = "Return the answer as a JSON object matching the schema."
JSON_ONLY_INSTRUCTION = (
    "Respond with a single JSON object and no other text, prose, or code fences."
)

# Gemini's responseSchema is an OpenAPI 3.0 subset and 400s on JSON Schema
# bookkeeping keys, so they are pruned rather than forwarded.
_GEMINI_UNSUPPORTED = frozenset(
    {"$schema", "$defs", "$id", "definitions", "additionalProperties", "strict"}
)
_GEMINI_RECURSE_INTO = ("items", "additionalItems", "not")
_GEMINI_RECURSE_LISTS = ("anyOf", "oneOf", "allOf", "prefixItems")


def structured_schema(request: ChatRequest) -> dict[str, Any] | None:
    """Extract the JSON Schema a request wants the model to satisfy.

    Accepts both OpenAI's wrapper (``{"name": ..., "schema": {...}}``) and a
    bare schema, since callers supply both in practice.

    Args:
        request: Normalized chat request.

    Returns:
        The schema object, or ``None`` when the request did not ask for one.

    Examples:
        >>> from enroute.types import Message, ResponseFormat
        >>> req = ChatRequest(
        ...     model="anthropic/claude-sonnet-4",
        ...     messages=[Message(role="user", content="hi")],
        ...     response_format=ResponseFormat(
        ...         type="json_schema",
        ...         json_schema={"name": "out", "schema": {"type": "object"}},
        ...     ),
        ... )
        >>> structured_schema(req)
        {'type': 'object'}
    """
    fmt = request.response_format
    if fmt is None or fmt.type != "json_schema":
        return None
    payload = fmt.json_schema or {}
    if not isinstance(payload, dict):
        return None
    schema = payload.get("schema", payload)
    if not isinstance(schema, dict) or not schema:
        return None
    return schema


def wants_json_object(request: ChatRequest) -> bool:
    """Whether the request asked for free-form JSON with no schema.

    Args:
        request: Normalized chat request.

    Returns:
        ``True`` for ``response_format={"type": "json_object"}``.
    """
    fmt = request.response_format
    return fmt is not None and fmt.type == "json_object"


def schema_tool_name(request: ChatRequest) -> str | None:
    """Name of the synthetic tool used to emulate structured output.

    Args:
        request: Normalized chat request.

    Returns:
        The tool name when emulation applies, otherwise ``None``. Emulation is
        skipped when the caller supplied real tools, because forcing ours would
        make theirs unreachable.
    """
    if structured_schema(request) is None:
        return None
    if request.tools:
        return None
    return STRUCTURED_TOOL_NAME


def gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Prune a JSON Schema down to what Gemini's ``responseSchema`` accepts.

    Args:
        schema: A JSON Schema object.

    Returns:
        A copy with unsupported bookkeeping keys removed, recursively.

    Examples:
        >>> gemini_schema({"type": "object", "additionalProperties": False})
        {'type': 'object'}
    """
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _GEMINI_UNSUPPORTED:
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {
                name: gemini_schema(sub) if isinstance(sub, dict) else sub
                for name, sub in value.items()
            }
        elif key in _GEMINI_RECURSE_INTO and isinstance(value, dict):
            cleaned[key] = gemini_schema(value)
        elif key in _GEMINI_RECURSE_LISTS and isinstance(value, list):
            cleaned[key] = [gemini_schema(sub) if isinstance(sub, dict) else sub for sub in value]
        else:
            cleaned[key] = value
    return cleaned

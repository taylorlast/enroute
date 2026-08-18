"""Live smoke for tool calling and structured output. Skipped without keys.

Mocked contract tests prove the translation is right for the shapes we expect.
These prove the shapes are the ones hosts actually send, which is the part that
drifts as providers ship new model families.
"""

from __future__ import annotations

import json

import pytest
from hosts import live_cases

from enroute import Enroute, Message
from enroute.types import FunctionDefinition, ResponseFormat, Tool

pytestmark = pytest.mark.live

TOOL_CASES = live_cases("tools")
SCHEMA_CASES = live_cases("response_format")
# Anthropic encrypts its thinking blocks, so there is no readable text to assert
# on; the signature round trip is covered by the contract tests instead.
REASONING_CASES = [case for case in live_cases() if case[0] != "anthropic"]

WEATHER_TOOL = Tool(
    function=FunctionDefinition(
        name="get_weather",
        description="Look up the current weather for a city.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
)
CITY_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "population": {"type": "integer"}},
    "required": ["city", "population"],
    "additionalProperties": False,
}


@pytest.mark.parametrize(("host", "model"), TOOL_CASES)
def test_live_streamed_tool_call_reassembles(host: str, model: str) -> None:
    with Enroute() as client:
        chunks = list(
            client.stream(
                model=model,
                messages=[Message(role="user", content="Weather in Oslo? Use the tool.")],
                tools=[WEATHER_TOOL],
                max_tokens=512,
                provider={"only": [host]},
            )
        )

    merged: dict[int, str] = {}
    names: dict[int, str] = {}
    for chunk in chunks:
        for fragment in chunk.delta.tool_calls or []:
            index = fragment["index"]
            function = fragment.get("function") or {}
            if function.get("name"):
                names[index] = function["name"]
            merged[index] = merged.get(index, "") + (function.get("arguments") or "")

    assert merged, "host streamed no tool call"
    assert set(names.values()) == {"get_weather"}
    for raw in merged.values():
        assert "city" in json.loads(raw)


@pytest.mark.parametrize(("host", "model"), REASONING_CASES)
def test_live_reasoning_arrives_before_any_content(host: str, model: str) -> None:
    """A thinking model must show progress instead of a silent pause.

    This is the failure that started it all: the model spends seconds reasoning,
    emits nothing, then dumps a short answer, and the caller cannot tell the
    difference between thinking and a hung connection.
    """
    # Deliberately hard. These models answer an easy question straight out and
    # skip thinking entirely, which would make the assertion about the prompt
    # rather than about the adapter.
    prompt = (
        "Find every integer triple (x, y, z) with 0 < x, y, z < 20 satisfying "
        "x^3 + y^3 = z^3 + 1. Show your working, then list them."
    )
    with Enroute() as client:
        chunks = list(
            client.stream(
                model=model,
                messages=[Message(role="user", content=prompt)],
                max_tokens=4096,
                provider={"only": [host]},
            )
        )

    assert "".join(c.delta.reasoning or "" for c in chunks), f"{host} streamed no reasoning"
    assert "".join(c.delta.content or "" for c in chunks)

    first_reasoning = next(i for i, c in enumerate(chunks) if c.delta.reasoning)
    first_content = next(i for i, c in enumerate(chunks) if c.delta.content)
    assert first_reasoning < first_content

    # Reasoning must never be mixed into content, or callers rendering the answer
    # would print the model's scratch work to end users.
    assert not any(c.delta.reasoning and c.delta.content for c in chunks)


@pytest.mark.parametrize(("host", "model"), SCHEMA_CASES)
def test_live_structured_output_returns_json_content(host: str, model: str) -> None:
    with Enroute() as client:
        resp = client.chat(
            model=model,
            messages=[Message(role="user", content="Population of Oslo, Norway.")],
            response_format=ResponseFormat(
                type="json_schema", json_schema={"name": "city", "schema": CITY_SCHEMA}
            ),
            max_tokens=512,
            provider={"only": [host]},
        )

    # The forced-tool emulation must stay invisible: JSON arrives as content.
    assert resp.message.tool_calls is None
    parsed = json.loads(resp.text or "")
    assert set(parsed) >= {"city", "population"}
    assert isinstance(parsed["population"], int)


@pytest.mark.parametrize(("host", "model"), SCHEMA_CASES)
def test_live_structured_output_streams_as_content(host: str, model: str) -> None:
    with Enroute() as client:
        chunks = list(
            client.stream(
                model=model,
                messages=[Message(role="user", content="Population of Lima, Peru.")],
                response_format=ResponseFormat(
                    type="json_schema", json_schema={"name": "city", "schema": CITY_SCHEMA}
                ),
                max_tokens=512,
                provider={"only": [host]},
            )
        )

    assert all(not c.delta.tool_calls for c in chunks)
    parsed = json.loads("".join(c.delta.content or "" for c in chunks))
    assert set(parsed) >= {"city", "population"}

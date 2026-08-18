"""Live streaming smoke. Skipped unless the host's key is present.

Pins ``provider.only`` so the router cannot silently serve a different host.
A pass is incremental ``delta.content``, a last-chunk ``usage``, and an OpenAI
wire object from ``to_openai()``.
"""

from __future__ import annotations

import os
import time

import pytest
from hosts import cheapest_model, live_cases

from enroute import Enroute, Message

pytestmark = pytest.mark.live

CASES = live_cases()


def _has_azure() -> bool:
    return bool(os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"))


def _has_bedrock() -> bool:
    return bool(
        os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        or (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))
    )


for _host, _available in (("azure", _has_azure()), ("bedrock", _has_bedrock())):
    if _available:
        _model = cheapest_model(_host)
        if _model is not None:
            CASES.append((_host, _model))


@pytest.mark.parametrize(("host", "model"), CASES)
def test_live_stream_is_openai_shaped(host: str, model: str) -> None:
    with Enroute() as client:
        chunks = list(
            client.stream(
                model=model,
                messages=[Message(role="user", content="Reply with the word pong only.")],
                # Thinking models spend output tokens before writing anything, so a
                # tight cap yields an empty answer and a misleading failure.
                max_tokens=1024,
                provider={"only": [host]},
            )
        )

    text = "".join(c.delta.content or "" for c in chunks)
    assert text
    assert {c.provider for c in chunks} == {host}
    payload = chunks[0].to_openai()
    assert payload["object"] == "chat.completion.chunk"
    assert "choices" in payload
    usage = next((c.usage for c in reversed(chunks) if c.usage), None)
    assert usage is not None
    assert usage.completion_tokens >= 0


@pytest.mark.parametrize(("host", "model"), CASES)
def test_live_stream_arrives_incrementally(host: str, model: str) -> None:
    """Output must arrive progressively rather than in one final burst.

    Asked for a long answer, because a short one fits in a single delta on most
    hosts and would pass even against a fully buffered transport. Reasoning
    counts as output: Gemini flushes its prose in a batch at the end but streams
    thought summaries as it goes, and a caller watching either one can tell the
    difference between a working stream and a hung socket.
    """
    started = time.perf_counter()
    arrivals: list[float] = []
    with Enroute() as client:
        for chunk in client.stream(
            model=model,
            messages=[
                Message(role="user", content="Write 300 words about a lighthouse. Plain prose.")
            ],
            max_tokens=4096,
            provider={"only": [host]},
        ):
            if chunk.delta.content or chunk.delta.reasoning:
                arrivals.append(time.perf_counter() - started)

    assert len(arrivals) > 1, "host sent everything in one delta"
    assert arrivals[0] < arrivals[-1]

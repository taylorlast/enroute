"""Live streaming smoke. Skipped unless the host's key is present.

Pins ``provider.only`` so the router cannot silently serve a different host.
A pass is incremental ``delta.content``, a last-chunk ``usage``, and an OpenAI
wire object from ``to_openai()``.
"""

from __future__ import annotations

import os

import pytest

from enroute import Enroute, Message

pytestmark = pytest.mark.live

CASES = [
    ("OPENAI_API_KEY", "openai", "openai/gpt-4o-mini"),
    ("ANTHROPIC_API_KEY", "anthropic", "anthropic/claude-sonnet-4"),
    ("GOOGLE_API_KEY", "google", "google/gemini-2.5-flash"),
]


def _has_azure() -> bool:
    return bool(os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"))


def _has_bedrock() -> bool:
    return bool(
        os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        or (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))
    )


if _has_azure():
    CASES.append(("AZURE_OPENAI_API_KEY", "azure", "openai/gpt-5.6-sol"))
if _has_bedrock():
    CASES.append(("AWS_REGION", "bedrock", "openai/gpt-5.6-sol"))


@pytest.mark.parametrize(("env_key", "host", "model"), CASES)
def test_live_stream_is_openai_shaped(env_key: str, host: str, model: str) -> None:
    if host == "azure" and not _has_azure():
        pytest.skip("Azure is not configured")
    if host == "bedrock" and not _has_bedrock():
        pytest.skip("Bedrock is not configured")
    if not os.environ.get(env_key):
        pytest.skip(f"{env_key} not set")

    with Enroute() as client:
        chunks = list(
            client.stream(
                model=model,
                messages=[Message(role="user", content="Reply with the word pong only.")],
                max_tokens=16,
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

"""Live provider smoke tests. Skipped unless keys are present."""

from __future__ import annotations

import os

import pytest

from enroute import Enroute, Message

pytestmark = pytest.mark.live


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_live() -> None:
    with Enroute(providers={"openai": os.environ["OPENAI_API_KEY"]}) as client:
        resp = client.chat(
            model="openai/gpt-4o-mini",
            messages=[Message(role="user", content="Reply with the word pong only.")],
            max_tokens=8,
        )
    assert resp.text

"""Capture traces with redaction and late labeling."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _shared import ScriptedProvider, ensure_out_dir
from enroute import Enroute, Message, Redactor
from enroute.tracing import SQLiteSink


def main() -> None:
    out = ensure_out_dir()
    sqlite = SQLiteSink(out / "traces.sqlite")
    redactor = Redactor(fields={"metadata.email"}, patterns=[r"\b\d{3}-\d{2}-\d{4}\b"])
    with Enroute(
        providers={"openai": ScriptedProvider("openai", "done")},
        sink=sqlite,
        redactor=redactor,
        capture_content=True,
    ) as client:
        resp = client.chat(
            model="openai/gpt-4o-mini",
            messages=[Message(role="user", content="ssn 123-45-6789")],
            metadata={"email": "user@example.com"},
        )
        trace_id = (resp.raw or {})["enroute_trace_id"]
        client.flush()
        client.label(trace_id, reward=1.0, labels={"resolved": True})
    loaded = SQLiteSink(out / "traces.sqlite").get(trace_id)
    assert loaded is not None
    print("email redacted:", loaded.metadata.get("email"))
    print("reward:", loaded.outcome.reward if loaded.outcome else None)


if __name__ == "__main__":
    main()

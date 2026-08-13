from pathlib import Path

from enroute.tracing import JSONLSink, Redactor, Sampler, SQLiteSink, Trace, TraceWriter
from enroute.tracing.schema import Outcome


def test_jsonl_sink(tmp_path: Path) -> None:
    sink = JSONLSink(tmp_path / "t.jsonl")
    sink.write(Trace(trace_id="a"))
    sink.write(Trace(trace_id="b"))
    sink.close()
    traces = JSONLSink(tmp_path / "t.jsonl").read_all()
    assert [t.trace_id for t in traces] == ["a", "b"]


def test_sqlite_sink_and_label(tmp_path: Path) -> None:
    sink = SQLiteSink(tmp_path / "t.sqlite")
    writer = TraceWriter(sink)
    writer.record(Trace(trace_id="x"))
    writer.flush()
    writer.label("x", reward=1.0, scores={"ok": 1.0})
    writer.close()
    loaded = SQLiteSink(tmp_path / "t.sqlite").get("x")
    assert loaded is not None
    assert loaded.outcome is not None
    assert loaded.outcome.reward == 1.0


def test_redactor_fields() -> None:
    redactor = Redactor(fields={"metadata.email"})
    trace = Trace(trace_id="t", metadata={"email": "a@b.com", "ok": 1})
    out = redactor.apply(trace)
    assert out.metadata["email"] == "[REDACTED]"
    assert out.metadata["ok"] == 1


def test_sampler_always_tags() -> None:
    sampler = Sampler(rate=0.0, always_tags={"keep"})
    assert sampler.accept(Trace(trace_id="t", tags={"keep": "1"}))
    assert not sampler.accept(Trace(trace_id="t2"))


def test_trace_label_helper() -> None:
    t = Trace(trace_id="t")
    t.label(scores={"a": 1.0}, reward=1.0)
    assert t.outcome == Outcome(scores={"a": 1.0}, reward=1.0, labels={}, feedback=None)

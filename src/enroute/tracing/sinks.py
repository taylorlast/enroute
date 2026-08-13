"""Trace sinks: JSONL, SQLite, OpenTelemetry, and fan-out.

Examples:
    >>> from pathlib import Path
    >>> from enroute.tracing.schema import Trace
    >>> from enroute.tracing.sinks import JSONLSink
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as d:
    ...     sink = JSONLSink(Path(d) / "t.jsonl")
    ...     sink.write(Trace(trace_id="abc"))
    ...     sink.close()
    ...     lines = (Path(d) / "t.jsonl").read_text().strip().splitlines()
    ...     len(lines) == 1
    True
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from enroute.tracing.schema import Trace


@runtime_checkable
class Sink(Protocol):
    """Destination for persisted traces."""

    def write(self, trace: Trace) -> None:
        """Persist a single trace."""
        ...

    def close(self) -> None:
        """Flush and release resources."""
        ...


class JSONLSink:
    """Append traces as JSON lines to a file.

    Args:
        path: Destination ``.jsonl`` path.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._closed = False

    def write(self, trace: Trace) -> None:
        """Append one JSON line.

        Args:
            trace: Trace to persist.
        """
        with self._lock:
            if self._closed:
                return
            self._fh.write(trace.model_dump_json() + "\n")
            self._fh.flush()

    def read_all(self) -> list[Trace]:
        """Read all traces from the file.

        Returns:
            List of traces in file order.
        """
        if not self.path.exists():
            return []
        traces: list[Trace] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    traces.append(Trace.model_validate_json(line))
        return traces

    def close(self) -> None:
        """Close the file handle."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._fh.close()


class SQLiteSink:
    """Persist traces in a SQLite database for querying.

    Args:
        path: Destination ``.sqlite`` path.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._closed = False
        # Background TraceWriter may call from a worker thread.
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    environment TEXT,
                    task_id TEXT,
                    created_at TEXT,
                    payload TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    def write(self, trace: Trace) -> None:
        """Upsert a trace by ``trace_id``.

        Args:
            trace: Trace to persist.
        """
        with self._lock:
            if self._closed:
                return
            self._conn.execute(
                """
                INSERT INTO traces (trace_id, environment, task_id, created_at, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(trace_id) DO UPDATE SET
                    environment=excluded.environment,
                    task_id=excluded.task_id,
                    created_at=excluded.created_at,
                    payload=excluded.payload
                """,
                (
                    trace.trace_id,
                    trace.environment,
                    trace.task_id,
                    trace.created_at.isoformat(),
                    trace.model_dump_json(),
                ),
            )
            self._conn.commit()

    def get(self, trace_id: str) -> Trace | None:
        """Fetch a trace by id.

        Args:
            trace_id: Trace id.

        Returns:
            The trace, or ``None``.
        """
        with self._lock:
            if self._closed:
                return None
            row = self._conn.execute(
                "SELECT payload FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
        if row is None:
            return None
        return Trace.model_validate_json(row[0])

    def query(
        self,
        *,
        environment: str | None = None,
        tag: str | None = None,
        limit: int = 100,
    ) -> list[Trace]:
        """Query traces with simple filters.

        Args:
            environment: Optional environment name filter.
            tag: Optional tag key that must be present.
            limit: Maximum rows.

        Returns:
            Matching traces, newest first.
        """
        sql = "SELECT payload FROM traces"
        clauses: list[str] = []
        params: list[Any] = []
        if environment is not None:
            clauses.append("environment = ?")
            params.append(environment)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            if self._closed:
                return []
            rows = self._conn.execute(sql, params).fetchall()
        traces = [Trace.model_validate_json(r[0]) for r in rows]
        if tag is not None:
            traces = [t for t in traces if tag in t.tags]
        return traces

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._conn.close()


class MultiSink:
    """Fan out writes to multiple sinks.

    Args:
        sinks: Child sinks.
    """

    def __init__(self, sinks: list[Sink]) -> None:
        self.sinks = sinks

    def write(self, trace: Trace) -> None:
        """Write to every child sink.

        Args:
            trace: Trace to persist.
        """
        for sink in self.sinks:
            sink.write(trace)

    def close(self) -> None:
        """Close every child sink."""
        for sink in self.sinks:
            sink.close()


class OTelSink:
    """Export traces as OpenTelemetry GenAI-style spans.

    The enroute :class:`~enroute.tracing.schema.Trace` remains canonical. This
    sink is an exporter onto the still-Development ``gen_ai.*`` conventions.

    Args:
        tracer_name: Tracer instrumentation name.

    Note:
        Requires the ``enroute[otel]`` extra.
    """

    def __init__(self, tracer_name: str = "enroute") -> None:
        try:
            from opentelemetry import trace as otel_trace
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "OTelSink requires the 'otel' extra: pip install enroute[otel]"
            ) from exc
        self._tracer = otel_trace.get_tracer(tracer_name)

    def write(self, trace: Trace) -> None:
        """Emit one span per LLM step plus a parent span for the trace.

        Args:
            trace: Trace to export.
        """
        with self._tracer.start_as_current_span(f"enroute.trace.{trace.trace_id}") as parent:
            parent.set_attribute("enroute.trace_id", trace.trace_id)
            if trace.environment:
                parent.set_attribute("enroute.environment", trace.environment)
            if trace.task_id:
                parent.set_attribute("enroute.task_id", trace.task_id)
            from enroute.tracing.schema import LLMCall

            for step in trace.steps:
                if not isinstance(step, LLMCall):
                    continue
                with self._tracer.start_as_current_span("gen_ai.chat") as span:
                    span.set_attribute("gen_ai.operation.name", "chat")
                    if step.provider:
                        span.set_attribute("gen_ai.provider.name", step.provider)
                    if step.model:
                        span.set_attribute("gen_ai.request.model", step.model)
                    if step.usage:
                        span.set_attribute("gen_ai.usage.input_tokens", step.usage.prompt_tokens)
                        span.set_attribute(
                            "gen_ai.usage.output_tokens", step.usage.completion_tokens
                        )
                    if step.latency_ms is not None:
                        span.set_attribute("enroute.latency_ms", step.latency_ms)
                    if step.error:
                        span.set_attribute("error.type", step.error)

    def close(self) -> None:
        """No-op; tracer provider lifecycle is owned by the host app."""
        return None


def traces_from_jsonl(path: str | Path) -> list[Trace]:
    """Load traces from a JSONL file without opening a writable sink.

    Args:
        path: Path to a JSONL file.

    Returns:
        Parsed traces.
    """
    path = Path(path)
    if not path.exists():
        return []
    out: list[Trace] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(Trace.model_validate_json(line))
    return out


# Silence unused import warning for json in type checkers that don't see dumps usage.
_ = json

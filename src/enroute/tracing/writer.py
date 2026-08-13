"""Non-blocking background trace writer.

Writes go through a queue so tracing never blocks a request path. Call
:meth:`TraceWriter.flush` or :meth:`TraceWriter.close` on shutdown.

Examples:
    >>> from pathlib import Path
    >>> import tempfile
    >>> from enroute.tracing.schema import Trace
    >>> from enroute.tracing.sinks import JSONLSink
    >>> from enroute.tracing.writer import TraceWriter
    >>> with tempfile.TemporaryDirectory() as d:
    ...     sink = JSONLSink(Path(d) / "t.jsonl")
    ...     writer = TraceWriter(sink)
    ...     writer.record(Trace(trace_id="x"))
    ...     writer.close()
    ...     True
    True
"""

from __future__ import annotations

import atexit
import contextlib
import queue
import threading
from typing import Any

from enroute.tracing.redaction import Redactor, Sampler
from enroute.tracing.schema import Outcome, Trace
from enroute.tracing.sinks import Sink


class TraceWriter:
    """Background writer that applies redaction/sampling then sinks traces.

    Args:
        sink: Destination sink.
        redactor: Optional redactor applied before write.
        sampler: Optional sampler; defaults to keep-all.
        maxsize: Queue size before ``record`` blocks.
    """

    def __init__(
        self,
        sink: Sink,
        *,
        redactor: Redactor | None = None,
        sampler: Sampler | None = None,
        maxsize: int = 1000,
    ) -> None:
        self.sink = sink
        self.redactor = redactor
        self.sampler = sampler or Sampler(rate=1.0)
        self._queue: queue.Queue[Trace | None] = queue.Queue(maxsize=maxsize)
        self._labels: dict[str, Outcome] = {}
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="enroute-trace-writer", daemon=True)
        self._closed = False
        self._thread.start()
        atexit.register(self._atexit_close)

    def record(self, trace: Trace) -> None:
        """Enqueue a trace for persistence.

        Args:
            trace: Trace to record.
        """
        if self._closed:
            return
        with self._lock:
            pending = self._labels.pop(trace.trace_id, None)
        if pending is not None:
            current = trace.outcome or Outcome()
            current.scores.update(pending.scores)
            if pending.reward is not None:
                current.reward = pending.reward
            current.labels.update(pending.labels)
            if pending.feedback is not None:
                current.feedback = pending.feedback
            trace.outcome = current
        if not self.sampler.accept(trace):
            return
        if self.redactor is not None:
            trace = self.redactor.apply(trace)
        self._queue.put(trace)

    def label(
        self,
        trace_id: str,
        *,
        scores: dict[str, float] | None = None,
        reward: float | None = None,
        labels: dict[str, Any] | None = None,
        feedback: str | None = None,
    ) -> None:
        """Attach a late label to a trace id.

        If the trace has not been written yet, the label is merged on record.
        If a SQLite sink is used and the trace already exists, it is updated.

        Args:
            trace_id: Trace id to label.
            scores: Named scores.
            reward: Scalar reward.
            labels: Discrete labels.
            feedback: Free-form feedback.
        """
        outcome = Outcome(
            scores=scores or {},
            reward=reward,
            labels=labels or {},
            feedback=feedback,
        )
        with self._lock:
            existing = self._labels.get(trace_id)
            if existing:
                existing.scores.update(outcome.scores)
                if outcome.reward is not None:
                    existing.reward = outcome.reward
                existing.labels.update(outcome.labels)
                if outcome.feedback is not None:
                    existing.feedback = outcome.feedback
            else:
                self._labels[trace_id] = outcome

        get = getattr(self.sink, "get", None)
        write = getattr(self.sink, "write", None)
        if callable(get) and callable(write):
            trace = get(trace_id)
            if trace is not None:
                trace.label(
                    scores=scores,
                    reward=reward,
                    labels=labels,
                    feedback=feedback,
                )
                if self.redactor is not None:
                    trace = self.redactor.apply(trace)
                write(trace)

    def flush(self) -> None:
        """Block until the queue is empty."""
        self._queue.join()

    def close(self) -> None:
        """Flush, stop the worker, and close the sink."""
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=5.0)
        self.sink.close()
        with contextlib.suppress(Exception):
            atexit.unregister(self._atexit_close)

    def _atexit_close(self) -> None:
        """Best-effort close registered with :mod:`atexit`."""
        with contextlib.suppress(Exception):
            self.close()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                self.sink.write(item)
            finally:
                self._queue.task_done()

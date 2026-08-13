"""Unified tracing: the same Trace object for production traffic and environments.

Examples:
    >>> from enroute.tracing import Trace, JSONLSink
    >>> Trace(trace_id="t1").trace_id
    't1'
"""

from __future__ import annotations

from enroute.tracing.redaction import Redactor, Sampler
from enroute.tracing.schema import Event, LLMCall, Outcome, Step, ToolCallStep, Trace
from enroute.tracing.sinks import JSONLSink, MultiSink, OTelSink, Sink, SQLiteSink
from enroute.tracing.writer import TraceWriter

__all__ = [
    "Event",
    "JSONLSink",
    "LLMCall",
    "MultiSink",
    "OTelSink",
    "Outcome",
    "Redactor",
    "SQLiteSink",
    "Sampler",
    "Sink",
    "Step",
    "ToolCallStep",
    "Trace",
    "TraceWriter",
]

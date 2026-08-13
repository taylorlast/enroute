"""Versioned datasets of traces for benchmarks and training.

Examples:
    >>> from enroute.environments.dataset import Dataset
    >>> from enroute.tracing.schema import Trace
    >>> ds = Dataset.from_traces("demo", [Trace(trace_id="a"), Trace(trace_id="b")])
    >>> len(ds)
    2
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from enroute.tracing.schema import Trace
from enroute.tracing.sinks import JSONLSink, SQLiteSink, traces_from_jsonl


class Dataset(BaseModel):
    """A named, content-hashed collection of traces.

    Attributes:
        name: Dataset name.
        version: Human version label.
        traces: Contained traces.
        content_hash: Hash of trace ids + outcomes for reproducibility.
        metadata: Arbitrary metadata.
    """

    name: str
    version: str = "0.1.0"
    traces: list[Trace] = Field(default_factory=list)
    content_hash: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Compute content hash after init."""
        if not self.content_hash:
            self.content_hash = self.compute_hash()

    def __len__(self) -> int:
        """Return the number of traces."""
        return len(self.traces)

    def compute_hash(self) -> str:
        """Compute a stable content hash.

        Returns:
            Hex SHA256 digest.
        """
        h = hashlib.sha256()
        for trace in sorted(self.traces, key=lambda t: t.trace_id):
            payload = {
                "trace_id": trace.trace_id,
                "outcome": trace.outcome.model_dump() if trace.outcome else None,
                "task_id": trace.task_id,
                "environment": trace.environment,
            }
            h.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
        return h.hexdigest()

    @classmethod
    def from_traces(
        cls,
        name: str,
        traces: Iterable[Trace],
        *,
        version: str = "0.1.0",
        metadata: dict[str, Any] | None = None,
    ) -> Dataset:
        """Build a dataset from an iterable of traces.

        Args:
            name: Dataset name.
            traces: Traces to include.
            version: Version label.
            metadata: Optional metadata.

        Returns:
            A new :class:`Dataset`.
        """
        return cls(
            name=name,
            version=version,
            traces=list(traces),
            metadata=metadata or {},
        )

    @classmethod
    def from_sink(
        cls,
        sink: JSONLSink | SQLiteSink | Path | str,
        name: str,
        *,
        where: Callable[[Trace], bool] | None = None,
        version: str = "0.1.0",
    ) -> Dataset:
        """Build a dataset from a sink or JSONL path.

        Args:
            sink: A :class:`JSONLSink`, :class:`SQLiteSink`, or filesystem path.
            name: Dataset name.
            where: Optional predicate filter.
            version: Version label.

        Returns:
            A filtered :class:`Dataset`.
        """
        if isinstance(sink, (str, Path)):
            traces = traces_from_jsonl(sink)
        elif isinstance(sink, JSONLSink):
            traces = sink.read_all()
        elif isinstance(sink, SQLiteSink):
            traces = sink.query(limit=100_000)
        else:  # pragma: no cover
            raise TypeError(f"unsupported sink type: {type(sink)}")
        if where is not None:
            traces = [t for t in traces if where(t)]
        return cls.from_traces(name, traces, version=version)

    def save(self, path: str | Path) -> None:
        """Save the dataset as JSONL plus a sidecar manifest.

        Args:
            path: Destination JSONL path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for trace in self.traces:
                fh.write(trace.model_dump_json() + "\n")
        manifest = {
            "name": self.name,
            "version": self.version,
            "content_hash": self.content_hash,
            "count": len(self.traces),
            "metadata": self.metadata,
        }
        path.with_suffix(path.suffix + ".manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path, name: str | None = None) -> Dataset:
        """Load a dataset from JSONL.

        Args:
            path: JSONL path.
            name: Optional override name.

        Returns:
            Loaded dataset.
        """
        path = Path(path)
        traces = traces_from_jsonl(path)
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        version = "0.1.0"
        metadata: dict[str, Any] = {}
        ds_name = name or path.stem
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            ds_name = name or manifest.get("name") or ds_name
            version = manifest.get("version") or version
            metadata = manifest.get("metadata") or {}
        return cls.from_traces(ds_name, traces, version=version, metadata=metadata)

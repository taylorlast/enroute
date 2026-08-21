"""Export datasets to Hugging Face-style record lists.

Examples:
    >>> from enroute.environments.dataset import Dataset
    >>> from enroute.environments.export.hf import to_huggingface_records
    >>> from enroute.tracing.schema import Trace
    >>> recs = to_huggingface_records(Dataset.from_traces("d", [Trace(trace_id="1")]))
    >>> recs[0]["trace_id"]
    '1'
"""

from __future__ import annotations

from typing import Any

from enroute.environments.dataset import Dataset


def to_huggingface_records(dataset: Dataset) -> list[dict[str, Any]]:
    """Convert a dataset to a list of HF-friendly dictionaries.

    Args:
        dataset: Source dataset.

    Returns:
        List of flat-ish records suitable for ``datasets.Dataset.from_list``.
    """
    records: list[dict[str, Any]] = []
    for trace in dataset.traces:
        records.append(
            {
                "trace_id": trace.trace_id,
                "environment": trace.environment,
                "environment_version": trace.environment_version,
                "environment_fingerprint": trace.environment_fingerprint,
                "task_id": trace.task_id,
                "model": trace.model,
                "reward": trace.outcome.reward if trace.outcome else None,
                "scores": trace.outcome.scores if trace.outcome else {},
                "labels": trace.outcome.labels if trace.outcome else {},
                "tags": trace.tags,
                "steps": [s.model_dump(mode="json") for s in trace.steps],
                "transitions": [t.model_dump(mode="json") for t in trace.transitions()],
                "returns": trace.returns(),
                "initial_state": trace.initial_state,
                "final_state": trace.final_state,
                "metrics": trace.metrics,
                "terminated": trace.terminated,
                "truncated": trace.truncated,
                "metadata": trace.metadata,
                "created_at": trace.created_at.isoformat(),
            }
        )
    return records

"""Pre-sink redaction and sampling.

Redactors and samplers run **before** a sink writes, so PII never reaches disk
when configured correctly.

Examples:
    >>> from enroute.tracing.redaction import Redactor, Sampler
    >>> from enroute.tracing.schema import Trace
    >>> r = Redactor(fields={"metadata.email"})
    >>> t = Trace(trace_id="t", metadata={"email": "a@b.com", "ok": 1})
    >>> redacted = r.apply(t)
    >>> redacted.metadata["email"]
    '[REDACTED]'
"""

from __future__ import annotations

import copy
import random
import re
from collections.abc import Callable
from typing import Any

from enroute.tracing.schema import Trace


class Redactor:
    """Redact sensitive fields and patterns from traces.

    Args:
        fields: Dot-paths to replace with ``[REDACTED]`` (e.g. ``metadata.email``).
        patterns: Regex patterns applied to all string values.
        replacement: Replacement string for matches.
        callables: Custom callables ``(trace) -> trace`` applied last.
        drop_content: If ``True``, strip message content from LLM request/response.
    """

    def __init__(
        self,
        *,
        fields: set[str] | None = None,
        patterns: list[str] | None = None,
        replacement: str = "[REDACTED]",
        callables: list[Callable[[Trace], Trace]] | None = None,
        drop_content: bool = False,
    ) -> None:
        self.fields = fields or set()
        self.patterns = [re.compile(p) for p in (patterns or [])]
        self.replacement = replacement
        self.callables = callables or []
        self.drop_content = drop_content

    def apply(self, trace: Trace) -> Trace:
        """Return a redacted copy of ``trace``.

        Args:
            trace: The original trace.

        Returns:
            A redacted deep copy.
        """
        data = trace.model_dump(mode="python")
        for path in self.fields:
            _set_path(data, path.split("."), self.replacement)
        if self.patterns:
            data = _apply_patterns(data, self.patterns, self.replacement)
        if self.drop_content:
            data = _drop_content(data)
        result = Trace.model_validate(data)
        for fn in self.callables:
            result = fn(result)
        return result


class Sampler:
    """Decide whether a trace should be persisted.

    Args:
        rate: Probability in ``[0, 1]`` that a trace is kept.
        always_tags: Tags that force retention when present.
    """

    def __init__(self, rate: float = 1.0, *, always_tags: set[str] | None = None) -> None:
        if not 0.0 <= rate <= 1.0:
            raise ValueError("rate must be between 0 and 1")
        self.rate = rate
        self.always_tags = always_tags or set()

    def accept(self, trace: Trace) -> bool:
        """Return whether ``trace`` should be written.

        Args:
            trace: Candidate trace.

        Returns:
            ``True`` if the trace should be persisted.

        Examples:
            >>> Sampler(rate=1.0).accept(Trace(trace_id="t"))
            True
        """
        if self.always_tags and self.always_tags.intersection(trace.tags):
            return True
        if self.rate >= 1.0:
            return True
        if self.rate <= 0.0:
            return False
        return random.random() < self.rate


def _set_path(obj: Any, parts: list[str], value: Any) -> None:
    if not parts:
        return
    head, *rest = parts
    if isinstance(obj, dict):
        if not rest:
            if head in obj:
                obj[head] = value
            return
        if head in obj:
            _set_path(obj[head], rest, value)


def _apply_patterns(obj: Any, patterns: list[re.Pattern[str]], replacement: str) -> Any:
    if isinstance(obj, str):
        result = obj
        for pattern in patterns:
            result = pattern.sub(replacement, result)
        return result
    if isinstance(obj, list):
        return [_apply_patterns(x, patterns, replacement) for x in obj]
    if isinstance(obj, dict):
        return {k: _apply_patterns(v, patterns, replacement) for k, v in obj.items()}
    return obj


def _omit_request_messages(req: Any) -> None:
    if isinstance(req, dict) and "messages" in req:
        for msg in req["messages"]:
            if isinstance(msg, dict) and "content" in msg:
                msg["content"] = "[CONTENT_OMITTED]"


def _omit_response_content(resp: Any) -> None:
    if isinstance(resp, dict):
        for choice in resp.get("choices") or []:
            message = choice.get("message")
            if isinstance(message, dict) and "content" in message:
                message["content"] = "[CONTENT_OMITTED]"


def _drop_content(data: dict[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(data)
    for step in data.get("steps") or []:
        step_type = step.get("type")
        if step_type == "llm":
            _omit_request_messages(step.get("request"))
            _omit_response_content(step.get("response"))
        elif step_type == "decision":
            _omit_request_messages(step.get("model_context"))
            _omit_response_content(step.get("model_output"))
            if "observation" in step and step["observation"] is not None:
                step["observation"] = "[CONTENT_OMITTED]"
    return data

"""Benchmark runner and report types.

Examples:
    >>> from enroute.benchmarks.runner import Report, ModelStats
    >>> r = Report(environment="demo", models={})
    >>> r.environment
    'demo'
"""

from __future__ import annotations

import json
import math
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from pydantic import BaseModel, Field

from enroute.client import Enroute
from enroute.environments.env import Environment, Rollout, TaskData


class ModelStats(BaseModel):
    """Aggregate stats for one model in a benchmark.

    Attributes:
        model: Model id.
        n: Number of successful rollouts.
        failures: Number of failed rollouts.
        mean_reward: Mean reward.
        mean_scores: Per-scorer means.
        mean_latency_ms: Mean end-to-end latency.
        p50_latency_ms: Median latency.
        p95_latency_ms: 95th percentile latency.
        mean_cost: Mean USD cost.
        total_cost: Total USD cost.
        failure_reasons: Count of failure reason strings.
    """

    model: str
    n: int = 0
    failures: int = 0
    mean_reward: float | None = None
    mean_scores: dict[str, float] = Field(default_factory=dict)
    mean_latency_ms: float | None = None
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    mean_cost: float | None = None
    total_cost: float = 0.0
    failure_reasons: dict[str, int] = Field(default_factory=dict)


class Report(BaseModel):
    """Benchmark report across models.

    Attributes:
        environment: Environment name.
        environment_version: Environment version.
        models: Per-model stats.
        win_rates: Pairwise win rates ``{model_a: {model_b: rate}}``.
        metadata: Arbitrary metadata.
    """

    environment: str
    environment_version: str = "0.1.0"
    models: dict[str, ModelStats] = Field(default_factory=dict)
    win_rates: dict[str, dict[str, float]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize the report as JSON.

        Returns:
            JSON string.
        """
        return self.model_dump_json(indent=2)

    def to_markdown(self) -> str:
        """Render a markdown summary table.

        Returns:
            Markdown string.
        """
        lines = [
            f"# Benchmark: {self.environment} ({self.environment_version})",
            "",
            (
                "| Model | N | Failures | Mean reward | Mean cost "
                "| p50 latency (ms) | p95 latency (ms) |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for model, stats in sorted(self.models.items()):
            lines.append(
                f"| {model} | {stats.n} | {stats.failures} | {_fmt(stats.mean_reward)} "
                f"| {_fmt(stats.mean_cost)} | {_fmt(stats.p50_latency_ms)} "
                f"| {_fmt(stats.p95_latency_ms)} |"
            )
        if self.win_rates:
            lines.extend(["", "## Win rates", ""])
            for a, opponents in self.win_rates.items():
                for b, rate in opponents.items():
                    lines.append(f"- `{a}` vs `{b}`: {rate:.2%}")
        return "\n".join(lines) + "\n"

    def compare(self, other: Report) -> dict[str, Any]:
        """Compare this report to another for CI regression checks.

        Args:
            other: Baseline report.

        Returns:
            Dict with per-model reward deltas and regressions list.
        """
        deltas: dict[str, float | None] = {}
        regressions: list[str] = []
        for model, stats in self.models.items():
            base = other.models.get(model)
            if base is None or stats.mean_reward is None or base.mean_reward is None:
                deltas[model] = None
                continue
            delta = stats.mean_reward - base.mean_reward
            deltas[model] = delta
            if delta < -1e-9:
                regressions.append(model)
        return {"deltas": deltas, "regressions": regressions}


class Benchmark:
    """Run an environment across models and aggregate a report.

    Args:
        env: Environment to evaluate.
        models: Model ids to compare.
        repeats: Times to run each task per model.
        concurrency: Max concurrent rollouts.
        client: Shared enroute client.
    """

    def __init__(
        self,
        env: Environment[Any, Any],
        models: list[str],
        *,
        client: Enroute,
        repeats: int = 1,
        concurrency: int = 4,
    ) -> None:
        self.env = env
        self.models = models
        self.client = client
        self.repeats = repeats
        self.concurrency = concurrency

    def run(self, tasks: list[TaskData] | None = None) -> Report:
        """Execute the benchmark.

        Args:
            tasks: Optional explicit task list; defaults to ``env.iter_tasks()``.

        Returns:
            Aggregated :class:`Report`.
        """
        task_list = list(tasks) if tasks is not None else list(self.env.iter_tasks())
        jobs: list[tuple[str, TaskData, int]] = []
        for model in self.models:
            for task in task_list:
                for repeat in range(self.repeats):
                    jobs.append((model, task, repeat))

        results: dict[str, list[Rollout | BaseException]] = {m: [] for m in self.models}

        def _run(job: tuple[str, TaskData, int]) -> tuple[str, Rollout | BaseException]:
            model, task, _repeat = job
            try:
                return model, self.env.spawn().rollout(task, self.client, model=model)
            except BaseException as exc:  # noqa: BLE001
                return model, exc

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = [pool.submit(_run, job) for job in jobs]
            for fut in as_completed(futures):
                model, result = fut.result()
                results[model].append(result)

        model_stats = {model: _aggregate(model, items) for model, items in results.items()}
        rewards: dict[str, list[float]] = {}
        for model, items in results.items():
            rewards[model] = [
                r.trace.outcome.reward
                for r in items
                if isinstance(r, Rollout) and r.trace.outcome and r.trace.outcome.reward is not None
            ]
        return Report(
            environment=self.env.name,
            environment_version=self.env.version,
            models=model_stats,
            win_rates=_win_rates(rewards),
            metadata={
                "repeats": self.repeats,
                "tasks": len(task_list),
                "environment_fingerprint": self.env.fingerprint(),
            },
        )


def _aggregate(model: str, items: list[Rollout | BaseException]) -> ModelStats:
    rewards: list[float] = []
    score_lists: dict[str, list[float]] = {}
    latencies: list[float] = []
    costs: list[float] = []
    failures = 0
    failure_reasons: dict[str, int] = {}

    for item in items:
        if isinstance(item, BaseException):
            failures += 1
            key = type(item).__name__
            failure_reasons[key] = failure_reasons.get(key, 0) + 1
            continue
        outcome = item.trace.outcome
        if outcome and outcome.reward is not None:
            rewards.append(outcome.reward)
        if outcome:
            for name, value in outcome.scores.items():
                score_lists.setdefault(name, []).append(value)
        # Prefer last decision / LLM step latency / cost.
        for step in reversed(item.trace.steps):
            if step.type == "decision":
                output = step.model_output
                if output is None:
                    continue
                latency = getattr(output, "latency_ms", None)
                if latency is None and isinstance(output, dict):
                    latency = output.get("latency_ms")
                usage = getattr(output, "usage", None)
                cost = usage.cost if usage is not None and hasattr(usage, "cost") else None
                if cost is None and isinstance(output, dict):
                    cost = (output.get("usage") or {}).get("cost")
                if latency is not None:
                    latencies.append(float(latency))
                if cost is not None:
                    costs.append(float(cost))
                break
            if step.type == "llm":
                if step.latency_ms is not None:
                    latencies.append(step.latency_ms)
                if step.cost is not None:
                    costs.append(step.cost)
                break

    return ModelStats(
        model=model,
        n=len(items) - failures,
        failures=failures,
        mean_reward=statistics.fmean(rewards) if rewards else None,
        mean_scores={k: statistics.fmean(v) for k, v in score_lists.items() if v},
        mean_latency_ms=statistics.fmean(latencies) if latencies else None,
        p50_latency_ms=_percentile(latencies, 0.50),
        p95_latency_ms=_percentile(latencies, 0.95),
        mean_cost=statistics.fmean(costs) if costs else None,
        total_cost=sum(costs),
        failure_reasons=failure_reasons,
    )


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = q * (len(ordered) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo)


def _win_rates(rewards: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    models = list(rewards)
    result: dict[str, dict[str, float]] = {}
    for i, a in enumerate(models):
        result[a] = {}
        for b in models[i + 1 :]:
            a_vals = rewards[a]
            b_vals = rewards[b]
            n = min(len(a_vals), len(b_vals))
            if n == 0:
                continue
            wins = sum(1 for j in range(n) if a_vals[j] > b_vals[j])
            ties = sum(1 for j in range(n) if a_vals[j] == b_vals[j])
            rate = (wins + 0.5 * ties) / n
            result[a][b] = rate
            result.setdefault(b, {})[a] = 1.0 - rate
    return result


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4g}"


# Keep json available for potential report helpers.
_ = json

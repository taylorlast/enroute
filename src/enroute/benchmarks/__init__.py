"""Benchmarks: run environments across models and produce reports.

Examples:
    >>> from enroute.benchmarks import Benchmark, Report
    >>> Benchmark.__name__
    'Benchmark'
"""

from __future__ import annotations

from enroute.benchmarks.runner import Benchmark, Report

__all__ = ["Benchmark", "Report"]

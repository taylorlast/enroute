# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-08-13

### Added

- Initial `enroute` package: unified LLM routing with traces, environments, and benchmarks.
- Native httpx providers for OpenAI-compatible APIs, Anthropic, and Google.
- `Enroute` client with hosted-gateway mode (`ENROUTE_API_KEY`) and BYOK `providers={...}`.
- Routing policies, fallbacks, retries, model catalog, and local cost estimation.
- Tracing with JSONL/SQLite/OTel sinks, redaction, sampling, and late labels.
- Environments (tasks, tools, scorers), versioned datasets, and benchmark reports.
- Docs site (MkDocs) and cookbook-style routing examples.

[Unreleased]: https://github.com/taylorlast/enroute/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/taylorlast/enroute/releases/tag/v0.1.0

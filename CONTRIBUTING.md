# Contributing to enroute

Thanks for helping. Documentation quality is part of the definition of done.

## Setup

```bash
uv sync --group dev --group docs
```

## Checks (same as CI)

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/enroute
uv run pytest -m "not live"
uv run mkdocs build --strict
```

## Module definition of done

A change that adds a public symbol is incomplete until:

1. Google-style docstring with `Args` / `Returns` / `Raises` and a runnable `Examples:` block
2. Doctests pass (`pytest --doctest-modules`)
3. Concept or guide page updated when behavior changes
4. Example under `examples/` still runs

## Tests

- `tests/unit` — pure logic, no network
- `tests/contract` — provider adapters against `respx` fixtures
- `tests/live` — real keys only; marked `@pytest.mark.live`

## Release

Tags matching `v*` publish to PyPI via Trusted Publishing.

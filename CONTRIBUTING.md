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

## Model catalog

`src/enroute/catalog/data/models.json` is the source of truth for model ids,
context windows, and pass-through pricing. It is static so every rate change is
reviewable in git history.

A weekly workflow refreshes it from OpenRouter's public catalog and, when
provider keys are configured as repository secrets, from each provider's own
model listing. It opens a PR rather than pushing to `main`, because a wrong
price costs real money on every request.

Run it yourself with:

```bash
uv run python -m enroute.catalog.sync --check          # report drift, exit 1 if stale
uv run python -m enroute.catalog.sync --write          # apply to models.json
uv run python -m enroute.catalog.sync --add-unserved   # include models no provider confirms
```

A model with no price is deliberately left unpriced. Downstream gateways treat
an unpriced model as unavailable, so it stays hidden until someone fills the
rate in. Models the upstream reference stops listing are reported but never
removed automatically, since dropping one breaks callers.

## Release

Tags matching `v*` publish to PyPI via Trusted Publishing.

Downstream deployments pin a tag, so after merging a catalog PR cut a release
and bump the pinned ref where it is consumed.

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

A weekly workflow refreshes it from OpenRouter's public catalog and opens a PR
rather than pushing to `main`, because a wrong price costs real money on every
request. It uses no secrets: this repository is public, and provider keys belong
to the deployments that consume the library.

Run it yourself with:

```bash
uv run python -m enroute.catalog.sync --check   # report drift, exit 1 if stale
uv run python -m enroute.catalog.sync --write   # apply price changes
uv run python -m enroute.catalog.sync --write --add openai/gpt-5.6-luna
uv run python -m enroute.catalog.sync --write --add-host azure
```

Four rules keep the file trustworthy:

- **Prices are applied automatically.** A stale rate misbills every request, so
  drift is the one thing the job fixes on its own.
- **Prices are list prices, never promotional ones.** OpenRouter's `/models` list
  reports what a caller pays after a discount and omits the discount field, so the
  sync reads `/endpoints` and divides the discount back out. Recording a promo
  would undercharge the moment it expires, and the gap is ours.
- **Additions are deliberate.** The catalog is curated, not a mirror. New models
  are listed in the report as candidates; `--add` brings one in.
- **Removals never happen automatically.** Dropping a model breaks callers, so
  the job reports and leaves it.

A model with no price is left unpriced. Downstream gateways treat an unpriced
model as unavailable, so it stays hidden until someone fills the rate in.

Each `endpoints` entry carries its own rate, because the same model costs
different amounts on different clouds and in different regions. The model-level
price tracks the first endpoint, which is the host routing prefers, so it
advertises what we will actually pay rather than the cheapest listing anywhere.
`--add-host <provider>` pulls a cloud's regional endpoints in for every model that
offers them. An endpoint whose provider has no configured key reports unavailable
and is skipped during routing, so listing a cloud before holding credentials for
it is safe.

Two kinds of upstream endpoint are deliberately excluded from pricing:

- **Service tiers** such as `flex` and `priority` are separate products with their
  own rates and latency characteristics, so one flat number cannot represent both.
- **Quantized deployments** such as `fp8` and `fp4` are cheaper because they are
  smaller, and pricing the full-precision model from them would misrepresent it.

The sync also reports models whose upstream pricing changes above a prompt-token
threshold. This schema stores one flat rate, so those requests bill low and the
difference is absorbed. Fixing it properly needs tiered pricing in the schema.

## Release

Tags matching `v*` publish to PyPI via Trusted Publishing.

Downstream deployments pin a tag, so after merging a catalog PR cut a release
and bump the pinned ref where it is consumed.

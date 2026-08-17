import json

import httpx
import pytest

from enroute.catalog.sync import (
    OPENROUTER_MODELS_URL,
    UpstreamModel,
    apply_diff,
    collect_served_slugs,
    diff_catalog,
    fetch_openrouter,
    normalize_model_id,
    parse_openrouter,
    render_report,
)

CURRENT = {
    "updated_at": "2026-01-01T00:00:00Z",
    "models": [
        {
            "id": "openai/gpt-x",
            "name": "OpenAI: GPT-X",
            "context_length": 128000,
            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        },
        {
            "id": "openai/retired",
            "name": "OpenAI: Retired",
            "context_length": 8192,
            "pricing": {"prompt": "0.000005", "completion": "0.00001"},
        },
        {
            "id": "openai/no-price",
            "name": "OpenAI: No Price",
            "context_length": 8192,
        },
    ],
}

UPSTREAM = {
    "openai/gpt-x": UpstreamModel(
        id="openai/gpt-x", prompt=1e-6, completion=4e-6, name="OpenAI: GPT-X"
    ),
    "openai/no-price": UpstreamModel(id="openai/no-price", name="OpenAI: No Price"),
    "openai/brand-new": UpstreamModel(
        id="openai/brand-new", prompt=2e-6, completion=8e-6, name="OpenAI: Brand New"
    ),
    "acme/unknown-author": UpstreamModel(id="acme/unknown-author", prompt=1e-6, completion=1e-6),
}


def test_normalize_strips_account_paths() -> None:
    assert normalize_model_id("accounts/fireworks/models/Llama-3.1-70B") == "llama-3.1-70b"
    assert normalize_model_id("openai/gpt-4o-mini") == "gpt-4o-mini"
    assert normalize_model_id("GPT-4O") == "gpt-4o"


def test_parse_openrouter_handles_unpriced_models() -> None:
    payload = {
        "data": [
            {
                "id": "openai/priced",
                "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                "context_length": 1000,
            },
            # OpenRouter reports -1 when it cannot price a model.
            {"id": "openai/byok", "pricing": {"prompt": "-1", "completion": "-1"}},
            {"id": "openai/free", "pricing": {"prompt": "0", "completion": "0"}},
        ]
    }
    models = parse_openrouter(payload)
    assert models["openai/priced"].priced is True
    assert models["openai/byok"].priced is False
    assert models["openai/free"].priced is True


def test_diff_detects_price_drift_and_additions() -> None:
    served = {"gpt-x", "brand-new", "no-price"}
    changes = diff_catalog(CURRENT, UPSTREAM, served, providers_checked=["openai"])

    assert [c.id for c in changes.added] == ["openai/brand-new"]
    assert changes.removed == ["openai/retired"]
    assert changes.unpriced == ["openai/no-price"]
    # Only the completion rate moved.
    assert [(c.id, c.field_name, c.new) for c in changes.repriced] == [
        ("openai/gpt-x", "completion", 4e-6)
    ]
    assert changes.empty is False


def test_diff_skips_models_no_provider_serves() -> None:
    changes = diff_catalog(CURRENT, UPSTREAM, {"gpt-x"}, providers_checked=["openai"])
    assert changes.added == []
    # Everything we hold that openai no longer lists is called out.
    assert "openai/no-price" in changes.unconfirmed


def test_diff_can_propose_unserved_models_for_known_authors() -> None:
    changes = diff_catalog(CURRENT, UPSTREAM, set(), add_unserved=True)
    added = [model.id for model in changes.added]
    assert "openai/brand-new" in added
    # acme is not a provider we can route to.
    assert "acme/unknown-author" not in added


def test_apply_writes_prices_and_hides_unpriced_additions() -> None:
    served = {"gpt-x", "brand-new", "no-price"}
    changes = diff_catalog(CURRENT, UPSTREAM, served)
    updated = apply_diff(CURRENT, UPSTREAM, changes)
    entries = {spec["id"]: spec for spec in updated["models"]}

    assert entries["openai/gpt-x"]["pricing"]["completion"] == "0.000004"
    assert entries["openai/brand-new"]["pricing"]["prompt"] == "0.000002"
    # Removals stay put so we never silently break a caller.
    assert "openai/retired" in entries
    assert sorted(entries) == list(entries)


def test_apply_moves_endpoints_that_tracked_the_default_price() -> None:
    current = {
        "models": [
            {
                "id": "openai/gpt-x",
                "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                "endpoints": [
                    # Tracking the default, so it must move.
                    {
                        "provider": "openai",
                        "upstream_id": "gpt-x",
                        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                    },
                    # A cheaper host with its own rate, which must not move.
                    {
                        "provider": "fireworks",
                        "upstream_id": "gpt-x",
                        "pricing": {"prompt": "0.0000005", "completion": "0.000002"},
                    },
                ],
            }
        ]
    }
    upstream = {"openai/gpt-x": UpstreamModel(id="openai/gpt-x", prompt=9e-6, completion=2e-6)}
    changes = diff_catalog(current, upstream, set())
    spec = apply_diff(current, upstream, changes)["models"][0]

    assert spec["pricing"]["prompt"] == "0.000009"
    assert spec["endpoints"][0]["pricing"]["prompt"] == "0.000009"
    assert spec["endpoints"][1]["pricing"]["prompt"] == "0.0000005"


def test_apply_leaves_new_unpriced_models_without_pricing() -> None:
    upstream = {"openai/mystery": UpstreamModel(id="openai/mystery")}
    changes = diff_catalog({"models": []}, upstream, {"mystery"})
    updated = apply_diff({"models": []}, upstream, changes)
    spec = updated["models"][0]
    assert spec["id"] == "openai/mystery"
    assert "pricing" not in spec


def test_apply_keeps_keys_in_canonical_order() -> None:
    current = {"models": [{"id": "openai/gpt-x", "endpoints": [], "name": "GPT-X"}]}
    upstream = {"openai/gpt-x": UpstreamModel(id="openai/gpt-x", prompt=1e-6, completion=2e-6)}
    changes = diff_catalog(current, upstream, set())
    spec = apply_diff(current, upstream, changes)["models"][0]
    assert list(spec) == ["id", "name", "pricing", "endpoints"]


def test_report_flags_unpriced_additions() -> None:
    changes = diff_catalog({"models": []}, {"a/b": UpstreamModel(id="a/b")}, {"b"})
    report = render_report(changes)
    assert "no price, stays hidden" in report


def test_report_is_quiet_when_in_sync() -> None:
    changes = diff_catalog({"models": []}, {}, set())
    assert "already matches upstream" in render_report(changes)


def test_fetch_openrouter_reads_the_public_endpoint() -> None:
    payload = {"data": [{"id": "openai/x", "pricing": {"prompt": "0.1", "completion": "0.2"}}]}

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == OPENROUTER_MODELS_URL
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        models = fetch_openrouter(client)
    assert models["openai/x"].prompt == 0.1


def test_collect_served_slugs_skips_providers_without_keys() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(200, json={"data": [{"id": "gpt-x"}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        served, checked = collect_served_slugs(client, env={"OPENAI_API_KEY": "test-key"})
    assert served == {"gpt-x"}
    assert checked == ["openai"]


def test_collect_served_slugs_tolerates_provider_outages() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        served, checked = collect_served_slugs(client, env={"OPENAI_API_KEY": "test-key"})
    assert served == set()
    assert checked == []


@pytest.mark.parametrize("slug", ["anthropic", "google"])
def test_collect_served_slugs_handles_non_openai_shapes(slug: str) -> None:
    env_name = {"anthropic": "ANTHROPIC_API_KEY", "google": "GOOGLE_API_KEY"}[slug]
    body = {
        "anthropic": {"data": [{"id": "claude-x"}]},
        "google": {"models": [{"name": "models/gemini-x"}]},
    }[slug]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        served, checked = collect_served_slugs(client, env={env_name: "test-key"})
    assert checked == [slug]
    assert served == {"claude-x"} if slug == "anthropic" else served == {"gemini-x"}


def test_bundled_catalog_round_trips() -> None:
    from enroute.catalog.sync import load_catalog

    document = load_catalog()
    assert document["models"]
    # The file we ship must stay valid JSON with sorted, unique ids.
    ids = [spec["id"] for spec in document["models"]]
    assert len(ids) == len(set(ids))
    assert json.dumps(document)

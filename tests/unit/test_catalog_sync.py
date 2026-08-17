import json

import httpx
import pytest

from enroute.catalog.sync import (
    OPENROUTER_MODELS_URL,
    UpstreamEndpoint,
    UpstreamModel,
    UpstreamTier,
    apply_diff,
    collect_served_slugs,
    diff_catalog,
    fetch_openrouter,
    normalize_model_id,
    normalize_region,
    parse_endpoint_tag,
    parse_endpoints,
    parse_openrouter,
    parse_tiers,
    render_report,
    standard_endpoints,
)


def host(
    provider: str | None = "openai",
    *,
    region: str = "us",
    tier: str = "standard",
    prompt: float | None = 1e-6,
    completion: float | None = 2e-6,
    discount: float | None = None,
    tiers: tuple[UpstreamTier, ...] = (),
    upstream_id: str = "gpt-x",
) -> UpstreamEndpoint:
    return UpstreamEndpoint(
        provider=provider,
        region=region,
        tier=tier,
        upstream_id=upstream_id,
        prompt=prompt,
        completion=completion,
        discount=discount,
        tiers=tiers,
        tag=f"{provider}/{region}",
    )


CURRENT = {
    "updated_at": "2026-01-01T00:00:00Z",
    "models": [
        {
            "id": "openai/gpt-x",
            "name": "OpenAI: GPT-X",
            "context_length": 128000,
            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
            "endpoints": [
                {
                    "provider": "openai",
                    "region": "us",
                    "upstream_id": "gpt-x",
                    "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                }
            ],
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

# gpt-x completion moved; no-price has no hosts; retired is gone upstream.
ENDPOINTS = {
    "openai/gpt-x": [host(completion=4e-6)],
    "openai/no-price": [],
    "openai/retired": [],
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


def test_endpoint_tags_split_into_provider_region_and_tier() -> None:
    assert parse_endpoint_tag("openai") == ("openai", "us", "standard")
    assert parse_endpoint_tag("openai/flex") == ("openai", "us", "flex")
    assert parse_endpoint_tag("azure/eu") == ("azure", "eu", "standard")
    assert parse_endpoint_tag("amazon-bedrock/us-east-1") == ("bedrock", "us", "standard")
    assert parse_endpoint_tag("google-vertex/global/priority") == ("vertex", "global", "priority")
    assert parse_endpoint_tag("google-ai-studio") == ("google", "us", "standard")
    # An unmapped vendor is reported rather than guessed at.
    assert parse_endpoint_tag("some-new-cloud")[0] is None


def test_quantized_and_program_variants_are_not_standard() -> None:
    # A 4-bit deployment is a different product and must not set the model's rate.
    assert parse_endpoint_tag("baseten/fp4") == ("baseten", "us", "fp4")
    assert parse_endpoint_tag("together/fp8")[2] == "fp8"
    assert parse_endpoint_tag("amazon-bedrock/claude-on-aws") == (
        "bedrock",
        "us",
        "claude-on-aws",
    )
    hosts = [host("baseten", tier="fp4", prompt=1e-9), host("baseten", prompt=5e-6)]
    assert standard_endpoints(hosts)[("baseten", "us")].prompt == 5e-6


def test_normalize_region_collapses_vendor_labels() -> None:
    assert normalize_region("us-east-1") == "us"
    assert normalize_region("europe-west4") == "eu"
    assert normalize_region("global") == "global"


def test_parse_endpoints_records_list_price_not_promo_price() -> None:
    # This is the bug the rework exists to prevent: /models reports 0.0000025 with
    # no discount field, while the real OpenAI list rate is 0.000005.
    payload = {
        "data": {
            "id": "openai/gpt-5.6-sol",
            "endpoints": [
                {
                    "name": "OpenAI | openai/gpt-5.6-sol-20260709",
                    "tag": "openai",
                    "pricing": {
                        "prompt": "0.0000025",
                        "completion": "0.000015",
                        "discount": 0.5,
                        "overrides": [{"min_prompt_tokens": 272000, "prompt": "0.000005"}],
                    },
                },
                {
                    "name": "Azure | openai/gpt-5.6-sol",
                    "tag": "azure/eu",
                    "pricing": {"prompt": "0.0000055", "completion": "0.000033"},
                },
            ],
        }
    }
    endpoints = parse_endpoints(payload)
    openai_host, azure_host = endpoints

    assert openai_host.prompt == 5e-6
    assert openai_host.completion == 3e-5
    assert openai_host.discount == 0.5
    assert openai_host.upstream_id == "gpt-5.6-sol-20260709"
    # The override lacked a completion rate, so it cannot be billed as a tier.
    assert openai_host.tiers == ()
    # An undiscounted host is recorded as-is.
    assert azure_host.provider == "azure"
    assert azure_host.region == "eu"
    assert azure_host.prompt == 5.5e-6


def test_standard_endpoints_excludes_service_tiers() -> None:
    hosts = [
        host(prompt=5e-6),
        host(tier="flex", prompt=2.5e-6),
        host(tier="priority", prompt=1e-5),
        host(None, prompt=1e-9),
        host("azure", prompt=None, completion=None),
    ]
    indexed = standard_endpoints(hosts)
    # Flex is genuinely cheaper but is a different service level, and an unmapped
    # or unpriced host cannot be billed.
    assert list(indexed) == [("openai", "us")]
    assert indexed[("openai", "us")].prompt == 5e-6


def test_parse_tiers_undiscounts_and_orders_thresholds() -> None:
    pricing = {
        "overrides": [
            {"min_prompt_tokens": 500000, "prompt": "0.000008", "completion": "0.00004"},
            {"min_prompt_tokens": 272000, "prompt": "0.0000025", "completion": "0.00001125"},
            # No completion rate, so this one cannot be billed.
            {"min_prompt_tokens": 100, "prompt": "0.000001"},
        ]
    }
    tiers = parse_tiers(pricing, 0.5)
    assert [t.min_prompt_tokens for t in tiers] == [272000, 500000]
    assert tiers[0].prompt == 5e-6
    assert tiers[0].completion == 2.25e-5


def test_diff_flags_promotions() -> None:
    endpoints = {"openai/gpt-x": [host(discount=0.5)]}
    changes = diff_catalog(CURRENT, UPSTREAM, set(), endpoints_by_model=endpoints)
    assert changes.discounted == ["openai/gpt-x"]
    assert "Running a promotion upstream" in render_report(changes)


def test_prompt_length_tiers_are_recorded_per_host() -> None:
    tier = UpstreamTier(min_prompt_tokens=272000, prompt=1e-5, completion=6e-5)
    endpoints = {"openai/gpt-x": [host(tiers=(tier,))]}
    changes = diff_catalog(CURRENT, UPSTREAM, set(), endpoints_by_model=endpoints)

    assert [(c.id, c.host) for c in changes.retiered] == [
        ("openai/gpt-x", "openai/us"),
        ("openai/gpt-x", "default"),
    ]
    assert changes.has_updates is True
    assert "above 272,000 tokens" in render_report(changes)

    spec = next(
        m for m in apply_diff(CURRENT, UPSTREAM, changes)["models"] if m["id"] == "openai/gpt-x"
    )
    assert spec["pricing"]["tiers"] == [
        {"min_prompt_tokens": 272000, "prompt": "0.00001", "completion": "0.00006"}
    ]
    assert spec["endpoints"][0]["pricing"]["tiers"][0]["min_prompt_tokens"] == 272000


def test_dropped_tiers_are_removed_not_left_stale() -> None:
    current = {
        "models": [
            {
                "id": "openai/gpt-x",
                "pricing": {
                    "prompt": "0.000001",
                    "completion": "0.000002",
                    "tiers": [
                        {
                            "min_prompt_tokens": 272000,
                            "prompt": "0.00001",
                            "completion": "0.00006",
                        }
                    ],
                },
                "endpoints": [
                    {
                        "provider": "openai",
                        "region": "us",
                        "upstream_id": "gpt-x",
                        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                    }
                ],
            }
        ]
    }
    endpoints = {"openai/gpt-x": [host()]}
    changes = diff_catalog(current, UPSTREAM, set(), endpoints_by_model=endpoints)
    spec = apply_diff(current, UPSTREAM, changes)["models"][0]
    assert "tiers" not in spec["pricing"]


def test_diff_reports_hosts_we_do_not_list() -> None:
    endpoints = {
        "openai/gpt-x": [
            host(),
            host("azure", region="eu"),
            host("bedrock", region="us"),
        ]
    }
    changes = diff_catalog(CURRENT, UPSTREAM, set(), endpoints_by_model=endpoints)
    assert changes.new_hosts == [("openai/gpt-x", "azure/eu"), ("openai/gpt-x", "bedrock/us")]
    # Reporting a host must not silently add or reprice it.
    spec = apply_diff(CURRENT, UPSTREAM, changes)["models"][0]
    assert len(spec.get("endpoints", [])) <= 1


def test_add_host_brings_in_regional_endpoints_with_their_own_rates() -> None:
    endpoints = {
        "openai/gpt-x": [
            host(),
            host("azure", region="us", prompt=5e-6, completion=3e-5, upstream_id="gpt-x-dep"),
            host("azure", region="eu", prompt=5.5e-6, completion=3.3e-5),
            host("bedrock", region="us", prompt=5.5e-6, completion=3.3e-5),
        ]
    }
    changes = diff_catalog(CURRENT, UPSTREAM, set(), endpoints_by_model=endpoints)
    updated = apply_diff(
        CURRENT,
        UPSTREAM,
        changes,
        add_hosts=["azure"],
        endpoints_by_model=endpoints,
    )
    spec = next(m for m in updated["models"] if m["id"] == "openai/gpt-x")
    hosts = [(e["provider"], e["region"]) for e in spec["endpoints"]]

    # Added in a deterministic order; ordered_endpoints() still prefers US at read time.
    assert hosts == [("openai", "us"), ("azure", "eu"), ("azure", "us")]
    # Each region keeps its own rate; EU is more expensive than US.
    azure_eu = next(e for e in spec["endpoints"] if e["region"] == "eu")
    assert azure_eu["pricing"]["prompt"] == "0.0000055"
    assert next(e for e in spec["endpoints"] if e["provider"] == "azure")["upstream_id"]


def test_add_host_is_idempotent() -> None:
    endpoints = {"openai/gpt-x": [host(), host("azure", region="eu")]}
    changes = diff_catalog(CURRENT, UPSTREAM, set(), endpoints_by_model=endpoints)
    once = apply_diff(CURRENT, UPSTREAM, changes, add_hosts=["azure"], endpoints_by_model=endpoints)
    twice = apply_diff(once, UPSTREAM, changes, add_hosts=["azure"], endpoints_by_model=endpoints)
    first = next(m for m in once["models"] if m["id"] == "openai/gpt-x")["endpoints"]
    second = next(m for m in twice["models"] if m["id"] == "openai/gpt-x")["endpoints"]
    assert len(first) == len(second) == 2


def test_diff_says_nothing_when_the_endpoint_lookup_fails() -> None:
    # A failed HTTP call must not read as "upstream dropped every price".
    changes = diff_catalog(CURRENT, UPSTREAM, set(), endpoints_by_model={})
    assert changes.repriced == []
    assert changes.has_updates is False


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


def test_diff_detects_price_drift_and_candidates() -> None:
    served = {"gpt-x", "brand-new", "no-price"}
    changes = diff_catalog(
        CURRENT, UPSTREAM, served, providers_checked=["openai"], endpoints_by_model=ENDPOINTS
    )

    assert [c.id for c in changes.candidates] == ["openai/brand-new"]
    assert changes.removed == ["openai/no-price", "openai/retired"]
    assert changes.unpriced == ["openai/no-price"]
    # Only completion moved, on the host and on the model default that tracks it.
    assert [(c.id, c.host, c.field_name, c.new) for c in changes.repriced] == [
        ("openai/gpt-x", "openai/us", "completion", 4e-6),
        ("openai/gpt-x", "default", "completion", 4e-6),
    ]
    assert changes.has_updates is True


def test_diff_skips_candidates_no_provider_serves() -> None:
    changes = diff_catalog(CURRENT, UPSTREAM, {"gpt-x"}, providers_checked=["openai"])
    assert changes.candidates == []
    # Everything we hold that openai no longer lists is called out.
    assert "openai/no-price" in changes.unconfirmed


def test_diff_proposes_only_routable_authors_when_unconfirmed() -> None:
    changes = diff_catalog(CURRENT, UPSTREAM, set())
    ids = [model.id for model in changes.candidates]
    assert "openai/brand-new" in ids
    # acme is not a provider we have an adapter for.
    assert "acme/unknown-author" not in ids


def test_apply_reprices_without_adding_candidates() -> None:
    changes = diff_catalog(CURRENT, UPSTREAM, set(), endpoints_by_model=ENDPOINTS)
    updated = apply_diff(CURRENT, UPSTREAM, changes)
    entries = {spec["id"]: spec for spec in updated["models"]}

    assert entries["openai/gpt-x"]["pricing"]["completion"] == "0.000004"
    assert entries["openai/gpt-x"]["endpoints"][0]["pricing"]["completion"] == "0.000004"
    # A curated catalog does not grow on its own.
    assert "openai/brand-new" not in entries
    # Removals stay put so we never silently break a caller.
    assert "openai/retired" in entries
    assert sorted(entries) == list(entries)


def test_apply_adds_only_requested_models() -> None:
    changes = diff_catalog(CURRENT, UPSTREAM, set())
    updated = apply_diff(CURRENT, UPSTREAM, changes, add=["openai/brand-new"])
    entries = {spec["id"]: spec for spec in updated["models"]}
    assert entries["openai/brand-new"]["pricing"]["prompt"] == "0.000002"
    assert "acme/unknown-author" not in entries


def test_apply_reprices_only_the_host_that_moved() -> None:
    current = {
        "models": [
            {
                "id": "openai/gpt-x",
                "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                "endpoints": [
                    {
                        "provider": "openai",
                        "region": "us",
                        "upstream_id": "gpt-x",
                        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                    },
                    {
                        "provider": "azure",
                        "region": "eu",
                        "upstream_id": "gpt-x",
                        "pricing": {"prompt": "0.0000011", "completion": "0.0000022"},
                    },
                ],
            }
        ]
    }
    upstream = {"openai/gpt-x": UpstreamModel(id="openai/gpt-x")}
    endpoints = {
        "openai/gpt-x": [
            host(prompt=9e-6, completion=2e-6),
            host("azure", region="eu", prompt=1.1e-6, completion=2.2e-6),
        ]
    }
    changes = diff_catalog(current, upstream, set(), endpoints_by_model=endpoints)
    spec = apply_diff(current, upstream, changes)["models"][0]

    assert spec["endpoints"][0]["pricing"]["prompt"] == "0.000009"
    # Azure did not move, so its rate is untouched.
    assert spec["endpoints"][1]["pricing"]["prompt"] == "0.0000011"
    # The default tracks the first host.
    assert spec["pricing"]["prompt"] == "0.000009"


def test_apply_leaves_new_unpriced_models_without_pricing() -> None:
    upstream = {"openai/mystery": UpstreamModel(id="openai/mystery")}
    changes = diff_catalog({"models": []}, upstream, {"mystery"})
    updated = apply_diff({"models": []}, upstream, changes, add=["openai/mystery"])
    spec = updated["models"][0]
    assert spec["id"] == "openai/mystery"
    # No price means downstream keeps it hidden rather than serving it for free.
    assert "pricing" not in spec


def test_curated_catalog_does_not_grow_on_a_schedule() -> None:
    # The real risk: hundreds of upstream models silently landing in the catalog.
    upstream = {f"openai/model-{i}": UpstreamModel(id=f"openai/model-{i}") for i in range(50)}
    changes = diff_catalog({"models": []}, upstream, set())
    assert len(changes.candidates) == 50
    assert changes.has_updates is False
    assert apply_diff({"models": []}, upstream, changes)["models"] == []


def test_apply_keeps_keys_in_canonical_order() -> None:
    current = {"models": [{"id": "openai/gpt-x", "endpoints": [], "name": "GPT-X"}]}
    upstream = {"openai/gpt-x": UpstreamModel(id="openai/gpt-x")}
    endpoints = {"openai/gpt-x": [host()]}
    changes = diff_catalog(current, upstream, set(), endpoints_by_model=endpoints)
    spec = apply_diff(current, upstream, changes)["models"][0]
    assert list(spec) == ["id", "name", "pricing", "endpoints"]


def test_report_lists_candidates_without_claiming_they_were_added() -> None:
    upstream = {"openai/new-thing": UpstreamModel(id="openai/new-thing")}
    report = render_report(diff_catalog({"models": []}, upstream, set()))
    assert "not carried" in report
    assert "`openai/new-thing`" in report
    assert "no upstream price" in report


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

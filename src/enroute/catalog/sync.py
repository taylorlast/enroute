"""Refresh the bundled model catalog from upstream sources.

The catalog ships as static JSON so pricing changes are reviewable in git.

Prices come from OpenRouter's per-model ``/endpoints`` response rather than its
``/models`` list. The list response reports the price a caller pays *after* any
promotion and omits the ``discount`` field entirely, so reading it would bake a
temporary discount in as if it were the standard rate. When the promotion ends,
the catalog would still bill the discounted price while the provider charges
full, and the difference comes out of our margin. ``/endpoints`` exposes
``discount``, which lets us record the undiscounted list price.

Only standard-tier endpoints set pricing. Flex and priority are separate service
levels with their own rates, so folding them into one number would misprice
whichever tier we did not pick.

Pricing is pass-through, so a model with no price must never go live. Anything we
cannot price is left unpriced, which keeps it unavailable downstream.

Run it with::

    python -m enroute.catalog.sync --check
    python -m enroute.catalog.sync --write --report catalog-report.md
    python -m enroute.catalog.sync --write --add openai/gpt-5.6-luna
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from enroute.catalog.models import normalize_region

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
DATA_PATH = Path(__file__).resolve().parent / "data" / "models.json"

# Mirrors enroute.client env handling so the sync sees the same providers.
PROVIDER_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "together": "TOGETHER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "baseten": "BASETEN_API_KEY",
    "xai": "XAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "meta": "META_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "zhipu": "ZAI_API_KEY",
}

OPENAI_COMPATIBLE_BASES: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "baseten": "https://inference.baseten.co/v1",
    "xai": "https://api.x.ai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "mistral": "https://api.mistral.ai/v1",
    "meta": "https://api.meta.ai/v1",
    "moonshot": "https://api.moonshot.ai/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "zhipu": "https://api.z.ai/api/paas/v4",
}

ANTHROPIC_MODELS_URL = "https://api.anthropic.com/v1/models"
ANTHROPIC_VERSION = "2023-06-01"
GOOGLE_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# OpenRouter endpoint tags look like ``provider[/qualifier...]``, where a
# qualifier is either a service tier or a region.
PROVIDER_TAG_MAP: dict[str, str] = {
    "openai": "openai",
    "azure": "azure",
    "amazon-bedrock": "bedrock",
    "anthropic": "anthropic",
    "google-vertex": "vertex",
    "google-ai-studio": "google",
    "google": "google",
    "fireworks": "fireworks",
    "baseten": "baseten",
    "together": "together",
    "groq": "groq",
    "deepseek": "deepseek",
    "mistral": "mistral",
    "moonshot": "moonshot",
    "xai": "xai",
    "qwen": "qwen",
    "alibaba": "qwen",
    "zhipu": "zhipu",
    "z-ai": "zhipu",
    "meta": "meta",
}
SERVICE_TIERS = frozenset({"flex", "priority", "batch", "scale"})
STANDARD_TIER = "standard"
_REGION_PREFIXES = ("us", "eu", "europe", "global", "ap", "asia", "sa", "ca", "me", "af")
_REGION_PATTERN = re.compile(r"^[a-z]{2}-[a-z]+-?\d*$")


def normalize_model_id(model_id: str) -> str:
    """Reduce a vendor model id to a comparable slug.

    Vendors name the same weights differently: Fireworks serves
    ``accounts/fireworks/models/llama-v3p1-70b`` where OpenRouter says
    ``meta/llama-3.1-70b``. Comparing bare trailing slugs catches most pairs.

    Args:
        model_id: Provider or catalog model id.

    Returns:
        A lowercase slug with author and account path segments removed.

    Examples:
        >>> normalize_model_id("accounts/fireworks/models/Llama-3.1-70B")
        'llama-3.1-70b'
        >>> normalize_model_id("openai/gpt-4o-mini")
        'gpt-4o-mini'
    """
    slug = model_id.strip().lower()
    if "/" in slug:
        slug = slug.rsplit("/", 1)[1]
    return slug


def _price(raw: Any) -> float | None:
    """Parse an OpenRouter price string into USD per token.

    Args:
        raw: Price field from the upstream payload.

    Returns:
        Price per token, or ``None`` when the vendor reports it as unknown.
    """
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    # OpenRouter uses -1 for models it cannot price (BYOK or request-scoped).
    if value < 0:
        return None
    return value


@dataclass(frozen=True)
class UpstreamModel:
    """A model as described by the upstream reference catalog.

    Attributes:
        id: Canonical ``author/slug`` id.
        name: Display name.
        context_length: Context window in tokens.
        prompt: USD per prompt token, or ``None`` when unknown.
        completion: USD per completion token, or ``None`` when unknown.
        modality: High-level modality string.
        input_modalities: Accepted input modalities.
        output_modalities: Produced output modalities.
        supported_parameters: Request parameters the model accepts.
    """

    id: str
    name: str | None = None
    context_length: int | None = None
    prompt: float | None = None
    completion: float | None = None
    modality: str | None = None
    input_modalities: tuple[str, ...] = ("text",)
    output_modalities: tuple[str, ...] = ("text",)
    supported_parameters: tuple[str, ...] = ()

    @property
    def author(self) -> str:
        """Author segment of the model id.

        Returns:
            The portion before the first slash, or the whole id.
        """
        return self.id.split("/", 1)[0] if "/" in self.id else self.id

    @property
    def priced(self) -> bool:
        """Whether both token prices are known.

        Returns:
            ``True`` when the model can be billed pass-through.
        """
        return self.prompt is not None and self.completion is not None


def is_region(qualifier: str) -> bool:
    """Decide whether an endpoint tag qualifier names a region.

    Args:
        qualifier: A segment after the provider in an endpoint tag.

    Returns:
        ``True`` when the segment looks like a region rather than a variant.

    Examples:
        >>> is_region("us-east-1"), is_region("global"), is_region("fp8")
        (True, True, False)
    """
    lowered = qualifier.lower()
    if lowered.startswith(_REGION_PREFIXES):
        return True
    return bool(_REGION_PATTERN.match(lowered))


def parse_endpoint_tag(tag: str) -> tuple[str | None, str, str]:
    """Split an OpenRouter endpoint tag into provider, region, and service tier.

    A qualifier that is neither a known service tier nor region-shaped is a
    deployment variant: quantizations such as ``fp8`` and ``fp4``, or program
    names such as ``claude-on-aws``. Those are returned as the tier so they are
    excluded from pricing, since a 4-bit deployment is not the same product as
    the full-precision model and must not set its rate.

    Args:
        tag: Endpoint tag such as ``azure/eu`` or ``openai/flex``.

    Returns:
        A triple of (provider slug or ``None`` when unmapped, region, tier).

    Examples:
        >>> parse_endpoint_tag("amazon-bedrock/us-east-1")
        ('bedrock', 'us', 'standard')
        >>> parse_endpoint_tag("openai/flex")
        ('openai', 'us', 'flex')
        >>> parse_endpoint_tag("baseten/fp4")
        ('baseten', 'us', 'fp4')
    """
    parts = [part for part in tag.split("/") if part]
    if not parts:
        return None, "us", STANDARD_TIER
    provider = PROVIDER_TAG_MAP.get(parts[0])
    region = "us"
    tier = STANDARD_TIER
    for qualifier in parts[1:]:
        if qualifier in SERVICE_TIERS or not is_region(qualifier):
            tier = qualifier
        else:
            region = normalize_region(qualifier)
    return provider, region, tier


@dataclass(frozen=True)
class UpstreamTier:
    """A long-prompt rate that supersedes the base rate.

    Attributes:
        min_prompt_tokens: Prompt size at which the rate takes over.
        prompt: Undiscounted USD per prompt token.
        completion: Undiscounted USD per completion token.
    """

    min_prompt_tokens: int
    prompt: float
    completion: float


@dataclass(frozen=True)
class UpstreamEndpoint:
    """A single inference host as OpenRouter describes it.

    Attributes:
        provider: Catalog provider slug, or ``None`` when the tag is unmapped.
        region: Coarse region key.
        tier: Service tier. Only ``standard`` is used for catalog pricing.
        upstream_id: Model id the host expects.
        prompt: Undiscounted USD per prompt token.
        completion: Undiscounted USD per completion token.
        discount: Promotional fraction OpenRouter applied, if any.
        tiers: Prompt-length rates, ordered by threshold.
        tag: Raw upstream tag, kept for reporting unmapped hosts.
    """

    provider: str | None
    region: str
    tier: str
    upstream_id: str
    prompt: float | None
    completion: float | None
    discount: float | None
    tiers: tuple[UpstreamTier, ...]
    tag: str

    @property
    def priced(self) -> bool:
        """Whether both token prices are known.

        Returns:
            ``True`` when the endpoint can be billed pass-through.
        """
        return self.prompt is not None and self.completion is not None


def _undiscount(rate: float | None, discount: float | None) -> float | None:
    """Recover a list price from a discounted rate.

    Args:
        rate: Price a caller currently pays.
        discount: Fraction taken off the list price.

    Returns:
        The undiscounted list price.

    Examples:
        >>> _undiscount(2.5, 0.5)
        5.0
        >>> _undiscount(2.5, None)
        2.5
    """
    if rate is None:
        return None
    if not discount or discount >= 1:
        return rate
    return rate / (1 - discount)


def parse_tiers(pricing: Mapping[str, Any], discount: float | None) -> tuple[UpstreamTier, ...]:
    """Read prompt-length rates from an endpoint's pricing block.

    A tier is only usable if both rates are present, since billing one side at the
    long-prompt rate and the other at the base rate matches neither.

    Args:
        pricing: The endpoint's ``pricing`` mapping.
        discount: Promotional fraction to divide back out.

    Returns:
        Tiers ordered by threshold.
    """
    tiers: list[UpstreamTier] = []
    for override in pricing.get("overrides") or []:
        threshold = override.get("min_prompt_tokens")
        prompt = _undiscount(_price(override.get("prompt")), discount)
        completion = _undiscount(_price(override.get("completion")), discount)
        if threshold is None or prompt is None or completion is None:
            continue
        tiers.append(
            UpstreamTier(min_prompt_tokens=int(threshold), prompt=prompt, completion=completion)
        )
    tiers.sort(key=lambda tier: tier.min_prompt_tokens)
    return tuple(tiers)


def parse_endpoints(payload: Mapping[str, Any]) -> list[UpstreamEndpoint]:
    """Convert an OpenRouter ``/endpoints`` payload into host records.

    Args:
        payload: Decoded ``/api/v1/models/{id}/endpoints`` response.

    Returns:
        Every host described upstream, with list rather than promotional prices.
    """
    data = payload.get("data") or {}
    model_id = data.get("id") or ""
    fallback_id = model_id.split("/", 1)[1] if "/" in model_id else model_id
    endpoints: list[UpstreamEndpoint] = []
    for entry in data.get("endpoints") or []:
        tag = entry.get("tag") or ""
        provider, region, tier = parse_endpoint_tag(tag)
        pricing = entry.get("pricing") or {}
        discount = pricing.get("discount")
        name = entry.get("name") or ""
        upstream_id = name.split("|", 1)[1].strip() if "|" in name else fallback_id
        if "/" in upstream_id:
            upstream_id = upstream_id.rsplit("/", 1)[1]
        endpoints.append(
            UpstreamEndpoint(
                provider=provider,
                region=region,
                tier=tier,
                upstream_id=upstream_id or fallback_id,
                prompt=_undiscount(_price(pricing.get("prompt")), discount),
                completion=_undiscount(_price(pricing.get("completion")), discount),
                discount=discount if isinstance(discount, (int, float)) else None,
                tiers=parse_tiers(pricing, discount),
                tag=tag,
            )
        )
    return endpoints


def fetch_endpoints(client: httpx.Client, model_id: str) -> list[UpstreamEndpoint]:
    """Fetch the hosts OpenRouter lists for one model.

    Args:
        client: HTTP client to use.
        model_id: Canonical ``author/slug`` id.

    Returns:
        Host records, empty when the model is unknown upstream.
    """
    try:
        response = client.get(f"{OPENROUTER_MODELS_URL}/{model_id}/endpoints")
        response.raise_for_status()
    except httpx.HTTPError:
        return []
    return parse_endpoints(response.json())


def standard_endpoints(
    endpoints: Iterable[UpstreamEndpoint],
) -> dict[tuple[str, str], UpstreamEndpoint]:
    """Index priced standard-tier hosts by provider and region.

    Args:
        endpoints: Host records for one model.

    Returns:
        Mapping of (provider, region) to endpoint, cheapest kept on collision.
    """
    indexed: dict[tuple[str, str], UpstreamEndpoint] = {}
    for endpoint in endpoints:
        if endpoint.provider is None or endpoint.tier != STANDARD_TIER:
            continue
        if not endpoint.priced:
            continue
        key = (endpoint.provider, endpoint.region)
        current = indexed.get(key)
        if current is None or (endpoint.prompt or 0) < (current.prompt or 0):
            indexed[key] = endpoint
    return indexed


@dataclass(frozen=True)
class PriceChange:
    """A per-token price that moved upstream.

    Attributes:
        id: Model id.
        field_name: Which rate changed (``prompt`` or ``completion``).
        old: Price currently in the catalog.
        new: Undiscounted list price reported upstream.
        provider: Host the rate belongs to, or ``None`` for the model default.
        region: Host region, or ``None`` for the model default.
    """

    id: str
    field_name: str
    old: float | None
    new: float | None
    provider: str | None = None
    region: str | None = None

    @property
    def host(self) -> str:
        """Human-readable host label.

        Returns:
            ``provider/region``, or ``default`` for the model-level price.
        """
        if self.provider is None:
            return "default"
        return f"{self.provider}/{self.region}"


@dataclass(frozen=True)
class TierChange:
    """Prompt-length rates that differ from what the catalog records.

    Attributes:
        id: Model id.
        tiers: Tiers reported upstream, at list prices.
        provider: Host the tiers belong to, or ``None`` for the model default.
        region: Host region, or ``None`` for the model default.
    """

    id: str
    tiers: tuple[UpstreamTier, ...]
    provider: str | None = None
    region: str | None = None

    @property
    def host(self) -> str:
        """Human-readable host label.

        Returns:
            ``provider/region``, or ``default`` for the model-level price.
        """
        if self.provider is None:
            return "default"
        return f"{self.provider}/{self.region}"

    def describe(self) -> str:
        """Summarize the thresholds and rates for a review body.

        Returns:
            A human-readable description, or a note that tiers were dropped.
        """
        if not self.tiers:
            return "no prompt-length pricing"
        return ", ".join(
            f"above {tier.min_prompt_tokens:,} tokens: {tier.prompt}/{tier.completion}"
            for tier in self.tiers
        )


@dataclass
class CatalogDiff:
    """What the sync found relative to the bundled catalog.

    Attributes:
        candidates: Upstream models we could carry but do not. Reported only; the
            catalog is curated, so adding one is an explicit choice.
        removed: Catalog ids the upstream reference no longer lists.
        repriced: Rates that moved upstream.
        unpriced: Catalog ids with no usable price, which stay hidden.
        unconfirmed: Catalog ids no configured provider currently lists.
        providers_checked: Provider slugs whose listings were fetched.
        discounted: Model ids where upstream is running a promotion. Recorded so a
            reviewer can see we deliberately kept the list price.
        retiered: Prompt-length rates that differ from the catalog's.
        new_hosts: Hosts upstream offers for models we carry but we do not list.
    """

    candidates: list[UpstreamModel] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    repriced: list[PriceChange] = field(default_factory=list)
    unpriced: list[str] = field(default_factory=list)
    unconfirmed: list[str] = field(default_factory=list)
    providers_checked: list[str] = field(default_factory=list)
    discounted: list[str] = field(default_factory=list)
    retiered: list[TierChange] = field(default_factory=list)
    new_hosts: list[tuple[str, str]] = field(default_factory=list)

    @property
    def has_updates(self) -> bool:
        """Whether anything would be written without an explicit request.

        Returns:
            ``True`` when a rate moved upstream.
        """
        return bool(self.repriced or self.retiered)

    @property
    def empty(self) -> bool:
        """Whether there is nothing at all worth reporting.

        Returns:
            ``True`` when the catalog matches upstream and nothing is pending.
        """
        return not (self.candidates or self.removed or self.repriced or self.retiered)


def parse_openrouter(payload: Mapping[str, Any]) -> dict[str, UpstreamModel]:
    """Convert an OpenRouter models payload into upstream records.

    Args:
        payload: Decoded ``/api/v1/models`` response.

    Returns:
        Upstream models keyed by canonical id.
    """
    models: dict[str, UpstreamModel] = {}
    for entry in payload.get("data") or []:
        model_id = entry.get("id")
        if not model_id:
            continue
        pricing = entry.get("pricing") or {}
        architecture = entry.get("architecture") or {}
        models[model_id] = UpstreamModel(
            id=model_id,
            name=entry.get("name"),
            context_length=entry.get("context_length"),
            prompt=_price(pricing.get("prompt")),
            completion=_price(pricing.get("completion")),
            modality=architecture.get("modality"),
            input_modalities=tuple(architecture.get("input_modalities") or ["text"]),
            output_modalities=tuple(architecture.get("output_modalities") or ["text"]),
            supported_parameters=tuple(entry.get("supported_parameters") or ()),
        )
    return models


def fetch_openrouter(client: httpx.Client) -> dict[str, UpstreamModel]:
    """Fetch the public OpenRouter catalog.

    Args:
        client: HTTP client to use.

    Returns:
        Upstream models keyed by canonical id.
    """
    response = client.get(OPENROUTER_MODELS_URL)
    response.raise_for_status()
    return parse_openrouter(response.json())


def fetch_provider_models(client: httpx.Client, slug: str, api_key: str) -> set[str]:
    """List the model slugs a provider currently serves.

    Args:
        client: HTTP client to use.
        slug: Provider slug.
        api_key: Provider API key.

    Returns:
        Normalized slugs the provider reports, empty when the call fails.
    """
    try:
        if slug == "anthropic":
            response = client.get(
                ANTHROPIC_MODELS_URL,
                headers={"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION},
            )
        elif slug == "google":
            response = client.get(GOOGLE_MODELS_URL, params={"key": api_key})
        else:
            base = OPENAI_COMPATIBLE_BASES.get(slug)
            if base is None:
                return set()
            response = client.get(f"{base}/models", headers={"Authorization": f"Bearer {api_key}"})
        response.raise_for_status()
    except httpx.HTTPError:
        return set()
    body = response.json()
    entries = body.get("data") or body.get("models") or []
    slugs: set[str] = set()
    for entry in entries:
        raw = entry.get("id") or entry.get("name") if isinstance(entry, dict) else entry
        if isinstance(raw, str) and raw:
            slugs.add(normalize_model_id(raw))
    return slugs


def collect_served_slugs(
    client: httpx.Client, env: Mapping[str, str] | None = None
) -> tuple[set[str], list[str]]:
    """Gather the slugs every configured provider reports.

    Args:
        client: HTTP client to use.
        env: Environment mapping to read keys from. Defaults to ``os.environ``.

    Returns:
        A pair of (normalized slugs, provider slugs that answered).
    """
    source = os.environ if env is None else env
    served: set[str] = set()
    checked: list[str] = []
    for slug, env_name in PROVIDER_ENV.items():
        api_key = source.get(env_name)
        if not api_key:
            continue
        listed = fetch_provider_models(client, slug, api_key)
        if listed:
            checked.append(slug)
            served |= listed
    return served, checked


def _catalog_prices(spec: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """Read the default prices off a catalog entry.

    Args:
        spec: A single entry from ``models.json``.

    Returns:
        A pair of (prompt, completion) prices per token.
    """
    pricing = spec.get("pricing") or {}
    return _price(pricing.get("prompt")), _price(pricing.get("completion"))


def _endpoint_key(endpoint: Mapping[str, Any]) -> tuple[str, str]:
    """Identify a catalog endpoint by provider and region.

    Args:
        endpoint: An ``endpoints`` entry from ``models.json``.

    Returns:
        A (provider, region) key.
    """
    return str(endpoint.get("provider") or ""), str(endpoint.get("region") or "us")


def _catalog_tiers(entry: Mapping[str, Any]) -> tuple[tuple[int, float, float], ...]:
    """Read prompt-length rates already recorded for a model or endpoint.

    Args:
        entry: A catalog entry or one of its ``endpoints`` items.

    Returns:
        Comparable (threshold, prompt, completion) triples ordered by threshold.
    """
    pricing = entry.get("pricing") or {}
    tiers: list[tuple[int, float, float]] = []
    for item in pricing.get("tiers") or []:
        threshold = item.get("min_prompt_tokens")
        prompt = _price(item.get("prompt"))
        completion = _price(item.get("completion"))
        if threshold is None or prompt is None or completion is None:
            continue
        tiers.append((int(threshold), prompt, completion))
    return tuple(sorted(tiers))


def _comparable(tiers: Iterable[UpstreamTier]) -> tuple[tuple[int, float, float], ...]:
    """Put upstream tiers in the same shape as catalog tiers.

    Args:
        tiers: Upstream tier records.

    Returns:
        Comparable triples ordered by threshold.
    """
    return tuple(sorted((t.min_prompt_tokens, t.prompt, t.completion) for t in tiers))


def _default_host(spec: Mapping[str, Any]) -> tuple[str, str] | None:
    """Find the host whose rate the model-level price should track.

    Args:
        spec: A catalog entry.

    Returns:
        The first endpoint's (provider, region) key, or ``None``.
    """
    for endpoint in spec.get("endpoints") or []:
        return _endpoint_key(endpoint)
    return None


def diff_catalog(
    current: Mapping[str, Any],
    upstream: Mapping[str, UpstreamModel],
    served: set[str],
    *,
    providers_checked: Iterable[str] = (),
    endpoints_by_model: Mapping[str, list[UpstreamEndpoint]] | None = None,
) -> CatalogDiff:
    """Compare the bundled catalog against upstream.

    Args:
        current: Decoded ``models.json``.
        upstream: Upstream models keyed by id.
        served: Normalized slugs configured providers report. Empty means
            availability could not be confirmed, so nothing is filtered on it.
        providers_checked: Provider slugs that answered.
        endpoints_by_model: Per-model host records. Models absent from this
            mapping are skipped for repricing rather than assumed unchanged, so a
            failed lookup never looks like a price drop.

    Returns:
        The differences a human needs to review.
    """
    result = CatalogDiff(providers_checked=sorted(providers_checked))
    entries = {spec["id"]: spec for spec in current.get("models", [])}
    hosts_by_model = endpoints_by_model or {}

    for model_id, spec in sorted(entries.items()):
        prompt, completion = _catalog_prices(spec)
        if prompt is None or completion is None:
            result.unpriced.append(model_id)
        if served and normalize_model_id(model_id) not in served:
            result.unconfirmed.append(model_id)

        hosts = hosts_by_model.get(model_id)
        if hosts is None:
            # Nothing upstream to compare against; say nothing rather than guess.
            if model_id not in upstream:
                result.removed.append(model_id)
            continue
        if not hosts:
            result.removed.append(model_id)
            continue

        if any(host.discount for host in hosts):
            result.discounted.append(model_id)

        indexed = standard_endpoints(hosts)
        ours = {_endpoint_key(e): e for e in spec.get("endpoints") or []}
        for key in sorted(indexed):
            if key not in ours:
                result.new_hosts.append((model_id, f"{key[0]}/{key[1]}"))

        default_key = _default_host(spec)
        for key, endpoint in sorted(indexed.items()):
            existing = ours.get(key)
            if existing is None:
                continue
            old_prompt, old_completion = _catalog_prices(existing)
            provider, region = key
            if _catalog_tiers(existing) != _comparable(endpoint.tiers):
                result.retiered.append(TierChange(model_id, endpoint.tiers, provider, region))
            if endpoint.prompt is not None and endpoint.prompt != old_prompt:
                result.repriced.append(
                    PriceChange(model_id, "prompt", old_prompt, endpoint.prompt, provider, region)
                )
            if endpoint.completion is not None and endpoint.completion != old_completion:
                result.repriced.append(
                    PriceChange(
                        model_id,
                        "completion",
                        old_completion,
                        endpoint.completion,
                        provider,
                        region,
                    )
                )

        # The model-level price advertises the default host's rate.
        reference = indexed.get(default_key) if default_key else None
        if reference is None and len(indexed) == 1:
            reference = next(iter(indexed.values()))
        if reference is not None:
            if _catalog_tiers(spec) != _comparable(reference.tiers):
                result.retiered.append(TierChange(model_id, reference.tiers))
            if reference.prompt is not None and reference.prompt != prompt:
                result.repriced.append(PriceChange(model_id, "prompt", prompt, reference.prompt))
            if reference.completion is not None and reference.completion != completion:
                result.repriced.append(
                    PriceChange(model_id, "completion", completion, reference.completion)
                )

    for model_id, candidate in sorted(upstream.items()):
        if model_id in entries:
            continue
        # Only authors we have an adapter for are routable at all.
        if candidate.author not in PROVIDER_ENV:
            continue
        if served and normalize_model_id(model_id) not in served:
            continue
        result.candidates.append(candidate)
    return result


SPEC_KEY_ORDER = (
    "id",
    "name",
    "context_length",
    "pricing",
    "architecture",
    "supported_parameters",
    "endpoints",
)


def order_spec_keys(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Put catalog entry keys in a stable order.

    A price added to an entry that had none would otherwise land at the end,
    which makes the review diff harder to read than it needs to be.

    Args:
        spec: A single catalog entry.

    Returns:
        The same entry with known keys first, in canonical order.

    Examples:
        >>> list(order_spec_keys({"endpoints": [], "id": "a/b"}))
        ['id', 'endpoints']
    """
    ordered = {key: spec[key] for key in SPEC_KEY_ORDER if key in spec}
    for key in spec:
        if key not in ordered:
            ordered[key] = spec[key]
    return ordered


def _spec_from_upstream(model: UpstreamModel) -> dict[str, Any]:
    """Build a catalog entry for a newly discovered model.

    Args:
        model: Upstream record.

    Returns:
        An entry shaped like the rest of ``models.json``. Pricing is omitted when
        unknown, which keeps the model unavailable until someone sets a rate.
    """
    spec: dict[str, Any] = {
        "id": model.id,
        "name": model.name or model.id,
        "context_length": model.context_length,
    }
    if model.priced:
        spec["pricing"] = {
            "prompt": f"{model.prompt:.10f}".rstrip("0"),
            "completion": f"{model.completion:.10f}".rstrip("0"),
        }
    spec["architecture"] = {
        "modality": model.modality,
        "input_modalities": list(model.input_modalities),
        "output_modalities": list(model.output_modalities),
    }
    spec["supported_parameters"] = list(model.supported_parameters)
    spec["endpoints"] = []
    return spec


def _with_tiers(pricing: Mapping[str, Any] | None, tiers: list[dict[str, Any]]) -> dict[str, Any]:
    """Attach prompt-length rates to a pricing block.

    Args:
        pricing: Existing pricing mapping, if any.
        tiers: Serialized tiers to record.

    Returns:
        A new pricing mapping. The key is dropped when there are no tiers, so a
        model that stops pricing long context does not keep a stale threshold.
    """
    updated = dict(pricing or {})
    if tiers:
        updated["tiers"] = tiers
    else:
        updated.pop("tiers", None)
    return updated


def apply_diff(
    current: Mapping[str, Any],
    upstream: Mapping[str, UpstreamModel],
    changes: CatalogDiff,
    *,
    add: Iterable[str] = (),
    add_hosts: Iterable[str] = (),
    endpoints_by_model: Mapping[str, list[UpstreamEndpoint]] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Produce an updated catalog document.

    Price drift is applied automatically because a stale rate bills the wrong
    amount on every request. Additions and removals are deliberate: the catalog
    is curated, and dropping a model breaks callers.

    Args:
        current: Decoded ``models.json``.
        upstream: Upstream models keyed by id.
        changes: Differences from :func:`diff_catalog`.
        add: Model ids to bring into the catalog.
        add_hosts: Provider slugs whose regional endpoints should be added to
            every carried model that upstream offers them for.
        endpoints_by_model: Per-model host records, required by ``add_hosts``.
        now: Timestamp to stamp the document with.

    Returns:
        A new catalog document ready to write.
    """
    stamped = now or datetime.now(timezone.utc)
    entries = {spec["id"]: dict(spec) for spec in current.get("models", [])}

    for change in changes.repriced:
        if change.new is None:
            continue
        spec = entries.get(change.id)
        if spec is None:
            continue
        formatted = f"{change.new:.10f}".rstrip("0")
        if change.provider is None:
            pricing = dict(spec.get("pricing") or {})
            pricing[change.field_name] = formatted
            spec["pricing"] = pricing
            continue
        # pricing_for() prefers the endpoint rate, so the host that actually moved
        # is the one to edit. Other hosts keep their own rates.
        endpoints = []
        for endpoint in spec.get("endpoints") or []:
            updated = dict(endpoint)
            if _endpoint_key(updated) == (change.provider, change.region):
                host_pricing = dict(updated.get("pricing") or {})
                host_pricing[change.field_name] = formatted
                updated["pricing"] = host_pricing
            endpoints.append(updated)
        if endpoints:
            spec["endpoints"] = endpoints

    for tier_change in changes.retiered:
        spec = entries.get(tier_change.id)
        if spec is None:
            continue
        payload = [
            {
                "min_prompt_tokens": tier.min_prompt_tokens,
                "prompt": f"{tier.prompt:.10f}".rstrip("0"),
                "completion": f"{tier.completion:.10f}".rstrip("0"),
            }
            for tier in tier_change.tiers
        ]
        if tier_change.provider is None:
            spec["pricing"] = _with_tiers(spec.get("pricing"), payload)
            continue
        endpoints = []
        for endpoint in spec.get("endpoints") or []:
            updated = dict(endpoint)
            if _endpoint_key(updated) == (tier_change.provider, tier_change.region):
                updated["pricing"] = _with_tiers(updated.get("pricing"), payload)
            endpoints.append(updated)
        if endpoints:
            spec["endpoints"] = endpoints

    for model_id in add:
        model = upstream.get(model_id)
        if model is not None:
            entries[model.id] = _spec_from_upstream(model)

    for provider in add_hosts:
        for model_id, hosts in (endpoints_by_model or {}).items():
            spec = entries.get(model_id)
            if spec is None:
                continue
            existing = {_endpoint_key(e) for e in spec.get("endpoints") or []}
            added = list(spec.get("endpoints") or [])
            for key, endpoint in sorted(standard_endpoints(hosts).items()):
                if key[0] != provider or key in existing:
                    continue
                added.append(
                    {
                        "provider": endpoint.provider,
                        "region": endpoint.region,
                        "upstream_id": endpoint.upstream_id,
                        "pricing": _with_tiers(
                            {
                                "prompt": f"{endpoint.prompt:.10f}".rstrip("0"),
                                "completion": f"{endpoint.completion:.10f}".rstrip("0"),
                            },
                            [
                                {
                                    "min_prompt_tokens": tier.min_prompt_tokens,
                                    "prompt": f"{tier.prompt:.10f}".rstrip("0"),
                                    "completion": f"{tier.completion:.10f}".rstrip("0"),
                                }
                                for tier in endpoint.tiers
                            ],
                        ),
                    }
                )
            if added:
                spec["endpoints"] = added

    return {
        "updated_at": stamped.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models": [order_spec_keys(entries[key]) for key in sorted(entries)],
    }


def render_report(changes: CatalogDiff) -> str:
    """Render the diff as markdown for a pull request body.

    Args:
        changes: Differences from :func:`diff_catalog`.

    Returns:
        A markdown summary.
    """
    lines = ["## Catalog sync", ""]
    if changes.providers_checked:
        lines.append(f"Providers checked: {', '.join(changes.providers_checked)}")
    else:
        lines.append("No provider keys configured, so availability is unconfirmed.")
    lines.append("")

    if changes.empty:
        lines.append("The catalog already matches upstream.")
        return "\n".join(lines)

    if changes.repriced:
        lines.append(f"### Repriced ({len(changes.repriced)})")
        lines.append("Undiscounted list prices, so a promotion never lowers what we bill.")
        for change in changes.repriced:
            lines.append(
                f"- `{change.id}` [{change.host}] {change.field_name}: {change.old} → {change.new}"
            )
        lines.append("")
    if changes.discounted:
        lines.append(f"### Running a promotion upstream ({len(changes.discounted)})")
        lines.append(
            "OpenRouter is discounting these. We record the list price, so our rate "
            "stays correct when the promotion ends."
        )
        for model_id in changes.discounted:
            lines.append(f"- `{model_id}`")
        lines.append("")
    if changes.retiered:
        lines.append(f"### Prompt-length pricing ({len(changes.retiered)})")
        lines.append(
            "These models charge a different rate above a prompt-token threshold. "
            "The tier replaces the base rate for the whole request."
        )
        for tier_change in changes.retiered:
            lines.append(f"- `{tier_change.id}` [{tier_change.host}] {tier_change.describe()}")
        lines.append("")
    if changes.new_hosts:
        lines.append(f"### Hosts we do not list ({len(changes.new_hosts)})")
        lines.append("Available upstream for models we already carry.")
        for model_id, host in changes.new_hosts:
            lines.append(f"- `{model_id}` — {host}")
        lines.append("")
    if changes.removed:
        lines.append(f"### No longer listed upstream ({len(changes.removed)})")
        lines.append("Left in place; remove by hand if you want to drop them.")
        for model_id in changes.removed:
            lines.append(f"- `{model_id}`")
        lines.append("")
    if changes.unpriced:
        lines.append(f"### Unpriced, hidden from the gateway ({len(changes.unpriced)})")
        for model_id in changes.unpriced:
            lines.append(f"- `{model_id}`")
        lines.append("")
    if changes.unconfirmed:
        lines.append(f"### Not listed by a configured provider ({len(changes.unconfirmed)})")
        for model_id in changes.unconfirmed:
            lines.append(f"- `{model_id}`")
        lines.append("")
    if changes.candidates:
        lines.append(f"### Available upstream, not carried ({len(changes.candidates)})")
        lines.append("Add one with `--add <id>` when you want to offer it.")
        lines.append("")
        lines.append("<details><summary>Show candidates</summary>")
        lines.append("")
        for model in changes.candidates:
            note = "" if model.priced else " — no upstream price"
            lines.append(f"- `{model.id}`{note}")
        lines.append("")
        lines.append("</details>")
        lines.append("")
    return "\n".join(lines)


def load_catalog(path: Path = DATA_PATH) -> dict[str, Any]:
    """Read the bundled catalog document.

    Args:
        path: Location of ``models.json``.

    Returns:
        The decoded document.
    """
    with path.open(encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    return data


def write_catalog(document: Mapping[str, Any], path: Path = DATA_PATH) -> None:
    """Write the catalog document back to disk.

    Args:
        document: Catalog document to serialize.
        path: Location of ``models.json``.
    """
    with path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    """Run the catalog sync.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit code. ``1`` under ``--check`` when the catalog is stale.
    """
    parser = argparse.ArgumentParser(description="Sync the bundled model catalog.")
    parser.add_argument("--write", action="store_true", help="Update models.json.")
    parser.add_argument(
        "--check", action="store_true", help="Exit non-zero when changes are pending."
    )
    parser.add_argument("--report", type=Path, help="Write a markdown report here.")
    parser.add_argument(
        "--add",
        action="append",
        default=[],
        metavar="MODEL_ID",
        help="Bring a model into the catalog. Repeatable.",
    )
    parser.add_argument(
        "--add-host",
        action="append",
        default=[],
        metavar="PROVIDER",
        help="Add a provider's regional endpoints to every model that offers them.",
    )
    parser.add_argument("--path", type=Path, default=DATA_PATH)
    args = parser.parse_args(argv)

    current = load_catalog(args.path)
    carried = [spec["id"] for spec in current.get("models", [])]
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        upstream = fetch_openrouter(client)
        served, checked = collect_served_slugs(client)
        # Only models we carry need per-host detail; the list response is enough
        # to enumerate candidates.
        endpoints_by_model = {model_id: fetch_endpoints(client, model_id) for model_id in carried}

    changes = diff_catalog(
        current,
        upstream,
        served,
        providers_checked=checked,
        endpoints_by_model=endpoints_by_model,
    )
    report = render_report(changes)
    print(report)
    if args.report:
        args.report.write_text(report, encoding="utf-8")

    unknown = [model_id for model_id in args.add if model_id not in upstream]
    if unknown:
        parser.error(f"unknown model ids: {', '.join(unknown)}")

    if args.write and (changes.has_updates or args.add or args.add_host):
        write_catalog(
            apply_diff(
                current,
                upstream,
                changes,
                add=args.add,
                add_hosts=args.add_host,
                endpoints_by_model=endpoints_by_model,
            ),
            args.path,
        )

    return 1 if args.check and changes.has_updates else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

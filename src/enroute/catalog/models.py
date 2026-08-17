"""Model catalog types and cost estimation.

Examples:
    >>> from enroute.catalog.models import ModelCatalog, estimate_cost
    >>> from enroute.types import Usage
    >>> cat = ModelCatalog()
    >>> spec = cat.get("openai/gpt-5.6-luna")
    >>> spec is not None
    True
    >>> estimate_cost(Usage.from_counts(1000, 500), spec) > 0
    True
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from enroute.errors import NotFoundError
from enroute.types import Usage

_US_HOST_ORDER = ("fireworks", "baseten")


def normalize_region(value: str) -> str:
    """Collapse a vendor region label to the catalog's coarse regions.

    Vendors name regions at finer grain than pricing varies, so ``us-east-1`` and
    ``us-west-2`` both bill as ``us``.

    Args:
        value: Region label such as ``us-east-1`` or ``europe-west4``.

    Returns:
        A short region key.

    Examples:
        >>> normalize_region("us-east-1"), normalize_region("europe-west4")
        ('us', 'eu')
        >>> normalize_region("global")
        'global'
    """
    lowered = value.lower()
    if lowered.startswith("us"):
        return "us"
    if lowered.startswith(("eu", "europe")):
        return "eu"
    if lowered.startswith("global"):
        return "global"
    return lowered


class PriceTier(BaseModel):
    """A rate that replaces the base rate once a prompt gets large enough.

    Attributes:
        min_prompt_tokens: Prompt size at which this tier starts applying.
        prompt: USD per prompt token at this tier.
        completion: USD per completion token at this tier.
    """

    min_prompt_tokens: int
    prompt: float
    completion: float


class ModelPricing(BaseModel):
    """Per-token pricing in USD.

    Attributes:
        prompt: USD per prompt token below the first tier threshold.
        completion: USD per completion token below the first tier threshold.
        tiers: Rates for larger prompts, if the model prices them differently.
    """

    prompt: float
    completion: float
    tiers: list[PriceTier] = Field(default_factory=list)

    def rates_for_prompt(self, prompt_tokens: int) -> tuple[float, float]:
        """Resolve the rates that apply to a request of a given prompt size.

        Providers that price long context replace the rate for the whole request
        rather than charging a higher rate only on tokens past the threshold, so
        the matching tier supersedes the base rate outright.

        Args:
            prompt_tokens: Prompt tokens in the request being billed.

        Returns:
            A (prompt rate, completion rate) pair.

        Examples:
            >>> pricing = ModelPricing(
            ...     prompt=5e-06,
            ...     completion=3e-05,
            ...     tiers=[PriceTier(min_prompt_tokens=272000, prompt=1e-05, completion=6e-05)],
            ... )
            >>> pricing.rates_for_prompt(1000)
            (5e-06, 3e-05)
            >>> pricing.rates_for_prompt(300000)
            (1e-05, 6e-05)
        """
        applicable = [tier for tier in self.tiers if prompt_tokens >= tier.min_prompt_tokens]
        if not applicable:
            return self.prompt, self.completion
        tier = max(applicable, key=lambda item: item.min_prompt_tokens)
        return tier.prompt, tier.completion


class Architecture(BaseModel):
    """Model modality metadata.

    Attributes:
        modality: High-level modality string.
        input_modalities: Accepted input modalities.
        output_modalities: Produced output modalities.
    """

    modality: str | None = None
    input_modalities: list[str] = Field(default_factory=lambda: ["text"])
    output_modalities: list[str] = Field(default_factory=lambda: ["text"])


class ModelEndpoint(BaseModel):
    """A concrete inference host for a catalog model.

    Attributes:
        provider: Host slug (``openai``, ``fireworks``, ``moonshot``, …).
        region: Serving region hint (``us``, ``cn``, ``global``).
        upstream_id: Model id the host expects.
        pricing: Pass-through price at this host.
    """

    provider: str
    region: str = "us"
    upstream_id: str
    pricing: ModelPricing | None = None


class ModelSpec(BaseModel):
    """A catalog entry for a model.

    Attributes:
        id: Model id in ``author/slug`` form.
        name: Human-readable display name.
        context_length: Maximum context window in tokens.
        pricing: Default per-token pricing (US-first host, then first endpoint).
        architecture: Modality metadata.
        supported_parameters: Request parameters the model accepts.
        endpoints: Inference hosts. Empty means the author is the only host.
        provider: Derived provider slug (author segment of ``id``).
    """

    id: str
    name: str | None = None
    context_length: int | None = None
    pricing: ModelPricing | None = None
    architecture: Architecture = Field(default_factory=Architecture)
    supported_parameters: list[str] = Field(default_factory=list)
    endpoints: list[ModelEndpoint] = Field(default_factory=list)

    @property
    def provider(self) -> str:
        """Author slug derived from the model id.

        Returns:
            The ``author`` segment of ``author/slug``, or the full id.
        """
        if "/" in self.id:
            return self.id.split("/", 1)[0]
        return self.id

    def ordered_endpoints(self) -> list[ModelEndpoint]:
        """Return endpoints with US hosts first (Fireworks, then Baseten).

        Returns:
            Ordered endpoints, or a single implicit author host when none are listed.
        """
        if not self.endpoints:
            upstream = self.id.split("/", 1)[1] if "/" in self.id else self.id
            return [
                ModelEndpoint(
                    provider=self.provider,
                    region="us",
                    upstream_id=upstream,
                    pricing=self.pricing,
                )
            ]
        us = [endpoint for endpoint in self.endpoints if endpoint.region == "us"]
        rest = [endpoint for endpoint in self.endpoints if endpoint.region != "us"]

        def us_key(endpoint: ModelEndpoint) -> int:
            if endpoint.provider in _US_HOST_ORDER:
                return _US_HOST_ORDER.index(endpoint.provider)
            return 50

        us.sort(key=us_key)
        return us + rest

    def host_providers(self) -> list[str]:
        """Return host slugs that can serve this model."""
        return [endpoint.provider for endpoint in self.ordered_endpoints()]

    def pricing_for(
        self, provider: str | None = None, region: str | None = None
    ) -> ModelPricing | None:
        """Return pricing for a host, or the default US-first price.

        Region matters because the same provider charges different rates in
        different regions: Azure EU lists above Azure US for identical models.
        When a region is given it must match, so a miss falls back to the default
        rather than billing a cheaper region's rate for a pricier one.

        Args:
            provider: Host slug. ``None`` uses the default endpoint.
            region: Host region. ``None`` matches the first endpoint for the provider.

        Returns:
            Per-token pricing, or ``None``.
        """
        endpoints = self.ordered_endpoints()
        if provider:
            for endpoint in endpoints:
                if endpoint.provider != provider or endpoint.pricing is None:
                    continue
                if region is not None and endpoint.region != region:
                    continue
                return endpoint.pricing
        if endpoints and endpoints[0].pricing is not None:
            return endpoints[0].pricing
        return self.pricing


def estimate_cost(
    usage: Usage,
    spec: ModelSpec | None,
    *,
    provider: str | None = None,
    region: str | None = None,
) -> float | None:
    """Estimate USD cost for a usage record against a model spec.

    Args:
        usage: Token usage.
        spec: Model catalog entry, or ``None``.
        provider: Optional host slug; bills at that host's pass-through price.
        region: Optional host region, for providers whose rates vary by region.

    Returns:
        Estimated USD cost, or ``None`` if pricing is unavailable.

    Examples:
        >>> from enroute.types import Usage
        >>> spec = ModelSpec(
        ...     id="openai/gpt-5.6-luna",
        ...     pricing=ModelPricing(prompt=0.0000002, completion=0.0000012),
        ... )
        >>> round(estimate_cost(Usage.from_counts(1_000_000, 0), spec) or 0, 2)
        0.2
    """
    if spec is None:
        return None
    pricing = spec.pricing_for(provider, region)
    if pricing is None:
        return None
    prompt_rate, completion_rate = pricing.rates_for_prompt(usage.prompt_tokens)
    return usage.prompt_tokens * prompt_rate + usage.completion_tokens * completion_rate


def _parse_tiers(raw: Any) -> list[PriceTier]:
    tiers: list[PriceTier] = []
    for item in raw or []:
        try:
            tiers.append(
                PriceTier(
                    min_prompt_tokens=int(item["min_prompt_tokens"]),
                    prompt=float(item["prompt"]),
                    completion=float(item["completion"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            # A malformed tier must not silently become the base rate.
            continue
    tiers.sort(key=lambda tier: tier.min_prompt_tokens)
    return tiers


def _parse_pricing(raw: dict[str, Any] | None) -> ModelPricing | None:
    if not raw:
        return None
    try:
        return ModelPricing(
            prompt=float(raw["prompt"]),
            completion=float(raw["completion"]),
            tiers=_parse_tiers(raw.get("tiers")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _dump_pricing(pricing: ModelPricing | None) -> dict[str, Any] | None:
    if pricing is None:
        return None
    payload: dict[str, Any] = {
        "prompt": str(pricing.prompt),
        "completion": str(pricing.completion),
    }
    if pricing.tiers:
        payload["tiers"] = [
            {
                "min_prompt_tokens": tier.min_prompt_tokens,
                "prompt": str(tier.prompt),
                "completion": str(tier.completion),
            }
            for tier in pricing.tiers
        ]
    return payload


def _parse_endpoint(raw: dict[str, Any]) -> ModelEndpoint | None:
    provider = raw.get("provider")
    upstream_id = raw.get("upstream_id")
    if not provider or not upstream_id:
        return None
    return ModelEndpoint(
        provider=str(provider),
        region=str(raw.get("region") or "us"),
        upstream_id=str(upstream_id),
        pricing=_parse_pricing(raw.get("pricing")),
    )


class ModelCatalog:
    """In-memory model catalog loaded from a bundled JSON snapshot.

    Args:
        path: Optional path to a catalog JSON file. Defaults to the bundled snapshot.

    Examples:
        >>> catalog = ModelCatalog()
        >>> len(catalog.models()) > 0
        True
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._models: dict[str, ModelSpec] = {}
        self._updated_at: str | None = None
        if path is None:
            self.load_bundled()
        else:
            self.load_path(Path(path))

    @property
    def updated_at(self) -> str | None:
        """Catalog snapshot timestamp, if known."""
        return self._updated_at

    def load_bundled(self) -> None:
        """Load the package-bundled catalog snapshot."""
        ref = resources.files("enroute.catalog.data").joinpath("models.json")
        with ref.open("r", encoding="utf-8") as fh:
            self._load_payload(json.load(fh))

    def load_path(self, path: Path) -> None:
        """Load a catalog from a filesystem path.

        Args:
            path: Path to a JSON catalog file.
        """
        with path.open("r", encoding="utf-8") as fh:
            self._load_payload(json.load(fh))

    def _load_payload(self, payload: dict[str, Any]) -> None:
        self._updated_at = payload.get("updated_at")
        models: dict[str, ModelSpec] = {}
        for item in payload.get("models") or []:
            endpoints = [
                endpoint
                for raw in item.get("endpoints") or []
                if (endpoint := _parse_endpoint(raw)) is not None
            ]
            pricing = _parse_pricing(item.get("pricing"))
            spec = ModelSpec(
                id=item["id"],
                name=item.get("name"),
                context_length=item.get("context_length"),
                pricing=pricing,
                architecture=Architecture(
                    modality=(item.get("architecture") or {}).get("modality"),
                    input_modalities=list(
                        (item.get("architecture") or {}).get("input_modalities") or ["text"]
                    ),
                    output_modalities=list(
                        (item.get("architecture") or {}).get("output_modalities") or ["text"]
                    ),
                ),
                supported_parameters=list(item.get("supported_parameters") or []),
                endpoints=endpoints,
            )
            if spec.pricing is None:
                spec.pricing = spec.pricing_for()
            models[spec.id] = spec
        self._models = models

    def models(self) -> list[ModelSpec]:
        """Return all model specs.

        Returns:
            A list of :class:`ModelSpec` entries.
        """
        return list(self._models.values())

    def get(self, model_id: str) -> ModelSpec | None:
        """Look up a model by id.

        Args:
            model_id: Model id in ``author/slug`` form.

        Returns:
            The matching :class:`ModelSpec`, or ``None``.
        """
        return self._models.get(model_id)

    def require(self, model_id: str) -> ModelSpec:
        """Look up a model by id or raise.

        Args:
            model_id: Model id in ``author/slug`` form.

        Returns:
            The matching :class:`ModelSpec`.

        Raises:
            NotFoundError: If the model is not in the catalog.
        """
        spec = self.get(model_id)
        if spec is None:
            raise NotFoundError(f"model not found in catalog: {model_id}", model=model_id)
        return spec

    def refresh_from_openrouter(self, url: str = "https://openrouter.ai/api/v1/models") -> int:
        """Refresh the in-memory catalog from OpenRouter's public models API.

        Args:
            url: Models endpoint URL.

        Returns:
            Number of models loaded.

        Note:
            This performs a network request and is intended for maintainer
            tooling, not runtime hot paths.
        """
        import httpx

        response = httpx.get(url, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        models = []
        for item in data.get("data") or []:
            models.append(
                {
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "context_length": item.get("context_length"),
                    "pricing": item.get("pricing"),
                    "architecture": item.get("architecture"),
                    "supported_parameters": item.get("supported_parameters"),
                }
            )
        self._load_payload({"updated_at": None, "models": models})
        return len(models)

    def write_snapshot(self, path: Path) -> None:
        """Write the current catalog to a JSON snapshot file.

        Args:
            path: Destination path.
        """
        payload = {
            "updated_at": self._updated_at,
            "models": [
                {
                    "id": spec.id,
                    "name": spec.name,
                    "context_length": spec.context_length,
                    "pricing": _dump_pricing(spec.pricing),
                    "architecture": spec.architecture.model_dump(),
                    "supported_parameters": spec.supported_parameters,
                    "endpoints": [
                        {
                            "provider": endpoint.provider,
                            "region": endpoint.region,
                            "upstream_id": endpoint.upstream_id,
                            "pricing": _dump_pricing(endpoint.pricing),
                        }
                        for endpoint in spec.endpoints
                    ],
                }
                for spec in self.models()
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

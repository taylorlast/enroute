"""Refresh the bundled model catalog from upstream sources.

The catalog ships as static JSON so pricing changes are reviewable in git. This
module rebuilds that file from two inputs:

* OpenRouter's public model list supplies canonical ``author/slug`` ids, context
  windows, modalities, and per-token pricing.
* Each configured provider's own model listing confirms we can actually serve a
  model today. Providers are optional; without keys, models are still proposed
  but flagged as unconfirmed.

Pricing is pass-through, so a model with no price must never go live. New models
are written without pricing when none can be resolved, which leaves them
unavailable until a human fills the rate in.

Upstream price changes are applied automatically because a stale rate bills the
wrong amount on every request. Adding a model is deliberate, since the catalog is
curated rather than a mirror of everything available.

Run it with::

    python -m enroute.catalog.sync --check
    python -m enroute.catalog.sync --write --report catalog-report.md
    python -m enroute.catalog.sync --write --add openai/gpt-5.6-luna
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

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


@dataclass(frozen=True)
class PriceChange:
    """A per-token price that moved upstream.

    Attributes:
        id: Model id.
        field_name: Which rate changed (``prompt`` or ``completion``).
        old: Price currently in the catalog.
        new: Price reported upstream.
    """

    id: str
    field_name: str
    old: float | None
    new: float | None


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
    """

    candidates: list[UpstreamModel] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    repriced: list[PriceChange] = field(default_factory=list)
    unpriced: list[str] = field(default_factory=list)
    unconfirmed: list[str] = field(default_factory=list)
    providers_checked: list[str] = field(default_factory=list)

    @property
    def has_updates(self) -> bool:
        """Whether anything would be written without an explicit request.

        Returns:
            ``True`` when a price moved upstream.
        """
        return bool(self.repriced)

    @property
    def empty(self) -> bool:
        """Whether there is nothing at all worth reporting.

        Returns:
            ``True`` when the catalog matches upstream and nothing is pending.
        """
        return not (self.candidates or self.removed or self.repriced)


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


def diff_catalog(
    current: Mapping[str, Any],
    upstream: Mapping[str, UpstreamModel],
    served: set[str],
    *,
    providers_checked: Iterable[str] = (),
) -> CatalogDiff:
    """Compare the bundled catalog against upstream.

    Args:
        current: Decoded ``models.json``.
        upstream: Upstream models keyed by id.
        served: Normalized slugs configured providers report. Empty means
            availability could not be confirmed, so nothing is filtered on it.
        providers_checked: Provider slugs that answered.

    Returns:
        The differences a human needs to review.
    """
    result = CatalogDiff(providers_checked=sorted(providers_checked))
    entries = {spec["id"]: spec for spec in current.get("models", [])}

    for model_id, spec in sorted(entries.items()):
        prompt, completion = _catalog_prices(spec)
        if prompt is None or completion is None:
            result.unpriced.append(model_id)
        if served and normalize_model_id(model_id) not in served:
            result.unconfirmed.append(model_id)
        candidate = upstream.get(model_id)
        if candidate is None:
            result.removed.append(model_id)
            continue
        if candidate.prompt is not None and candidate.prompt != prompt:
            result.repriced.append(PriceChange(model_id, "prompt", prompt, candidate.prompt))
        if candidate.completion is not None and candidate.completion != completion:
            result.repriced.append(
                PriceChange(model_id, "completion", completion, candidate.completion)
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


def apply_diff(
    current: Mapping[str, Any],
    upstream: Mapping[str, UpstreamModel],
    changes: CatalogDiff,
    *,
    add: Iterable[str] = (),
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
        pricing = dict(spec.get("pricing") or {})
        pricing[change.field_name] = formatted
        spec["pricing"] = pricing
        # pricing_for() prefers the endpoint rate, so a host that was tracking the
        # default has to move with it. Hosts with their own rate are left alone.
        endpoints = []
        for endpoint in spec.get("endpoints") or []:
            updated = dict(endpoint)
            host_pricing = dict(updated.get("pricing") or {})
            if host_pricing and _price(host_pricing.get(change.field_name)) == change.old:
                host_pricing[change.field_name] = formatted
                updated["pricing"] = host_pricing
            endpoints.append(updated)
        if endpoints:
            spec["endpoints"] = endpoints

    for model_id in add:
        model = upstream.get(model_id)
        if model is not None:
            entries[model.id] = _spec_from_upstream(model)

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
        for change in changes.repriced:
            lines.append(f"- `{change.id}` {change.field_name}: {change.old} → {change.new}")
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
    parser.add_argument("--path", type=Path, default=DATA_PATH)
    args = parser.parse_args(argv)

    current = load_catalog(args.path)
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        upstream = fetch_openrouter(client)
        served, checked = collect_served_slugs(client)

    changes = diff_catalog(current, upstream, served, providers_checked=checked)
    report = render_report(changes)
    print(report)
    if args.report:
        args.report.write_text(report, encoding="utf-8")

    unknown = [model_id for model_id in args.add if model_id not in upstream]
    if unknown:
        parser.error(f"unknown model ids: {', '.join(unknown)}")

    if args.write and (changes.has_updates or args.add):
        write_catalog(apply_diff(current, upstream, changes, add=args.add), args.path)

    return 1 if args.check and changes.has_updates else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

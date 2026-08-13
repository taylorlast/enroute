"""Model catalog types and cost estimation.

Examples:
    >>> from enroute.catalog.models import ModelCatalog, estimate_cost
    >>> from enroute.types import Usage
    >>> cat = ModelCatalog()
    >>> spec = cat.get("openai/gpt-4o-mini")
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


class ModelPricing(BaseModel):
    """Per-token pricing in USD.

    Attributes:
        prompt: USD per prompt token.
        completion: USD per completion token.
    """

    prompt: float
    completion: float


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


class ModelSpec(BaseModel):
    """A catalog entry for a model.

    Attributes:
        id: Model id in ``author/slug`` form.
        name: Human-readable display name.
        context_length: Maximum context window in tokens.
        pricing: Per-token pricing.
        architecture: Modality metadata.
        supported_parameters: Request parameters the model accepts.
        provider: Derived provider slug (author segment of ``id``).
    """

    id: str
    name: str | None = None
    context_length: int | None = None
    pricing: ModelPricing | None = None
    architecture: Architecture = Field(default_factory=Architecture)
    supported_parameters: list[str] = Field(default_factory=list)

    @property
    def provider(self) -> str:
        """Provider slug derived from the model id.

        Returns:
            The ``author`` segment of ``author/slug``, or the full id.
        """
        if "/" in self.id:
            return self.id.split("/", 1)[0]
        return self.id


def estimate_cost(usage: Usage, spec: ModelSpec | None) -> float | None:
    """Estimate USD cost for a usage record against a model spec.

    Args:
        usage: Token usage.
        spec: Model catalog entry, or ``None``.

    Returns:
        Estimated USD cost, or ``None`` if pricing is unavailable.

    Examples:
        >>> from enroute.types import Usage
        >>> spec = ModelSpec(
        ...     id="openai/gpt-4o-mini",
        ...     pricing=ModelPricing(prompt=0.00000015, completion=0.0000006),
        ... )
        >>> round(estimate_cost(Usage.from_counts(1_000_000, 0), spec) or 0, 2)
        0.15
    """
    if spec is None or spec.pricing is None:
        return None
    return (
        usage.prompt_tokens * spec.pricing.prompt
        + usage.completion_tokens * spec.pricing.completion
    )


def _parse_pricing(raw: dict[str, Any] | None) -> ModelPricing | None:
    if not raw:
        return None
    try:
        return ModelPricing(prompt=float(raw["prompt"]), completion=float(raw["completion"]))
    except (KeyError, TypeError, ValueError):
        return None


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
            pricing = _parse_pricing(item.get("pricing"))
            arch_raw = item.get("architecture") or {}
            spec = ModelSpec(
                id=item["id"],
                name=item.get("name"),
                context_length=item.get("context_length"),
                pricing=pricing,
                architecture=Architecture(
                    modality=arch_raw.get("modality"),
                    input_modalities=list(arch_raw.get("input_modalities") or ["text"]),
                    output_modalities=list(arch_raw.get("output_modalities") or ["text"]),
                ),
                supported_parameters=list(item.get("supported_parameters") or []),
            )
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
                    "id": m.id,
                    "name": m.name,
                    "context_length": m.context_length,
                    "pricing": (
                        {"prompt": str(m.pricing.prompt), "completion": str(m.pricing.completion)}
                        if m.pricing
                        else None
                    ),
                    "architecture": m.architecture.model_dump(),
                    "supported_parameters": m.supported_parameters,
                }
                for m in self.models()
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

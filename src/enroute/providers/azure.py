"""Azure OpenAI adapter.

Azure speaks the OpenAI Chat Completions API on its ``/openai/v1`` route, so the
wire format, tool calls, and SSE streaming are inherited unchanged. Three things
differ and are handled here:

* The key goes in an ``api-key`` header. Azure reserves ``Authorization: Bearer``
  for Microsoft Entra ID tokens, which are supported via ``use_bearer_auth``.
* The ``model`` field carries a *deployment* name, which the customer chooses when
  they provision the model. It frequently does not match the model id, so a
  deployment map is accepted and there is no safe way to guess one.
* Rates differ by region, and EU capacity lists above US, so the catalog prices
  each Azure region separately rather than sharing one Azure rate.

Examples:
    >>> from enroute.providers.azure import normalize_azure_base_url
    >>> normalize_azure_base_url("https://acme.openai.azure.com")
    'https://acme.openai.azure.com/openai/v1'
"""

from __future__ import annotations

from enroute.providers.openai_compatible import OpenAICompatible

AZURE_V1_SUFFIX = "/openai/v1"


def normalize_azure_base_url(endpoint: str) -> str:
    """Point a resource endpoint at the v1 inference route.

    Args:
        endpoint: Resource endpoint, with or without the route suffix.

    Returns:
        A base URL ending in ``/openai/v1``.

    Examples:
        >>> normalize_azure_base_url("https://acme.openai.azure.com/")
        'https://acme.openai.azure.com/openai/v1'
        >>> normalize_azure_base_url("https://acme.openai.azure.com/openai/v1")
        'https://acme.openai.azure.com/openai/v1'
    """
    trimmed = endpoint.rstrip("/")
    if trimmed.endswith(AZURE_V1_SUFFIX):
        return trimmed
    return f"{trimmed}{AZURE_V1_SUFFIX}"


class AzureOpenAIProvider(OpenAICompatible):
    """Azure OpenAI via the v1 chat completions route.

    Args:
        api_key: Azure API key, or an Entra ID token when ``use_bearer_auth`` is set.
        endpoint: Resource endpoint, such as ``https://acme.openai.azure.com``.
        deployments: Maps model ids to deployment names. Both the full
            ``author/slug`` and the bare slug are accepted as keys.
        api_version: Optional explicit version, for opting into ``preview``.
        use_bearer_auth: Send the credential as a bearer token for Entra ID.
        base_url: Overrides ``endpoint`` entirely.
        region: Serving region, recorded so billing can charge that region's rate.
        name: Provider slug.
        timeout_s: Request timeout in seconds.
        default_headers: Extra headers.

    Raises:
        ConfigurationError: If neither ``endpoint`` nor ``base_url`` is given.

    Examples:
        >>> provider = AzureOpenAIProvider(
        ...     "key",
        ...     endpoint="https://acme.openai.azure.com",
        ...     deployments={"openai/gpt-5.6-sol": "sol-prod"},
        ... )
        >>> provider._model_id("openai/gpt-5.6-sol")
        'sol-prod'
        >>> provider.close()
    """

    name = "azure"
    # Azure has no shared host, so there is no usable default.
    default_base_url = ""
    strip_model_prefix = True

    def __init__(
        self,
        api_key: str,
        *,
        endpoint: str | None = None,
        deployments: dict[str, str] | None = None,
        api_version: str | None = None,
        use_bearer_auth: bool = False,
        base_url: str | None = None,
        region: str = "us",
        name: str | None = None,
        timeout_s: float | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        from enroute.errors import ConfigurationError

        resolved = base_url or (normalize_azure_base_url(endpoint) if endpoint else None)
        if not resolved:
            raise ConfigurationError(
                "azure requires endpoint=https://<resource>.openai.azure.com",
                provider=name or self.name,
            )
        self.deployments = deployments or {}
        self.region = region
        self.use_bearer_auth = use_bearer_auth
        super().__init__(
            api_key,
            base_url=resolved,
            name=name,
            timeout_s=timeout_s,
            default_headers=default_headers,
        )
        if api_version:
            self._client.params = {"api-version": api_version}
            self._aclient.params = {"api-version": api_version}

    def _build_headers(self) -> dict[str, str]:
        if self.use_bearer_auth:
            return super()._build_headers()
        # Azure keeps Authorization for Entra ID tokens.
        return {
            "api-key": self.config.api_key,
            "Content-Type": "application/json",
            **self.config.default_headers,
        }

    def _model_id(self, model: str) -> str:
        """Resolve a model id to the deployment that serves it.

        Args:
            model: Model id, either ``author/slug`` or a bare slug.

        Returns:
            The configured deployment name, or the bare slug when unmapped.
        """
        if model in self.deployments:
            return self.deployments[model]
        bare = model.split("/", 1)[1] if model.count("/") == 1 else model
        return self.deployments.get(bare, bare)

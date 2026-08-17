# Azure and Bedrock

Enterprises usually cannot send traffic to a model lab directly: the contract, the
data residency commitment, and the invoice all sit with a cloud vendor. Azure and
Bedrock resell the same flagship models under those terms, so the catalog lists
them as hosts alongside the labs.

Both are configured from the environment and picked up automatically.

## Azure OpenAI

```bash
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
export AZURE_OPENAI_REGION=eu          # us, eu, or global — must match the resource
export AZURE_OPENAI_DEPLOYMENTS='{"openai/gpt-5.6-sol": "sol-prod"}'
```

Azure requires a **deployment map**. A deployment name is chosen by whoever
provisions the model, so it cannot be derived from the model id — `sol-prod` and
`gpt-5-6-sol-eastus` are both ordinary choices. Unmapped models fall back to the
bare slug, which works only if the deployment happens to be named that way.

Set `AZURE_OPENAI_REGION` to the region the resource actually lives in. EU
capacity lists above US for identical models, and the router will not offer a
route it would have to bill at another region's rate.

For Microsoft Entra ID instead of an API key, pass the token yourself:

```python
from enroute import Enroute
from enroute.providers import AzureOpenAIProvider

client = Enroute(
    providers={
        "azure": AzureOpenAIProvider(
            entra_token,
            endpoint="https://acme.openai.azure.com",
            deployments={"openai/gpt-5.6-sol": "sol-prod"},
            region="eu",
            use_bearer_auth=True,
        )
    }
)
```

## Amazon Bedrock

With a Bedrock API key:

```bash
export AWS_BEARER_TOKEN_BEDROCK=...
export AWS_REGION=us-east-1
export BEDROCK_MODEL_IDS='{"openai/gpt-5.6-sol": "us.openai.gpt-5.6-sol"}'
```

With IAM credentials, which most enterprises mandate over long-lived keys:

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...          # when using temporary credentials
export AWS_REGION=eu-central-1
```

Requests are signed with SigV4 when IAM credentials are used and sent as a bearer
token otherwise. An explicit key wins over ambient credentials, since instance
credentials are usually inherited rather than chosen.

Bedrock model ids are its own (`us.anthropic.claude-sonnet-4-6`) and include a
cross-region inference profile prefix, so `BEDROCK_MODEL_IDS` is required for
anything whose Bedrock id differs from the catalog slug.

`AWS_REGION` is a full AWS region. It is collapsed to the catalog's coarse region
for pricing, because rates vary by continent rather than by zone: `eu-central-1`
and `eu-west-1` both bill as `eu`.

## Streaming

Both stream. Azure uses server-sent events like OpenAI. Bedrock replies in the
binary `vnd.amazon.eventstream` framing, which is decoded into the same
`StreamChunk` type, so callers see no difference:

```python
for chunk in client.stream(model="openai/gpt-5.6-sol", messages=[...]):
    print(chunk.delta.content or "", end="")
```

Usage on a Bedrock stream arrives only in the trailing `metadata` event. Enroute
waits for it before costing the request, so a stream is billed on real token
counts rather than an estimate.

## Cost

Streams and non-streams cost the same way, at the serving host's rate:

```python
response.usage.cost      # USD, at this host and region
response.provider        # "azure"
response.region          # "eu"
```

Models that price long context carry rate tiers, and a tier replaces the base
rate for the whole request once the prompt crosses its threshold. See
[Routing](../concepts/routing.md).

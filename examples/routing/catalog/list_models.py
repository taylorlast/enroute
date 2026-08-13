"""List model ids available in the enroute catalog.

uv run python examples/routing/catalog/list_models.py
uv run python examples/routing/catalog/list_models.py --provider openai
"""

import argparse

from enroute import ModelCatalog

parser = argparse.ArgumentParser()
parser.add_argument("--provider", help="Filter by provider, e.g. openai")
parser.add_argument("--refresh", action="store_true", help="Refresh from OpenRouter (network)")
args = parser.parse_args()

catalog = ModelCatalog()
if args.refresh:
    print(f"refreshed {catalog.refresh_from_openrouter()} models")

models = catalog.models()
if args.provider:
    models = [m for m in models if m.provider == args.provider]

for spec in sorted(models, key=lambda m: m.id):
    price_in = f"${spec.pricing.prompt * 1_000_000:.2f}/1M" if spec.pricing else "-"
    price_out = f"${spec.pricing.completion * 1_000_000:.2f}/1M" if spec.pricing else "-"
    print(f"{spec.id:50}  ctx={spec.context_length}  in={price_in}  out={price_out}")

print(f"\n{len(models)} model(s)")

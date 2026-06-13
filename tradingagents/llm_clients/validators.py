"""Model name validators for each provider.

``VALID_MODELS`` and ``validate_model`` are derived from the provider registry
(:mod:`tradingagents.llm_clients.provider_registry`): providers flagged
``accepts_any_model`` (ollama, openrouter) skip validation entirely, and the
remaining providers are validated against the registry's curated model lists.
"""

from .provider_registry import accepts_any_model as _accepts_any_model
from .provider_registry import known_models as _known_models


# Curated providers only: the catalog minus the ones that accept any model id.
# (Providers without a curated list — Azure, OpenRouter — are simply absent and
# fall through to "accept" in validate_model.)
VALID_MODELS = {
    provider: models
    for provider, models in _known_models().items()
    if not _accepts_any_model(provider)
}


def validate_model(provider: str, model: str) -> bool:
    """Check if model name is valid for the given provider.

    For providers flagged ``accepts_any_model`` (ollama, openrouter) — and for
    providers with no curated list at all (e.g. azure) — any model is accepted.
    """
    provider_lower = provider.lower()

    if _accepts_any_model(provider_lower):
        return True

    if provider_lower not in VALID_MODELS:
        return True

    return model in VALID_MODELS[provider_lower]

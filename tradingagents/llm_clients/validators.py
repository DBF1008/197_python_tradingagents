"""Model name validators for each provider.

Uses the central registry's ``accepts_any_model`` flag to decide which
providers skip model-name validation (Ollama, OpenRouter, Azure — any
model name is valid). Providers with a static model list are validated
against ``model_catalog.get_known_models()``.
"""

from .model_catalog import get_known_models
from .registry import PROVIDERS

# Providers where any model name is accepted (local runtimes, dynamic
# catalogs, free-form deployment names). Derived from the registry's
# accepts_any_model flag instead of a hardcoded tuple.
_ACCEPTS_ANY = frozenset(
    key for key, pdef in PROVIDERS.items() if pdef.accepts_any_model
)


VALID_MODELS = {
    provider: models
    for provider, models in get_known_models().items()
    if provider not in _ACCEPTS_ANY
}


def validate_model(provider: str, model: str) -> bool:
    """Check if model name is valid for the given provider.

    For providers with ``accepts_any_model=True`` (ollama, openrouter,
    azure), any model is accepted.
    """
    provider_lower = provider.lower()

    if provider_lower in _ACCEPTS_ANY:
        return True

    if provider_lower not in VALID_MODELS:
        return True

    return model in VALID_MODELS[provider_lower]

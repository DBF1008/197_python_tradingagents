"""Shared model catalog for CLI selections and validation.

The catalog is derived from the provider registry
(:mod:`tradingagents.llm_clients.provider_registry`), which owns the curated
per-provider model lists. This module keeps the historical accessors
(``get_model_options``, ``get_known_models``, ``MODEL_OPTIONS``) and type
aliases as the stable import surface. To change a model list, edit the registry.
"""

from __future__ import annotations

from typing import Dict, List

from .provider_registry import PROVIDERS, ModelOption, ProviderModeOptions
from .provider_registry import known_models as _registry_known_models
from .provider_registry import model_options as _registry_model_options

__all__ = [
    "ModelOption",
    "ProviderModeOptions",
    "MODEL_OPTIONS",
    "get_model_options",
    "get_known_models",
]


# Provider -> {mode -> [(label, model_id), ...]}, only for providers with a
# curated list (OpenRouter is fetched dynamically and Azure takes any
# deployment name, so neither appears here). Dual-region providers share one
# list object, so e.g. ``MODEL_OPTIONS["qwen"] is MODEL_OPTIONS["qwen-cn"]``.
MODEL_OPTIONS: ProviderModeOptions = {
    spec.key: spec.models for spec in PROVIDERS.values() if spec.models is not None
}


def get_model_options(provider: str, mode: str) -> List[ModelOption]:
    """Return shared model options for a provider and selection mode."""
    return _registry_model_options(provider, mode)


def get_known_models() -> Dict[str, List[str]]:
    """Build known model names from the shared CLI catalog."""
    return _registry_known_models()

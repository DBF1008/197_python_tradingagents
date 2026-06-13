"""Canonical provider registry — single source of truth for provider metadata.

Every piece of provider-level knowledge (display name, API key env var,
default base URL, OpenAI-compatibility, model catalog reference, dual-region
grouping) lives here. Other modules (``api_key_env``, ``openai_client``,
``factory``, ``validators``, ``cli/utils``) derive their data from this
registry instead of maintaining parallel tables.

Adding a new provider means editing this file — not six others.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .model_catalog import (
    _ANTHROPIC_MODELS,
    _DEEPSEEK_MODELS,
    _GLM_MODELS,
    _GOOGLE_MODELS,
    _MINIMAX_MODELS,
    _OLLAMA_MODELS,
    _OPENAI_MODELS,
    _QWEN_MODELS,
    _XAI_MODELS,
)


# ---------------------------------------------------------------------------
# ProviderDef — the immutable metadata record for one provider
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderDef:
    """Immutable metadata for one LLM provider (or regional variant).

    Attributes:
        display_name: Human-readable name shown in CLI dropdowns.
        api_key_env: Environment variable holding the API key, or ``None``
            for providers that don't authenticate (e.g. Ollama).
        base_url: Default OpenAI-compatible API endpoint. ``None`` means
            "use the SDK's built-in default" (OpenAI, Anthropic, Google)
            or "requires separate env-based config" (Azure).
        openai_compatible: ``True`` if this provider is routed through
            ``OpenAIClient`` (Chat Completions API). ``False`` for
            providers with their own client class (Anthropic, Google, Azure).
        accepts_any_model: ``True`` if model validation should be skipped
            (Ollama, OpenRouter, Azure — any model name is valid).
        models: ``{"quick": [...], "deep": [...]}`` model-option tuples for
            CLI dropdowns and validation, or ``None`` if models are dynamic
            (OpenRouter) or free-form (Azure deployment names).
        region_group: Shared tag for dual-region sibling providers
            (e.g. ``"qwen"`` for both ``qwen`` and ``qwen-cn``). ``None``
            for providers without regional variants.
    """

    display_name: str
    api_key_env: Optional[str]
    base_url: Optional[str]
    openai_compatible: bool
    accepts_any_model: bool = False
    models: Optional[Dict[str, List[Tuple[str, str]]]] = None
    region_group: Optional[str] = None


# ---------------------------------------------------------------------------
# PROVIDERS — the registry itself, in CLI dropdown order
# ---------------------------------------------------------------------------

PROVIDERS: Dict[str, ProviderDef] = {
    # --- Native-SDK providers (not OpenAI-compatible) ---
    "openai": ProviderDef(
        display_name="OpenAI",
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        openai_compatible=True,
        models=_OPENAI_MODELS,
    ),
    "google": ProviderDef(
        display_name="Google",
        api_key_env="GOOGLE_API_KEY",
        base_url=None,  # SDK default: generativelanguage.googleapis.com
        openai_compatible=False,
        models=_GOOGLE_MODELS,
    ),
    "anthropic": ProviderDef(
        display_name="Anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com/",
        openai_compatible=False,
        models=_ANTHROPIC_MODELS,
    ),
    "azure": ProviderDef(
        display_name="Azure OpenAI",
        api_key_env="AZURE_OPENAI_API_KEY",
        base_url=None,  # requires AZURE_OPENAI_ENDPOINT env var
        openai_compatible=False,
        accepts_any_model=True,
        models=None,  # deployment names are free-form
    ),
    # --- OpenAI-compatible third-party providers ---
    "xai": ProviderDef(
        display_name="xAI",
        api_key_env="XAI_API_KEY",
        base_url="https://api.x.ai/v1",
        openai_compatible=True,
        models=_XAI_MODELS,
    ),
    "deepseek": ProviderDef(
        display_name="DeepSeek",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        openai_compatible=True,
        models=_DEEPSEEK_MODELS,
    ),
    # --- Dual-region: Qwen (DashScope) ---
    "qwen": ProviderDef(
        display_name="Qwen",
        api_key_env="DASHSCOPE_API_KEY",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        openai_compatible=True,
        models=_QWEN_MODELS,
        region_group="qwen",
    ),
    "qwen-cn": ProviderDef(
        display_name="Qwen (China)",
        api_key_env="DASHSCOPE_CN_API_KEY",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        openai_compatible=True,
        models=_QWEN_MODELS,
        region_group="qwen",
    ),
    # --- Dual-region: GLM (Zhipu) ---
    "glm": ProviderDef(
        display_name="GLM",
        api_key_env="ZHIPU_API_KEY",
        base_url="https://api.z.ai/api/paas/v4/",
        openai_compatible=True,
        models=_GLM_MODELS,
        region_group="glm",
    ),
    "glm-cn": ProviderDef(
        display_name="GLM (China)",
        api_key_env="ZHIPU_CN_API_KEY",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        openai_compatible=True,
        models=_GLM_MODELS,
        region_group="glm",
    ),
    # --- Dual-region: MiniMax ---
    "minimax": ProviderDef(
        display_name="MiniMax",
        api_key_env="MINIMAX_API_KEY",
        base_url="https://api.minimax.io/v1",
        openai_compatible=True,
        models=_MINIMAX_MODELS,
        region_group="minimax",
    ),
    "minimax-cn": ProviderDef(
        display_name="MiniMax (China)",
        api_key_env="MINIMAX_CN_API_KEY",
        base_url="https://api.minimaxi.com/v1",
        openai_compatible=True,
        models=_MINIMAX_MODELS,
        region_group="minimax",
    ),
    # --- Special providers ---
    "openrouter": ProviderDef(
        display_name="OpenRouter",
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        openai_compatible=True,
        accepts_any_model=True,
        models=None,  # fetched dynamically from API
    ),
    "ollama": ProviderDef(
        display_name="Ollama",
        api_key_env=None,  # local runtime, no authentication
        base_url="http://localhost:11434/v1",
        openai_compatible=True,
        accepts_any_model=True,
        models=_OLLAMA_MODELS,
    ),
}


# ---------------------------------------------------------------------------
# Public accessor functions
# ---------------------------------------------------------------------------


def get_provider(key: str) -> ProviderDef:
    """Return the provider definition for ``key`` (case-insensitive).

    Raises ``KeyError`` if the provider is not registered.
    """
    k = key.lower().strip()
    try:
        return PROVIDERS[k]
    except KeyError:
        raise KeyError(
            f"Unknown provider: {key!r}. "
            f"Known providers: {sorted(PROVIDERS)}"
        ) from None


def has_provider(key: str) -> bool:
    """Check whether ``key`` is a registered provider (case-insensitive)."""
    return key.lower().strip() in PROVIDERS


def all_providers() -> List[ProviderDef]:
    """All registered providers in registration (CLI dropdown) order."""
    return list(PROVIDERS.values())


def provider_keys() -> List[str]:
    """All provider key strings in registration order."""
    return list(PROVIDERS.keys())


def openai_compatible_providers() -> Tuple[str, ...]:
    """Provider keys routed through ``OpenAIClient``."""
    return tuple(k for k, v in PROVIDERS.items() if v.openai_compatible)


def get_base_url(key: str) -> Optional[str]:
    """Default base URL for a provider, or ``None`` for SDK-default providers."""
    return get_provider(key).base_url


def get_region_siblings(group: str) -> List[ProviderDef]:
    """Return all providers sharing the same ``region_group``.

    For example, ``get_region_siblings("qwen")`` returns both the
    international and China ``ProviderDef`` entries.
    """
    return [v for v in PROVIDERS.values() if v.region_group == group]

"""Single source of truth for provider-level LLM metadata.

Every provider-level fact — display name, default base URLs (CLI-facing and
client-facing), the env var override, the API-key env var, the curated model
list, which OpenAI-compatible chat subclass to use, whether the Responses API
applies, and how dual-region providers pair up — is declared here, once per
provider key, as a :class:`ProviderSpec`.

The older modules (:mod:`model_catalog`, :mod:`api_key_env`, :mod:`validators`,
and the base-URL/subclass branches in :mod:`openai_client`/:mod:`factory`, plus
the CLI provider table and region menus in :mod:`cli.utils`) now *derive* their
behavior from this registry instead of each maintaining a parallel table. That
removes the "fix a regional endpoint in one table, forget the other two" class
of bug: a dual-region provider's URL, key, and model list are stated in exactly
one place here.

Per-*model* API quirks (which model id rejects ``tool_choice``, needs
``reasoning_split``, etc.) deliberately live elsewhere, in
:mod:`tradingagents.llm_clients.capabilities`, because they are keyed by model
id, not by provider — a single provider hosts models with differing quirks. The
registry only carries the *provider*→chat-subclass routing
(:attr:`ProviderSpec.chat_variant`); the subclass then consults the capability
table per model.

This module is a dependency *leaf*: it imports nothing from
``tradingagents.llm_clients.*`` or ``cli.*`` so that all of those consumers can
import it without creating an import cycle.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

ModelOption = Tuple[str, str]
# One provider's curated lists: selection mode ("quick"/"deep") -> options.
ModeOptions = Dict[str, List[ModelOption]]
# The whole catalog: provider key -> mode -> options. Kept so model_catalog can
# re-export its historical ``ProviderModeOptions`` alias with the same meaning.
ProviderModeOptions = Dict[str, ModeOptions]


# --------------------------------------------------------------------------- #
# Curated model lists (CLI dropdown). Dual-region providers reference the SAME
# object so the sharing is explicit and impossible to half-update.
# --------------------------------------------------------------------------- #

OPENAI_MODELS: ModeOptions = {
    "quick": [
        ("GPT-5.4 Mini - Fast, strong coding and tool use", "gpt-5.4-mini"),
        ("GPT-5.4 Nano - Cheapest, high-volume tasks", "gpt-5.4-nano"),
        ("GPT-5.5 - Latest frontier, 1M context", "gpt-5.5"),
        ("GPT-4.1 - Smartest non-reasoning model", "gpt-4.1"),
    ],
    "deep": [
        ("GPT-5.5 - Latest frontier, 1M context", "gpt-5.5"),
        ("GPT-5.4 - Previous-gen frontier, 1M context, cost-effective", "gpt-5.4"),
        ("GPT-5.2 - Strong reasoning, cost-effective", "gpt-5.2"),
        ("GPT-5.5 Pro - Most capable, expensive ($30/$180 per 1M tokens)", "gpt-5.5-pro"),
    ],
}

ANTHROPIC_MODELS: ModeOptions = {
    "quick": [
        ("Claude Sonnet 4.6 - Best speed and intelligence balance", "claude-sonnet-4-6"),
        ("Claude Haiku 4.5 - Fastest with near-frontier intelligence", "claude-haiku-4-5"),
        ("Claude Sonnet 4.5 - High-performance for agents and coding", "claude-sonnet-4-5"),
    ],
    "deep": [
        ("Claude Opus 4.8 - Latest frontier, agentic coding and reasoning", "claude-opus-4-8"),
        ("Claude Opus 4.7 - Previous frontier, long-running agents", "claude-opus-4-7"),
        ("Claude Opus 4.6 - Frontier intelligence, agents and coding", "claude-opus-4-6"),
        ("Claude Sonnet 4.6 - Best speed and intelligence balance", "claude-sonnet-4-6"),
    ],
}

GOOGLE_MODELS: ModeOptions = {
    "quick": [
        ("Gemini 3.5 Flash - Latest, frontier agentic + coding (GA)", "gemini-3.5-flash"),
        ("Gemini 3.1 Flash Lite - Most cost-efficient (GA)", "gemini-3.1-flash-lite"),
        ("Gemini 2.5 Flash - Balanced, stable", "gemini-2.5-flash"),
        ("Gemini 2.5 Flash Lite - Fast, low-cost", "gemini-2.5-flash-lite"),
    ],
    "deep": [
        ("Gemini 3.1 Pro - Reasoning-first, complex workflows (preview)", "gemini-3.1-pro-preview"),
        ("Gemini 3.5 Flash - Latest GA, strong agentic + coding", "gemini-3.5-flash"),
        ("Gemini 2.5 Pro - Stable pro model", "gemini-2.5-pro"),
        ("Gemini 2.5 Flash - Balanced, stable", "gemini-2.5-flash"),
    ],
}

XAI_MODELS: ModeOptions = {
    "quick": [
        ("Grok 4.3 - Latest flagship, fast with built-in reasoning", "grok-4.3"),
        ("Grok Build 0.1 - Coding-specialized, 256K ctx", "grok-build-0.1"),
        ("Grok 4 Fast (Non-Reasoning) - Speed optimized", "grok-4-fast-non-reasoning"),
    ],
    "deep": [
        ("Grok 4.3 - Latest flagship, built-in reasoning, 1M ctx", "grok-4.3"),
        ("Grok 4.20 (Reasoning) - Previous-gen reasoning", "grok-4.20-0309-reasoning"),
        ("Grok 4 Fast (Reasoning) - High-performance", "grok-4-fast-reasoning"),
        ("Grok 4 - Flagship (dated build)", "grok-4-0709"),
    ],
}

DEEPSEEK_MODELS: ModeOptions = {
    "quick": [
        ("DeepSeek V4 Flash - Latest V4 fast model", "deepseek-v4-flash"),
        ("DeepSeek V3.2", "deepseek-chat"),
        ("Custom model ID", "custom"),
    ],
    "deep": [
        ("DeepSeek V4 Pro - Latest V4 flagship model", "deepseek-v4-pro"),
        ("DeepSeek V3.2 (thinking)", "deepseek-reasoner"),
        ("DeepSeek V3.2", "deepseek-chat"),
        ("Custom model ID", "custom"),
    ],
}

# GLM via Z.AI (international) and BigModel (China) host the same model IDs.
# Source: docs.z.ai (GLM Coding Plan supported models + LLM guides).
# All GLM 4.7+ entries support thinking mode via thinking={"type":"enabled"}.
GLM_MODELS: ModeOptions = {
    "quick": [
        ("GLM-5-Turbo - Fast, switchable thinking modes", "glm-5-turbo"),
        ("GLM-4.7 - Previous-gen flagship", "glm-4.7"),
        ("GLM-4.5-Air - Lightweight, cost-efficient", "glm-4.5-air"),
        ("Custom model ID", "custom"),
    ],
    "deep": [
        ("GLM-5.1 - Latest flagship, 204K ctx", "glm-5.1"),
        ("GLM-5 - Flagship, 204K ctx", "glm-5"),
        ("GLM-4.7 - Previous-gen flagship", "glm-4.7"),
        ("Custom model ID", "custom"),
    ],
}

# Qwen global (dashscope-intl) and CN (dashscope) endpoints share model IDs.
# Source: modelstudio.console.alibabacloud.com (Featured Models).
#
# Only versioned IDs are exposed; the version-less aliases (qwen-plus,
# qwen-flash) auto-upgrade when Alibaba rotates the backing model, so users who
# want a specific generation pick it explicitly and users who want auto-latest
# enter the alias via "Custom model ID".
QWEN_MODELS: ModeOptions = {
    "quick": [
        ("Qwen 3.6 Flash - Latest fast, agentic coding + vision-language", "qwen3.6-flash"),
        ("Qwen 3.5 Flash - Previous-gen fast", "qwen3.5-flash"),
        ("Custom model ID", "custom"),
    ],
    "deep": [
        ("Qwen 3.7 Max - Latest flagship reasoning agent, 1M ctx", "qwen3.7-max"),
        ("Qwen 3.6 Plus - Vision-language, agentic coding", "qwen3.6-plus"),
        ("Qwen 3.5 Plus - Previous-gen flagship", "qwen3.5-plus"),
        ("Custom model ID", "custom"),
    ],
}

# MiniMax global (.io) and China (.com) regions share model IDs. Full official
# lineup per platform.minimax.io/docs/api-reference/text-openai-api. All M2.x
# models share a 204,800-token context window.
MINIMAX_MODELS: ModeOptions = {
    "quick": [
        ("MiniMax-M2.7-highspeed - Faster M2.7, 204K ctx, ~100 TPS", "MiniMax-M2.7-highspeed"),
        ("MiniMax-M2.5-highspeed - Previous-gen highspeed, 204K ctx", "MiniMax-M2.5-highspeed"),
        ("MiniMax-M2.1-highspeed - M2.1 highspeed, 204K ctx", "MiniMax-M2.1-highspeed"),
        ("Custom model ID", "custom"),
    ],
    "deep": [
        ("MiniMax-M2.7 - Flagship, SOTA on coding/agent benchmarks, 204K ctx", "MiniMax-M2.7"),
        ("MiniMax-M2.7-highspeed - Same quality as M2.7, ~100 TPS", "MiniMax-M2.7-highspeed"),
        ("MiniMax-M2.5 - Previous-gen flagship, 204K ctx", "MiniMax-M2.5"),
        ("MiniMax-M2.1 - Earlier M2 line, 204K ctx", "MiniMax-M2.1"),
        ("MiniMax-M2 - Base M2, 204K ctx", "MiniMax-M2"),
        ("Custom model ID", "custom"),
    ],
}

# Ollama labels intentionally omit a "(local)" marker: the endpoint is
# configurable via OLLAMA_BASE_URL, so the same labels apply whether ollama-serve
# runs on localhost or a remote host. "Custom model ID" (kept LAST so the curated
# defaults aren't pushed off-screen) lets users pick anything they've pulled.
OLLAMA_MODELS: ModeOptions = {
    "quick": [
        ("Qwen3:latest (8B)", "qwen3:latest"),
        ("GPT-OSS:latest (20B)", "gpt-oss:latest"),
        ("GLM-4.7-Flash:latest (30B)", "glm-4.7-flash:latest"),
        ("Custom model ID", "custom"),
    ],
    "deep": [
        ("GLM-4.7-Flash:latest (30B)", "glm-4.7-flash:latest"),
        ("GPT-OSS:latest (20B)", "gpt-oss:latest"),
        ("Qwen3:latest (8B)", "qwen3:latest"),
        ("Custom model ID", "custom"),
    ],
}


# --------------------------------------------------------------------------- #
# Provider specification
# --------------------------------------------------------------------------- #

# Client backends. Only "openai" providers route through the OpenAI-compatible
# OpenAIClient; the others have dedicated clients.
ClientKind = str  # "openai" | "anthropic" | "google" | "azure"

# Which NormalizedChatOpenAI subclass an OpenAI-compatible provider uses. Only
# meaningful when ``client == "openai"``.
ChatVariant = str  # "default" | "deepseek" | "minimax"


@dataclass(frozen=True)
class ProviderSpec:
    """Everything provider-level the rest of the codebase needs to know."""

    key: str
    display_name: str
    client: ClientKind
    api_key_env: Optional[str]
    # What the CLI dropdown shows and cli.utils.provider_default_url returns.
    menu_base_url: Optional[str] = None
    # What gets injected into the client. None => use the SDK's own default
    # endpoint (and, for OpenAI, no explicit base-URL/key injection at all).
    # Differs from menu_base_url only for native OpenAI (menu shows the
    # canonical /v1 URL for display; the client passes None so langchain uses
    # its default and auto-reads OPENAI_API_KEY).
    client_base_url: Optional[str] = None
    # Env var that overrides the base URL at call time (only Ollama, today).
    base_url_env: Optional[str] = None
    # Curated CLI model list. None for providers whose models are dynamic
    # (OpenRouter, fetched live) or arbitrary (Azure deployment names). A
    # provider appears in MODEL_OPTIONS / known_models iff this is not None.
    models: Optional[ModeOptions] = None
    # Skip model validation and accept any model id. NOT derivable from
    # ``models is None``: Ollama has a curated list AND accepts arbitrary
    # pulled models, so it needs this flag while still carrying ``models``.
    accepts_any_model: bool = False
    chat_variant: ChatVariant = "default"
    uses_responses_api: bool = False
    # Dual-region grouping. Both endpoints of a pair carry the same
    # ``region_group``; the secondary (China) endpoint sets in_main_menu=False
    # so it is reached only through the region sub-prompt.
    region_group: Optional[str] = None
    region_label: Optional[str] = None
    in_main_menu: bool = True


# Declared in CLI dropdown order. The "-cn" secondary endpoints sit next to
# their international sibling for readability but are kept out of the main menu
# (in_main_menu=False) — they are selected via the region sub-prompt. Keys,
# display names, env vars, and URLs here are the canonical values; do not
# duplicate them elsewhere.
PROVIDERS: Dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        key="openai",
        display_name="OpenAI",
        client="openai",
        api_key_env="OPENAI_API_KEY",
        menu_base_url="https://api.openai.com/v1",
        client_base_url=None,  # SDK default + langchain auto-reads OPENAI_API_KEY
        models=OPENAI_MODELS,
        uses_responses_api=True,
    ),
    "google": ProviderSpec(
        key="google",
        display_name="Google",
        client="google",
        api_key_env="GOOGLE_API_KEY",
        menu_base_url=None,  # uses the Gemini SDK default endpoint
        models=GOOGLE_MODELS,
    ),
    "anthropic": ProviderSpec(
        key="anthropic",
        display_name="Anthropic",
        client="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        menu_base_url="https://api.anthropic.com/",
        models=ANTHROPIC_MODELS,
    ),
    "xai": ProviderSpec(
        key="xai",
        display_name="xAI",
        client="openai",
        api_key_env="XAI_API_KEY",
        menu_base_url="https://api.x.ai/v1",
        client_base_url="https://api.x.ai/v1",
        models=XAI_MODELS,
    ),
    "deepseek": ProviderSpec(
        key="deepseek",
        display_name="DeepSeek",
        client="openai",
        api_key_env="DEEPSEEK_API_KEY",
        menu_base_url="https://api.deepseek.com",
        client_base_url="https://api.deepseek.com",
        models=DEEPSEEK_MODELS,
        chat_variant="deepseek",
    ),
    # Qwen — Alibaba DashScope. International and China accounts cannot share a
    # key (#758), so the two endpoints are distinct providers that share one
    # model list.
    "qwen": ProviderSpec(
        key="qwen",
        display_name="Qwen",
        client="openai",
        api_key_env="DASHSCOPE_API_KEY",
        menu_base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        client_base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        models=QWEN_MODELS,
        region_group="qwen",
        region_label="International",
    ),
    "qwen-cn": ProviderSpec(
        key="qwen-cn",
        display_name="Qwen",
        client="openai",
        api_key_env="DASHSCOPE_CN_API_KEY",
        menu_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        client_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        models=QWEN_MODELS,
        region_group="qwen",
        region_label="China",
        in_main_menu=False,
    ),
    # GLM — Zhipu. Z.AI (international, ZHIPU_API_KEY) is the top-level default;
    # BigModel (China, ZHIPU_CN_API_KEY) is the "-cn" endpoint.
    "glm": ProviderSpec(
        key="glm",
        display_name="GLM",
        client="openai",
        api_key_env="ZHIPU_API_KEY",
        menu_base_url="https://api.z.ai/api/paas/v4/",
        client_base_url="https://api.z.ai/api/paas/v4/",
        models=GLM_MODELS,
        region_group="glm",
        region_label="International",
    ),
    "glm-cn": ProviderSpec(
        key="glm-cn",
        display_name="GLM",
        client="openai",
        api_key_env="ZHIPU_CN_API_KEY",
        menu_base_url="https://open.bigmodel.cn/api/paas/v4/",
        client_base_url="https://open.bigmodel.cn/api/paas/v4/",
        models=GLM_MODELS,
        region_group="glm",
        region_label="China",
        in_main_menu=False,
    ),
    # MiniMax — global (.io) and China (.com) accounts cannot share a key.
    "minimax": ProviderSpec(
        key="minimax",
        display_name="MiniMax",
        client="openai",
        api_key_env="MINIMAX_API_KEY",
        menu_base_url="https://api.minimax.io/v1",
        client_base_url="https://api.minimax.io/v1",
        models=MINIMAX_MODELS,
        chat_variant="minimax",
        region_group="minimax",
        region_label="Global",
    ),
    "minimax-cn": ProviderSpec(
        key="minimax-cn",
        display_name="MiniMax",
        client="openai",
        api_key_env="MINIMAX_CN_API_KEY",
        menu_base_url="https://api.minimaxi.com/v1",
        client_base_url="https://api.minimaxi.com/v1",
        models=MINIMAX_MODELS,
        chat_variant="minimax",
        region_group="minimax",
        region_label="China",
        in_main_menu=False,
    ),
    "openrouter": ProviderSpec(
        key="openrouter",
        display_name="OpenRouter",
        client="openai",
        api_key_env="OPENROUTER_API_KEY",
        menu_base_url="https://openrouter.ai/api/v1",
        client_base_url="https://openrouter.ai/api/v1",
        models=None,  # fetched dynamically by the CLI
        accepts_any_model=True,
    ),
    "azure": ProviderSpec(
        key="azure",
        display_name="Azure OpenAI",
        client="azure",
        api_key_env="AZURE_OPENAI_API_KEY",
        menu_base_url=None,  # endpoint configured via AZURE_OPENAI_ENDPOINT
        models=None,  # any deployed model name
    ),
    "ollama": ProviderSpec(
        key="ollama",
        display_name="Ollama",
        client="openai",
        api_key_env=None,  # local runtimes do not authenticate
        menu_base_url="http://localhost:11434/v1",
        client_base_url="http://localhost:11434/v1",
        base_url_env="OLLAMA_BASE_URL",
        models=OLLAMA_MODELS,
        accepts_any_model=True,
    ),
}


# --------------------------------------------------------------------------- #
# Accessors. Lookups are case-insensitive and never raise for unknown providers
# (they return None / a permissive default) so callers can probe freely; the
# model-list accessors keep their historical KeyError-on-missing contract.
# --------------------------------------------------------------------------- #


def get_spec(provider: str) -> Optional[ProviderSpec]:
    """Return the :class:`ProviderSpec` for ``provider``, or None if unknown."""
    return PROVIDERS.get(provider.lower())


def _env_override(spec: ProviderSpec) -> Optional[str]:
    """Call-time base-URL override from ``spec.base_url_env`` if set & present."""
    if spec.base_url_env:
        return os.environ.get(spec.base_url_env) or None
    return None


def resolve_base_url(provider: str) -> Optional[str]:
    """Base URL to inject into the client, with env override applied at call time.

    Returns the provider's ``client_base_url`` (None for native OpenAI, which
    means "use the SDK default and inject no key"), unless an env override
    (e.g. ``OLLAMA_BASE_URL``) is set. Unknown providers return None.

    Evaluated at call time, not import time, so a test that monkeypatches the
    env after import still sees the override.
    """
    spec = get_spec(provider)
    if spec is None:
        return None
    return _env_override(spec) or spec.client_base_url


def menu_url(provider: str) -> Optional[str]:
    """Default backend URL shown in the CLI provider dropdown.

    Mirrors the historical ``cli.utils.provider_default_url``: only providers
    listed in the main menu resolve to a URL (the region "-cn" endpoints, which
    are reached via the secondary prompt, return None), the Ollama env override
    is honored, and unknown providers return None.
    """
    spec = get_spec(provider)
    if spec is None or not spec.in_main_menu:
        return None
    return _env_override(spec) or spec.menu_base_url


def api_key_env(provider: str) -> Optional[str]:
    """Env var holding ``provider``'s API key, or None if it needs/has none."""
    spec = get_spec(provider)
    return spec.api_key_env if spec else None


def is_openai_compatible(provider: str) -> bool:
    """True if ``provider`` routes through the OpenAI-compatible client."""
    spec = get_spec(provider)
    return bool(spec and spec.client == "openai")


def client_kind(provider: str) -> Optional[ClientKind]:
    """Return which client backend ``provider`` uses, or None if unknown."""
    spec = get_spec(provider)
    return spec.client if spec else None


def chat_variant(provider: str) -> ChatVariant:
    """Return the OpenAI-compatible chat subclass key for ``provider``."""
    spec = get_spec(provider)
    return spec.chat_variant if spec else "default"


def uses_responses_api(provider: str) -> bool:
    """True if ``provider`` should use the OpenAI Responses API."""
    spec = get_spec(provider)
    return bool(spec and spec.uses_responses_api)


def accepts_any_model(provider: str) -> bool:
    """True if model-name validation should be skipped for ``provider``."""
    spec = get_spec(provider)
    return bool(spec and spec.accepts_any_model)


def model_options(provider: str, mode: str) -> List[ModelOption]:
    """Curated model options for ``provider`` and selection ``mode``.

    Raises KeyError for providers without a curated list (Azure, OpenRouter) or
    for an unknown provider/mode, matching the historical
    ``model_catalog.get_model_options`` contract.
    """
    spec = get_spec(provider)
    if spec is None or spec.models is None:
        raise KeyError(provider)
    return spec.models[mode]


def known_models() -> Dict[str, List[str]]:
    """Map each provider with a curated list to its sorted unique model ids."""
    return {
        spec.key: sorted(
            {value for options in spec.models.values() for _, value in options}
        )
        for spec in PROVIDERS.values()
        if spec.models is not None
    }


def main_menu_providers() -> List[ProviderSpec]:
    """Specs shown in the top-level CLI provider dropdown, in display order."""
    return [spec for spec in PROVIDERS.values() if spec.in_main_menu]


def region_group_of(provider: str) -> Optional[str]:
    """Return ``provider``'s dual-region group key, or None if it has none."""
    spec = get_spec(provider)
    return spec.region_group if spec else None


def region_members(group: str) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """Return ``(provider_key, resolved_base_url, region_label)`` for a group.

    Ordered with the main-menu (international/global) endpoint first, then the
    "-cn" endpoint — matching the order the region sub-prompts present them.
    """
    members = [spec for spec in PROVIDERS.values() if spec.region_group == group]
    members.sort(key=lambda spec: not spec.in_main_menu)
    return [(spec.key, resolve_base_url(spec.key), spec.region_label) for spec in members]

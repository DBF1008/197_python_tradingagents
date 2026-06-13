"""Canonical provider -> API-key env-var mapping.

A single source of truth for which environment variable holds the API key for
each supported LLM provider. Used by the CLI's interactive key prompt
(cli/utils.ensure_api_key) and by anything else that needs to ask "does this
provider require a key, and which env var is it?".

The mapping itself is derived from the provider registry
(:mod:`tradingagents.llm_clients.provider_registry`) — to add a provider or
change its key env var, edit the registry's :data:`ProviderSpec`, not this file.
``PROVIDER_API_KEY_ENV`` and ``get_api_key_env`` remain here as the stable
import surface.
"""

from __future__ import annotations

from typing import Optional

from .provider_registry import PROVIDERS
from .provider_registry import api_key_env as _registry_api_key_env


# Derived from the registry so the CLI can never present a provider whose key
# env var isn't defined. Dual-region providers each carry their own account
# (keys are not interchangeable between international and China endpoints), and
# local runtimes (ollama) authenticate with no key (None).
PROVIDER_API_KEY_ENV: dict[str, Optional[str]] = {
    key: spec.api_key_env for key, spec in PROVIDERS.items()
}


def get_api_key_env(provider: str) -> Optional[str]:
    """Return the env var name for `provider`'s API key, or None if not applicable.

    Unknown providers also return None — callers should treat that as
    "no key check possible" rather than as "no key required".
    """
    return _registry_api_key_env(provider)

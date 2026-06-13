"""Canonical provider -> API-key env-var mapping.

Derived from the central provider registry (``registry.py``). When adding
a new provider, register it in the registry — the mapping below and the
CLI's interactive key-prompt will pick it up automatically.
"""

from __future__ import annotations

from typing import Optional

from .registry import PROVIDERS, get_provider


PROVIDER_API_KEY_ENV: dict[str, Optional[str]] = {
    key: pdef.api_key_env for key, pdef in PROVIDERS.items()
}


def get_api_key_env(provider: str) -> Optional[str]:
    """Return the env var name for `provider`'s API key, or None if not applicable.

    Unknown providers also return None — callers should treat that as
    "no key check possible" rather than as "no key required".
    """
    try:
        return get_provider(provider).api_key_env
    except KeyError:
        return None

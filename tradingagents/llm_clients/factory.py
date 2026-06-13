from typing import Optional

from . import provider_registry
from .base_client import BaseLLMClient


def create_llm_client(
    provider: str,
    model: str,
    base_url: Optional[str] = None,
    **kwargs,
) -> BaseLLMClient:
    """Create an LLM client for the specified provider.

    The provider→client-backend routing is derived from the provider registry
    (:mod:`tradingagents.llm_clients.provider_registry`). Provider client
    modules are imported lazily so that simply importing this factory (e.g.
    during test collection) does not pull in heavy LLM SDKs or fail when their
    API keys are absent. The registry itself is a lightweight dependency leaf.

    Args:
        provider: LLM provider name
        model: Model name/identifier
        base_url: Optional base URL for API endpoint
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured BaseLLMClient instance

    Raises:
        ValueError: If provider is not supported
    """
    provider_lower = provider.lower()
    spec = provider_registry.get_spec(provider_lower)
    if spec is None:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    if spec.client == "openai":
        from .openai_client import OpenAIClient
        return OpenAIClient(model, base_url, provider=provider_lower, **kwargs)

    if spec.client == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(model, base_url, **kwargs)

    if spec.client == "google":
        from .google_client import GoogleClient
        return GoogleClient(model, base_url, **kwargs)

    if spec.client == "azure":
        from .azure_client import AzureOpenAIClient
        return AzureOpenAIClient(model, base_url, **kwargs)

    raise ValueError(f"Unsupported LLM provider: {provider}")

"""Tests for the central provider registry and cross-module consistency.

Verifies that the registry is the single source of truth for provider
metadata, that dual-region providers are correctly configured, that
special providers (Ollama, OpenRouter, Azure) are handled correctly,
and that CLI-derived data matches the registry.
"""

from __future__ import annotations

import pytest

from tradingagents.llm_clients.registry import (
    PROVIDERS,
    ProviderDef,
    all_providers,
    get_base_url,
    get_provider,
    get_region_siblings,
    has_provider,
    openai_compatible_providers,
    provider_keys,
)


# ---------------------------------------------------------------------------
# Registry consistency
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRegistryConsistency:
    def test_all_14_providers_registered(self):
        """Every known provider key has a registry entry."""
        expected = {
            "openai", "google", "anthropic", "azure", "xai", "deepseek",
            "qwen", "qwen-cn", "glm", "glm-cn", "minimax", "minimax-cn",
            "openrouter", "ollama",
        }
        assert expected == set(provider_keys())

    def test_provider_keys_are_lowercase(self):
        assert all(k == k.lower() for k in provider_keys())

    def test_frozen_dataclass(self):
        """ProviderDef instances are immutable."""
        with pytest.raises(Exception):
            get_provider("openai").display_name = "changed"  # type: ignore[misc]

    def test_case_insensitive_lookup(self):
        assert get_provider("OpenAI") is get_provider("openai")
        assert get_provider("QWEN-CN") is get_provider("qwen-cn")
        assert get_provider("DeepSeek") is get_provider("deepseek")

    def test_has_provider(self):
        assert has_provider("openai")
        assert has_provider("OpenAI")
        assert not has_provider("not-a-real-provider")

    def test_unknown_provider_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown provider"):
            get_provider("not-a-real-provider")

    def test_all_providers_returns_list(self):
        providers = all_providers()
        assert isinstance(providers, list)
        assert len(providers) == 14
        assert all(isinstance(p, ProviderDef) for p in providers)

    def test_registry_order_matches_cli_dropdown(self):
        """Registration order determines CLI dropdown order."""
        keys = provider_keys()
        # The main dropdown shows these in order (regional variants filtered)
        dropdown_keys = [
            k for k in keys if k not in ("qwen-cn", "glm-cn", "minimax-cn")
        ]
        assert dropdown_keys.index("openai") < dropdown_keys.index("google")
        assert dropdown_keys.index("google") < dropdown_keys.index("anthropic")
        assert dropdown_keys.index("deepseek") < dropdown_keys.index("qwen")
        assert dropdown_keys.index("qwen") < dropdown_keys.index("glm")
        assert dropdown_keys.index("glm") < dropdown_keys.index("minimax")
        assert dropdown_keys.index("minimax") < dropdown_keys.index("ollama")


# ---------------------------------------------------------------------------
# Dual-region providers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDualRegionProviders:
    @pytest.mark.parametrize(
        "group,keys",
        [
            ("qwen", ["qwen", "qwen-cn"]),
            ("glm", ["glm", "glm-cn"]),
            ("minimax", ["minimax", "minimax-cn"]),
        ],
    )
    def test_siblings_share_models(self, group, keys):
        """Dual-region siblings must reference the same model dict object."""
        siblings = get_region_siblings(group)
        assert len(siblings) == 2
        # Identity check — must be the same object, not just equal
        assert siblings[0].models is siblings[1].models

    @pytest.mark.parametrize("group", ["qwen", "glm", "minimax"])
    def test_siblings_have_different_urls(self, group):
        siblings = get_region_siblings(group)
        assert siblings[0].base_url != siblings[1].base_url

    @pytest.mark.parametrize("group", ["qwen", "glm", "minimax"])
    def test_siblings_have_different_api_keys(self, group):
        siblings = get_region_siblings(group)
        assert siblings[0].api_key_env != siblings[1].api_key_env

    @pytest.mark.parametrize("group", ["qwen", "glm", "minimax"])
    def test_siblings_share_region_group(self, group):
        siblings = get_region_siblings(group)
        assert all(s.region_group == group for s in siblings)

    def test_glm_international_url_correct(self):
        """Regression: GLM international must use api.z.ai, not open.bigmodel.cn."""
        glm = get_provider("glm")
        assert "api.z.ai" in glm.base_url
        assert "bigmodel" not in glm.base_url

    def test_glm_cn_url_correct(self):
        glm_cn = get_provider("glm-cn")
        assert "open.bigmodel.cn" in glm_cn.base_url

    def test_qwen_international_url_correct(self):
        qwen = get_provider("qwen")
        assert "dashscope-intl" in qwen.base_url

    def test_qwen_cn_url_correct(self):
        qwen_cn = get_provider("qwen-cn")
        assert "dashscope-intl" not in qwen_cn.base_url
        assert "dashscope" in qwen_cn.base_url

    def test_minimax_international_url_correct(self):
        minimax = get_provider("minimax")
        assert "minimax.io" in minimax.base_url

    def test_minimax_cn_url_correct(self):
        minimax_cn = get_provider("minimax-cn")
        assert "minimaxi.com" in minimax_cn.base_url


# ---------------------------------------------------------------------------
# Special providers: Ollama, OpenRouter, Azure
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSpecialProviders:
    def test_ollama_accepts_any_model(self):
        assert get_provider("ollama").accepts_any_model is True

    def test_ollama_has_no_api_key(self):
        assert get_provider("ollama").api_key_env is None

    def test_ollama_base_url_default(self):
        assert get_provider("ollama").base_url == "http://localhost:11434/v1"

    def test_ollama_is_openai_compatible(self):
        assert get_provider("ollama").openai_compatible is True

    def test_ollama_has_model_suggestions(self):
        """Ollama has model suggestions but accepts any model."""
        assert get_provider("ollama").models is not None

    def test_openrouter_accepts_any_model(self):
        assert get_provider("openrouter").accepts_any_model is True

    def test_openrouter_has_dynamic_models(self):
        """OpenRouter models are fetched dynamically, not in registry."""
        assert get_provider("openrouter").models is None

    def test_azure_accepts_any_model(self):
        assert get_provider("azure").accepts_any_model is True

    def test_azure_has_no_models(self):
        """Azure deployment names are free-form."""
        assert get_provider("azure").models is None

    def test_azure_has_no_base_url(self):
        """Azure uses AZURE_OPENAI_ENDPOINT env var, not a fixed URL."""
        assert get_provider("azure").base_url is None


# ---------------------------------------------------------------------------
# DeepSeek/MiniMax capability alignment with registry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCapabilityRegistryAlignment:
    """Verify the registry's provider routing aligns with capability quirks."""

    def test_deepseek_is_openai_compatible(self):
        assert get_provider("deepseek").openai_compatible is True

    def test_minimax_both_regions_openai_compatible(self):
        assert get_provider("minimax").openai_compatible is True
        assert get_provider("minimax-cn").openai_compatible is True

    def test_deepseek_models_in_catalog(self):
        """All DeepSeek model IDs in the catalog resolve capabilities."""
        from tradingagents.llm_clients.capabilities import get_capabilities

        models = get_provider("deepseek").models
        for mode_options in models.values():
            for _, model_id in mode_options:
                if model_id == "custom":
                    continue
                caps = get_capabilities(model_id)
                assert caps.preferred_structured_method != "none"

    def test_minimax_models_have_reasoning_split(self):
        """All MiniMax M2.x models in the catalog require reasoning_split."""
        from tradingagents.llm_clients.capabilities import get_capabilities

        for key in ("minimax", "minimax-cn"):
            models = get_provider(key).models
            for mode_options in models.values():
                for _, model_id in mode_options:
                    if model_id == "custom":
                        continue
                    if model_id.startswith("MiniMax-M2"):
                        caps = get_capabilities(model_id)
                        assert caps.requires_reasoning_split is True, (
                            f"{key}/{model_id} should require reasoning_split"
                        )

    def test_deepseek_thinking_models_reject_tool_choice(self):
        """DeepSeek V4 thinking models reject tool_choice."""
        from tradingagents.llm_clients.capabilities import get_capabilities

        for model_id in ("deepseek-v4-flash", "deepseek-v4-pro", "deepseek-reasoner"):
            caps = get_capabilities(model_id)
            assert caps.supports_tool_choice is False
            assert caps.requires_reasoning_content_roundtrip is True


# ---------------------------------------------------------------------------
# OpenAI-compatible routing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOpenAICompatibleRouting:
    def test_openai_compatible_set(self):
        compat = set(openai_compatible_providers())
        expected = {
            "openai", "xai", "deepseek",
            "qwen", "qwen-cn",
            "glm", "glm-cn",
            "minimax", "minimax-cn",
            "ollama", "openrouter",
        }
        assert compat == expected

    def test_non_openai_providers(self):
        compat = set(openai_compatible_providers())
        for key in ("anthropic", "google", "azure"):
            assert key not in compat
            assert get_provider(key).openai_compatible is False


# ---------------------------------------------------------------------------
# API key env derivation from registry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestApiKeyEnvDerivation:
    def test_api_key_env_matches_registry(self):
        from tradingagents.llm_clients.api_key_env import PROVIDER_API_KEY_ENV

        for key, pdef in PROVIDERS.items():
            assert PROVIDER_API_KEY_ENV[key] == pdef.api_key_env, (
                f"api_key_env mismatch for {key}"
            )

    def test_api_key_env_count(self):
        from tradingagents.llm_clients.api_key_env import PROVIDER_API_KEY_ENV

        assert len(PROVIDER_API_KEY_ENV) == len(PROVIDERS)


# ---------------------------------------------------------------------------
# Base URL derivation from registry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBaseUrlDerivation:
    def test_get_base_url_known_provider(self):
        assert get_base_url("deepseek") == "https://api.deepseek.com"

    def test_get_base_url_ollama(self):
        assert get_base_url("ollama") == "http://localhost:11434/v1"

    def test_get_base_url_google_none(self):
        """Google uses SDK default — no explicit base URL."""
        assert get_base_url("google") is None

    def test_get_base_url_azure_none(self):
        """Azure uses AZURE_OPENAI_ENDPOINT env var — no fixed URL."""
        assert get_base_url("azure") is None


# ---------------------------------------------------------------------------
# CLI derivation from registry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCliDerivation:
    def test_provider_default_url_matches_registry(self):
        from cli.utils import provider_default_url

        for key, pdef in PROVIDERS.items():
            if key in ("qwen-cn", "glm-cn", "minimax-cn"):
                continue  # accessed via region sub-prompts
            if key == "ollama":
                continue  # Ollama has env override
            cli_url = provider_default_url(key)
            assert cli_url == pdef.base_url, (
                f"{key}: CLI={cli_url}, registry={pdef.base_url}"
            )

    def test_glm_cli_url_is_international(self):
        """Regression: CLI must show international URL for GLM default."""
        from cli.utils import provider_default_url

        url = provider_default_url("glm")
        assert "api.z.ai" in url

    def test_provider_table_has_all_dropdown_providers(self):
        from cli.utils import _llm_provider_table

        table_keys = {pk for _, pk, _ in _llm_provider_table()}
        # Main dropdown should have all non-variant providers
        for key in provider_keys():
            if key not in ("qwen-cn", "glm-cn", "minimax-cn"):
                assert key in table_keys, f"{key} missing from CLI dropdown"

    def test_provider_table_excludes_variants(self):
        from cli.utils import _llm_provider_table

        table_keys = {pk for _, pk, _ in _llm_provider_table()}
        assert "qwen-cn" not in table_keys
        assert "glm-cn" not in table_keys
        assert "minimax-cn" not in table_keys


# ---------------------------------------------------------------------------
# Validator alignment with registry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidatorAlignment:
    def test_accepts_any_model_providers_skip_validation(self):
        from tradingagents.llm_clients.validators import validate_model

        for key, pdef in PROVIDERS.items():
            if pdef.accepts_any_model:
                assert validate_model(key, "literally-anything"), (
                    f"{key} should accept any model"
                )

    def test_catalog_providers_validate_known_models(self):
        from tradingagents.llm_clients.validators import validate_model

        for key, pdef in PROVIDERS.items():
            if pdef.models is None:
                continue
            for mode_options in pdef.models.values():
                for _, model_id in mode_options:
                    if model_id == "custom":
                        continue
                    assert validate_model(key, model_id), (
                        f"{key}/{model_id} should validate"
                    )

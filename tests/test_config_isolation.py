"""Config isolation: runtime configs must be deep, independent copies.

Regression coverage for the shallow-copy pollution bug: ``DEFAULT_CONFIG.copy()``
(a *shallow* copy) and ``config or DEFAULT_CONFIG`` (no copy at all) let one
run's overrides of nested ``dict``/``list`` values — ``data_vendors``,
``tool_vendors``, ``benchmark_map``, ``global_news_queries`` — mutate the
module-level ``DEFAULT_CONFIG`` in place. The mutation then bled into every later
run, test, or graph instance, e.g. a provider/vendor override clobbering a
benchmark tweak, or a news-query edit crossing instances.

The fix routes all runtime-config construction through
``get_default_config()`` (deep copy) and makes ``TradingAgentsGraph`` deep-copy
whatever config it is handed, so:

1. mutating a runtime config never touches ``DEFAULT_CONFIG``;
2. two graph instances never share mutable nested state;
3. the TRADINGAGENTS_* env-override semantics are unchanged (they are baked into
   ``DEFAULT_CONFIG`` at import and therefore reflected in every copy).
"""

from __future__ import annotations

import importlib
import os
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

import tradingagents.default_config as default_config_module
from tradingagents.default_config import DEFAULT_CONFIG, get_default_config
from tradingagents.graph.trading_graph import TradingAgentsGraph

# The nested containers that a shallow copy would alias.
NESTED_KEYS = ("data_vendors", "tool_vendors", "benchmark_map", "global_news_queries")

_GRAPH_NS = "tradingagents.graph.trading_graph"


def _build_graph(config=None):
    """Construct a ``TradingAgentsGraph`` with its heavy collaborators mocked.

    This exercises the *real* ``__init__`` — where the config deep-copy lives —
    without creating LLM clients, mutating the dataflows global config, or
    compiling a LangGraph. The trivial collaborators (``Propagator``,
    ``Reflector``, ``SignalProcessor``, ``ConditionalLogic``) are left real
    because their constructors only store arguments. The returned graph stays
    valid after the patches exit; the tests only read ``graph.config``.
    """
    client = MagicMock()
    client.get_llm.return_value = MagicMock()
    with ExitStack() as stack:
        stack.enter_context(patch(f"{_GRAPH_NS}.create_llm_client", return_value=client))
        stack.enter_context(patch(f"{_GRAPH_NS}.set_config"))
        stack.enter_context(patch(f"{_GRAPH_NS}.TradingMemoryLog"))
        stack.enter_context(patch(f"{_GRAPH_NS}.ToolNode"))
        stack.enter_context(patch(f"{_GRAPH_NS}.GraphSetup"))
        stack.enter_context(patch(f"{_GRAPH_NS}.os.makedirs"))
        return TradingAgentsGraph(config=config)


@pytest.mark.unit
class TestGetDefaultConfig:
    """The factory hands back fully isolated, mutation-safe copies."""

    def test_returns_equal_but_distinct_object(self):
        cfg = get_default_config()
        assert cfg == DEFAULT_CONFIG
        assert cfg is not DEFAULT_CONFIG

    def test_nested_containers_are_not_aliased(self):
        cfg = get_default_config()
        for key in NESTED_KEYS:
            assert cfg[key] == DEFAULT_CONFIG[key]
            assert cfg[key] is not DEFAULT_CONFIG[key], key

    def test_two_calls_are_independent(self):
        a = get_default_config()
        b = get_default_config()
        for key in NESTED_KEYS:
            assert a[key] is not b[key], key
        a["data_vendors"]["news_data"] = "alpha_vantage"
        assert b["data_vendors"]["news_data"] == "yfinance"

    def test_deep_mutation_does_not_pollute_default(self):
        """Every kind of nested mutation must stay local to the copy."""
        cfg = get_default_config()
        # scalar
        cfg["llm_provider"] = "anthropic"
        # nested dict value change + new key
        cfg["data_vendors"]["core_stock_apis"] = "alpha_vantage"
        cfg["tool_vendors"]["get_stock_data"] = "alpha_vantage"
        cfg["benchmark_map"][".NS"] = "CHANGED"
        cfg["benchmark_map"]["NEW_SUFFIX"] = "X"
        # list append + in-place element rewrite
        cfg["global_news_queries"].append("extra macro query")
        cfg["global_news_queries"][0] = "rewritten"

        assert DEFAULT_CONFIG["llm_provider"] == "openai"
        assert DEFAULT_CONFIG["data_vendors"]["core_stock_apis"] == "yfinance"
        assert "get_stock_data" not in DEFAULT_CONFIG["tool_vendors"]
        assert DEFAULT_CONFIG["benchmark_map"][".NS"] == "^NSEI"
        assert "NEW_SUFFIX" not in DEFAULT_CONFIG["benchmark_map"]
        assert "extra macro query" not in DEFAULT_CONFIG["global_news_queries"]
        assert (
            DEFAULT_CONFIG["global_news_queries"][0]
            == "Federal Reserve interest rates inflation"
        )

    def test_mirrors_cli_and_main_mutation_pattern(self):
        """Same shape as cli/main.py run_analysis() and main.py: take a runtime
        config, overwrite provider/vendor knobs; the shared default is untouched."""
        config = get_default_config()
        config["llm_provider"] = "google"
        config["quick_think_llm"] = "gemini-3-flash-preview"
        config["data_vendors"]["news_data"] = "alpha_vantage"

        assert DEFAULT_CONFIG["llm_provider"] == "openai"
        assert DEFAULT_CONFIG["quick_think_llm"] == "gpt-5.4-mini"
        assert DEFAULT_CONFIG["data_vendors"]["news_data"] == "yfinance"


@pytest.mark.unit
class TestGraphInstanceIsolation:
    """Two ``TradingAgentsGraph`` instances never share mutable nested state."""

    def test_default_config_graph_is_isolated_copy(self):
        ta = _build_graph()  # config=None -> get_default_config()
        assert ta.config is not DEFAULT_CONFIG
        for key in NESTED_KEYS:
            assert ta.config[key] is not DEFAULT_CONFIG[key], key

    def test_two_graphs_do_not_cross_contaminate(self):
        ta1 = _build_graph()
        ta2 = _build_graph()

        assert ta1.config is not ta2.config
        for key in NESTED_KEYS:
            assert ta1.config[key] is not ta2.config[key], key

        # Mutate ta1's config across providers, vendors, benchmark, and news.
        ta1.config["llm_provider"] = "anthropic"
        ta1.config["data_vendors"]["core_stock_apis"] = "alpha_vantage"
        ta1.config["tool_vendors"]["get_news"] = "alpha_vantage"
        ta1.config["benchmark_map"][".HK"] = "CHANGED"
        ta1.config["global_news_queries"].append("ta1 only query")

        # ta2 stays pristine ...
        assert ta2.config["llm_provider"] == "openai"
        assert ta2.config["data_vendors"]["core_stock_apis"] == "yfinance"
        assert "get_news" not in ta2.config["tool_vendors"]
        assert ta2.config["benchmark_map"][".HK"] == "^HSI"
        assert "ta1 only query" not in ta2.config["global_news_queries"]

        # ... and so does the module-level default.
        assert DEFAULT_CONFIG["llm_provider"] == "openai"
        assert DEFAULT_CONFIG["data_vendors"]["core_stock_apis"] == "yfinance"
        assert "get_news" not in DEFAULT_CONFIG["tool_vendors"]
        assert DEFAULT_CONFIG["benchmark_map"][".HK"] == "^HSI"
        assert "ta1 only query" not in DEFAULT_CONFIG["global_news_queries"]

    def test_passed_config_is_deep_copied_not_aliased(self):
        """A programmatic caller's dict is copied, so neither side can mutate the
        other's nested state after construction."""
        caller_cfg = get_default_config()
        ta = _build_graph(config=caller_cfg)

        assert ta.config is not caller_cfg
        for key in NESTED_KEYS:
            assert ta.config[key] is not caller_cfg[key], key

        # Caller mutates its own dict after construction -> graph unaffected.
        caller_cfg["data_vendors"]["news_data"] = "alpha_vantage"
        caller_cfg["benchmark_map"][".T"] = "CHANGED"
        caller_cfg["global_news_queries"].append("caller only")
        assert ta.config["data_vendors"]["news_data"] == "yfinance"
        assert ta.config["benchmark_map"][".T"] == "^N225"
        assert "caller only" not in ta.config["global_news_queries"]

        # Graph mutates its own config -> caller dict and default unaffected.
        ta.config["tool_vendors"]["get_stock_data"] = "alpha_vantage"
        assert "get_stock_data" not in caller_cfg["tool_vendors"]
        assert "get_stock_data" not in DEFAULT_CONFIG["tool_vendors"]


def _reload_with_env(monkeypatch, **overrides):
    """Clear all TRADINGAGENTS_* overrides, set the given ones, reload the module.

    Mirrors tests/test_env_overrides.py so the override semantics are validated
    through the same lens, but here we assert via ``get_default_config()``.
    """
    for key in list(default_config_module._ENV_OVERRIDES):
        monkeypatch.delenv(key, raising=False)
    for key, val in overrides.items():
        monkeypatch.setenv(key, val)
    return importlib.reload(default_config_module)


@pytest.fixture(autouse=True, scope="module")
def _restore_pristine_default_config():
    """Reload default_config to an env-free state after this module's tests so a
    reload performed by an env test cannot leak into later test modules."""
    yield
    for key in list(default_config_module._ENV_OVERRIDES):
        os.environ.pop(key, None)
    importlib.reload(default_config_module)


@pytest.mark.unit
class TestEnvOverrideSemanticsPreserved:
    """Existing TRADINGAGENTS_* override behaviour is unchanged by the deep-copy
    factory: overrides are baked into DEFAULT_CONFIG and reflected in each copy."""

    def test_factory_reflects_string_overrides(self, monkeypatch):
        dc = _reload_with_env(
            monkeypatch,
            TRADINGAGENTS_LLM_PROVIDER="google",
            TRADINGAGENTS_DEEP_THINK_LLM="gemini-3-pro-preview",
            TRADINGAGENTS_OUTPUT_LANGUAGE="Chinese",
        )
        cfg = dc.get_default_config()
        assert cfg["llm_provider"] == "google"
        assert cfg["deep_think_llm"] == "gemini-3-pro-preview"
        assert cfg["output_language"] == "Chinese"
        # The copy matches the (overridden) module default exactly.
        assert cfg == dc.DEFAULT_CONFIG

    def test_factory_reflects_int_and_bool_coercion(self, monkeypatch):
        dc = _reload_with_env(
            monkeypatch,
            TRADINGAGENTS_MAX_DEBATE_ROUNDS="3",
            TRADINGAGENTS_CHECKPOINT_ENABLED="true",
        )
        cfg = dc.get_default_config()
        assert cfg["max_debate_rounds"] == 3
        assert isinstance(cfg["max_debate_rounds"], int)
        assert cfg["checkpoint_enabled"] is True

    def test_factory_default_when_env_absent(self, monkeypatch):
        dc = _reload_with_env(monkeypatch)
        cfg = dc.get_default_config()
        assert cfg["llm_provider"] == "openai"
        assert cfg["max_debate_rounds"] == 1
        assert cfg["checkpoint_enabled"] is False

    def test_empty_env_value_is_passthrough(self, monkeypatch):
        dc = _reload_with_env(
            monkeypatch,
            TRADINGAGENTS_LLM_PROVIDER="",
            TRADINGAGENTS_MAX_DEBATE_ROUNDS="",
        )
        cfg = dc.get_default_config()
        assert cfg["llm_provider"] == "openai"
        assert cfg["max_debate_rounds"] == 1

    def test_override_and_isolation_compose(self, monkeypatch):
        """Even with an env override applied, the factory still returns an
        isolated copy: mutating it must not touch the overridden default."""
        dc = _reload_with_env(monkeypatch, TRADINGAGENTS_LLM_PROVIDER="anthropic")
        cfg = dc.get_default_config()
        assert cfg["llm_provider"] == "anthropic"

        cfg["data_vendors"]["news_data"] = "alpha_vantage"
        cfg["global_news_queries"].append("extra")
        assert dc.DEFAULT_CONFIG["data_vendors"]["news_data"] == "yfinance"
        assert "extra" not in dc.DEFAULT_CONFIG["global_news_queries"]
        # The scalar override is still in place on the default.
        assert dc.DEFAULT_CONFIG["llm_provider"] == "anthropic"

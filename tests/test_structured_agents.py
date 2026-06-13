"""Tests for structured-output agents (Trader, Research Manager, Sentiment Analyst).

The Portfolio Manager has its own coverage in tests/test_memory_log.py
(which exercises the full memory-log → PM injection cycle).  This file
covers the parallel schemas, render functions, and graceful-fallback
behavior we added for the Trader, Research Manager, and Sentiment Analyst
so they share the same deterministic output shape.
"""

import logging
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from tradingagents.agents.analysts.sentiment_analyst import create_sentiment_analyst
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.schemas import (
    PortfolioRating,
    ResearchPlan,
    SentimentBand,
    SentimentReport,
    TraderAction,
    TraderProposal,
    render_research_plan,
    render_sentiment_report,
    render_trader_proposal,
)
from tradingagents.agents.trader.trader import create_trader
from tradingagents.agents.utils.structured import StructuredBinding, bind_structured


# ---------------------------------------------------------------------------
# Render functions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRenderTraderProposal:
    def test_minimal_required_fields(self):
        p = TraderProposal(action=TraderAction.HOLD, reasoning="Balanced setup; no edge.")
        md = render_trader_proposal(p)
        assert "**Action**: Hold" in md
        assert "**Reasoning**: Balanced setup; no edge." in md
        # The trailing FINAL TRANSACTION PROPOSAL line is preserved for the
        # analyst stop-signal text and any external code that greps for it.
        assert "FINAL TRANSACTION PROPOSAL: **HOLD**" in md

    def test_optional_fields_included_when_present(self):
        p = TraderProposal(
            action=TraderAction.BUY,
            reasoning="Strong technicals + fundamentals.",
            entry_price=189.5,
            stop_loss=178.0,
            position_sizing="6% of portfolio",
        )
        md = render_trader_proposal(p)
        assert "**Action**: Buy" in md
        assert "**Entry Price**: 189.5" in md
        assert "**Stop Loss**: 178.0" in md
        assert "**Position Sizing**: 6% of portfolio" in md
        assert "FINAL TRANSACTION PROPOSAL: **BUY**" in md

    def test_optional_fields_omitted_when_absent(self):
        p = TraderProposal(action=TraderAction.SELL, reasoning="Guidance cut.")
        md = render_trader_proposal(p)
        assert "Entry Price" not in md
        assert "Stop Loss" not in md
        assert "Position Sizing" not in md
        assert "FINAL TRANSACTION PROPOSAL: **SELL**" in md


@pytest.mark.unit
class TestRenderResearchPlan:
    def test_required_fields(self):
        p = ResearchPlan(
            recommendation=PortfolioRating.OVERWEIGHT,
            rationale="Bull case carried; tailwinds intact.",
            strategic_actions="Build position over two weeks; cap at 5%.",
        )
        md = render_research_plan(p)
        assert "**Recommendation**: Overweight" in md
        assert "**Rationale**: Bull case carried" in md
        assert "**Strategic Actions**: Build position" in md

    def test_all_5_tier_ratings_render(self):
        for rating in PortfolioRating:
            p = ResearchPlan(
                recommendation=rating,
                rationale="r",
                strategic_actions="s",
            )
            md = render_research_plan(p)
            assert f"**Recommendation**: {rating.value}" in md


# ---------------------------------------------------------------------------
# Trader agent: structured happy path + fallback
# ---------------------------------------------------------------------------


def _make_trader_state():
    return {
        "company_of_interest": "NVDA",
        "investment_plan": "**Recommendation**: Buy\n**Rationale**: ...\n**Strategic Actions**: ...",
    }


def _structured_trader_llm(captured: dict, proposal: TraderProposal | None = None):
    """Build a MagicMock LLM whose with_structured_output binding captures the
    prompt and returns a real TraderProposal so render_trader_proposal works.
    """
    if proposal is None:
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="Strong setup.",
        )
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or proposal
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
class TestTraderAgent:
    def test_structured_path_produces_rendered_markdown(self):
        captured = {}
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="AI capex cycle intact; institutional flows constructive.",
            entry_price=189.5,
            stop_loss=178.0,
            position_sizing="6% of portfolio",
        )
        llm = _structured_trader_llm(captured, proposal)
        trader = create_trader(llm)
        result = trader(_make_trader_state())
        plan = result["trader_investment_plan"]
        assert "**Action**: Buy" in plan
        assert "**Entry Price**: 189.5" in plan
        assert "FINAL TRANSACTION PROPOSAL: **BUY**" in plan
        # The same rendered markdown is also added to messages for downstream agents.
        assert plan in result["messages"][0].content

    def test_prompt_includes_investment_plan(self):
        captured = {}
        llm = _structured_trader_llm(captured)
        trader = create_trader(llm)
        trader(_make_trader_state())
        # The investment plan is in the user message of the captured prompt.
        prompt = captured["prompt"]
        assert any("Proposed Investment Plan" in m["content"] for m in prompt)

    def test_falls_back_to_freetext_when_structured_unavailable(self):
        plain_response = (
            "**Action**: Sell\n\nGuidance cut hits margins.\n\n"
            "FINAL TRANSACTION PROPOSAL: **SELL**"
        )
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
        llm.invoke.return_value = MagicMock(content=plain_response)
        trader = create_trader(llm)
        result = trader(_make_trader_state())
        assert result["trader_investment_plan"] == plain_response


# ---------------------------------------------------------------------------
# Research Manager agent: structured happy path + fallback
# ---------------------------------------------------------------------------


def _make_rm_state():
    return {
        "company_of_interest": "NVDA",
        "investment_debate_state": {
            "history": "Bull and bear arguments here.",
            "bull_history": "Bull says...",
            "bear_history": "Bear says...",
            "current_response": "",
            "judge_decision": "",
            "count": 1,
        },
    }


def _structured_rm_llm(captured: dict, plan: ResearchPlan | None = None):
    if plan is None:
        plan = ResearchPlan(
            recommendation=PortfolioRating.HOLD,
            rationale="Balanced view across both sides.",
            strategic_actions="Hold current position; reassess after earnings.",
        )
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or plan
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
class TestResearchManagerAgent:
    def test_structured_path_produces_rendered_markdown(self):
        captured = {}
        plan = ResearchPlan(
            recommendation=PortfolioRating.OVERWEIGHT,
            rationale="Bull case is stronger; AI tailwind intact.",
            strategic_actions="Build position gradually over two weeks.",
        )
        llm = _structured_rm_llm(captured, plan)
        rm = create_research_manager(llm)
        result = rm(_make_rm_state())
        ip = result["investment_plan"]
        assert "**Recommendation**: Overweight" in ip
        assert "**Rationale**: Bull case" in ip
        assert "**Strategic Actions**: Build position" in ip

    def test_prompt_uses_5_tier_rating_scale(self):
        """The RM prompt must list all five tiers so the schema enum matches user expectations."""
        captured = {}
        llm = _structured_rm_llm(captured)
        rm = create_research_manager(llm)
        rm(_make_rm_state())
        prompt = captured["prompt"]
        for tier in ("Buy", "Overweight", "Hold", "Underweight", "Sell"):
            assert f"**{tier}**" in prompt, f"missing {tier} in prompt"

    def test_falls_back_to_freetext_when_structured_unavailable(self):
        plain_response = "**Recommendation**: Sell\n\n**Rationale**: ...\n\n**Strategic Actions**: ..."
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
        llm.invoke.return_value = MagicMock(content=plain_response)
        rm = create_research_manager(llm)
        result = rm(_make_rm_state())
        assert result["investment_plan"] == plain_response


# ---------------------------------------------------------------------------
# Sentiment Analyst: schema, render, structured happy path + fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRenderSentimentReport:
    def test_header_contains_band_and_score(self):
        report = SentimentReport(
            overall_band=SentimentBand.BULLISH,
            overall_score=7.2,
            confidence="high",
            narrative="Source breakdown here.",
        )
        md = render_sentiment_report(report)
        assert "**Overall Sentiment:** **Bullish**" in md
        assert "(Score: 7.2/10)" in md

    def test_header_contains_confidence(self):
        report = SentimentReport(
            overall_band=SentimentBand.NEUTRAL,
            overall_score=5.0,
            confidence="low",
            narrative="Limited data.",
        )
        assert "**Confidence:** Low" in render_sentiment_report(report)

    def test_narrative_preserved_in_output(self):
        narrative = "## Breakdown\n\nStockTwits: 70% bullish.\n\n| Signal | Direction |\n|---|---|\n| News | Neutral |"
        report = SentimentReport(
            overall_band=SentimentBand.MILDLY_BULLISH,
            overall_score=6.0,
            confidence="medium",
            narrative=narrative,
        )
        assert narrative in render_sentiment_report(report)

    def test_all_six_bands_render(self):
        for band in SentimentBand:
            report = SentimentReport(
                overall_band=band, overall_score=5.0,
                confidence="medium", narrative="n",
            )
            assert band.value in render_sentiment_report(report)

    def test_score_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            SentimentReport(
                overall_band=SentimentBand.BULLISH, overall_score=11.0,
                confidence="high", narrative="n",
            )


def _make_sentiment_state():
    return {
        "company_of_interest": "NVDA",
        "trade_date": "2026-01-15",
        "asset_type": "stock",
        "messages": [],
    }


def _structured_sentiment_llm(captured: dict, report: SentimentReport | None = None):
    """MagicMock LLM whose structured binding captures the prompt and returns
    a real SentimentReport so render_sentiment_report works."""
    if report is None:
        report = SentimentReport(
            overall_band=SentimentBand.BULLISH, overall_score=7.5,
            confidence="high",
            narrative="StockTwits 75% bullish. News constructive. Reddit upbeat.",
        )
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or report
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
class TestSentimentAnalystAgent:
    def test_structured_path_produces_rendered_markdown(self):
        captured = {}
        report = SentimentReport(
            overall_band=SentimentBand.MILDLY_BEARISH, overall_score=4.0,
            confidence="medium", narrative="Mixed signals across sources.",
        )
        analyst = create_sentiment_analyst(_structured_sentiment_llm(captured, report))
        sr = analyst(_make_sentiment_state())["sentiment_report"]
        assert "**Overall Sentiment:** **Mildly Bearish**" in sr
        assert "(Score: 4.0/10)" in sr
        assert "Mixed signals across sources." in sr

    def test_sentiment_report_also_in_messages(self):
        captured = {}
        analyst = create_sentiment_analyst(_structured_sentiment_llm(captured))
        result = analyst(_make_sentiment_state())
        assert len(result["messages"]) == 1
        assert result["sentiment_report"] == result["messages"][0].content

    def test_prompt_contains_ticker(self):
        captured = {}
        create_sentiment_analyst(_structured_sentiment_llm(captured))(_make_sentiment_state())
        assert any("NVDA" in str(m) for m in captured["prompt"])

    def test_falls_back_to_freetext_when_structured_unavailable(self):
        plain = "**Overall Sentiment:** **Bearish** (Score: 3.0/10)\n**Confidence:** Low\n\nLimited data."
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
        llm.invoke.return_value = MagicMock(content=plain)
        assert create_sentiment_analyst(llm)(_make_sentiment_state())["sentiment_report"] == plain

    def test_falls_back_to_freetext_when_structured_call_fails(self):
        plain = "Fallback free-text sentiment."
        structured = MagicMock()
        structured.invoke.side_effect = ValueError("bad JSON from model")
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        llm.invoke.return_value = MagicMock(content=plain)
        assert create_sentiment_analyst(llm)(_make_sentiment_state())["sentiment_report"] == plain


# ---------------------------------------------------------------------------
# Sticky fallback — bind_structured / StructuredBinding (the unit that changed)
# ---------------------------------------------------------------------------


def _render(result):
    """Trivial render used by binding-level tests: turn a structured result
    into a deterministic string so the happy path is distinguishable from the
    free-text path (which returns ``response.content`` instead)."""
    return f"RENDERED::{result}"


def _binding_llm(*, structured_side_effect=None, structured_return="OBJ", plain_content="PLAIN"):
    """Build a MagicMock LLM and its structured binding for StructuredBinding tests.

    The structured binding's ``.invoke`` either raises (``structured_side_effect``)
    or returns ``structured_return``; the plain ``llm.invoke`` returns an object
    whose ``.content`` is ``plain_content``. Returns ``(llm, structured)`` so a
    test can assert call counts on both the structured and free-text paths.
    """
    structured = MagicMock()
    if structured_side_effect is not None:
        structured.invoke.side_effect = structured_side_effect
    else:
        structured.invoke.return_value = structured_return
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    llm.invoke.return_value = MagicMock(content=plain_content)
    return llm, structured


@pytest.mark.unit
class TestStructuredBindingStickyFallback:
    """The sticky free-text fallback, exercised directly on the helper.

    Pinning the behavior here (independently of any agent's wiring) covers the
    four required cases: first failure becomes sticky, the success path is
    untouched, distinct bindings are isolated, and the original
    unsupported-at-bind branch still degrades to free text.
    """

    def test_bind_structured_returns_a_binding(self):
        llm, _ = _binding_llm()
        binding = bind_structured(llm, TraderProposal, "Trader")
        assert isinstance(binding, StructuredBinding)
        assert binding.uses_structured_output is True

    def test_runtime_failure_makes_fallback_sticky(self):
        """First runtime failure disables structured; later calls don't retry it."""
        llm, structured = _binding_llm(
            structured_side_effect=ValueError("400 response_format unsupported")
        )
        binding = bind_structured(llm, TraderProposal, "Trader")
        assert binding.uses_structured_output is True  # supported at bind time

        first = binding.invoke("p1", _render)
        second = binding.invoke("p2", _render)

        assert first == "PLAIN"
        assert second == "PLAIN"
        # Structured attempted exactly once, then permanently disabled.
        assert structured.invoke.call_count == 1
        assert llm.invoke.call_count == 2
        assert binding.uses_structured_output is False

    def test_sticky_fallback_warns_exactly_once(self, caplog):
        """The recurring incompatibility must not spam a warning per call."""
        llm, _ = _binding_llm(structured_side_effect=ValueError("400"))
        binding = bind_structured(llm, TraderProposal, "Trader")
        with caplog.at_level(
            logging.WARNING, logger="tradingagents.agents.utils.structured"
        ):
            binding.invoke("p", _render)
            binding.invoke("p", _render)
            binding.invoke("p", _render)
        failures = [
            r for r in caplog.records
            if "structured-output invocation failed" in r.getMessage()
        ]
        assert len(failures) == 1

    def test_successful_structured_calls_stay_structured(self):
        """Success path keeps using structured output and never goes sticky."""
        llm, structured = _binding_llm(structured_return="OBJ")
        binding = bind_structured(llm, TraderProposal, "Trader")

        assert binding.invoke("p", _render) == "RENDERED::OBJ"
        assert binding.invoke("p", _render) == "RENDERED::OBJ"

        assert structured.invoke.call_count == 2
        assert llm.invoke.call_count == 0  # plain fallback untouched
        assert binding.uses_structured_output is True

    def test_unsupported_at_bind_uses_freetext_every_call(self, caplog):
        """Original branch: provider unsupported at bind time → free text on every call."""
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
        llm.invoke.return_value = MagicMock(content="PLAIN")

        with caplog.at_level(
            logging.WARNING, logger="tradingagents.agents.utils.structured"
        ):
            binding = bind_structured(llm, TraderProposal, "Trader")

        assert binding.uses_structured_output is False
        assert binding.invoke("p", _render) == "PLAIN"
        assert binding.invoke("p", _render) == "PLAIN"
        # Bound once at creation; not re-attempted on each invoke.
        assert llm.with_structured_output.call_count == 1
        assert llm.invoke.call_count == 2
        assert any(
            "does not support with_structured_output" in r.getMessage()
            for r in caplog.records
        )

    def test_distinct_bindings_do_not_share_sticky_state(self):
        """One binding degrading must not affect another (per agent + per schema)."""
        # Binding A: structured always fails -> goes sticky.
        llm_a, _ = _binding_llm(structured_side_effect=ValueError("400"))
        # Binding B: independent llm + a different schema; structured works.
        llm_b, structured_b = _binding_llm(structured_return="B")
        binding_a = bind_structured(llm_a, TraderProposal, "A")
        binding_b = bind_structured(llm_b, ResearchPlan, "B")

        binding_a.invoke("p", _render)  # trip A into sticky fallback
        assert binding_a.uses_structured_output is False

        # B is unaffected by A's degradation.
        assert binding_b.invoke("p", _render) == "RENDERED::B"
        assert binding_b.invoke("p", _render) == "RENDERED::B"
        assert binding_b.uses_structured_output is True
        assert structured_b.invoke.call_count == 2
        assert llm_b.invoke.call_count == 0


@pytest.mark.unit
class TestAgentStickyFallback:
    """The sticky fallback as wired through the real agent factories.

    Confirms a runtime structured failure on one call is not retried on the
    next, the markdown output contract holds on the free-text path, the
    success path is untouched, and one agent degrading does not bleed into a
    different agent with a different schema.
    """

    def test_trader_sticky_fallback_across_repeated_calls(self):
        plain_response = (
            "**Action**: Hold\n\nNo edge in the setup.\n\n"
            "FINAL TRANSACTION PROPOSAL: **HOLD**"
        )
        structured = MagicMock()
        structured.invoke.side_effect = ValueError("400 response_format unsupported")
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        llm.invoke.return_value = MagicMock(content=plain_response)

        trader = create_trader(llm)
        first = trader(_make_trader_state())
        second = trader(_make_trader_state())

        # Output contract preserved via free text on both calls.
        assert first["trader_investment_plan"] == plain_response
        assert second["trader_investment_plan"] == plain_response
        # Structured attempted once; the second call did not retry it.
        assert structured.invoke.call_count == 1
        assert llm.invoke.call_count == 2

    def test_trader_repeated_structured_success_never_falls_back(self):
        captured = {}
        llm = _structured_trader_llm(
            captured, TraderProposal(action=TraderAction.BUY, reasoning="Strong setup.")
        )
        structured = llm.with_structured_output.return_value
        trader = create_trader(llm)

        a = trader(_make_trader_state())["trader_investment_plan"]
        b = trader(_make_trader_state())["trader_investment_plan"]

        assert "FINAL TRANSACTION PROPOSAL: **BUY**" in a
        assert "FINAL TRANSACTION PROPOSAL: **BUY**" in b
        assert structured.invoke.call_count == 2
        assert llm.invoke.call_count == 0  # plain path untouched on success

    def test_sticky_fallback_isolated_across_agents_and_schemas(self):
        # Trader: structured always fails -> sticky free-text.
        trader_plain = (
            "**Action**: Sell\n\nGuidance cut hits margins.\n\n"
            "FINAL TRANSACTION PROPOSAL: **SELL**"
        )
        t_structured = MagicMock()
        t_structured.invoke.side_effect = ValueError("400")
        t_llm = MagicMock()
        t_llm.with_structured_output.return_value = t_structured
        t_llm.invoke.return_value = MagicMock(content=trader_plain)
        trader = create_trader(t_llm)

        # Research Manager: independent llm + a different schema; structured works.
        rm_captured = {}
        rm_llm = _structured_rm_llm(
            rm_captured,
            ResearchPlan(
                recommendation=PortfolioRating.OVERWEIGHT,
                rationale="Bull case is stronger.",
                strategic_actions="Build position gradually.",
            ),
        )
        rm_structured = rm_llm.with_structured_output.return_value
        rm = create_research_manager(rm_llm)

        # Trip the trader into its sticky fallback (twice).
        trader(_make_trader_state())
        trader(_make_trader_state())
        # The RM shares no state with the trader and still uses structured.
        rm_plan = rm(_make_rm_state())["investment_plan"]
        rm(_make_rm_state())

        assert t_structured.invoke.call_count == 1  # trader stuck on free text
        assert t_llm.invoke.call_count == 2
        assert "**Recommendation**: Overweight" in rm_plan  # RM rendered structured
        assert rm_structured.invoke.call_count == 2  # RM never went sticky
        assert rm_llm.invoke.call_count == 0

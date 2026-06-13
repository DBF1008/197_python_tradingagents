"""Tests for the sticky-fallback behavior of ``bind_structured`` / ``invoke_structured_or_freetext``.

Covers the four scenarios the fix must get right:

1. **Sticky fallback**: after the first structured-output invocation
   failure, all subsequent calls for the same agent skip the structured
   path entirely (no repeated 400s, no duplicate warnings).
2. **Agent/schema isolation**: a runtime failure in one agent's binding
   does not poison another agent's binding, even when both share the
   same underlying LLM.
3. **Success path unaffected**: structured calls that succeed continue
   to go through the render function on every invocation.
4. **Bind-time unsupported branch**: the original behavior where
   ``with_structured_output`` raises ``NotImplementedError`` /
   ``AttributeError`` at bind time is preserved — every call uses
   free-text generation directly.
"""

from unittest.mock import MagicMock

import pytest

from tradingagents.agents.utils.structured import (
    _StructuredBinding,
    bind_structured,
    invoke_structured_or_freetext,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm(structured_side_effect=None, plain_content="plain response",
              structured_return=None):
    """Build a mock LLM with a configurable structured-output binding."""
    structured = MagicMock()
    if structured_side_effect is not None:
        structured.invoke.side_effect = structured_side_effect
    elif structured_return is not None:
        structured.invoke.return_value = structured_return
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    llm.invoke.return_value = MagicMock(content=plain_content)
    return llm, structured


def _identity_render(obj):
    """Minimal render: return ``str(obj)`` — sufficient for mock return values."""
    return str(obj)


# ---------------------------------------------------------------------------
# 1. Sticky fallback — structured failure is remembered across calls
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStickyFallback:
    def test_first_failure_then_subsequent_calls_skip_structured(self):
        """After a structured invocation failure the binding is poisoned.

        The second call must not call ``structured_llm.invoke`` at all —
        it goes straight to the free-text path via ``plain_llm.invoke``.
        """
        llm, structured = _make_llm(
            structured_side_effect=ValueError("provider 400: bad tool_choice"),
            plain_content="first-fallback",
        )
        binding = bind_structured(llm, object, "TestAgent")

        # First call: structured fails → falls back to plain
        r1 = invoke_structured_or_freetext(
            binding, llm, "prompt", _identity_render, "TestAgent"
        )
        assert r1 == "first-fallback"
        assert structured.invoke.call_count == 1
        assert llm.invoke.call_count == 1

        # Reconfigure plain response for the second call to prove
        # the free-text path is actually invoked (not a cached result).
        llm.invoke.return_value = MagicMock(content="second-fallback")

        # Second call: structured path is skipped entirely
        r2 = invoke_structured_or_freetext(
            binding, llm, "prompt2", _identity_render, "TestAgent"
        )
        assert r2 == "second-fallback"
        # structured.invoke NOT called again — still 1
        assert structured.invoke.call_count == 1
        # plain_llm.invoke called again — now 2
        assert llm.invoke.call_count == 2

    def test_three_consecutive_calls_all_skip_structured(self):
        """The sticky fallback holds for any number of subsequent calls."""
        llm, structured = _make_llm(
            structured_side_effect=RuntimeError("persistent failure"),
            plain_content="fallback",
        )
        binding = bind_structured(llm, object, "TestAgent")

        for i in range(3):
            result = invoke_structured_or_freetext(
                binding, llm, f"prompt-{i}", _identity_render, "TestAgent"
            )
            assert result == "fallback"

        # structured called exactly once (the first call), plain called 3 times
        # (first call's fallback + 2 subsequent calls that skip structured)
        assert structured.invoke.call_count == 1
        assert llm.invoke.call_count == 3

    def test_binding_poisoned_flag_is_set_after_failure(self):
        """Internal: the binding's ``_poisoned`` flag flips after a failure."""
        llm, structured = _make_llm(
            structured_side_effect=ValueError("bad JSON"),
        )
        binding = bind_structured(llm, object, "TestAgent")
        assert binding._poisoned is False

        invoke_structured_or_freetext(
            binding, llm, "prompt", _identity_render, "TestAgent"
        )
        assert binding._poisoned is True

    def test_first_call_still_falls_back_to_freetext(self):
        """The first structured failure still produces a usable free-text result."""
        llm, structured = _make_llm(
            structured_side_effect=ValueError("malformed JSON"),
            plain_content="usable free-text output",
        )
        binding = bind_structured(llm, object, "TestAgent")

        result = invoke_structured_or_freetext(
            binding, llm, "prompt", _identity_render, "TestAgent"
        )
        assert result == "usable free-text output"


# ---------------------------------------------------------------------------
# 2. Agent/schema isolation — one binding's failure doesn't affect another
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAgentIsolation:
    def test_separate_bindings_are_independent(self):
        """Two bindings from the same LLM don't share poisoned state."""
        llm_a, structured_a = _make_llm(
            structured_side_effect=ValueError("agent-A failure"),
            plain_content="A-fallback",
        )
        llm_b, structured_b = _make_llm(structured_return="B-structured")

        binding_a = bind_structured(llm_a, object, "AgentA")
        binding_b = bind_structured(llm_b, object, "AgentB")

        # Agent A: structured fails → poisons its binding
        invoke_structured_or_freetext(
            binding_a, llm_a, "prompt", _identity_render, "AgentA"
        )
        assert binding_a._poisoned is True
        assert binding_b._poisoned is False

        # Agent B: structured still works normally
        result_b = invoke_structured_or_freetext(
            binding_b, llm_b, "prompt", _identity_render, "AgentB"
        )
        assert result_b == "B-structured"
        assert structured_b.invoke.call_count == 1

    def test_trader_and_rm_share_llm_but_isolate_failures(self):
        """Trader and Research Manager share the same underlying LLM.

        If Trader's structured call fails, it must not affect RM's binding.
        This mirrors the real graph where all agents are created from the
        same ``llm`` instance.
        """
        from tradingagents.agents.schemas import (
            ResearchPlan,
            TraderProposal,
            render_research_plan,
            render_trader_proposal,
        )

        shared_llm = MagicMock()

        # First with_structured_output call (Trader) → broken binding
        broken_structured = MagicMock()
        broken_structured.invoke.side_effect = ValueError("Trader schema incompatible")

        # Second with_structured_output call (RM) → working binding
        working_structured = MagicMock()
        working_structured.invoke.return_value = ResearchPlan(
            recommendation="Buy",
            rationale="test",
            strategic_actions="test",
        )

        shared_llm.with_structured_output.side_effect = [
            broken_structured,
            working_structured,
        ]
        shared_llm.invoke.return_value = MagicMock(content="trader-fallback")

        trader_binding = bind_structured(shared_llm, TraderProposal, "Trader")
        rm_binding = bind_structured(shared_llm, ResearchPlan, "Research Manager")

        # Trader: structured fails → fallback
        trader_result = invoke_structured_or_freetext(
            trader_binding, shared_llm, "p",
            render_trader_proposal, "Trader",
        )
        assert trader_result == "trader-fallback"
        assert trader_binding._poisoned is True
        assert rm_binding._poisoned is False

        # RM: structured still works — produces rendered markdown
        rm_result = invoke_structured_or_freetext(
            rm_binding, shared_llm, "p",
            render_research_plan, "Research Manager",
        )
        assert "**Recommendation**" in rm_result
        assert working_structured.invoke.call_count == 1

    def test_three_agents_isolated(self):
        """Three agents with separate bindings — only one fails."""
        llm_a, s_a = _make_llm(structured_side_effect=RuntimeError("A breaks"))
        llm_b, s_b = _make_llm(plain_content="B-ok")
        llm_c, s_c = _make_llm(plain_content="C-ok")

        b_a = bind_structured(llm_a, object, "A")
        b_b = bind_structured(llm_b, object, "B")
        b_c = bind_structured(llm_c, object, "C")

        # A fails
        invoke_structured_or_freetext(
            b_a, llm_a, "p", _identity_render, "A"
        )
        assert b_a._poisoned is True
        assert b_b._poisoned is False
        assert b_c._poisoned is False

        # B and C still go through structured path
        invoke_structured_or_freetext(
            b_b, llm_b, "p", _identity_render, "B"
        )
        invoke_structured_or_freetext(
            b_c, llm_c, "p", _identity_render, "C"
        )
        assert s_b.invoke.call_count == 1
        assert s_c.invoke.call_count == 1


# ---------------------------------------------------------------------------
# 3. Success path unaffected — structured calls that work keep working
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSuccessPathUnaffected:
    def test_successful_structured_calls_continue_to_use_render(self):
        """When the structured call succeeds, it is used on every invocation."""
        llm, structured = _make_llm()
        structured.invoke.side_effect = lambda p: f"structured({p})"
        binding = bind_structured(llm, object, "TestAgent")

        r1 = invoke_structured_or_freetext(
            binding, llm, "p1", _identity_render, "TestAgent"
        )
        r2 = invoke_structured_or_freetext(
            binding, llm, "p2", _identity_render, "TestAgent"
        )
        r3 = invoke_structured_or_freetext(
            binding, llm, "p3", _identity_render, "TestAgent"
        )

        assert r1 == "structured(p1)"
        assert r2 == "structured(p2)"
        assert r3 == "structured(p3)"
        # structured called 3 times, plain never called
        assert structured.invoke.call_count == 3
        assert llm.invoke.call_count == 0

    def test_binding_not_poisoned_after_success(self):
        """Successful invocations never set the poisoned flag."""
        llm, structured = _make_llm()
        structured.invoke.return_value = "ok"
        binding = bind_structured(llm, object, "TestAgent")

        invoke_structured_or_freetext(
            binding, llm, "p", _identity_render, "TestAgent"
        )
        assert binding._poisoned is False

    def test_render_function_is_called_on_structured_result(self):
        """The render function transforms the structured result into markdown."""
        from tradingagents.agents.schemas import (
            TraderAction,
            TraderProposal,
            render_trader_proposal,
        )

        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="Strong setup.",
            entry_price=100.0,
        )
        llm, structured = _make_llm()
        structured.invoke.return_value = proposal
        binding = bind_structured(llm, object, "Trader")

        result = invoke_structured_or_freetext(
            binding, llm, "p", render_trader_proposal, "Trader"
        )
        assert "**Action**: Buy" in result
        assert "**Entry Price**: 100.0" in result
        assert "FINAL TRANSACTION PROPOSAL: **BUY**" in result

    def test_interleaved_success_and_different_agents(self):
        """Two agents: one always succeeds, they don't interfere."""
        llm_a, s_a = _make_llm()
        s_a.invoke.side_effect = lambda p: f"A({p})"
        llm_b, s_b = _make_llm()
        s_b.invoke.side_effect = lambda p: f"B({p})"

        b_a = bind_structured(llm_a, object, "A")
        b_b = bind_structured(llm_b, object, "B")

        results = []
        for i in range(3):
            results.append(
                invoke_structured_or_freetext(
                    b_a, llm_a, f"x{i}", _identity_render, "A"
                )
            )
            results.append(
                invoke_structured_or_freetext(
                    b_b, llm_b, f"y{i}", _identity_render, "B"
                )
            )

        assert results == ["A(x0)", "B(y0)", "A(x1)", "B(y1)", "A(x2)", "B(y2)"]


# ---------------------------------------------------------------------------
# 4. Bind-time unsupported branch — with_structured_output raises at bind
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBindTimeUnsupported:
    def test_not_implemented_error_at_bind_uses_freetext_every_time(self):
        """``NotImplementedError`` from ``with_structured_output`` → all calls
        go through free-text. ``plain_llm.invoke`` is called on every call."""
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
        llm.invoke.return_value = MagicMock(content="free-text")

        binding = bind_structured(llm, object, "TestAgent")
        assert binding.structured_llm is None

        for i in range(3):
            result = invoke_structured_or_freetext(
                binding, llm, f"p{i}", _identity_render, "TestAgent"
            )
            assert result == "free-text"

        assert llm.invoke.call_count == 3

    def test_attribute_error_at_bind_uses_freetext(self):
        """``AttributeError`` from ``with_structured_output`` is also handled."""
        llm = MagicMock()
        llm.with_structured_output.side_effect = AttributeError("no such method")
        llm.invoke.return_value = MagicMock(content="attr-error-fallback")

        binding = bind_structured(llm, object, "TestAgent")
        assert binding.structured_llm is None

        result = invoke_structured_or_freetext(
            binding, llm, "p", _identity_render, "TestAgent"
        )
        assert result == "attr-error-fallback"

    def test_bind_time_structured_llm_is_none_in_binding(self):
        """The binding records ``None`` for ``structured_llm`` on bind failure."""
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("nope")

        binding = bind_structured(llm, object, "TestAgent")
        assert isinstance(binding, _StructuredBinding)
        assert binding.structured_llm is None
        assert binding.plain_llm is llm
        assert binding.agent_name == "TestAgent"
        assert binding._poisoned is False  # not poisoned — never had a chance


# ---------------------------------------------------------------------------
# 5. Legacy backward-compat path — raw structured LLM without binding
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLegacyBackwardCompat:
    def test_raw_structured_llm_success(self):
        """Passing a raw structured LLM (not a binding) still works."""
        structured = MagicMock()
        structured.invoke.return_value = "raw-ok"
        plain_llm = MagicMock()

        result = invoke_structured_or_freetext(
            structured, plain_llm, "p", _identity_render, "Legacy"
        )
        assert result == "raw-ok"
        assert plain_llm.invoke.call_count == 0

    def test_raw_structured_llm_failure_falls_back_without_sticky(self):
        """Raw structured LLM failure falls back but does NOT track state.

        Since there is no binding to poison, each call retries the
        structured path. This preserves the original single-shot fallback
        behavior for any caller that bypasses ``bind_structured``.
        """
        structured = MagicMock()
        structured.invoke.side_effect = ValueError("bad JSON")
        plain_llm = MagicMock()
        plain_llm.invoke.return_value = MagicMock(content="plain")

        # Call 1: structured fails → fallback
        r1 = invoke_structured_or_freetext(
            structured, plain_llm, "p1", _identity_render, "Legacy"
        )
        assert r1 == "plain"
        assert structured.invoke.call_count == 1
        assert plain_llm.invoke.call_count == 1

        # Call 2: structured is tried again (no sticky state) → fails → fallback
        r2 = invoke_structured_or_freetext(
            structured, plain_llm, "p2", _identity_render, "Legacy"
        )
        assert r2 == "plain"
        assert structured.invoke.call_count == 2  # called again
        assert plain_llm.invoke.call_count == 2

    def test_none_structured_llm_uses_plain(self):
        """Passing ``None`` as structured_llm goes straight to plain_llm."""
        plain_llm = MagicMock()
        plain_llm.invoke.return_value = MagicMock(content="direct-plain")

        result = invoke_structured_or_freetext(
            None, plain_llm, "p", _identity_render, "TestAgent"
        )
        assert result == "direct-plain"


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEdgeCases:
    def test_different_exception_types_all_poison(self):
        """The fallback catches any Exception subclass, not just ValueError."""
        for exc_class in (ValueError, RuntimeError, TypeError, KeyError):
            llm, structured = _make_llm(
                structured_side_effect=exc_class("fail"),
                plain_content=f"fallback-{exc_class.__name__}",
            )
            binding = bind_structured(llm, object, f"Agent-{exc_class.__name__}")

            # First call: fails and poisons
            invoke_structured_or_freetext(
                binding, llm, "p", _identity_render, f"Agent-{exc_class.__name__}"
            )
            assert binding._poisoned is True

            # Second call: skips structured
            llm.invoke.return_value = MagicMock(content="second-call")
            result = invoke_structured_or_freetext(
                binding, llm, "p", _identity_render, f"Agent-{exc_class.__name__}"
            )
            assert result == "second-call"
            assert structured.invoke.call_count == 1

    def test_binding_preserves_agent_name(self):
        """The binding stores the agent_name for log messages."""
        llm, _ = _make_llm()
        binding = bind_structured(llm, object, "Sentiment Analyst")
        assert binding.agent_name == "Sentiment Analyst"

    def test_binding_stores_plain_llm_reference(self):
        """The binding keeps a reference to the original plain LLM."""
        llm, _ = _make_llm()
        binding = bind_structured(llm, object, "TestAgent")
        assert binding.plain_llm is llm

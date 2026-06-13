"""Tests for resume_policy: compatibility evaluation and verdict formatting."""

import unittest
from typing import Any, Dict, List

from tradingagents.graph.checkpoint_manifest import RunManifest
from tradingagents.graph.resume_policy import (
    ResumeDecision,
    ResumeRejectedError,
    ResumeVerdict,
    evaluate_resume,
    format_resume_summary,
)


def _manifest(**overrides: Any) -> RunManifest:
    """Build a RunManifest with sensible defaults."""
    defaults = dict(
        thread_id="abc123",
        ticker="TEST",
        trade_date="2026-04-20",
        asset_type="stock",
        selected_analysts=("market", "social", "news", "fundamentals"),
        llm_provider="openai",
        deep_think_llm="gpt-5.5",
        quick_think_llm="gpt-5.4-mini",
        temperature=0.2,
        backend_url=None,
        last_completed_step=3,
        created_at="2026-04-20T10:00:00+00:00",
        updated_at="2026-04-20T10:00:00+00:00",
    )
    defaults.update(overrides)
    return RunManifest(**defaults)


def _config(**overrides: Any) -> Dict[str, Any]:
    """Build a DEFAULT_CONFIG-like dict."""
    defaults: Dict[str, Any] = {
        "llm_provider": "openai",
        "deep_think_llm": "gpt-5.5",
        "quick_think_llm": "gpt-5.4-mini",
        "temperature": 0.2,
        "backend_url": None,
    }
    defaults.update(overrides)
    return defaults


_DEFAULT_ANALYSTS = ["market", "social", "news", "fundamentals"]


class TestEvaluateResume(unittest.TestCase):
    """Core policy logic."""

    def test_fresh_start_no_checkpoint(self):
        """No manifest → FRESH_START."""
        v = evaluate_resume(
            existing=None,
            current_analysts=_DEFAULT_ANALYSTS,
            current_asset_type="stock",
            current_config=_config(),
        )
        self.assertEqual(v.decision, ResumeDecision.FRESH_START)
        self.assertIsNone(v.manifest)

    def test_matching_config_resumes(self):
        """All fields match → RESUME."""
        m = _manifest()
        v = evaluate_resume(
            existing=m,
            current_analysts=_DEFAULT_ANALYSTS,
            current_asset_type="stock",
            current_config=_config(),
        )
        self.assertEqual(v.decision, ResumeDecision.RESUME)
        self.assertEqual(v.manifest, m)
        self.assertEqual(v.incompatible_fields, ())
        self.assertEqual(v.drifted_fields, ())

    def test_different_analysts_rejects(self):
        """Analyst set change → REJECT."""
        m = _manifest(selected_analysts=("market", "social", "news", "fundamentals"))
        v = evaluate_resume(
            existing=m,
            current_analysts=["market", "social"],  # fewer analysts
            current_asset_type="stock",
            current_config=_config(),
        )
        self.assertEqual(v.decision, ResumeDecision.REJECT)
        self.assertIn("selected_analysts", v.incompatible_fields)

    def test_different_analyst_order_rejects(self):
        """Same analysts but different order → REJECT (order matters for topology)."""
        m = _manifest(selected_analysts=("market", "social", "news", "fundamentals"))
        v = evaluate_resume(
            existing=m,
            current_analysts=["fundamentals", "news", "social", "market"],
            current_asset_type="stock",
            current_config=_config(),
        )
        self.assertEqual(v.decision, ResumeDecision.REJECT)
        self.assertIn("selected_analysts", v.incompatible_fields)

    def test_different_asset_type_rejects(self):
        """stock → crypto → REJECT."""
        m = _manifest(asset_type="stock")
        v = evaluate_resume(
            existing=m,
            current_analysts=_DEFAULT_ANALYSTS,
            current_asset_type="crypto",
            current_config=_config(),
        )
        self.assertEqual(v.decision, ResumeDecision.REJECT)
        self.assertIn("asset_type", v.incompatible_fields)

    def test_llm_drift_warns(self):
        """Model change → WARN_RESUME."""
        m = _manifest(quick_think_llm="gpt-5.4-mini")
        v = evaluate_resume(
            existing=m,
            current_analysts=_DEFAULT_ANALYSTS,
            current_asset_type="stock",
            current_config=_config(quick_think_llm="gpt-5.5-nano"),
        )
        self.assertEqual(v.decision, ResumeDecision.WARN_RESUME)
        self.assertIn("quick_think_llm", v.drifted_fields)
        self.assertEqual(v.incompatible_fields, ())

    def test_provider_drift_warns(self):
        """Provider change → WARN_RESUME."""
        m = _manifest(llm_provider="openai")
        v = evaluate_resume(
            existing=m,
            current_analysts=_DEFAULT_ANALYSTS,
            current_asset_type="stock",
            current_config=_config(llm_provider="anthropic"),
        )
        self.assertEqual(v.decision, ResumeDecision.WARN_RESUME)
        self.assertIn("llm_provider", v.drifted_fields)

    def test_temperature_drift_warns(self):
        """Temperature change → WARN_RESUME."""
        m = _manifest(temperature=0.2)
        v = evaluate_resume(
            existing=m,
            current_analysts=_DEFAULT_ANALYSTS,
            current_asset_type="stock",
            current_config=_config(temperature=0.7),
        )
        self.assertEqual(v.decision, ResumeDecision.WARN_RESUME)
        self.assertIn("temperature", v.drifted_fields)

    def test_multiple_drifts_accumulate(self):
        """Multiple soft drifts are all reported."""
        m = _manifest()
        v = evaluate_resume(
            existing=m,
            current_analysts=_DEFAULT_ANALYSTS,
            current_asset_type="stock",
            current_config=_config(
                llm_provider="anthropic",
                deep_think_llm="claude-4-sonnet",
                temperature=0.9,
            ),
        )
        self.assertEqual(v.decision, ResumeDecision.WARN_RESUME)
        self.assertIn("llm_provider", v.drifted_fields)
        self.assertIn("deep_think_llm", v.drifted_fields)
        self.assertIn("temperature", v.drifted_fields)
        self.assertEqual(len(v.drifted_fields), 3)

    def test_force_overrides_reject(self):
        """force=True downgrades REJECT to WARN_RESUME."""
        m = _manifest(selected_analysts=("market", "social"))
        v = evaluate_resume(
            existing=m,
            current_analysts=["market", "social", "news"],
            current_asset_type="stock",
            current_config=_config(),
            force=True,
        )
        self.assertEqual(v.decision, ResumeDecision.WARN_RESUME)
        self.assertIn("selected_analysts", v.incompatible_fields)

    def test_force_does_not_affect_clean_resume(self):
        """force=True on a clean match is a no-op."""
        m = _manifest()
        v = evaluate_resume(
            existing=m,
            current_analysts=_DEFAULT_ANALYSTS,
            current_asset_type="stock",
            current_config=_config(),
            force=True,
        )
        self.assertEqual(v.decision, ResumeDecision.RESUME)

    def test_hard_and_soft_combined(self):
        """Both hard and soft incompatibilities: hard wins unless forced."""
        m = _manifest(
            selected_analysts=("market", "social"),
            llm_provider="openai",
        )
        v = evaluate_resume(
            existing=m,
            current_analysts=["market"],
            current_asset_type="stock",
            current_config=_config(llm_provider="anthropic"),
        )
        self.assertEqual(v.decision, ResumeDecision.REJECT)
        self.assertIn("selected_analysts", v.incompatible_fields)
        self.assertIn("llm_provider", v.drifted_fields)

    def test_hard_and_soft_forced(self):
        """Forced resume reports both incompatible and drifted fields."""
        m = _manifest(
            selected_analysts=("market", "social"),
            llm_provider="openai",
        )
        v = evaluate_resume(
            existing=m,
            current_analysts=["market"],
            current_asset_type="stock",
            current_config=_config(llm_provider="anthropic"),
            force=True,
        )
        self.assertEqual(v.decision, ResumeDecision.WARN_RESUME)
        self.assertIn("selected_analysts", v.incompatible_fields)
        self.assertIn("llm_provider", v.drifted_fields)


class TestResumeRejectedError(unittest.TestCase):
    def test_error_carries_verdict(self):
        m = _manifest(selected_analysts=("market",))
        v = evaluate_resume(
            existing=m,
            current_analysts=["market", "social"],
            current_asset_type="stock",
            current_config=_config(),
        )
        err = ResumeRejectedError(v)
        self.assertIs(err.verdict, v)
        self.assertIn("selected_analysts", str(err))


class TestFormatResumeSummary(unittest.TestCase):
    def test_resume_summary_contains_key_fields(self):
        m = _manifest()
        v = ResumeVerdict(
            decision=ResumeDecision.RESUME,
            reason="Configuration matches.",
            manifest=m,
        )
        text = format_resume_summary(v)
        self.assertIn("RESUME", text)
        self.assertIn("TEST", text)
        self.assertIn("2026-04-20", text)
        self.assertIn("openai", text)
        self.assertIn("gpt-5.5", text)

    def test_reject_summary_shows_incompatible(self):
        m = _manifest()
        v = ResumeVerdict(
            decision=ResumeDecision.REJECT,
            reason="Incompatible analysts.",
            manifest=m,
            incompatible_fields=("selected_analysts",),
        )
        text = format_resume_summary(v)
        self.assertIn("REJECT", text)
        self.assertIn("selected_analysts", text)

    def test_fresh_start_summary(self):
        v = ResumeVerdict(
            decision=ResumeDecision.FRESH_START,
            reason="No checkpoint found.",
        )
        text = format_resume_summary(v)
        self.assertIn("FRESH_START", text)


if __name__ == "__main__":
    unittest.main()

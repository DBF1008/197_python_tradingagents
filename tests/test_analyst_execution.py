import unittest

from tradingagents.graph.analyst_execution import (
    AnalystWallTimeTracker,
    build_analyst_execution_plan,
    get_initial_analyst_node,
    sync_analyst_tracker_from_chunk,
)


class AnalystExecutionPlanTests(unittest.TestCase):
    def test_build_plan_preserves_selected_order(self):
        plan = build_analyst_execution_plan(["news", "market"], concurrency_limit=2)

        self.assertEqual([spec.key for spec in plan.specs], ["news", "market"])
        self.assertEqual(plan.concurrency_limit, 2)
        self.assertEqual(plan.specs[0].agent_node, "News Analyst")
        self.assertEqual(plan.specs[0].tool_node, "tools_news")
        self.assertEqual(plan.specs[0].clear_node, "Msg Clear News")

    def test_rejects_unknown_analyst_keys(self):
        with self.assertRaises(ValueError):
            build_analyst_execution_plan(["market", "macro"])

    def test_requires_positive_concurrency_limit(self):
        with self.assertRaises(ValueError):
            build_analyst_execution_plan(["market"], concurrency_limit=0)

    def test_get_initial_analyst_node_uses_plan_metadata(self):
        plan = build_analyst_execution_plan(["fundamentals", "news"])

        self.assertEqual(
            get_initial_analyst_node(plan),
            "Fundamentals Analyst",
        )

    def test_social_key_displays_as_sentiment_analyst(self):
        # The wire key stays "social" for saved-config back-compat, but the
        # user-visible agent_node label must match the v0.2.5 rename so the
        # wall-time summary and any future consumer of agent_node says
        # "Sentiment Analyst" rather than the legacy "Social Analyst".
        plan = build_analyst_execution_plan(["social"])
        spec = plan.specs[0]
        self.assertEqual(spec.key, "social")
        self.assertEqual(spec.agent_node, "Sentiment Analyst")
        self.assertEqual(spec.report_key, "sentiment_report")


class AnalystWallTimeTrackerTests(unittest.TestCase):
    def test_records_wall_time_when_analyst_completes(self):
        plan = build_analyst_execution_plan(["market", "news"])
        tracker = AnalystWallTimeTracker(plan)

        tracker.mark_started("market", started_at=10.0)
        tracker.mark_completed("market", completed_at=13.5)

        self.assertEqual(tracker.get_wall_times(), {"market": 3.5})

    def test_formats_summary_in_plan_order(self):
        plan = build_analyst_execution_plan(["news", "market"])
        tracker = AnalystWallTimeTracker(plan)

        tracker.mark_started("market", started_at=20.0)
        tracker.mark_completed("market", completed_at=22.25)
        tracker.mark_started("news", started_at=10.0)
        tracker.mark_completed("news", completed_at=14.0)

        self.assertEqual(
            tracker.format_summary(),
            "Analyst wall time: News 4.00s | Market 2.25s",
        )

    def test_syncs_wall_time_from_sequential_chunks(self):
        plan = build_analyst_execution_plan(["market", "news"])
        tracker = AnalystWallTimeTracker(plan)

        sync_analyst_tracker_from_chunk(tracker, {}, now=10.0)
        self.assertEqual(tracker.get_wall_times(), {})

        sync_analyst_tracker_from_chunk(
            tracker,
            {"market_report": "done"},
            now=13.0,
        )
        self.assertEqual(tracker.get_wall_times(), {"market": 3.0})

        sync_analyst_tracker_from_chunk(
            tracker,
            {"market_report": "done", "news_report": "done"},
            now=18.0,
        )
        self.assertEqual(
            tracker.get_wall_times(),
            {"market": 3.0, "news": 5.0},
        )


class ConcurrentAnalystTrackingTests(unittest.TestCase):
    """Tests for sync_analyst_tracker_from_chunk with concurrency_limit > 1."""

    def test_concurrent_start_marks_both_analysts(self):
        """concurrency_limit=2, empty chunk → both analysts marked started."""
        plan = build_analyst_execution_plan(["market", "news"], concurrency_limit=2)
        tracker = AnalystWallTimeTracker(plan)

        sync_analyst_tracker_from_chunk(tracker, {}, now=10.0)

        self.assertTrue(tracker.is_started("market"))
        self.assertTrue(tracker.is_started("news"))
        self.assertFalse(tracker.is_completed("market"))
        self.assertFalse(tracker.is_completed("news"))
        self.assertEqual(tracker.get_wall_times(), {})

    def test_interleaved_completion(self):
        """Two analysts start together, complete at different times."""
        plan = build_analyst_execution_plan(["market", "news"], concurrency_limit=2)
        tracker = AnalystWallTimeTracker(plan)

        # t=10: both start (empty chunk, capacity=2)
        sync_analyst_tracker_from_chunk(tracker, {}, now=10.0)

        # t=15: news completes first
        sync_analyst_tracker_from_chunk(
            tracker, {"news_report": "done"}, now=15.0,
        )
        self.assertTrue(tracker.is_completed("news"))
        self.assertFalse(tracker.is_completed("market"))
        self.assertEqual(tracker.get_wall_times(), {"news": 5.0})

        # t=20: market completes
        sync_analyst_tracker_from_chunk(
            tracker,
            {"market_report": "done", "news_report": "done"},
            now=20.0,
        )
        self.assertTrue(tracker.is_completed("market"))
        self.assertEqual(
            tracker.get_wall_times(),
            {"market": 10.0, "news": 5.0},
        )

    def test_repeated_chunk_preserves_earliest_start(self):
        """Multiple empty chunks: setdefault preserves earliest start time."""
        plan = build_analyst_execution_plan(["market", "news"], concurrency_limit=2)
        tracker = AnalystWallTimeTracker(plan)

        sync_analyst_tracker_from_chunk(tracker, {}, now=10.0)
        sync_analyst_tracker_from_chunk(tracker, {}, now=15.0)
        sync_analyst_tracker_from_chunk(tracker, {}, now=20.0)

        # Both should still show start time from first chunk (t=10)
        sync_analyst_tracker_from_chunk(
            tracker, {"market_report": "done"}, now=25.0,
        )
        self.assertEqual(tracker.get_wall_times(), {"market": 15.0})

    def test_serial_backward_compat(self):
        """concurrency_limit=1 behaves identically to original serial logic."""
        plan = build_analyst_execution_plan(["market", "news"], concurrency_limit=1)
        tracker = AnalystWallTimeTracker(plan)

        # t=10: only market starts (capacity=1)
        sync_analyst_tracker_from_chunk(tracker, {}, now=10.0)
        self.assertTrue(tracker.is_started("market"))
        self.assertFalse(tracker.is_started("news"))

        # t=13: market completes
        sync_analyst_tracker_from_chunk(
            tracker, {"market_report": "done"}, now=13.0,
        )
        self.assertEqual(tracker.get_wall_times(), {"market": 3.0})

        # After market completes, news gets started using freed capacity
        self.assertTrue(tracker.is_started("news"))

        # t=18: news completes
        sync_analyst_tracker_from_chunk(
            tracker,
            {"market_report": "done", "news_report": "done"},
            now=18.0,
        )
        self.assertEqual(
            tracker.get_wall_times(),
            {"market": 3.0, "news": 5.0},
        )

    def test_simultaneous_completion(self):
        """Both reports in same chunk, concurrency_limit=2."""
        plan = build_analyst_execution_plan(["market", "news"], concurrency_limit=2)
        tracker = AnalystWallTimeTracker(plan)

        # t=10: both start
        sync_analyst_tracker_from_chunk(tracker, {}, now=10.0)

        # t=13: both complete in same chunk
        sync_analyst_tracker_from_chunk(
            tracker,
            {"market_report": "done", "news_report": "done"},
            now=13.0,
        )
        self.assertEqual(
            tracker.get_wall_times(),
            {"market": 3.0, "news": 3.0},
        )

    def test_capacity_reuse_after_completion(self):
        """One finishes, freed slot used by next analyst in plan order."""
        plan = build_analyst_execution_plan(
            ["market", "news", "fundamentals"], concurrency_limit=2,
        )
        tracker = AnalystWallTimeTracker(plan)

        # t=10: market and news start (capacity=2), fundamentals waits
        sync_analyst_tracker_from_chunk(tracker, {}, now=10.0)
        self.assertTrue(tracker.is_started("market"))
        self.assertTrue(tracker.is_started("news"))
        self.assertFalse(tracker.is_started("fundamentals"))

        # t=15: market completes → capacity freed → fundamentals starts
        sync_analyst_tracker_from_chunk(
            tracker, {"market_report": "done"}, now=15.0,
        )
        self.assertTrue(tracker.is_completed("market"))
        self.assertFalse(tracker.is_completed("news"))
        self.assertTrue(tracker.is_started("fundamentals"))
        self.assertEqual(tracker.get_wall_times(), {"market": 5.0})

        # t=20: news completes
        sync_analyst_tracker_from_chunk(
            tracker,
            {"market_report": "done", "news_report": "done"},
            now=20.0,
        )
        self.assertEqual(
            tracker.get_wall_times(),
            {"market": 5.0, "news": 10.0},
        )

        # t=25: fundamentals completes
        sync_analyst_tracker_from_chunk(
            tracker,
            {
                "market_report": "done",
                "news_report": "done",
                "fundamentals_report": "done",
            },
            now=25.0,
        )
        self.assertEqual(
            tracker.get_wall_times(),
            {"market": 5.0, "news": 10.0, "fundamentals": 10.0},
        )

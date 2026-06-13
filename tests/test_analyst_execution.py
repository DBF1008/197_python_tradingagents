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


class AnalystTrackerConcurrencyTests(unittest.TestCase):
    def test_concurrent_startup_marks_whole_window_active(self):
        # With concurrency_limit=2, both analysts launch together. The first
        # chunk carries no report yet (analysts are still calling tools), so
        # both must be marked active rather than only the first.
        plan = build_analyst_execution_plan(
            ["market", "news"], concurrency_limit=2
        )
        tracker = AnalystWallTimeTracker(plan)

        sync_analyst_tracker_from_chunk(tracker, {}, now=10.0)

        self.assertEqual(set(tracker.active_keys()), {"market", "news"})
        self.assertEqual(tracker.get_wall_times(), {})

    def test_interleaved_completion_records_each_real_wall_time(self):
        # Two analysts start at t=10. News finishes first at t=14, market
        # finishes later at t=17. Market's wall time must span from its own
        # start (10), not collapse to the completion chunk time, and news
        # finishing first must not disturb market's clock.
        plan = build_analyst_execution_plan(
            ["market", "news"], concurrency_limit=2
        )
        tracker = AnalystWallTimeTracker(plan)

        sync_analyst_tracker_from_chunk(tracker, {}, now=10.0)
        sync_analyst_tracker_from_chunk(
            tracker, {"news_report": "done"}, now=14.0
        )
        sync_analyst_tracker_from_chunk(
            tracker, {"market_report": "done"}, now=17.0
        )

        self.assertEqual(
            tracker.get_wall_times(),
            {"market": 7.0, "news": 4.0},
        )

    def test_completion_opens_next_slot_in_sliding_window(self):
        # concurrency_limit=2 over three analysts: market+news run first;
        # when market completes, fundamentals must enter the window and its
        # clock must start at the slot-open time (15), not at its report.
        plan = build_analyst_execution_plan(
            ["market", "news", "fundamentals"], concurrency_limit=2
        )
        tracker = AnalystWallTimeTracker(plan)

        sync_analyst_tracker_from_chunk(tracker, {}, now=10.0)
        self.assertEqual(set(tracker.active_keys()), {"market", "news"})

        sync_analyst_tracker_from_chunk(
            tracker, {"market_report": "done"}, now=15.0
        )
        self.assertEqual(tracker.get_wall_times(), {"market": 5.0})
        self.assertEqual(set(tracker.active_keys()), {"news", "fundamentals"})

        sync_analyst_tracker_from_chunk(
            tracker, {"news_report": "done"}, now=20.0
        )
        sync_analyst_tracker_from_chunk(
            tracker, {"fundamentals_report": "done"}, now=22.0
        )

        self.assertEqual(
            tracker.get_wall_times(),
            {"market": 5.0, "news": 10.0, "fundamentals": 7.0},
        )

    def test_long_running_analyst_accumulates_until_report_lands(self):
        # An analyst with no report across several chunks keeps its original
        # start; its wall time is measured only when the report finally lands.
        plan = build_analyst_execution_plan(
            ["market", "news"], concurrency_limit=2
        )
        tracker = AnalystWallTimeTracker(plan)

        sync_analyst_tracker_from_chunk(tracker, {}, now=10.0)
        sync_analyst_tracker_from_chunk(
            tracker, {"news_report": "done"}, now=14.0
        )
        # market still running across an empty chunk
        sync_analyst_tracker_from_chunk(tracker, {}, now=20.0)
        self.assertEqual(tracker.active_keys(), ["market"])
        sync_analyst_tracker_from_chunk(
            tracker, {"market_report": "done"}, now=25.0
        )

        self.assertEqual(
            tracker.get_wall_times(),
            {"market": 15.0, "news": 4.0},
        )

    def test_repeated_report_chunk_is_idempotent(self):
        # Streaming may re-deliver an already-finished report (or batch it with
        # later ones). A repeated report must not reset or extend wall time.
        plan = build_analyst_execution_plan(
            ["market", "news"], concurrency_limit=2
        )
        tracker = AnalystWallTimeTracker(plan)

        sync_analyst_tracker_from_chunk(tracker, {}, now=10.0)
        sync_analyst_tracker_from_chunk(
            tracker, {"market_report": "done"}, now=13.0
        )
        # market_report re-appears in later chunks at unrelated times
        sync_analyst_tracker_from_chunk(
            tracker, {"market_report": "done"}, now=99.0
        )
        sync_analyst_tracker_from_chunk(
            tracker,
            {"market_report": "done", "news_report": "done"},
            now=100.0,
        )

        self.assertEqual(
            tracker.get_wall_times(),
            {"market": 3.0, "news": 90.0},
        )

    def test_serial_mode_keeps_single_active_analyst(self):
        # concurrency_limit=1 must behave exactly as before: only one analyst
        # active at a time, the next starting only once the prior completes.
        plan = build_analyst_execution_plan(
            ["market", "news"], concurrency_limit=1
        )
        tracker = AnalystWallTimeTracker(plan)

        sync_analyst_tracker_from_chunk(tracker, {}, now=10.0)
        self.assertEqual(tracker.active_keys(), ["market"])

        sync_analyst_tracker_from_chunk(
            tracker, {"market_report": "done"}, now=13.0
        )
        self.assertEqual(tracker.active_keys(), ["news"])
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

    def test_mark_initial_active_anchors_window_before_streaming(self):
        # The CLI starts the initial window before the first chunk arrives.
        # That pre-stream start (t=5) is what a later report measures against,
        # which is exactly the case the old serial logic collapsed to ~0s.
        plan = build_analyst_execution_plan(
            ["market", "news", "fundamentals"], concurrency_limit=2
        )
        tracker = AnalystWallTimeTracker(plan)

        tracker.mark_initial_active(started_at=5.0)
        self.assertEqual(set(tracker.active_keys()), {"market", "news"})

        sync_analyst_tracker_from_chunk(
            tracker, {"market_report": "done"}, now=12.0
        )
        self.assertEqual(tracker.get_wall_times(), {"market": 7.0})

    def test_mark_initial_active_serial_starts_only_first(self):
        plan = build_analyst_execution_plan(
            ["market", "news"], concurrency_limit=1
        )
        tracker = AnalystWallTimeTracker(plan)

        tracker.mark_initial_active(started_at=5.0)

        self.assertEqual(tracker.active_keys(), ["market"])
